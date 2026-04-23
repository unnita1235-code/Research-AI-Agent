"""LangGraph nodes: query generation, Tavily search, fact extraction, report writing.

LLM behavior is controlled by ``LLM_PROVIDER``:

- **anthropic** (default if unset): Claude 3.5 Sonnet (needs credits).
- **heuristic**: no LLM — templates + snippets from Tavily only (needs ``TAVILY_API_KEY``).
- **ollama**: local model via Ollama (``langchain-ollama``; needs Ollama running).

The search node always uses Tavily.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from urllib.parse import urlparse, urlunparse
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
from tavily import AsyncTavilyClient

from research_options import ResearchOptions, options_from_state
from state import ResearchState

logger = logging.getLogger(__name__)

# https://docs.anthropic.com/en/docs/about-claude/models
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"

# Trim long snippets so downstream LLM context stays bounded
_MAX_SNIPPET_CHARS = 3000
_MAX_HEURISTIC_FACTS = 24
# Per-query cap so `asyncio.gather` on many queries does not run past ~3m budget for the whole graph
TAVILY_QUERY_TIMEOUT = float(os.environ.get("TAVILY_QUERY_TIMEOUT", "50"))
# Drop near-duplicate *fact text* (different URLs) in fact extractor
_FACT_OVERLAP_JACCARD = float(os.environ.get("FACT_DEDUPE_JACCARD", "0.78"))
# Stop-words for Jaccard on overlap (minimal)
_STOP = frozenset(
    "a an the to of and or for in on at by is are was were be been being with as it its this that these"
    " from into over per than then not no".split()
)


def _llm_mode() -> str:
    v = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    return v or "anthropic"


def _get_chat() -> BaseChatModel:
    """Claude, or Ollama when ``LLM_PROVIDER=ollama``."""
    m = _llm_mode()
    if m == "heuristic":
        raise RuntimeError("_get_chat() is not used in heuristic mode")
    if m == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(  # type: ignore[return-value]
            model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=0.2,
        )
    if m in ("anthropic", "claude", ""):
        return ChatAnthropic(
            model=CLAUDE_3_5_SONNET,
            temperature=0.2,
            max_tokens=4096,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER={m!r}. Use: anthropic, ollama, or heuristic (no credit needed)."
    )


def _heuristic_search_queries(topic: str) -> list[str]:
    t = (topic or "").strip()
    if not t:
        return []
    return [
        f"{t} overview",
        f"{t} definition",
        f"{t} how it works",
        f"{t} examples",
        f"{t} applications",
    ]


def _normalize_url_key(u: str) -> str:
    p = urlparse((u or "").strip())
    if not p.netloc:
        return ""
    path = p.path or ""
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((p.scheme or "https", p.netloc.lower(), path, "", p.query, "")).lower()


def _word_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t and t not in _STOP and len(t) > 1}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _word_set(a), _word_set(b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    if not u:
        return 0.0
    return len(sa & sb) / len(u)


def _text_before_link_bullet(line: str) -> str:
    m = re.match(r"^-\s*(.+?)\s*—\s*\[", line, flags=re.S)
    return m.group(1).strip() if m else re.sub(r"^\s*[-*]\s*", "", (line or "").split("—", 1)[0]).strip()


def _source_keys_from_results(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for s in rows:
        if not isinstance(s, dict):
            continue
        k = _normalize_url_key((s.get("url") or "").strip())
        if k and k not in seen:
            seen.add(k)
            order.append(k)
    return order


def _slice_sources_by_score(rows: list[dict[str, Any]], max_n: int) -> list[dict[str, Any]]:
    if max_n <= 0 or len(rows) <= max_n:
        return rows
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        s = r.get("score")
        f = -1.0 if s is None else float(s)
        scored.append((f, -i, r))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [t[2] for t in scored[:max_n]]


def _dedupe_tavily_merged(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate **URLs** (multi-query overlap); keep the row with the higher score or longer text."""
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in merged:
        url = (item.get("url") or "").strip()
        k = _normalize_url_key(url)
        if not k:
            continue
        if k not in best:
            order.append(k)
            best[k] = item
            continue
        old = best[k]
        s_item = -1.0 if item.get("score") is None else float(item.get("score"))
        s_old = -1.0 if old.get("score") is None else float(old.get("score"))
        if s_item > s_old or (s_item == s_old and len(item.get("content") or "") > len(old.get("content") or "")):
            best[k] = item
    return [best[k] for k in order]


def _dedupe_overlapping_fact_lines(
    lines: list[str], *, jaccard_min: float = _FACT_OVERLAP_JACCARD
) -> list[str]:
    """Remove redundant information when the same (or very similar) claim appears under different links."""
    out: list[str] = []
    for line in lines:
        text = _text_before_link_bullet(line)
        if not text:
            continue
        is_dup = False
        for prev in out:
            prev_text = _text_before_link_bullet(prev)
            if _jaccard(text, prev_text) >= jaccard_min:
                is_dup = True
                break
        if not is_dup:
            out.append(line)
    return out


def _allowed_http_urls(sources: list[dict[str, Any]]) -> set[str]:
    """Only URLs that appeared in source snippets (citation allowlist)."""
    out: set[str] = set()
    for s in sources:
        if not isinstance(s, dict):
            continue
        u = (s.get("url") or "").strip()
        p = urlparse(u)
        if p.scheme in ("http", "https") and p.netloc:
            out.add(u)
    return out


def _is_allowed_url(url: str, allow: set[str]) -> bool:
    nu = _normalize_url_key(url)
    if not nu:
        return False
    for a in allow:
        if _normalize_url_key(a) == nu:
            return True
    return False


def _heuristic_extract_facts(
    raw: list[dict[str, Any]], *, allowed_urls: set[str] | None = None
) -> list[str]:
    """Slice Tavily snippets into bullet lines; no model. Requires URL in allowlist when provided."""
    allow = allowed_urls
    if allow is None:
        allow = _allowed_http_urls([r for r in raw if isinstance(r, dict)])
    lines: list[str] = []
    for s in _dedupe_tavily_merged([r for r in raw if isinstance(r, dict)]):
        if len(lines) >= _MAX_HEURISTIC_FACTS:
            break
        content = (s.get("content") or "").strip()
        if not content:
            continue
        content = re.sub(r"\s+", " ", content)
        if len(content) > 500:
            content = content[:500].rsplit(" ", 1)[0] + "…"
        title = (s.get("title") or "Source").strip() or "Source"
        url = (s.get("url") or "").strip()
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.netloc:
            continue
        if not _is_allowed_url(url, allow):
            continue
        line = f"- {content} — [{title}]({url})"
        lines.append(line)
    out = _dedupe_fact_strings(lines)
    return _dedupe_overlapping_fact_lines(out, jaccard_min=_FACT_OVERLAP_JACCARD)


def _url_to_cite_index(facts: list[str]) -> dict[str, int]:
    """Map raw URL (as in markdown) to 1-based cite index, order of first appearance in facts."""
    order: list[str] = []
    for f in facts:
        for m_ in re.finditer(r"\((https?://[^)]+)\)", f):
            u = m_.group(1)
            if u not in order:
                order.append(u)
    return {u: i + 1 for i, u in enumerate(order)}


def _fact_text_and_cite_nums(line: str, url_to_n: dict[str, int]) -> tuple[str, list[int]]:
    """Strip trailing ``— [t](url)`` from a fact line; return claim text and 1-based cite indices."""
    nums: list[int] = []
    for m_ in re.finditer(r"\[([^\]]+)\]\((https?://[^)]+)\)", line):
        u = m_.group(2)
        n = url_to_n.get(u)
        if n is not None:
            nums.append(n)
    t = re.sub(r"\s*—\s*\[([^\]]+)\]\((https?://[^)]+)\)\s*$", "", line).strip()
    if t.startswith("- "):
        t = t[2:].strip()
    return (t or line.strip(), nums)


def _format_cited_report(topic: str, facts: list[str], ro: ResearchOptions) -> str:
    """Build markdown from pre-verified ``facts``; style from ``ro``."""
    aud = ro.audience
    aud_line = {
        "beginner": "_Audience: non-specialist; avoid heavy jargon and define terms when needed._\n",
        "general": "_Audience: informed reader; balance depth and clarity._\n",
        "expert": "_Audience: specialist; you may use domain terminology and denser phrasing where helpful._\n",
    }.get(aud, "")

    url_to_n = _url_to_cite_index(facts)
    ref_block_lines: list[str] = []
    for u, i in sorted(url_to_n.items(), key=lambda x: x[1]):
        ref_block_lines.append(f"{i}. {u}")
    ref_block = "\n".join(ref_block_lines) or "_No URLs in snippets._\n"

    if ro.output_style == "narrative" and facts:
        lines_out: list[str] = []
        for i, f in enumerate(facts, start=1):
            t, nums_ = _fact_text_and_cite_nums(f, url_to_n)
            cite_str = "".join(f" [{n}]" for n in nums_) if nums_ else ""
            lines_out.append(f"{i}. {t}{cite_str}")
        body = "\n\n".join(lines_out)
    else:
        body = "\n\n".join(facts) if facts else "_No source-verified facts to include._\n"

    title_line = f"# {topic}\n\n"
    return (
        f"{title_line}"
        f"{aud_line}\n"
        "## Findings\n\n"
        f"{body}\n\n"
        "## References\n\n"
        f"{ref_block}\n"
    )


def _heuristic_breadth_queries(topic: str, n: int) -> list[str]:
    t = (topic or "").strip()
    if not t:
        return []
    base = _heuristic_search_queries(topic)  # 5 templates
    if n <= len(base):
        return base[:n]
    extra = [
        f"{t} 2024 2025",
        f"{t} case studies",
    ]
    out = list(base)
    for e in extra:
        if len(out) >= n:
            break
        if e not in out:
            out.append(e)
    while len(out) < n:
        out.append(f"{t} research")
    return out[:n]


class _QueryList(BaseModel):
    """3–7 distinct search strings for the topic (breadth from options)."""

    queries: list[str] = Field(
        min_length=3,
        max_length=7,
        description="Distinct, specific search queries to cover the topic from different angles.",
    )

    @field_validator("queries")
    @classmethod
    def _strip_non_empty(cls, qs: list[str]) -> list[str]:
        out = [q.strip() for q in qs if q and q.strip()]
        if len(out) < 3 or len(out) > 7:
            raise ValueError("Must provide between 3 and 7 non-empty queries")
        return out


class _CitedFact(BaseModel):
    fact: str
    source_url: str
    source_title: str = ""


class _FactExtractionResult(BaseModel):
    facts: list[_CitedFact] = Field(
        default_factory=list,
        description="Unique facts, figures, and claims, each tied to a source URL from the input only.",
    )


def _filter_result_item(raw: dict[str, Any], query: str) -> dict[str, Any]:
    content = (raw.get("content") or raw.get("raw_content") or "").strip()
    if len(content) > _MAX_SNIPPET_CHARS:
        content = content[:_MAX_SNIPPET_CHARS] + "…"
    return {
        "query": query,
        "url": (raw.get("url") or "").strip(),
        "title": (raw.get("title") or "").strip(),
        "content": content,
        "score": raw.get("score"),
    }


def _normalize_fact_key(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").lower().strip())
    t = re.sub(r"^[\-\*\u2022\d\.\s]+", "", t)
    return t[:200]


def _dedupe_fact_strings(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = _normalize_fact_key(line)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _tavily_client() -> AsyncTavilyClient:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY is not set")
    return AsyncTavilyClient(api_key=key)


async def query_generator_node(state: ResearchState) -> dict[str, list[str]]:
    """Generate distinct search queries for ``state['topic']`` (count from options.search_breadth)."""
    topic = (state.get("topic") or "").strip()
    if not topic:
        return {"search_queries": []}

    ro = options_from_state(state)
    b = int(ro.search_breadth)

    if _llm_mode() == "heuristic":
        return {"search_queries": _heuristic_breadth_queries(topic, b)}

    llm = _get_chat().with_structured_output(_QueryList)
    out = await llm.ainvoke(
        [
            SystemMessage(
                content="You create search engine queries. Output only the structured list: "
                f"{b} short, non-overlapping queries that cover definitions, key facts, "
                "comparisons, and recent context when relevant. No questions unless they "
                "are effective as search keywords."
            ),
            HumanMessage(
                content=f"Research topic:\n{topic}\n\nProduce exactly {b} distinct queries for this topic."
            ),
        ],
    )
    if isinstance(out, _QueryList):
        return {"search_queries": out.queries}
    if isinstance(out, dict) and "queries" in out:
        return {"search_queries": out["queries"]}  # type: ignore[return-value]
    raise TypeError("Unexpected structured output from query generator")


async def tavily_searcher_node(state: ResearchState) -> dict[str, Any]:
    """Run one Tavily search per query in parallel. Supports wave-2 merge + ``max_sources`` cap."""
    ro = options_from_state(state)
    max_res = int(ro.max_results_per_query)
    max_src = int(ro.max_sources)

    queries = [q for q in (state.get("search_queries") or []) if q and str(q).strip()]
    pre_gap = state.get("pre_gap_source_urls")
    if not queries and pre_gap is not None:
        return {"pre_gap_source_urls": None}
    if not queries:
        # No queries and no second-wave merge: leave state unchanged (do not clear sources).
        return {}

    client = _tavily_client()

    async def _one(q: str) -> tuple[str, dict[str, Any] | None]:
        try:
            res = await asyncio.wait_for(
                client.search(
                    str(q).strip(),
                    max_results=max_res,
                    search_depth="basic",
                ),
                timeout=TAVILY_QUERY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Tavily search timed out for %r (>%ss)", q, TAVILY_QUERY_TIMEOUT)
            return str(q), None
        except Exception as e:  # noqa: BLE001
            logger.warning("Tavily search failed for %r: %s", q, e)
            return str(q), None
        return str(q), res if isinstance(res, dict) else None

    pairs = await asyncio.gather(*(_one(q) for q in queries), return_exceptions=True)

    merged: list[dict[str, Any]] = []
    for p in pairs:
        if isinstance(p, Exception):
            logger.warning("Tavily task error: %s", p)
            continue
        q, res = p
        if res is None:
            continue
        for item in (res.get("results") or []):
            if isinstance(item, dict):
                merged.append(_filter_result_item(item, q))

    wave_keys: list[str] | None = None
    if pre_gap is not None:
        old = [r for r in (state.get("source_results") or []) if isinstance(r, dict)]
        wave_keys = list(pre_gap)
        combined = _dedupe_tavily_merged(old + _dedupe_tavily_merged(merged))
    else:
        combined = _dedupe_tavily_merged(merged)

    combined = _slice_sources_by_score(combined, max_src)
    out: dict[str, Any] = {"source_results": combined}
    if pre_gap is not None:
        out["pre_gap_source_urls"] = None
        out["wave2_url_keys"] = wave_keys
    return out


async def fact_extractor_node(state: ResearchState) -> dict[str, Any]:
    """From ``source_results``, extract facts; in wave-2 only new (post-gap) sources are used, then merged."""
    raw = state.get("source_results") or []
    w2k = state.get("wave2_url_keys")
    prior_lines = list(state.get("extracted_facts") or [])

    if not raw and not prior_lines:
        return {"extracted_facts": []}

    if _llm_mode() == "heuristic":
        if w2k is not None:
            base = set(w2k)
            src = _dedupe_tavily_merged([r for r in raw if isinstance(r, dict)])
            src = [
                r
                for r in src
                if _normalize_url_key((r.get("url") or "").strip()) not in base
            ]
            if not src:
                return {"extracted_facts": prior_lines, "wave2_url_keys": None}
            new_lines = _heuristic_extract_facts(src)
            merged = _dedupe_overlapping_fact_lines(
                prior_lines + new_lines, jaccard_min=_FACT_OVERLAP_JACCARD
            )
            return {"extracted_facts": merged, "wave2_url_keys": None}
        return {
            "extracted_facts": _heuristic_extract_facts(
                [r for r in raw if isinstance(r, dict)]
            )
        }

    if w2k is not None:
        base = set(w2k)
        src = _dedupe_tavily_merged([r for r in raw if isinstance(r, dict)])
        src = [
            r
            for r in src
            if _normalize_url_key((r.get("url") or "").strip()) not in base
        ]
        if not src:
            return {"extracted_facts": prior_lines, "wave2_url_keys": None}
    else:
        # Same URL de-dupe as tavily output; full pass
        src = _dedupe_tavily_merged([r for r in raw if isinstance(r, dict)])
    pack = [
        {
            "url": s.get("url", ""),
            "title": s.get("title", ""),
            "snippet": (s.get("content") or "")[:_MAX_SNIPPET_CHARS],
        }
        for s in src
    ]
    if not pack:
        if w2k is not None:
            return {"extracted_facts": prior_lines, "wave2_url_keys": None}
        return {"extracted_facts": []}
    allowed = _allowed_http_urls(src)
    if not allowed:
        if w2k is not None:
            return {"extracted_facts": prior_lines, "wave2_url_keys": None}
        return {"extracted_facts": []}

    llm = _get_chat().with_structured_output(_FactExtractionResult)
    out = await llm.ainvoke(
        [
            SystemMessage(
                content="You extract only short factual claims that are directly supported by the snippet for that URL. "
                "Set source_url to one of the exact 'url' strings from the input JSON; never invent, aggregate, or guess URLs. "
                "If a claim cannot be tied to a specific url from the list, drop it. "
                "Deduplicate paraphrases of the same claim."
            ),
            HumanMessage(
                content="Sources (JSON list). Allowed URLs are exactly the 'url' field values in this list — no other URLs are valid.\n"
                f"{_safe_json_dumps(pack)}\n"
                "Return each fact with source_url and source_title exactly as in the entry you used."
            ),
        ],
    )
    if isinstance(out, _FactExtractionResult):
        facts = out.facts
    elif isinstance(out, dict) and "facts" in out:
        facts = _FactExtractionResult.model_validate(out).facts
    else:
        raise TypeError("Unexpected structured output from fact extractor")

    lines: list[str] = []
    for f in facts:
        u = f.source_url.strip()
        t = f.fact.strip()
        if not t or not _is_allowed_url(u, allowed):
            continue
        title = f.source_title.strip() or "Source"
        line = f"- {t} — [{title}]({u})"
        lines.append(line)
    lines = _dedupe_fact_strings(lines)
    new_only = _dedupe_overlapping_fact_lines(lines, jaccard_min=_FACT_OVERLAP_JACCARD)
    if w2k is not None:
        out_lines = _dedupe_overlapping_fact_lines(
            prior_lines + new_only, jaccard_min=_FACT_OVERLAP_JACCARD
        )
        return {"extracted_facts": out_lines, "wave2_url_keys": None}
    return {"extracted_facts": new_only}


def _safe_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)[: 120_000]


async def report_writer_node(state: ResearchState) -> dict[str, str]:
    """Render from ``extracted_facts``; style/audience from options."""
    topic = (state.get("topic") or "").strip() or "Research"
    facts = state.get("extracted_facts") or []
    ro = options_from_state(state)
    return {"final_report": _format_cited_report(topic, facts, ro)}


class _GapList(BaseModel):
    """2–4 follow-up search strings to fill coverage gaps after an initial pass."""

    queries: list[str] = Field(
        min_length=2,
        max_length=4,
        description="Narrow queries to find missing angles, limitations, or recent work.",
    )

    @field_validator("queries")
    @classmethod
    def _gap_q(cls, qs: list[str]) -> list[str]:
        out = [q.strip() for q in qs if q and str(q).strip()]
        if len(out) < 2 or len(out) > 4:
            raise ValueError("2–4 non-empty gap queries required")
        return out


def _heuristic_gap_queries(topic: str) -> list[str]:
    t = (topic or "").strip() or "topic"
    return [
        f"{t} open problems and future work",
        f"{t} recent survey or review 2024 2025",
        f"{t} limitations criticisms trade-offs",
    ]


async def gap_query_node(state: ResearchState) -> dict[str, Any]:
    """Plan 2–4 follow-up queries; snapshot pre-merge URL keys for wave-2."""
    raw = [r for r in (state.get("source_results") or []) if isinstance(r, dict)]
    pre = _source_keys_from_results(raw)
    topic = (state.get("topic") or "").strip()
    if not pre:
        return {"search_queries": [], "pre_gap_source_urls": None}

    if _llm_mode() == "heuristic":
        return {"search_queries": _heuristic_gap_queries(topic), "pre_gap_source_urls": pre}

    facts = state.get("extracted_facts") or []
    fact_blob = "\n".join(facts[: 48])[: 12_000]
    llm = _get_chat().with_structured_output(_GapList)
    out = await llm.ainvoke(
        [
            SystemMessage(
                content="You design follow-up **search engine queries** (not essay titles) to find gap coverage: "
                "missing angles, trade-offs, recent updates, or limitations. Output only structured queries."
            ),
            HumanMessage(
                content=f"Topic:\n{topic}\n\nInitial findings (markdown bullets, may be truncated):\n{fact_blob}\n\n"
                "Return 2–4 short keyword-friendly queries to query a web search API."
            ),
        ],
    )
    if isinstance(out, _GapList):
        return {"search_queries": out.queries, "pre_gap_source_urls": pre}
    if isinstance(out, dict) and "queries" in out:
        return {"search_queries": out["queries"], "pre_gap_source_urls": pre}  # type: ignore[return-value]
    raise TypeError("Unexpected structured output from gap query planner")


async def fact_second_node(state: ResearchState) -> dict[str, Any]:
    """Run fact extraction only for wave-2; no-op if the second Tavily pass did not schedule merging."""
    if not state.get("wave2_url_keys"):
        return {}
    return await fact_extractor_node(state)


__all__ = [
    "CLAUDE_3_5_SONNET",
    "query_generator_node",
    "tavily_searcher_node",
    "fact_extractor_node",
    "fact_second_node",
    "gap_query_node",
    "report_writer_node",
]
