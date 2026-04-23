"""Optional LLM helpers: topic suggestions and executive one-liner (not on critical graph path)."""

from __future__ import annotations

import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from nodes import CLAUDE_3_5_SONNET, _get_chat, _llm_mode

logger = logging.getLogger(__name__)

# Curated when LLM is unavailable or disabled
STATIC_SUGGEST_TOPICS: list[str] = [
    "RAG with vector databases: trade-offs in 2025",
    "Async Python asyncio event loop in production",
    "EU AI Act: obligations for high-risk systems",
    "LangGraph state machines vs linear pipelines",
    "Retrieval metrics: nDCG vs recall@k for search",
    "Ollama vs cloud LLM for local agents",
    "FastAPI background tasks and SSE patterns",
    "Structured output with Pydantic and LLMs",
]


def suggest_topics_enabled() -> bool:
    return (os.environ.get("ENABLE_SUGGEST_TOPICS") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def executive_summary_enabled() -> bool:
    return (os.environ.get("ENABLE_EXECUTIVE_SUMMARY") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


class _TopicList(BaseModel):
    topics: list[str] = Field(min_length=3, max_length=8, description="Short research query strings.")


def _chat_for_summary() -> BaseChatModel:
    """Heuristic mode has no default chat; use Anthropic or Ollama when configured."""
    m = _llm_mode()
    if m == "heuristic" and (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return ChatAnthropic(
            model=CLAUDE_3_5_SONNET,
            temperature=0.3,
            max_tokens=220,
        )
    if m == "heuristic":
        raise ValueError(
            "Executive summary in heuristic mode needs ANTHROPIC_API_KEY, or set LLM_PROVIDER=ollama/anthropic."
        )
    return _get_chat()


async def _suggest_with_anthropic(seed: str | None) -> list[str] | None:
    if (os.environ.get("ANTHROPIC_API_KEY") or "").strip() and _llm_mode() in ("anthropic", "claude", ""):
        try:
            llm = ChatAnthropic(
                model=CLAUDE_3_5_SONNET,
                temperature=0.5,
                max_tokens=400,
            )
            structured = llm.with_structured_output(_TopicList)
            human = "Suggest 5–7 specific, diverse search topics for a deep research agent. "
            if (seed or "").strip():
                human += f"User hint: {seed}\n"
            else:
                human += "Vary domains (tech, policy, science, tools), no overlap.\n"
            human += "Output field topics (array of strings) only in the schema."
            out = await structured.ainvoke(
                [
                    SystemMessage(
                        content="You output only valid structured data. Topics must be under 90 chars, no quotes inside."
                    ),
                    HumanMessage(content=human),
                ],
            )
            if isinstance(out, _TopicList):
                return [t.strip() for t in out.topics if t and str(t).strip()][:8]
        except Exception as e:  # noqa: BLE001
            logger.warning("Suggest-topics LLM failed: %s", e)
    return None


async def suggest_topic_strings_async(seed: str | None) -> list[str]:
    if not suggest_topics_enabled():
        return list(STATIC_SUGGEST_TOPICS)
    got = await _suggest_with_anthropic(seed)
    if got and len(got) >= 3:
        return got
    return list(STATIC_SUGGEST_TOPICS)


async def executive_one_liner_async(report_markdown: str) -> str:
    """2–3 sentence evidence-bound overview; no new URLs or facts."""
    m = (report_markdown or "").strip()
    if len(m) < 40:
        return ""
    chat = _chat_for_summary()
    # Strip huge reports for token safety
    clip = m[: 24_000] + ("…" if len(m) > 24_000 else "")
    msg = await chat.ainvoke(
        [
            SystemMessage(
                content="You write a short (2-3 sentence) professional overview. "
                "You may only restate or synthesize what is explicitly in the user markdown; "
                "do not add new facts, numbers, or URLs. If you cannot, say the report is too thin."
            ),
            HumanMessage(
                content="Report (markdown, may contain # headings and bullet facts with links):\n\n" + clip
            ),
        ],
    )
    c = msg.content
    if isinstance(c, str):
        text = c
    else:
        text = "".join(
            p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in c
        )
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty summary from model")
    return text


def ask_feature_enabled() -> bool:
    return (os.environ.get("ENABLE_ASK") or "1").strip().lower() in ("1", "true", "yes")


def counterarguments_enabled() -> bool:
    return (os.environ.get("ENABLE_COUNTERARGUMENTS") or "0").strip().lower() in ("1", "true", "yes")


def tldr_enabled() -> bool:
    return (os.environ.get("ENABLE_TLDR") or "0").strip().lower() in ("1", "true", "yes")


def translate_enabled() -> bool:
    return (os.environ.get("ENABLE_TRANSLATE") or "0").strip().lower() in ("1", "true", "yes")


def _clip(s: str, n: int) -> str:
    t = (s or "").strip()
    if len(t) <= n:
        return t
    return t[:n] + "…"


async def ask_report_async(
    question: str,
    *,
    report: str,
    fact_lines: list[str],
) -> str:
    """Answer a question using only the given facts + report (RAG-lite)."""
    q = (question or "").strip()
    if len(q) < 2:
        raise ValueError("Question too short")
    chat = _chat_for_summary()
    facts_block = _clip("\n".join(fact_lines[: 120]), 20_000)
    rep = _clip(report, 18_000)
    msg = await chat.ainvoke(
        [
            SystemMessage(
                content="You answer using ONLY the context below. If the answer is not in the context, say you "
                "cannot find it in the provided material. Do not add URLs or facts that are not implied by the context."
            ),
            HumanMessage(
                content="FACT LINES (may be truncated):\n"
                f"{facts_block}\n\nREPORT (markdown, truncated):\n{rep}\n\nUSER QUESTION:\n{q}"
            ),
        ],
    )
    c = msg.content
    if isinstance(c, str):
        text = c
    else:
        text = "".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in c)
    return (text or "").strip() or "No answer from context."


async def counterarguments_async(report: str) -> str:
    chat = _chat_for_summary()
    clip = _clip(report, 20_000)
    msg = await chat.ainvoke(
        [
            SystemMessage(
                content="List possible limitations, counterarguments, or confounding factors implied by or consistent "
                "with this report only. Do not add new web facts. Use short markdown bullet points."
            ),
            HumanMessage(content=clip),
        ],
    )
    c = msg.content
    t = c if isinstance(c, str) else str(c)
    return (t or "").strip()


async def tldr_async(report: str) -> str:
    chat = _chat_for_summary()
    clip = _clip(report, 24_000)
    msg = await chat.ainvoke(
        [
            SystemMessage(
                content="Write a 3–5 bullet TL;DR of the main points in the report. Only restate what is in the text."
            ),
            HumanMessage(content=clip),
        ],
    )
    c = msg.content
    t = c if isinstance(c, str) else str(c)
    return (t or "").strip()


async def translate_report_async(report: str, target_lang: str) -> str:
    lang = (target_lang or "Spanish").strip() or "Spanish"
    chat = _chat_for_summary()
    clip = _clip(report, 28_000)
    msg = await chat.ainvoke(
        [
            SystemMessage(
                content="Translate the user markdown to the target language. Preserve markdown structure, links, and "
                "citation numbers. Do not add new information."
            ),
            HumanMessage(content=f"Target language: {lang}\n\n{clip}"),
        ],
    )
    c = msg.content
    t = c if isinstance(c, str) else str(c)
    return (t or "").strip()
