"""Compile the deep-research LangGraph: query → search → facts → [gap → search → facts] → report.

With ``options.depth == 2``, a gap-planner and second Tavily pass run before the report.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from langgraph.graph import END, START, StateGraph

from research_options import ResearchOptions, options_from_state
from nodes import (
    fact_extractor_node,
    fact_second_node,
    gap_query_node,
    query_generator_node,
    report_writer_node,
    tavily_searcher_node,
)
from state import ResearchState

# Node ids (LangGraph identifiers)
NODE_QUERY = "query_generator"
NODE_TAVILY1 = "tavily_pass1"
NODE_FACT1 = "fact_pass1"
NODE_GAP = "gap_planner"
NODE_TAVILY2 = "tavily_pass2"
NODE_FACT2 = "fact_pass2"
NODE_REPORT = "report_writer"

# Backward-compatible names for importers
NODE_TAVILY = NODE_TAVILY1
NODE_FACTS = NODE_FACT1

_DISPLAY_NAME = {
    NODE_QUERY: "Query Generator",
    NODE_TAVILY1: "Tavily Searcher",
    NODE_TAVILY2: "Tavily Searcher (wave 2)",
    NODE_FACT1: "Fact Extractor",
    NODE_FACT2: "Fact Extractor (wave 2)",
    NODE_GAP: "Gap query planner",
    NODE_REPORT: "Report Writer",
}

OnStateUpdate: TypeAlias = Callable[[ResearchState, int, str], Awaitable[None]]


def _route_after_f1(state: ResearchState) -> str:
    ro = options_from_state(state)
    if ro.depth >= 2:
        return "deeper"
    return "write"


def build_research_graph() -> object:
    """Build and compile the **StateGraph** (async nodes: ``ainvoke`` / ``astream``)."""
    g = StateGraph(ResearchState)
    g.add_node(NODE_QUERY, query_generator_node)
    g.add_node(NODE_TAVILY1, tavily_searcher_node)
    g.add_node(NODE_FACT1, fact_extractor_node)
    g.add_node(NODE_GAP, gap_query_node)
    g.add_node(NODE_TAVILY2, tavily_searcher_node)
    g.add_node(NODE_FACT2, fact_second_node)
    g.add_node(NODE_REPORT, report_writer_node)
    g.add_edge(START, NODE_QUERY)
    g.add_edge(NODE_QUERY, NODE_TAVILY1)
    g.add_edge(NODE_TAVILY1, NODE_FACT1)
    g.add_conditional_edges(NODE_FACT1, _route_after_f1, {"deeper": NODE_GAP, "write": NODE_REPORT})
    g.add_edge(NODE_GAP, NODE_TAVILY2)
    g.add_edge(NODE_TAVILY2, NODE_FACT2)
    g.add_edge(NODE_FACT2, NODE_REPORT)
    g.add_edge(NODE_REPORT, END)
    return g.compile()


def _has_pipeline_output(s: ResearchState) -> bool:
    """True once any node has written (skip LangGraph’s initial empty *values* tick)."""
    return bool(
        (s.get("search_queries") or [])
        or (s.get("source_results") or [])
        or (s.get("extracted_facts") or [])
        or (s.get("final_report") or "").strip()
    )


def _infer_step_label(s: ResearchState) -> str:
    """Map cumulative state to a rough label for *values* stream / UI (order matters for depth-2)."""
    if (s.get("final_report") or "").strip():
        return _DISPLAY_NAME[NODE_REPORT]
    if s.get("wave2_url_keys"):
        return _DISPLAY_NAME[NODE_FACT2]
    if s.get("pre_gap_source_urls") is not None:
        return _DISPLAY_NAME[NODE_GAP]
    if s.get("extracted_facts"):
        return _DISPLAY_NAME[NODE_FACT1]
    if s.get("source_results"):
        return _DISPLAY_NAME[NODE_TAVILY1]
    if s.get("search_queries"):
        return _DISPLAY_NAME[NODE_QUERY]
    return "Initial"


def _print_state_update(s: ResearchState, step_index: int) -> None:
    """Compact snapshot for console (real-time progress)."""
    label = _infer_step_label(s)
    topic = s.get("topic") or ""
    t_preview = topic if len(topic) <= 100 else topic[:100] + "…"
    print(f"\n--- Step {step_index}: {label} ---", flush=True)
    print(f"  topic: {t_preview!r}", flush=True)
    sq = s.get("search_queries") or []
    nq = len(sq)
    print(f"  search_queries: {nq}", flush=True)
    if sq:
        show = sq if nq <= 5 else sq[:5] + [f"… ({nq - 5} more)"]
        for i, q in enumerate(show, 1):
            print(f"    {i}. {q}", flush=True)
    ns = len(s.get("source_results") or [])
    print(f"  source_results: {ns} items", flush=True)
    nf = len(s.get("extracted_facts") or [])
    print(f"  extracted_facts: {nf} items", flush=True)
    fr = s.get("final_report") or ""
    if fr:
        prev = fr[:500].replace("\n", " ")
        if len(fr) > 500:
            prev += "…"
        print(f"  final_report: {len(fr)} chars (preview) {prev!r}", flush=True)
    else:
        print("  final_report: (empty)", flush=True)


async def run_research(
    topic: str,
    *,
    stream_progress: bool = True,
    on_state_update: OnStateUpdate | None = None,
    options: dict[str, Any] | None = None,
) -> ResearchState:
    """Run the research graph. Optional ``options`` (depth, audience, max_sources, …)."""
    g = build_research_graph()
    base = ResearchOptions().model_dump()
    base.update({**(options or {})})
    init: ResearchState = {
        "topic": (topic or "").strip(),
        "options": base,
        "search_queries": [],
        "source_results": [],
        "extracted_facts": [],
        "final_report": "",
    }

    use_stream = stream_progress or (on_state_update is not None)
    if not use_stream:
        out = await g.ainvoke(init)  # type: ignore[assignment]
        return out  # type: ignore[return-value]

    if on_state_update is None and stream_progress:
        print(
            f"\nFlow: START → {_DISPLAY_NAME[NODE_QUERY]} → {_DISPLAY_NAME[NODE_TAVILY1]} → "
            f"{_DISPLAY_NAME[NODE_FACT1]} → (optional) {_DISPLAY_NAME[NODE_GAP]} → … → "
            f"{_DISPLAY_NAME[NODE_REPORT]} → END\n"
            f"Topic: {init['topic']!r}\n",
            flush=True,
        )

    last: ResearchState = init
    step = 0
    first = True
    async for state in g.astream(init, stream_mode="values"):
        last = state  # type: ignore[assignment]
        if first and not _has_pipeline_output(last):
            first = False
            continue
        first = False
        step += 1
        label = _infer_step_label(last)
        if on_state_update is not None:
            await on_state_update(last, step, label)
        elif stream_progress:
            _print_state_update(last, step)

    return last


__all__ = [
    "NODE_FACT1",
    "NODE_FACT2",
    "NODE_FACTS",
    "NODE_GAP",
    "NODE_QUERY",
    "NODE_REPORT",
    "NODE_TAVILY",
    "NODE_TAVILY1",
    "NODE_TAVILY2",
    "OnStateUpdate",
    "build_research_graph",
    "run_research",
]
