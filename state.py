"""Graph state: topic, options, search pipeline, and report output."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ResearchState(TypedDict):
    """State carried through the deep-research graph."""

    topic: str
    options: NotRequired[dict[str, Any]]
    # First wave: from gap planner before second Tavily merge; normalized URL keys (see nodes).
    pre_gap_source_urls: NotRequired[list[str] | None]
    # After second Tavily: normalized URL keys for sources that existed before wave-2 add.
    wave2_url_keys: NotRequired[list[str] | None]
    search_queries: list[str]
    source_results: list[dict[str, Any]]
    extracted_facts: list[str]
    final_report: str
