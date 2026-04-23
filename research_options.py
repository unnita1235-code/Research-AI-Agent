"""Pydantic options for a research run (API + graph state)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ResearchOptions(BaseModel):
    """User-tunable research parameters (stored on jobs and in graph state as a dict)."""

    depth: int = Field(1, ge=1, le=2, description="1 = one search pass, 2 = gap-filling second pass")
    audience: str = Field("general", description="beginner, general, or expert")
    max_sources: int = Field(24, ge=3, le=200)
    max_results_per_query: int = Field(5, ge=1, le=10)
    output_style: str = Field("bullets", description="bullets or narrative (numbered with [n] cites)")
    search_breadth: int = Field(4, ge=3, le=7, description="Number of first-pass search queries")

    @field_validator("audience")
    @classmethod
    def _audience(cls, v: str) -> str:
        a = (v or "general").strip().lower()
        if a in ("beginner", "general", "expert"):
            return a
        return "general"

    @field_validator("output_style")
    @classmethod
    def _style(cls, v: str) -> str:
        s = (v or "bullets").strip().lower()
        if s in ("bullets", "narrative"):
            return s
        return "bullets"

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "audience": self.audience,
            "max_sources": self.max_sources,
            "max_results_per_query": self.max_results_per_query,
            "output_style": self.output_style,
            "search_breadth": self.search_breadth,
        }


def options_from_state(state: Any) -> ResearchOptions:
    """Build ResearchOptions from graph state (dict with optional options key)."""
    if isinstance(state, dict):
        raw = state.get("options")
        if isinstance(raw, ResearchOptions):
            return raw
        if isinstance(raw, dict):
            return ResearchOptions.model_validate(
                {**ResearchOptions().model_dump(), **raw}
            )
    return ResearchOptions()

