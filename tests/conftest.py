"""Test environment: set before ``main`` is imported (no real API keys required)."""

from __future__ import annotations

import os

# Isolated from developer .env: CI and pytest need deterministic, offline-safe values.
os.environ.setdefault("TAVILY_API_KEY", "ci-synthetic-tavily-key-xxxxxxxx")
os.environ.setdefault("LLM_PROVIDER", "heuristic")
os.environ.setdefault("ANTHROPIC_API_KEY", "")
os.environ.setdefault("ENABLE_SUGGEST_TOPICS", "0")
os.environ.setdefault("ENABLE_JOB_DB", "0")
os.environ.setdefault("RATE_LIMIT_PER_MIN", "0")
