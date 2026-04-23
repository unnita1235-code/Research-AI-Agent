"""Smoke tests: no live Tavily/LLM calls (avoids background research jobs in CI)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from graph import build_research_graph
from research_options import ResearchOptions
from state import ResearchState


def test_research_options_defaults() -> None:
    o = ResearchOptions()
    assert o.depth == 1
    assert o.audience in ("beginner", "general", "expert")
    assert o.output_style in ("bullets", "narrative")


def test_graph_compiles() -> None:
    g = build_research_graph()
    assert g is not None


def test_research_state_shape() -> None:
    s: ResearchState = {
        "topic": "x",
        "options": {"depth": 1},
        "search_queries": [],
        "source_results": [],
        "extracted_facts": [],
        "final_report": "",
    }
    assert s["topic"] == "x"


def test_app_routes() -> None:
    from main import app

    c = TestClient(app)
    r = c.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "llm_provider" in body
    assert "app_version" in body

    r2 = c.get("/static/index.html")
    assert r2.status_code == 200

    r3 = c.get("/", follow_redirects=False)
    assert r3.status_code == 302

    r4 = c.post("/research/suggest-topics", json={})
    assert r4.status_code == 200
    assert "topics" in r4.json()

    r5 = c.get("/jobs")
    assert r5.status_code == 200
    assert "jobs" in r5.json()

    r6 = c.get("/research/00000000-0000-0000-0000-000000000000")
    assert r6.status_code == 404


def test_research_in_validation() -> None:
    from main import app

    c = TestClient(app)
    r = c.post("/research", json={"topic": ""})
    assert r.status_code == 422
