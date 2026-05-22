import pytest
from pydantic import ValidationError

from nodes import (
    _normalize_url_key,
    _jaccard,
    _dedupe_fact_strings,
    _dedupe_overlapping_fact_lines,
    _heuristic_extract_facts,
    _heuristic_search_queries,
    _heuristic_gap_queries,
    _format_cited_report,
)
from research_options import ResearchOptions, options_from_state
from vector_store import rag_enabled, embed_facts

def test_normalize_url_key_strips_trailing_slash():
    assert _normalize_url_key("https://Example.COM/path/") == "https://example.com/path"

def test_normalize_url_key_lowercases_domain():
    assert _normalize_url_key("HTTPS://DOCS.PYTHON.ORG/3/library/") == "https://docs.python.org/3/library"

def test_normalize_url_key_empty_string():
    assert _normalize_url_key("") == ""

def test_jaccard_identical_strings():
    s = "machine learning model training is important"
    assert _jaccard(s, s) == 1.0

def test_jaccard_completely_different():
    assert _jaccard("machine learning model training", "deep sea diving techniques") == 0.0

def test_jaccard_partial_overlap():
    score = _jaccard("machine learning model training", "model training techniques")
    assert 0.0 < score < 1.0

def test_dedupe_fact_strings_removes_exact_duplicates():
    input_facts = [
        "- Fact A — [S](https://a.com)",
        "- Fact A — [S](https://a.com)",
        "- Fact B — [S](https://b.com)"
    ]
    out = _dedupe_fact_strings(input_facts)
    assert len(out) == 2
    assert out[0] == "- Fact A — [S](https://a.com)"

def test_dedupe_fact_strings_preserves_unique():
    input_facts = [
        "- Fact A — [A](https://a.com)",
        "- Fact B — [B](https://b.com)",
        "- Fact C — [C](https://c.com)",
        "- Fact D — [D](https://d.com)",
        "- Fact E — [E](https://e.com)",
    ]
    out = _dedupe_fact_strings(input_facts)
    assert len(out) == 5

def test_dedupe_overlapping_fact_lines_removes_paraphrase():
    # Jaccard needs to be >= 0.78 for these
    # We construct two strings that share almost all words
    input_facts = [
        "- This is a very specific machine learning model training technique used widely — [S1](https://a.com)",
        "- This is a very specific machine learning model training technique used widely — [S2](https://b.com)"
    ]
    out = _dedupe_overlapping_fact_lines(input_facts)
    assert len(out) == 1

def test_heuristic_extract_facts_returns_bullets():
    raw = [
        {"url": "https://a.com", "content": "This is a fact about AI.", "title": "AI Site"}
    ]
    out = _heuristic_extract_facts(raw)
    assert len(out) == 1
    assert out[0].startswith("- ")

def test_heuristic_extract_facts_skips_non_http_urls():
    raw = [
        {"url": "ftp://badscheme.com/file", "content": "Something", "title": "FTP"}
    ]
    out = _heuristic_extract_facts(raw)
    assert len(out) == 0

def test_heuristic_extract_facts_skips_empty_content():
    raw = [
        {"url": "https://a.com", "content": "", "title": "Empty"}
    ]
    out = _heuristic_extract_facts(raw)
    assert len(out) == 0

def test_heuristic_search_queries_returns_five():
    out = _heuristic_search_queries("quantum computing")
    assert len(out) == 5
    for q in out:
        assert "quantum computing" in q

def test_heuristic_gap_queries_returns_three():
    out = _heuristic_gap_queries("climate change")
    assert len(out) == 3

def test_format_cited_report_structure():
    ro = ResearchOptions()
    facts = ["- A fact — [Source](https://x.com)"]
    res = _format_cited_report("Test Topic", facts, ro)
    assert "# Test Topic" in res
    assert "## Findings" in res
    assert "## References" in res

def test_format_cited_report_empty_facts():
    ro = ResearchOptions()
    res = _format_cited_report("Empty", [], ro)
    assert "# Empty" in res

def test_research_options_audience_validator_normalizes_invalid():
    ro = ResearchOptions(audience="UNKNOWN")
    assert ro.audience == "general"

def test_research_options_style_validator_normalizes_invalid():
    ro = ResearchOptions(output_style="fancy")
    assert ro.output_style == "bullets"

def test_research_options_depth_bounds():
    ResearchOptions(depth=1)
    ResearchOptions(depth=2)
    with pytest.raises(ValidationError):
        ResearchOptions(depth=0)
    with pytest.raises(ValidationError):
        ResearchOptions(depth=3)

def test_options_from_state_with_missing_key():
    state = {"topic": "x"}
    ro = options_from_state(state)
    assert isinstance(ro, ResearchOptions)

def test_vector_store_rag_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    assert rag_enabled() is False

def test_vector_store_embed_facts_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    assert embed_facts("job1", ["fact"]) is False
