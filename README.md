# Deep Research Agent

Deep Research Agent — a FastAPI + LangGraph pipeline that converts a research topic into a cited Markdown report via multi-step web search, LLM fact extraction, ChromaDB RAG, and structured synthesis.

[![CI](https://github.com/unnita1235-code/Research-AI-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/unnita1235-code/Research-AI-Agent/actions/workflows/ci.yml) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Architecture

```mermaid
flowchart LR
    User --> QueryGenerator
    QueryGenerator --> TavilySearcher
    TavilySearcher --> FactExtractor
    FactExtractor --> GapPlanner
    GapPlanner --> TavilySearcherWave2[TavilySearcher (wave 2)]
    TavilySearcherWave2 --> FactExtractorWave2[FactExtractor (wave 2)]
    FactExtractorWave2 --> ChromaDBRAG
    ChromaDBRAG --> ReportWriter[ReportWriter (LLM)]
    ReportWriter --> CitedMarkdownReport
```

## Features

- **Multi-pass depth:** Supports a 2-pass search approach (gap planning and follow-up searching).
- **SSE streaming:** Real-time progress updates sent to the client via Server-Sent Events.
- **SQLite job history:** Keeps a local record of executed research jobs (`data/jobs.sqlite`).
- **Export formats:** Outputs reports in Markdown (MD), HTML, and DOCX formats.
- **Q&A on report:** Allows follow-up questions and answers based on the generated report.
- **TL;DR:** Optionally generates a short summary of the findings.
- **Translation:** Supports translating the final report.
- **Counterarguments:** Generates opposing perspectives and limitations.
- **Executive summary:** A concise overview of the core topic before the detailed findings.
- **Heuristic mode:** Can be run without an LLM (`LLM_PROVIDER=heuristic`), relying solely on templates and snippets.
- **Ollama support:** Run locally and privately using an Ollama LLM setup.
- **Docker deployment:** Fully containerized with a simple `docker-compose.yml` for instant startup.
- **ChromaDB RAG:** Uses a local vector store and `sentence-transformers` to fetch relevant facts for LLM synthesis (`ENABLE_RAG=1`).

## Quick Start

You can run the application in three ways. First, copy `.env.example` to `.env` and fill in your API keys.

**1. Docker (one command):**
```bash
docker compose up
```

**2. Python local:**
```bash
pip install -r requirements.txt
python main.py serve
```

**3. CLI:**
```bash
python main.py "quantum computing"
```

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `anthropic` (default), `heuristic` (no LLM, only Tavily), or `ollama` (local). |
| `ANTHROPIC_API_KEY` | Your Anthropic Claude API key. |
| `TAVILY_API_KEY` | Your Tavily search API key (required for web searches). |
| `OLLAMA_MODEL` | Local Ollama model name (e.g., `llama3.2`). |
| `OLLAMA_BASE_URL` | Local Ollama API URL (e.g., `http://127.0.0.1:11434`). |
| `ENABLE_SUGGEST_TOPICS` | Set `1` to enable dynamic topic suggestions on the frontend. |
| `ENABLE_EXECUTIVE_SUMMARY` | Set `1` to prepend an executive summary to the report. |
| `ENABLE_ASK` | Set `1` to enable Q&A functionality. |
| `ENABLE_COUNTERARGUMENTS` | Set `1` to enable counterargument generation. |
| `ENABLE_TLDR` | Set `1` to generate a TL;DR section. |
| `ENABLE_TRANSLATE` | Set `1` to enable translation endpoints. |
| `ENABLE_JOB_DB` | Set `1` to save jobs to SQLite. Set `0` to disable history. |
| `JOB_DB_PATH` | Path to the SQLite database (e.g., `./data/jobs.sqlite`). |
| `RATE_LIMIT_PER_MIN` | Rate limit for POST `/research` per IP address. `0` disables limits. |
| `ENABLE_RAG` | Set `1` to enable ChromaDB + sentence-transformers vector store RAG. |

## Deep Research Architecture

The application is built around a dynamic LangGraph state machine. It takes a user topic and first translates it into specific search queries. The engine then uses Tavily to perform parallel web searches. Facts are extracted from the retrieved HTML and deduplicated using a Jaccard similarity scoring function to ensure maximum uniqueness of claims.

If the user sets a two-pass depth, the engine activates a gap-planner that reviews the currently extracted facts and generates follow-up queries to find missing context or counter-perspectives. These new queries are passed to a second wave of Tavily searches and fact extraction. Finally, the collected facts are embedded into an ephemeral ChromaDB vector store for Retrieval-Augmented Generation (RAG). The Anthropic Claude 3.5 Sonnet LLM then retrieves the most relevant subset of these facts and synthesizes a highly structured, fully cited Markdown report. In heuristic mode, no LLM is used; the application relies strictly on search snippets and templating logic.

## Testing

Run the standard pytest suite:
```bash
python -m pytest -v
```

Generate a test coverage report:
```bash
python -m pytest --cov=. --cov-report=term-missing
```

## License

MIT License. See `LICENSE` for details.
