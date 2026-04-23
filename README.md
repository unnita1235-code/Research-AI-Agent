# Deep Research Agent

FastAPI + LangGraph pipeline: query generation → Tavily search → fact extraction → cited Markdown report, with optional second pass, SQLite job history, and Q&A on completed reports.

## Setup

- Python 3.11+
- Copy `.env.example` to `.env` and set `TAVILY_API_KEY`. For full LLM features, set `ANTHROPIC_API_KEY` or `LLM_PROVIDER=heuristic` to avoid Anthropic.
- `pip install -r requirements.txt`
- For tests/CI: `pip install -r requirements-dev.txt`

## Tests & CI

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Or run the full local check (matches [GitHub Actions](.github/workflows/ci.yml)):

- **Windows:** `powershell -ExecutionPolicy Bypass -File scripts/verify.ps1`
- **macOS/Linux:** `bash scripts/verify.sh`

Pushes to `main` run automated tests on GitHub.

## Run the web UI

```bash
python main.py serve
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (or your host/port).

## Run once from the CLI

```bash
python main.py "your research topic"
```

## License

Add your license here if applicable.
