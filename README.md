# Deep Research Agent

FastAPI + LangGraph pipeline: query generation → Tavily search → fact extraction → cited Markdown report, with optional second pass, SQLite job history, and Q&A on completed reports.

## Setup

- Python 3.11+
- Copy `.env.example` to `.env` and set `TAVILY_API_KEY`. For full LLM features, set `ANTHROPIC_API_KEY` or `LLM_PROVIDER=heuristic` to avoid Anthropic.
- `pip install -r requirements.txt`

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
