# AGENTS.md

## Cursor Cloud specific instructions

### Services

| Service | Command | Notes |
|---|---|---|
| FastAPI backend + static frontend | `python3 main.py serve --host 0.0.0.0 --port 8000` | Serves both API and SPA at `/static/index.html` |

### Running tests

```
python3 -m pytest -v
```

Tests run offline in heuristic mode with a synthetic `TAVILY_API_KEY` (set in `tests/conftest.py`). No real API keys are needed for tests.

### Environment setup

- Copy `.env.example` to `.env`. The default `LLM_PROVIDER=heuristic` mode works without Anthropic credits.
- `TAVILY_API_KEY` is required for actual research jobs (even in heuristic mode). Without it, the server starts and the UI loads, but `POST /research` returns 503.
- The `python` command is not available; always use `python3`.

### Key endpoints

- `GET /config` — server configuration and feature flags
- `POST /research` — start a research job (needs `TAVILY_API_KEY`)
- `POST /research/suggest-topics` — works without API keys
- `GET /jobs` — list completed jobs (requires `ENABLE_JOB_DB=1`, on by default)
