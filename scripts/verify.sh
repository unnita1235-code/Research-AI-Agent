#!/usr/bin/env bash
set -euo pipefail
export TAVILY_API_KEY="local-verify-synthetic"
export ANTHROPIC_API_KEY=""
export LLM_PROVIDER="heuristic"
export ENABLE_SUGGEST_TOPICS="0"
export ENABLE_JOB_DB="0"
export RATE_LIMIT_PER_MIN="0"
cd "$(dirname "$0")/.."
python -m pip install -U pip wheel
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m compileall -q -f main.py graph.py nodes.py state.py job_store.py research_extras.py research_options.py
python -m compileall -q -f tests
python -m pytest
echo "OK: verify passed"
