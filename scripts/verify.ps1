# Local verification (same steps as GitHub Actions CI)
$ErrorActionPreference = "Stop"
$env:TAVILY_API_KEY = "local-verify-synthetic"
$env:ANTHROPIC_API_KEY = ""
$env:LLM_PROVIDER = "heuristic"
$env:ENABLE_SUGGEST_TOPICS = "0"
$env:ENABLE_JOB_DB = "0"
$env:RATE_LIMIT_PER_MIN = "0"
Set-Location $PSScriptRoot\..
python -m pip install -U pip wheel
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m compileall -q -f main.py graph.py nodes.py state.py job_store.py research_extras.py research_options.py
python -m compileall -q -f tests
python -m pytest
Write-Host "OK: verify passed" -ForegroundColor Green
