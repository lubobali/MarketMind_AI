# MarketMind AI — Project Instructions

## Overview
- **MarketMind AI**: Real-time stock market streaming analytics + AI agent
- **Capstone project** for Databricks/Spark bootcamp
- **GitHub**: https://github.com/lubobali/MarketMind_AI
- **Forgejo**: git.lubot.ai (mirrored)
- **Architecture**: Yahoo Finance → Kafka → Spark Structured Streaming → Delta Live Tables → AI Agent → Gradio UI

## Tech Stack
- Databricks (clusters, notebooks, Delta Live Tables, MLflow)
- Apache Spark (Structured Streaming, DataFrames, SQL)
- Delta Lake (medallion architecture: Bronze → Silver → Gold)
- Kafka (real-time ingestion)
- MLflow (model tracking + serving)
- Gradio (chat UI)
- Plotly (charts)
- yfinance (stock data)
- VADER/TextBlob (sentiment analysis)

## How to Build & Test (RECR Loop)
**Follow these rules automatically every session. No reminders needed.**

### Commands
- Run all tests: `python3 -m pytest tests/ -q`
- Run specific: `python3 -m pytest -q -k test_name`
- Lint: `ruff check .` | Format: `ruff format .`
- Pre-commit hooks auto-run ruff + pytest on every commit
- Forgejo CI runs full suite on every push

### RECR Loop — How to Write Code
1. **R**equirements: Write the TEST first that defines the behavior
2. **E**xecute: Implement ONE task to make that test pass
3. **C**heck: Run tests, verify green
4. **R**epeat: Next task
- Keep each task SHORT — one test at a time
- Only accept changes that move a test from red to green

### What to Mock vs What to Keep Real
- **MOCK**: External APIs (yfinance, NewsAPI, Spark session in unit tests)
- **REAL**: Internal logic (schemas, transformations, sentiment scoring, UDFs)
- "Mock external boundaries, not internal business logic"

### Test Patterns
- **Parametrize**: `@pytest.mark.parametrize` for multiple inputs
- **Fixtures**: conftest.py for shared test data
- **Markers**: @pytest.mark.fast (unit), @pytest.mark.slow (integration/Spark)

### Quality Rules
- ZERO test failures before moving forward — no exceptions
- ALL new code: write test FIRST, then implement
- Never weaken a test to make it pass — fix the implementation
- CI must pass before any merge to main

### CI Rules
- CI installs from `requirements.txt` — same deps everywhere
- Never skip tests or ignore directories in CI
- Any new pip dependency MUST be added to `requirements.txt` immediately
- If CI passes, code is good. If CI fails, fix before merging.

## Folder Structure
```
marketmind-ai/
├── notebooks/          # Databricks notebooks (9 total)
├── config/             # Schemas, settings
├── utils/              # Technical indicators, sentiment helpers
├── tests/              # All tests (unit + integration)
│   ├── unit/
│   └── integration/
├── screenshots/        # Pipeline DAG, charts, UI screenshots
├── pyproject.toml      # pytest + ruff config
├── requirements.txt    # Python dependencies
└── CLAUDE.md           # This file
```
