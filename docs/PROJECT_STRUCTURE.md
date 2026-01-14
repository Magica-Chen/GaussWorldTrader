# Gauss World Trader - Project Structure

## Overview

This repo is simplified around two entry points:
- `main_cli.py` for CLI workflows
- `dashboard.py` for the Streamlit dashboard

## Directory Structure

```
GaussWorldTrader/
├── main_cli.py            # CLI entry point
├── dashboard.py           # Streamlit launcher
├── README.md              # Project documentation
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Dependencies
├── watchlist.json         # Default watchlist
│
├── config/                # Configuration management
│   ├── __init__.py
│   └── config.py
│
├── src/                   # Source code
│   ├── account/           # Account and portfolio management
│   ├── agent/             # AI/LLM analysis utilities
│   ├── analysis/          # Indicators and analytics
│   ├── data/              # Market data providers
│   ├── option_strategy/   # Options strategies (e.g., wheel)
│   ├── strategy/          # Strategy template + built-ins
│   ├── trade/             # Trading engine + backtester
│   ├── ui/                # Streamlit dashboard
│   └── utils/             # Shared utilities
│
├── examples/              # Example scripts
├── tests/                 # Smoke tests
└── docs/                  # Documentation
```

## Strategy Template

All strategies live in `src/strategy/strategies.py` and share a common template
in `src/strategy/template.py`.

Key concepts:
- `StrategyMeta`: metadata (name, label, category, dashboard visibility)
- `StrategyBase`: base class with `generate_signals`
- `StrategyRegistry`: factory and list utilities

## Entry Points

### CLI
```
python main_cli.py list-strategies
python main_cli.py run-strategy momentum AAPL MSFT
python main_cli.py backtest momentum AAPL
```

### Dashboard
```
python dashboard.py
```
