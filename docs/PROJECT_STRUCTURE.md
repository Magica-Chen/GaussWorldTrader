# Gauss World Trader - Project Structure

## Overview

This repo is simplified around three entry points:
- `main_cli.py` for CLI workflows
- `dashboard.py` for the Streamlit dashboard
- `live_script.py` for the unified live trading CLI

## Directory Structure

```
GaussWorldTrader/
├── main_cli.py            # CLI entry point
├── dashboard.py           # Streamlit launcher
├── live_script.py         # Unified live trading CLI
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
│   ├── agent/             # AI/LLM analysis, watchlist, notifications
│   │   ├── watchlist_manager.py  # Watchlist management
│   │   └── ...
│   ├── analysis/          # Indicators and analytics
│   ├── data/              # Market data providers
│   ├── pic/               # Images used in code (logo2.png)
│   ├── strategy/          # Strategy template + built-ins
│   ├── trade/             # Trading engine + backtester + live trading helpers
│   ├── ui/                # Streamlit dashboard (mixin-based)
│   │   ├── dashboard.py        # Main dashboard orchestrator
│   │   ├── market_views.py     # Market overview views
│   │   ├── account_views.py    # Account/portfolio views
│   │   ├── trading_views.py    # Trading/backtest views
│   │   ├── analysis_views.py   # Analysis/news views
│   │   ├── ui_components.py    # Reusable UI components
│   │   └── dashboard_utils.py  # Dashboard utilities
│   └── utils/             # Core utilities (asset_utils, timezone_utils, logger)
│
├── examples/              # Example scripts
├── tests/                 # Smoke tests (gitignored)
└── docs/                  # Documentation and README images
    ├── logo.png
    ├── logo3.png
    ├── screenshot1.png
    └── screenshot2.png
```

## Strategy Template

All strategies live under:
- `src/strategy/stock/`
- `src/strategy/crypto/`
- `src/strategy/option/`

Key concepts:
- `StrategyBase` and specialized base classes in `src/strategy/base.py`
- `StrategyRegistry` in `src/strategy/registry.py`

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
