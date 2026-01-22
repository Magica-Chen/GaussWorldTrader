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
│   ├── agent/             # AI/LLM analysis, watchlist, live utils
│   │   ├── watchlist_manager.py  # Watchlist management
│   │   ├── live_utils.py         # Live trading utilities
│   │   └── ...
│   ├── analysis/          # Indicators and analytics
│   ├── data/              # Market data providers
│   ├── option_strategy/   # Options strategies (e.g., wheel)
│   ├── pic/               # Images used in code (logo2.png)
│   ├── script/            # Live trading modules per asset
│   ├── strategy/          # Strategy template + built-ins
│   ├── trade/             # Trading engine + backtester
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
