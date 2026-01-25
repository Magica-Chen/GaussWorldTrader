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
│   ├── account/           # Account and position management
│   ├── agent/             # LLM helpers, watchlist, notifications
│   │   ├── watchlist_manager.py  # Watchlist management
│   │   └── ...
│   ├── analysis/          # Technical analysis (metrics re-exported)
│   ├── data/              # Market data providers
│   ├── pic/               # Images used in code (logo2.png)
│   ├── strategy/          # Strategy base, registry, per-asset strategies
│   ├── trade/             # Trading engines, execution, backtester, live trading, portfolio analytics
│   │   ├── execution.py   # ExecutionEngine (sizing, order type, account limits)
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
- `StrategyBase`, `BaseOptionStrategy`, `SignalSnapshot`, `ActionPlan`, and helpers in `src/strategy/base.py`
- `StrategyRegistry` in `src/strategy/registry.py`
- `crypto_momentum` is a factory alias for `MomentumStrategy` with crypto defaults

Strategy flow:
- `get_signal()` computes indicators and detects signals.
- `get_action_plan()` turns signals into an abstract plan (target price, stop loss, take profit, intent).
- `generate_signals()` remains as a legacy wrapper for dashboard/backtests.

## Execution Layer

The execution layer converts action plans into concrete orders:
- `ExecutionEngine` in `src/trade/execution.py` sizes quantity, applies order type policy, and enforces
  account constraints (fractional trading, shorting, margin).
- Order type `auto` uses limit orders when a target price is present (with minimal price improvement);
  otherwise it uses market orders.

## Live Trading Helpers

Live trading modules live in `src/trade/`:
- `src/trade/live_trading_stock.py`
- `src/trade/live_trading_crypto.py`
- `src/trade/live_trading_option.py`
- `src/trade/live_trading_base.py` (strategy -> plan -> execution loop)
- `src/trade/live_runner.py` (shared websocket runner)

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

### Live Trading CLI
```
python live_script.py
```
