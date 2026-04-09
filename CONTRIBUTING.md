# Contributing to Gauss World Trader

Thank you for considering contributing to Gauss World Trader! This guide covers everything you need to know to get started.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Running the Linter](#running-the-linter)
- [Adding a Strategy](#adding-a-strategy)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## Code of Conduct

Be respectful, constructive, and collaborative. Harassment of any kind is not tolerated.

---

## Development Setup

**Requirements:** Python 3.12 or higher.

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/GaussWorldTrader.git
cd GaussWorldTrader

# 2. Create a virtual environment (conda recommended)
conda create -n gaussworldtrader-dev python=3.12
conda activate gaussworldtrader-dev

# 3. Install the package with dev extras
pip install -e ".[dev]"

# 4. Copy and fill in the environment file
cp .env.example .env
```

---

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and [Black](https://black.readthedocs.io/) for formatting. Both are installed as part of the `dev` extras.

Key conventions:
- Line length: **100 characters**
- Target Python version: **3.12+**
- Use type annotations for all public functions and methods
- Follow the existing pattern of docstrings (Google-style, brief)

---

## Running the Linter

```bash
# Check for lint errors
ruff check .

# Auto-fix safe issues
ruff check --fix .

# Format with Black
black .

# Type-check with mypy
mypy src/
```

All pull requests must pass `ruff check .` without errors before they can be merged.

---

## Adding a Strategy

1. Create a new file in the appropriate directory:
   - `src/strategy/stock/` for stock strategies
   - `src/strategy/crypto/` for crypto strategies
   - `src/strategy/option/` for options strategies

2. Subclass `StrategyBase` (or `BaseOptionStrategy` for options) and fill in `meta` and `summary`:

```python
from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal

class MyStrategy(StrategyBase):
    meta = StrategyMeta(
        name="my_strategy",
        label="My Strategy",
        category="signal",
        description="One-line description of what this strategy does.",
        visible_in_dashboard=True,
        default_params={"lookback": 20},
    )
    summary = "Brief intro explaining the logic, indicators used, and intended market regime."

    def get_signal(self, symbol, current_date, current_price, current_data, historical_data, portfolio=None):
        ...  # compute indicators, return a SignalSnapshot

    def get_action_plan(self, signal, current_price, current_date):
        ...  # translate the snapshot into an ActionPlan

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        ...  # legacy wrapper used by dashboard and backtests
```

3. Register the strategy in `src/strategy/registry.py` and export it from the relevant `__init__.py`.

4. Verify it appears in the CLI:

```bash
python main_cli.py list-strategies
```

5. Run a quick backtest to confirm correctness:

```bash
python main_cli.py backtest --strategy my_strategy AAPL --days 60
```

---

## Pull Request Process

1. **Branch** off `main` using a descriptive name (e.g., `feature/add-rsi-strategy` or `fix/momentum-signal-edge-case`).
2. **Make focused commits** — one logical change per commit.
3. **Update documentation** if you change public interfaces or add new features.
4. **Open a PR** against `main`, fill in the description template, and link any related issues.
5. Address any review feedback before merging.

---

## Reporting Bugs

Please [open an issue](https://github.com/Magica-Chen/GaussWorldTrader/issues/new) and include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behaviour
- Relevant log output or error traceback

---

## Requesting Features

Open an issue with the `enhancement` label and describe:

- The problem you are trying to solve
- Your proposed solution or API
- Any relevant context (strategy logic, asset type, etc.)
