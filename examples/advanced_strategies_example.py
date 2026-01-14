#!/usr/bin/env python3
"""
Advanced Strategies Example

Demonstrates listing strategies and running a simple backtest.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import AlpacaDataProvider
from src.strategy import get_strategy_registry
from src.trade import Backtester
from src.utils.timezone_utils import now_et


def main() -> None:
    registry = get_strategy_registry()

    print("Available strategies:")
    for name in registry.list_strategies():
        meta = registry.get_meta(name)
        print(f"- {meta.label} ({name})")

    symbol = "AAPL"
    strategy_name = "momentum"
    provider = AlpacaDataProvider()

    start_date = now_et() - timedelta(days=180)
    data = provider.get_bars(symbol, "1Day", start_date)
    if data.empty:
        print("No data available for backtest.")
        return

    backtester = Backtester(initial_cash=100000, commission=0.01)
    backtester.add_data(symbol, data)

    strategy_instance = registry.create(strategy_name)

    def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
        return strategy_instance.generate_signals(
            current_date, current_prices, current_data, historical_data, portfolio
        )

    results = backtester.run_backtest(strategy_func, symbols=[symbol])
    summary = results.get("summary", {})

    print("Backtest summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
