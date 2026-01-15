#!/usr/bin/env python3
"""
Wheel strategy example using live Alpaca option chain data.

Requirements:
- ALPACA_API_KEY and ALPACA_SECRET_KEY set in your environment or .env
- Options data entitlement on Alpaca
"""

import os
import sys
from datetime import timedelta
from typing import Dict, List, Tuple

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import Config
from src.data import AlpacaDataProvider
from src.strategy import WheelStrategy
from src.trade import Portfolio
from src.utils.timezone_utils import now_et


def build_market_snapshot(
    provider: AlpacaDataProvider,
    symbols: List[str],
    timeframe: str,
    lookback_days: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
    start_date = now_et() - timedelta(days=lookback_days)
    current_prices: Dict[str, float] = {}
    current_data: Dict[str, Dict[str, float]] = {}
    historical_data: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        data = provider.get_bars(symbol, timeframe, start_date)
        if data.empty:
            continue

        latest = data.iloc[-1]
        current_prices[symbol] = float(latest["close"])
        current_data[symbol] = {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
        }
        historical_data[symbol] = data

    return current_prices, current_data, historical_data


def main() -> int:
    if not Config.validate_alpaca_config():
        print("Missing Alpaca credentials.")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment or .env.")
        return 1

    symbols = sys.argv[1:] or ["AAPL"]
    timeframe = "1Day"
    lookback_days = 90

    provider = AlpacaDataProvider()
    strategy = WheelStrategy()
    strategy.symbol_list = symbols
    portfolio = Portfolio(initial_cash=100000)

    current_prices, current_data, historical_data = build_market_snapshot(
        provider, symbols, timeframe, lookback_days
    )

    if not current_prices:
        print("No equity data returned for the requested symbols.")
        return 1

    signals = strategy.generate_signals(
        current_date=now_et(),
        current_prices=current_prices,
        current_data=current_data,
        historical_data=historical_data,
        portfolio=portfolio,
    )

    if not signals:
        print("No wheel signals generated.")
        print("If you expected signals, confirm options data access on Alpaca.")
        return 0

    print("Wheel signals:")
    for signal in signals:
        strike = signal.get("strike_price")
        exp_date = signal.get("expiration_date")
        premium = signal.get("premium")
        option_type = signal.get("option_type", "")
        print(
            f"- {signal.get('symbol')}: {signal.get('action')} "
            f"qty={signal.get('quantity')} type={option_type} "
            f"strike={strike} exp={exp_date} premium={premium} "
            f"reason={signal.get('reason', '')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
