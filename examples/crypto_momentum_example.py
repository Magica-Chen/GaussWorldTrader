#!/usr/bin/env python3
"""
Crypto momentum strategy example using the current strategy framework.

Strategy logic:
- Short momentum (ROC) = (close - close_n) / close_n
- Long momentum (ROC) = (close - close_n) / close_n
- BUY when short momentum crosses above long momentum by the threshold
- SELL when short momentum crosses below long momentum by the threshold
"""

import os
import sys
from datetime import timedelta
from typing import Dict, Tuple

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import Config
from src.data import AlpacaDataProvider
from src.strategy import CryptoMomentumStrategy
from src.trade import Portfolio
from src.utils.timezone_utils import now_et


def build_market_snapshot(
    provider: AlpacaDataProvider,
    symbol: str,
    timeframe: str,
    lookback_days: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
    start_date = now_et() - timedelta(days=lookback_days)
    data = provider.get_bars(symbol, timeframe, start_date)
    if data.empty:
        raise ValueError(f"No market data returned for {symbol}")

    latest = data.iloc[-1]
    current_prices = {symbol: float(latest["close"])}
    current_data = {
        symbol: {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
        }
    }
    historical_data = {symbol: data}
    return current_prices, current_data, historical_data


def _format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


def main() -> int:
    if not Config.validate_alpaca_config():
        print("Missing Alpaca credentials.")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment or .env.")
        return 1

    symbol = "BTC/USD"
    timeframe = "1Hour"
    lookback_days = 7

    provider = AlpacaDataProvider()
    strategy = CryptoMomentumStrategy()
    portfolio = Portfolio()

    current_prices, current_data, historical_data = build_market_snapshot(
        provider, symbol, timeframe, lookback_days
    )

    latest_price = current_prices[symbol]
    bars = historical_data[symbol]
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Bars: {len(bars)} ({bars.index[0]} to {bars.index[-1]})")
    print(f"Latest price: {_format_price(latest_price)}")

    signals = strategy.generate_signals(
        current_date=now_et(),
        current_prices=current_prices,
        current_data=current_data,
        historical_data=historical_data,
        portfolio=portfolio,
    )

    if not signals:
        print("No signals generated.")
    else:
        print("Signals:")
        for signal in signals:
            print(
                f"- {signal['symbol']}: {signal['action']} "
                f"qty={signal['quantity']} price={_format_price(signal.get('price'))} "
                f"stop={_format_price(signal.get('stop_loss'))} "
                f"take={_format_price(signal.get('take_profit'))} "
                f"reason={signal.get('reason', '')}"
            )

    plan = strategy.generate_trading_plan(
        current_date=now_et(),
        current_prices=current_prices,
        current_data=current_data,
        historical_data=historical_data,
        portfolio=portfolio,
    )

    if not plan:
        print("No trading plan generated.")
    else:
        print("Trading plan:")
        for item in plan:
            print(
                f"- {item['symbol']}: {item['action']} "
                f"qty={item['quantity']} price={_format_price(item.get('price'))} "
                f"plan_type={item.get('plan_type')} strategy={item.get('strategy')} "
                f"reason={item.get('reason', '')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
