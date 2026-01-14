#!/usr/bin/env python3
"""
Simple CLI entry point for Gauss World Trader.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import typer

from src.data import AlpacaDataProvider
from src.strategy import get_strategy_registry
from src.trade import Backtester, TradingEngine
from src.utils.timezone_utils import now_et

app = typer.Typer(add_completion=False)


def _load_symbols(symbols: Optional[List[str]]) -> List[str]:
    if symbols:
        return [s.upper() for s in symbols]
    watchlist_path = Path("watchlist.json")
    if watchlist_path.exists():
        data = json.loads(watchlist_path.read_text())
        return [s.upper() for s in data.get("watchlist", [])]
    return ["AAPL", "MSFT", "GOOGL"]


@app.command("list-strategies")
def list_strategies(dashboard_only: bool = False) -> None:
    """List available strategies."""
    registry = get_strategy_registry()
    strategies = registry.list_strategies(dashboard_only=dashboard_only)
    for strategy in strategies:
        meta = registry.get_meta(strategy)
        visibility = "dashboard" if meta.visible_in_dashboard else "non-dashboard"
        print(f"{meta.label} ({strategy}) - {visibility}")


@app.command("account-info")
def account_info() -> None:
    """Show account information."""
    engine = TradingEngine()
    info = engine.get_account_info()
    if not info:
        raise typer.Exit(1)
    for key, value in info.items():
        print(f"{key}: {value}")


@app.command("run-strategy")
def run_strategy(
    strategy: str = typer.Option("momentum", help="Strategy name"),
    symbols: Optional[List[str]] = typer.Argument(None, help="Symbols to run"),
    days: int = typer.Option(60, help="Lookback window in days"),
    execute: bool = typer.Option(False, help="Place orders for signals"),
) -> None:
    """Run a strategy on recent data and print signals."""
    registry = get_strategy_registry()
    symbols_list = _load_symbols(symbols)

    provider = AlpacaDataProvider()
    start_date = now_et() - timedelta(days=days)

    historical_data = {}
    current_prices = {}
    current_data = {}

    for symbol in symbols_list:
        data = provider.get_bars(symbol, "1Day", start_date)
        if data.empty:
            print(f"No data for {symbol}")
            continue
        historical_data[symbol] = data
        current_prices[symbol] = float(data["close"].iloc[-1])
        current_data[symbol] = {
            "open": float(data["open"].iloc[-1]),
            "high": float(data["high"].iloc[-1]),
            "low": float(data["low"].iloc[-1]),
            "close": float(data["close"].iloc[-1]),
            "volume": float(data["volume"].iloc[-1]),
        }

    if not historical_data:
        raise typer.Exit(1)

    try:
        strategy_instance = registry.create(strategy)
    except KeyError as exc:
        print(exc)
        raise typer.Exit(1)
    signals = strategy_instance.generate_signals(
        current_date=now_et(),
        current_prices=current_prices,
        current_data=current_data,
        historical_data=historical_data,
        portfolio=None,
    )

    if not signals:
        print("No signals generated.")
        return

    print("Signals:")
    for signal in signals:
        print(signal)

    if execute:
        engine = TradingEngine()
        for signal in signals:
            if signal["action"] == "BUY":
                engine.place_market_order(signal["symbol"], signal["quantity"], side="buy")
            elif signal["action"] == "SELL":
                engine.place_market_order(signal["symbol"], signal["quantity"], side="sell")


@app.command("backtest")
def backtest(
    strategy: str = typer.Option("momentum", help="Strategy name"),
    symbols: Optional[List[str]] = typer.Argument(None, help="Symbols to backtest"),
    days: int = typer.Option(365, help="Backtest window"),
    initial_cash: float = typer.Option(100000, help="Initial cash"),
) -> None:
    """Run a simple backtest."""
    registry = get_strategy_registry()
    symbols_list = _load_symbols(symbols)

    provider = AlpacaDataProvider()
    start_date = now_et() - timedelta(days=days)

    backtester = Backtester(initial_cash=initial_cash, commission=0.01)
    for symbol in symbols_list:
        data = provider.get_bars(symbol, "1Day", start_date)
        if data.empty:
            print(f"No data for {symbol}")
            continue
        backtester.add_data(symbol, data)

    try:
        strategy_instance = registry.create(strategy)
    except KeyError as exc:
        print(exc)
        raise typer.Exit(1)

    def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
        return strategy_instance.generate_signals(
            current_date, current_prices, current_data, historical_data, portfolio
        )

    results = backtester.run_backtest(strategy_func, symbols=symbols_list)
    if not results:
        print("Backtest produced no results.")
        return

    summary = results.get("summary", {})
    print("Backtest Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
