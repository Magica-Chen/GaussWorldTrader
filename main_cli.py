#!/usr/bin/env python3
"""
Simple CLI entry point for Gauss World Trader.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import typer

from src.data import AlpacaDataProvider
from src.strategy import get_strategy_registry
from src.trade import Backtester, TradingEngine
from src.utils.timezone_utils import now_et
from src.utils.watchlist_manager import WatchlistManager

app = typer.Typer(add_completion=False)


def _load_symbols(symbols: Optional[List[str]]) -> List[str]:
    if symbols:
        return [s.upper() for s in symbols]
    watchlist_path = Path("watchlist.json")
    if watchlist_path.exists():
        manager = WatchlistManager()
        return [s.upper() for s in manager.get_watchlist(asset_type="stock")]
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


@app.command("stream-market")
def stream_market(
    symbols: Optional[List[str]] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Symbols to stream (repeatable or space-separated)",
    ),
    asset_type: str = typer.Option(
        "stock",
        "--asset-type",
        help="Asset type to stream: stock, crypto, or option",
    ),
    crypto_loc: str = typer.Option(
        "eu-1",
        "--crypto-loc",
        help="Crypto feed location: us, us-1, eu-1",
    ),
    stream_type: str = typer.Option("trades", help="trades, quotes, or bars"),
    max_messages: int = typer.Option(0, help="Stop after N messages (0 = unlimited)"),
    raw: bool = typer.Option(False, help="Print raw stream payloads"),
) -> None:
    """Stream live market data from Alpaca (CLI only)."""
    provider = AlpacaDataProvider()
    symbols_list = _load_symbols(symbols)
    if len(symbols_list) > 30:
        print("Alpaca basic accounts support up to 30 symbols per websocket.")
        raise typer.Exit(1)

    asset_type = asset_type.strip().lower()
    if asset_type not in {"stock", "crypto", "option"}:
        print("asset-type must be one of: stock, crypto, option")
        raise typer.Exit(1)

    stream_type = stream_type.strip().lower()
    if stream_type not in {"trades", "quotes", "bars"}:
        print("stream-type must be one of: trades, quotes, bars")
        raise typer.Exit(1)

    if asset_type == "crypto":
        try:
            stream = provider.create_crypto_stream(raw_data=raw, loc=crypto_loc)
        except ValueError as exc:
            print(exc)
            raise typer.Exit(1)
    elif asset_type == "option":
        stream = provider.create_option_stream(raw_data=raw)
    else:
        stream = provider.create_stock_stream(raw_data=raw)
    max_messages = max_messages if max_messages > 0 else None
    message_count = {"count": 0}

    def _get_field(data, attr: str, raw_key: str):
        if hasattr(data, attr):
            return getattr(data, attr)
        if isinstance(data, dict):
            return data.get(raw_key)
        return None

    def _maybe_stop() -> None:
        if max_messages and message_count["count"] >= max_messages:
            stream.stop()

    async def handle_trade(data) -> None:
        message_count["count"] += 1
        if raw:
            print(data, flush=True)
        else:
            symbol = _get_field(data, "symbol", "S")
            price = _get_field(data, "price", "p")
            size = _get_field(data, "size", "s")
            timestamp = _get_field(data, "timestamp", "t")
            exchange = _get_field(data, "exchange", "x")
            print(f"{symbol} trade {price} x{size} @ {timestamp} {exchange}", flush=True)
        _maybe_stop()

    async def handle_quote(data) -> None:
        message_count["count"] += 1
        if raw:
            print(data, flush=True)
        else:
            symbol = _get_field(data, "symbol", "S")
            bid_price = _get_field(data, "bid_price", "bp")
            bid_size = _get_field(data, "bid_size", "bs")
            ask_price = _get_field(data, "ask_price", "ap")
            ask_size = _get_field(data, "ask_size", "as")
            timestamp = _get_field(data, "timestamp", "t")
            print(
                f"{symbol} quote {bid_price}@{bid_size} / {ask_price}@{ask_size} @ {timestamp}",
                flush=True,
            )
        _maybe_stop()

    async def handle_bar(data) -> None:
        message_count["count"] += 1
        if raw:
            print(data, flush=True)
        else:
            symbol = _get_field(data, "symbol", "S")
            close = _get_field(data, "close", "c")
            volume = _get_field(data, "volume", "v")
            timestamp = _get_field(data, "timestamp", "t")
            print(f"{symbol} bar close={close} volume={volume} @ {timestamp}", flush=True)
        _maybe_stop()

    if stream_type == "trades":
        stream.subscribe_trades(handle_trade, *symbols_list)
    elif stream_type == "quotes":
        stream.subscribe_quotes(handle_quote, *symbols_list)
    else:
        stream.subscribe_bars(handle_bar, *symbols_list)

    print(
        f"Streaming {asset_type} {stream_type} for {', '.join(symbols_list)}. Ctrl+C to stop.",
        flush=True,
    )
    try:
        stream.run()
    except KeyboardInterrupt:
        stream.stop()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
