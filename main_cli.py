#!/usr/bin/env python3
"""
Simple CLI entry point for Gauss World Trader.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import typer

from src.account.account_manager import AccountManager
from src.backtest import Backtester
from src.data import AlpacaDataProvider
from src.settings import get_alpaca_base_url
from src.strategy import get_strategy_registry
from src.trade.engine import ExecutionEngine, TradingStockEngine
from src.utils.timezone_utils import now_et
from src.watchlist import WatchlistManager

app = typer.Typer(add_completion=False)


def _configure_numba_cache() -> None:
    if os.getenv("NUMBA_CACHE_DIR"):
        return
    cache_dir = Path("/tmp/gaussworldtrader-numba-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


def _load_symbols(symbols: list[str] | None) -> list[str]:
    if symbols:
        return [s.upper() for s in symbols]
    watchlist_path = Path("watchlist.json")
    if watchlist_path.exists():
        manager = WatchlistManager()
        return [s.upper() for s in manager.get_watchlist(asset_type="stock")]
    return ["AAPL", "MSFT", "GOOGL"]


def _parse_quantity_overrides(entries: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for entry in entries:
        if not entry or "=" not in entry:
            continue
        symbol, qty = entry.split("=", 1)
        symbol = symbol.strip().upper()
        try:
            overrides[symbol] = float(qty.strip())
        except ValueError:
            continue
    return overrides


def _coerce_param_value(value: str):
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." not in text and "e" not in lowered:
            return int(text)
        return float(text)
    except ValueError:
        return text


def _parse_strategy_params(entries: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for entry in entries:
        if not entry or "=" not in entry:
            continue
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            continue
        params[key] = _coerce_param_value(raw_value)
    return params


def _print_backtest_results(results: dict[str, object]) -> None:
    summary = results.get("summary", results)
    print("Backtest Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    benchmark = results.get("benchmark")
    if isinstance(benchmark, dict):
        print("\nBenchmark:")
        for key, value in benchmark.items():
            print(f"{key}: {value}")

    splits = results.get("split_summaries")
    if isinstance(splits, list):
        print("\nWalk-Forward Splits:")
        for split in splits:
            split_id = split.get("split", "?")
            total_return = split.get("total_return_percentage")
            sharpe = split.get("sharpe_ratio")
            drawdown = split.get("max_drawdown_percentage")
            print(
                f"split {split_id}: return={total_return}, "
                f"sharpe={sharpe}, max_drawdown={drawdown}"
            )


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
    engine = TradingStockEngine()
    info = engine.get_account_info()
    if not info:
        raise typer.Exit(1)
    for key, value in info.items():
        print(f"{key}: {value}")


@app.command("run-strategy")
def run_strategy(
    strategy: str = typer.Option("momentum", help="Strategy name"),
    symbols: list[str] | None = typer.Argument(None, help="Symbols to run"),
    days: int = typer.Option(60, help="Lookback window in days"),
    execute: bool = typer.Option(False, help="Place orders for signals"),
    order_type: str = typer.Option(
        "auto",
        help="Order type: auto, market, or limit",
    ),
    allow_sell_to_open: bool = typer.Option(
        False,
        help="Allow sell-to-open (shorting) when account supports it",
    ),
    quantity: list[str] | None = typer.Option(
        None,
        "--quantity",
        "-q",
        help="Per-symbol quantity override, e.g. AAPL=10",
    ),
    strategy_param: list[str] | None = typer.Option(
        None,
        "--strategy-param",
        "-p",
        help="Strategy param override, e.g. risk_pct=0.03",
    ),
) -> None:
    """Run a strategy on recent data and print signals."""
    registry = get_strategy_registry()
    symbols_list = _load_symbols(symbols)
    strategy_params = _parse_strategy_params(strategy_param or [])
    evaluation_time = now_et()

    provider = AlpacaDataProvider()
    start_date = evaluation_time - timedelta(days=days)

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
        strategy_instance = registry.create(strategy, strategy_params or None)
    except KeyError as exc:
        print(exc)
        raise typer.Exit(1) from exc
    qty_overrides = _parse_quantity_overrides(quantity or [])
    order_type = order_type.lower().strip()
    if order_type not in {"auto", "market", "limit"}:
        print("order-type must be one of: auto, market, limit")
        raise typer.Exit(1)

    account_manager = AccountManager(base_url=get_alpaca_base_url())
    account_config = account_manager.get_account_configurations()
    fractional_enabled = bool(account_config.get("fractional_trading", False))

    engine = TradingStockEngine(allow_fractional=fractional_enabled)
    executor = ExecutionEngine(
        trading_engine=engine,
        asset_type="stock",
        allow_sell_to_open=allow_sell_to_open,
        order_type=order_type,
        execute=execute,
        account_manager=account_manager,
    )
    context = executor.load_context()
    positions = {p.get("symbol"): p for p in engine.get_current_positions()}

    risk_pct = float(strategy_instance.params.get("risk_pct", 0.05))
    decisions = []

    for symbol in symbols_list:
        price = current_prices.get(symbol)
        data = historical_data.get(symbol)
        if price is None or data is None:
            continue
        snapshot = strategy_instance.get_signal(
            symbol=symbol,
            current_date=evaluation_time,
            current_price=price,
            current_data=current_data.get(symbol, {}),
            historical_data=data,
            portfolio=None,
        )
        if snapshot is None:
            continue
        plan = strategy_instance.get_action_plan(snapshot, price, evaluation_time)
        if not plan or plan.action == "HOLD":
            continue
        decision = executor.build_decision(
            action_plan=plan,
            context=context,
            position=positions.get(symbol, {"qty": 0.0, "side": "flat"}),
            risk_pct=risk_pct,
            current_price=price,
            override_qty=qty_overrides.get(symbol),
            order_pref=order_type,
        )
        if decision is None:
            continue
        decisions.append(decision)

    if not decisions:
        print("No actionable decisions.")
        return

    print("Decisions:")
    for decision in decisions:
        print(decision)

    for decision in decisions:
        executor.execute_decision(decision)


@app.command("backtest")
def backtest(
    strategy: str = typer.Option("momentum", help="Strategy name"),
    symbols: list[str] | None = typer.Argument(None, help="Symbols to backtest"),
    days: int = typer.Option(365, help="Backtest window"),
    initial_cash: float = typer.Option(100000, help="Initial cash"),
    walk_forward: bool = typer.Option(False, help="Run walk-forward evaluation"),
    splits: int = typer.Option(5, help="Walk-forward splits"),
    benchmark: str | None = typer.Option(
        None,
        help="Optional benchmark symbol for buy-and-hold comparison",
    ),
    strategy_param: list[str] | None = typer.Option(
        None,
        "--strategy-param",
        "-p",
        help="Strategy param override, e.g. mode=fast",
    ),
) -> None:
    """Run a simple backtest."""
    registry = get_strategy_registry()
    symbols_list = _load_symbols(symbols)
    strategy_params = _parse_strategy_params(strategy_param or [])
    if strategy == "multi_agent" and "mode" not in strategy_params:
        strategy_params["mode"] = "fast"
    benchmark_symbol = benchmark.upper() if benchmark else None

    provider = AlpacaDataProvider()
    start_date = now_et() - timedelta(days=days)

    backtester = Backtester(initial_cash=initial_cash, commission=0.01)
    fetch_symbols = list(symbols_list)
    if benchmark_symbol and benchmark_symbol not in fetch_symbols:
        fetch_symbols.append(benchmark_symbol)
    for symbol in fetch_symbols:
        data = provider.get_bars(symbol, "1Day", start_date)
        if data.empty:
            print(f"No data for {symbol}")
            continue
        backtester.add_data(symbol, data)

    try:
        strategy_instance = registry.create(strategy, strategy_params or None)
    except KeyError as exc:
        print(exc)
        raise typer.Exit(1) from exc

    def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
        return strategy_instance.generate_signals(
            current_date, current_prices, current_data, historical_data, portfolio
        )

    if walk_forward:
        results = backtester.run_walk_forward(
            strategy_func,
            splits=splits,
            symbols=symbols_list,
            benchmark_symbol=benchmark_symbol,
            strategy=strategy_instance,
        )
    else:
        results = backtester.run_backtest(
            strategy_func,
            symbols=symbols_list,
            benchmark_symbol=benchmark_symbol,
            strategy=strategy_instance,
        )
    if not results:
        print("Backtest produced no results.")
        return

    _print_backtest_results(results)


@app.command("stream-market")
def stream_market(
    symbols: list[str] | None = typer.Option(
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
            raise typer.Exit(1) from exc
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
    _configure_numba_cache()
    app()


if __name__ == "__main__":
    main()
