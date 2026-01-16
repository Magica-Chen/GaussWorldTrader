#!/usr/bin/env python3
"""Live stock trading script using momentum signals."""
from __future__ import annotations

import argparse
import logging
from typing import List

from src.trade.live_trading_stock import LiveTradingStock
from src.trade.live_runner import run_live_engines
from src.trade.stock_engine import TradingStockEngine
from src.utils.live_utils import merge_symbol_sources, parse_symbol_args, positions_for_asset_type
from src.utils.timezone_utils import format_duration
from src.utils.watchlist_manager import WatchlistManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live stock trading script.")
    parser.add_argument("--symbol", help="Stock ticker to trade.")
    parser.add_argument(
        "--symbols",
        action="append",
        help="Comma-separated tickers (e.g., AAPL,MSFT). Can be repeated.",
    )
    parser.add_argument("--timeframe", default="1Hour", help="Bar timeframe for signals.")
    parser.add_argument("--lookback-days", type=int, default=30, help="Historical lookback days.")
    parser.add_argument("--risk-pct", type=float, default=0.05, help="Portfolio risk per trade.")
    parser.add_argument("--stop-loss-pct", type=float, default=0.03, help="Stop-loss percent.")
    parser.add_argument("--take-profit-pct", type=float, default=0.06, help="Take-profit percent.")
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute live trades (use --no-execute for dry run).",
    )
    parser.add_argument(
        "--auto-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-close on stop/take (use --no-auto-exit to only monitor).",
    )
    parser.add_argument(
        "--fractional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow fractional shares (default: whole shares only).",
    )
    parser.add_argument(
        "--extended-hours",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trade during extended hours (4AM-8PM ET).",
    )
    return parser.parse_args()


def _get_default_symbols() -> List[str]:
    manager = WatchlistManager()
    watchlist_symbols = manager.get_watchlist(asset_type="stock")
    position_symbols: List[str] = []
    try:
        engine = TradingStockEngine()
        position_symbols = positions_for_asset_type(engine.get_current_positions(), "stock")
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load stock positions: %s", exc)
    defaults = merge_symbol_sources("stock", watchlist_symbols, position_symbols)
    return defaults or ["AAPL"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    parsed_symbols = parse_symbol_args(args.symbols, args.symbol)
    symbols = merge_symbol_sources(
        "stock",
        parsed_symbols if parsed_symbols else _get_default_symbols(),
    )
    engines = [
        LiveTradingStock(
            symbol=symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            risk_pct=args.risk_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            execute=args.execute,
            auto_exit=args.auto_exit,
            allow_fractional=args.fractional,
            extended_hours=args.extended_hours,
        )
        for symbol in symbols
    ]
    if engines and not engines[0].is_market_open():
        remaining = engines[0].seconds_until_market_open()
        logging.warning(
            "NOT in market period. Market opens in %s", format_duration(remaining)
        )
        return
    for engine in engines:
        engine.logger.info(
            "Live trading %s (execute=%s, auto_exit=%s, fractional=%s, extended=%s)",
            engine.symbol,
            args.execute,
            args.auto_exit,
            args.fractional,
            args.extended_hours,
        )
    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)


if __name__ == "__main__":
    main()
