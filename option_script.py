#!/usr/bin/env python3
"""Live options trading script using wheel strategy."""
from __future__ import annotations

import argparse
import logging
from typing import List

from src.trade.live_trading_option import LiveTradingOption
from src.trade.live_runner import run_live_engines
from src.trade.option_engine import TradingOptionEngine
from src.utils.live_utils import merge_symbol_sources, parse_symbol_args
from src.utils.timezone_utils import format_duration
from src.utils.watchlist_manager import WatchlistManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live options trading script.")
    parser.add_argument("--underlying", help="Underlying stock symbol.")
    parser.add_argument(
        "--symbols",
        action="append",
        help="Comma-separated underlying symbols (e.g., AAPL,MSFT). Can be repeated.",
    )
    parser.add_argument("--timeframe", default="1Day", help="Bar timeframe for signals.")
    parser.add_argument("--lookback-days", type=int, default=30, help="Historical lookback days.")
    parser.add_argument("--risk-pct", type=float, default=0.08, help="Portfolio risk per trade.")
    parser.add_argument("--stop-loss-pct", type=float, default=0.50, help="Stop-loss percent.")
    parser.add_argument("--take-profit-pct", type=float, default=0.50, help="Take-profit percent.")
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
        "--roll-days",
        type=int,
        default=5,
        help="Days before expiry to warn about rolling positions.",
    )
    return parser.parse_args()


def _get_default_symbols() -> List[str]:
    manager = WatchlistManager()
    watchlist_symbols = manager.get_watchlist(asset_type="option")
    position_symbols: List[str] = []
    try:
        engine = TradingOptionEngine()
        for pos in engine.get_option_positions():
            underlying = pos.get("underlying")
            if underlying:
                position_symbols.append(underlying)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load option positions: %s", exc)
    defaults = merge_symbol_sources("option", watchlist_symbols, position_symbols)
    return defaults or ["AAPL"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    parsed_symbols = parse_symbol_args(args.symbols, args.underlying)
    underlyings = merge_symbol_sources(
        "option",
        parsed_symbols if parsed_symbols else _get_default_symbols(),
    )
    engines = [
        LiveTradingOption(
            underlying_symbol=underlying,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            risk_pct=args.risk_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            execute=args.execute,
            auto_exit=args.auto_exit,
            roll_days_before_expiry=args.roll_days,
        )
        for underlying in underlyings
    ]
    if engines and not engines[0].is_market_open():
        remaining = engines[0].seconds_until_market_open()
        logging.warning(
            "NOT in market period. Market opens in %s", format_duration(remaining)
        )
        return
    for engine in engines:
        engine.logger.info(
            "Live options trading on %s (execute=%s, auto_exit=%s, roll_days=%d)",
            engine.underlying_symbol,
            args.execute,
            args.auto_exit,
            args.roll_days,
        )
    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)


if __name__ == "__main__":
    main()
