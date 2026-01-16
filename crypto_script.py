#!/usr/bin/env python3
"""Live cryptocurrency trading script using momentum signals."""
from __future__ import annotations

import argparse
import logging
from typing import List

from src.trade.crypto_engine import TradingCryptoEngine
from src.trade.live_trading_crypto import LiveTradingCrypto
from src.trade.live_runner import run_live_engines
from src.utils.live_utils import merge_symbol_sources, parse_symbol_args
from src.utils.watchlist_manager import WatchlistManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live cryptocurrency trading script.")
    parser.add_argument("--symbol", help="Crypto pair to trade (e.g., BTC/USD).")
    parser.add_argument(
        "--symbols",
        action="append",
        help="Comma-separated crypto pairs (e.g., BTC/USD,ETH/USD). Can be repeated.",
    )
    parser.add_argument("--timeframe", default="1Hour", help="Bar timeframe for signals.")
    parser.add_argument("--lookback-days", type=int, default=14, help="Historical lookback days.")
    parser.add_argument("--crypto-loc", default="us", help="Crypto stream feed: us, us-1, eu-1.")
    parser.add_argument("--risk-pct", type=float, default=0.10, help="Portfolio risk per trade.")
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
    return parser.parse_args()


def _get_default_symbols() -> List[str]:
    manager = WatchlistManager()
    watchlist_symbols = manager.get_watchlist(asset_type="crypto")
    position_symbols: List[str] = []
    try:
        engine = TradingCryptoEngine()
        position_symbols = [
            pos.get("symbol")
            for pos in engine.get_crypto_positions()
            if pos.get("symbol")
        ]
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load crypto positions: %s", exc)
    defaults = merge_symbol_sources("crypto", watchlist_symbols, position_symbols)
    return defaults or ["BTC/USD"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    parsed_symbols = parse_symbol_args(args.symbols, args.symbol)
    symbols = merge_symbol_sources(
        "crypto",
        parsed_symbols if parsed_symbols else _get_default_symbols(),
    )
    engines = [
        LiveTradingCrypto(
            symbol=symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            crypto_loc=args.crypto_loc,
            risk_pct=args.risk_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            execute=args.execute,
            auto_exit=args.auto_exit,
        )
        for symbol in symbols
    ]
    for engine in engines:
        engine.logger.info(
            "Live trading %s (execute=%s, auto_exit=%s)",
            engine.symbol,
            args.execute,
            args.auto_exit,
        )
    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)


if __name__ == "__main__":
    main()
