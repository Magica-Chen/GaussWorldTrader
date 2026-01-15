#!/usr/bin/env python3
"""Live cryptocurrency trading script using momentum signals."""
from __future__ import annotations

import argparse
import logging
from typing import Iterable, List

from src.trade.live_trading_crypto import LiveTradingCrypto
from src.trade.live_runner import run_live_engines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live cryptocurrency trading script.")
    parser.add_argument("--symbol", default="BTC/USD", help="Crypto pair to trade (e.g., BTC/USD).")
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


def _parse_symbols(symbols: Iterable[str] | None, fallback: str) -> List[str]:
    items = list(symbols) if symbols else [fallback]
    parsed: List[str] = []
    for item in items:
        for part in item.split(","):
            symbol = part.strip()
            if symbol:
                parsed.append(symbol)
    return parsed or [fallback]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    symbols = _parse_symbols(args.symbols, args.symbol)
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
