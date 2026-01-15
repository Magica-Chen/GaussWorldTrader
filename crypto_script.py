#!/usr/bin/env python3
"""Live cryptocurrency trading script using momentum signals."""
from __future__ import annotations

import argparse
import logging

from src.trade.live_trading_crypto import LiveTradingCrypto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live cryptocurrency trading script.")
    parser.add_argument("--symbol", default="BTC/USD", help="Crypto pair to trade (e.g., BTC/USD).")
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    trader = LiveTradingCrypto(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        crypto_loc=args.crypto_loc,
        risk_pct=args.risk_pct,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        execute=args.execute,
        auto_exit=args.auto_exit,
    )
    trader.logger.info(
        "Live trading %s (execute=%s, auto_exit=%s)", trader.symbol, args.execute, args.auto_exit
    )
    trader.start()


if __name__ == "__main__":
    main()
