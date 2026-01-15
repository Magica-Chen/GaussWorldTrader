#!/usr/bin/env python3
"""Live options trading script using wheel strategy."""
from __future__ import annotations

import argparse
import logging

from src.trade.live_trading_option import LiveTradingOption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live options trading script.")
    parser.add_argument("--underlying", default="AAPL", help="Underlying stock symbol.")
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    trader = LiveTradingOption(
        underlying_symbol=args.underlying,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
        risk_pct=args.risk_pct,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        execute=args.execute,
        auto_exit=args.auto_exit,
        roll_days_before_expiry=args.roll_days,
    )
    trader.logger.info(
        "Live options trading on %s (execute=%s, auto_exit=%s, roll_days=%d)",
        trader.underlying_symbol, args.execute, args.auto_exit, args.roll_days
    )
    trader.start()


if __name__ == "__main__":
    main()
