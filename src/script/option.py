#!/usr/bin/env python3
"""Live options trading module using wheel strategy."""
from __future__ import annotations

import logging
from typing import List, Optional

from src.trade.live_trading_option import LiveTradingOption
from src.trade.live_runner import run_live_engines
from src.trade.option_engine import TradingOptionEngine
from src.utils.live_utils import merge_symbol_sources
from src.utils.timezone_utils import format_duration
from src.utils.watchlist_manager import WatchlistManager


def get_default_option_symbols() -> List[str]:
    """Get default option underlying symbols from watchlist and open positions."""
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


def create_option_engines(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1Day",
    lookback_days: int = 30,
    risk_pct: float = 0.08,
    stop_loss_pct: float = 0.50,
    take_profit_pct: float = 0.50,
    execute: bool = True,
    auto_exit: bool = True,
    roll_days: int = 5,
    strategy: str = "wheel",
) -> List[LiveTradingOption]:
    """Create option trading engines without starting them.

    Returns:
        List of configured LiveTradingOption engines.
    """
    underlyings = symbols if symbols else get_default_option_symbols()
    underlyings = merge_symbol_sources("option", underlyings)

    return [
        LiveTradingOption(
            underlying_symbol=underlying,
            timeframe=timeframe,
            lookback_days=lookback_days,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execute=execute,
            auto_exit=auto_exit,
            roll_days_before_expiry=roll_days,
            strategy=strategy,
        )
        for underlying in underlyings
    ]


def run_option_trading(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1Day",
    lookback_days: int = 30,
    risk_pct: float = 0.08,
    stop_loss_pct: float = 0.50,
    take_profit_pct: float = 0.50,
    execute: bool = True,
    auto_exit: bool = True,
    roll_days: int = 5,
    strategy: str = "wheel",
) -> None:
    """Run live options trading.

    Args:
        symbols: List of underlying symbols (e.g., ["AAPL", "MSFT"]).
                 If None, uses watchlist and open positions.
        timeframe: Bar timeframe for signals.
        lookback_days: Historical lookback days.
        risk_pct: Portfolio risk per trade.
        stop_loss_pct: Stop-loss percentage.
        take_profit_pct: Take-profit percentage.
        execute: Execute live trades (False for dry run).
        auto_exit: Auto-close on stop/take-profit.
        roll_days: Days before expiry to warn about rolling positions.
        strategy: Strategy name to use for signals.
    """
    engines = create_option_engines(
        symbols=symbols,
        timeframe=timeframe,
        lookback_days=lookback_days,
        risk_pct=risk_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        execute=execute,
        auto_exit=auto_exit,
        roll_days=roll_days,
        strategy=strategy,
    )

    if engines and not engines[0].is_market_open():
        remaining = engines[0].seconds_until_market_open()
        logging.warning(
            "NOT in market period. Market opens in %s", format_duration(remaining)
        )
        return

    for engine in engines:
        engine.logger.info(
            "Live options trading on %s (execute=%s, auto_exit=%s, roll_days=%d)",
            engine.underlying_symbol, execute, auto_exit, roll_days,
        )

    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)
