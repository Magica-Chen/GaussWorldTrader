#!/usr/bin/env python3
"""Live stock trading module using momentum signals."""
from __future__ import annotations

import logging
from typing import List, Optional

from src.trade.live_trading_stock import LiveTradingStock
from src.trade.live_runner import run_live_engines
from src.trade.stock_engine import TradingStockEngine
from src.utils.live_utils import merge_symbol_sources, positions_for_asset_type
from src.utils.timezone_utils import format_duration
from src.utils.watchlist_manager import WatchlistManager


def get_default_stock_symbols() -> List[str]:
    """Get default stock symbols from watchlist and open positions."""
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


def create_stock_engines(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1Hour",
    lookback_days: int = 30,
    risk_pct: float = 0.05,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    execute: bool = True,
    auto_exit: bool = True,
    fractional: bool = False,
    extended_hours: bool = False,
    strategy: str = "momentum",
) -> List[LiveTradingStock]:
    """Create stock trading engines without starting them.

    Returns:
        List of configured LiveTradingStock engines.
    """
    trading_symbols = symbols if symbols else get_default_stock_symbols()
    trading_symbols = merge_symbol_sources("stock", trading_symbols)

    return [
        LiveTradingStock(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execute=execute,
            auto_exit=auto_exit,
            allow_fractional=fractional,
            extended_hours=extended_hours,
            strategy=strategy,
        )
        for symbol in trading_symbols
    ]


def run_stock_trading(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1Hour",
    lookback_days: int = 30,
    risk_pct: float = 0.05,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    execute: bool = True,
    auto_exit: bool = True,
    fractional: bool = False,
    extended_hours: bool = False,
    strategy: str = "momentum",
) -> None:
    """Run live stock trading.

    Args:
        symbols: List of stock tickers (e.g., ["AAPL", "MSFT"]).
                 If None, uses watchlist and open positions.
        timeframe: Bar timeframe for signals.
        lookback_days: Historical lookback days.
        risk_pct: Portfolio risk per trade.
        stop_loss_pct: Stop-loss percentage.
        take_profit_pct: Take-profit percentage.
        execute: Execute live trades (False for dry run).
        auto_exit: Auto-close on stop/take-profit.
        fractional: Allow fractional shares.
        extended_hours: Trade during extended hours (4AM-8PM ET).
        strategy: Strategy name to use for signals.
    """
    engines = create_stock_engines(
        symbols=symbols,
        timeframe=timeframe,
        lookback_days=lookback_days,
        risk_pct=risk_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        execute=execute,
        auto_exit=auto_exit,
        fractional=fractional,
        extended_hours=extended_hours,
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
            "Live trading %s (execute=%s, auto_exit=%s, fractional=%s, extended=%s)",
            engine.symbol, execute, auto_exit, fractional, extended_hours,
        )

    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)
