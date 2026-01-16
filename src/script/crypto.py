#!/usr/bin/env python3
"""Live cryptocurrency trading module using momentum signals."""
from __future__ import annotations

import logging
from typing import List, Optional

from src.trade.crypto_engine import TradingCryptoEngine
from src.trade.live_trading_crypto import LiveTradingCrypto
from src.trade.live_runner import run_live_engines
from src.utils.live_utils import merge_symbol_sources
from src.utils.watchlist_manager import WatchlistManager


def get_default_crypto_symbols() -> List[str]:
    """Get default crypto symbols from watchlist and open positions."""
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


def create_crypto_engines(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1Hour",
    lookback_days: int = 14,
    crypto_loc: str = "us",
    risk_pct: float = 0.10,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    execute: bool = True,
    auto_exit: bool = True,
    strategy: str = "crypto_momentum",
) -> List[LiveTradingCrypto]:
    """Create crypto trading engines without starting them.

    Returns:
        List of configured LiveTradingCrypto engines.
    """
    trading_symbols = symbols if symbols else get_default_crypto_symbols()
    trading_symbols = merge_symbol_sources("crypto", trading_symbols)

    return [
        LiveTradingCrypto(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            crypto_loc=crypto_loc,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execute=execute,
            auto_exit=auto_exit,
            strategy=strategy,
        )
        for symbol in trading_symbols
    ]


def run_crypto_trading(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1Hour",
    lookback_days: int = 14,
    crypto_loc: str = "us",
    risk_pct: float = 0.10,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    execute: bool = True,
    auto_exit: bool = True,
    strategy: str = "crypto_momentum",
) -> None:
    """Run live cryptocurrency trading.

    Args:
        symbols: List of crypto pairs (e.g., ["BTC/USD", "ETH/USD"]).
                 If None, uses watchlist and open positions.
        timeframe: Bar timeframe for signals.
        lookback_days: Historical lookback days.
        crypto_loc: Crypto stream feed location (us, us-1, eu-1).
        risk_pct: Portfolio risk per trade.
        stop_loss_pct: Stop-loss percentage.
        take_profit_pct: Take-profit percentage.
        execute: Execute live trades (False for dry run).
        auto_exit: Auto-close on stop/take-profit.
        strategy: Strategy name to use for signals.
    """
    engines = create_crypto_engines(
        symbols=symbols,
        timeframe=timeframe,
        lookback_days=lookback_days,
        crypto_loc=crypto_loc,
        risk_pct=risk_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        execute=execute,
        auto_exit=auto_exit,
        strategy=strategy,
    )

    for engine in engines:
        engine.logger.info(
            "Live trading %s (execute=%s, auto_exit=%s)",
            engine.symbol, execute, auto_exit,
        )

    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)
