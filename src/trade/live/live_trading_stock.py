"""Live stock trading with market hours awareness."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


from src.strategy.registry import get_strategy_registry
from src.trade.engine import TradingStockEngine
from src.utils.asset_utils import merge_symbol_sources, positions_for_asset_type
from src.utils.timezone_utils import format_duration
from src.watchlist import WatchlistManager

from .session_policy import SessionAwareTrading
from .live_runner import run_live_engines
from .live_trading_base import LiveTradingEngine

if TYPE_CHECKING:
    from src.strategy.base import StrategyBase


class LiveTradingStock(SessionAwareTrading, LiveTradingEngine):
    """Live trading engine for stocks.

    Features:
    - Broker-calendar session hours, including holidays and early closes
    - Signal cycles based on timeframe
    - PDT rules consideration
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        lookback_days: int = 30,
        risk_pct: float = 0.05,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        execute: bool = True,
        auto_exit: bool = True,
        allow_fractional: bool = False,
        extended_hours: bool = False,
        strategy: str = "momentum",
        strategy_params: dict[str, Any] | None = None,
        allow_sell_to_open: bool = False,
        order_type: str = "auto",
    ) -> None:
        self.allow_fractional = allow_fractional
        self.extended_hours = extended_hours
        self.strategy_name = strategy
        self.strategy_params = dict(strategy_params or {})
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execute=execute,
            auto_exit=auto_exit,
            asset_type="stock",
            allow_sell_to_open=allow_sell_to_open,
            order_type=order_type,
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize stock symbol (uppercase, trimmed)."""
        return symbol.strip().upper()

    def _get_trading_engine(self) -> TradingStockEngine:
        """Return stock trading engine."""
        return TradingStockEngine(allow_fractional=self.allow_fractional)

    def _get_strategy(self) -> StrategyBase:
        """Return the configured strategy."""
        return get_strategy_registry().create(
            self.strategy_name,
            self.strategy_params,
        )

    def _create_stream(self) -> Any:
        """Create stock data stream."""
        return self.provider.create_stock_stream(raw_data=False)

    def _subscribe_to_stream(self, handler: Any, symbol: str) -> None:
        """Subscribe to stock trade stream."""
        self._stream.subscribe_trades(handler, symbol)

    def _get_display_symbol(self) -> str:
        """Stock symbols don't need conversion."""
        return self.symbol


def get_default_stock_symbols() -> list[str]:
    """Get default stock symbols from watchlist and open positions."""
    manager = WatchlistManager()
    watchlist_symbols = manager.get_watchlist(asset_type="stock")
    engine = TradingStockEngine()
    position_symbols = positions_for_asset_type(engine.get_current_positions(), "stock")
    defaults = merge_symbol_sources("stock", watchlist_symbols, position_symbols)
    return defaults or ["AAPL"]


def create_stock_engines(
    symbols: list[str] | None = None,
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
    strategy_params: dict[str, Any] | None = None,
    allow_sell_to_open: bool = False,
    order_type: str = "auto",
) -> list[LiveTradingStock]:
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
            strategy_params=strategy_params,
            allow_sell_to_open=allow_sell_to_open,
            order_type=order_type,
        )
        for symbol in trading_symbols
    ]


def run_stock_trading(
    symbols: list[str] | None = None,
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
    allow_sell_to_open: bool = False,
    order_type: str = "auto",
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
        allow_sell_to_open=allow_sell_to_open,
        order_type=order_type,
    )

    if engines and not engines[0].is_market_open():
        remaining = engines[0].seconds_until_market_open()
        logging.warning("NOT in market period. Market opens in %s", format_duration(remaining))
        return

    for engine in engines:
        engine.logger.info(
            "Live trading %s (execute=%s, auto_exit=%s, fractional=%s, extended=%s)",
            engine.symbol,
            execute,
            auto_exit,
            fractional,
            extended_hours,
        )

    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)
