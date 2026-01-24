"""Live options trading with expiration awareness."""
from __future__ import annotations

from datetime import datetime, time, timedelta
import logging
from typing import Any, List, Optional

import pytz

from src.agent.watchlist_manager import WatchlistManager
from src.strategy.base import StrategyBase
from src.strategy.registry import get_strategy_registry
from src.utils.asset_utils import merge_symbol_sources
from src.utils.timezone_utils import format_duration

from .live_trading_base import LiveTradingEngine, PositionState
from .live_runner import run_live_engines
from .option_engine import TradingOptionEngine


EASTERN = pytz.timezone("US/Eastern")


class LiveTradingOption(LiveTradingEngine):
    """Live trading engine for options.

    Features:
    - Market hours awareness (9:30 AM - 4:00 PM ET)
    - Expiration date tracking
    - Position rolling support
    - Underlying price monitoring
    """

    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        underlying_symbol: str,
        timeframe: str = "1Day",
        lookback_days: int = 30,
        risk_pct: float = 0.08,
        stop_loss_pct: float = 0.50,
        take_profit_pct: float = 0.50,
        execute: bool = True,
        auto_exit: bool = True,
        roll_days_before_expiry: int = 5,
        strategy: str = "wheel",
    ) -> None:
        self.underlying_symbol = underlying_symbol.strip().upper()
        self.roll_days_before_expiry = roll_days_before_expiry
        self.strategy_name = strategy
        super().__init__(
            symbol=underlying_symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execute=execute,
            auto_exit=auto_exit,
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize underlying symbol (uppercase, trimmed)."""
        return symbol.strip().upper()

    def _get_trading_engine(self) -> TradingOptionEngine:
        """Return options trading engine."""
        return TradingOptionEngine()

    def _get_strategy(self) -> StrategyBase:
        """Return the configured strategy."""
        return get_strategy_registry().create(self.strategy_name)

    def _create_stream(self) -> Any:
        """Create stock data stream for underlying."""
        return self.provider.create_stock_stream(raw_data=False)

    def _subscribe_to_stream(self, handler: Any, symbol: str) -> None:
        """Subscribe to underlying stock trade stream."""
        self._stream.subscribe_trades(handler, symbol)

    def _get_signal_interval_seconds(self) -> float:
        """Return seconds until next signal check, respecting market hours."""
        if not self._is_market_open():
            return self._seconds_until_market_open()

        interval_secs = self._seconds_until_next_interval()
        now = datetime.now(EASTERN)
        today_close = now.replace(
            hour=self.MARKET_CLOSE.hour, minute=self.MARKET_CLOSE.minute,
            second=0, microsecond=0
        )
        secs_to_close = max(0.0, (today_close - now).total_seconds())

        if interval_secs > secs_to_close:
            return self._seconds_until_market_open()

        return interval_secs

    def is_market_open(self) -> bool:
        """Expose market open status for scripts."""
        return self._is_market_open()

    def seconds_until_market_open(self) -> float:
        """Expose seconds until next market open for scripts."""
        return self._seconds_until_market_open()

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EASTERN)

        if now.weekday() >= 5:
            return False

        current_time = now.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def _seconds_until_market_open(self) -> float:
        """Calculate seconds until market opens."""
        now = datetime.now(EASTERN)

        days_ahead = 0
        if now.weekday() == 5:
            days_ahead = 2
        elif now.weekday() == 6:
            days_ahead = 1
        elif now.time() > self.MARKET_CLOSE:
            days_ahead = 1
            if now.weekday() == 4:
                days_ahead = 3

        next_open = now.replace(
            hour=self.MARKET_OPEN.hour, minute=self.MARKET_OPEN.minute,
            second=0, microsecond=0
        )
        if days_ahead > 0:
            next_open += timedelta(days=days_ahead)
        elif now.time() >= self.MARKET_OPEN:
            next_open += timedelta(days=1)

        return max(1.0, (next_open - now).total_seconds())

    def _get_display_symbol(self) -> str:
        """Return underlying symbol for display."""
        return self.underlying_symbol

    def _run_signal_cycle(self) -> None:
        """Run signal cycle with expiration checking."""
        self._check_expiring_positions()
        super()._run_signal_cycle()

    def _check_expiring_positions(self) -> None:
        """Check for positions nearing expiration."""
        if not isinstance(self.engine, TradingOptionEngine):
            return

        expiring = self.engine.get_expiring_positions(days=self.roll_days_before_expiry)
        for pos in expiring:
            days_left = pos.get('days_to_expiration', 999)
            self.logger.warning(
                "Position %s expires in %d days - consider rolling",
                pos.get('symbol'), days_left
            )

    def _refresh_position_state(self) -> None:
        """Refresh position state for options."""
        if not isinstance(self.engine, TradingOptionEngine):
            super()._refresh_position_state()
            return

        positions = self.engine.get_option_positions()

        with self._lock:
            if not positions:
                self.position = PositionState()
                return

            total_value = sum(float(p.get('market_value', 0)) for p in positions)
            total_qty = sum(float(p.get('qty', 0)) for p in positions)

            if total_qty == 0:
                self.position = PositionState()
                return

            side = "long" if total_qty > 0 else "short"
            avg_cost = sum(float(p.get('cost_basis', 0)) for p in positions) / abs(total_qty)

            self.position = PositionState(
                qty=total_qty,
                side=side,
                entry_price=avg_cost,
                stop_loss=self.position.stop_loss,
                take_profit=self.position.take_profit,
            )


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
