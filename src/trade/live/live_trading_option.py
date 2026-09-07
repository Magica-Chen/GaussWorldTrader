"""Live options trading with expiration awareness."""

from __future__ import annotations

import logging
from typing import Any, List, Optional


from src.watchlist import WatchlistManager
from src.strategy.base import StrategyBase
from src.strategy.registry import get_strategy_registry
from src.utils.asset_utils import merge_symbol_sources
from src.utils.timezone_utils import format_duration

from .live_trading_base import LiveTradingEngine, PositionState
from .session_policy import SessionAwareTrading
from .live_runner import run_live_engines
from src.trade.engine import TradingOptionEngine


class LiveTradingOption(SessionAwareTrading, LiveTradingEngine):
    """Live trading engine for options.

    Features:
    - Market hours awareness (9:30 AM - 4:00 PM ET)
    - Expiration date tracking
    - Position rolling support
    - Underlying price monitoring
    """

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
        allow_sell_to_open: bool = False,
        order_type: str = "auto",
    ) -> None:
        if execute:
            raise ValueError(
                'Unattended option execution requires the Gauss session gateway and options '
                'readiness gates. Use execute=False for legacy underlying research.'
            )
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
            asset_type="option",
            allow_sell_to_open=allow_sell_to_open,
            order_type=order_type,
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
                "Position %s expires in %d days - consider rolling", pos.get('symbol'), days_left
            )

    def _refresh_position_state(self) -> None:
        """Refresh position state for options."""
        if not isinstance(self.engine, TradingOptionEngine):
            super()._refresh_position_state()
            return

        positions = [
            p
            for p in self.engine.get_option_positions()
            if p.get('underlying') == self.underlying_symbol and float(p.get('qty', 0)) != 0
        ]

        with self._lock:
            self.option_legs = {p['symbol']: dict(p) for p in positions}
            if not positions:
                self.position = PositionState()
                return

            total_qty = sum(float(p.get('qty', 0)) for p in positions)
            gross_qty = sum(abs(float(p['qty'])) for p in positions)
            side = "mixed" if len(positions) > 1 else ("long" if total_qty > 0 else "short")

            self.position = PositionState(
                qty=gross_qty,
                side=side,
                entry_price=None,
            )

    def _monitor_position(self, price: float) -> None:
        """Underlying trades are signal evidence; contract premiums are managed by Gauss."""
        return


def get_default_option_symbols() -> List[str]:
    """Get default option underlying symbols from watchlist and open positions."""
    manager = WatchlistManager()
    watchlist_symbols = manager.get_watchlist(asset_type="option")
    position_symbols: List[str] = []
    engine = TradingOptionEngine()
    for pos in engine.get_option_positions():
        underlying = pos.get("underlying")
        if underlying:
            position_symbols.append(underlying)
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
    allow_sell_to_open: bool = False,
    order_type: str = "auto",
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
            allow_sell_to_open=allow_sell_to_open,
            order_type=order_type,
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
    allow_sell_to_open: bool = False,
    order_type: str = "auto",
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
        allow_sell_to_open=allow_sell_to_open,
        order_type=order_type,
    )

    if engines and not engines[0].is_market_open():
        remaining = engines[0].seconds_until_market_open()
        logging.warning("NOT in market period. Market opens in %s", format_duration(remaining))
        return

    for engine in engines:
        engine.logger.info(
            "Live options trading on %s (execute=%s, auto_exit=%s, roll_days=%d)",
            engine.underlying_symbol,
            execute,
            auto_exit,
            roll_days,
        )

    if len(engines) == 1:
        engines[0].start()
    else:
        run_live_engines(engines)
