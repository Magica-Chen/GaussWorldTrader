"""Live stock trading with market hours awareness."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

import pytz

from src.strategy.stock.momentum import MomentumStrategy

from .live_trading_base import LiveTradingEngine
from .stock_engine import TradingStockEngine


EASTERN = pytz.timezone("US/Eastern")


class LiveTradingStock(LiveTradingEngine):
    """Live trading engine for stocks.

    Features:
    - Market hours awareness (9:30 AM - 4:00 PM ET)
    - Extended hours support (optional)
    - Fractional shares support (optional)
    - PDT rules consideration
    """

    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EXTENDED_OPEN = time(4, 0)
    EXTENDED_CLOSE = time(20, 0)

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
    ) -> None:
        self.allow_fractional = allow_fractional
        self.extended_hours = extended_hours
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execute=execute,
            auto_exit=auto_exit,
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize stock symbol (uppercase, trimmed)."""
        return symbol.strip().upper()

    def _get_trading_engine(self) -> TradingStockEngine:
        """Return stock trading engine."""
        return TradingStockEngine(allow_fractional=self.allow_fractional)

    def _get_strategy(self) -> MomentumStrategy:
        """Return stock momentum strategy."""
        return MomentumStrategy()

    def _create_stream(self) -> Any:
        """Create stock data stream."""
        return self.provider.create_stock_stream(raw_data=False)

    def _subscribe_to_stream(self, handler: Any) -> None:
        """Subscribe to stock trade stream."""
        self._stream.subscribe_trades(handler, self.symbol)

    def _get_signal_interval_seconds(self) -> float:
        """Return seconds until next signal check, respecting market hours."""
        if not self._is_market_open():
            return self._seconds_until_market_open()

        now = datetime.now(EASTERN)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        close_time = (
            self.EXTENDED_CLOSE if self.extended_hours else self.MARKET_CLOSE
        )
        today_close = now.replace(
            hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0
        )

        if next_hour.time() > close_time:
            return self._seconds_until_market_open()

        return max(1.0, (next_hour - now).total_seconds())

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EASTERN)

        if now.weekday() >= 5:
            return False

        current_time = now.time()

        if self.extended_hours:
            return self.EXTENDED_OPEN <= current_time <= self.EXTENDED_CLOSE

        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def _seconds_until_market_open(self) -> float:
        """Calculate seconds until market opens."""
        now = datetime.now(EASTERN)
        open_time = self.EXTENDED_OPEN if self.extended_hours else self.MARKET_OPEN

        days_ahead = 0
        if now.weekday() == 5:
            days_ahead = 2
        elif now.weekday() == 6:
            days_ahead = 1
        elif now.time() > (self.EXTENDED_CLOSE if self.extended_hours else self.MARKET_CLOSE):
            days_ahead = 1
            if now.weekday() == 4:
                days_ahead = 3

        next_open = now.replace(
            hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0
        )
        if days_ahead > 0:
            next_open += timedelta(days=days_ahead)
        elif now.time() >= open_time:
            next_open += timedelta(days=1)

        return max(1.0, (next_open - now).total_seconds())

    def _get_display_symbol(self) -> str:
        """Stock symbols don't need conversion."""
        return self.symbol
