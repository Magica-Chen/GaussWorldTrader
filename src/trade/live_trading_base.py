"""Abstract base for live trading with streaming data and signal orchestration."""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.data import AlpacaDataProvider
from src.utils.timezone_utils import now_et
from src.utils.validators import convert_crypto_symbol_for_display

if TYPE_CHECKING:
    from src.strategy.base import StrategyBase
    from .trading_engine import TradingEngine


@dataclass
class PositionState:
    """Track position state for live trading."""
    qty: float = 0.0
    side: str = "flat"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class LiveTradingEngine(ABC):
    """Abstract base for live trading with streaming data and signal orchestration.

    Provides common functionality for:
    - Real-time data streaming
    - Position state tracking
    - Signal generation cycles
    - Stop-loss and take-profit monitoring
    - Order execution with dry-run support
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int,
        risk_pct: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        execute: bool,
        auto_exit: bool,
    ) -> None:
        self.raw_symbol = symbol
        self.symbol = self._normalize_symbol(symbol)
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.execute = execute
        self.auto_exit = auto_exit

        self.provider = AlpacaDataProvider()
        self.engine = self._get_trading_engine()
        self.strategy = self._get_strategy()
        self.strategy.params.update({
            "risk_pct": risk_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        })

        self._lock = threading.Lock()
        self._exit_lock = threading.Lock()
        self._exit_in_progress = False
        self._last_monitor_log = 0.0
        self._latest_price: Optional[float] = None
        self._latest_timestamp: Optional[datetime] = None
        self.position = PositionState()

        self._stream = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.symbol}")

    @abstractmethod
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for the specific asset type."""
        pass

    @abstractmethod
    def _get_trading_engine(self) -> "TradingEngine":
        """Return the appropriate trading engine instance."""
        pass

    @abstractmethod
    def _get_strategy(self) -> "StrategyBase":
        """Return the appropriate strategy instance."""
        pass

    @abstractmethod
    def _create_stream(self) -> Any:
        """Create and return the data stream for this asset type."""
        pass

    @abstractmethod
    def _get_signal_interval_seconds(self) -> float:
        """Return seconds until next signal check."""
        pass

    def is_market_open(self) -> bool:
        """Return True if the market is open for this asset type."""
        return True

    def seconds_until_market_open(self) -> float:
        """Return seconds until the market opens for this asset type."""
        return 0.0

    def _timeframe_to_seconds(self) -> int:
        """Convert timeframe string to seconds."""
        mapping = {
            "1Min": 60,
            "5Min": 300,
            "15Min": 900,
            "30Min": 1800,
            "1Hour": 3600,
            "1Day": 86400,
            "1Week": 604800,
            "1Month": 2592000,
        }
        return mapping.get(self.timeframe, 3600)

    def _seconds_until_next_interval(self) -> float:
        """Calculate seconds until next timeframe boundary."""
        interval = self._timeframe_to_seconds()
        now = datetime.now(timezone.utc)
        epoch = now.timestamp()
        next_boundary = ((int(epoch) // interval) + 1) * interval
        return max(1.0, next_boundary - epoch)

    @abstractmethod
    def _subscribe_to_stream(self, handler: Any, symbol: str) -> None:
        """Subscribe to the appropriate stream data for a symbol."""
        pass

    def _get_display_symbol(self) -> str:
        """Get display-friendly symbol for logging and position matching."""
        return convert_crypto_symbol_for_display(self.symbol)

    def start(self) -> None:
        """Main entry point - starts streaming and signal loop."""
        self._start_stream()
        try:
            self.run_signal_loop()
        finally:
            self._stop_stream()

    def run_signal_loop(self) -> None:
        """Run the signal loop without starting a stream."""
        self._stop_event.clear()
        self._refresh_position_state()

        while not self._stop_event.is_set():
            self._run_signal_cycle()
            sleep_seconds = self._get_signal_interval_seconds()
            next_run = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
            self.logger.info(
                "Next signal check at %s (sleep %.0fs)",
                next_run.isoformat(timespec="seconds"),
                sleep_seconds,
            )
            self._stop_event.wait(timeout=sleep_seconds)

    def stop(self) -> None:
        """Stop the live trading engine."""
        self._stop_event.set()
        self._stop_stream()

    def _start_stream(self) -> None:
        """Start the data streaming thread."""
        self._stream = self._create_stream()
        self._subscribe_to_stream(self.handle_trade, self.symbol)

        def _run_stream() -> None:
            try:
                self._stream.run()
            except Exception as exc:
                self.logger.error("Stream stopped: %s", exc)

        self._stream_thread = threading.Thread(
            target=_run_stream, name=f"{self.__class__.__name__}_stream", daemon=True
        )
        self._stream_thread.start()
        self.logger.info("Streaming started for %s", self.symbol)

    def _stop_stream(self) -> None:
        """Stop the data streaming thread."""
        if self._stream:
            try:
                self._stream.stop()
            except Exception as exc:
                self.logger.error("Failed to stop stream: %s", exc)
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)

    def _refresh_position_state(self) -> None:
        """Refresh position state from the trading engine."""
        positions = self.engine.get_current_positions()
        display_symbol = self._get_display_symbol()
        match = next((pos for pos in positions if pos.get("symbol") == display_symbol), None)

        with self._lock:
            if not match:
                self.position = PositionState()
                return

            qty = float(match.get("qty", 0.0))
            side = "long" if qty > 0 else "short" if qty < 0 else "flat"
            cost_basis = float(match.get("cost_basis", 0.0))
            entry_price = (cost_basis / abs(qty)) if qty else None

            stop_loss = self.position.stop_loss
            take_profit = self.position.take_profit
            if entry_price and side != "flat":
                if stop_loss is None:
                    stop_loss = self.strategy.calculate_stop_loss(entry_price, side)
                if take_profit is None:
                    take_profit = self.strategy.calculate_take_profit(entry_price, side)

            self.position = PositionState(
                qty=qty,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

    def _run_signal_cycle(self) -> None:
        """Run one signal generation and execution cycle."""
        self._refresh_position_state()
        signal = self._get_latest_signal()
        if not signal:
            self.logger.info("Signal: HOLD")
            return

        action = signal.get("action", "HOLD")
        self.logger.info("Signal: %s (%s)", action, signal.get("reason", ""))

        if action == "BUY":
            if self.position.side == "long":
                self.logger.info("Already long; skipping buy.")
                return
            self._place_order("buy", signal)
        elif action == "SELL":
            if self.position.side != "long":
                self.logger.info("No long position to close; skipping sell.")
                return
            self._close_position("signal_sell")

    def _get_latest_signal(self) -> Optional[Dict[str, Any]]:
        """Generate the latest signal from the strategy."""
        start_date = now_et() - timedelta(days=self.lookback_days)
        bars = self.provider.get_bars(self.symbol, self.timeframe, start_date)
        if bars.empty:
            self.logger.warning("No bars returned for %s", self.symbol)
            return None

        current_price = float(bars["close"].iloc[-1])
        current_data = {
            "open": float(bars["open"].iloc[-1]),
            "high": float(bars["high"].iloc[-1]),
            "low": float(bars["low"].iloc[-1]),
            "close": current_price,
            "volume": float(bars["volume"].iloc[-1]),
        }

        account_info = self.engine.get_account_info()
        portfolio_value = float(
            account_info.get("portfolio_value") or account_info.get("equity") or 100000
        )

        class _PortfolioProxy:
            def __init__(self, value: float) -> None:
                self.value = value

            def get_portfolio_value(self, _prices: Dict[str, float]) -> float:
                return self.value

        signals = self.strategy.generate_signals(
            current_date=now_et(),
            current_prices={self.symbol: current_price},
            current_data={self.symbol: current_data},
            historical_data={self.symbol: bars},
            portfolio=_PortfolioProxy(portfolio_value),
        )
        return signals[0] if signals else None

    def _place_order(self, side: str, signal: Dict[str, Any]) -> None:
        """Place an order based on a signal."""
        qty = float(signal.get("quantity", 0.0))
        if qty <= 0:
            self.logger.info("Signal quantity is zero; skipping order.")
            return

        if not self.execute:
            self.logger.info("DRY RUN: would %s %s %s", side, qty, self.symbol)
            return

        self.engine.place_market_order(self.symbol, qty, side=side)
        time.sleep(2)
        with self._lock:
            self.position.stop_loss = signal.get("stop_loss")
            self.position.take_profit = signal.get("take_profit")
        self._refresh_position_state()
        self.logger.info("Order placed: %s %s %s", side, qty, self.symbol)

    def _close_position(self, reason: str) -> None:
        """Close the current position."""
        if not self.execute:
            self.logger.info("DRY RUN: would close position (%s)", reason)
            return

        with self._exit_lock:
            if self._exit_in_progress:
                return
            self._exit_in_progress = True

        def _do_close() -> None:
            try:
                self.engine.close_position(self._get_display_symbol())
                self.logger.info("Closed position: %s", reason)
            except Exception as exc:
                self.logger.error("Failed to close position: %s", exc)
            finally:
                with self._exit_lock:
                    self._exit_in_progress = False
                self._refresh_position_state()

        threading.Thread(
            target=_do_close, name=f"{self.__class__.__name__}_close", daemon=True
        ).start()

    def _monitor_position(self, price: float) -> None:
        """Monitor position for stop-loss and take-profit exits."""
        with self._lock:
            position = self.position
            if position.side == "flat" or not position.entry_price or position.qty == 0:
                return

            entry = position.entry_price
            qty = position.qty
            side = position.side

            if side == "long":
                pnl = (price - entry) * qty
                pnl_pct = (price - entry) / entry
            else:
                pnl = (entry - price) * abs(qty)
                pnl_pct = (entry - price) / entry

            now_ts = time.time()
            if now_ts - self._last_monitor_log > 10:
                self.logger.info(
                    "Price %.2f | P/L %.2f (%.2f%%) | stop %.2f | take %.2f",
                    price,
                    pnl,
                    pnl_pct * 100,
                    position.stop_loss or 0.0,
                    position.take_profit or 0.0,
                )
                self._last_monitor_log = now_ts

            if not self.auto_exit:
                return

            if side == "long":
                if position.stop_loss and price <= position.stop_loss:
                    self._close_position("stop_loss")
                elif position.take_profit and price >= position.take_profit:
                    self._close_position("take_profit")
            else:
                if position.stop_loss and price >= position.stop_loss:
                    self._close_position("stop_loss")
                elif position.take_profit and price <= position.take_profit:
                    self._close_position("take_profit")

    @staticmethod
    def _get_field(data: Any, attr: str, raw_key: str) -> Any:
        """Extract field from data object or dict."""
        if hasattr(data, attr):
            return getattr(data, attr)
        if isinstance(data, dict):
            return data.get(raw_key)
        return None

    async def handle_trade(self, data: Any) -> None:
        """Handle a trade update from the stream."""
        price = self._get_field(data, "price", "p")
        timestamp = self._get_field(data, "timestamp", "t")
        if price is None:
            return
        self._handle_trade_update(float(price), timestamp)

    def _handle_trade_update(self, price: float, timestamp: Any) -> None:
        """Update latest price and monitor position for a trade."""
        with self._lock:
            self._latest_price = price
            self._latest_timestamp = timestamp if isinstance(timestamp, datetime) else None
        self._monitor_position(price)
