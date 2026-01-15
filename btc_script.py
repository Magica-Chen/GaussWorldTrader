#!/usr/bin/env python3
"""
Live BTC/USD trading loop using Alpaca streaming data and momentum signals.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.data import AlpacaDataProvider
from src.strategy.crypto.momentum import CryptoMomentumStrategy
from src.trade import TradingEngine
from src.utils.timezone_utils import now_et
from src.utils.validators import convert_crypto_symbol_for_display


def _get_field(data: Any, attr: str, raw_key: str) -> Any:
    if hasattr(data, attr):
        return getattr(data, attr)
    if isinstance(data, dict):
        return data.get(raw_key)
    return None


def _normalize_crypto_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if "/" in symbol:
        return symbol
    if symbol.endswith("USD") and len(symbol) > 3:
        return f"{symbol[:-3]}/USD"
    return symbol


def _seconds_until_next_hour() -> float:
    now = datetime.now(timezone.utc)
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return max(1.0, (next_hour - now).total_seconds())


@dataclass
class PositionState:
    qty: float = 0.0
    side: str = "flat"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class BtcLiveTrader:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int,
        crypto_loc: str,
        risk_pct: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        execute: bool,
        auto_exit: bool,
    ) -> None:
        self.symbol = _normalize_crypto_symbol(symbol)
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.crypto_loc = crypto_loc
        self.execute = execute
        self.auto_exit = auto_exit

        self.provider = AlpacaDataProvider()
        self.engine = TradingEngine()
        self.strategy = CryptoMomentumStrategy(
            params={
                "risk_pct": risk_pct,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
            }
        )

        self._lock = threading.Lock()
        self._exit_lock = threading.Lock()
        self._exit_in_progress = False
        self._last_monitor_log = 0.0
        self._latest_price: Optional[float] = None
        self._latest_timestamp: Optional[datetime] = None
        self.position = PositionState()

        self._stream = None
        self._stream_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger("btc_live_trader")

    def start(self) -> None:
        self._start_stream()
        self._refresh_position_state()

        try:
            while True:
                self._run_signal_cycle()
                sleep_seconds = _seconds_until_next_hour()
                next_run = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
                self.logger.info(
                    "Next signal check at %s (sleep %.0fs)",
                    next_run.isoformat(timespec="seconds"),
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            self.logger.info("Stopping live trader.")
        finally:
            self._stop_stream()

    def _start_stream(self) -> None:
        try:
            self._stream = self.provider.create_crypto_stream(raw_data=False, loc=self.crypto_loc)
        except Exception as exc:
            raise RuntimeError(f"Unable to start crypto stream: {exc}") from exc

        async def handle_trade(data: Any) -> None:
            price = _get_field(data, "price", "p")
            timestamp = _get_field(data, "timestamp", "t")
            if price is None:
                return
            with self._lock:
                self._latest_price = float(price)
                self._latest_timestamp = timestamp if isinstance(timestamp, datetime) else None
            self._monitor_position(float(price))

        self._stream.subscribe_trades(handle_trade, self.symbol)

        def _run_stream() -> None:
            try:
                self._stream.run()
            except Exception as exc:
                self.logger.error("Stream stopped: %s", exc)

        self._stream_thread = threading.Thread(target=_run_stream, name="btc_stream", daemon=True)
        self._stream_thread.start()
        self.logger.info("Streaming BTC trades from feed %s", self.crypto_loc)

    def _stop_stream(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
            except Exception as exc:
                self.logger.error("Failed to stop stream: %s", exc)
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)

    def _refresh_position_state(self) -> None:
        positions = self.engine.get_current_positions()
        display_symbol = convert_crypto_symbol_for_display(self.symbol)
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
        portfolio_value = float(account_info.get("portfolio_value") or account_info.get("equity") or 100000)

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
        if not self.execute:
            self.logger.info("DRY RUN: would close position (%s)", reason)
            return

        with self._exit_lock:
            if self._exit_in_progress:
                return
            self._exit_in_progress = True

        def _do_close() -> None:
            try:
                self.engine.close_position(self.symbol)
                self.logger.info("Closed position: %s", reason)
            except Exception as exc:
                self.logger.error("Failed to close position: %s", exc)
            finally:
                with self._exit_lock:
                    self._exit_in_progress = False
                self._refresh_position_state()

        threading.Thread(target=_do_close, name="btc_close_position", daemon=True).start()

    def _monitor_position(self, price: float) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live BTC/USD trading script.")
    parser.add_argument("--symbol", default="BTC/USD", help="Crypto pair to trade.")
    parser.add_argument("--timeframe", default="1Hour", help="Bar timeframe for signals.")
    parser.add_argument("--lookback-days", type=int, default=14, help="Historical lookback.")
    parser.add_argument("--crypto-loc", default="eu-1", help="Crypto stream feed: us, us-1, eu-1.")
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
    trader = BtcLiveTrader(
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
    trader.logger.info("Live trading %s (execute=%s, auto_exit=%s)", trader.symbol, args.execute, args.auto_exit)
    trader.start()


if __name__ == "__main__":
    main()
