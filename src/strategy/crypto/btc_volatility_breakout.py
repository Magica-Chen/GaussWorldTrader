"""BTC volatility breakout strategy for crypto live trading.

Designed for BTC/USD on 1H/4H bars:
- Buy strength only when price breaks above a recent rolling high
- Require higher-timeframe trend confirmation via EMA filter
- Require volatility expansion via ATR percentage filter
- Exit when momentum fades back below the fast EMA or recent range low

This is intentionally long-only to match the current Alpaca crypto engine style.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import ActionPlan, SignalSnapshot, StrategyBase, StrategyMeta, StrategySignal
from src.strategy.utils import latest_price, safe_series


ta = TechnicalAnalysis()


class BTCVolatilityBreakoutStrategy(StrategyBase):
    """Trend-filtered Donchian-style breakout strategy for BTC."""

    meta = StrategyMeta(
        name="btc_volatility_breakout",
        label="BTC Volatility Breakout",
        category="signal",
        description=(
            "Long-only BTC breakout strategy using rolling highs/lows, EMA trend "
            "confirmation, and ATR-based volatility expansion filters."
        ),
        asset_type="crypto",
        default_params={
            "breakout_lookback": 20,
            "exit_lookback": 10,
            "fast_ema": 20,
            "slow_ema": 50,
            "atr_period": 14,
            "min_atr_pct": 0.012,
            "breakout_buffer_pct": 0.0015,
            "risk_pct": 0.10,
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 4.0,
            "qty_precision": 6,
            "min_qty": 0.000001,
        },
        visible_in_dashboard=True,
    )
    summary = (
        "BTC long-only volatility breakout. Buys when price breaks above the prior "
        "20-bar high while above fast/slow EMAs and ATR% confirms expansion. Sells "
        "when price loses the fast EMA or breaks below the recent exit range low. "
        "Uses ATR-multiple stop-loss and take-profit levels."
    )

    def _position_size(self, price: float, portfolio_value: float, risk_pct: float) -> float:
        if price <= 0 or portfolio_value <= 0:
            return 0.0
        quantity = (portfolio_value * risk_pct) / price
        precision = int(self.params.get("qty_precision", 6))
        min_qty = float(self.params.get("min_qty", 0.0))
        quantity = round(quantity, precision)
        if min_qty > 0 and quantity < min_qty:
            return 0.0
        return quantity

    def generate_signals(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> List[Dict[str, Any]]:
        signals: List[StrategySignal] = []
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            price = current_prices.get(symbol, latest_price(data))
            snapshot = self.get_signal(
                symbol=symbol,
                current_date=current_date,
                current_price=price,
                current_data=current_data.get(symbol, {}),
                historical_data=data,
                portfolio=portfolio,
            )
            if snapshot is None:
                continue
            plan = self.get_action_plan(snapshot, price, current_date)
            if not plan or plan.action == "HOLD":
                continue

            portfolio_value = getattr(
                portfolio, "get_portfolio_value", lambda *_: 100000
            )(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)
            if quantity <= 0:
                continue
            signals.append(self._plan_to_signal(plan, quantity, price))

        return self._normalize(signals)

    def get_signal(
        self,
        symbol: str,
        current_date: datetime,
        current_price: float,
        current_data: Dict[str, Any],
        historical_data: pd.DataFrame,
        portfolio: Any = None,
    ) -> Optional[SignalSnapshot]:
        breakout_lookback = int(self.params["breakout_lookback"])
        exit_lookback = int(self.params["exit_lookback"])
        fast_ema_period = int(self.params["fast_ema"])
        slow_ema_period = int(self.params["slow_ema"])
        atr_period = int(self.params["atr_period"])
        min_atr_pct = float(self.params["min_atr_pct"])
        breakout_buffer_pct = float(self.params["breakout_buffer_pct"])

        min_bars = max(breakout_lookback + 1, exit_lookback + 1, slow_ema_period + 1, atr_period + 2)
        if len(historical_data) < min_bars:
            return None
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(historical_data.columns):
            return None

        close = historical_data["close"]
        high = historical_data["high"]
        low = historical_data["low"]

        fast_ema = ta.ema(close, fast_ema_period)
        slow_ema = ta.ema(close, slow_ema_period)
        atr = ta.atr(high, low, close, atr_period)

        fast_val = safe_series(fast_ema)
        slow_val = safe_series(slow_ema)
        atr_val = safe_series(atr)
        if current_price <= 0:
            return None
        atr_pct = atr_val / current_price if current_price else 0.0

        prior_breakout_high = float(high.shift(1).rolling(window=breakout_lookback).max().iloc[-1])
        prior_exit_low = float(low.shift(1).rolling(window=exit_lookback).min().iloc[-1])
        breakout_level = prior_breakout_high * (1 + breakout_buffer_pct)

        in_uptrend = current_price > fast_val > slow_val
        breakout_triggered = current_price > breakout_level
        volatility_confirmed = atr_pct >= min_atr_pct
        exit_triggered = current_price < fast_val or current_price < prior_exit_low

        if breakout_triggered and in_uptrend and volatility_confirmed:
            signal = "BUY"
            reason = (
                f"breakout above {breakout_level:.2f} with EMA trend confirmation "
                f"and ATR% {atr_pct:.2%}"
            )
            strength = max(0.0, (current_price - breakout_level) / breakout_level) + atr_pct
        elif exit_triggered:
            signal = "SELL"
            reason = (
                f"exit on loss of momentum: price below fast EMA or {exit_lookback}-bar low"
            )
            strength = max(
                0.0,
                (fast_val - current_price) / fast_val if fast_val else 0.0,
                (prior_exit_low - current_price) / prior_exit_low if prior_exit_low else 0.0,
            )
        else:
            signal = "HOLD"
            reason = "waiting for confirmed breakout or exit trigger"
            strength = 0.0

        return SignalSnapshot(
            symbol=symbol,
            signal=signal,
            indicators={
                "fast_ema": float(fast_val),
                "slow_ema": float(slow_val),
                "atr": float(atr_val),
                "atr_pct": float(atr_pct),
                "breakout_level": float(breakout_level),
                "exit_level": float(prior_exit_low),
                "trend_ok": 1.0 if in_uptrend else 0.0,
                "volatility_ok": 1.0 if volatility_confirmed else 0.0,
            },
            signal_strength=float(strength),
            reason=reason,
            timestamp=current_date,
        )

    def get_action_plan(
        self,
        signal: SignalSnapshot,
        current_price: float,
        current_date: datetime,
    ) -> Optional[ActionPlan]:
        if signal.signal == "HOLD":
            return None

        atr = float(signal.indicators.get("atr", 0.0))
        stop_mult = float(self.params.get("stop_loss_atr_mult", 2.0))
        take_mult = float(self.params.get("take_profit_atr_mult", 4.0))

        if signal.signal == "BUY":
            stop_loss = current_price - (atr * stop_mult) if atr > 0 else self.calculate_stop_loss(current_price, "long")
            take_profit = current_price + (atr * take_mult) if atr > 0 else self.calculate_take_profit(current_price, "long")
        else:
            stop_loss = None
            take_profit = None

        return ActionPlan(
            symbol=signal.symbol,
            action=signal.signal,
            target_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=signal.reason,
            strength=abs(signal.signal_strength),
            timestamp=signal.timestamp or current_date,
            metadata=signal.indicators,
        )
