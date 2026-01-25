"""Trend following strategy.

Signals are generated from fast/slow SMA crossovers to capture trend direction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import ActionPlan, SignalSnapshot, StrategyBase, StrategyMeta, StrategySignal
from src.strategy.utils import latest_price, safe_series


ta = TechnicalAnalysis()


class TrendFollowingStrategy(StrategyBase):
    meta = StrategyMeta(
        name="trend_following",
        label="Trend Following",
        category="signal",
        description="Trades on moving average crossovers.",
        asset_type="stock",
        default_params={"fast": 20, "slow": 50, "risk_pct": 0.05},
        visible_in_dashboard=True,
    )
    summary = (
        "Captures trend direction using moving average crossovers. "
        "Fast SMA vs slow SMA crossover. BUY when SMA_fast > SMA_slow, "
        "SELL when SMA_fast < SMA_slow. Size = portfolio_value * risk_pct / price."
    )

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

            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(
                current_prices
            )
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
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])

        if len(historical_data) < slow + 1:
            return None
        fast_sma = ta.sma(historical_data["close"], fast)
        slow_sma = ta.sma(historical_data["close"], slow)
        fast_val = safe_series(fast_sma)
        slow_val = safe_series(slow_sma)

        if fast_val > slow_val:
            signal = "BUY"
            reason = "fast SMA above slow SMA"
        elif fast_val < slow_val:
            signal = "SELL"
            reason = "fast SMA below slow SMA"
        else:
            signal = "HOLD"
            reason = "SMAs equal"

        strength = 0.0 if slow_val == 0 else (fast_val - slow_val) / abs(slow_val)
        return SignalSnapshot(
            symbol=symbol,
            signal=signal,
            indicators={
                "fast_sma": float(fast_val),
                "slow_sma": float(slow_val),
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

        side = "long" if signal.signal == "BUY" else "short"
        stop_loss = self.calculate_stop_loss(current_price, side)
        take_profit = self.calculate_take_profit(current_price, side)

        return ActionPlan(
            symbol=signal.symbol,
            action=signal.signal,
            target_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=signal.reason,
            strength=abs(signal.signal_strength),
            timestamp=signal.timestamp or current_date,
        )
