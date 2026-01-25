"""Scalping strategy.

Trades short-term mean reversion around a fast EMA using a tight deviation band.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import ActionPlan, SignalSnapshot, StrategyBase, StrategyMeta, StrategySignal
from src.strategy.utils import latest_price, safe_series


ta = TechnicalAnalysis()


class ScalpingStrategy(StrategyBase):
    meta = StrategyMeta(
        name="scalping",
        label="Scalping",
        category="signal",
        description="Short-term mean reversion around a short EMA.",
        asset_type="stock",
        default_params={"ema_period": 5, "band_pct": 0.003, "risk_pct": 0.02},
        visible_in_dashboard=True,
    )
    summary = (
        "Trades short-term mean reversion around a fast EMA using a tight band. "
        "EMA_p defines the mean. Deviation = (price - EMA_p) / EMA_p. "
        "BUY if deviation <= -band_pct, SELL if deviation >= band_pct. "
        "Size = portfolio_value * risk_pct / price."
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
        period = int(self.params["ema_period"])
        band = float(self.params["band_pct"])

        if len(historical_data) < period + 1:
            return None
        ema = ta.ema(historical_data["close"], period)
        ema_val = safe_series(ema)
        if ema_val <= 0:
            return None
        deviation = (current_price - ema_val) / ema_val

        if deviation <= -band:
            signal = "BUY"
            reason = f"{deviation:.2%} below EMA{period}"
        elif deviation >= band:
            signal = "SELL"
            reason = f"{deviation:.2%} above EMA{period}"
        else:
            signal = "HOLD"
            reason = "within scalping band"

        return SignalSnapshot(
            symbol=symbol,
            signal=signal,
            indicators={
                "ema": float(ema_val),
                "deviation": float(deviation),
                "band_pct": float(band),
            },
            signal_strength=float(deviation),
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
