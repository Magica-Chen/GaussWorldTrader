"""Value strategy.

Compares the current price to a long-term SMA and trades when the discount
or premium crosses the configured threshold.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import ActionPlan, SignalSnapshot, StrategyBase, StrategyMeta, StrategySignal
from src.strategy.utils import latest_price, safe_series


ta = TechnicalAnalysis()


class ValueStrategy(StrategyBase):
    meta = StrategyMeta(
        name="value",
        label="Value",
        category="signal",
        description="Looks for price discounts to a long-term average.",
        asset_type="stock",
        default_params={"sma_period": 50, "discount_pct": 0.03, "risk_pct": 0.04},
        visible_in_dashboard=True,
    )
    summary = (
        "Trades when price discounts or premiums to a long-term average are extreme. "
        "SMA_L = mean(close[-L:]); deviation = (price - SMA_L) / SMA_L. "
        "BUY if deviation <= -discount_pct, SELL if >= discount_pct. "
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
        period = int(self.params["sma_period"])
        discount = float(self.params["discount_pct"])

        if len(historical_data) < period + 1:
            return None
        sma = ta.sma(historical_data["close"], period)
        sma_value = safe_series(sma)
        if sma_value <= 0:
            return None
        deviation = (current_price - sma_value) / sma_value

        if deviation <= -discount:
            signal = "BUY"
            reason = f"{deviation:.2%} below SMA{period}"
        elif deviation >= discount:
            signal = "SELL"
            reason = f"{deviation:.2%} above SMA{period}"
        else:
            signal = "HOLD"
            reason = "within value band"

        return SignalSnapshot(
            symbol=symbol,
            signal=signal,
            indicators={
                "sma": float(sma_value),
                "deviation": float(deviation),
                "discount_pct": float(discount),
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
