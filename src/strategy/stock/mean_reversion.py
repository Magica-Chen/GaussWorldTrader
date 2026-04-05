"""Mean reversion strategy using Bollinger Bands and RSI confirmation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import (
    ActionPlan,
    SignalSnapshot,
    StrategyBase,
    StrategyMeta,
    StrategySignal,
)
from src.strategy.utils import latest_price, safe_series


ta = TechnicalAnalysis()


class MeanReversionStrategy(StrategyBase):
    """Trade reversions from stretched Bollinger Band moves."""

    meta = StrategyMeta(
        name="mean_reversion",
        label="Mean Reversion",
        category="signal",
        description="Bollinger Band reversions confirmed by RSI extremes.",
        asset_type="stock",
        default_params={
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "rsi_period": 14,
            "oversold_rsi": 30.0,
            "overbought_rsi": 70.0,
            "risk_pct": 0.04,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
        },
        visible_in_dashboard=True,
    )
    summary = (
        "Looks for mean reversion opportunities when price stretches beyond the "
        "Bollinger Bands and RSI confirms an extreme condition."
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
            if plan is None or plan.action == "HOLD":
                continue

            portfolio_value = getattr(
                portfolio,
                "get_portfolio_value",
                lambda *_: 100000,
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
        bb_period = int(self.params["bb_period"])
        rsi_period = int(self.params["rsi_period"])
        std_dev = float(self.params["bb_std_dev"])
        oversold = float(self.params["oversold_rsi"])
        overbought = float(self.params["overbought_rsi"])

        min_bars = max(bb_period, rsi_period) + 1
        if len(historical_data) < min_bars:
            return None

        close = historical_data["close"]
        upper, middle, lower = ta.bollinger_bands(close, bb_period, std_dev)
        rsi = ta.rsi(close, rsi_period)
        upper_value = safe_series(upper)
        middle_value = safe_series(middle)
        lower_value = safe_series(lower)
        rsi_value = safe_series(rsi)

        if current_price <= lower_value and rsi_value <= oversold:
            signal = "BUY"
            reason = "price below lower Bollinger Band with oversold RSI"
        elif current_price >= upper_value and rsi_value >= overbought:
            signal = "SELL"
            reason = "price above upper Bollinger Band with overbought RSI"
        else:
            signal = "HOLD"
            reason = "no mean reversion edge"

        band_width = upper_value - lower_value
        strength = 0.0 if band_width <= 0 else (current_price - middle_value) / band_width
        return SignalSnapshot(
            symbol=symbol,
            signal=signal,
            indicators={
                "upper_band": float(upper_value),
                "middle_band": float(middle_value),
                "lower_band": float(lower_value),
                "rsi": float(rsi_value),
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
