"""Statistical arbitrage strategy.

Uses a rolling z-score of recent returns to trade mean reversion extremes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd

from src.strategy.base import ActionPlan, SignalSnapshot, StrategyBase, StrategyMeta, StrategySignal
from src.strategy.utils import latest_price


class StatisticalArbitrageStrategy(StrategyBase):
    meta = StrategyMeta(
        name="statistical_arbitrage",
        label="Statistical Arbitrage",
        category="signal",
        description="Mean reversion using z-score of recent returns.",
        asset_type="stock",
        default_params={"window": 20, "zscore": 1.5, "risk_pct": 0.03},
        visible_in_dashboard=True,
    )
    summary = (
        "Uses rolling return z-scores to trade mean reversion extremes. "
        "Compute returns r over window; z = (r_t - mean(r)) / std(r). "
        "BUY if z <= -zscore, SELL if z >= zscore. "
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
        window = int(self.params["window"])
        zscore = float(self.params["zscore"])

        if len(historical_data) < window + 2:
            return None
        returns = historical_data["close"].pct_change().dropna().iloc[-window:]
        if returns.empty:
            return None
        mean = float(returns.mean())
        std = float(returns.std()) or 1e-6
        score = (returns.iloc[-1] - mean) / std

        if score <= -zscore:
            signal = "BUY"
            reason = f"z-score {score:.2f} oversold"
        elif score >= zscore:
            signal = "SELL"
            reason = f"z-score {score:.2f} overbought"
        else:
            signal = "HOLD"
            reason = "z-score within range"

        return SignalSnapshot(
            symbol=symbol,
            signal=signal,
            indicators={
                "zscore": float(score),
                "threshold": float(zscore),
                "mean": float(mean),
                "std": float(std),
            },
            signal_strength=float(score),
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
