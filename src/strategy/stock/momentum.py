"""Momentum strategy."""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd

from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal


def _latest_price(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    return float(data["close"].iloc[-1])


class MomentumStrategy(StrategyBase):
    meta = StrategyMeta(
        name="momentum",
        label="Momentum",
        category="signal",
        description="Buys strength and sells weakness using recent returns.",
        asset_type="stock",
        default_params={"lookback": 20, "threshold": 0.02, "risk_pct": 0.05},
        visible_in_dashboard=True,
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
        lookback = int(self.params["lookback"])
        threshold = float(self.params["threshold"])

        for symbol, data in historical_data.items():
            if len(data) < lookback + 1:
                continue
            window = data["close"].iloc[-lookback:]
            returns = (window.iloc[-1] - window.iloc[0]) / window.iloc[0]
            price = current_prices.get(symbol, _latest_price(data))
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if returns >= threshold:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason=f"{returns:.2%} momentum",
                        timestamp=current_date,
                    )
                )
            elif returns <= -threshold:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason=f"{returns:.2%} drawdown",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)
