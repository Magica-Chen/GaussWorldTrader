"""Gaussian process strategy.

Uses a rolling return z-score as a proxy for uncertainty-aware mean reversion.
"""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd

from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal
from src.strategy.utils import latest_price


class GaussianProcessStrategy(StrategyBase):
    meta = StrategyMeta(
        name="gaussian_process",
        label="Gaussian Process",
        category="signal",
        description="Proxy uncertainty-aware mean reversion strategy.",
        asset_type="stock",
        default_params={"window": 30, "zscore": 1.0, "risk_pct": 0.03},
        visible_in_dashboard=True,
    )
    summary = (
        "Uses return z-scores as a proxy for uncertainty-aware mean reversion. "
        "z = (r_t - mean(r)) / std(r). "
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
        window = int(self.params["window"])
        zscore = float(self.params["zscore"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < window + 2:
                continue
            returns = data["close"].pct_change().dropna().iloc[-window:]
            if returns.empty:
                continue
            mean = float(returns.mean())
            std = float(returns.std()) or 1e-6
            score = (returns.iloc[-1] - mean) / std
            price = current_prices.get(symbol, latest_price(data))
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if score <= -zscore:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason="uncertainty signal BUY",
                        timestamp=current_date,
                    )
                )
            elif score >= zscore:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason="uncertainty signal SELL",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)
