"""Statistical arbitrage strategy.

Uses a rolling z-score of recent returns to trade mean reversion extremes.
"""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd

from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal
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
                        reason=f"z-score {score:.2f} oversold",
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
                        reason=f"z-score {score:.2f} overbought",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)
