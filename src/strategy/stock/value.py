"""Value strategy.

Compares the current price to a long-term SMA and trades when the discount
or premium crosses the configured threshold.
"""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal
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
        period = int(self.params["sma_period"])
        discount = float(self.params["discount_pct"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < period + 1:
                continue
            sma = ta.sma(data["close"], period)
            sma_value = safe_series(sma)
            price = current_prices.get(symbol, latest_price(data))
            if sma_value <= 0:
                continue
            deviation = (price - sma_value) / sma_value
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if deviation <= -discount:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} below SMA{period}",
                        timestamp=current_date,
                    )
                )
            elif deviation >= discount:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} above SMA{period}",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)
