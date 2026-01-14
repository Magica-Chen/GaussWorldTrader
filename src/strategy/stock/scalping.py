"""Scalping strategy.

Trades short-term mean reversion around a fast EMA using a tight deviation band.
"""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal
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
        period = int(self.params["ema_period"])
        band = float(self.params["band_pct"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < period + 1:
                continue
            ema = ta.ema(data["close"], period)
            ema_val = safe_series(ema)
            price = current_prices.get(symbol, latest_price(data))
            if ema_val <= 0:
                continue
            deviation = (price - ema_val) / ema_val
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if deviation <= -band:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} below EMA{period}",
                        timestamp=current_date,
                    )
                )
            elif deviation >= band:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} above EMA{period}",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)
