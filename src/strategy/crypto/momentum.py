"""Crypto momentum starter strategy.

Uses lookback returns to trigger buy/sell signals for crypto pairs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from src.strategy.base import BaseCryptoStrategy, StrategyMeta, StrategySignal
from src.strategy.utils import latest_price


class CryptoMomentumStrategy(BaseCryptoStrategy):
    meta = StrategyMeta(
        name="crypto_momentum",
        label="Crypto Momentum",
        category="signal",
        description="Starter crypto momentum strategy using recent returns.",
        asset_type="crypto",
        default_params={"lookback": 20, "threshold": 0.02, "risk_pct": 0.05},
        visible_in_dashboard=False,
    )
    summary = (
        "Uses lookback returns to trigger buy/sell signals for crypto pairs. "
        "Return = (P_t - P_{t-L}) / P_{t-L}. BUY if return >= threshold, "
        "SELL if return <= -threshold. Size = portfolio_value * risk_pct / price."
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
            price = current_prices.get(symbol, latest_price(data))
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
