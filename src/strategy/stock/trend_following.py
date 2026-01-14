"""Trend following strategy."""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal


ta = TechnicalAnalysis()


def _latest_price(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    return float(data["close"].iloc[-1])


def _safe_series(series: pd.Series, default: float = 0.0) -> float:
    if series.empty:
        return default
    return float(series.iloc[-1])


class TrendFollowingStrategy(StrategyBase):
    meta = StrategyMeta(
        name="trend_following",
        label="Trend Following",
        category="signal",
        description="Trades on moving average crossovers.",
        asset_type="stock",
        default_params={"fast": 20, "slow": 50, "risk_pct": 0.05},
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
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < slow + 1:
                continue
            fast_sma = ta.sma(data["close"], fast)
            slow_sma = ta.sma(data["close"], slow)
            fast_val = _safe_series(fast_sma)
            slow_val = _safe_series(slow_sma)
            price = current_prices.get(symbol, _latest_price(data))
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if fast_val > slow_val:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason="fast SMA above slow SMA",
                        timestamp=current_date,
                    )
                )
            elif fast_val < slow_val:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason="fast SMA below slow SMA",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)

    def generate_trading_plan(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> List[Dict[str, Any]]:
        return super().generate_trading_plan(
            current_date,
            current_prices,
            current_data,
            historical_data,
            portfolio,
        )
