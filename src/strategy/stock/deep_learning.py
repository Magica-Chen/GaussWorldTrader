"""Deep learning strategy template."""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd

from src.strategy.base import StrategyBase, StrategyMeta
from src.strategy.stock.momentum import MomentumStrategy


class DeepLearningStrategy(StrategyBase):
    meta = StrategyMeta(
        name="deep_learning",
        label="Deep Learning",
        category="ml",
        description="Template deep learning strategy (rule-based placeholder).",
        asset_type="stock",
        default_params={"lookback": 15, "threshold": 0.02, "risk_pct": 0.04},
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
        base = MomentumStrategy(
            {
                "lookback": self.params["lookback"],
                "threshold": self.params["threshold"],
                "risk_pct": self.params["risk_pct"],
            }
        )
        return base.generate_signals(current_date, current_prices, current_data, historical_data, portfolio)

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
