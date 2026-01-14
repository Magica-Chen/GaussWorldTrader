"""
Strategy registry and factory.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import StrategyBase, StrategyMeta
from .option import WheelStrategy
from .crypto import CryptoMomentumStrategy
from .stock import (
    MomentumStrategy,
    ValueStrategy,
    TrendFollowingStrategy,
    ScalpingStrategy,
    StatisticalArbitrageStrategy,
)


class StrategyRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[StrategyBase]] = {
            "momentum": MomentumStrategy,
            "value": ValueStrategy,
            "trend_following": TrendFollowingStrategy,
            "scalping": ScalpingStrategy,
            "statistical_arbitrage": StatisticalArbitrageStrategy,
            "crypto_momentum": CryptoMomentumStrategy,
            "wheel": WheelStrategy,
        }

    def list_strategies(self, dashboard_only: bool = False) -> List[str]:
        if not dashboard_only:
            return sorted(self._registry.keys())
        return sorted(
            name
            for name, cls in self._registry.items()
            if cls.meta.visible_in_dashboard
        )

    def get_meta(self, name: str) -> StrategyMeta:
        if name not in self._registry:
            raise KeyError(f"Unknown strategy: {name}")
        return self._registry[name].meta

    def create(self, name: str, params: Optional[Dict] = None) -> StrategyBase:
        if name not in self._registry:
            raise KeyError(f"Unknown strategy: {name}")
        return self._registry[name](params)


_registry = StrategyRegistry()


def get_strategy_registry() -> StrategyRegistry:
    return _registry
