"""
Strategy registry and factory.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base import StrategyBase, StrategyMeta
from .option import WheelStrategy
from .stock import (
    MomentumStrategy,
    ValueStrategy,
    TrendFollowingStrategy,
    ScalpingStrategy,
    StatisticalArbitrageStrategy,
)


# Factory type: either a class or a callable that returns a strategy
StrategyFactory = Union[Type[StrategyBase], Callable[[Optional[Dict]], StrategyBase]]


def _create_crypto_momentum(params: Optional[Dict[str, Any]] = None) -> StrategyBase:
    """Factory for crypto momentum strategy with proper defaults."""
    merged = {"asset_type": "crypto", **(params or {})}
    return MomentumStrategy(merged)


# Metadata for crypto_momentum (used by get_meta)
_CRYPTO_MOMENTUM_META = StrategyMeta(
    name="crypto_momentum",
    label="Crypto Momentum",
    category="signal",
    description="Dual momentum crossover strategy for crypto with risk management.",
    asset_type="crypto",
    default_params={
        "short_period": 12,
        "long_period": 26,
        "threshold": 0.005,
        "risk_pct": 0.10,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "qty_precision": 6,
        "min_qty": 0.000001,
    },
    visible_in_dashboard=True,
)


class StrategyRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, StrategyFactory] = {
            "momentum": MomentumStrategy,
            "value": ValueStrategy,
            "trend_following": TrendFollowingStrategy,
            "scalping": ScalpingStrategy,
            "statistical_arbitrage": StatisticalArbitrageStrategy,
            "crypto_momentum": _create_crypto_momentum,
            "wheel": WheelStrategy,
        }
        # Separate meta registry for factories that aren't classes
        self._meta_overrides: Dict[str, StrategyMeta] = {
            "crypto_momentum": _CRYPTO_MOMENTUM_META,
        }

    def list_strategies(self, dashboard_only: bool = False) -> List[str]:
        if not dashboard_only:
            return sorted(self._registry.keys())
        result = []
        for name, factory in self._registry.items():
            meta = self._meta_overrides.get(name) or getattr(factory, "meta", None)
            if meta and meta.visible_in_dashboard:
                result.append(name)
        return sorted(result)

    def get_meta(self, name: str) -> StrategyMeta:
        if name not in self._registry:
            raise KeyError(f"Unknown strategy: {name}")
        # Check override first, then class attribute
        if name in self._meta_overrides:
            return self._meta_overrides[name]
        factory = self._registry[name]
        return factory.meta

    def create(self, name: str, params: Optional[Dict] = None) -> StrategyBase:
        if name not in self._registry:
            raise KeyError(f"Unknown strategy: {name}")
        return self._registry[name](params)


_registry = StrategyRegistry()


def get_strategy_registry() -> StrategyRegistry:
    return _registry
