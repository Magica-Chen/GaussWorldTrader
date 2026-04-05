"""
Strategy registry and factory.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

from .base import StrategyBase, StrategyMeta
from .crypto import BTCVolatilityBreakoutStrategy
from .option import VerticalSpreadStrategy, WheelStrategy
from .stock import (
    MacroFactorStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    ScalpingStrategy,
    StatisticalArbitrageStrategy,
    TrendFollowingStrategy,
    ValueStrategy,
)

# Factory type: either a class or a callable that returns a strategy
StrategyFactory = Union[type[StrategyBase], Callable[[dict | None], StrategyBase]]


def _create_crypto_momentum(params: dict[str, Any] | None = None) -> StrategyBase:
    """Factory for crypto momentum strategy with proper defaults."""
    merged = {"asset_type": "crypto", **(params or {})}
    return MomentumStrategy(merged)


def _create_multi_agent(params: dict[str, Any] | None = None) -> StrategyBase:
    """Lazy factory for the multi-agent strategy."""
    from .multi_agent_strategy import MultiAgentStrategy

    return MultiAgentStrategy(params)


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

_MULTI_AGENT_META = StrategyMeta(
    name="multi_agent",
    label="Multi-Agent",
    category="meta",
    description="Committee-style strategy using technical, fundamental, and sentiment "
    "analysts.",
    asset_type="stock",
    default_params={
        "risk_pct": 0.05,
        "llm_provider": "openai",
        "llm_model": None,
        "mode": "llm",
        "debate_enabled": False,
        "max_concurrent_tasks": 4,
        "max_cost_per_run": None,
    },
    visible_in_dashboard=True,
)


class StrategyRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, StrategyFactory] = {
            "momentum": MomentumStrategy,
            "value": ValueStrategy,
            "trend_following": TrendFollowingStrategy,
            "scalping": ScalpingStrategy,
            "statistical_arbitrage": StatisticalArbitrageStrategy,
            "mean_reversion": MeanReversionStrategy,
            "macro_factor": MacroFactorStrategy,
            "crypto_momentum": _create_crypto_momentum,
            "multi_agent": _create_multi_agent,
            "btc_volatility_breakout": BTCVolatilityBreakoutStrategy,
            "wheel": WheelStrategy,
            "vertical_spread": VerticalSpreadStrategy,
        }
        # Separate meta registry for factories that aren't classes
        self._meta_overrides: dict[str, StrategyMeta] = {
            "crypto_momentum": _CRYPTO_MOMENTUM_META,
            "multi_agent": _MULTI_AGENT_META,
        }

    def list_strategies(self, dashboard_only: bool = False) -> list[str]:
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

    def create(self, name: str, params: dict | None = None) -> StrategyBase:
        if name not in self._registry:
            raise KeyError(f"Unknown strategy: {name}")
        return self._registry[name](params)


_registry = StrategyRegistry()


def get_strategy_registry() -> StrategyRegistry:
    return _registry
