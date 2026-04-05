"""Multi-agent meta-strategy built on top of the standard strategy interface."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from src.agent.multi_agent import MultiAgentOrchestrator
from src.strategy.base import (
    ActionPlan,
    MarketDataContext,
    SignalSnapshot,
    StrategyBase,
    StrategyMeta,
    StrategySignal,
)

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return float(value)


class MultiAgentStrategy(StrategyBase):
    """Strategy wrapper around the multi-agent decision system."""

    meta = StrategyMeta(
        name="multi_agent",
        label="Multi-Agent",
        category="meta",
        description="Committee-style strategy using technical, fundamental, and sentiment "
        "analysts.",
        asset_type="stock",
        visible_in_dashboard=True,
        default_params={
            "risk_pct": 0.05,
            "llm_provider": "openai",
            "llm_model": None,
            "mode": "llm",
            "debate_enabled": False,
            "max_concurrent_tasks": 4,
            "max_cost_per_run": None,
        },
    )
    summary = (
        "Meta-strategy that runs technical, fundamental, and sentiment analysts in "
        "parallel, applies a math-only risk review, and asks a final LLM decision "
        "maker for the action plan."
    )

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        env_params = {
            "llm_provider": os.getenv(
                "MULTI_AGENT_LLM_PROVIDER",
                self.meta.default_params["llm_provider"],
            ),
            "llm_model": os.getenv("MULTI_AGENT_LLM_MODEL") or None,
            "mode": os.getenv("MULTI_AGENT_MODE", self.meta.default_params["mode"]),
            "debate_enabled": _env_flag(
                "MULTI_AGENT_DEBATE_ENABLED",
                bool(self.meta.default_params["debate_enabled"]),
            ),
            "max_cost_per_run": _env_optional_float("MULTI_AGENT_MAX_COST_PER_RUN"),
        }
        merged_params = {**self.meta.default_params, **env_params, **(params or {})}
        super().__init__(merged_params)
        self.orchestrator = MultiAgentOrchestrator(
            llm_provider=str(self.params["llm_provider"]),
            llm_model=self.params.get("llm_model"),
            mode=str(self.params["mode"]),
            debate_enabled=bool(self.params["debate_enabled"]),
            max_concurrent_tasks=int(self.params["max_concurrent_tasks"]),
            base_position_pct=float(self.params["risk_pct"]),
            max_cost_per_run=self._optional_float(self.params.get("max_cost_per_run")),
        )

    def generate_signals(
        self,
        current_date: datetime,
        current_prices: dict[str, float],
        current_data: dict[str, Any],
        historical_data: dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> list[dict[str, Any]]:
        market_context = self._build_market_context(
            current_date,
            current_prices,
            historical_data,
            portfolio,
        )
        results = self._run_async(
            self.orchestrator.evaluate_symbols(sorted(historical_data), market_context)
        )
        signals: list[StrategySignal] = []
        for symbol, result in results.items():
            snapshot = self._snapshot_from_result(symbol, current_date, result)
            plan = self.get_action_plan(snapshot, current_prices[symbol], current_date)
            if plan is None or plan.action == "HOLD":
                continue
            risk_pct = min(
                float(self.params["risk_pct"]),
                float(result["risk_assessment"].max_position_pct),
            )
            quantity = self._position_size(
                current_prices[symbol],
                market_context.portfolio_value,
                risk_pct,
            )
            if quantity <= 0:
                continue
            signals.append(self._plan_to_signal(plan, quantity, current_prices[symbol]))
        return self._normalize(signals)

    def get_signal(
        self,
        symbol: str,
        current_date: datetime,
        current_price: float,
        current_data: dict[str, Any],
        historical_data: pd.DataFrame,
        portfolio: Any = None,
    ) -> SignalSnapshot | None:
        market_context = self._build_market_context(
            current_date,
            {symbol: current_price},
            {symbol: historical_data},
            portfolio,
        )
        result = self._run_async(self.orchestrator.evaluate_symbol(symbol, market_context))
        return self._snapshot_from_result(symbol, current_date, result)

    def get_action_plan(
        self,
        signal: SignalSnapshot,
        current_price: float,
        current_date: datetime,
    ) -> ActionPlan | None:
        decision = signal.metadata["decision"]
        if decision["action"] == "HOLD":
            return None
        return ActionPlan(
            symbol=signal.symbol,
            action=decision["action"],
            target_price=decision["target_price"],
            stop_loss=decision["stop_loss"],
            take_profit=decision["take_profit"],
            reason=decision["reason"],
            strength=signal.signal_strength,
            timestamp=signal.timestamp or current_date,
            metadata=signal.metadata,
        )

    def _build_market_context(
        self,
        current_date: datetime,
        current_prices: dict[str, float],
        historical_data: dict[str, pd.DataFrame],
        portfolio: Any,
    ) -> MarketDataContext:
        portfolio_value = self._portfolio_value(portfolio, current_prices)
        available_cash = float(getattr(portfolio, "cash", portfolio_value))
        return MarketDataContext(
            current_date=current_date,
            current_prices=current_prices,
            historical_bars=historical_data,
            portfolio_value=portfolio_value,
            available_cash=available_cash,
            current_positions=getattr(portfolio, "positions", None),
        )

    def _portfolio_value(
        self,
        portfolio: Any,
        current_prices: dict[str, float],
    ) -> float:
        if portfolio is None:
            return 100000.0
        getter = getattr(portfolio, "get_portfolio_value", None)
        if getter is None:
            raise ValueError("Portfolio object must define get_portfolio_value")
        return float(getter(current_prices))

    def _snapshot_from_result(
        self,
        symbol: str,
        current_date: datetime,
        result: dict[str, Any],
    ) -> SignalSnapshot:
        decision = result["decision"]
        risk_assessment = result["risk_assessment"]
        metadata = {
            "decision": asdict(decision),
            "risk_assessment": asdict(risk_assessment),
            "reports": [asdict(report) for report in result["reports"]],
            "debate_positions": [
                asdict(position) for position in result["debate_positions"]
            ],
            "usage": result.get("usage", {}),
        }
        return SignalSnapshot(
            symbol=symbol,
            signal=decision.action,
            indicators={
                "decision_confidence": float(decision.confidence),
                "max_position_pct": float(risk_assessment.max_position_pct),
                "volatility_pct": float(risk_assessment.volatility_pct),
            },
            signal_strength=float(decision.confidence),
            reason=decision.reason,
            timestamp=current_date,
            metadata=metadata,
        )

    def _run_async(self, coroutine: Any) -> Any:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coroutine)

    def _optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        return float(value)
