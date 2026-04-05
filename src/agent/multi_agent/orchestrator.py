"""Async coordinator for the multi-agent trading workflow."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.agent.multi_agent.agents import (
    BaseAnalystAgent,
    DecisionMakerAgent,
    FundamentalAnalystAgent,
    RiskManagerAgent,
    SentimentAnalystAgent,
    TechnicalAnalystAgent,
)
from src.agent.multi_agent.types import AgentReport, DebatePosition
from src.llm import create_provider
from src.strategy.stock import MomentumStrategy, TrendFollowingStrategy, ValueStrategy

if TYPE_CHECKING:
    from src.llm import BaseLLMProvider
    from src.strategy.base import SignalSnapshot
    from src.strategy.base import MarketDataContext


class MultiAgentOrchestrator:
    """Coordinate the multi-agent decision flow for one or more symbols."""

    def __init__(
        self,
        llm_provider: str,
        llm_model: str | None = None,
        debate_enabled: bool = False,
        max_concurrent_tasks: int = 4,
        base_position_pct: float = 0.05,
        mode: str = "llm",
        max_cost_per_run: float | None = None,
        finnhub_key: str | None = None,
        fred_key: str | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.mode = mode.strip().lower()
        if self.mode not in {"llm", "fast"}:
            raise ValueError("mode must be either 'llm' or 'fast'")
        if self.mode == "fast" and debate_enabled:
            raise ValueError("Fast mode does not support debate")

        self.llm = None
        if self.mode == "llm":
            self.llm = create_provider(llm_provider, model=llm_model)
        self.debate_enabled = debate_enabled
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_cost_per_run = max_cost_per_run
        self.semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self.technical_agent = None
        self.fundamental_agent = None
        self.sentiment_agent = None
        self.risk_manager = RiskManagerAgent(base_position_pct=base_position_pct)
        self.decision_maker = None
        if self.llm is not None:
            self.technical_agent = TechnicalAnalystAgent(self.llm)
            self.fundamental_agent = FundamentalAnalystAgent(
                self.llm,
                llm_provider=llm_provider,
                llm_model=llm_model,
                finnhub_key=finnhub_key,
                fred_key=fred_key,
            )
            self.sentiment_agent = SentimentAnalystAgent(
                self.llm,
                finnhub_key=finnhub_key,
            )
            self.decision_maker = DecisionMakerAgent(self.llm)
        self.fast_signal_agents = {
            "technical": TrendFollowingStrategy(),
            "fundamental": ValueStrategy(),
            "sentiment": MomentumStrategy(),
        }

    async def evaluate_symbols(
        self,
        symbols: list[str],
        market_context: MarketDataContext,
    ) -> dict[str, dict[str, Any]]:
        tasks = [self.evaluate_symbol(symbol, market_context) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {result["decision"].symbol: result for result in results}

    async def evaluate_symbol(
        self,
        symbol: str,
        market_context: MarketDataContext,
    ) -> dict[str, Any]:
        run_id = f"{symbol}:{market_context.current_date.isoformat()}:{uuid4().hex}"
        try:
            reports = await self._collect_reports(symbol, market_context, run_id=run_id)
            debate_positions: list[DebatePosition] = []
            if self.debate_enabled:
                debate_positions = await self._run_debate(
                    symbol,
                    reports,
                    run_id=run_id,
                )
            risk_assessment = await self._run_agent(
                self.risk_manager,
                symbol,
                market_context,
                reports=reports,
            )
            if self.mode == "fast":
                decision = self._build_fast_decision(
                    symbol,
                    market_context,
                    reports,
                    risk_assessment,
                )
            else:
                self._enforce_cost_limit(run_id, symbol, "pre-decision")
                decision = await self._run_agent(
                    self._require_decision_maker(),
                    symbol,
                    market_context,
                    reports=reports,
                    risk_assessment=risk_assessment,
                    debate_positions=debate_positions,
                    run_id=run_id,
                )
            usage = self._usage_summary(run_id)
            self._enforce_cost_limit(run_id, symbol, "post-decision")
            self._log_usage(symbol, usage)
            self.logger.info(
                "multi-agent decision complete for %s: %s",
                symbol,
                decision.action,
            )
            return {
                "symbol": symbol,
                "reports": reports,
                "debate_positions": debate_positions,
                "risk_assessment": risk_assessment,
                "decision": decision,
                "usage": usage,
            }
        finally:
            self._clear_usage(run_id)

    async def _collect_reports(
        self,
        symbol: str,
        market_context: MarketDataContext,
        run_id: str,
    ) -> list[AgentReport]:
        if self.mode == "fast":
            return self._collect_fast_reports(symbol, market_context)
        tasks = [
            self._run_agent(
                self._require_agent(self.technical_agent, "technical"),
                symbol,
                market_context,
                run_id=run_id,
            ),
            self._run_agent(
                self._require_agent(self.fundamental_agent, "fundamental"),
                symbol,
                market_context,
                run_id=run_id,
            ),
            self._run_agent(
                self._require_agent(self.sentiment_agent, "sentiment"),
                symbol,
                market_context,
                run_id=run_id,
            ),
        ]
        reports = await asyncio.gather(*tasks)
        return list(reports)

    async def _run_debate(
        self,
        symbol: str,
        reports: list[AgentReport],
        run_id: str,
    ) -> list[DebatePosition]:
        tasks = [
            self._run_debate_position(symbol, reports, "bull", run_id),
            self._run_debate_position(symbol, reports, "bear", run_id),
        ]
        positions = await asyncio.gather(*tasks)
        return list(positions)

    def _collect_fast_reports(
        self,
        symbol: str,
        market_context: MarketDataContext,
    ) -> list[AgentReport]:
        current_price = float(market_context.current_prices[symbol])
        bars = market_context.historical_bars[symbol]
        scales = {
            "technical": 8.0,
            "fundamental": 12.0,
            "sentiment": 20.0,
        }
        reports = []
        for role, strategy in self.fast_signal_agents.items():
            snapshot = strategy.get_signal(
                symbol=symbol,
                current_date=market_context.current_date,
                current_price=current_price,
                current_data={},
                historical_data=bars,
                portfolio=None,
            )
            reports.append(self._fast_report(role, strategy.meta.name, snapshot, scales[role]))
        return reports

    def _fast_report(
        self,
        role: str,
        source_strategy: str,
        snapshot: SignalSnapshot | None,
        scale: float,
    ) -> AgentReport:
        if snapshot is None:
            return AgentReport(
                action="HOLD",
                confidence=0.0,
                thesis=f"Insufficient history for fast {role} analysis",
                agent_name=role,
                summary=f"{role} warmup incomplete",
                risk_flags=["warmup_incomplete"],
                metrics={"source_strategy": source_strategy},
            )

        confidence = min(1.0, abs(float(snapshot.signal_strength)) * scale)
        if snapshot.signal == "HOLD":
            confidence = min(0.49, max(0.1, confidence))
        else:
            confidence = max(0.55, confidence)

        return AgentReport(
            action=snapshot.signal,
            confidence=confidence,
            thesis=snapshot.reason,
            agent_name=role,
            summary=snapshot.reason,
            key_points=[snapshot.reason],
            metrics={
                "source_strategy": source_strategy,
                **snapshot.indicators,
            },
        )

    def _build_fast_decision(
        self,
        symbol: str,
        market_context: MarketDataContext,
        reports: list[AgentReport],
        risk_assessment: Any,
    ) -> Any:
        action_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        for report in reports:
            action = report.action.upper()
            if action not in action_scores:
                action = "HOLD"
            action_scores[action] += max(0.0, float(report.confidence))

        buy_score = action_scores["BUY"]
        sell_score = action_scores["SELL"]
        active_score = buy_score + sell_score
        action = "HOLD"
        dominant_score = action_scores["HOLD"]
        if active_score > 0:
            if buy_score > sell_score:
                dominant_action = "BUY"
                dominant_score = buy_score
            elif sell_score > buy_score:
                dominant_action = "SELL"
                dominant_score = sell_score
            else:
                dominant_action = "HOLD"
            dominance = 0.0 if active_score == 0 else dominant_score / active_score
            if dominant_action != "HOLD" and dominance >= 0.55:
                action = dominant_action

        total_score = sum(action_scores.values())
        confidence = 0.0 if total_score == 0 else dominant_score / total_score
        participating_agents = [
            report.agent_name
            for report in reports
            if action != "HOLD" and report.action.upper() == action
        ]
        dissenting_agents = [
            report.agent_name
            for report in reports
            if report.action.upper() not in {action, "HOLD"}
        ]

        price = float(market_context.current_prices[symbol])
        stop_loss = None
        take_profit = None
        if action == "BUY":
            stop_loss = price * (1 - risk_assessment.stop_loss_pct)
            take_profit = price * (1 + risk_assessment.take_profit_pct)
        elif action == "SELL":
            stop_loss = price * (1 + risk_assessment.stop_loss_pct)
            take_profit = price * (1 - risk_assessment.take_profit_pct)

        from src.agent.multi_agent.types import ConsensusDecision

        reason = (
            "Fast mode weighted vote: "
            f"BUY={buy_score:.2f}, SELL={sell_score:.2f}, "
            f"HOLD={action_scores['HOLD']:.2f}, risk={risk_assessment.risk_level}"
        )
        return ConsensusDecision(
            action=action,
            confidence=confidence,
            reason=reason,
            symbol=symbol,
            target_price=price if action != "HOLD" else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            participating_agents=participating_agents,
            dissenting_agents=dissenting_agents,
            debate_summary="",
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self.semaphore is None or self._semaphore_loop is not loop:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            self._semaphore_loop = loop
        return self.semaphore

    async def _run_agent(
        self,
        agent: BaseAnalystAgent,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> Any:
        async with self._get_semaphore():
            return await agent.analyze(symbol, market_context, **kwargs)

    async def _run_debate_position(
        self,
        symbol: str,
        reports: list[AgentReport],
        side: str,
        run_id: str,
    ) -> DebatePosition:
        async with self._get_semaphore():
            return await asyncio.to_thread(
                self._generate_debate_position,
                symbol,
                reports,
                side,
                self._require_llm(),
                run_id,
            )

    def _generate_debate_position(
        self,
        symbol: str,
        reports: list[AgentReport],
        side: str,
        llm: BaseLLMProvider,
        run_id: str,
    ) -> DebatePosition:
        prompt = (
            f"You are taking the {side} side in a trading debate.\n"
            f"Symbol: {symbol}\n"
            f"Analyst reports:\n{json.dumps([asdict(report) for report in reports], indent=2)}\n\n"
            f"Build the strongest {side} case and rebut the opposing side."
        )
        position = llm.generate_structured(
            prompt,
            DebatePosition,
            context={
                "run_id": run_id,
                "symbol": symbol,
                "agent_role": f"debate_{side}",
            },
        )
        position.symbol = symbol
        position.side = side
        return position

    def _usage_summary(self, run_id: str) -> dict[str, Any]:
        if self.llm is None:
            return {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "latency_ms": 0.0,
            }
        return self.llm.get_usage_summary(run_id)

    def _enforce_cost_limit(
        self,
        run_id: str,
        symbol: str,
        stage: str,
    ) -> None:
        if self.llm is None or self.max_cost_per_run is None:
            return
        usage = self.llm.get_usage_summary(run_id)
        estimated_cost = usage.get("estimated_cost_usd")
        if estimated_cost is None or estimated_cost <= self.max_cost_per_run:
            return
        raise RuntimeError(
            f"multi-agent run for {symbol} exceeded max cost during {stage}: "
            f"${estimated_cost:.4f} > ${self.max_cost_per_run:.4f}"
        )

    def _log_usage(self, symbol: str, usage: dict[str, Any]) -> None:
        if usage["calls"] == 0:
            self.logger.info("multi-agent usage for %s: fast mode, no LLM calls", symbol)
            return
        cost = usage["estimated_cost_usd"]
        cost_text = "n/a" if cost is None else f"${cost:.4f}"
        self.logger.info(
            "multi-agent usage for %s: calls=%s tokens=%s cost=%s latency_ms=%.1f",
            symbol,
            usage["calls"],
            usage["total_tokens"],
            cost_text,
            usage["latency_ms"],
        )

    def _clear_usage(self, run_id: str) -> None:
        if self.llm is not None:
            self.llm.clear_usage_events(run_id)

    def _require_agent(
        self,
        agent: BaseAnalystAgent | None,
        role: str,
    ) -> BaseAnalystAgent:
        if agent is None:
            raise ValueError(f"{role} agent is unavailable in {self.mode} mode")
        return agent

    def _require_decision_maker(self) -> DecisionMakerAgent:
        if self.decision_maker is None:
            raise ValueError(f"decision agent is unavailable in {self.mode} mode")
        return self.decision_maker

    def _require_llm(self) -> BaseLLMProvider:
        if self.llm is None:
            raise ValueError(f"LLM provider is unavailable in {self.mode} mode")
        return self.llm
