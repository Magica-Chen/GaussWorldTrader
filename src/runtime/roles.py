"""Four session responsibilities over a frozen research boundary and controlled gateway."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal, ROUND_DOWN
from uuid import uuid4

from .evidence import SnapshotDataReader, evaluate_rules
from .models import (
    Alternative,
    Candidate,
    Instrument,
    OrderIntent,
    Rule,
    SessionPlan,
    StrategyApproval,
    ValidationRecord,
)
from .store import digest
from .options import opening_legs, quote_checks


class PostGauss:
    name = "PostGauss"

    def run(self, service, session, now):
        """Screen frozen observations and contract metadata; return explicit no-trade reviews."""
        from .options import select_alternatives

        approvals = service.approvals(now)
        account = service.account_profile.model_copy(
            update={
                "id": uuid4().hex,
                "created_at": now,
                "reserved_capital": service.account_profile.reserved_capital
                + service.reserved_capital(),
            }
        )
        service.store.put("account_profiles", account, service.scope)
        snapshot = service.evidence.freeze(session.id, account, now, approvals)
        reader = SnapshotDataReader(service.store, snapshot)
        candidates, rejections, suitability_ids = [], [], []
        metadata = {}
        for row in reversed(service.store.list("instrument_versions", service.scope, 100000)):
            if row["created_at"] <= now.isoformat():
                instrument = Instrument.model_validate(row["instrument"])
                metadata[instrument.symbol] = instrument
        for symbol in service.config.symbols[: service.config.scan_universe_limit]:
            bars = reader.bars(symbol)
            if bars.empty:
                rejections.append({"symbol": symbol, "reason": "MISSING_COMPLETE_OHLC"})
                continue
            alternatives, rationales = [], {}
            for approval in approvals:
                selector = approval.option_selector
                if approval.asset_type == "option" and selector is None:
                    rejections.append({"symbol": symbol, "reason": "OPTION_SELECTOR_REQUIRED"})
                    continue
                signal_strategy = selector.signal_strategy_id if selector else approval.strategy_id
                if len(bars) < approval.minimum_history:
                    rejections.append({"symbol": symbol, "reason": "INSUFFICIENT_HISTORY"})
                    continue
                # These registered signal implementations operate exclusively on supplied bars.
                from src.strategy.registry import get_strategy_registry

                registry = get_strategy_registry()
                pure_strategies = {
                    "momentum",
                    "trend_following",
                    "mean_reversion",
                    "value",
                    "scalping",
                    "statistical_arbitrage",
                    "multi_agent",
                }
                if signal_strategy not in pure_strategies:
                    service.store.put(
                        "strategy_experiments",
                        {
                            "id": uuid4().hex,
                            "strategy_id": approval.strategy_id,
                            "reason": "SNAPSHOT_ADAPTER_REQUIRED",
                            "snapshot_id": snapshot.id,
                        },
                        service.scope,
                    )
                    continue
                params = json.loads(approval.parameter_json)
                if signal_strategy == "multi_agent" and params.get("mode") != "fast":
                    rejections.append(
                        {"symbol": symbol, "reason": "PAID_ANALYSIS_REQUIRES_BUDGET_WORKER"}
                    )
                    continue
                strategy = registry.create(signal_strategy, params)
                price = Decimal(str(bars.iloc[-1]["close"]))
                signal = strategy.get_signal(
                    symbol=symbol,
                    current_date=snapshot.data_context.signal_as_of,
                    current_price=float(price),
                    current_data={},
                    historical_data=bars,
                    portfolio=None,
                )
                required_signal = "SELL" if selector and "put" in selector.structure else "BUY"
                if signal is None or signal.signal != required_signal:
                    rejections.append(
                        {
                            "symbol": symbol,
                            "strategy": approval.strategy_id,
                            "reason": "SCREEN_NO_DIRECTIONAL_SIGNAL",
                        }
                    )
                    continue
                action = strategy.get_action_plan(
                    signal, float(price), snapshot.data_context.signal_as_of
                )
                stop = Decimal(str(action.stop_loss)) if action and action.stop_loss else None
                if stop is None or stop <= 0 or (required_signal == "BUY" and stop >= price):
                    rejections.append({"symbol": symbol, "reason": "APPROVED_STOP_REQUIRED"})
                    continue
                if approval.asset_type == "option":
                    contracts = [
                        item
                        for item in metadata.values()
                        if item.asset_type == "option" and item.underlying == symbol
                    ]
                    quotes = {
                        event.symbol: event
                        for event in reader.market()
                        if event.event_type == "quote"
                        and event.effective_event_time <= session.close
                    }
                    found, issues = select_alternatives(
                        approval,
                        contracts,
                        quotes,
                        price,
                        stop,
                        session.close,
                        quote_max_age=300,
                        max_spread=service.config.policy.max_quote_spread_pct,
                        available_at=now,
                    )
                    if not found:
                        rejections.append(
                            {"symbol": symbol, "reason": "NO_OPTION_ALTERNATIVE", "details": issues}
                        )
                    for option in found:
                        option = option.model_copy(
                            update={
                                "evidence_ids": tuple(
                                    dict.fromkeys(
                                        option.evidence_ids
                                        + tuple(e.id for e in reader.market(symbol))
                                    )
                                )
                            }
                        )
                        alternatives.append(option)
                        rationales[option.id] = signal.reason
                else:
                    capacity = Decimal(str(bars.iloc[-1]["volume"])) * Decimal(".01")
                    instrument = metadata.get(symbol)
                    if not instrument or capacity <= 0:
                        rejections.append(
                            {"symbol": symbol, "reason": "MISSING_INSTRUMENT_OR_CAPACITY"}
                        )
                        continue
                    instrument = instrument.model_copy(
                        update={"liquidity_capacity": min(instrument.liquidity_capacity, capacity)}
                    )
                    alternative = Alternative(
                        created_at=now,
                        strategy_id=approval.strategy_id,
                        strategy_version=approval.strategy_version,
                        instrument=instrument,
                        entry_price=price,
                        stop_price=stop,
                        expected_net_value=approval.validated_net_expectancy,
                        evidence_ids=tuple(e.id for e in reader.market(symbol)),
                    )
                    alternatives.append(alternative)
                    rationales[alternative.id] = signal.reason
            report = service.suitability.assess(
                account, alternatives, service.config.data_profile, approvals
            )
            service.store.put("suitability_reports", report, service.scope)
            suitability_ids.append(report.id)
            feasible = {item.alternative_id for item in report.alternatives if item.feasible}
            for alternative in alternatives:
                if alternative.id not in feasible:
                    continue
                candidate = Candidate(
                    created_at=now,
                    symbol=symbol,
                    snapshot_id=snapshot.id,
                    account_profile_id=account.id,
                    suitability_report_id=report.id,
                    alternative=alternative,
                    rationale=rationales[alternative.id],
                )
                service.store.put("candidates", candidate, service.scope)
                candidates.append(candidate.id)
        if not suitability_ids:
            report = service.suitability.assess(account, [], service.config.data_profile, approvals)
            service.store.put("suitability_reports", report, service.scope)
            suitability_ids.append(report.id)
        review = {
            "id": uuid4().hex,
            "session_id": session.id,
            "snapshot_id": snapshot.id,
            "account_profile_id": account.id,
            "data_profile": str(service.config.data_profile),
            "signal_as_of": snapshot.data_context.signal_as_of.isoformat(),
            "created_at": now.isoformat(),
            "candidates": candidates,
            "rejections": rejections,
            "suitability_report_ids": suitability_ids,
            "outcome": "CANDIDATES" if candidates else "NO_TRADE",
            "reconciliation_time": service.last_reconciliation,
            "cost": "0",
        }
        service.store.put("session_reviews", review, service.scope)
        return [review["id"], snapshot.id, *candidates, *suitability_ids]


class CloseGauss:
    name = "CloseGauss"

    def run(self, service, session, now):
        from .models import AccountProfile
        from .research import ResearchJobRunner

        grouped = {}
        for row in service.store.list("candidates", service.scope):
            candidate = Candidate.model_validate(row)
            snapshot = service.store.get("snapshots", candidate.snapshot_id)
            if snapshot["target_session_id"] == session.id:
                grouped.setdefault((candidate.symbol, candidate.snapshot_id), []).append(candidate)
        results = []
        for (symbol, snapshot_id), candidates in list(grouped.items())[
            : service.config.active_candidate_limit
        ]:
            reader = SnapshotDataReader(service.store, service.store.get("snapshots", snapshot_id))
            account = AccountProfile.model_validate(
                service.store.get("account_profiles", reader.snapshot.account_profile_id)
            )
            approvals = [
                StrategyApproval.model_validate(service.store.get("strategy_approvals", key))
                for key in reader.snapshot.strategy_approval_ids
            ]
            alternatives = [candidate.alternative for candidate in candidates]
            report = service.suitability.assess(
                account, alternatives, service.config.data_profile, approvals
            )
            service.store.put("suitability_reports", report, service.scope)
            bars = reader.bars(symbol)
            from src.agent.multi_agent.orchestrator import MultiAgentOrchestrator
            from src.strategy.base import MarketDataContext

            context = MarketDataContext(
                current_date=reader.snapshot.data_context.signal_as_of,
                current_prices={symbol: float(bars.iloc[-1]["close"])},
                historical_bars={symbol: bars},
                portfolio_value=float(account.equity),
                available_cash=float(account.cash),
                current_positions={p.symbol: str(p.quantity) for p in account.positions},
            )
            analysis = asyncio.run(
                MultiAgentOrchestrator(
                    llm_provider="openai", mode="fast", snapshot_reader=reader
                ).evaluate_symbol(symbol, context)
            )
            runner = ResearchJobRunner(
                service.store, service.config, service.scope, clock=lambda: now
            )
            model = getattr(service.config, "research_model", None)
            pricing_id = getattr(service.config, "research_pricing_id", None)
            pricing = service.store.get("model_pricing", pricing_id) if pricing_id else None
            annotation = runner.run(
                reader.snapshot,
                account,
                self.name,
                session.id,
                model=model,
                pricing=pricing,
                max_output_tokens=getattr(service.config, "research_max_output_tokens", 1024),
                max_retries=getattr(service.config, "research_max_retries", 0),
                budget_account=service.account_profile,
            )
            counter = [n.id for n in reader.news(symbol) if n.blocking]
            analysis_veto = annotation.annotation is not None and annotation.annotation.outcome in {
                "COUNTER_EVIDENCE",
                "EXPERIMENT",
            }
            research = {
                "id": uuid4().hex,
                "session_id": session.id,
                "candidate_ids": [c.id for c in candidates],
                "snapshot_id": reader.snapshot.id,
                "account_profile_id": account.id,
                "suitability_report_id": report.id,
                "data_profile": str(service.config.data_profile),
                "created_at": now.isoformat(),
                "analysis": {
                    key: asdict(value)
                    if hasattr(value, "__dataclass_fields__")
                    else [asdict(v) if hasattr(v, "__dataclass_fields__") else v for v in value]
                    if isinstance(value, list)
                    else value
                    for key, value in analysis.items()
                },
                "annotation": annotation.model_dump(mode="json"),
                "counter_evidence": counter,
                "outcome": "PLAN"
                if report.selected_alternative_id and not counter and not analysis_veto
                else "NO_TRADE",
                "cost": str(annotation.cost_usd),
            }
            service.store.put("research_reports", research, service.scope)
            results.extend([research["id"], report.id])
            if research["outcome"] != "PLAN":
                continue
            candidate = next(
                c for c in candidates if c.alternative.id == report.selected_alternative_id
            )
            alternative = candidate.alternative
            approval = next(
                a
                for a in approvals
                if (a.strategy_id, a.strategy_version)
                == (alternative.strategy_id, alternative.strategy_version)
            )
            target = service.calendar.current_or_next(session.close + timedelta(seconds=1))
            schedule = target.schedule(service.config.policy)
            if schedule["entry_start"] >= min(schedule["entry_cutoff"], schedule["closing_start"]):
                continue
            price = alternative.entry_price
            tick = max(item.tick_size for item in alternative.instruments)
            bound = (price * Decimal("1.001") / tick).to_integral_value(rounding=ROUND_DOWN) * tick
            trigger = alternative.underlying_entry_price or price
            stop = alternative.underlying_stop_price or alternative.stop_price
            bearish = bool(approval.option_selector and "put" in approval.option_selector.structure)
            plan = SessionPlan(
                created_at=now,
                target_session_id=target.id,
                snapshot_id=reader.snapshot.id,
                candidate_id=candidate.id,
                account_id=account.account_id,
                environment=account.environment,
                account_profile_id=account.id,
                suitability_report_id=report.id,
                data_profile=service.config.data_profile,
                data_policy_version=service.config.data_policy_version,
                signal_as_of=reader.snapshot.data_context.signal_as_of,
                strategy_id=approval.strategy_id,
                strategy_version=approval.strategy_version,
                approval_id=approval.id,
                alternative=alternative,
                entry_rules=(
                    Rule(
                        rule_type="completed_bar_condition",
                        operator="below" if bearish else "above",
                        value=trigger,
                        timeframe=service.config.signal_timeframe,
                    ),
                    Rule(rule_type="no_blocking_event"),
                    Rule(rule_type="time_in_window"),
                ),
                invalidation_rules=(
                    Rule(
                        rule_type="completed_bar_condition",
                        operator="above" if bearish else "below",
                        value=stop,
                        timeframe=service.config.signal_timeframe,
                    ),
                ),
                max_buy_price=bound,
                valid_from=schedule["entry_start"],
                entry_expires_at=min(schedule["entry_cutoff"], schedule["closing_start"]),
                close_at=schedule["closing_start"],
                risk_policy_id=service.config.policy.id,
                exit_policy_id=f"{approval.strategy_id}:{approval.strategy_version}:intraday-v1",
                execution_eligible=not account.hypothetical,
            )
            with service.store.transaction():
                service.store.put("plans", plan, service.scope)
                service.store.transition_plan(plan.id, "RESEARCH_COMPLETE", scope=service.scope)
                service.store.transition_plan(plan.id, "PENDING_VALIDATION", scope=service.scope)
            results.append(plan.id)
        if not results:
            suitability = service.suitability.assess(
                service.account_profile, [], service.config.data_profile, service.approvals(now)
            )
            service.store.put("suitability_reports", suitability, service.scope)
            report = {
                "id": uuid4().hex,
                "session_id": session.id,
                "outcome": "NO_TRADE",
                "reason": "NO_FEASIBLE_CANDIDATES",
                "account_profile_id": service.account_profile.id,
                "suitability_report_id": suitability.id,
                "data_profile": str(service.config.data_profile),
                "created_at": now.isoformat(),
                "cost": "0",
            }
            service.store.put("research_reports", report, service.scope)
            results.extend([report["id"], suitability.id])
        return results


class PreGauss:
    name = "PreGauss"

    def run(self, service, session, now):
        results = []
        approvals = {a.id: a for a in service.approvals(now)}
        for row in service.store.list("plans", service.scope):
            plan = SessionPlan.model_validate(row)
            if plan.target_session_id != session.id:
                continue
            state = service.store.projection("plan_state", plan.id)
            if state and state["state"] in {
                "REJECTED",
                "INVALIDATED",
                "SUPERSEDED",
                "ENTRY_EXPIRED",
            }:
                continue
            approval = approvals.get(plan.approval_id)
            report = service.suitability.assess(
                service.account_profile,
                [plan.alternative],
                service.config.data_profile,
                [approval] if approval else [],
                reserved=service.reserved_capital(),
            )
            service.store.put("suitability_reports", report, service.scope)
            reasons = list(report.alternatives[0].reasons)
            outcome = "ELIGIBLE"
            if now >= plan.entry_expires_at:
                outcome = "ENTRY_EXPIRED"
                reasons.append("ENTRY_WINDOW_EXPIRED")
            elif service.evidence.blockers(plan.alternative.signal_symbol, now):
                outcome = "INVALIDATED"
                reasons.append("CURRENT_NEWS_SAFETY_VETO")
            elif (
                plan.data_profile != service.config.data_profile
                or plan.data_policy_version != service.config.data_policy_version
            ):
                outcome = "REVIEW_REQUIRED"
                reasons.append("DATA_POLICY_CHANGED")
            elif plan.account_id != service.account_profile.account_id:
                outcome = "REJECTED"
                reasons.append("ACCOUNT_SCOPE_MISMATCH")
            else:
                if not service.price_ready:
                    reasons.append("MARKET_COVERAGE_UNAVAILABLE")
                if (
                    now >= session.open
                    and not service.evidence.context(
                        now, (plan.alternative.signal_symbol,)
                    ).complete
                ):
                    reasons.append("PLAN_MARKET_COVERAGE_UNAVAILABLE")
                if approval and approval.requires_news and not service.news_ready:
                    reasons.append("NEWS_COVERAGE_UNAVAILABLE")
                if (
                    approval
                    and approval.requires_event_calendar
                    and not service.event_calendar_ready
                ):
                    reasons.append("EVENT_CALENDAR_UNAVAILABLE")
                # A pre-open validation may defer current quotes until the permitted session.
                if approval and approval.requires_current_quote and now >= session.open:
                    max_age = (
                        service.config.option_quote_max_age_seconds
                        if approval.asset_type == "option"
                        else service.config.stock_quote_max_age_seconds
                    )
                    quotes = [
                        service.evidence.current_quote(
                            item.symbol, now, approval.required_feed, max_age
                        )
                        for item in plan.alternative.instruments
                    ]
                    if approval.asset_type == "option":
                        issue = quote_checks(
                            quotes,
                            now,
                            max_age=max_age,
                            max_skew=approval.option_selector.maximum_quote_skew_seconds
                            if approval.option_selector
                            else Decimal("1"),
                            max_spread=service.config.policy.max_quote_spread_pct,
                        )
                        if issue:
                            reasons.append(issue)
                    elif any(q is None for q in quotes):
                        reasons.append("CURRENT_GENUINE_QUOTE_REQUIRED")
                observations = service.evidence.market(now, plan.alternative.signal_symbol)
                if evaluate_rules(
                    plan.invalidation_rules,
                    observations,
                    now,
                    sessions=tuple(service.calendar.sessions.values()),
                ):
                    outcome = "INVALIDATED"
                    reasons.append("PRICE_INVALIDATION")
                if reasons:
                    outcome = "DEFERRED" if outcome != "INVALIDATED" else outcome
            validation = ValidationRecord(
                created_at=now,
                plan_id=plan.id,
                plan_version=plan.version,
                account_profile_id=service.account_profile.id,
                suitability_report_id=report.id,
                signal_as_of=service.evidence.clock.signal_as_of(now),
                event_watermark=service.store.watermark(service.scope),
                outcome=outcome,
                reasons=tuple(reasons),
            )
            with service.store.transaction():
                service.store.put("validations", validation, service.scope)
                service.store.transition_plan(plan.id, outcome, reasons, scope=service.scope)
            results.append(validation.id)
        return results


class LiveGauss:
    name = "LiveGauss"

    def run(self, service, session, now):
        results = []
        for row in service.store.list("plans", service.scope):
            plan = SessionPlan.model_validate(row)
            if plan.target_session_id != session.id:
                continue
            state = service.store.projection("plan_state", plan.id)
            if not state or state["state"] != "ELIGIBLE":
                continue
            if now >= plan.entry_expires_at:
                service.store.transition_plan(
                    plan.id, "ENTRY_EXPIRED", ("ENTRY_WINDOW_EXPIRED",), scope=service.scope
                )
                continue
            if service.store.projection(
                "orders", digest([plan.id, plan.version, "initial-entry"])[:32]
            ):
                continue
            observations = service.evidence.market(now, plan.alternative.signal_symbol)
            if evaluate_rules(
                plan.invalidation_rules,
                observations,
                now,
                sessions=tuple(service.calendar.sessions.values()),
            ):
                service.store.transition_plan(
                    plan.id, "INVALIDATED", ("PRICE_INVALIDATION",), scope=service.scope
                )
                continue
            if not evaluate_rules(
                plan.entry_rules,
                observations,
                now,
                blocking=bool(service.evidence.blockers(plan.alternative.signal_symbol, now)),
                valid_from=plan.valid_from,
                expires_at=plan.entry_expires_at,
                sessions=tuple(service.calendar.sessions.values()),
            ):
                continue
            if not service.entries_ready:
                service.store.put(
                    "decisions",
                    {
                        "id": uuid4().hex,
                        "plan_id": plan.id,
                        "outcome": "NO_TRADE",
                        "reason": "RUNTIME_NOT_READY",
                        "created_at": now.isoformat(),
                    },
                    service.scope,
                )
                continue
            price = plan.max_buy_price
            intent = OrderIntent(
                created_at=now,
                id=digest([plan.id, plan.version, "initial-entry"])[:32],
                account_id=service.config.account_id,
                environment=service.config.environment,
                purpose="open",
                plan_id=plan.id,
                plan_version=plan.version,
                legs=opening_legs(plan.alternative),
                limit_price=price,
                maximum_buy_price=plan.max_buy_price,
                expires_at=min(plan.entry_expires_at, now + timedelta(seconds=10)),
                reason="Approved completed-observation trigger",
            )
            decision = service.risk.approve(
                intent,
                service.account_profile,
                now,
                entry_ready=service.entries_ready,
                news_ready=service.news_ready,
                event_calendar_ready=service.event_calendar_ready,
            )
            if decision.approved:
                result = service.gateway.submit(intent, decision, now)
                results.append(result["id"])
            else:
                results.append(decision.id)
        return results
