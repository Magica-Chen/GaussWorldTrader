"""Single deterministic authority for reservations, submission and actual exposure."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

from .models import (
    AccountProfile,
    AccountOperationalState,
    ExecutionMode,
    DataProfile,
    OrderIntent,
    PositionGroup,
    RiskDecision,
    SessionPlan,
    StrategyApproval,
    TradingSession,
)
from .store import TERMINAL_ORDERS, digest
from .suitability import AccountSuitabilityService
from .evidence import evaluate_rules
from .options import opening_legs, quote_checks, structure_error


def validate_structure(legs, now, policy, purpose="open"):
    if any(not leg.instrument.tradable for leg in legs):
        return "INSTRUMENT_NOT_TRADABLE"
    if any(leg.instrument.asset_type == "option" for leg in legs):
        if not policy.options_enabled and purpose == "open":
            return "OPTIONS_LIFECYCLE_DISABLED"
        for leg in legs:
            item = leg.instrument
            if (
                item.asset_type != "option"
                or not item.standard_contract
                or item.multiplier != 100
                or not item.contract_id
                or not item.expiry
                or item.expiry < now.date().isoformat()
                or (purpose == "open" and item.expiry == now.date().isoformat())
            ):
                return "OPTION_CONTRACT_OR_EXPIRY_INVALID"
            if item.quantity_increment != 1:
                return "WHOLE_CONTRACT_REQUIRED"
        if len(legs) > 1:
            if not policy.spreads_enabled and purpose == "open":
                return "SPREAD_LIFECYCLE_DISABLED"
            a, b = [leg.instrument for leg in legs]
            if (
                a.symbol == b.symbol
                or a.underlying != b.underlying
                or a.expiry != b.expiry
                or a.option_type != b.option_type
                or a.strike == b.strike
                or {leg.side for leg in legs} != {"buy", "sell"}
                or any(leg.ratio != 1 for leg in legs)
            ):
                return "UNSUPPORTED_OPTION_STRUCTURE"
            bought = next(leg.instrument for leg in legs if leg.side == "buy")
            sold = next(leg.instrument for leg in legs if leg.side == "sell")
            if legs[0].position_intent.endswith("_open") and not (
                (bought.option_type == "call" and bought.strike < sold.strike)
                or (bought.option_type == "put" and bought.strike > sold.strike)
            ):
                return "ONLY_DEBIT_VERTICAL_ENTRIES_SUPPORTED"
        elif legs[0].position_intent == "sell_to_open":
            return "UNCOVERED_SHORT_PROHIBITED"
    elif len(legs) > 1:
        return "MULTILEG_STOCK_UNSUPPORTED"
    elif legs[0].position_intent == "sell_to_open":
        return "SHORT_ENTRY_NOT_APPROVED"
    return None


class RiskGate:
    def __init__(self, store, config, scope, evidence, calendar):
        self.store, self.config, self.scope = store, config, scope
        self.evidence, self.calendar = evidence, calendar
        self.suitability = AccountSuitabilityService(config.policy)

    def approve(self, intent, account, now, *, entry_ready, news_ready, event_calendar_ready):
        intent = OrderIntent.model_validate(intent)
        if intent.purpose == "close" and (
            isinstance(account, AccountOperationalState)
            or isinstance(account, dict)
            and "equity" not in account
        ):
            account = AccountOperationalState.model_validate(account)
        else:
            account = AccountProfile.model_validate(account)
        policy = self.config.policy
        with self.store.transaction():
            previous = self.store.get("risk_decisions", intent.id)
            if previous:
                return RiskDecision.model_validate(previous)
            reasons = []
            quantity = Decimal("0")
            capital = Decimal("0")
            loss = Decimal("0")
            suitability_id = None
            if (
                intent.account_id != self.config.account_id
                or account.account_id != intent.account_id
                or account.environment != intent.environment
                or intent.environment != self.config.environment
                or getattr(account, "hypothetical", False)
            ):
                reasons.append("ACCOUNT_SCOPE_MISMATCH")
            if (
                not 0
                <= (now - account.observed_at).total_seconds()
                <= policy.account_max_age_seconds
            ):
                reasons.append("ACCOUNT_STALE")
            if intent.expires_at <= now:
                reasons.append("INTENT_EXPIRED")
            structure_issue = validate_structure(intent.legs, now, policy, intent.purpose)
            if structure_issue:
                reasons.append(structure_issue)
            tick = intent.legs[0].instrument.tick_size
            if intent.limit_price % tick:
                reasons.append("INVALID_PRICE_TICK")
            if (
                intent.maximum_buy_price is not None
                and intent.limit_price > intent.maximum_buy_price
            ):
                reasons.append("BUY_PRICE_BOUND_EXCEEDED")
            if (
                intent.minimum_sell_price is not None
                and intent.limit_price < intent.minimum_sell_price
            ):
                reasons.append("SELL_PRICE_BOUND_EXCEEDED")
            if (
                intent.purpose == "open"
                and len(intent.legs) > 1
                and getattr(intent, "limit_effect", "debit") != "debit"
            ):
                reasons.append("ONLY_DEBIT_VERTICAL_ENTRIES_SUPPORTED")
            session_id = "exit"
            if intent.purpose == "close":
                try:
                    if not self.calendar.is_open(now):
                        reasons.append("EXIT_SESSION_CLOSED")
                except RuntimeError:
                    reasons.append("EXIT_CALENDAR_UNAVAILABLE")
                group = self.store.projection("position_groups", intent.group_id)
                if (
                    not group
                    or group.get("account_id") != intent.account_id
                    or group.get("environment") != intent.environment
                ):
                    reasons.append("MANAGED_POSITION_REQUIRED")
                bounds = []
                for leg in intent.legs:
                    held = next(
                        (p for p in account.positions if p.symbol == leg.instrument.symbol), None
                    )
                    if (
                        held is None
                        or held.quantity == 0
                        or (leg.side == "sell" and held.quantity <= 0)
                        or (leg.side == "buy" and held.quantity >= 0)
                    ):
                        reasons.append("CLOSE_QUANTITY_OR_SIDE_INVALID")
                    else:
                        if (
                            held.account_id != account.account_id
                            or held.environment != account.environment
                        ):
                            reasons.append("CLOSE_HOLDING_SCOPE_MISMATCH")
                        if (
                            not 0
                            <= (now - held.reconciled_at).total_seconds()
                            <= policy.account_max_age_seconds
                        ):
                            reasons.append("CLOSE_HOLDING_RECONCILIATION_STALE")
                        bounds.append(abs(held.quantity) / leg.ratio)
                pending = [
                    r
                    for r in self.store.projected("orders", self.scope)
                    if r.get("group_id") == intent.group_id
                    and r.get("purpose") == "close"
                    and r.get("state") not in TERMINAL_ORDERS
                ]
                if pending:
                    reasons.append("EXIT_ALREADY_PENDING")
                if bounds:
                    quantity = min(bounds)
                    if intent.requested_quantity is not None:
                        if intent.requested_quantity > quantity:
                            reasons.append("CLOSE_EXCEEDS_CONFIRMED_HOLDING")
                        quantity = min(quantity, intent.requested_quantity)
                quotes = []
                for leg in intent.legs:
                    feed = "opra" if leg.instrument.asset_type == "option" else "sip"
                    max_age = (
                        self.config.option_quote_max_age_seconds
                        if leg.instrument.asset_type == "option"
                        else self.config.stock_quote_max_age_seconds
                    )
                    quotes.append(
                        self.evidence.current_quote(leg.instrument.symbol, now, feed, max_age)
                    )
                if any(quote is None for quote in quotes):
                    reasons.append("CLOSE_CURRENT_GENUINE_QUOTES_REQUIRED")
                elif len(intent.legs) > 1:
                    quote_issue = quote_checks(
                        quotes,
                        now,
                        max_age=self.config.option_quote_max_age_seconds,
                        max_skew=Decimal("1"),
                        max_spread=policy.max_quote_spread_pct,
                    )
                    if quote_issue:
                        reasons.append(quote_issue)
                    net = sum(
                        (
                            quote.bid if leg.side == "sell" else -quote.ask
                            for leg, quote in zip(intent.legs, quotes)
                        ),
                        Decimal(0),
                    )
                    expected = "credit" if net > 0 else "debit"
                    if net == 0 or getattr(intent, "limit_effect", "debit") != expected:
                        reasons.append("CLOSE_DEBIT_CREDIT_EFFECT_MISMATCH")
            else:
                if not entry_ready:
                    reasons.append("RUNTIME_NOT_READY")
                if account.trading_blocked:
                    reasons.append("ACCOUNT_RESTRICTED")
                if (
                    self.config.execution_mode == ExecutionMode.LIVE
                    and self.config.data_profile == DataProfile.FREE_DELAYED
                ):
                    reasons.append("FREE_DELAYED_LIVE_ENTRY_DISABLED")
                plan_row = self.store.get("plans", intent.plan_id)
                if not plan_row:
                    reasons.append("PLAN_UNKNOWN")
                else:
                    plan = SessionPlan.model_validate(plan_row)
                    state = self.store.projection("plan_state", plan.id)
                    if state is None or state["state"] != "ELIGIBLE":
                        reasons.append("PLAN_NOT_ELIGIBLE")
                    if (
                        not plan.execution_eligible
                        or plan.environment != intent.environment
                        or plan.account_id != account.account_id
                    ):
                        reasons.append("PLAN_SCOPE_MISMATCH")
                    if intent.plan_version != plan.version:
                        reasons.append("PLAN_VERSION_MISMATCH")
                    if (
                        plan.data_profile != self.config.data_profile
                        or plan.data_policy_version != self.config.data_policy_version
                    ):
                        reasons.append("DATA_POLICY_MISMATCH")
                    if plan.risk_policy_id != policy.id:
                        reasons.append("RISK_POLICY_MISMATCH")
                    expected_legs = opening_legs(plan.alternative)
                    if len(intent.legs) != len(expected_legs) or any(
                        actual.instrument != expected.instrument
                        or actual.side != expected.side
                        or actual.ratio != expected.ratio
                        or actual.position_intent != expected.position_intent
                        for actual, expected in zip(intent.legs, expected_legs)
                    ):
                        # New selectors/structures require immutable replacement plan with concrete instrument.
                        reasons.append("PLAN_INSTRUMENT_MISMATCH")
                    if intent.limit_price > plan.max_buy_price:
                        reasons.append("PLAN_PRICE_BOUND_EXCEEDED")
                    session_id = plan.target_session_id
                    session = self.calendar.get(session_id)
                    schedule = session.schedule(policy)
                    if not (
                        max(plan.valid_from, schedule["entry_start"])
                        <= now
                        < min(plan.entry_expires_at, schedule["entry_cutoff"])
                    ):
                        reasons.append("ENTRY_WINDOW_CLOSED")
                    approval_row = self.store.get("strategy_approvals", plan.approval_id)
                    approval = (
                        StrategyApproval.model_validate(approval_row) if approval_row else None
                    )
                    if (
                        approval is None
                        or approval.account_id != account.account_id
                        or approval.environment != account.environment
                        or approval.strategy_version != plan.strategy_version
                        or approval.strategy_id != plan.strategy_id
                        or approval.expiry <= now
                        or self.config.execution_mode not in approval.execution_modes
                        or self.config.data_profile not in approval.profiles
                    ):
                        reasons.append("STRATEGY_APPROVAL_INVALID")
                    if plan.strategy_id not in self.config.strategy_allowlist:
                        reasons.append("STRATEGY_NOT_IN_MANDATE")
                    if approval:
                        if approval.requires_news and not news_ready:
                            reasons.append("NEWS_COVERAGE_UNAVAILABLE")
                        if approval.requires_event_calendar and not event_calendar_ready:
                            reasons.append("EVENT_CALENDAR_UNAVAILABLE")
                        observations = self.evidence.market(now, plan.alternative.signal_symbol)
                        if (
                            not observations
                            or (now - observations[-1].effective_event_time).total_seconds()
                            > approval.maximum_signal_latency_seconds
                        ):
                            reasons.append("SIGNAL_STALE_OR_MISSING")
                        if self.evidence.blockers(plan.alternative.signal_symbol, now):
                            reasons.append("BLOCKING_NEWS_EVENT")
                        if not evaluate_rules(
                            plan.entry_rules,
                            observations,
                            now,
                            blocking=bool(
                                self.evidence.blockers(plan.alternative.signal_symbol, now)
                            ),
                            valid_from=plan.valid_from,
                            expires_at=plan.entry_expires_at,
                            sessions=tuple(self.calendar.sessions.values()),
                        ):
                            reasons.append("TRIGGER_NOT_SATISFIED")
                        if approval.requires_current_quote:
                            max_age = (
                                self.config.option_quote_max_age_seconds
                                if plan.alternative.instrument.asset_type == "option"
                                else self.config.stock_quote_max_age_seconds
                            )
                            quotes = [
                                self.evidence.current_quote(
                                    instrument.symbol, now, approval.required_feed, max_age
                                )
                                for instrument in plan.alternative.instruments
                            ]
                            if plan.alternative.instrument.asset_type == "option":
                                error = quote_checks(
                                    quotes,
                                    now,
                                    max_age=max_age,
                                    max_skew=approval.option_selector.maximum_quote_skew_seconds
                                    if approval.option_selector
                                    else Decimal("1"),
                                    max_spread=policy.max_quote_spread_pct,
                                )
                                if error:
                                    reasons.append(error)
                                structure = structure_error(
                                    plan.alternative.model_copy(
                                        update={"entry_price": intent.limit_price}
                                    )
                                )
                                if structure:
                                    reasons.append(structure)
                                if (
                                    len(quotes) == 2
                                    and all(quotes)
                                    and intent.limit_price < quotes[0].bid - quotes[1].ask
                                ):
                                    reasons.append("FINAL_QUOTE_PRICE_INVALID")
                            elif not quotes[0]:
                                reasons.append("CURRENT_GENUINE_QUOTE_REQUIRED")
                            elif (quotes[0].ask - quotes[0].bid) / quotes[
                                0
                            ].ask > policy.max_quote_spread_pct:
                                reasons.append("QUOTE_SPREAD_TOO_WIDE")
                            elif (
                                intent.limit_price < quotes[0].bid
                                or intent.limit_price > plan.max_buy_price
                            ):
                                reasons.append("FINAL_QUOTE_PRICE_INVALID")
                    rows = self.store.db.execute(
                        "SELECT * FROM reservations WHERE scope=? AND state!=?",
                        (self.scope, "RELEASED"),
                    ).fetchall()
                    reserved = sum((Decimal(r["capital"]) for r in rows), Decimal("0"))
                    # Conservative reservation policy deliberately retains pending capital even if broker BP reflects it.
                    pending = sum(1 for r in rows if r["session_id"] == session_id)
                    attempts = sum(
                        1
                        for o in self.store.projected("orders", self.scope)
                        if o.get("session_id") == session_id
                        and o.get("purpose") == "open"
                        and o["id"] == o["intent_id"]
                    )
                    groups = [
                        g
                        for g in self.store.projected("position_groups", self.scope)
                        if g["state"] != "CLOSED"
                    ]
                    if pending >= policy.max_new_entry_groups_per_session:
                        reasons.append("ENTRY_QUOTA_EXHAUSTED")
                    if (
                        len(groups) + sum(1 for r in rows if not r["first_fill"])
                        >= policy.max_open_position_groups
                    ):
                        reasons.append("POSITION_GROUP_LIMIT")
                    if attempts >= policy.max_entry_order_attempts_per_session:
                        reasons.append("ATTEMPT_LIMIT")
                    if any(
                        p.symbol in {i.symbol for i in plan.alternative.instruments}
                        and p.quantity != 0
                        for p in account.positions
                    ) or any(r["symbol"] == plan.alternative.signal_symbol for r in rows):
                        reasons.append("DUPLICATE_EXPOSURE")
                    if any(
                        o.get("plan_id") == plan.id
                        for o in self.store.projected("orders", self.scope)
                    ):
                        reasons.append("PLAN_ALREADY_ATTEMPTED")
                    if account.daily_pnl is None:
                        reasons.append("DAILY_PNL_UNRECONCILED")
                    elif account.daily_pnl <= -account.equity * policy.daily_loss_pause_pct:
                        reasons.append("DAILY_LOSS_PAUSE")
                    actual = plan.alternative.model_copy(update={"entry_price": intent.limit_price})
                    report = self.suitability.assess(
                        account,
                        [actual],
                        self.config.data_profile,
                        [approval] if approval else [],
                        reserved=reserved,
                    )
                    self.store.put("suitability_reports", report, self.scope)
                    suitability_id = report.id
                    sizing = report.alternatives[0]
                    reasons.extend(sizing.reasons)
                    quantity = sizing.quantity
                    if intent.requested_quantity is not None:
                        if intent.requested_quantity > quantity:
                            reasons.append("REQUEST_EXCEEDS_SAFE_SIZE")
                        quantity = min(intent.requested_quantity, quantity)
                    capital = quantity * intent.limit_price * intent.legs[0].instrument.multiplier
                    loss = (
                        sizing.planned_loss
                        if sizing.quantity == quantity
                        else sizing.planned_loss * quantity / sizing.quantity
                        if sizing.quantity
                        else Decimal("0")
                    )
            if quantity <= 0:
                reasons.append("NO_FEASIBLE_SIZE")
            if any(quantity % leg.instrument.quantity_increment for leg in intent.legs):
                reasons.append("INVALID_QUANTITY_INCREMENT")
            decision = RiskDecision(
                id=intent.id,
                intent_id=intent.id,
                approved=not reasons,
                policy_id=policy.id,
                account_profile_id=account.id,
                suitability_report_id=suitability_id,
                data_policy_version=self.config.data_policy_version,
                event_watermark=self.store.watermark(self.scope),
                approved_quantity=quantity if not reasons else Decimal("0"),
                capital_reserved=capital if not reasons else Decimal("0"),
                loss_reserved=loss if not reasons else Decimal("0"),
                reasons=tuple(dict.fromkeys(reasons)),
                expires_at=min(intent.expires_at, now + timedelta(seconds=10)),
            )
            self.store.put("order_intents", intent, self.scope)
            self.store.put("risk_decisions", decision, self.scope)
            if decision.approved:
                if intent.purpose == "open":
                    self.store.db.execute(
                        "INSERT INTO reservations VALUES(?,?,?,?,?,?,?,0)",
                        (
                            intent.id,
                            self.scope,
                            session_id,
                            str(capital),
                            str(loss),
                            plan.alternative.signal_symbol,
                            "RESERVED",
                        ),
                    )
                self.store.emit("submit_intent", {"intent_id": intent.id}, self.scope, intent.id)
            return decision


class ExecutionGateway:
    def __init__(self, store, config, scope, broker, evidence):
        self.store, self.config, self.scope, self.broker, self.evidence = (
            store,
            config,
            scope,
            broker,
            evidence,
        )

    def _armed_for_plan(self, plan):
        arm = self.store.projection("arming", self.scope)
        if (
            not self.config.live_trading_enabled
            or self.config.account_id not in self.config.account_id_allowlist
            or not arm
            or arm.get("risk_policy_id") != self.config.policy.id
            or arm.get("data_policy_version") != self.config.data_policy_version
            or arm.get("deployment_version") != self.config.deployment_version
        ):
            return False
        return any(
            item.get("approval_id") == plan.approval_id
            and item.get("strategy_id") == plan.strategy_id
            and item.get("strategy_version") == plan.strategy_version
            and self.config.data_profile in item.get("data_profiles", ())
            for item in arm.get("approved_strategy_versions", ())
        )

    def submit(self, intent, decision, now):
        with self.store.transaction():
            recorded_intent = self.store.get("order_intents", intent.id)
            recorded_decision = self.store.get("risk_decisions", intent.id)
            if recorded_intent is None or recorded_decision is None:
                raise ValueError("durable risk-approved intent required")
            if (
                OrderIntent.model_validate(recorded_intent) != intent
                or RiskDecision.model_validate(recorded_decision) != decision
            ):
                raise ValueError("intent or risk approval differs from durable authority")
            if (
                intent.account_id != self.config.account_id
                or intent.environment != self.config.environment
            ):
                raise ValueError("gateway account/environment mismatch")
            if self.config.execution_mode == ExecutionMode.PAPER and intent.environment != "paper":
                raise ValueError("paper gateway cannot reach live account")
            existing = self.store.projection("orders", intent.id)
            if existing:
                return existing
            if not decision.approved or decision.intent_id != intent.id:
                raise ValueError("risk-approved intent required")
            state = (
                "SHADOW"
                if self.config.execution_mode in {ExecutionMode.SHADOW, ExecutionMode.REPLAY}
                else "SUBMITTING"
            )
            row = {
                "id": intent.id,
                "intent_id": intent.id,
                "client_order_id": "gauss-" + intent.id[:40],
                "account_id": intent.account_id,
                "environment": intent.environment,
                "purpose": intent.purpose,
                "plan_id": intent.plan_id,
                "group_id": intent.group_id or "group-" + intent.id,
                "state": state,
                "filled_quantity": "0",
                "quantity": str(decision.approved_quantity),
                "created_at": now.isoformat(),
                "decision_time": now.isoformat(),
                "session_id": (self.store.get("plans", intent.plan_id) or {}).get(
                    "target_session_id"
                ),
                "signal_as_of": (self.store.get("plans", intent.plan_id) or {}).get("signal_as_of"),
            }
            reasons = []
            if now >= decision.expires_at:
                reasons.append("RISK_APPROVAL_EXPIRED")
            if intent.purpose == "open":
                projection = self.store.projection("plan_state", intent.plan_id)
                if not projection or projection["state"] != "ELIGIBLE":
                    reasons.append("PLAN_NO_LONGER_ELIGIBLE")
                if self.store.watermark(self.scope) != decision.event_watermark + 1:
                    # submit_intent itself advances watermark by one; any intervening event invalidates approval.
                    reasons.append("EVENT_WATERMARK_CHANGED")
                if self.config.execution_mode == ExecutionMode.LIVE:
                    armed_plan = SessionPlan.model_validate(self.store.get("plans", intent.plan_id))
                    if not self._armed_for_plan(armed_plan):
                        reasons.append("LIVE_NOT_ARMED")
                    if self.config.data_profile == DataProfile.FREE_DELAYED:
                        reasons.append("FREE_DELAYED_LIVE_ENTRY_DISABLED")
            if reasons:
                row.update(state="REJECTED", reasons=reasons)
            self.store.project("orders", intent.id, row, self.scope)
            if row["state"] in {"REJECTED", "SHADOW"}:
                self.store.db.execute(
                    "UPDATE reservations SET state=? WHERE intent_id=?", ("RELEASED", intent.id)
                )
                return row
        # Once SUBMITTING is durable, crashes/timeouts are unknown until broker reconciliation.
        try:
            broker_order = self.broker.submit(
                intent, decision.approved_quantity, row["client_order_id"]
            )
        except Exception as exc:
            row.update(state="UNKNOWN", error=type(exc).__name__)
            self.store.project("orders", intent.id, row, self.scope)
            self.store.put(
                "incidents",
                {
                    "id": uuid4().hex,
                    "reason": "SUBMISSION_UNKNOWN",
                    "intent_id": intent.id,
                    "created_at": now.isoformat(),
                },
                self.scope,
            )
            return row
        row.update(
            state="ACKNOWLEDGED",
            broker_id=str(broker_order["id"]),
            raw_status=str(broker_order.get("status", "")),
        )
        self.store.project("orders", intent.id, row, self.scope)
        return row

    def cancel_entries(self, now, plan_id=None):
        result = []
        for row in self.store.projected("orders", self.scope):
            if plan_id is not None and row.get("plan_id") != plan_id:
                continue
            if row["purpose"] != "open" or row["state"] in TERMINAL_ORDERS | {"SHADOW"}:
                continue
            if self.config.execution_mode in {ExecutionMode.REPLAY, ExecutionMode.SHADOW}:
                continue
            if not row.get("broker_id"):
                result.append({"intent_id": row["intent_id"], "state": "UNKNOWN_RECONCILE_FIRST"})
                continue
            row.pop("_revision", None)
            row.update(state="CANCEL_PENDING", cancel_requested_at=now.isoformat())
            with self.store.transaction():
                self.store.project("orders", row["id"], row, self.scope)
                self.store.put(
                    "order_controls",
                    {
                        "id": uuid4().hex,
                        "order_id": row["id"],
                        "intent_id": row["intent_id"],
                        "action": "CANCEL_REQUESTED",
                        "created_at": now.isoformat(),
                    },
                    self.scope,
                )
            try:
                self.broker.cancel(row["broker_id"])
            except Exception as exc:
                row["cancel_error"] = type(exc).__name__
                self.store.project("orders", row["id"], row, self.scope)
                self.store.put(
                    "incidents",
                    {
                        "id": uuid4().hex,
                        "reason": "CANCELLATION_UNCONFIRMED",
                        "order_id": row["id"],
                        "intent_id": row["intent_id"],
                        "created_at": now.isoformat(),
                    },
                    self.scope,
                )
            result.append(row)
        return result

    def replace(
        self, order_id, limit_price, now, *, account, entry_ready, news_ready, event_calendar_ready
    ):
        """Replace a reconciled single-leg limit order with the same total quantity.

        Every child remains attached to the original reservation and group. A pending
        or unknown replacement must be reconciled before any further attempt.
        """
        account = (
            AccountOperationalState.model_validate(account)
            if isinstance(account, AccountOperationalState)
            or isinstance(account, dict)
            and "equity" not in account
            else AccountProfile.model_validate(account)
        )
        price = Decimal(str(limit_price))
        policy = self.config.policy
        with self.store.transaction():
            parent = self.store.projection("orders", order_id)
            if (
                not parent
                or parent.get("account_id") != self.config.account_id
                or parent.get("environment") != self.config.environment
            ):
                raise ValueError("REPLACEMENT_ORDER_SCOPE_MISMATCH")
            root_id = parent["intent_id"]
            original = OrderIntent.model_validate(self.store.get("order_intents", root_id))
            original_decision = RiskDecision.model_validate(
                self.store.get("risk_decisions", root_id)
            )
            if not original_decision.approved:
                raise ValueError("DURABLE_RISK_APPROVAL_REQUIRED")
            if len(original.legs) != 1:
                raise ValueError("MULTILEG_REPLACEMENT_MECHANICS_NOT_APPROVED")
            family = [
                row
                for row in self.store.projected("orders", self.scope)
                if row["intent_id"] == root_id
            ]
            descendants = [row for row in family if row.get("replaces_order_id") == order_id]
            if descendants:
                # A repeated call returns the existing work, including an unknown result.
                previous = descendants[-1]
                if Decimal(previous["limit_price"]) == price:
                    return previous
                raise ValueError("REPLACEMENT_ALREADY_EXISTS_RECONCILE_FIRST")
            if parent["state"] not in {"OPEN", "PARTIALLY_FILLED", "SHADOW"}:
                raise ValueError("REPLACEMENT_REQUIRES_RECONCILED_OPEN_ORDER")
            attempts = len(family) - 1
            if attempts >= policy.max_replacements_per_intent:
                raise ValueError("REPLACEMENT_ATTEMPT_LIMIT")
            instrument = original.legs[0].instrument
            if not price.is_finite() or price <= 0 or price % instrument.tick_size:
                raise ValueError("INVALID_REPLACEMENT_PRICE_TICK")
            if (
                account.account_id != original.account_id
                or account.environment != original.environment
                or getattr(account, "hypothetical", False)
            ):
                raise ValueError("REPLACEMENT_ACCOUNT_SCOPE_MISMATCH")
            if (
                not 0
                <= (now - account.observed_at).total_seconds()
                <= policy.account_max_age_seconds
            ):
                raise ValueError("REPLACEMENT_ACCOUNT_STALE")
            current_account = self.store.projection(
                "operational_account"
                if isinstance(account, AccountOperationalState)
                else "account",
                self.scope,
            )
            if current_account is None or current_account["id"] != account.id:
                raise ValueError("REPLACEMENT_REQUIRES_CURRENT_RECONCILIATION")
            if original.maximum_buy_price is not None and price > original.maximum_buy_price:
                raise ValueError("REPLACEMENT_BUY_BOUND_EXCEEDED")
            if original.minimum_sell_price is not None and price < original.minimum_sell_price:
                raise ValueError("REPLACEMENT_SELL_BOUND_EXCEEDED")
            old_price = Decimal(parent.get("limit_price", str(original.limit_price)))
            if price == old_price:
                raise ValueError("REPLACEMENT_REQUIRES_CHANGED_PRICE")
            capital = Decimal(0)
            loss = Decimal(0)
            if original.purpose == "open":
                if not isinstance(account, AccountProfile):
                    raise ValueError("REPLACEMENT_FINANCIAL_ACCOUNT_REQUIRED")
                if not entry_ready or account.trading_blocked:
                    raise ValueError("REPLACEMENT_ENTRIES_PAUSED")
                plan = SessionPlan.model_validate(self.store.get("plans", original.plan_id))
                state = self.store.projection("plan_state", plan.id)
                approval = StrategyApproval.model_validate(
                    self.store.get("strategy_approvals", plan.approval_id)
                )
                if (
                    not state
                    or state["state"] != "ELIGIBLE"
                    or not plan.valid_from <= now < plan.entry_expires_at
                ):
                    raise ValueError("REPLACEMENT_PLAN_INELIGIBLE_OR_EXPIRED")
                session_record = self.store.get("sessions", plan.target_session_id)
                if session_record is None:
                    raise ValueError("REPLACEMENT_CALENDAR_UNAVAILABLE")
                session = TradingSession.model_validate(session_record)
                schedule = session.schedule(policy)
                if not schedule["entry_start"] <= now < schedule["entry_cutoff"]:
                    raise ValueError("REPLACEMENT_SESSION_ENTRY_WINDOW_CLOSED")
                if (
                    plan.account_id != account.account_id
                    or plan.environment != account.environment
                    or plan.risk_policy_id != policy.id
                    or plan.data_profile != self.config.data_profile
                    or plan.data_policy_version != self.config.data_policy_version
                    or plan.version != original.plan_version
                    or not plan.execution_eligible
                ):
                    raise ValueError("REPLACEMENT_PLAN_AUTHORITY_CHANGED")
                if (
                    approval.expiry <= now
                    or approval.account_id != account.account_id
                    or approval.environment != account.environment
                    or approval.strategy_id != plan.strategy_id
                    or approval.strategy_version != plan.strategy_version
                    or self.config.data_profile not in approval.profiles
                    or self.config.execution_mode not in approval.execution_modes
                ):
                    raise ValueError("REPLACEMENT_STRATEGY_APPROVAL_INVALID")
                if price > plan.max_buy_price:
                    raise ValueError("REPLACEMENT_PLAN_PRICE_BOUND_EXCEEDED")
                if self.config.execution_mode == ExecutionMode.LIVE:
                    if (
                        self.config.data_profile == DataProfile.FREE_DELAYED
                        or not self._armed_for_plan(plan)
                    ):
                        raise ValueError("REPLACEMENT_LIVE_AUTHORITY_INVALID")
                if (approval.requires_news and not news_ready) or (
                    approval.requires_event_calendar and not event_calendar_ready
                ):
                    raise ValueError("REPLACEMENT_EVENT_COVERAGE_UNAVAILABLE")
                symbol = plan.alternative.signal_symbol
                observations = self.evidence.market(now, symbol)
                if (
                    not observations
                    or (now - observations[-1].effective_event_time).total_seconds()
                    > approval.maximum_signal_latency_seconds
                ):
                    raise ValueError("REPLACEMENT_SIGNAL_STALE")
                if self.evidence.blockers(symbol, now) or not evaluate_rules(
                    plan.entry_rules,
                    observations,
                    now,
                    blocking=bool(self.evidence.blockers(symbol, now)),
                    valid_from=plan.valid_from,
                    expires_at=plan.entry_expires_at,
                    sessions=(session,),
                ):
                    raise ValueError("REPLACEMENT_TRIGGER_OR_EVENT_INVALID")
                if evaluate_rules(plan.invalidation_rules, observations, now, sessions=(session,)):
                    raise ValueError("REPLACEMENT_PLAN_INVALIDATED")
                if (
                    account.daily_pnl is None
                    or account.daily_pnl <= -account.equity * policy.daily_loss_pause_pct
                ):
                    raise ValueError("REPLACEMENT_DAILY_LOSS_OR_PNL_BLOCK")
                rows = self.store.db.execute(
                    "SELECT * FROM reservations WHERE scope=? AND state!=?",
                    (self.scope, "RELEASED"),
                ).fetchall()
                reservation = next((row for row in rows if row["intent_id"] == root_id), None)
                if reservation is None:
                    raise ValueError("REPLACEMENT_RESERVATION_MISSING")
                other_capital = sum(
                    (Decimal(row["capital"]) for row in rows if row["intent_id"] != root_id),
                    Decimal(0),
                )
                alternative = plan.alternative.model_copy(update={"entry_price": price})
                report = AccountSuitabilityService(policy).assess(
                    account,
                    [alternative],
                    self.config.data_profile,
                    [approval],
                    reserved=other_capital,
                )
                self.store.put("suitability_reports", report, self.scope)
                sizing = report.alternatives[0]
                if not sizing.feasible or sizing.quantity < original_decision.approved_quantity:
                    raise ValueError("REPLACEMENT_NO_LONGER_AFFORDABLE")
                capital = max(
                    Decimal(reservation["capital"]),
                    price * original_decision.approved_quantity * instrument.multiplier,
                )
                loss = max(
                    Decimal(reservation["loss"]),
                    sizing.planned_loss * original_decision.approved_quantity / sizing.quantity,
                )
                if approval.requires_current_quote:
                    max_age = (
                        self.config.option_quote_max_age_seconds
                        if instrument.asset_type == "option"
                        else self.config.stock_quote_max_age_seconds
                    )
                    quote = self.evidence.current_quote(
                        instrument.symbol, now, approval.required_feed, max_age
                    )
                    if (
                        not quote
                        or (quote.ask - quote.bid) / quote.ask > policy.max_quote_spread_pct
                    ):
                        raise ValueError("REPLACEMENT_CURRENT_QUOTE_REQUIRED")
            else:
                group = self.store.projection("position_groups", original.group_id)
                if not group or group["state"] == "CLOSED":
                    raise ValueError("REPLACEMENT_MANAGED_EXPOSURE_REQUIRED")
                held = next(
                    (leg for leg in account.positions if leg.symbol == instrument.symbol), None
                )
                total_filled = sum(
                    (Decimal(row.get("filled_quantity", "0")) for row in family), Decimal(0)
                )
                remaining = original_decision.approved_quantity - total_filled
                if held is None or remaining <= 0 or abs(held.quantity) < remaining:
                    raise ValueError("REPLACEMENT_CLOSE_EXCEEDS_CURRENT_HOLDING")
                if (original.legs[0].side == "sell" and held.quantity <= 0) or (
                    original.legs[0].side == "buy" and held.quantity >= 0
                ):
                    raise ValueError("REPLACEMENT_CLOSE_SIDE_INVALID")
                feed = "opra" if instrument.asset_type == "option" else "sip"
                quote = self.evidence.current_quote(
                    instrument.symbol,
                    now,
                    feed,
                    self.config.option_quote_max_age_seconds
                    if instrument.asset_type == "option"
                    else self.config.stock_quote_max_age_seconds,
                )
                if not quote:
                    raise ValueError("REPLACEMENT_EXIT_QUOTE_REQUIRED")
            child_id = digest([root_id, order_id, attempts + 1, str(price)])[:32]
            child = {
                key: value
                for key, value in parent.items()
                if key
                not in {"_revision", "broker_id", "raw_status", "replaced_by", "reconciled_at"}
            }
            child.update(
                id=child_id,
                client_order_id="gauss-" + child_id,
                state="SHADOW"
                if self.config.execution_mode in {ExecutionMode.SHADOW, ExecutionMode.REPLAY}
                else "SUBMITTING",
                filled_quantity="0",
                limit_price=str(price),
                replacement_number=attempts + 1,
                replaces_order_id=order_id,
                replaces=parent.get("broker_id"),
                reservation_id=root_id,
                created_at=now.isoformat(),
                decision_time=now.isoformat(),
            )
            audit = {
                "id": child_id,
                "intent_id": root_id,
                "parent_order_id": order_id,
                "account_profile_id": account.id,
                "event_watermark": self.store.watermark(self.scope),
                "limit_price": str(price),
                "quantity": str(original_decision.approved_quantity),
                "created_at": now.isoformat(),
                "policy_id": policy.id,
                "data_policy_version": self.config.data_policy_version,
            }
            self.store.put("replacement_decisions", audit, self.scope)
            self.store.project("orders", child_id, child, self.scope)
            if original.purpose == "open":
                self.store.db.execute(
                    "UPDATE reservations SET capital=?,loss=? WHERE intent_id=?",
                    (str(capital), str(loss), root_id),
                )
            if child["state"] == "SHADOW":
                return child
            if not parent.get("broker_id"):
                raise ValueError("REPLACEMENT_BROKER_ID_UNRECONCILED")
            parent.pop("_revision", None)
            parent.update(state="REPLACE_PENDING", pending_replacement=child_id)
            self.store.project("orders", order_id, parent, self.scope)
        try:
            result = self.broker.replace(
                parent["broker_id"], limit_price=price, client_order_id=child["client_order_id"]
            )
        except Exception as exc:
            child.update(state="UNKNOWN", error=type(exc).__name__)
            self.store.project("orders", child_id, child, self.scope)
            self.store.put(
                "incidents",
                {
                    "id": child_id + ":unknown",
                    "reason": "REPLACEMENT_UNKNOWN",
                    "intent_id": root_id,
                    "order_id": child_id,
                    "created_at": now.isoformat(),
                },
                self.scope,
            )
            return child
        child.update(
            state="ACKNOWLEDGED",
            broker_id=str(result["id"]),
            raw_status=str(result.get("status", "")),
        )
        self.store.project("orders", child_id, child, self.scope)
        return child


BROKER_STATES = {
    "new": "OPEN",
    "accepted": "ACKNOWLEDGED",
    "pending_new": "ACKNOWLEDGED",
    "partially_filled": "PARTIALLY_FILLED",
    "filled": "FILLED",
    "canceled": "CANCELLED",
    "expired": "EXPIRED",
    "rejected": "REJECTED",
    "pending_cancel": "CANCEL_PENDING",
    "pending_replace": "UNKNOWN",
    "replaced": "CANCELLED",
    "done_for_day": "OPEN",
    "suspended": "UNKNOWN",
    "stopped": "UNKNOWN",
    "calculated": "UNKNOWN",
    "held": "OPEN",
}


class Reconciler:
    def __init__(self, store, config, scope, broker):
        self.store, self.config, self.scope, self.broker = store, config, scope, broker
        self._startup_recovery_complete = False

    def _incident(self, reason, now, **details):
        self.store.put(
            "incidents",
            {"id": uuid4().hex, "reason": reason, "created_at": now.isoformat(), **details},
            self.scope,
        )

    def reconcile(self, now):
        # All potentially slow broker reads finish before the transactional write phase.
        financial_error = None
        try:
            account = self.broker.account(self.config.policy, now)
        except Exception as exc:
            financial_error = type(exc).__name__
            identity = self.broker.identity()
            account = AccountOperationalState(
                account_id=str(identity["account_id"]),
                environment=identity["environment"],
                positions=(),
                observed_at=now,
            )
        if (
            account.account_id != self.config.account_id
            or account.environment != self.config.environment
            or getattr(account, "hypothetical", False)
        ):
            raise ValueError("BROKER_ACCOUNT_ENVIRONMENT_MISMATCH")
        positions = self.broker.positions(account.account_id, now)
        orders = self.broker.orders()
        activities = self.broker.activities()
        known = self.store.projected("orders", self.scope)
        by_client = {str(order.get("client_order_id")): order for order in orders}
        for local in known:
            if local["state"] == "SHADOW":
                continue
            if local["client_order_id"] not in by_client and local["state"] not in TERMINAL_ORDERS:
                order = self.broker.order_by_client_id(local["client_order_id"])
                if order is not None:
                    by_client[local["client_order_id"]] = order
        with self.store.transaction():
            if not self._startup_recovery_complete:
                for value in self.store.list("order_intents", self.scope):
                    intent = OrderIntent.model_validate(value)
                    decision = self.store.get("risk_decisions", intent.id)
                    if (
                        not decision
                        or not decision["approved"]
                        or self.store.projection("orders", intent.id)
                    ):
                        continue
                    # The gateway persists SUBMITTING before a network write. Absence
                    # at startup proves this work had never reached that boundary.
                    plan = self.store.get("plans", intent.plan_id) or {}
                    row = {
                        "id": intent.id,
                        "intent_id": intent.id,
                        "client_order_id": "gauss-" + intent.id[:40],
                        "account_id": intent.account_id,
                        "environment": intent.environment,
                        "purpose": intent.purpose,
                        "plan_id": intent.plan_id,
                        "group_id": intent.group_id or "group-" + intent.id,
                        "state": "REJECTED",
                        "filled_quantity": "0",
                        "quantity": decision["approved_quantity"],
                        "session_id": plan.get("target_session_id"),
                        "created_at": now.isoformat(),
                        "reasons": ["PRE_SUBMISSION_RECOVERY"],
                    }
                    self.store.project("orders", intent.id, row, self.scope)
                    self.store.db.execute(
                        "UPDATE reservations SET state=? WHERE intent_id=?", ("RELEASED", intent.id)
                    )
                    self._incident("PRE_SUBMISSION_RECOVERY", now, intent_id=intent.id)
            if financial_error:
                self._incident("ACCOUNT_FINANCIAL_FIELDS_UNAVAILABLE", now, error=financial_error)
            for activity in activities:
                key = str(activity.get("id") or activity.get("activity_id") or "")
                if not key:
                    raise ValueError("broker activity lacks stable identity")
                if not self.store.get("activities", key):
                    self.store.put("activities", {**activity, "id": key}, self.scope)
                    self.store.emit("broker_activity", activity, self.scope, key)
            # Re-read projections under the account lock: a submission can have completed
            # while REST was running. Missing rows remain unknown until another cycle.
            for local in self.store.projected("orders", self.scope):
                if local["state"] == "SHADOW":
                    continue
                order = by_client.get(local["client_order_id"])
                if order is None:
                    continue
                raw_status = str(getattr(order.get("status"), "value", order.get("status")))
                state = BROKER_STATES.get(raw_status, "UNKNOWN")
                filled = Decimal(str(order.get("filled_qty") or "0"))
                previous = Decimal(local.get("filled_quantity", "0"))
                if not filled.is_finite() or filled < 0:
                    self._incident("BROKER_INVALID_FILL_QUANTITY", now, order_id=local["id"])
                    continue
                if filled < previous:
                    self._incident("BROKER_FILL_QUANTITY_REGRESSION", now, order_id=local["id"])
                    continue
                if local["state"] in TERMINAL_ORDERS and state not in TERMINAL_ORDERS:
                    # Late cumulative fills remain real even if a stale state event follows
                    # terminal confirmation; terminal order mechanics remain authoritative.
                    state = local["state"]
                if filled > previous:
                    average = (
                        Decimal(str(order["filled_avg_price"]))
                        if order.get("filled_avg_price")
                        else None
                    )
                    prior_average = (
                        Decimal(local["filled_average_price"])
                        if local.get("filled_average_price")
                        else None
                    )
                    delta_price = average
                    if average is not None and prior_average is not None and previous:
                        delta_price = (filled * average - previous * prior_average) / (
                            filled - previous
                        )
                    fill_id = f"{order['id']}:{filled}"
                    self.store.put(
                        "fills",
                        {
                            "id": fill_id,
                            "order_id": str(order["id"]),
                            "local_order_id": local["id"],
                            "intent_id": local["intent_id"],
                            "plan_id": local.get("plan_id"),
                            "group_id": local.get("group_id"),
                            "quantity": str(filled - previous),
                            "cumulative_quantity": str(filled),
                            "price": str(delta_price) if delta_price is not None else None,
                            "broker_filled_at": order.get("filled_at"),
                            "received_at": now.isoformat(),
                        },
                        self.scope,
                    )
                local.pop("_revision", None)
                local.update(
                    state=state,
                    broker_id=str(order["id"]),
                    raw_status=raw_status,
                    filled_quantity=str(filled),
                    filled_average_price=order.get("filled_avg_price"),
                    reconciled_at=now.isoformat(),
                    replaces=order.get("replaces") or local.get("replaces"),
                    replaced_by=order.get("replaced_by"),
                )
                self.store.project("orders", local["id"], local, self.scope)
            projection = self.store.projected("orders", self.scope)
            families = {}
            for local in projection:
                if local["state"] != "SHADOW":
                    families.setdefault(local["intent_id"], []).append(local)
            for intent_id, family in families.items():
                total_filled = sum(
                    (Decimal(row.get("filled_quantity", "0")) for row in family), Decimal(0)
                )
                unresolved = any(row["state"] not in TERMINAL_ORDERS for row in family)
                if total_filled:
                    self.store.db.execute(
                        "UPDATE reservations SET first_fill=1,state=? WHERE intent_id=?",
                        ("EXPOSURE", intent_id),
                    )
                    if not unresolved:
                        # Confirmed terminal fills are now included in actual account exposure.
                        # Keep first_fill and the entry slot; release only pending cash capacity.
                        self.store.db.execute(
                            "UPDATE reservations SET capital=? WHERE intent_id=?", ("0", intent_id)
                        )
                elif not unresolved:
                    self.store.db.execute(
                        "UPDATE reservations SET state=? WHERE intent_id=?", ("RELEASED", intent_id)
                    )
            assigned = set()
            roots = [
                row
                for row in projection
                if row["purpose"] == "open"
                and row["id"] == row["intent_id"]
                and row["state"] != "SHADOW"
            ]
            for local in roots:
                family = families.get(local["intent_id"], [])
                total_filled = sum(
                    (Decimal(row.get("filled_quantity", "0")) for row in family), Decimal(0)
                )
                group_orders = [
                    row for row in projection if row.get("group_id") == local["group_id"]
                ]
                unresolved = tuple(
                    row["id"]
                    for row in group_orders
                    if row["state"] not in TERMINAL_ORDERS | {"SHADOW"}
                )
                previous = self.store.projection("position_groups", local["group_id"])
                lifecycle = self.store.projection("group_lifecycle", local["group_id"]) or {}
                if (
                    previous
                    and previous["state"] == "CLOSED"
                    and total_filled <= Decimal(lifecycle.get("entry_filled_quantity", "0"))
                ):
                    continue
                if total_filled == 0 and not unresolved:
                    continue
                intent = OrderIntent.model_validate(
                    self.store.get("order_intents", local["intent_id"])
                )
                symbols = {leg.instrument.symbol for leg in intent.legs}
                # Pending zero-fill orders cannot adopt an unrelated broker position.
                legs = tuple(
                    position.model_copy(update={"group_id": local["group_id"]})
                    for position in positions
                    if total_filled > 0
                    and position.symbol in symbols
                    and position.symbol not in assigned
                )
                assigned.update(position.symbol for position in legs)
                state = (
                    "OPEN"
                    if any(leg.quantity for leg in legs)
                    else "PENDING_ENTRY"
                    if unresolved
                    else "CLOSED"
                )
                if state == "OPEN" and any(row["state"] == "PARTIALLY_FILLED" for row in family):
                    state = "PARTIALLY_OPEN"
                if any(
                    row["purpose"] == "close" and row["state"] not in TERMINAL_ORDERS | {"SHADOW"}
                    for row in group_orders
                ):
                    state = "EXIT_PENDING"
                if total_filled > Decimal(local["quantity"]):
                    state = "RECONCILIATION_REQUIRED"
                    self._incident(
                        "REPLACEMENT_CHAIN_FILL_EXCEEDS_AUTHORIZED_QUANTITY",
                        now,
                        intent_id=intent.id,
                    )
                plan = self.store.get("plans", intent.plan_id) or {}
                group = PositionGroup(
                    id=local["group_id"],
                    account_id=account.account_id,
                    environment=account.environment,
                    plan_id=intent.plan_id,
                    strategy_id=plan.get("strategy_id"),
                    exit_policy_id=plan.get("exit_policy_id", "manual-adoption-required"),
                    legs=legs,
                    state=state,
                    unresolved_orders=unresolved,
                    stop_price=(previous or {}).get("stop_price")
                    or plan.get("alternative", {}).get("stop_price"),
                    close_at=(previous or {}).get("close_at") or plan.get("close_at"),
                )
                self.store.project(
                    "position_groups", group.id, group.model_dump(mode="json"), self.scope
                )
                self.store.project(
                    "group_lifecycle",
                    group.id,
                    {"entry_filled_quantity": str(total_filled)},
                    self.scope,
                )
                if group.flat:
                    self.store.db.execute(
                        "UPDATE reservations SET capital=?,loss=? WHERE intent_id=?",
                        ("0", "0", local["intent_id"]),
                    )
            current_symbols = {position.symbol for position in positions if position.quantity}
            for previous in self.store.projected("position_groups", self.scope):
                if previous["id"].startswith("unmanaged:") and not any(
                    leg["symbol"] in current_symbols for leg in previous.get("legs", [])
                ):
                    unresolved = tuple(
                        row["id"]
                        for row in projection
                        if row.get("group_id") == previous["id"]
                        and row["state"] not in TERMINAL_ORDERS | {"SHADOW"}
                    )
                    previous.pop("_revision", None)
                    previous.update(
                        state="EXIT_PENDING" if unresolved else "CLOSED",
                        legs=[],
                        unresolved_orders=list(unresolved),
                    )
                    self.store.project("position_groups", previous["id"], previous, self.scope)
            for position in positions:
                if position.symbol in assigned or not position.quantity:
                    continue
                previous = self.store.projection("position_groups", position.group_id) or {}
                adopted = (
                    previous.get("exit_policy_id", "manual-adoption-required")
                    != "manual-adoption-required"
                )
                unresolved = tuple(
                    row["id"]
                    for row in projection
                    if row.get("group_id") == position.group_id
                    and row["state"] not in TERMINAL_ORDERS | {"SHADOW"}
                )
                group = PositionGroup(
                    id=position.group_id,
                    account_id=account.account_id,
                    environment=account.environment,
                    exit_policy_id=previous.get("exit_policy_id", "manual-adoption-required"),
                    legs=(position,),
                    state="EXIT_PENDING"
                    if unresolved
                    else "OPEN"
                    if adopted
                    else "RECONCILIATION_REQUIRED",
                    unresolved_orders=unresolved,
                    stop_price=previous.get("stop_price"),
                    close_at=previous.get("close_at"),
                )
                self.store.project(
                    "position_groups", group.id, group.model_dump(mode="json"), self.scope
                )
                incident_id = "unmanaged:" + account.account_id + ":" + position.symbol
                if not self.store.get("incidents", incident_id):
                    self.store.put(
                        "incidents",
                        {
                            "id": incident_id,
                            "reason": "UNMANAGED_OR_ASSIGNMENT_EXPOSURE",
                            "symbol": position.symbol,
                            "created_at": now.isoformat(),
                        },
                        self.scope,
                    )
            operational = AccountOperationalState(
                account_id=account.account_id,
                environment=account.environment,
                positions=positions,
                observed_at=now,
                trading_blocked=account.trading_blocked,
            )
            self.store.project(
                "operational_account", self.scope, operational.model_dump(mode="json"), self.scope
            )
            if isinstance(account, AccountProfile):
                exposure = sum(
                    (
                        max(
                            abs(position.cost_basis),
                            abs(position.market_value)
                            if getattr(position, "market_value", None) is not None
                            else Decimal(0),
                        )
                        for position in positions
                    ),
                    Decimal(0),
                ) + getattr(account, "out_of_scope_exposure", Decimal(0))
                account = account.model_copy(update={"positions": positions, "exposure": exposure})
                self.store.put("account_profiles", account, self.scope)
                self.store.project(
                    "account", self.scope, account.model_dump(mode="json"), self.scope
                )
            else:
                account = operational
        self._startup_recovery_complete = True
        return account


def liquidation_mark(group, quotes, now, max_age=5, max_skew=2):
    """Conservative synthetic mark; zero net contracts does not mean flat."""
    total = Decimal("0")
    times = []
    for leg in group.legs:
        if not leg.quantity:
            continue
        quote = quotes.get(leg.symbol)
        if (
            quote is None
            or quote.quality_class != "genuine"
            or (leg.asset_type == "option" and quote.feed != "opra")
        ):
            raise ValueError("GENUINE_LEG_QUOTE_REQUIRED")
        if not 0 <= (now - quote.effective_event_time).total_seconds() <= max_age:
            raise ValueError("STALE_LEG_QUOTE")
        price = quote.bid if leg.quantity > 0 else quote.ask
        if price is None:
            raise ValueError("MISSING_LIQUIDATION_SIDE")
        total += leg.quantity * leg.multiplier * price
        times.append(quote.effective_event_time)
    if times and (max(times) - min(times)).total_seconds() > max_skew:
        raise ValueError("CROSS_LEG_TIMESTAMP_SKEW")
    return total
