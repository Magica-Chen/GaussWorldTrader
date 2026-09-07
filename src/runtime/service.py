"""Persistent session runtime and read-only UI client with authenticated command queue."""

from __future__ import annotations

import hmac
import hashlib
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from uuid import uuid4

from .calendar import SessionCalendar
from .evidence import EvidenceService
from .models import (
    AccountProfile,
    AccountOperationalState,
    AgentRun,
    DataProfile,
    ExecutionMode,
    Instrument,
    MarketEvent,
    OrderIntent,
    OrderLeg,
    PositionGroup,
    RuntimeConfig,
    StrategyApproval,
    TradingSession,
    utc_now,
)
from .risk import ExecutionGateway, Reconciler, RiskGate, liquidation_mark
from .roles import CloseGauss, LiveGauss, PostGauss, PreGauss
from .store import DEFAULT_DATABASE_PATH, Store, TERMINAL_ORDERS, digest, canonical
from .suitability import AccountSuitabilityService, compare_capital

ROLES = ("PostGauss", "CloseGauss", "PreGauss", "LiveGauss")
COMMANDS = {
    "pause_entries",
    "resume_entries",
    "cancel_pending_entries",
    "manage_only",
    "flatten",
    "stop",
    "change_profile",
    "approve_strategy",
    "arm_live",
    "run_task",
    "reconcile",
    "adopt_position",
    "approve_model_pricing",
}


class SessionClient:
    """Separate UI process; never acquires account ownership or starts collectors."""

    def __init__(self, database_path=None, account_id=None, environment=None):
        self.database_path = database_path or os.getenv(
            "GAUSS_DATABASE_PATH", DEFAULT_DATABASE_PATH
        )
        self.account_id, self.environment = account_id, environment

    def _store(self, read_only=True):
        return Store(self.database_path, read_only=read_only)

    def _scope(self, store):
        if self.account_id and self.environment:
            return f"{self.environment}:{self.account_id}"
        states = store.projected("runtime")
        if len(states) > 1:
            raise ValueError("select explicit account_id/environment for multiple account scopes")
        return states[0]["scope"] if states else None

    def status(self):
        if not Path(self.database_path).exists():
            return {
                "runtime_state": "NOT_STARTED",
                "roles": {r: {"state": "IDLE"} for r in ROLES},
                "readiness": ["No persisted runtime state"],
                "supervision": "NOT_RUNNING",
            }
        store = self._store()
        try:
            scope = self._scope(store)
            result = store.projection("runtime", scope) if scope else None
            return result or {
                "runtime_state": "NOT_STARTED",
                "roles": {r: {"state": "IDLE"} for r in ROLES},
                "readiness": ["No persisted runtime state"],
                "supervision": "NOT_RUNNING",
            }
        finally:
            store.close()

    def health(self):
        return self.status().get(
            "health",
            {
                "price_data": "UNKNOWN",
                "news": "UNKNOWN",
                "broker_stream": "UNKNOWN",
                "reconciliation": "UNKNOWN",
                "model_workers": "IDLE",
            },
        )

    def account(self):
        if not Path(self.database_path).exists():
            return {}
        store = self._store()
        try:
            scope = self._scope(store)
            return store.projection("account", scope) or {} if scope else {}
        finally:
            store.close()

    def plans(self):
        if not Path(self.database_path).exists():
            return []
        store = self._store()
        try:
            scope = self._scope(store)
            return (
                [
                    {
                        **p,
                        "eligibility": (store.projection("plan_state", p["id"]) or {}).get(
                            "state", "DRAFT"
                        ),
                    }
                    for p in store.list("plans", scope)
                ]
                if scope
                else []
            )
        finally:
            store.close()

    def report(self, session_id=None):
        if not Path(self.database_path).exists():
            return {"status": self.status(), "plans": []}
        store = self._store()
        try:
            scope = self._scope(store)
            result = {
                key: store.list(key, scope) if scope else []
                for key in (
                    "agent_runs",
                    "snapshots",
                    "candidates",
                    "plans",
                    "plan_events",
                    "validations",
                    "decisions",
                    "fills",
                    "suitability_reports",
                    "capital_scenarios",
                    "operator_commands",
                    "incidents",
                    "session_reviews",
                    "research_reports",
                    "strategy_approvals",
                    "model_pricing",
                    "data_capabilities",
                    "costs",
                )
            }
            result["command_results"] = store.projected("command_results", scope) if scope else []
            result["orders"] = store.projected("orders", scope) if scope else []
            result["position_groups"] = store.projected("position_groups", scope) if scope else []
            result["positions"] = [
                leg for group in result["position_groups"] for leg in group.get("legs", [])
            ]
            result["risk_decisions"] = store.list("risk_decisions", scope) if scope else []
            result["status"] = store.projection("runtime", scope) if scope else self.status()
            result["account"] = store.projection("account", scope) if scope else {}
            result["costs"] = store.list("cost_ledger", scope) if scope else []
            from .research import research_cost_summary

            result["research_costs"] = (
                research_cost_summary(store, scope, session_id if session_id != "latest" else None)
                if scope
                else {}
            )
            result["pnl"] = {
                "trading_pnl": (result["account"] or {}).get("daily_pnl"),
                "operating_costs": str(result["research_costs"].get("settled_usd", "0")),
                "uncertain_model_costs": str(result["research_costs"].get("uncertain_usd", "0")),
                "reserved_model_costs": str(result["research_costs"].get("reserved_usd", "0")),
                "method": "Broker equity change less external cashflows; costs reported separately.",
            }
            if session_id and session_id != "latest":
                result["requested_session_id"] = session_id
                result["plans"] = [
                    p for p in result["plans"] if p["target_session_id"] == session_id
                ]
                plan_ids = {p["id"] for p in result["plans"]}
                snapshot_ids = {p["snapshot_id"] for p in result["plans"]}
                candidate_ids = {p["candidate_id"] for p in result["plans"]}
                result["snapshots"] = [
                    v
                    for v in result["snapshots"]
                    if v["id"] in snapshot_ids or v["target_session_id"] == session_id
                ]
                result["candidates"] = [v for v in result["candidates"] if v["id"] in candidate_ids]
                for key in ("plan_events", "validations", "decisions"):
                    result[key] = [v for v in result[key] if v.get("plan_id") in plan_ids]
                result["orders"] = [
                    v for v in result["orders"] if v.get("session_id") == session_id
                ]
                intent_ids = {v["intent_id"] for v in result["orders"]}
                result["fills"] = [v for v in result["fills"] if v.get("intent_id") in intent_ids]
                result["session_reviews"] = [
                    v for v in result["session_reviews"] if v.get("session_id") == session_id
                ]
            return result
        finally:
            store.close()

    def command(self, action, *, operator="local", confirmed=False, payload=None, token=None):
        if action not in COMMANDS:
            raise ValueError("unsupported operator command")
        store = self._store(read_only=False)
        try:
            scope = self._scope(store)
            if not scope:
                raise RuntimeError("no runtime account is available for commands")
            status = store.projection("runtime", scope) or {}
            token_env = status.get("control_token_env", "GAUSS_CONTROL_TOKEN")
            expected = os.getenv(token_env, "")
            if not expected or not token or not hmac.compare_digest(expected, str(token)):
                store.put(
                    "incidents",
                    {
                        "id": uuid4().hex,
                        "reason": "UNAUTHENTICATED_COMMAND_REJECTED",
                        "action": action,
                        "created_at": utc_now().isoformat(),
                    },
                    scope,
                )
                raise PermissionError("a configured matching control token is required")
            if not operator.strip():
                raise ValueError("operator identity is required")
            if (
                action
                in {
                    "flatten",
                    "stop",
                    "arm_live",
                    "approve_strategy",
                    "adopt_position",
                    "approve_model_pricing",
                }
                and not confirmed
            ):
                raise ValueError("explicit confirmation is required for this command")
            body = {
                "id": uuid4().hex,
                "scope": scope,
                "action": action,
                "operator": operator,
                "confirmed": confirmed,
                "payload": payload or {},
                "created_at": utc_now().isoformat(),
                "state": "QUEUED",
            }
            body["signature"] = hmac.new(
                str(token).encode(), canonical(body).encode(), hashlib.sha256
            ).hexdigest()
            with store.transaction():
                store.put("operator_commands", body, scope)
                store.emit("operator_command", body, scope, body["id"])
            return body
        finally:
            store.close()

    def scenarios(self, values, *, data_profile=None):
        store = self._store(read_only=False)
        try:
            scope = self._scope(store)
            config_row = store.projection("configuration", scope) if scope else None
            config = (
                RuntimeConfig.model_validate(
                    {k: v for k, v in config_row.items() if k != "_revision"}
                )
                if config_row
                else RuntimeConfig()
            )
            profile = DataProfile(data_profile or config.data_profile)
            from .models import Alternative

            alternatives = (
                [
                    Alternative.model_validate(c["alternative"])
                    for c in store.list("candidates", scope)
                ]
                if scope
                else []
            )
            approvals = (
                [
                    StrategyApproval.model_validate(a)
                    for a in store.list("strategy_approvals", scope)
                ]
                if scope
                else []
            )
            report = compare_capital(values, alternatives, profile, config.policy, approvals)
            record = {"id": uuid4().hex, "created_at": utc_now().isoformat(), **report}
            store.put("capital_scenarios", record, scope or "research:scenarios")
            return record
        finally:
            store.close()

    def close(self):
        pass


class SessionService(SessionClient):
    def __init__(
        self, config, broker=None, market=None, calendar=None, clock=None, event_calendar=None
    ):
        self.config = RuntimeConfig.model_validate(config)
        super().__init__(self.config.database_path, self.config.account_id, self.config.environment)
        self.store = Store(self.config.database_path)
        self.broker, self.market = broker, market
        self.calendar = calendar or SessionCalendar(provider=broker)
        from .events import FileEventCalendar

        self.event_calendar = event_calendar or (
            FileEventCalendar(self.config.event_calendar_path)
            if self.config.event_calendar_path
            else None
        )
        self.clock = clock or utc_now
        self.owner = uuid4().hex
        self.scope = f"{self.config.environment}:{self.config.account_id}"
        self.state = "STARTING"
        self.entries_paused = True
        self.account_profile = None
        self.operational_profile = None
        self.last_reconciliation = None
        self.news_ready = False
        self.price_ready = False
        self.event_calendar_ready = False
        self.calendar_ready = False
        self._started = False
        self._owns_lease = False
        self._closed = False
        self._force_role = None
        self._closing = False
        self.health_state = {
            "price_data": "UNKNOWN",
            "news": "UNKNOWN",
            "broker_stream": "UNKNOWN",
            "reconciliation": "UNKNOWN",
            "model_workers": "IDLE",
        }
        self.role_states = {r: {"state": "IDLE"} for r in ROLES}
        self.roles = {r.name: r for r in (PostGauss(), CloseGauss(), PreGauss(), LiveGauss())}
        self._research_pool = ThreadPoolExecutor(
            max_workers=self.config.research_max_parallel_jobs, thread_name_prefix="gauss-research"
        )
        self._collection_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gauss-collector"
        )
        self._research_futures = {}
        self._collection_future = None
        self._news_future = None
        self._last_collection = None
        self._wire()

    def _wire(self):
        self.evidence = EvidenceService(self.store, self.config, self.scope)
        if self.market and hasattr(self.market, "event_sink"):
            self.market.event_sink = self.evidence.ingest_market
        self.suitability = AccountSuitabilityService(self.config.policy)
        self.risk = RiskGate(self.store, self.config, self.scope, self.evidence, self.calendar)
        self.gateway = ExecutionGateway(
            self.store, self.config, self.scope, self.broker, self.evidence
        )
        self.reconciler = Reconciler(self.store, self.config, self.scope, self.broker)

    @property
    def entries_ready(self):
        return bool(
            not self.entries_paused
            and self.state == "READY"
            and self.price_ready
            and self.account_profile is not None
            and self.health_state["reconciliation"] == "HEALTHY"
        )

    def _start(self, now):
        if self.broker is None:
            raise RuntimeError("BROKER_READ_ADAPTER_REQUIRED_NO_SYNTHETIC_BALANCE")
        try:
            account = self.broker.account(self.config.policy, now)
        except Exception:
            identity = self.broker.identity() if hasattr(self.broker, "identity") else None
            if not identity:
                raise
            account = AccountOperationalState(
                account_id=identity["account_id"],
                environment=identity["environment"],
                positions=(),
                observed_at=now,
            )
        if isinstance(account, AccountProfile) and account.hypothetical:
            raise ValueError("actual runtime cannot use hypothetical capital")
        if not self.config.account_id:
            self.config = self.config.model_copy(update={"account_id": account.account_id})
            self.account_id = account.account_id
            self.scope = f"{self.config.environment}:{account.account_id}"
            self._wire()
        if (
            account.account_id != self.config.account_id
            or account.environment != self.config.environment
        ):
            raise ValueError("BROKER_ACCOUNT_ENVIRONMENT_MISMATCH")
        if (
            self.config.account_id_allowlist
            and account.account_id not in self.config.account_id_allowlist
        ):
            raise ValueError("ACCOUNT_NOT_IN_ALLOWLIST")
        self.store.acquire_lease(self.scope, self.owner, utc_now(), seconds=120)
        self._owns_lease = True
        previous = self.store.projection("runtime", self.scope)
        self.entries_paused = previous.get("entries_paused", False) if previous else False
        self.state = "RECONCILING"
        try:
            self.calendar.refresh(now)
            self.calendar.current_or_next(now)
            self.calendar_ready = True
        except Exception as exc:
            self.calendar_ready = False
            self.entries_paused = True
            self.state = "MANAGE_ONLY"
            self._incident("SESSION_CALENDAR_UNAVAILABLE", now, error=type(exc).__name__)
        for session in self.calendar.sessions.values():
            if not self.store.get("sessions", session.id):
                self.store.put("sessions", session, self.scope)
        self.store.project(
            "configuration", self.scope, self.config.model_dump(mode="json"), self.scope
        )
        if hasattr(self.broker, "start_stream"):
            self.broker.start_stream()
        self._started = True
        self._reconcile(now)
        self._probe(now)
        self._publish(now)

    def _incident(self, reason, now, **details):
        record = {
            "id": details.pop("incident_id", None) or uuid4().hex,
            "reason": reason,
            "created_at": now.isoformat(),
            **details,
        }
        # Persist before attempting a fallible console transport.
        self.store.put("incidents", record, self.scope)
        try:
            if getattr(self, "event_sink", None) is not None:
                self.event_sink({"event": "incident", **record})
                return
            print(
                json.dumps({"event": "incident", **record}, default=str),
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            # The durable ledger remains visible through SessionClient.report().
            pass

    def _reconcile(self, now):
        try:
            account = self.reconciler.reconcile(now)
            self.operational_profile = account
            if isinstance(account, AccountOperationalState):
                self.account_profile = None
                self.last_reconciliation = now.isoformat()
                self.health_state["reconciliation"] = "HEALTHY"
                self.health_state["account_financials"] = "UNAVAILABLE"
                self.state = "MANAGE_ONLY"
                self.entries_paused = True
                self._incident("ACCOUNT_FINANCIALS_UNAVAILABLE_OPERATIONS_CONTINUE", now)
                return
            self.health_state["account_financials"] = "HEALTHY"
            # Session reference is immutable; external cashflows are subtracted from equity change.
            try:
                session = self.calendar.current_or_next(now)
                self.calendar_ready = True
            except RuntimeError:
                self.calendar_ready = False
                self.account_profile = account
                self.last_reconciliation = now.isoformat()
                self.health_state["reconciliation"] = "HEALTHY"
                self.state = "MANAGE_ONLY"
                self.entries_paused = True
                return
            reference_id = f"{self.scope}:{session.id}"
            reference = self.store.get("pnl_reference", reference_id)
            if reference is None:
                reference = {
                    "id": reference_id,
                    "equity": str(account.equity),
                    "created_at": now.isoformat(),
                    "activity_ids": [a["id"] for a in self.store.list("activities", self.scope)],
                }
                self.store.put("pnl_reference", reference, self.scope)
            cashflows = Decimal("0")
            for activity in self.store.list("activities", self.scope):
                if activity["id"] in reference["activity_ids"]:
                    continue
                if activity.get("activity_type") in {"CSD", "CSW", "ACATC", "ACATS"}:
                    cashflows += Decimal(str(activity.get("net_amount", "0")))
            pnl = (
                account.daily_pnl
                if account.daily_pnl is not None
                else account.equity - Decimal(reference["equity"]) - cashflows
            )
            account = account.model_copy(update={"id": uuid4().hex, "daily_pnl": pnl})
            self.store.put("account_profiles", account, self.scope)
            self.store.project("account", self.scope, account.model_dump(mode="json"), self.scope)
            self.account_profile = account
            self.last_reconciliation = now.isoformat()
            self.health_state["reconciliation"] = "HEALTHY"
            self.health_state["broker_stream"] = getattr(
                self.broker, "stream_health", "REST_RECONCILED"
            )
            unresolved = [
                g
                for g in self.store.projected("position_groups", self.scope)
                if g["state"] == "RECONCILIATION_REQUIRED"
            ]
            if unresolved:
                self.state = "MANAGE_ONLY"
                self.entries_paused = True
            elif self.state not in {"MANAGE_ONLY", "STOPPING"}:
                self.state = "ENTRY_PAUSED" if self.entries_paused else "READY"
        except Exception as exc:
            self.health_state["reconciliation"] = "FAILED"
            self.state = "MANAGE_ONLY"
            details = {"error": type(exc).__name__}
            if isinstance(exc, ImportError):
                details["detail"] = str(exc)
            self._incident("RECONCILIATION_FAILED", now, **details)

    def _market_symbols(self):
        holdings = [
            p.symbol
            for p in (self.operational_profile.positions if self.operational_profile else ())
            if p.asset_type == "stock"
        ]
        return tuple(
            dict.fromkeys(holdings + list(self.config.symbols[: self.config.scan_universe_limit]))
        )

    def _probe(self, now):
        if self.market:
            try:
                for capability in self.market.probe(
                    self.config.account_id, now, self._market_symbols()
                ):
                    self.store.put("data_capabilities", capability, self.scope)
                self._last_probe = now
            except Exception as exc:
                self._incident("CAPABILITY_PROBE_FAILED", now, error=type(exc).__name__)
        self._refresh_event_calendar(now)

    def _refresh_event_calendar(self, now):
        if self.event_calendar is None:
            return
        try:
            snapshot = self.event_calendar.snapshot(now)
            if not snapshot.published_at <= now <= snapshot.valid_until:
                raise ValueError("event calendar is unverified/stale")
            body = snapshot.model_dump(mode="json")
            record_id = digest(body)
            self.store.put("event_calendar_versions", {**body, "id": record_id}, self.scope)
            for event in snapshot.events:
                value = event.model_dump(mode="json")
                value.update(
                    id=digest([event.id, record_id]),
                    source_event_id=event.id,
                    calendar_version_id=record_id,
                    published_at=snapshot.published_at.isoformat(),
                )
                self.store.put("scheduled_events", value, self.scope)
            self.store.project(
                "event_calendar",
                self.scope,
                {"version_id": record_id, "valid_until": snapshot.valid_until.isoformat()},
                self.scope,
            )
            self.event_calendar_ready = snapshot.verified
            self.health_state["event_calendar"] = "HEALTHY" if snapshot.verified else "PARTIAL"
        except Exception as exc:
            self.event_calendar_ready = False
            self.health_state["event_calendar"] = "FAILED"
            self._incident("EVENT_CALENDAR_UNAVAILABLE", now, error=type(exc).__name__)

    def _collect(self, now):
        if self.market is None:
            # Injected replay writes exact evidence before stepping the runtime.
            self.price_ready = bool(self.evidence.market(now))
            return
        holdings = [
            p.symbol
            for p in (self.operational_profile.positions if self.operational_profile else ())
            if p.asset_type == "stock"
        ]
        options = [
            p.symbol
            for p in (self.operational_profile.positions if self.operational_profile else ())
            if p.asset_type == "option"
        ]
        symbols = self._market_symbols()
        # Existing exposure has priority over discovery, metadata and history.
        for label, fetch in (
            (
                "HELD_OPTION_QUOTES",
                lambda: self.market.collect_options(tuple(options), now) if options else [],
            ),
            (
                "HELD_STOCK_QUOTES",
                lambda: self.market.collect_quotes(tuple(holdings), now)
                if holdings and hasattr(self.market, "collect_quotes")
                else [],
            ),
        ):
            try:
                for event in fetch():
                    self.evidence.ingest_market(event)
            except Exception as exc:
                self._incident(label + "_UNAVAILABLE", now, error=type(exc).__name__)
        try:
            metadata_exhausted = False
            versions = {
                row["instrument"]["symbol"]: row["instrument"]
                for row in reversed(self.store.list("instrument_versions", self.scope, 100000))
            }
            for symbol in symbols:
                previous = versions.get(symbol)
                if (
                    previous
                    and (
                        now - datetime.fromisoformat(previous["created_at"].replace("Z", "+00:00"))
                    ).total_seconds()
                    < 3600
                ):
                    continue
                observed = self.evidence.market(now, symbol, "bar")
                capacity = observed[-1].volume * Decimal(".01") if observed else Decimal("0")
                try:
                    item = self.broker.instrument(symbol, capacity)
                except Exception as exc:
                    self._incident(
                        "DISCOVERY_METADATA_DEFERRED", now, symbol=symbol, error=type(exc).__name__
                    )
                    if "BUDGET_EXHAUSTED" in str(exc):
                        metadata_exhausted = True
                        break
                    continue
                self.store.put(
                    "instrument_versions",
                    {
                        "id": uuid4().hex,
                        "created_at": utc_now().isoformat(),
                        "instrument": item.model_dump(mode="json"),
                    },
                    self.scope,
                )
            selected_options = set(options)
            for approval in self.approvals(now):
                if approval.option_selector and not metadata_exhausted:
                    for underlying in self.config.symbols:
                        try:
                            contracts = self.broker.option_contracts(
                                underlying, approval.option_selector, now
                            )
                        except Exception as exc:
                            self._incident(
                                "OPTION_DISCOVERY_DEFERRED",
                                now,
                                symbol=underlying,
                                error=type(exc).__name__,
                            )
                            if "BUDGET_EXHAUSTED" in str(exc):
                                metadata_exhausted = True
                                break
                            continue
                        for instrument in contracts:
                            self.store.put(
                                "instrument_versions",
                                {
                                    "id": uuid4().hex,
                                    "created_at": utc_now().isoformat(),
                                    "instrument": instrument.model_dump(mode="json"),
                                },
                                self.scope,
                            )
                            selected_options.add(instrument.symbol)
            for plan in self.store.list("plans", self.scope):
                state = self.store.projection("plan_state", plan["id"])
                if not state or state["state"] not in {
                    "ELIGIBLE",
                    "DEFERRED",
                    "PENDING_VALIDATION",
                    "REVIEW_REQUIRED",
                }:
                    continue
                alternative = plan["alternative"]
                for key in ("instrument", "short_instrument"):
                    item = alternative.get(key)
                    if item and item["asset_type"] == "option":
                        selected_options.add(item["symbol"])
            events, news = self.market.collect(symbols, now)
            for event in events:
                self.evidence.ingest_market(event)
            metadata = {
                row["instrument"]["symbol"]: row["instrument"]
                for row in reversed(self.store.list("instrument_versions", self.scope, 100000))
            }
            for symbol in symbols:
                parsed = [
                    event
                    for event in events
                    if event.symbol == symbol and event.event_type == "bar"
                ]
                if parsed and symbol in metadata:
                    latest = max(parsed, key=lambda event: event.effective_event_time)
                    item = Instrument.model_validate(metadata[symbol]).model_copy(
                        update={"liquidity_capacity": latest.volume * Decimal(".01")}
                    )
                    self.store.put(
                        "instrument_versions",
                        {
                            "id": uuid4().hex,
                            "created_at": utc_now().isoformat(),
                            "instrument": item.model_dump(mode="json"),
                        },
                        self.scope,
                    )
            for event in news:
                self.evidence.ingest_news(event)
            candidate_options = selected_options - set(options)
            if candidate_options:
                for event in self.market.collect_options(tuple(candidate_options), now):
                    self.evidence.ingest_market(event)
                    size = min(event.bid_size or Decimal("0"), event.ask_size or Decimal("0"))
                    rows = self.store.list("instrument_versions", self.scope, 100000)
                    prior = next(
                        (row for row in rows if row["instrument"]["symbol"] == event.symbol), None
                    )
                    if prior and size > 0:
                        item = Instrument.model_validate(prior["instrument"]).model_copy(
                            update={"liquidity_capacity": size}
                        )
                        self.store.put(
                            "instrument_versions",
                            {
                                "id": uuid4().hex,
                                "created_at": utc_now().isoformat(),
                                "instrument": item.model_dump(mode="json"),
                            },
                            self.scope,
                        )
            self.news_ready = self.market.health.get("news") == "HEALTHY"
            self.health_state["news"] = self.market.health.get("news", "UNKNOWN")
            self.health_state["price_data"] = self.market.health.get("price_data", "UNKNOWN")
            capabilities = self.store.list("data_capabilities", self.scope)
            required = (
                "stock_latest"
                if self.config.data_profile == DataProfile.SUBSCRIBED_REALTIME
                else "stock_historical"
            )
            current = next((c for c in capabilities if c["endpoint"] == required), None)
            entitled = bool(
                current
                and current["outcome"] == "AVAILABLE"
                and datetime.fromisoformat(current["expires_at"]) > now
            )
            self.price_ready = entitled and not self.market.gaps
        except Exception as exc:
            self.price_ready = False
            self.health_state["price_data"] = "FAILED"
            self._incident("COLLECTION_FAILED", now, error=type(exc).__name__)
        for gap in getattr(self.market, "gaps", ()):
            incident_id = digest([self.scope, "market_gap", gap])
            if self.store.get("incidents", incident_id) is None:
                self._incident(
                    gap.get("reason", "MARKET_STREAM_GAP"),
                    now,
                    incident_id=incident_id,
                    symbol=gap.get("symbol"),
                    observed_at=gap.get("at"),
                    channel="market_collection",
                )

    def approvals(self, now):
        results = []
        for row in self.store.list("strategy_approvals", self.scope):
            approval = StrategyApproval.model_validate(row)
            if (
                approval.account_id != self.config.account_id
                or approval.environment != self.config.environment
                or approval.expiry <= now
            ):
                continue
            if self.config.data_profile not in approval.profiles:
                continue
            if approval.strategy_id not in self.config.strategy_allowlist:
                continue
            results.append(approval)
        return results

    def approved_strategy_versions(self, now):
        return sorted(
            (
                {
                    "approval_id": a.id,
                    "strategy_id": a.strategy_id,
                    "strategy_version": a.strategy_version,
                    "data_profiles": [str(profile) for profile in a.profiles],
                }
                for a in self.approvals(now)
            ),
            key=lambda row: (row["strategy_id"], row["strategy_version"], row["approval_id"]),
        )

    def instrument(self, symbol, capacity):
        if hasattr(self.broker, "operational_instrument"):
            return self.broker.operational_instrument(symbol, capacity)
        if hasattr(self.broker, "instrument"):
            return self.broker.instrument(symbol, capacity)
        raise RuntimeError("BROKER_INSTRUMENT_METADATA_REQUIRED")

    def reserved_capital(self):
        rows = self.store.db.execute(
            "SELECT capital FROM reservations WHERE scope=? AND state!=?", (self.scope, "RELEASED")
        ).fetchall()
        return sum((Decimal(row["capital"]) for row in rows), Decimal("0"))

    def _recover_market_gaps(self, now):
        if not self.market or not self.market.gaps or not self.calendar_ready:
            return
        metadata = {
            row["instrument"]["symbol"]: row["instrument"]
            for row in reversed(self.store.list("instrument_versions", self.scope, 100000))
        }
        events = self.evidence.market(now)
        for gap in tuple(self.market.gaps):
            stamp = datetime.fromisoformat(gap["at"])
            option_gap = "OPTION" in gap["reason"]
            symbols = (
                [gap["symbol"]]
                if gap.get("symbol")
                else (
                    [symbol for symbol, item in metadata.items() if item["asset_type"] == "option"]
                    if option_gap
                    else list(self.config.symbols)
                )
            )
            if not symbols:
                continue
            restored = True
            quote_ids = []
            for symbol in symbols:
                item = metadata.get(symbol, {})
                is_option = item.get("asset_type") == "option" or option_gap
                feed = "opra" if is_option else "sip"
                max_age = (
                    self.config.option_quote_max_age_seconds
                    if is_option
                    else self.config.stock_quote_max_age_seconds
                )
                quote = self.evidence.current_quote(symbol, now, feed, max_age)
                if quote is None or quote.effective_event_time < stamp:
                    restored = False
                    break
                quote_ids.append(quote.id)
                underlying = item.get("underlying") or symbol
                minutes = {
                    event.effective_event_time
                    for event in events
                    if event.symbol == underlying
                    and event.event_type == "bar"
                    and event.timeframe == "1Min"
                }
                relevant = [
                    session
                    for session in self.calendar.sessions.values()
                    if session.close >= stamp and session.open <= now
                ]
                for session in relevant:
                    end = min(now.replace(second=0, microsecond=0), session.close)
                    beginning = max(
                        stamp.replace(second=0, microsecond=0) + timedelta(minutes=1),
                        session.open + timedelta(minutes=1),
                    )
                    while beginning <= end:
                        if beginning not in minutes:
                            restored = False
                            break
                        beginning += timedelta(minutes=1)
                    if not restored:
                        break
                if not restored:
                    break
            if restored:
                try:
                    self.market.gaps.remove(gap)
                except ValueError:
                    continue
                self._incident(
                    "FEED_GAP_RECOVERED",
                    now,
                    gap=gap,
                    quote_ids=quote_ids,
                    method="per-symbol completed-minute REST recovery plus fresh genuine quote",
                )
        if not self.market.gaps:
            self.health_state["price_data"] = "HEALTHY_RECOVERED"
            self.market.health["price_data"] = "HEALTHY"
            # Entitlements remain independently required after a transport recovery.
            required = (
                "stock_latest"
                if self.config.data_profile == DataProfile.SUBSCRIBED_REALTIME
                else "stock_historical"
            )
            capability = next(
                (
                    v
                    for v in self.store.list("data_capabilities", self.scope)
                    if v["endpoint"] == required
                ),
                None,
            )
            self.price_ready = bool(
                capability
                and capability["outcome"] == "AVAILABLE"
                and datetime.fromisoformat(capability["expires_at"]) > now
            )

    def _safety_events(self, now):
        for plan in self.store.list("plans", self.scope):
            current = self.store.projection("plan_state", plan["id"])
            if current and current["state"] in {
                "ELIGIBLE",
                "DEFERRED",
                "PENDING_VALIDATION",
                "REVIEW_REQUIRED",
            }:
                if self.evidence.blockers(
                    plan["alternative"]["instrument"].get("underlying")
                    or plan["alternative"]["instrument"]["symbol"],
                    now,
                ):
                    self.store.transition_plan(
                        plan["id"], "INVALIDATED", ("CURRENT_NEWS_SAFETY_VETO",), scope=self.scope
                    )
        # Scheduled events veto the declared window; absence of coverage never means no events.
        current_calendar = self.store.projection("event_calendar", self.scope)
        for event in self.store.list("scheduled_events", self.scope):
            if not event.get("blocks_entries", True):
                continue
            if current_calendar and event.get("calendar_version_id") != current_calendar.get(
                "version_id"
            ):
                continue
            stamp = datetime.fromisoformat(event["occurs_at"].replace("Z", "+00:00"))
            if abs((stamp - now).total_seconds()) <= int(event.get("block_window_seconds", 1800)):
                for plan in self.store.list("plans", self.scope):
                    if (
                        event.get("symbols")
                        and (
                            plan["alternative"]["instrument"].get("underlying")
                            or plan["alternative"]["instrument"]["symbol"]
                        )
                        not in event["symbols"]
                    ):
                        continue
                    state = self.store.projection("plan_state", plan["id"])
                    if state and state["state"] == "ELIGIBLE":
                        self.store.transition_plan(
                            plan["id"], "DEFERRED", ("SCHEDULED_EVENT_WINDOW",), scope=self.scope
                        )

    def _cancel_invalidated_entries(self, now):
        for order in self.store.projected("orders", self.scope):
            if order.get("purpose") != "open" or order.get("state") in TERMINAL_ORDERS | {
                "SHADOW",
                "CANCEL_PENDING",
            }:
                continue
            plan = self.store.projection("plan_state", order.get("plan_id"))
            if plan and plan["state"] in {"INVALIDATED", "ENTRY_EXPIRED", "REJECTED", "SUPERSEDED"}:
                try:
                    self.gateway.cancel_entries(now, plan_id=order["plan_id"])
                except Exception as exc:
                    self._incident(
                        "INVALIDATED_ENTRY_CANCEL_FAILED",
                        now,
                        intent_id=order["intent_id"],
                        error=type(exc).__name__,
                    )

    def _supervise(self, now, flatten=False):
        supervision_account = self.account_profile or self.operational_profile
        if not supervision_account:
            return
        for row in self.store.projected("position_groups", self.scope):
            group = PositionGroup.model_validate({k: v for k, v in row.items() if k != "_revision"})
            if group.flat:
                continue
            if group.exit_policy_id == "manual-adoption-required" and not flatten:
                continue
            if self.health_state["reconciliation"] != "HEALTHY":
                self._incident("EXIT_NEEDS_RECONCILIATION", now, group_id=group.id)
                continue
            quotes = {}
            trigger = flatten or (group.close_at is not None and now >= group.close_at)
            for leg in group.legs:
                feed = "opra" if leg.asset_type == "option" else "sip"
                max_age = (
                    self.config.option_quote_max_age_seconds
                    if leg.asset_type == "option"
                    else self.config.stock_quote_max_age_seconds
                )
                quote = self.evidence.current_quote(leg.symbol, now, feed, max_age)
                if quote:
                    quotes[leg.symbol] = quote
                    if (
                        group.stop_price
                        and leg.asset_type == "stock"
                        and quote.bid <= group.stop_price
                    ):
                        trigger = True
                if self.evidence.blockers(leg.symbol, now):
                    trigger = True
            if group.plan_id:
                plan = self.store.get("plans", group.plan_id)
                if plan:
                    symbol = (
                        plan["alternative"]["instrument"].get("underlying")
                        or plan["alternative"]["instrument"]["symbol"]
                    )
                    if self.evidence.blockers(symbol, now):
                        trigger = True
            if (
                group.legs
                and all(leg.asset_type == "option" for leg in group.legs)
                and len(quotes) == len(group.legs)
            ):
                try:
                    marked = liquidation_mark(
                        group, quotes, now, self.config.option_quote_max_age_seconds
                    )
                    units = min(abs(leg.quantity) for leg in group.legs if leg.quantity)
                    premium = marked / (units * group.legs[0].multiplier)
                    if group.stop_price is not None and premium <= group.stop_price:
                        trigger = True
                    if group.take_profit is not None and premium >= group.take_profit:
                        trigger = True
                except ValueError:
                    self._incident("OPTION_MARK_UNAVAILABLE", now, group_id=group.id)
            if not trigger:
                continue
            if len(quotes) != len([l for l in group.legs if l.quantity]):
                self._incident("DEGRADED_EXIT_CURRENT_QUOTE_UNAVAILABLE", now, group_id=group.id)
                continue
            try:
                liquidation_mark(group, quotes, now, self.config.policy.quote_max_age_seconds)
                legs = []
                net = Decimal("0")
                for position in group.legs:
                    if not position.quantity:
                        continue
                    instrument = self.instrument(position.symbol, abs(position.quantity))
                    side = "sell" if position.quantity > 0 else "buy"
                    legs.append(
                        OrderLeg(
                            instrument=instrument, side=side, position_intent=side + "_to_close"
                        )
                    )
                    quote = quotes[position.symbol]
                    net += quote.bid if side == "sell" else -quote.ask
                if not legs or not net:
                    continue
                tick = max(leg.instrument.tick_size for leg in legs)
                limit = (abs(net) / tick).to_integral_value(
                    rounding=ROUND_CEILING if net > 0 else ROUND_FLOOR
                ) * tick
                if limit <= 0:
                    self._incident("EXIT_PRICE_BELOW_VALID_TICK", now, group_id=group.id)
                    continue
                intent = OrderIntent(
                    created_at=now,
                    account_id=self.config.account_id,
                    environment=self.config.environment,
                    purpose="close",
                    group_id=group.id,
                    legs=tuple(legs),
                    limit_price=limit,
                    minimum_sell_price=abs(net) if net > 0 else None,
                    maximum_buy_price=abs(net) if net < 0 else None,
                    limit_effect="credit" if net > 0 else "debit",
                    expires_at=now + timedelta(seconds=10),
                    reason="Confirmed holding exit policy",
                )
                decision = self.risk.approve(
                    intent,
                    supervision_account,
                    now,
                    entry_ready=False,
                    news_ready=self.news_ready,
                    event_calendar_ready=self.event_calendar_ready,
                )
                if decision.approved:
                    self.gateway.submit(intent, decision, now)
                else:
                    self._incident(
                        "EXIT_REJECTED", now, group_id=group.id, reasons=decision.reasons
                    )
            except Exception as exc:
                self._incident(
                    "EXIT_MECHANICS_FAILED", now, group_id=group.id, error=type(exc).__name__
                )

    def _commands(self, now):
        def handle(topic, body):
            if topic != "operator_command":
                return
            action = body.get("action")
            payload = body.get("payload", {})
            try:
                signed = {key: value for key, value in body.items() if key != "signature"}
                token = os.getenv(self.config.control_token_env, "")
                expected = (
                    hmac.new(token.encode(), canonical(signed).encode(), hashlib.sha256).hexdigest()
                    if token
                    else ""
                )
                if (
                    not token
                    or not hmac.compare_digest(expected, str(body.get("signature", "")))
                    or body.get("scope") != self.scope
                ):
                    raise PermissionError("operator command signature is invalid")
                created = datetime.fromisoformat(body["created_at"])
                if (
                    created.tzinfo is None
                    or not -30 <= (utc_now() - created).total_seconds() <= 3600
                ):
                    raise PermissionError("operator command timestamp is invalid or expired")
                if action not in COMMANDS:
                    raise ValueError("unsupported operator command")
                if action in {
                    "flatten",
                    "stop",
                    "arm_live",
                    "approve_strategy",
                    "adopt_position",
                    "approve_model_pricing",
                } and not body.get("confirmed"):
                    raise ValueError("explicit confirmation required")
                if action == "pause_entries":
                    self.entries_paused = True
                    self.state = "ENTRY_PAUSED"
                elif action == "manage_only":
                    self.entries_paused = True
                    self.state = "MANAGE_ONLY"
                elif action == "resume_entries":
                    self._reconcile(now)
                    self._probe(now)
                    if (
                        self.health_state["reconciliation"] != "HEALTHY"
                        or not self.price_ready
                        or self.account_profile is None
                    ):
                        raise RuntimeError("readiness recheck failed")
                    if any(
                        g["state"] == "RECONCILIATION_REQUIRED"
                        for g in self.store.projected("position_groups", self.scope)
                    ):
                        raise RuntimeError("unmanaged exposure requires adoption")
                    self.entries_paused = False
                    self.state = "READY"
                elif action == "cancel_pending_entries":
                    self.gateway.cancel_entries(now)
                elif action == "flatten":
                    self.entries_paused = True
                    self.state = "MANAGE_ONLY"
                    self._supervise(now, flatten=True)
                elif action == "stop":
                    self.entries_paused = True
                    self.state = "STOPPING"
                    self._closing = True
                elif action == "reconcile":
                    self._reconcile(now)
                elif action == "run_task":
                    role = payload["role"]
                    if role not in ROLES:
                        raise ValueError("unknown role")
                    self._force_role = role
                elif action == "change_profile":
                    profile = DataProfile(payload["data_profile"])
                    self.config = self.config.model_copy(
                        update={"data_profile": profile, "data_policy_version": uuid4().hex}
                    )
                    self.entries_paused = True
                    self.state = "ENTRY_PAUSED"
                    self.price_ready = False
                    for plan in self.store.list("plans", self.scope):
                        state = self.store.projection("plan_state", plan["id"])
                        if state and state["state"] in {"ELIGIBLE", "DEFERRED"}:
                            self.store.transition_plan(
                                plan["id"],
                                "REVIEW_REQUIRED",
                                ("DATA_POLICY_CHANGED",),
                                scope=self.scope,
                            )
                    if self.market:
                        self.market.close()
                        self.market.config = self.config
                        self.market.cursors = {}
                        self.market.subscribed = set()
                        self.market.stream = None
                    self._wire()
                    self.store.project(
                        "configuration", self.scope, self.config.model_dump(mode="json"), self.scope
                    )
                elif action == "approve_strategy":
                    approval = StrategyApproval.model_validate(payload["approval"])
                    if (
                        approval.account_id != self.config.account_id
                        or approval.environment != self.config.environment
                        or approval.approved_by != body["operator"]
                    ):
                        raise ValueError("strategy approval identity mismatch")
                    from src.strategy.registry import get_strategy_registry

                    if approval.strategy_id not in get_strategy_registry().list_strategies():
                        raise ValueError("unknown strategies belong in experiment queue")
                    self.store.put("strategy_approvals", approval, self.scope)
                elif action == "approve_model_pricing":
                    from .research import ModelPricing

                    pricing = ModelPricing.model_validate(payload["pricing"])
                    if pricing.verified_by != body["operator"] or pricing.expires_at <= now:
                        raise ValueError("pricing requires current operator-verified source")
                    self.store.put("model_pricing", pricing, self.scope)
                elif action == "arm_live":
                    if (
                        not self.config.live_trading_enabled
                        or self.config.execution_mode != ExecutionMode.LIVE
                        or self.config.data_profile == DataProfile.FREE_DELAYED
                    ):
                        raise ValueError("live configuration gates not satisfied")
                    required = {
                        "account_id": self.config.account_id,
                        "environment": self.config.environment,
                        "risk_policy_id": self.config.policy.id,
                        "data_policy_version": self.config.data_policy_version,
                        "data_profile": str(self.config.data_profile),
                        "deployment_version": self.config.deployment_version,
                        "approved_strategy_versions": self.approved_strategy_versions(now),
                    }
                    if not required["approved_strategy_versions"]:
                        raise ValueError("live arming requires explicit approved strategy versions")
                    if any(payload.get(k) != v for k, v in required.items()):
                        raise ValueError(
                            "live arming must explicitly match account/policy/deployment versions"
                        )
                    self.store.project(
                        "arming",
                        self.scope,
                        {**required, "operator": body["operator"], "created_at": now.isoformat()},
                        self.scope,
                    )
                elif action == "adopt_position":
                    row = self.store.projection("position_groups", payload["group_id"])
                    if not row or row.get("exit_policy_id") != "manual-adoption-required":
                        raise ValueError("unmanaged position required")
                    stop = Decimal(payload["stop_price"])
                    close_at = datetime.fromisoformat(payload["close_at"])
                    if stop <= 0 or close_at.tzinfo is None:
                        raise ValueError("explicit stop and timezone-aware close required")
                    row.pop("_revision", None)
                    row.update(
                        exit_policy_id="operator-adopted-v1",
                        stop_price=str(stop),
                        close_at=close_at.isoformat(),
                        state="OPEN",
                    )
                    self.store.project("position_groups", row["id"], row, self.scope)
                self.store.project(
                    "command_results",
                    body["id"],
                    {
                        "id": body["id"],
                        "state": "COMPLETED",
                        "action": action,
                        "completed_at": now.isoformat(),
                    },
                    self.scope,
                )
            except Exception as exc:
                self.store.project(
                    "command_results",
                    body["id"],
                    {"id": body["id"], "state": "REJECTED", "reason": str(exc), "action": action},
                    self.scope,
                )
                self._incident("COMMAND_REJECTED", now, action=action, error=str(exc))

        def enqueue(topic, body):
            if topic != "operator_command":
                return
            command_id = body.get("id") or digest(body)
            if self.store.projection("command_work", command_id) is None:
                self.store.project(
                    "command_work",
                    command_id,
                    {"id": command_id, "state": "READY", "command": body},
                    self.scope,
                )

        self.store.consume("runtime:" + self.scope, enqueue, scope=self.scope, limit=1000)
        # Network operations run after the durable consumer transaction has committed.
        for work in self.store.projected("command_work", self.scope):
            if work["state"] != "READY":
                continue
            if self.store.projection("command_results", work["id"]) is None:
                if "id" not in work["command"]:
                    self._incident("MALFORMED_OPERATOR_COMMAND", now)
                else:
                    handle("operator_command", work["command"])
            self.store.project(
                "command_work", work["id"], {"id": work["id"], "state": "DONE"}, self.scope
            )

    def _finish_role(self, run, output=None, error=None):
        now = self.clock()
        if error is None:
            run = run.model_copy(
                update={
                    "state": "COMPLETED",
                    "completed_at": now,
                    "output_ids": tuple(output or ()),
                }
            )
        else:
            run = run.model_copy(
                update={
                    "state": "FAILED",
                    "completed_at": now,
                    "failure": type(error).__name__ + ": " + str(error),
                }
            )
            self._incident("ROLE_FAILED", now, role=run.role, error=run.failure)
        self.store.finish_job(run.job_id, run.state, run.model_dump(mode="json"))
        self.store.put("agent_runs", run, self.scope)
        self.role_states[run.role] = run.model_dump(mode="json")

    def _drain_research(self):
        for key, (future, run) in list(self._research_futures.items()):
            if not future.done():
                continue
            try:
                self._finish_role(run, future.result())
            except Exception as exc:
                self._finish_role(run, error=exc)
            del self._research_futures[key]
        self.health_state["model_workers"] = "RUNNING" if self._research_futures else "IDLE"

    def _observe_post(self, session, now):
        if not self.operational_profile:
            return
        bucket = int(now.timestamp() // max(60, self.config.free_poll_interval_seconds))
        record_id = digest(
            [self.scope, "post-observation", session.id, bucket, self.config.data_policy_version]
        )
        if self.store.get("session_reviews", record_id):
            return
        context = self.evidence.context(now, self.config.symbols)
        latest_suitability = self.store.list("suitability_reports", self.scope, 1)
        self.store.put(
            "session_reviews",
            {
                "id": record_id,
                "review_type": "ROLLING_OBSERVATION",
                "session_id": session.id,
                "created_at": now.isoformat(),
                "account_profile_id": self.account_profile.id if self.account_profile else None,
                "operational_account_id": self.operational_profile.id,
                "data_profile": str(self.config.data_profile),
                "data_policy_version": self.config.data_policy_version,
                "signal_as_of": context.signal_as_of.isoformat(),
                "watermarks": {symbol: stamp.isoformat() for symbol, stamp in context.watermarks},
                "broker_reconciled_at": self.last_reconciliation,
                "trading_pnl": str(self.account_profile.daily_pnl)
                if self.account_profile
                else None,
                "cash": str(self.account_profile.cash) if self.account_profile else None,
                "exposure": str(self.account_profile.exposure) if self.account_profile else None,
                "current_safety_news_ids": [
                    event.id for event in self.evidence.news(now, safety=True)
                ],
                "suitability_report_id": latest_suitability[0]["id"]
                if latest_suitability
                else None,
                "outcome": "OBSERVATION",
                "candidate_authority": False,
                "market_reaction_observed_through": context.signal_as_of.isoformat(),
            },
            self.scope,
        )

    def _run_role(self, role, session, now, job_key):
        inputs = digest(
            [
                job_key,
                self.config.data_profile,
                self.config.data_policy_version,
                self.config.policy.id,
                self.account_profile.model_dump(mode="json") if self.account_profile else None,
            ]
        )
        job_id = f"{self.scope}:{role}:{job_key}:{self.config.data_policy_version}"
        if role in {"PostGauss", "CloseGauss"} and (
            job_id in self._research_futures
            or len(self._research_futures) >= self.config.research_max_parallel_jobs
        ):
            return
        if not self.store.claim_job(
            job_id, inputs, self.owner, now, self.config.policy.research_seconds
        ):
            return
        run = AgentRun(
            role=role,
            job_id=job_id,
            input_hash=inputs,
            account_profile_id=self.account_profile.id if self.account_profile else None,
            data_profile=self.config.data_profile,
            state="RUNNING",
            created_at=now,
        )
        self.role_states[role] = {"state": "RUNNING", "job_id": job_id}
        if role in {"PostGauss", "CloseGauss"}:
            # The role gets a fixed config/account/time view and no broker/gateway objects.
            view = copy.copy(self)
            view.account_profile = self.account_profile.model_copy(deep=True)
            view.config = self.config.model_copy(deep=True)
            view.evidence = EvidenceService(self.store, view.config, self.scope)
            view.suitability = AccountSuitabilityService(view.config.policy)
            view.broker = view.market = view.gateway = view.risk = view.reconciler = None
            view.calendar = SessionCalendar(tuple(self.calendar.sessions.values()))
            future = self._research_pool.submit(self.roles[role].run, view, session, now)
            self._research_futures[job_id] = (future, run)
            return
        try:
            self._finish_role(run, self.roles[role].run(self, session, now))
        except Exception as exc:
            self._finish_role(run, error=exc)

    def _schedule_collection(self, now):
        if self.market is None:
            self._collect(now)
            return
        if self._collection_future and self._collection_future.done():
            self._collection_future.result()
            self._collection_future = None
        if self._news_future and self._news_future.done():
            self._news_future.result()
            self._news_future = None
        interval = (
            self.config.free_poll_interval_seconds
            if self.config.data_profile == DataProfile.FREE_DELAYED
            else self.config.poll_seconds
        )
        if self._collection_future is None and (
            self._last_collection is None
            or (now - self._last_collection).total_seconds() >= interval
        ):
            self._last_collection = now
            self._collection_future = self._collection_pool.submit(self._collect, now)
        if hasattr(self.market, "collect_news") and self._news_future is None:
            self._news_future = self._collection_pool.submit(self._collect_news, now)

    def _collect_news(self, now):
        try:
            for event in self.market.collect_news(self.config.symbols, now):
                self.evidence.ingest_news(event)
            self.news_ready = self.market.health.get("news") == "HEALTHY"
            self.health_state["news"] = self.market.health.get("news", "UNKNOWN")
        except Exception as exc:
            self.news_ready = False
            self.health_state["news"] = "FAILED"
            self._incident("NEWS_COLLECTION_FAILED", now, error=type(exc).__name__)

    def run_once(self, task=None):
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("wall clock must be timezone aware")
        now = now.astimezone(timezone.utc)
        if self._closed:
            raise RuntimeError("runtime is closed")
        if not self._started:
            self._start(now)
        self.store.acquire_lease(self.scope, self.owner, utc_now(), seconds=120)
        if self.calendar.provider and getattr(self, "_calendar_date", None) != now.date():
            try:
                self.calendar.refresh(now)
                self._calendar_date = now.date()
                self.calendar_ready = True
            except Exception as exc:
                self.calendar_ready = False
                self._incident("CALENDAR_REFRESH_FAILED", now, error=type(exc).__name__)
        self._reconcile(now)
        if not getattr(self, "_last_probe", None) or (now - self._last_probe).total_seconds() > 240:
            self._probe(now)
        self._drain_research()
        self._schedule_collection(now)
        if self.market and hasattr(self.market, "events"):
            while not self.market.events.empty():
                self.evidence.ingest_market(self.market.events.get_nowait())
        self._recover_market_gaps(now)
        self._refresh_event_calendar(now)
        self._safety_events(now)
        self._cancel_invalidated_entries(now)
        self._commands(now)
        self._supervise(now)
        if self.account_profile is not None and self.calendar_ready:
            current = self.calendar.current_or_next(now)
            previous = self.calendar.previous(now)
            schedule = current.schedule(self.config.policy)
            if previous and now < current.open:
                self._observe_post(previous, now)
            requested = task or self._force_role
            if requested and requested not in ROLES:
                raise ValueError("unknown role")
            self._force_role = None
            # Market completeness belongs to each symbol and actual close, not current age alone.
            post_ready = bool(previous and self.evidence.clock.signal_as_of(now) >= previous.close)
            if post_ready and self.config.symbols:
                for symbol in self.config.symbols:
                    bars = self.evidence.market(now, symbol, "bar")
                    bars = [
                        b for b in bars if previous.open < b.effective_event_time <= previous.close
                    ]
                    if not bars or max(b.effective_event_time for b in bars) < previous.close:
                        post_ready = False
            if post_ready and requested in {None, "PostGauss"}:
                self._run_role("PostGauss", previous, now, previous.id)
            close_target = (
                self.calendar.current_or_next(previous.close + timedelta(seconds=1))
                if previous
                else None
            )
            close_useful = (
                close_target is not None
                and now < close_target.schedule(self.config.policy)["entry_cutoff"]
            )
            if (
                post_ready
                and close_useful
                and now >= previous.close + timedelta(minutes=self.config.research_offset_minutes)
                and requested in {None, "CloseGauss"}
            ):
                post_id = f"{self.scope}:PostGauss:{previous.id}:{self.config.data_policy_version}"
                row = self.store.db.execute(
                    "SELECT state FROM jobs WHERE id=?", (post_id,)
                ).fetchone()
                if row and row["state"] == "COMPLETED":
                    self._run_role("CloseGauss", previous, now, previous.id)
            if schedule["pre_start"] <= now < schedule["entry_cutoff"] and requested in {
                None,
                "PreGauss",
            }:
                bucket = int(now.timestamp() // (self.config.pre_review_interval_minutes * 60))
                self._run_role("PreGauss", current, now, f"{current.id}:{bucket}")
            if schedule["entry_start"] <= now < schedule["entry_cutoff"] and requested in {
                None,
                "LiveGauss",
            }:
                self._run_role(
                    "LiveGauss",
                    current,
                    now,
                    f"{current.id}:{int(now.timestamp() // self.config.poll_seconds)}",
                )
        self._publish(now)
        if self._closing:
            self._closing = False
            self.request_shutdown()
        return self.status()

    def _publish(self, now):
        try:
            session = self.calendar.current_or_next(now)
            schedule = {k: v.isoformat() for k, v in session.schedule(self.config.policy).items()}
            session_value = session.model_dump(mode="json")
        except RuntimeError:
            session_value = None
            schedule = {}
        context = self.evidence.context(now, self.config.symbols)
        blockers = []
        if not self.calendar_ready:
            blockers.append("SESSION_CALENDAR_UNAVAILABLE")
        if not self.price_ready:
            blockers.append("MARKET_DATA_NOT_READY")
        if not self.news_ready:
            blockers.append("NEWS_COVERAGE_UNAVAILABLE")
        if not self.event_calendar_ready:
            blockers.append("EVENT_CALENDAR_UNAVAILABLE")
        if not self.approvals(now):
            blockers.append("NO_APPROVED_STRATEGIES")
        if self.health_state["reconciliation"] != "HEALTHY":
            blockers.append("BROKER_NOT_RECONCILED")
        state = {
            "scope": self.scope,
            "account_id": self.config.account_id,
            "environment": self.config.environment,
            "runtime_state": self.state,
            "execution_mode": str(self.config.execution_mode),
            "data_profile": str(self.config.data_profile),
            "data_policy_version": self.config.data_policy_version,
            "risk_policy_id": self.config.policy.id,
            "deployment_version": self.config.deployment_version,
            "approved_strategy_versions": self.approved_strategy_versions(now),
            "roles": self.role_states,
            "session": session_value,
            "schedule": schedule,
            "signal_as_of": context.signal_as_of.isoformat(),
            "configured_delay_seconds": context.configured_delay_seconds,
            "watermarks": dict((k, v.isoformat()) for k, v in context.watermarks),
            "entries_paused": self.entries_paused,
            "entries_ready": self.entries_ready,
            "updated_at": now.isoformat(),
            "last_reconciliation": self.last_reconciliation,
            "health": self.health_state,
            "account_financials_ready": self.account_profile is not None,
            "readiness": blockers,
            "supervision": "RUNNING" if not self._closed else "ENDED",
            "control_token_env": self.config.control_token_env,
        }
        self.store.project("runtime", self.scope, state, self.scope)

    def run_forever(self):
        try:
            while not self._closed:
                self.run_once()
                time.sleep(min(60, self.config.poll_seconds))
        except KeyboardInterrupt:
            return self.request_shutdown()

    def request_shutdown(self, acknowledge_unmanaged_exposure=False):
        if self._closed:
            return {"stopped": True, "supervision": "ENDED"}
        if not self._owns_lease:
            # Failed startup never supervised this account. Close local resources only;
            # do not publish state, cancel orders, or record an account shutdown.
            self._closed = True
            self._research_pool.shutdown(wait=True, cancel_futures=True)
            self._collection_pool.shutdown(wait=True, cancel_futures=True)
            if self.market:
                self.market.close()
            if self.event_calendar and hasattr(self.event_calendar, "close"):
                self.event_calendar.close()
            if self.broker and hasattr(self.broker, "close"):
                self.broker.close()
            self.store.close()
            return {"stopped": True, "supervision": "NOT_STARTED"}
        now = self.clock()
        self.entries_paused = True
        self.state = "MANAGE_ONLY"
        if self._started:
            self.gateway.cancel_entries(now)
            self._reconcile(now)
        groups = [
            g for g in self.store.projected("position_groups", self.scope) if g["state"] != "CLOSED"
        ]
        orders = [
            o
            for o in self.store.projected("orders", self.scope)
            if o["state"] not in TERMINAL_ORDERS | {"SHADOW"}
        ]
        if (
            groups or orders or self.health_state["reconciliation"] == "FAILED"
        ) and not acknowledge_unmanaged_exposure:
            self._publish(now)
            return {
                "stopped": False,
                "runtime_state": "MANAGE_ONLY",
                "remaining_groups": groups,
                "unresolved_orders": orders,
                "reason": "Supervision continues until exposure resolves or shutdown is explicitly acknowledged.",
            }
        self._incident(
            "SUPERVISION_ENDING_ACKNOWLEDGED"
            if groups or orders or self.health_state["reconciliation"] == "FAILED"
            else "CLEAN_SHUTDOWN",
            now,
            remaining_groups=len(groups),
            unresolved_orders=len(orders),
            reconciliation=self.health_state["reconciliation"],
        )
        self.state = "STOPPING"
        self._closed = True
        self._publish(now)
        self._research_pool.shutdown(wait=True, cancel_futures=True)
        self._collection_pool.shutdown(wait=True, cancel_futures=True)
        self._drain_research()
        if self.market:
            self.market.close()
        if self.event_calendar and hasattr(self.event_calendar, "close"):
            self.event_calendar.close()
        if self.broker and hasattr(self.broker, "close"):
            self.broker.close()
        self.store.release_lease(self.scope, self.owner)
        self.store.close()
        return {
            "stopped": True,
            "supervision": "ENDED",
            "remaining_groups": groups,
            "unresolved_orders": orders,
        }

    def close(self):
        return self.request_shutdown()

    def doctor(self):
        now = self.clock()
        result = {
            "execution_mode": str(self.config.execution_mode),
            "data_profile": str(self.config.data_profile),
            "signal_as_of": self.evidence.clock.signal_as_of(now).isoformat(),
            "broker_writes": False,
            "checks": {},
            "blockers": [],
        }
        try:
            account = self.broker.account(self.config.policy, now)
            result["account"] = account.model_dump(mode="json")
            self.calendar.refresh(now)
            result["session"] = self.calendar.current_or_next(now).model_dump(mode="json")
            result["checks"]["broker"] = "VERIFIED"
            if self.config.account_id and self.config.account_id != account.account_id:
                result["blockers"].append("ACCOUNT_ID_MISMATCH")
            capabilities = (
                self.market.probe(account.account_id, now, self.config.symbols)
                if self.market
                else []
            )
            result["data_capabilities"] = [c.model_dump(mode="json") for c in capabilities]
            result["blockers"].extend(
                f"{c.endpoint}:{c.outcome}" for c in capabilities if c.outcome != "AVAILABLE"
            )
            if not self.config.symbols:
                result["blockers"].append("NO_SYMBOLS_CONFIGURED")
            actual_scope = f"{account.environment}:{account.account_id}"
            approved = [
                StrategyApproval.model_validate(row)
                for row in self.store.list("strategy_approvals", actual_scope)
            ]
            if not any(
                a.expiry > now
                and a.strategy_id in self.config.strategy_allowlist
                and self.config.data_profile in a.profiles
                for a in approved
            ):
                result["blockers"].append("NO_APPROVED_STRATEGIES")
        except Exception as exc:
            result["blockers"].append(type(exc).__name__ + ": " + str(exc))
        result["ready"] = not result["blockers"]
        return result


def build_service(config=None):
    if config is None:
        from src.settings import get_gauss_config

        config = get_gauss_config()
    config = RuntimeConfig.model_validate(config)
    from src.settings import get_config

    credentials = get_config().alpaca
    if not credentials.api_key or not credentials.secret_key:
        raise RuntimeError("Alpaca credentials required for current operational account state")
    from .broker import AlpacaBroker, AlpacaMarket

    broker = AlpacaBroker(credentials.api_key, credentials.secret_key, config.environment)
    market = AlpacaMarket(credentials.api_key, credentials.secret_key, config)
    from .events import configured_event_calendar

    return SessionService(config, broker=broker, market=market,
                          event_calendar=configured_event_calendar(config))


def doctor(config=None):
    service = build_service(config)
    try:
        return service.doctor()
    finally:
        # Doctor never starts ownership, streams, commands or gateway submission.
        service.store.close()
        service.market.close()
        service.broker.close()
        if service.event_calendar and hasattr(service.event_calendar, "close"):
            service.event_calendar.close()


def replay_fixture(path, *, database_path=None):
    """Replay into a fresh isolated ledger; fixture paths can never target operational state."""
    import tempfile

    data = json.loads(Path(path).read_text()) if not isinstance(path, dict) else copy.deepcopy(path)
    config = RuntimeConfig.model_validate(data["config"])
    if config.execution_mode not in {ExecutionMode.REPLAY, ExecutionMode.SHADOW}:
        raise ValueError("fixture replay cannot select broker-writing execution modes")
    if database_path is not None:
        target = Path(database_path)
        if target.exists() and target.stat().st_size:
            raise ValueError("replay requires a fresh empty database")
    else:
        target = Path(tempfile.mkdtemp(prefix="gauss-replay-")) / "session.sqlite3"
    config = config.model_copy(update={"database_path": str(target)})
    sessions = tuple(TradingSession.model_validate(v) for v in data["sessions"])
    if not data["steps"]:
        raise ValueError("replay requires at least one wall-clock step")
    state = {
        "now": datetime.fromisoformat(data["steps"][0]["wall_time"].replace("Z", "+00:00")),
        "account": AccountProfile.model_validate(data["account"]),
        "orders": [],
        "positions": [],
        "activities": [],
    }
    if state["account"].hypothetical:
        raise ValueError(
            "replay operational account requires explicit actual test scope; scenarios use comparison"
        )

    class ReplayBroker:
        def account(self, policy, now):
            return state["account"].model_copy(
                update={"id": uuid4().hex, "observed_at": now, "created_at": now}
            )

        def positions(self, account_id, now):
            from .models import PositionLeg

            return tuple(PositionLeg.model_validate(v) for v in state["positions"])

        def orders(self):
            return state["orders"]

        def activities(self):
            return state["activities"]

        def order_by_client_id(self, key):
            return next((o for o in state["orders"] if o["client_order_id"] == key), None)

        def instrument(self, symbol, capacity):
            return Instrument(symbol=symbol, liquidity_capacity=capacity)

        def submit(self, *args):
            raise AssertionError("replay broker writes are forbidden")

        def cancel(self, *args):
            raise AssertionError("replay broker writes are forbidden")

    service = SessionService(
        config,
        broker=ReplayBroker(),
        calendar=SessionCalendar(sessions),
        clock=lambda: state["now"],
    )
    reports = []
    try:
        if not config.account_id:
            service.config = config.model_copy(update={"account_id": state["account"].account_id})
            service.account_id = state["account"].account_id
            service.scope = f"{config.environment}:{service.account_id}"
            service._wire()
        for approval in data.get("approvals", []):
            service.store.put(
                "strategy_approvals", StrategyApproval.model_validate(approval), service.scope
            )
        for value in data.get("instruments", []):
            instrument = Instrument.model_validate(value)
            service.store.put(
                "instrument_versions",
                {
                    "id": uuid4().hex,
                    "created_at": state["now"].isoformat(),
                    "instrument": instrument.model_dump(mode="json"),
                },
                service.scope,
            )
        for step in data["steps"]:
            stamp = datetime.fromisoformat(step["wall_time"].replace("Z", "+00:00"))
            if stamp < state["now"]:
                raise ValueError("replay wall-clock steps must be monotonic")
            state["now"] = stamp
            if "account" in step:
                state["account"] = AccountProfile.model_validate(step["account"])
            for key in ("orders", "positions", "activities"):
                if key in step:
                    state[key] = step[key]
            for event in step.get("market_events", []):
                service.evidence.ingest_market(event)
                parsed = MarketEvent.model_validate(event)
                if parsed.event_type == "bar" and parsed.volume > 0:
                    instrument = Instrument(
                        symbol=parsed.symbol, liquidity_capacity=parsed.volume * Decimal(".01")
                    )
                    service.store.put(
                        "instrument_versions",
                        {
                            "id": uuid4().hex,
                            "created_at": state["now"].isoformat(),
                            "instrument": instrument.model_dump(mode="json"),
                        },
                        service.scope,
                    )
            for event in step.get("news_events", []):
                service.evidence.ingest_news(event)
            service.news_ready = step.get("news_ready", False)
            service.event_calendar_ready = step.get("event_calendar_ready", False)
            reports.append(service.run_once())
            for _ in range(4):
                pending = list(service._research_futures.values())
                if not pending:
                    break
                for future, _run in pending:
                    while not future.done():
                        service._reconcile(state["now"])
                        service._supervise(state["now"])
                        time.sleep(0.01)
                service._drain_research()
                service.run_once()
        report = service.report()
        report["replay_steps"] = reports
        report["broker_writes"] = 0
        report["database_path"] = str(target)
        report["execution_model"] = "decision-only; no inferred fills from delayed signal prices"
        return report
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
