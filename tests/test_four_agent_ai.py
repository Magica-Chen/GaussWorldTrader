"""Offline four-stage AI decisions, asynchronous supervision, and quota stops."""

import json
import re
import socket
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from src.llm.codex_cli import CodexCLIError, CodexQuotaWindow, CodexResult, CodexUsage
from src.runtime.calendar import SessionCalendar
from src.runtime.models import (
    AccountProfile,
    Instrument,
    MarketEvent,
    NewsEvent,
    PositionGroup,
    PositionLeg,
    RuntimeConfig,
    SessionPlan,
    StrategyApproval,
    TradingSession,
)
from src.runtime.research import ResearchAnnotation, ResearchJobRunner, record_codex_quota
from src.runtime.roles import CloseGauss, LiveGauss, PostGauss, PreGauss
from src.runtime.service import SessionClient, SessionService

FIXTURE = Path(__file__).parent / "fixtures" / "gauss_session_day.json"


def quota(used=20):
    return CodexUsage((CodexQuotaWindow("codex", "primary", used, None, 300),), time.time())


class OfflineBroker:
    def __init__(self, account):
        self.template, self.rows, self.holdings, self.writes = account, [], (), []
        self.reads = 0

    def account(self, policy, now):
        self.reads += 1
        return self.template.model_copy(update={"id": uuid4().hex, "observed_at": now})

    def positions(self, account_id, now):
        return tuple(p.model_copy(update={"reconciled_at": now}) for p in self.holdings)

    def orders(self):
        return self.rows

    def activities(self):
        return []

    def instrument(self, symbol, capacity):
        return Instrument(symbol=symbol, liquidity_capacity=capacity)

    def order_by_client_id(self, client_id):
        return next((r for r in self.rows if r["client_order_id"] == client_id), None)

    def submit(self, intent, quantity, client_id):
        self.writes.append(("submit", intent, quantity))
        row = {
            "id": "offline-" + intent.id,
            "client_order_id": client_id,
            "status": "new",
            "filled_qty": "0",
        }
        self.rows.append(row)
        return row

    def cancel(self, broker_id):
        self.writes.append(("cancel", broker_id))
        next(r for r in self.rows if r["id"] == broker_id)["status"] = "canceled"

    def replace(self, *args, **kwargs):
        self.writes.append(("replace", args, kwargs))
        raise AssertionError("AI tests never need a broker replacement")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("AI integration tests must never contact a network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("src.llm.codex_cli.read_codex_usage", lambda **kwargs: quota())


@pytest.fixture
def model(monkeypatch):
    calls, outcomes = [], {}

    def run(prompt, **kwargs):
        payload = json.loads(prompt.split("\n", 1)[1])
        calls.append(payload)
        result = outcomes.get(payload["role"], "SUPPORT")
        if isinstance(result, Exception):
            raise result
        annotation = ResearchAnnotation(
            outcome=result,
            summary="Offline evidence review for " + payload["role"],
            source_ids=tuple(payload["source_ids"][:1]),
            uncertainties=("Synthetic evidence only",),
            experiment_hypothesis=None,
        )
        return CodexResult(annotation.model_dump_json(), annotation, 100, 20, usage_limits=quota())

    monkeypatch.setattr("src.llm.codex_cli.run_codex", run)
    return calls, outcomes


class Scenario:
    def __init__(self, tmp_path, *, symbols=("TEST_STOCK",), fixture=None):
        self.fixture = fixture or json.loads(FIXTURE.read_text())
        identity = "offline-ai-" + uuid4().hex
        self.now = datetime.fromisoformat(self.fixture["steps"][0]["wall_time"])
        config = RuntimeConfig.model_validate(
            {
                **self.fixture["config"],
                "account_id": identity,
                "database_path": str(tmp_path / "ai.sqlite3"),
                "payload_directory": str(tmp_path / "evidence"),
                "research_backend": "codex_cli",
                "execution_mode": "paper",
                "data_profile": "SUBSCRIBED_REALTIME",
                "symbols": symbols,
                "notification_channels": (),
            }
        )
        account = AccountProfile.model_validate({**self.fixture["account"], "account_id": identity})
        self.broker = OfflineBroker(account)
        self.sessions = tuple(TradingSession.model_validate(r) for r in self.fixture["sessions"])
        self.service = SessionService(
            config,
            broker=self.broker,
            calendar=SessionCalendar(self.sessions),
            clock=lambda: self.now,
        )
        self.service._start(self.now)
        self.scope, self.store = self.service.scope, self.service.store
        for source in self.fixture["approvals"]:
            approval = StrategyApproval.model_validate(
                {
                    **source,
                    "account_id": identity,
                    "execution_modes": ["paper", "shadow"],
                    "requires_current_quote": True,
                }
            )
            self.store.put("strategy_approvals", approval, self.scope)
        for row in self.fixture["instruments"]:
            self.store.put(
                "instrument_versions",
                {
                    "id": row["symbol"],
                    "created_at": self.now.isoformat(),
                    "instrument": Instrument.model_validate(row).model_dump(mode="json"),
                },
                self.scope,
            )
        self.advance(0)

    def ready(self):
        self.service._reconcile(self.now)
        self.service._quota_checked_at = self.now
        record_codex_quota(self.store, self.scope, quota(), now=self.now)
        self.service.state, self.service.entries_paused = "READY", False
        self.service.price_ready = self.service.news_ready = True
        self.service.event_calendar_ready = self.service.calendar_ready = True

    def advance(self, step):
        row = self.fixture["steps"][step]
        self.now = datetime.fromisoformat(row["wall_time"].replace("Z", "+00:00"))
        for event in row.get("market_events", ()):
            self.service.evidence.ingest_market(MarketEvent.model_validate(event))
        self.ready()
        if step == 4:
            for symbol in self.service.config.symbols:
                self.service.evidence.ingest_market(
                    MarketEvent(
                        symbol=symbol,
                        source="offline-quote",
                        source_version=self.now.isoformat(),
                        event_type="quote",
                        effective_event_time=self.now,
                        received_at=self.now,
                        available_at=self.now,
                        feed="sip",
                        quality_class="genuine",
                        bid="100.05",
                        ask="100.06",
                    )
                )

    def finish(self):
        for future in list(self.service._ai_futures.values()):
            future.result(timeout=5)
        for future, _ in list(self.service._research_futures.values()):
            future.result(timeout=5)
        self.service._drain_research()

    def research(self):
        PostGauss().run(self.service, self.sessions[0], self.now)
        self.advance(1)
        CloseGauss().run(self.service, self.sessions[0], self.now)
        return [SessionPlan.model_validate(r) for r in self.store.list("plans", self.scope)]

    def prepare_live(self):
        plans = self.research()
        assert plans, "fixture must produce a feasible AI-supported conditional plan"
        self.advance(3)
        PreGauss().run(self.service, self.sessions[1], self.now)
        self.finish()
        PreGauss().run(self.service, self.sessions[1], self.now)
        assert all(self.store.projection("plan_state", p.id)["state"] == "ELIGIBLE" for p in plans)
        self.advance(4)
        return plans

    def close(self):
        self.finish()
        self.service.request_shutdown(acknowledge_unmanaged_exposure=True)


@pytest.fixture
def scenario(tmp_path, model):
    value = Scenario(tmp_path)
    yield value
    value.close()


def test_all_four_stages_review_evidence_before_automatic_paper_submission(scenario, model):
    calls, _ = model
    scenario.prepare_live()
    live = LiveGauss()
    live.run(scenario.service, scenario.sessions[1], scenario.now)
    assert scenario.broker.writes == [], "pending LiveGauss review cannot submit"
    scenario.finish()
    live.run(scenario.service, scenario.sessions[1], scenario.now)
    assert {c["role"] for c in calls} == {"PostGauss", "CloseGauss", "PreGauss", "LiveGauss"}
    assert len(scenario.broker.writes) == 1
    assert scenario.broker.writes[0][0] == "submit"
    assert scenario.broker.writes[0][1].purpose == "open"
    assert all(c["source_ids"] for c in calls)
    assert all(r["billing"] == "subscription" for r in scenario.store.projected("codex_usage"))
    assert not scenario.store.projected("research_spend")


def test_out_of_bound_quote_preserves_entry_for_later_fresh_executable_ask(scenario, model):
    calls, _ = model
    plan = scenario.prepare_live()[0]
    live = LiveGauss()
    tick = plan.alternative.instrument.tick_size

    def quote(ask):
        scenario.now += timedelta(seconds=1)
        scenario.ready()
        scenario.service.evidence.ingest_market(
            MarketEvent(
                symbol=plan.alternative.instrument.symbol,
                source="offline-executable-quote",
                source_version=scenario.now.isoformat(),
                event_type="quote",
                effective_event_time=scenario.now,
                received_at=scenario.now,
                available_at=scenario.now,
                feed="sip",
                quality_class="genuine",
                bid=ask - tick,
                ask=ask,
            )
        )

    quote(plan.max_buy_price + tick)
    assert live.run(scenario.service, scenario.sessions[1], scenario.now) == []
    assert scenario.store.list("order_intents", scenario.scope) == []
    assert scenario.store.list("risk_decisions", scenario.scope) == []
    assert not any(call["role"] == "LiveGauss" for call in calls)
    quote(plan.max_buy_price - tick * Decimal("3.5"))
    live.run(scenario.service, scenario.sessions[1], scenario.now)
    assert scenario.broker.writes == [], "fresh quote still needs LiveGauss AI support"
    scenario.finish()
    live.run(scenario.service, scenario.sessions[1], scenario.now)
    assert len(scenario.broker.writes) == 1
    intent = scenario.broker.writes[0][1]
    assert intent.limit_price == plan.max_buy_price - tick * 3
    assert intent.limit_price < intent.maximum_buy_price == plan.max_buy_price


def test_first_research_waits_for_retired_legacy_holdings_and_confirmed_flat_order(
    tmp_path, model, monkeypatch
):
    dates = {"08": "11", "09": "14", "10": "15", "11": "18"}
    fixture = json.loads(
        re.sub(
            r"2026-09-(08|09|10|11)",
            lambda match: "2026-09-" + dates[match.group(1)],
            FIXTURE.read_text(),
        )
    )
    value = Scenario(tmp_path, fixture=fixture)
    calls, _ = model
    try:
        previous, current = value.sessions[:2]
        assert previous.open.weekday() == 4 and current.open.weekday() == 0
        value.now = previous.close + timedelta(days=1)
        value.broker.holdings = (
            PositionLeg(
                account_id=value.service.config.account_id,
                environment="paper",
                group_id="unmanaged:LEGACY",
                symbol="LEGACY",
                asset_type="stock",
                quantity="100",
                cost_basis="10000",
                reconciled_at=value.now,
            ),
        )
        value.ready()
        assert not value.service.entries_ready
        value.service.run_once(task="PostGauss")
        value.service.run_once(task="CloseGauss")
        assert value.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        assert calls == []

        monkeypatch.setenv(value.service.config.control_token_env, "offline-adoption-secret")
        command = SessionClient(
            value.service.config.database_path,
            account_id=value.service.config.account_id,
            environment="paper",
        ).command(
            "adopt_position",
            operator="offline-owner",
            confirmed=True,
            token="offline-adoption-secret",
            payload={
                "group_id": "unmanaged:LEGACY",
                "stop_price": "50",
                "close_at": current.open.isoformat(),
                "retire_before_research": True,
            },
        )
        value.service._commands(value.now)
        assert value.store.projection("command_results", command["id"])["state"] == "COMPLETED"
        value.ready()
        assert value.store.projection("position_groups", "unmanaged:LEGACY")[
            "retire_before_research"
        ]
        assert not value.service.entries_ready
        value.service.run_once(task="PostGauss")
        assert value.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0

        value.now = current.open
        value.ready()
        value.service.evidence.ingest_market(
            MarketEvent(
                symbol="LEGACY",
                source="offline",
                source_version="legacy-exit-quote",
                event_type="quote",
                effective_event_time=value.now,
                received_at=value.now,
                available_at=value.now,
                feed="sip",
                quality_class="genuine",
                bid="100",
                ask="100.01",
            )
        )
        value.service.run_once(task="PostGauss")
        assert (
            len(value.broker.writes) == 1
        ), "retirement supervision continues while research waits"
        assert value.broker.writes[0][1].purpose == "close"
        assert value.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0

        value.broker.holdings = ()
        value.service.run_once(task="PostGauss")
        group = value.store.projection("position_groups", "unmanaged:LEGACY")
        assert group["legs"] == [] and group["unresolved_orders"]
        assert not value.service.entries_ready
        assert value.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        value.broker.rows[0].update(
            status="filled",
            filled_qty="100",
            filled_avg_price="100",
            filled_at=value.now.isoformat(),
        )
        value.service.run_once(task="PostGauss")
        value.finish()
        assert value.store.projection("position_groups", "unmanaged:LEGACY")["state"] == "CLOSED"
        assert value.service._research_holdings_ready()
        value.service.entries_paused = True
        assert not value.service.entries_ready, "flat holdings must not override an operator pause"
        value.service.run_once(task="CloseGauss")
        value.finish()
        jobs = value.store.db.execute(
            "SELECT id,state FROM jobs WHERE id LIKE ?", (value.scope + ":%",)
        ).fetchall()
        assert len(jobs) == 2 and all(row["state"] == "COMPLETED" for row in jobs)
        assert {call["role"] for call in calls} == {"PostGauss", "CloseGauss"}
        review = value.store.list("session_reviews", value.scope)[0]
        snapshot = value.store.get("snapshots", review["snapshot_id"])
        assert datetime.fromisoformat(snapshot["data_context"]["signal_as_of"]) == previous.close
        account = value.store.get("account_profiles", snapshot["account_profile_id"])
        assert datetime.fromisoformat(account["observed_at"]) == current.open
        plans = value.store.list("plans", value.scope)
        assert plans and all(plan["target_session_id"] == current.id for plan in plans)
        count = len(calls)
        value.service.run_once(task="PostGauss")
        value.service.run_once(task="CloseGauss")
        value.finish()
        assert len(calls) == count, "completed jobs remain immutable and are never reset"
    finally:
        value.close()


@pytest.mark.parametrize(
    "environment,retire,reason",
    [
        ("live", True, "paper account"),
        ("paper", "true", "boolean"),
    ],
)
def test_retirement_adoption_requires_boolean_and_paper_account(
    tmp_path, monkeypatch, environment, retire, reason
):
    now = datetime(2026, 9, 14, 13, 30, tzinfo=UTC)
    config = RuntimeConfig(
        database_path=str(tmp_path / "adoption.sqlite3"),
        account_id="offline-adoption",
        environment=environment,
    )
    service = SessionService(config, clock=lambda: now)
    group = PositionGroup(
        account_id=config.account_id,
        environment=environment,
        exit_policy_id="manual-adoption-required",
        state="RECONCILIATION_REQUIRED",
        legs=(
            PositionLeg(
                account_id=config.account_id,
                environment=environment,
                group_id="unmanaged:LEGACY",
                symbol="LEGACY",
                asset_type="stock",
                quantity="1",
                reconciled_at=now,
            ),
        ),
    )
    service.store.project("position_groups", group.id, group.model_dump(mode="json"), service.scope)
    monkeypatch.setenv(config.control_token_env, "offline-adoption-secret")
    try:
        command = SessionClient(
            config.database_path, account_id=config.account_id, environment=environment
        ).command(
            "adopt_position",
            confirmed=True,
            token="offline-adoption-secret",
            payload={
                "group_id": group.id,
                "stop_price": "50",
                "close_at": now.isoformat(),
                "retire_before_research": retire,
            },
        )
        service._commands(now)
        result = service.store.projection("command_results", command["id"])
        assert result["state"] == "REJECTED" and reason in result["reason"]
        assert (
            service.store.projection("position_groups", group.id)["exit_policy_id"]
            == "manual-adoption-required"
        )
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


@pytest.mark.parametrize("stage", ["PostGauss", "CloseGauss"])
def test_research_stage_requires_ai_support_before_plan_publication(scenario, model, stage):
    calls, outcomes = model
    outcomes[stage] = "INSUFFICIENT_EVIDENCE"
    assert scenario.research() == []
    assert stage in {c["role"] for c in calls}
    assert scenario.broker.writes == []


@pytest.mark.parametrize("stage", ["PreGauss", "LiveGauss"])
@pytest.mark.parametrize("result", ["COUNTER_EVIDENCE", "INSUFFICIENT_EVIDENCE", "FAILED"])
def test_failed_or_unsupportive_current_review_prevents_entry(scenario, model, stage, result):
    _, outcomes = model
    outcomes[stage] = CodexCLIError("CODEX_TIMEOUT") if result == "FAILED" else result
    plans = scenario.research()
    scenario.advance(3)
    pre = PreGauss()
    pre.run(scenario.service, scenario.sessions[1], scenario.now)
    scenario.finish()
    pre.run(scenario.service, scenario.sessions[1], scenario.now)
    if stage == "PreGauss":
        assert scenario.store.projection("plan_state", plans[0].id)["state"] == "DEFERRED"
    scenario.advance(4)
    live = LiveGauss()
    live.run(scenario.service, scenario.sessions[1], scenario.now)
    scenario.finish()
    live.run(scenario.service, scenario.sessions[1], scenario.now)
    assert scenario.broker.writes == []


def test_candidate_context_changes_research_cache_without_cross_symbol_reuse(tmp_path, model):
    calls, _ = model
    value = Scenario(tmp_path, symbols=("TEST_STOCK", "TEST_BLOCKED"))
    try:
        plans = value.research()
        assert len(plans) == 2
        close_calls = [c for c in calls if c["role"] == "CloseGauss"]
        assert len(close_calls) == 2
        assert {tuple(c["decision_context"]["symbols"]) for c in close_calls} == {
            ("TEST_STOCK",),
            ("TEST_BLOCKED",),
        }
        assert all(
            {e["symbol"] for e in c["evidence"]} == set(c["decision_context"]["symbols"])
            for c in close_calls
        )
        reports = value.store.list("research_reports", value.scope)
        assert len({r["annotation"]["input_hash"] for r in reports}) == 2
        CloseGauss().run(value.service, value.sessions[0], value.now)
        assert len([c for c in calls if c["role"] == "CloseGauss"]) == 2
    finally:
        value.close()


def test_quota_latch_prevents_new_research_even_if_provider_would_now_allow_it(scenario, model):
    calls, _ = model
    account = scenario.service.account_profile
    snapshot = scenario.service.evidence.freeze(scenario.sessions[0].id, account, scenario.now)
    record_codex_quota(scenario.store, scenario.scope, quota(95), now=scenario.now)
    outcome = ResearchJobRunner(
        scenario.store, scenario.service.config, scenario.scope, clock=lambda: scenario.now
    ).run(
        snapshot,
        account,
        "PostGauss",
        scenario.sessions[0].id,
        decision_context={"symbols": ["TEST_STOCK"]},
    )
    assert outcome.state == "BLOCKED"
    assert calls == []
    record_codex_quota(scenario.store, scenario.scope, quota(10), now=scenario.now)
    assert scenario.service.monitor_only, "healthy reading cannot silently clear exhaustion latch"


def test_nightly_post_snapshot_stays_at_regular_close_despite_later_or_weekend_bars(scenario):
    from src.runtime.evidence import SnapshotDataReader

    previous = scenario.sessions[0]
    PostGauss().run(scenario.service, previous, scenario.now)
    baseline = scenario.store.list("session_reviews", scenario.scope)[0]
    baseline_candidate = scenario.store.get("candidates", baseline["candidates"][0])
    scenario.advance(4)  # The next session is now trading with different observations.
    contaminated = []
    for name, stamp in (
        ("after-close", previous.close + timedelta(minutes=1)),
        ("pre-open", previous.open - timedelta(minutes=1)),
        ("weekend", previous.open - timedelta(days=2)),
        ("later-session", scenario.sessions[1].open + timedelta(minutes=1)),
    ):
        event = MarketEvent(
            id="contamination-" + name,
            symbol="TEST_STOCK",
            source="offline",
            source_version=name,
            event_type="bar",
            effective_event_time=stamp,
            interval_start=stamp - timedelta(minutes=1),
            received_at=scenario.now,
            available_at=scenario.now,
            feed="sip",
            quality_class="genuine",
            price="9000",
            open_price="8990",
            high_price="9010",
            low_price="8980",
            volume="10000",
        )
        contaminated.append(event.id)
        scenario.service.evidence.ingest_market(event)
    PostGauss().run(scenario.service, previous, scenario.now)
    review = scenario.store.list("session_reviews", scenario.scope)[0]
    reader = SnapshotDataReader(
        scenario.store, scenario.store.get("snapshots", review["snapshot_id"])
    )
    assert reader.snapshot.data_context.signal_as_of == previous.close
    assert not set(contaminated).intersection(reader.snapshot.evidence_ids)
    assert all(event.effective_event_time <= previous.close for event in reader.market())
    assert reader.bars("TEST_STOCK").index.max() == previous.close
    candidate = scenario.store.get("candidates", review["candidates"][0])
    for field in ("entry_price", "stop_price", "take_profit_price"):
        assert candidate["alternative"][field] == baseline_candidate["alternative"][field]


@pytest.mark.parametrize("role", ["PreGauss", "LiveGauss"])
def test_slow_current_review_does_not_block_broker_reconciliation(scenario, monkeypatch, role):
    if role == "PreGauss":
        plans = scenario.research()
        scenario.advance(3)
    else:
        plans = scenario.prepare_live()
    import src.runtime.research as research

    original = research.review_stage
    started, release, cycle_finished = Event(), Event(), Event()
    failures = []

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(5), "test must release the deliberately slow review"
        return original(*args, **kwargs)

    monkeypatch.setattr(research, "review_stage", slow)
    assert (
        scenario.service.request_ai_review(plans[0], scenario.sessions[1], role, scenario.now)
        is None
    )
    assert started.wait(2)
    future = next(iter(scenario.service._ai_futures.values()))
    assert not future.done()
    scenario.broker.holdings = (
        PositionLeg(
            account_id=scenario.service.config.account_id,
            environment="paper",
            group_id="unmanaged:EXISTING",
            symbol="EXISTING",
            asset_type="stock",
            quantity="2",
            cost_basis="20",
            reconciled_at=scenario.now,
        ),
    )
    reads = scenario.broker.reads

    def cycle():
        try:
            scenario.service.run_once(task="LiveGauss")
        except Exception as exc:
            failures.append(exc)
        finally:
            cycle_finished.set()

    from threading import Thread

    thread = Thread(target=cycle, daemon=True)
    thread.start()
    try:
        assert cycle_finished.wait(2), "a model review must not block the operational cycle"
        assert not failures
        assert not future.done(), "cycle must finish while the model still waits"
        assert scenario.broker.reads > reads
        assert any(
            leg["symbol"] == "EXISTING" and Decimal(leg["quantity"]) == 2
            for group in scenario.store.projected("position_groups", scenario.scope)
            for leg in group["legs"]
        )
        assert scenario.broker.writes == []
    finally:
        release.set()
        thread.join(5)
        scenario.finish()


def test_expired_live_support_requires_new_review_before_entry(scenario, model):
    calls, outcomes = model
    plan = scenario.prepare_live()[0]
    assert (
        scenario.service.request_ai_review(plan, scenario.sessions[1], "LiveGauss", scenario.now)
        is None
    )
    scenario.finish()
    assert (
        scenario.service.request_ai_review(
            plan, scenario.sessions[1], "LiveGauss", scenario.now
        ).annotation.outcome
        == "SUPPORT"
    )
    interval = scenario.service.config.ai_review_interval_seconds
    scenario.now = datetime.fromtimestamp(
        (int(scenario.now.timestamp() // interval) + 1) * interval, UTC
    )
    scenario.ready()
    scenario.service.evidence.ingest_market(
        MarketEvent(
            symbol=plan.alternative.instrument.symbol,
            source="offline-renewed-quote",
            source_version=scenario.now.isoformat(),
            event_type="quote",
            effective_event_time=scenario.now,
            received_at=scenario.now,
            available_at=scenario.now,
            feed="sip",
            quality_class="genuine",
            bid="100.05",
            ask="100.06",
        )
    )
    outcomes["LiveGauss"] = CodexCLIError("CODEX_TIMEOUT")
    LiveGauss().run(scenario.service, scenario.sessions[1], scenario.now)
    assert scenario.broker.writes == []
    scenario.finish()
    LiveGauss().run(scenario.service, scenario.sessions[1], scenario.now)
    assert scenario.broker.writes == []
    assert len([call for call in calls if call["role"] == "LiveGauss"]) == 2


def test_new_symbol_news_invalidates_cached_live_support(scenario, model):
    calls, _ = model
    plan = scenario.prepare_live()[0]
    scenario.service.request_ai_review(plan, scenario.sessions[1], "LiveGauss", scenario.now)
    scenario.finish()
    assert (
        scenario.service.request_ai_review(
            plan, scenario.sessions[1], "LiveGauss", scenario.now
        ).annotation.outcome
        == "SUPPORT"
    )
    scenario.now += timedelta(seconds=1)
    scenario.service.evidence.ingest_news(
        NewsEvent(
            provider="offline",
            article_id="changed-symbol-context",
            source_version="v1",
            published_at=scenario.now,
            updated_at=scenario.now,
            received_at=scenario.now,
            available_at=scenario.now,
            symbols=("TEST_STOCK",),
            headline="New synthetic evidence",
            content_hash="offline-only",
            blocking=False,
            verified=True,
        )
    )
    assert (
        scenario.service.request_ai_review(plan, scenario.sessions[1], "LiveGauss", scenario.now)
        is None
    )
    scenario.finish()
    assert len([call for call in calls if call["role"] == "LiveGauss"]) == 2
    assert scenario.broker.writes == []


@pytest.mark.parametrize("used_percent", [95, 100])
def test_authoritative_quota_stop_keeps_fill_monitoring_and_blocks_all_actions(
    scenario, model, monkeypatch, used_percent
):
    calls, _ = model
    scenario.prepare_live()
    LiveGauss().run(scenario.service, scenario.sessions[1], scenario.now)
    scenario.finish()
    LiveGauss().run(scenario.service, scenario.sessions[1], scenario.now)
    assert len(scenario.broker.writes) == 1
    analysis_count = len(calls)
    monkeypatch.setattr("src.llm.codex_cli.read_codex_usage", lambda **kwargs: quota(used_percent))
    scenario.service._quota_checked_at = None
    scenario.service._refresh_codex_quota(scenario.now)
    scenario.service._quota_future.result(timeout=3)
    scenario.service._refresh_codex_quota(scenario.now)
    assert scenario.service.monitor_only
    control = scenario.store.projection("automation_control", scenario.scope)
    assert control["reason"] == "CODEX_QUOTA_MONITOR_ONLY"
    assert control["remaining_percent"] <= 5

    # The order existed before the quota stop, and may still fill at the broker.
    scenario.broker.rows[0].update(
        status="filled",
        filled_qty="1",
        filled_avg_price="100.06",
        filled_at=scenario.now.isoformat(),
    )
    scenario.broker.holdings = (
        PositionLeg(
            account_id=scenario.service.config.account_id,
            environment="paper",
            group_id="unmanaged:TEST_STOCK",
            symbol="TEST_STOCK",
            asset_type="stock",
            quantity="1",
            cost_basis="100.06",
            reconciled_at=scenario.now,
        ),
    )
    reads = scenario.broker.reads
    scenario.now += timedelta(seconds=1)
    scenario.service.run_once()
    scenario.finish()
    assert scenario.broker.reads > reads
    assert len(scenario.store.list("fills", scenario.scope)) == 1
    assert any(
        Decimal(leg["quantity"]) == 1
        for group in scenario.store.projected("position_groups", scenario.scope)
        for leg in group["legs"]
    )
    assert len(calls) == analysis_count
    assert len(scenario.broker.writes) == 1
    shutdown = scenario.service.request_shutdown()
    assert shutdown["stopped"] is False, "monitoring stays active while exposure remains"
    assert len(scenario.broker.writes) == 1
