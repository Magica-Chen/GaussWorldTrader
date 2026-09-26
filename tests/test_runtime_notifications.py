"""Offline notification contracts; all transports are injected doubles."""

import threading
from datetime import UTC, datetime, timedelta

import pytest

from src.runtime.notifications import RuntimeNotifications
from src.runtime.store import Store

SCOPE = "paper:notification-test"


class Clock:
    now = datetime(2026, 9, 28, 15, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class Transport:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []

    def send(self, subject, message):
        self.calls.append((subject, message))
        outcome = self.outcomes.pop(0) if self.outcomes else True
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "notifications.sqlite3"
    store, clock, clients = Store(path), Clock(), []

    def make(providers, scope=SCOPE):
        client = RuntimeNotifications(path, scope, providers=providers, clock=clock)
        clients.append(client)
        return client

    yield store, clock, make
    for client in clients:
        client.close()
    store.close()


def decision(store, identifier="decision-1", scope=SCOPE):
    store.put(
        "decisions",
        {
            "id": identifier,
            "plan_id": "plan-1",
            "outcome": "NO_TRADE",
            "reason": "RUNTIME_NOT_READY",
        },
        scope,
    )


def order(store, state="ACKNOWLEDGED", identifier="entry-1", broker_id="broker-1", scope=SCOPE):
    intent = {
        "id": identifier,
        "purpose": "open",
        "order_type": "limit",
        "limit_price": "100",
        "legs": [{"side": "buy", "instrument": {"symbol": "AAPL"}}],
    }
    store.put("order_intents", intent, scope)
    body = {"id": identifier, "intent_id": identifier, "quantity": "10", "state": state}
    if broker_id:
        body["broker_id"] = broker_id
    store.project("orders", identifier, body, scope)


def fill(store, cumulative, delta, identifier=None):
    store.put(
        "fills",
        {
            "id": identifier or f"broker-1:{cumulative}",
            "order_id": "broker-1",
            "local_order_id": "entry-1",
            "intent_id": "entry-1",
            "quantity": str(delta),
            "cumulative_quantity": str(cumulative),
            "price": "99.50",
        },
        SCOPE,
    )


def test_disabled_channels_never_capture_or_send(setup, monkeypatch):
    store, clock, make = setup
    monkeypatch.setenv("NOTIFICATION_EMAIL_ENABLED", "false")
    monkeypatch.setenv("NOTIFICATION_SLACK_ENABLED", "false")
    client = make(None)
    decision(store)
    assert client.capture() == 0
    assert client.dispatch_once() == 0
    assert client.status() == {
        "channels": [],
        "missing_channels": [],
        "enabled": False,
        "pending": 0,
        "retrying": 0,
        "delivered": 0,
        "worker_error": None,
    }


def test_selected_slack_channel_excludes_configured_email(setup, monkeypatch):
    store, clock, make = setup
    slack, email = Transport(), Transport()
    monkeypatch.setattr(
        "src.runtime.notifications.configured_providers", lambda: {"slack": slack, "email": email}
    )
    client = RuntimeNotifications(store.path, SCOPE, channels=("slack",), clock=clock)
    try:
        decision(store)
        client.capture()
        assert client.dispatch_once() == 1
        assert client.status()["channels"] == ["slack"]
        assert email.calls == []
    finally:
        client.close()


def test_requested_unconfigured_channel_is_reported_for_readiness(setup, monkeypatch):
    store, clock, make = setup
    monkeypatch.setattr("src.runtime.notifications.configured_providers", lambda: {})
    client = RuntimeNotifications(store.path, SCOPE, channels=("slack",), clock=clock)
    try:
        assert client.status()["missing_channels"] == ["slack"]
        assert not client.status()["enabled"]
    finally:
        client.close()


def test_ai_decision_and_quota_monitor_only_incident_are_explained(setup):
    store, clock, make = setup
    transport = Transport()
    client = make({"slack": transport})
    store.put(
        "ai_decisions",
        {
            "id": "codex-pre-1",
            "role": "PreGauss",
            "symbol": "AAPL",
            "outcome": "NO_TRADE",
            "summary": "Defer entry because the earnings event is unresolved.",
            "result": {"private_internal": "never forward this"},
        },
        SCOPE,
    )
    store.put(
        "incidents",
        {
            "id": "quota-1",
            "reason": "CODEX_QUOTA_MONITOR_ONLY",
            "remaining_percent": 5,
        },
        SCOPE,
    )
    client.capture()
    assert client.dispatch_once() == 2
    messages = "\n".join(message for _, message in transport.calls)
    assert "earnings event" in messages
    assert "monitoring continue" in messages
    assert "Remaining percent: 5" in messages
    assert "never forward this" not in messages


def test_four_stage_decisions_are_durable_without_changing_risk_watermark(setup):
    store, clock, make = setup
    transport = Transport()
    client = make({"slack": transport})
    store.emit("submit_intent", {"intent_id": "existing"}, SCOPE)
    watermark = store.watermark(SCOPE)
    store.put("session_reviews", {"id": "post", "outcome": "NO_TRADE", "candidates": []}, SCOPE)
    store.put("research_reports", {"id": "close", "outcome": "PLAN"}, SCOPE)
    store.put("plan_events", {"id": "pre", "plan_id": "plan-1", "state": "ELIGIBLE"}, SCOPE)
    store.put("risk_decisions", {"id": "live", "approved": True, "approved_quantity": "10"}, SCOPE)
    assert client.capture() == 4
    assert transport.calls == []
    assert store.watermark(SCOPE) == watermark
    assert client.dispatch_once() == 4
    assert store.watermark(SCOPE) == watermark
    assert {
        next(line for line in message.splitlines() if line.startswith("Role:"))
        for _, message in transport.calls
    } == {"Role: PostGauss", "Role: CloseGauss", "Role: PreGauss", "Role: LiveGauss"}
    assert len(store.list("notification_receipts", SCOPE)) == 4
    client.capture()
    assert client.dispatch_once() == 0


@pytest.mark.parametrize("state", ["SUBMITTING", "UNKNOWN", "SHADOW", "REJECTED"])
def test_unconfirmed_or_simulated_orders_are_not_announced_as_submitted(setup, state):
    store, clock, make = setup
    transport = Transport()
    client = make({"slack": transport})
    order(store, state, broker_id=None)
    store.emit("submit_intent", {"intent_id": "entry-1"}, SCOPE)
    client.capture()
    assert client.dispatch_once() == 0
    assert transport.calls == []


def test_reconciliation_ack_and_cumulative_partial_fills_send_once(setup):
    store, clock, make = setup
    transport = Transport()
    client = make({"email": transport})
    order(store, "UNKNOWN", broker_id=None)
    client.capture()
    order(store, "FILLED")
    fill(store, 4, 4)
    fill(store, 10, 6)
    client.capture()
    assert client.dispatch_once() == 3
    subjects = [subject for subject, _ in transport.calls]
    assert "SUBMITTED" in subjects[0]
    assert "PARTIAL_FILL" in subjects[1]
    assert "FILLED:" in subjects[2]
    assert sum("PARTIAL_FILL" in subject for subject in subjects) == 1
    assert sum("FILLED:" in subject for subject in subjects) == 1
    assert all("AAPL" in subject for subject in subjects)
    assert any("Fill quantity: 6" in message for _, message in transport.calls)
    fill(store, 10, 6)
    client.capture()
    assert client.dispatch_once() == 0


def test_failed_channel_retries_after_restart_without_resending_success(setup):
    store, clock, make = setup
    slack, email = Transport(), Transport([RuntimeError("secret webhook URL"), True])
    client = make({"slack": slack, "email": email})
    decision(store)
    client.capture()
    assert client.dispatch_once() == 1
    rows = store.projected("notification_deliveries", SCOPE)
    assert next(row for row in rows if row["channel"] == "email")["last_error"] == "RuntimeError"
    assert "secret webhook" not in str(rows)
    client.close()
    restarted = make({"slack": slack, "email": email})
    assert restarted.dispatch_once() == 0
    clock.advance(5)
    assert restarted.dispatch_once() == 1
    assert len(slack.calls) == 1
    assert len(email.calls) == 2
    assert restarted.status()["pending"] == 0


def test_false_transport_result_is_not_a_delivery_receipt(setup):
    store, clock, make = setup
    client = make({"slack": Transport([False])})
    decision(store)
    client.capture()
    assert client.dispatch_once() == 0
    assert store.list("notification_receipts", SCOPE) == []
    assert client.status()["retrying"] == 1


def test_activation_baselines_history_but_restart_keeps_queued_events(setup):
    store, clock, make = setup
    decision(store, "old")
    order(store, identifier="old-order", broker_id="old-broker")
    slack = Transport()
    client = make({"slack": slack})
    client.capture()
    assert client.dispatch_once() == 0
    decision(store, "new")
    client.capture()
    client.close()
    email = Transport()
    restarted = make({"slack": slack, "email": email})
    restarted.capture()
    assert restarted.dispatch_once() == 1
    assert len(slack.calls) == 1
    assert email.calls == []


def test_accounts_do_not_receive_each_others_decisions_or_orders(setup):
    store, clock, make = setup
    transport = Transport()
    client = make({"slack": transport})
    decision(store, scope="paper:other-account")
    order(store, scope="paper:other-account")
    client.capture()
    assert client.dispatch_once() == 0


def test_crashed_inflight_delivery_is_reclaimed_only_after_lease(setup):
    store, clock, make = setup
    transport = Transport()
    client = make({"slack": transport})
    decision(store)
    client.capture()
    pending = store.projected("notification_deliveries", SCOPE)[0]
    pending.update(
        state="SENDING",
        token="dead-worker",
        attempts=1,
        lease_expires_at=(clock() + timedelta(seconds=300)).isoformat(),
    )
    store.project("notification_deliveries", pending["id"], pending, SCOPE)
    client.close()
    restarted = make({"slack": transport})
    assert restarted.dispatch_once() == 0
    clock.advance(301)
    assert restarted.dispatch_once() == 1
    assert store.projected("notification_deliveries", SCOPE)[0]["attempts"] == 2


def test_capture_failure_rolls_back_delivery_queue_and_cursor(setup, monkeypatch):
    store, clock, make = setup
    client = make({"slack": Transport()})
    decision(store)
    original = client._enqueue

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interrupted capture")

    monkeypatch.setattr(client, "_enqueue", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        client.capture()
    assert store.projected("notification_deliveries", SCOPE) == []
    monkeypatch.setattr(client, "_enqueue", original)
    assert client.capture() == 1
    assert client.dispatch_once() == 1


def test_market_data_volume_cannot_delay_decision_notifications(setup):
    store, clock, make = setup
    client = make({"slack": Transport()})
    for number in range(20):
        store.put("market_events", {"id": str(number), "symbol": "AAPL"}, SCOPE)
    decision(store)
    assert client.capture(limit=1) == 1
    assert client.dispatch_once() == 1


def test_slow_transport_does_not_hold_database_lock_or_block_shutdown(setup):
    store, clock, make = setup
    started, release = threading.Event(), threading.Event()

    class Slow:
        def send(self, subject, message):
            started.set()
            assert release.wait(5)
            return True

    client = make({"slack": Slow()})
    decision(store)
    client.capture()
    client.start()
    try:
        assert started.wait(2)
        store.db.execute("PRAGMA busy_timeout=100")
        store.put("incidents", {"id": "concurrent", "reason": "INDEPENDENT_SUPERVISION"}, SCOPE)
        assert client.close(timeout=0.01) is False
    finally:
        release.set()
        client._thread.join(2)
    assert not client._thread.is_alive()
    assert len(store.list("notification_receipts", SCOPE)) == 1


def test_repeating_incidents_are_throttled_without_losing_first_alert(setup):
    store, clock, make = setup
    transport = Transport()
    client = make({"slack": transport})
    for i in range(3):
        store.put("incidents", {"id": str(i), "reason": "BROKER_DOWN"}, SCOPE)
    client.capture()
    assert client.dispatch_once() == 1
    clock.advance(900)
    store.put("incidents", {"id": "later", "reason": "BROKER_DOWN"}, SCOPE)
    client.capture()
    assert client.dispatch_once() == 1
