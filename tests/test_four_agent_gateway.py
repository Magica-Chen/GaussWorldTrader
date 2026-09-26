"""Broker-double tests for current risk reservations and restart semantics."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Event

import pytest

from src.runtime.calendar import SessionCalendar
from src.runtime.evidence import EvidenceService
from src.runtime.models import (
    AccountProfile,
    Alternative,
    Instrument,
    MarketEvent,
    AccountOperationalState,
    OrderIntent,
    OrderLeg,
    PositionGroup,
    PositionLeg,
    RiskPolicy,
    Rule,
    RuntimeConfig,
    SessionPlan,
    StrategyApproval,
    TradingSession,
)
from src.runtime.risk import ExecutionGateway, Reconciler, RiskGate
from src.runtime.store import Store


NOW = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)


class BrokerDouble:
    def __init__(self):
        self.writes = []
        self.rows = []
        self.holdings = ()
        self.timeout = False

    def account(self, policy, now):
        return AccountProfile(
            account_id="fixture-account",
            environment="paper",
            equity="10000",
            cash="10000",
            buying_power="10000",
            currency="USD",
            observed_at=now,
            mandate_id=policy.id,
            daily_pnl="0",
        )

    def submit(self, intent, quantity, client_id):
        self.writes.append((intent, quantity, client_id))
        row = {
            "id": "broker-" + intent.id,
            "client_order_id": client_id,
            "status": "new",
            "filled_qty": "0",
        }
        self.rows.append(row)
        if self.timeout:
            raise TimeoutError("accepted then connection dropped")
        return row

    def orders(self):
        return self.rows

    def positions(self, account_id, now):
        return self.holdings

    def activities(self):
        return []

    def order_by_client_id(self, client_id):
        return next((row for row in self.rows if row["client_order_id"] == client_id), None)

    def replace(self, broker_id, *, limit_price, client_order_id):
        parent = next(row for row in self.rows if row['id'] == broker_id)
        child = {
            'id': 'broker-' + client_order_id,
            'client_order_id': client_order_id,
            'status': 'new',
            'filled_qty': '0',
            'replaces': broker_id,
        }
        parent.update(status='replaced', replaced_by=child['id'])
        self.rows.append(child)
        self.writes.append(('replace', limit_price, client_order_id))
        if self.timeout:
            raise TimeoutError('replacement accepted then timeout')
        return child


def setup(tmp_path, *, mode="shadow", maximum_groups=5):
    policy = RiskPolicy(max_open_position_groups=maximum_groups)
    config = RuntimeConfig(
        account_id="fixture-account",
        execution_mode=mode,
        policy=policy,
        symbols=("TEST_STOCK",),
        strategy_allowlist=("fixture",),
    )
    store = Store(tmp_path / "session.db")
    scope = "paper:fixture-account"
    session = TradingSession(
        id="fixture-session",
        session_date="2026-09-08",
        calendar_version="fixture-v1",
        open=NOW.replace(hour=13, minute=30),
        close=NOW.replace(hour=20, minute=0),
    )
    evidence = EvidenceService(store, config, scope)
    calendar = SessionCalendar((session,))
    store.put('sessions', session, scope)
    broker = BrokerDouble()
    return store, config, scope, evidence, calendar, broker


def seed_plan(
    store,
    config,
    scope,
    evidence,
    symbol="TEST_STOCK",
    take_profit=None,
    approval_changes=None,
    expected_net_value="1",
):
    instrument = Instrument(symbol=symbol, liquidity_capacity="3")
    alternative = Alternative(
        strategy_id="fixture",
        strategy_version="v1",
        instrument=instrument,
        entry_price="100",
        stop_price="99",
        take_profit_price=take_profit,
        expected_net_value=expected_net_value,
        evidence_ids=("fixture-source",),
    )
    approved = StrategyApproval(
        strategy_id="fixture",
        strategy_version="v1",
        account_id=config.account_id,
        environment="paper",
        approved_by="fixture-operator",
        profiles=("FREE_DELAYED",),
        execution_modes=("shadow", "paper", "replay"),
        requires_current_quote=False,
        maximum_signal_latency_seconds=1200,
        minimum_history=2,
        requires_news=False,
        requires_event_calendar=False,
        validation_reference="fixture-only",
        validated_net_expectancy="1",
        expiry=NOW + timedelta(days=1),
    )
    if approval_changes:
        approved = StrategyApproval.model_validate({**approved.model_dump(), **approval_changes})
    plan = SessionPlan(
        target_session_id="fixture-session",
        snapshot_id="fixture-snapshot",
        candidate_id="fixture-candidate",
        account_id=config.account_id,
        environment="paper",
        account_profile_id="fixture-account-profile",
        suitability_report_id="fixture-suitability",
        data_profile=config.data_profile,
        data_policy_version=config.data_policy_version,
        signal_as_of=NOW - timedelta(seconds=902),
        strategy_id="fixture",
        strategy_version="v1",
        approval_id=approved.id,
        alternative=alternative,
        entry_rules=(Rule(rule_type="completed_bar_condition", operator="above", value="99"),),
        max_buy_price="100.10",
        valid_from=NOW - timedelta(hours=1),
        entry_expires_at=NOW + timedelta(hours=3),
        close_at=NOW + timedelta(hours=4),
        risk_policy_id=config.policy.id,
        exit_policy_id="fixed-exit-v1",
    )
    store.put("strategy_approvals", approved, scope)
    store.put("plans", plan, scope)
    for state in ("RESEARCH_COMPLETE", "PENDING_VALIDATION", "ELIGIBLE"):
        store.transition_plan(plan.id, state, scope=scope)
    evidence.ingest_market(
        MarketEvent(
            symbol=symbol,
            source="fixture",
            source_version="1",
            event_type="bar",
            effective_event_time=NOW - timedelta(seconds=902),
            received_at=NOW,
            available_at=NOW,
            interval_start=NOW - timedelta(seconds=962),
            feed="sip",
            quality_class="genuine",
            price="100",
        )
    )
    intent = OrderIntent(
        account_id=config.account_id,
        environment="paper",
        purpose="open",
        plan_id=plan.id,
        plan_version=1,
        legs=(OrderLeg(instrument=instrument, side="buy", position_intent="buy_to_open"),),
        limit_price="100",
        maximum_buy_price="100.10",
        expires_at=NOW + timedelta(minutes=1),
        reason="synthetic",
    )
    return plan, intent


def approve(gate, intent, broker, config):
    return gate.approve(
        intent,
        broker.account(config.policy, NOW),
        NOW,
        entry_ready=True,
        news_ready=True,
        event_calendar_ready=True,
    )


def test_t03_shadow_gateway_never_calls_broker(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path)
    plan, intent = seed_plan(store, config, scope, evidence)
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = approve(gate, intent, broker, config)
    assert decision.approved, decision.reasons
    row = ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    assert row["state"] == "SHADOW"
    assert broker.writes == []
    store.close()


def test_t13_timeout_reconciles_same_client_without_duplicate_submission(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    plan, intent = seed_plan(store, config, scope, evidence)
    broker.timeout = True
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = approve(gate, intent, broker, config)
    assert decision.approved, decision.reasons
    gateway = ExecutionGateway(store, config, scope, broker, evidence)
    assert gateway.submit(intent, decision, NOW)["state"] == "UNKNOWN"
    store.close()
    reopened = Store(tmp_path / "session.db")
    Reconciler(reopened, config, scope, broker).reconcile(NOW + timedelta(seconds=1))
    assert reopened.projection("orders", intent.id)["state"] == "OPEN"
    ExecutionGateway(reopened, config, scope, broker, evidence).submit(
        intent, decision, NOW + timedelta(seconds=1)
    )
    assert len(broker.writes) == 1
    reopened.close()


def test_t14_concurrent_candidates_cannot_claim_same_position_slot(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, maximum_groups=1)
    intents = [
        seed_plan(store, config, scope, evidence, symbol)[1]
        for symbol in ("TEST_STOCK", "TEST_OTHER")
    ]
    gate = RiskGate(store, config, scope, evidence, calendar)
    with ThreadPoolExecutor(max_workers=2) as workers:
        decisions = list(workers.map(lambda intent: approve(gate, intent, broker, config), intents))
    assert sum(decision.approved for decision in decisions) == 1
    assert "POSITION_GROUP_LIMIT" in next(
        decision.reasons for decision in decisions if not decision.approved
    )
    store.close()


def test_t15_quantity_override_cannot_exceed_safe_size(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path)
    plan, intent = seed_plan(store, config, scope, evidence)
    intent = OrderIntent.model_validate({**intent.model_dump(), "requested_quantity": "10000"})
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    assert not decision.approved
    assert "REQUEST_EXCEEDS_SAFE_SIZE" in decision.reasons
    store.close()


def test_t12_news_between_risk_and_submission_revokes_approval(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    plan, intent = seed_plan(store, config, scope, evidence)
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    assert decision.approved, decision.reasons
    store.emit("critical_news", {"symbol": "TEST_STOCK"}, scope)
    result = ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    assert result["state"] == "REJECTED"
    assert broker.writes == []
    store.close()


def test_t16_partial_fill_remains_exposure_after_cancel(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    plan, intent = seed_plan(store, config, scope, evidence)
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    broker.rows[0].update(
        status="canceled", filled_qty="1", filled_avg_price="100", filled_at=NOW.isoformat()
    )
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment="paper",
            group_id="unmanaged",
            symbol="TEST_STOCK",
            asset_type="stock",
            quantity="1",
            cost_basis="100",
            reconciled_at=NOW,
        ),
    )
    Reconciler(store, config, scope, broker).reconcile(NOW)
    group = store.projection("position_groups", "group-" + intent.id)
    assert group["state"] == "OPEN"
    assert Decimal(group["legs"][0]["quantity"]) == 1
    assert (
        store.db.execute(
            "SELECT first_fill FROM reservations WHERE intent_id=?", (intent.id,)
        ).fetchone()[0]
        == 1
    )
    store.close()


def test_t12_late_news_after_submission_cancels_and_reconciles_fill_through_service(tmp_path):
    from src.runtime.models import NewsEvent
    from src.runtime.service import SessionService

    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    plan, intent = seed_plan(store, config, scope, evidence)
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    cancelled = []

    def cancel(broker_id):
        cancelled.append(broker_id)
        next(row for row in broker.rows if row["id"] == broker_id)["status"] = "canceled"

    broker.cancel = cancel
    received = NOW + timedelta(seconds=1)
    evidence.ingest_news(
        NewsEvent(
            provider="fixture",
            article_id="late-announcement",
            source_version="v1",
            published_at=received,
            updated_at=received,
            received_at=received,
            available_at=received,
            symbols=("TEST_STOCK",),
            headline="Synthetic trading halt",
            content_hash="fixture-halt",
            blocking=True,
            verified=True,
        )
    )
    service = SessionService(
        config.model_copy(update={"database_path": str(tmp_path / "session.db")}),
        broker=broker,
        calendar=calendar,
        clock=lambda: received,
    )
    try:
        service.run_once()
        assert service.store.projection("plan_state", plan.id)["state"] == "INVALIDATED"
        assert cancelled == [broker.rows[0]["id"]]
        assert service.store.projection("orders", intent.id)["state"] == "CANCEL_PENDING"
        assert service.evidence.news(received) == []  # current safety never becomes delayed signal
        assert (
            service.store.db.execute("SELECT state FROM reservations").fetchone()[0] != "RELEASED"
        )

        # The broker confirms a fill racing the cancellation on the next operational cycle.
        broker.rows[0].update(
            filled_qty="1", filled_avg_price="100", filled_at=received.isoformat()
        )
        broker.holdings = (
            PositionLeg(
                account_id=config.account_id,
                environment="paper",
                group_id="unmanaged:TEST_STOCK",
                symbol="TEST_STOCK",
                asset_type="stock",
                quantity="1",
                cost_basis="100",
                reconciled_at=received,
            ),
        )
        service.run_once()
        group = service.store.projection("position_groups", "group-" + intent.id)
        assert group["state"] == "OPEN"
        assert Decimal(group["legs"][0]["quantity"]) == 1
        assert service.store.db.execute("SELECT first_fill FROM reservations").fetchone()[0] == 1
        assert len(service.store.list("fills", scope)) == 1
        assert len(broker.writes) == 1
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
        store.close()


@pytest.mark.parametrize(
    "activity_type,option_quantity,stock_quantity",
    [
        ("OPEXC", "1", "100"),
        ("OPASN", "-1", "-100"),
    ],
)
def test_t29_assignment_or_exercise_reconciles_residual_stock(
    tmp_path, activity_type, option_quantity, stock_quantity
):
    store, config, scope, evidence, calendar, broker = setup(tmp_path)
    symbol = "TEST260918C00100000"
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment="paper",
            group_id="unmanaged:" + symbol,
            symbol=symbol,
            asset_type="option",
            quantity=option_quantity,
            multiplier="100",
            cost_basis="100",
            reconciled_at=NOW,
        ),
    )
    reconciler = Reconciler(store, config, scope, broker)
    reconciler.reconcile(NOW)
    assert (
        store.projection("position_groups", "unmanaged:" + symbol)["state"]
        == "RECONCILIATION_REQUIRED"
    )

    arrival = NOW + timedelta(seconds=1)
    activity = {
        "id": "fixture-" + activity_type,
        "activity_type": activity_type,
        "symbol": symbol,
        "qty": "1",
        "date": arrival.isoformat(),
    }
    broker.activities = lambda: [activity]
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment="paper",
            group_id="unmanaged:TEST",
            symbol="TEST",
            asset_type="stock",
            quantity=stock_quantity,
            cost_basis="10000",
            market_value="10100",
            reconciled_at=arrival,
        ),
    )
    account = reconciler.reconcile(arrival)
    reconciler.reconcile(arrival)
    assert store.projection("position_groups", "unmanaged:" + symbol)["state"] == "CLOSED"
    residual = store.projection("position_groups", "unmanaged:TEST")
    assert residual["state"] == "RECONCILIATION_REQUIRED"
    assert residual["exit_policy_id"] == "manual-adoption-required"
    assert Decimal(residual["legs"][0]["quantity"]) == Decimal(stock_quantity)
    assert not PositionGroup.model_validate(
        {key: value for key, value in residual.items() if key != "_revision"}
    ).flat
    assert account.exposure == Decimal("10100")
    assert len(store.list("activities", scope)) == 1
    assert any(
        row["reason"] == "UNMANAGED_OR_ASSIGNMENT_EXPOSURE" and row["symbol"] == "TEST"
        for row in store.list("incidents", scope)
    )
    assert broker.writes == []
    store.close()


def replacement_setup(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode='paper')
    plan, intent = seed_plan(store, config, scope, evidence)
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    gateway = ExecutionGateway(store, config, scope, broker, evidence)
    gateway.submit(intent, decision, NOW)
    account = Reconciler(store, config, scope, broker).reconcile(NOW)
    return store, config, scope, evidence, broker, plan, intent, gateway, account


def replace(gateway, order_id, account, price='100.05'):
    return gateway.replace(
        order_id,
        price,
        NOW,
        account=account,
        entry_ready=True,
        news_ready=True,
        event_calendar_ready=True,
    )


def test_t13_unknown_replacement_retains_single_reservation_and_stable_child(tmp_path):
    store, config, scope, evidence, broker, plan, intent, gateway, account = replacement_setup(
        tmp_path
    )
    broker.timeout = True
    child = replace(gateway, intent.id, account)
    assert child['state'] == 'UNKNOWN'
    assert replace(gateway, intent.id, account)['id'] == child['id']
    assert len(broker.writes) == 2  # one submit and one replace
    reservation = store.db.execute('SELECT * FROM reservations').fetchone()
    assert reservation['intent_id'] == intent.id and Decimal(reservation['capital']) == Decimal(
        '300.15'
    )
    assert store.db.execute('SELECT count(*) FROM reservations').fetchone()[0] == 1
    Reconciler(store, config, scope, broker).reconcile(NOW)
    assert store.projection('orders', child['id'])['state'] == 'OPEN'
    assert store.projection('orders', intent.id)['state'] == 'CANCELLED'
    assert store.db.execute('SELECT state FROM reservations').fetchone()[0] != 'RELEASED'
    store.close()


def test_t16_replacement_child_fill_counts_once_and_releases_only_pending_cash(tmp_path):
    store, config, scope, evidence, broker, plan, intent, gateway, account = replacement_setup(
        tmp_path
    )
    child = replace(gateway, intent.id, account)
    broker.rows[-1].update(
        status='canceled', filled_qty='1', filled_avg_price='100.05', filled_at=NOW.isoformat()
    )
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment='paper',
            group_id='unmanaged:TEST_STOCK',
            symbol='TEST_STOCK',
            asset_type='stock',
            quantity='1',
            cost_basis='100.05',
            reconciled_at=NOW,
        ),
    )
    reconciler = Reconciler(store, config, scope, broker)
    reconciler.reconcile(NOW)
    reconciler.reconcile(NOW)
    assert len(store.list('fills')) == 1
    assert store.projection('position_groups', 'group-' + intent.id)['state'] == 'OPEN'
    reservation = store.db.execute('SELECT * FROM reservations').fetchone()
    assert reservation['first_fill'] == 1 and reservation['state'] == 'EXPOSURE'
    assert Decimal(reservation['capital']) == 0
    store.close()


def test_t17_late_old_fill_with_rejected_replacement_retains_managed_exposure(tmp_path):
    store, config, scope, evidence, broker, plan, intent, gateway, account = replacement_setup(
        tmp_path
    )
    child = replace(gateway, intent.id, account)
    broker.rows[0].update(
        status='filled', filled_qty='3', filled_avg_price='100', filled_at=NOW.isoformat()
    )
    broker.rows[-1].update(status='rejected')
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment='paper',
            group_id='unmanaged:TEST_STOCK',
            symbol='TEST_STOCK',
            asset_type='stock',
            quantity='3',
            cost_basis='300',
            reconciled_at=NOW,
        ),
    )
    Reconciler(store, config, scope, broker).reconcile(NOW)
    assert store.projection('position_groups', 'group-' + intent.id)['state'] == 'OPEN'
    assert len(store.list('fills')) == 1
    assert store.db.execute('SELECT first_fill FROM reservations').fetchone()[0] == 1
    store.close()


def test_t15_forged_risk_decision_and_modified_intent_never_reach_broker(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode='paper')
    plan, intent = seed_plan(store, config, scope, evidence)
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    gateway = ExecutionGateway(store, config, scope, broker, evidence)
    with pytest.raises(ValueError, match='durable authority'):
        gateway.submit(
            intent, decision.model_copy(update={'approved_quantity': Decimal('99999')}), NOW
        )
    with pytest.raises(ValueError, match='durable authority'):
        gateway.submit(intent.model_copy(update={'limit_price': Decimal('1000')}), decision, NOW)
    assert broker.writes == []
    store.close()


def test_t26_closed_group_does_not_adopt_later_external_same_symbol_position(tmp_path):
    store, config, scope, evidence, broker, plan, intent, gateway, account = replacement_setup(
        tmp_path
    )
    broker.rows[0].update(
        status='filled', filled_qty='3', filled_avg_price='100', filled_at=NOW.isoformat()
    )
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment='paper',
            group_id='unmanaged:TEST_STOCK',
            symbol='TEST_STOCK',
            asset_type='stock',
            quantity='3',
            cost_basis='300',
            reconciled_at=NOW,
        ),
    )
    reconciler = Reconciler(store, config, scope, broker)
    reconciler.reconcile(NOW)
    broker.holdings = ()
    reconciler.reconcile(NOW)
    assert store.projection('position_groups', 'group-' + intent.id)['state'] == 'CLOSED'
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment='paper',
            group_id='unmanaged:TEST_STOCK',
            symbol='TEST_STOCK',
            asset_type='stock',
            quantity='1',
            cost_basis='100',
            reconciled_at=NOW,
        ),
    )
    reconciler.reconcile(NOW)
    assert store.projection('position_groups', 'group-' + intent.id)['state'] == 'CLOSED'
    assert (
        store.projection('position_groups', 'unmanaged:TEST_STOCK')['state']
        == 'RECONCILIATION_REQUIRED'
    )
    store.close()


def test_t30_reconciliation_lookup_happens_outside_account_transaction(tmp_path):
    store, config, scope, evidence, broker, plan, intent, gateway, account = replacement_setup(
        tmp_path
    )
    rows = broker.rows
    broker.orders = lambda: []

    def lookup(client_id):
        assert not store.db.in_transaction
        return rows[0]

    broker.order_by_client_id = lookup
    Reconciler(store, config, scope, broker).reconcile(NOW)
    store.close()


def test_t04_invalid_financial_data_preserves_operational_holdings_and_close_authority(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path)

    def invalid(*args):
        raise ValueError('non-finite equity')

    broker.account = invalid
    broker.identity = lambda: {'account_id': config.account_id, 'environment': 'paper'}
    position = PositionLeg(
        account_id=config.account_id,
        environment='paper',
        group_id='unmanaged:TEST_STOCK',
        symbol='TEST_STOCK',
        asset_type='stock',
        quantity='1',
        cost_basis='100',
        reconciled_at=NOW,
    )
    broker.holdings = (position,)
    actual = Reconciler(store, config, scope, broker).reconcile(NOW)
    assert isinstance(actual, AccountOperationalState)
    assert actual.positions[0].quantity == 1
    assert 'equity' not in actual.model_dump()
    evidence.ingest_market(
        MarketEvent(
            symbol='TEST_STOCK',
            source='fixture',
            source_version='exit-quote',
            event_type='quote',
            effective_event_time=NOW,
            received_at=NOW,
            available_at=NOW,
            feed='sip',
            quality_class='genuine',
            bid='100',
            ask='100.01',
        )
    )
    intent = OrderIntent(
        account_id=config.account_id,
        environment='paper',
        purpose='close',
        group_id=position.group_id,
        legs=(
            OrderLeg(
                instrument=Instrument(symbol='TEST_STOCK', liquidity_capacity='1'),
                side='sell',
                position_intent='sell_to_close',
            ),
        ),
        requested_quantity='1',
        limit_price='100',
        expires_at=NOW + timedelta(seconds=10),
        reason='confirmed protective exit',
    )
    result = RiskGate(store, config, scope, evidence, calendar).approve(
        intent, actual, NOW, entry_ready=False, news_ready=False, event_calendar_ready=False
    )
    assert result.approved, result.reasons
    store.close()


def test_t13_restart_rejects_reserved_work_that_never_reached_submitting(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode='paper')
    plan, intent = seed_plan(store, config, scope, evidence)
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    assert decision.approved
    store.close()
    store = Store(tmp_path / 'session.db')
    Reconciler(store, config, scope, broker).reconcile(NOW)
    order = store.projection('orders', intent.id)
    assert order['state'] == 'REJECTED' and order['reasons'] == ['PRE_SUBMISSION_RECOVERY']
    assert store.db.execute('SELECT state FROM reservations').fetchone()[0] == 'RELEASED'
    assert broker.writes == []
    store.close()


def test_replacements_are_bounded(tmp_path):
    store, config, scope, evidence, broker, plan, intent, gateway, account = replacement_setup(
        tmp_path
    )
    first = replace(gateway, intent.id, account)
    account = Reconciler(store, config, scope, broker).reconcile(NOW)
    second = replace(gateway, first['id'], account, '100.06')
    account = Reconciler(store, config, scope, broker).reconcile(NOW)
    with pytest.raises(ValueError, match='ATTEMPT_LIMIT'):
        replace(gateway, second['id'], account, '100.07')
    assert store.db.execute('SELECT count(*) FROM reservations').fetchone()[0] == 1
    assert len(broker.writes) == 3
    store.close()


def test_cancel_failure_on_parent_keeps_cancelling_replacement_child(tmp_path):
    store, config, scope, evidence, broker, plan, intent, gateway, account = replacement_setup(
        tmp_path
    )
    child = replace(gateway, intent.id, account)
    calls = []

    def cancel(broker_id):
        calls.append(broker_id)
        assert (
            store.projection(
                'orders', intent.id if broker_id == broker.rows[0]['id'] else child['id']
            )['state']
            == 'CANCEL_PENDING'
        )
        if broker_id == broker.rows[0]['id']:
            raise TimeoutError('cancellation status unknown')

    broker.cancel = cancel
    gateway.cancel_entries(NOW)
    assert len(calls) == 2
    assert store.projection('orders', child['id'])['state'] == 'CANCEL_PENDING'
    assert store.db.execute('SELECT state FROM reservations').fetchone()[0] != 'RELEASED'
    store.close()


def seed_confirmed_entry(store, config, scope, evidence, calendar, broker, take_profit=None):
    plan, intent = seed_plan(store, config, scope, evidence, take_profit=take_profit)
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = approve(gate, intent, broker, config)
    assert decision.approved, decision.reasons
    ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    broker.rows[0].update(
        status="filled", filled_qty="1", filled_avg_price="100", filled_at=NOW.isoformat()
    )
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment="paper",
            group_id="unmanaged:TEST_STOCK",
            symbol="TEST_STOCK",
            asset_type="stock",
            quantity="1",
            cost_basis="100",
            reconciled_at=NOW,
        ),
    )
    account = Reconciler(store, config, scope, broker).reconcile(NOW)
    evidence.ingest_market(
        MarketEvent(
            symbol="TEST_STOCK",
            source="fixture",
            source_version="held-current-quote",
            event_type="quote",
            effective_event_time=NOW,
            received_at=NOW,
            available_at=NOW,
            feed="sip",
            quality_class="genuine",
            bid="100",
            ask="100.01",
        )
    )
    return plan, intent, account


def close_confirmed_entry(config, entry):
    return OrderIntent(
        account_id=config.account_id,
        environment=config.environment,
        purpose="close",
        group_id="group-" + entry.id,
        legs=(
            OrderLeg(
                instrument=entry.legs[0].instrument, side="sell", position_intent="sell_to_close"
            ),
        ),
        requested_quantity="1",
        limit_price="100",
        expires_at=NOW + timedelta(seconds=10),
        reason="synthetic confirmed holding exit",
    )


def maintenance_quote(evidence, now, bid, ask):
    evidence.ingest_market(
        MarketEvent(
            symbol="TEST_STOCK",
            source="fixture",
            source_version=now.isoformat(),
            event_type="quote",
            effective_event_time=now,
            received_at=now,
            available_at=now,
            feed="sip",
            quality_class="genuine",
            bid=bid,
            ask=ask,
        )
    )


def maintain(gateway, account, now):
    return gateway.maintain_orders(
        now,
        account=account,
        entry_ready=True,
        news_ready=True,
        event_calendar_ready=True,
    )


def test_stock_profit_target_survives_reconciliation_and_restart(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, entry, _ = seed_confirmed_entry(
        store, config, scope, evidence, calendar, broker, take_profit="106"
    )
    group_id = "group-" + entry.id
    assert Decimal(store.projection("position_groups", group_id)["take_profit"]) == 106
    store.close()
    store = Store(tmp_path / "session.db")
    Reconciler(store, config, scope, broker).reconcile(NOW + timedelta(seconds=1))
    group = store.projection("position_groups", group_id)
    assert Decimal(group["take_profit"]) == 106
    assert Decimal(group["stop_price"]) == 99
    store.close()


def test_stock_profit_target_submits_one_bounded_exit_through_service(tmp_path):
    from src.runtime.service import SessionService

    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, entry, account = seed_confirmed_entry(
        store, config, scope, evidence, calendar, broker, take_profit="106"
    )
    now = NOW + timedelta(seconds=1)
    maintenance_quote(evidence, now, "106", "106.01")
    broker.instrument = lambda symbol, capacity: entry.legs[0].instrument
    service = SessionService(
        config.model_copy(update={"database_path": str(tmp_path / "session.db")}),
        broker=broker,
        calendar=calendar,
        clock=lambda: now,
    )
    try:
        service.account_profile = account
        service.health_state["reconciliation"] = "HEALTHY"
        service._supervise(now)
        assert len(broker.writes) == 2  # entry and profit-taking close
        exit_intent, quantity, _ = broker.writes[-1]
        assert exit_intent.purpose == "close" and quantity == 1
        assert exit_intent.limit_price == 106
        assert exit_intent.minimum_sell_price == Decimal("105.47")
        service._supervise(now)
        assert len(broker.writes) == 2  # pending close is maintained, never duplicated
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
        store.close()


def test_maintenance_reprices_entry_within_original_price_bound(tmp_path):
    store, config, scope, evidence, broker, _, intent, gateway, _ = replacement_setup(tmp_path)
    now = NOW + timedelta(seconds=config.order_reprice_seconds)
    account = Reconciler(store, config, scope, broker).reconcile(now)
    maintenance_quote(evidence, now, "100.19", "100.20")
    changes = maintain(gateway, account, now)
    assert len(changes) == 1
    child = changes[0]
    assert Decimal(child["limit_price"]) == intent.maximum_buy_price
    assert child["intent_id"] == intent.id
    assert len(broker.writes) == 2
    assert store.db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 1
    # Acknowledged replacement waits for current broker reconciliation.
    assert maintain(gateway, account, now) == []
    assert len(broker.writes) == 2
    store.close()


def test_maintenance_unknown_replacement_waits_even_after_deadline(tmp_path):
    store, config, scope, evidence, broker, _, _, gateway, _ = replacement_setup(tmp_path)
    now = NOW + timedelta(seconds=config.order_reprice_seconds)
    account = Reconciler(store, config, scope, broker).reconcile(now)
    maintenance_quote(evidence, now, "100.04", "100.05")
    broker.timeout = True
    assert maintain(gateway, account, now)[0]["state"] == "UNKNOWN"
    assert maintain(gateway, account, NOW + timedelta(seconds=config.order_cancel_seconds)) == []
    assert len(broker.writes) == 2
    store.close()


def test_maintenance_cancels_stale_entry_without_releasing_unconfirmed_reservation(tmp_path):
    store, config, scope, evidence, broker, _, intent, gateway, account = replacement_setup(
        tmp_path
    )
    calls = []
    broker.cancel = lambda broker_id: calls.append(broker_id)
    now = NOW + timedelta(seconds=config.order_cancel_seconds)
    changes = maintain(gateway, account, now)
    assert changes[0]["state"] == "CANCEL_PENDING"
    assert calls == [broker.rows[0]["id"]]
    assert maintain(gateway, account, now) == []
    assert store.db.execute("SELECT state FROM reservations").fetchone()[0] != "RELEASED"
    broker.rows[0]["status"] = "canceled"
    Reconciler(store, config, scope, broker).reconcile(now)
    assert store.projection("orders", intent.id)["state"] == "CANCELLED"
    assert store.db.execute("SELECT state FROM reservations").fetchone()[0] == "RELEASED"
    store.close()


def test_maintenance_skips_stale_or_wide_quote_without_broker_write(tmp_path):
    store, config, scope, evidence, broker, _, _, gateway, _ = replacement_setup(tmp_path)
    now = NOW + timedelta(seconds=config.order_reprice_seconds)
    account = Reconciler(store, config, scope, broker).reconcile(now)
    maintenance_quote(evidence, NOW, "100.04", "100.05")
    assert maintain(gateway, account, now) == []
    maintenance_quote(evidence, now, "99", "103")
    assert maintain(gateway, account, now) == []
    assert len(broker.writes) == 1
    store.close()


def test_close_maintenance_respects_floor_then_waits_for_confirmed_cancel(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    intent = close_confirmed_entry(config, entry).model_copy(
        update={"minimum_sell_price": Decimal("99.50")}
    )
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = gate.approve(
        intent, account, NOW, entry_ready=False, news_ready=False, event_calendar_ready=False
    )
    assert decision.approved, decision.reasons
    gateway = ExecutionGateway(store, config, scope, broker, evidence)
    gateway.submit(intent, decision, NOW)
    now = NOW + timedelta(seconds=config.order_reprice_seconds)
    broker.holdings = tuple(p.model_copy(update={"reconciled_at": now}) for p in broker.holdings)
    account = Reconciler(store, config, scope, broker).reconcile(now)
    maintenance_quote(evidence, now, "99.40", "99.41")
    child = maintain(gateway, account, now)[0]
    assert Decimal(child["limit_price"]) == Decimal("99.50")
    assert child["purpose"] == "close"
    account = Reconciler(store, config, scope, broker).reconcile(now)
    calls = []
    broker.cancel = lambda broker_id: calls.append(broker_id)
    deadline = NOW + timedelta(seconds=config.order_cancel_seconds)
    assert maintain(gateway, account, deadline)[0]["state"] == "CANCEL_PENDING"
    assert calls == [broker.rows[-1]["id"]]
    retry = close_confirmed_entry(config, entry).model_copy(
        update={"expires_at": deadline + timedelta(seconds=10)}
    )
    blocked = gate.approve(
        retry, account, deadline, entry_ready=False, news_ready=False, event_calendar_ready=False
    )
    assert not blocked.approved and "EXIT_ALREADY_PENDING" in blocked.reasons
    broker.rows[-1]["status"] = "canceled"
    broker.holdings = tuple(
        p.model_copy(update={"reconciled_at": deadline}) for p in broker.holdings
    )
    account = Reconciler(store, config, scope, broker).reconcile(deadline)
    maintenance_quote(evidence, deadline, "99.40", "99.41")
    retry = close_confirmed_entry(config, entry).model_copy(
        update={"expires_at": deadline + timedelta(seconds=10)}
    )
    allowed = gate.approve(
        retry, account, deadline, entry_ready=False, news_ready=False, event_calendar_ready=False
    )
    assert allowed.approved, allowed.reasons
    assert (
        Decimal(store.projection("position_groups", "group-" + entry.id)["legs"][0]["quantity"])
        == 1
    )
    store.close()


def quota_monitor_only(store, scope):
    store.project(
        "automation_control",
        scope,
        {
            "monitor_only": True,
            "reason": "CODEX_QUOTA_MONITOR_ONLY",
        },
        scope,
    )


def test_quota_monitor_only_blocks_submission_without_releasing_risk_reservation(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, intent = seed_plan(store, config, scope, evidence)
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    assert decision.approved
    reservation = dict(store.db.execute("SELECT * FROM reservations").fetchone())
    quota_monitor_only(store, scope)
    result = ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    assert result["state"] == "MONITOR_ONLY"
    assert result["reasons"] == ["CODEX_QUOTA_MONITOR_ONLY"]
    assert broker.writes == []
    assert store.projection("orders", intent.id) is None
    assert dict(store.db.execute("SELECT * FROM reservations").fetchone()) == reservation
    store.close()


def test_quota_monitor_only_blocks_replace_cancel_and_maintenance_but_reconciles_fills(tmp_path):
    store, config, scope, evidence, broker, _, intent, gateway, account = replacement_setup(
        tmp_path
    )
    before = store.projection("orders", intent.id)
    reservation = dict(store.db.execute("SELECT * FROM reservations").fetchone())
    broker.cancel = lambda broker_id: pytest.fail("quota freeze must not call broker cancellation")
    quota_monitor_only(store, scope)
    assert replace(gateway, intent.id, account)["state"] == "MONITOR_ONLY"
    assert gateway.cancel_entries(NOW)[0]["state"] == "MONITOR_ONLY"
    assert gateway._cancel_order(before, NOW, "fixture")["state"] == "MONITOR_ONLY"
    assert maintain(gateway, account, NOW)[0]["state"] == "MONITOR_ONLY"
    assert store.projection("orders", intent.id) == before
    assert dict(store.db.execute("SELECT * FROM reservations").fetchone()) == reservation
    assert len(broker.writes) == 1
    broker.rows[0].update(
        status="filled", filled_qty="1", filled_avg_price="100", filled_at=NOW.isoformat()
    )
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment="paper",
            group_id="unmanaged",
            symbol="TEST_STOCK",
            asset_type="stock",
            quantity="1",
            reconciled_at=NOW,
        ),
    )
    Reconciler(store, config, scope, broker).reconcile(NOW)
    assert store.projection("orders", intent.id)["state"] == "FILLED"
    assert (
        Decimal(store.projection("position_groups", "group-" + intent.id)["legs"][0]["quantity"])
        == 1
    )
    assert len(broker.writes) == 1
    store.close()


@pytest.mark.parametrize("operation", ["submit", "replace", "cancel"])
def test_quota_latching_after_local_preparation_stops_network_delivery(
    tmp_path, monkeypatch, operation
):
    if operation == "submit":
        store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
        _, intent = seed_plan(store, config, scope, evidence)
        decision = approve(
            RiskGate(store, config, scope, evidence, calendar), intent, broker, config
        )
        gateway = ExecutionGateway(store, config, scope, broker, evidence)
        initial_order = None
        write = lambda: gateway.submit(intent, decision, NOW)
    else:
        store, config, scope, evidence, broker, _, intent, gateway, account = replacement_setup(
            tmp_path
        )
        initial_order = store.projection("orders", intent.id)
        broker.cancel = lambda broker_id: pytest.fail("cancel must remain definitely unsent")
        write = (
            (lambda: replace(gateway, intent.id, account))
            if operation == "replace"
            else (lambda: gateway._cancel_order(initial_order, NOW, "fixture"))
        )
    initial_write_count = len(broker.writes)
    prepared, frozen = Event(), Event()
    original_check = gateway._execution_block
    checks = 0

    def check(*args):
        nonlocal checks
        checks += 1
        if checks == 2:
            prepared.set()
            assert frozen.wait(3), "concurrent quota worker must latch before network delivery"
        return original_check(*args)

    def latch():
        assert prepared.wait(3)
        quota_monitor_only(store, scope)
        frozen.set()

    monkeypatch.setattr(gateway, "_execution_block", check)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(latch)
        result = write()
        future.result(timeout=3)
    assert len(broker.writes) == initial_write_count
    assert result["reasons"] == ["CODEX_QUOTA_MONITOR_ONLY"]
    if operation == "cancel":
        assert result["state"] == "MONITOR_ONLY"
        after = store.projection("orders", intent.id)
        assert after["state"] == initial_order["state"]
        assert "cancel_requested_at" not in after
        assert any(
            row["action"] == "CANCEL_NOT_SENT" for row in store.list("order_controls", scope)
        )
    else:
        assert result["state"] == "REJECTED" and result["delivery_state"] == "NOT_SENT"
        assert "broker_id" not in result
        if operation == "replace":
            after = store.projection("orders", intent.id)
            assert after["state"] == initial_order["state"]
            assert "pending_replacement" not in after
    Reconciler(store, config, scope, broker).reconcile(NOW)
    assert not any(
        row["state"] in {"UNKNOWN", "SUBMITTING", "CANCEL_PENDING", "REPLACE_PENDING"}
        for row in store.projected("orders", scope)
    )
    store.close()


def trial_plan(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    policy = config.policy.model_copy(
        update={
            "max_capital_allocation_pct": Decimal(".02"),
            "max_portfolio_allocation_pct": Decimal(".10"),
            "planned_loss_per_trade_pct": Decimal(".001"),
            "daily_loss_pause_pct": Decimal(".01"),
        }
    )
    config = config.model_copy(update={"policy": policy, "research_backend": "codex_cli"})
    _, intent = seed_plan(
        store,
        config,
        scope,
        evidence,
        expected_net_value=None,
        approval_changes={
            "evidence_level": "paper_trial",
            "symbols": ("TEST_STOCK",),
            "validated_net_expectancy": None,
        },
    )
    return store, config, scope, evidence, calendar, broker, intent


def test_explicit_strategy_symbol_restriction_rejects_otherwise_valid_entry(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, intent = seed_plan(
        store, config, scope, evidence, approval_changes={"symbols": ("OTHER_SYMBOL",)}
    )
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    assert not decision.approved and "STRATEGY_SYMBOL_NOT_APPROVED" in decision.reasons
    assert broker.writes == []
    assert store.db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0
    store.close()


@pytest.mark.parametrize("boundary", ["approve", "submit", "replace"])
@pytest.mark.parametrize(
    "restriction,reason",
    [
        ("backend", "PAPER_TRIAL_CODEX_REQUIRED"),
        ("live", "PAPER_TRIAL_PAPER_ONLY"),
    ],
)
def test_paper_trial_cannot_outlive_paper_only_codex_authority(
    tmp_path, boundary, restriction, reason
):
    store, config, scope, evidence, calendar, broker, intent = trial_plan(tmp_path)
    invalid = (
        config.model_copy(update={"research_backend": "deterministic"})
        if restriction == "backend"
        else config.model_copy(update={"execution_mode": "live"})
    )
    if boundary == "approve":
        decision = approve(
            RiskGate(store, invalid, scope, evidence, calendar), intent, broker, invalid
        )
        assert not decision.approved and reason in decision.reasons
    else:
        decision = approve(
            RiskGate(store, config, scope, evidence, calendar), intent, broker, config
        )
        assert decision.approved, decision.reasons
        if boundary == "submit":
            result = ExecutionGateway(store, invalid, scope, broker, evidence).submit(
                intent, decision, NOW
            )
            assert result["state"] == "REJECTED" and reason in result["reasons"]
        else:
            ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
            account = Reconciler(store, config, scope, broker).reconcile(NOW)
            expected = (
                "REPLACEMENT_" + reason
                if restriction == "backend"
                else "REPLACEMENT_(PAPER_TRIAL_PAPER_ONLY|STRATEGY_APPROVAL_INVALID)"
            )
            with pytest.raises(ValueError, match=expected):
                replace(
                    ExecutionGateway(store, invalid, scope, broker, evidence), intent.id, account
                )
    assert len(broker.writes) == (1 if boundary == "replace" else 0)
    store.close()


@pytest.mark.parametrize("session_offset", [-1, 1])
def test_t09_wrong_session_plan_rejected_while_existing_group_remains_open(
    tmp_path, session_offset
):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, held_entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    baseline, proposal = seed_plan(store, config, scope, evidence, "TEST_OTHER")
    other = calendar.get("fixture-session").model_copy(
        update={
            "id": "different-session",
            "session_date": (NOW + timedelta(days=session_offset)).date().isoformat(),
            "open": NOW.replace(hour=13, minute=30) + timedelta(days=session_offset),
            "close": NOW.replace(hour=20, minute=0) + timedelta(days=session_offset),
        }
    )
    calendar.sessions[other.id] = other
    store.put("sessions", other, scope)
    wrong = baseline.model_copy(update={"id": "wrong-session-plan", "target_session_id": other.id})
    store.put("plans", wrong, scope)
    for state in ("RESEARCH_COMPLETE", "PENDING_VALIDATION", "ELIGIBLE"):
        store.transition_plan(wrong.id, state, scope=scope)
    proposal = proposal.model_copy(update={"plan_id": wrong.id})
    gate = RiskGate(store, config, scope, evidence, calendar)
    result = gate.approve(
        proposal, account, NOW, entry_ready=True, news_ready=True, event_calendar_ready=True
    )
    assert not result.approved and "ENTRY_WINDOW_CLOSED" in result.reasons
    assert (
        store.db.execute(
            "SELECT count(*) FROM reservations WHERE intent_id=?", (proposal.id,)
        ).fetchone()[0]
        == 0
    )
    assert store.projection("position_groups", "group-" + held_entry.id)["state"] == "OPEN"
    assert len(broker.writes) == 1
    store.close()


def test_t09_superseded_plan_revokes_pending_authority_and_preserves_holdings(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, held_entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    plan, intent = seed_plan(store, config, scope, evidence, "TEST_OTHER")
    decision = RiskGate(store, config, scope, evidence, calendar).approve(
        intent, account, NOW, entry_ready=True, news_ready=True, event_calendar_ready=True
    )
    assert decision.approved, decision.reasons
    store.transition_plan(plan.id, "SUPERSEDED", ("REPLACED_BY_REVIEWED_PLAN",), scope=scope)
    result = ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    assert result["state"] == "REJECTED"
    assert "PLAN_NO_LONGER_ELIGIBLE" in result["reasons"]
    assert (
        store.db.execute(
            "SELECT state FROM reservations WHERE intent_id=?", (intent.id,)
        ).fetchone()[0]
        == "RELEASED"
    )
    assert store.projection("position_groups", "group-" + held_entry.id)["state"] == "OPEN"
    assert len(broker.writes) == 1
    store.close()


def test_t18_broker_rejected_exit_preserves_open_group_and_confirmed_quantity(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    exit_intent = close_confirmed_entry(config, entry)
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = gate.approve(
        exit_intent, account, NOW, entry_ready=False, news_ready=False, event_calendar_ready=False
    )
    assert decision.approved, decision.reasons
    ExecutionGateway(store, config, scope, broker, evidence).submit(exit_intent, decision, NOW)
    broker.rows[-1].update(status="rejected", filled_qty="0")
    Reconciler(store, config, scope, broker).reconcile(NOW)
    assert store.projection("orders", exit_intent.id)["state"] == "REJECTED"
    group = store.projection("position_groups", "group-" + entry.id)
    assert group["state"] == "OPEN"
    assert Decimal(group["legs"][0]["quantity"]) == 1
    assert not PositionGroup.model_validate(
        {key: value for key, value in group.items() if key != "_revision"}
    ).flat
    retry = close_confirmed_entry(config, entry)
    retry_decision = gate.approve(
        retry, account, NOW, entry_ready=False, news_ready=False, event_calendar_ready=False
    )
    assert retry_decision.approved, retry_decision.reasons
    assert len(broker.writes) == 2  # entry and rejected exit; no implicit resend
    store.close()


@pytest.mark.parametrize(
    "blocker,reason",
    [
        ("daily_loss", "DAILY_LOSS_PAUSE"),
        ("entry_quota", "ENTRY_QUOTA_EXHAUSTED"),
    ],
)
def test_t25_entry_pause_limit_rejects_new_group_but_allows_confirmed_close(
    tmp_path, blocker, reason
):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    if blocker == "entry_quota":
        config = config.model_copy(
            update={
                "policy": config.policy.model_copy(update={"max_new_entry_groups_per_session": 1})
            }
        )
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    if blocker == "daily_loss":
        account = account.model_copy(
            update={"daily_pnl": -account.equity * config.policy.daily_loss_pause_pct}
        )
    _, new_intent = seed_plan(store, config, scope, evidence, "TEST_OTHER")
    gate = RiskGate(store, config, scope, evidence, calendar)
    blocked = gate.approve(
        new_intent, account, NOW, entry_ready=True, news_ready=True, event_calendar_ready=True
    )
    assert not blocked.approved and reason in blocked.reasons
    assert (
        store.db.execute(
            "SELECT count(*) FROM reservations WHERE intent_id=?", (new_intent.id,)
        ).fetchone()[0]
        == 0
    )
    closing = gate.approve(
        close_confirmed_entry(config, entry),
        account,
        NOW,
        entry_ready=False,
        news_ready=False,
        event_calendar_ready=False,
    )
    assert closing.approved, closing.reasons
    assert closing.approved_quantity == 1
    assert store.projection("position_groups", "group-" + entry.id)["state"] == "OPEN"
    assert len(broker.writes) == 1
    store.close()


def market_close(config, entry, **changes):
    values = close_confirmed_entry(config, entry).model_dump()
    values.update(order_type="market", limit_price=None, reference_price=Decimal("100"))
    values.update(changes)
    return OrderIntent.model_validate(values)


def test_market_stock_exit_uses_confirmed_quantity_and_preserves_actual_fill_price(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    config = config.model_copy(update={"stock_exit_order_type": "market"})
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    intent = market_close(config, entry)
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = gate.approve(
        intent, account, NOW, entry_ready=False, news_ready=False, event_calendar_ready=False
    )
    assert decision.approved, decision.reasons
    assert decision.approved_quantity == 1
    gateway = ExecutionGateway(store, config, scope, broker, evidence)
    row = gateway.submit(intent, decision, NOW)
    assert row["order_type"] == "market"
    assert row["limit_price"] is None and row["reference_price"] == "100"
    assert intent.minimum_sell_price is None and intent.maximum_buy_price is None
    assert len(broker.writes) == 2
    account = Reconciler(store, config, scope, broker).reconcile(NOW)
    with pytest.raises(ValueError, match="MARKET_ORDER_REPLACEMENT_NOT_ALLOWED"):
        replace(gateway, intent.id, account, "99")
    assert maintain(gateway, account, NOW + timedelta(seconds=config.order_cancel_seconds)) == []
    assert any(row["reason"] == "MARKET_EXIT_UNFILLED" for row in store.list("incidents", scope))
    maintain(gateway, account, NOW + timedelta(seconds=config.order_cancel_seconds + 1))
    assert (
        sum(row["reason"] == "MARKET_EXIT_UNFILLED" for row in store.list("incidents", scope)) == 1
    )
    broker.rows[-1].update(
        status="filled", filled_qty="1", filled_avg_price="98.75", filled_at=NOW.isoformat()
    )
    broker.holdings = ()
    Reconciler(store, config, scope, broker).reconcile(NOW)
    actual = next(fill for fill in store.list("fills", scope) if fill["intent_id"] == intent.id)
    assert Decimal(actual["price"]) == Decimal("98.75"), "market fill is not the reference quote"
    assert store.projection("position_groups", "group-" + entry.id)["state"] == "CLOSED"
    store.close()


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("disabled", "MARKET_STOCK_EXIT_POLICY_DISABLED"),
        ("stale_quote", "CLOSE_CURRENT_GENUINE_QUOTES_REQUIRED"),
        ("changed_reference", "MARKET_EXIT_REFERENCE_QUOTE_CHANGED"),
        ("excess_quantity", "CLOSE_EXCEEDS_CONFIRMED_HOLDING"),
        ("wide_quote", "MARKET_EXIT_QUOTE_SPREAD_INVALID"),
    ],
)
def test_market_stock_exit_still_requires_current_risk_authority(tmp_path, failure, reason):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    config = config.model_copy(update={"stock_exit_order_type": "market"})
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    changes = {}
    now = NOW
    if failure == "disabled":
        config = config.model_copy(update={"stock_exit_order_type": "limit"})
    elif failure == "stale_quote":
        now += timedelta(seconds=config.stock_quote_max_age_seconds + 1)
    elif failure == "changed_reference":
        changes["reference_price"] = Decimal("99")
    elif failure == "excess_quantity":
        changes["requested_quantity"] = Decimal("2")
    elif failure == "wide_quote":
        now += timedelta(seconds=1)
        maintenance_quote(evidence, now, "100", "110")
    intent = market_close(config, entry, **changes)
    decision = RiskGate(store, config, scope, evidence, calendar).approve(
        intent,
        account,
        now,
        entry_ready=False,
        news_ready=False,
        event_calendar_ready=False,
    )
    assert not decision.approved and reason in decision.reasons
    assert len(broker.writes) == 1
    store.close()


def test_market_policy_revoked_between_approval_and_submission_cannot_reach_broker(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    config = config.model_copy(update={"stock_exit_order_type": "market"})
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    intent = market_close(config, entry)
    decision = RiskGate(store, config, scope, evidence, calendar).approve(
        intent,
        account,
        NOW,
        entry_ready=False,
        news_ready=False,
        event_calendar_ready=False,
    )
    assert decision.approved
    disabled = config.model_copy(update={"stock_exit_order_type": "limit"})
    row = ExecutionGateway(store, disabled, scope, broker, evidence).submit(intent, decision, NOW)
    assert row["state"] == "REJECTED"
    assert "MARKET_STOCK_EXIT_POLICY_DISABLED" in row["reasons"]
    assert len(broker.writes) == 1
    store.close()


@pytest.mark.parametrize(
    "change,reason",
    [
        ("fabricated_account", "MARKET_EXIT_ACCOUNT_CHANGED"),
        ("account_changed", "MARKET_EXIT_ACCOUNT_CHANGED"),
        ("holding_closed", "MARKET_EXIT_CURRENT_HOLDING_REQUIRED"),
        ("quote_changed", "MARKET_EXIT_REFERENCE_QUOTE_CHANGED"),
        ("missing_calendar", "MARKET_EXIT_SESSION_CLOSED_OR_UNAVAILABLE"),
    ],
)
def test_market_gateway_rechecks_current_authority_after_risk_approval(tmp_path, change, reason):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    config = config.model_copy(update={"stock_exit_order_type": "market"})
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    intent = market_close(config, entry)
    candidate = (
        account.model_copy(update={"id": "fabricated-fresh-account"})
        if change == "fabricated_account"
        else account
    )
    decision = RiskGate(store, config, scope, evidence, calendar).approve(
        intent,
        candidate,
        NOW,
        entry_ready=False,
        news_ready=False,
        event_calendar_ready=False,
    )
    assert decision.approved, decision.reasons
    now = NOW + timedelta(seconds=1)
    if change == "account_changed":
        refreshed = account.model_copy(update={"id": "new-account", "observed_at": now})
        store.project("account", scope, refreshed.model_dump(mode="json"), scope)
    elif change == "holding_closed":
        refreshed = account.model_copy(update={"positions": (), "observed_at": now})
        store.project("account", scope, refreshed.model_dump(mode="json"), scope)
    elif change == "quote_changed":
        maintenance_quote(evidence, now, "99.98", "99.99")
    elif change == "missing_calendar":
        store.db.execute("DELETE FROM records WHERE kind='sessions'")
    row = ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, now)
    assert row["state"] == "REJECTED" and reason in row["reasons"]
    assert len(broker.writes) == 1
    store.close()


def test_quota_freeze_also_prevents_approved_market_protective_exit(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    config = config.model_copy(update={"stock_exit_order_type": "market"})
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    intent = market_close(config, entry)
    decision = RiskGate(store, config, scope, evidence, calendar).approve(
        intent,
        account,
        NOW,
        entry_ready=False,
        news_ready=False,
        event_calendar_ready=False,
    )
    assert decision.approved
    quota_monitor_only(store, scope)
    row = ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    assert row["state"] == "MONITOR_ONLY"
    assert len(broker.writes) == 1
    assert store.projection("orders", intent.id) is None
    store.close()


@pytest.mark.parametrize(
    "trigger,expected_type,bid",
    [
        ("stop", "market", "98.99"),
        ("time", "market", "100"),
        ("flatten", "market", "100"),
        ("news", "market", "100"),
        ("take_profit", "limit", "106"),
    ],
)
def test_service_selects_market_for_urgent_stock_exits_and_limit_for_profit(
    tmp_path, trigger, expected_type, bid
):
    from src.runtime.models import NewsEvent
    from src.runtime.service import SessionService

    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    config = config.model_copy(update={"stock_exit_order_type": "market"})
    _, entry, account = seed_confirmed_entry(
        store,
        config,
        scope,
        evidence,
        calendar,
        broker,
        take_profit="106",
    )
    now = NOW + timedelta(seconds=1)
    maintenance_quote(evidence, now, bid, str(Decimal(bid) + Decimal(".01")))
    if trigger == "time":
        group = store.projection("position_groups", "group-" + entry.id)
        group.pop("_revision")
        group["close_at"] = now.isoformat()
        store.project("position_groups", group["id"], group, scope)
    if trigger == "news":
        evidence.ingest_news(
            NewsEvent(
                provider="fixture",
                article_id="exit-news",
                source_version="v1",
                published_at=now,
                updated_at=now,
                received_at=now,
                available_at=now,
                symbols=("TEST_STOCK",),
                headline="Synthetic material event",
                content_hash="fixture",
                blocking=True,
                verified=True,
            )
        )
    broker.instrument = lambda symbol, capacity: entry.legs[0].instrument
    service = SessionService(
        config.model_copy(update={"database_path": str(tmp_path / "session.db")}),
        broker=broker,
        calendar=calendar,
        clock=lambda: now,
    )
    try:
        service.account_profile = account
        service.health_state["reconciliation"] = "HEALTHY"
        service._supervise(now, flatten=trigger == "flatten")
        assert len(broker.writes) == 2
        intent, quantity, _ = broker.writes[-1]
        assert intent.order_type == expected_type
        assert intent.purpose == "close" and quantity == 1
        if expected_type == "market":
            assert intent.limit_price is None
            assert intent.reference_price == Decimal(bid)
            assert intent.maximum_buy_price is None and intent.minimum_sell_price is None
        else:
            assert intent.limit_price == Decimal(bid) and intent.reference_price is None
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
        store.close()


@pytest.mark.parametrize(
    "bid,ask,expected_type",
    [
        ("100", "100.01", None),
        ("104.99", "105", "market"),
        ("94.99", "95.01", None),
        ("94.99", "95", "limit"),
    ],
)
def test_adopted_short_stock_exits_use_cover_ask_and_short_direction(
    tmp_path, bid, ask, expected_type
):
    from src.runtime.service import SessionService

    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    config = config.model_copy(update={"stock_exit_order_type": "market"})
    broker.holdings = (
        PositionLeg(
            account_id=config.account_id,
            environment="paper",
            group_id="unmanaged:TEST_STOCK",
            symbol="TEST_STOCK",
            asset_type="stock",
            quantity="-2",
            cost_basis="-200",
            reconciled_at=NOW,
        ),
    )
    account = Reconciler(store, config, scope, broker).reconcile(NOW)
    group = store.projected("position_groups", scope)[0]
    group.pop("_revision")
    group.update(
        exit_policy_id="operator-adopted-v1",
        stop_price="105",
        take_profit="95",
        state="OPEN",
        close_at=(NOW + timedelta(hours=1)).isoformat(),
    )
    store.project("position_groups", group["id"], group, scope)
    maintenance_quote(evidence, NOW, bid, ask)
    broker.instrument = lambda symbol, capacity: Instrument(
        symbol=symbol, liquidity_capacity=capacity
    )
    service = SessionService(
        config.model_copy(update={"database_path": str(tmp_path / "session.db")}),
        broker=broker,
        calendar=calendar,
        clock=lambda: NOW,
    )
    try:
        service.account_profile = account
        service.health_state["reconciliation"] = "HEALTHY"
        service._supervise(NOW)
        if expected_type is None:
            assert broker.writes == []
        else:
            assert len(broker.writes) == 1
            intent, quantity, _ = broker.writes[0]
            assert intent.order_type == expected_type and quantity == 2
            assert intent.legs[0].side == "buy"
            assert intent.legs[0].position_intent == "buy_to_close"
            executable_price = (
                intent.reference_price if expected_type == "market" else intent.limit_price
            )
            assert executable_price == Decimal(ask)
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
        store.close()


def test_t55_foreign_account_plan_cannot_reserve_or_reach_gateway(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    original, intent = seed_plan(store, config, scope, evidence)
    foreign = original.model_copy(
        update={"id": "foreign-account-plan", "account_id": "other-account"}
    )
    store.put("plans", foreign, scope)
    for state in ("RESEARCH_COMPLETE", "PENDING_VALIDATION", "ELIGIBLE"):
        store.transition_plan(foreign.id, state, scope=scope)
    intent = intent.model_copy(update={"plan_id": foreign.id})
    decision = approve(RiskGate(store, config, scope, evidence, calendar), intent, broker, config)
    assert not decision.approved and "PLAN_SCOPE_MISMATCH" in decision.reasons
    assert store.db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0
    with pytest.raises(ValueError, match="risk-approved"):
        ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    assert store.projected("orders", scope) == []
    assert broker.writes == []
    store.close()


def test_t50_pre_gauss_reassesses_revoked_options_permission_from_frozen_plan(tmp_path):
    from test_four_agent_options import alternatives, approval
    from src.runtime.service import SessionService
    from src.runtime.evidence import EvidenceClock
    from src.runtime.roles import PreGauss
    from src.runtime.options import opening_legs

    store, base, scope, evidence, calendar, broker = setup(tmp_path)
    config = RuntimeConfig.model_validate(
        {
            **base.model_dump(),
            "database_path": str(tmp_path / "session.db"),
            "data_profile": "SUBSCRIBED_REALTIME",
            "policy": RiskPolicy(options_enabled=True, spreads_enabled=True),
        }
    )
    evidence.config = config
    evidence.clock = EvidenceClock(config.data_profile, 0)
    original, _ = seed_plan(store, base, scope, evidence)
    store.transition_plan(original.id, "SUPERSEDED", scope=scope)
    alternative, quotes = alternatives()[0][0], alternatives()[2]
    permit = approval()
    store.put("strategy_approvals", permit, scope)
    for quote in quotes.values():
        evidence.ingest_market(quote)
    evidence.ingest_market(
        MarketEvent(
            symbol="TEST_STOCK",
            source="fixture",
            source_version="current-signal",
            event_type="bar",
            effective_event_time=NOW - timedelta(seconds=1),
            received_at=NOW,
            available_at=NOW,
            interval_start=NOW - timedelta(seconds=61),
            feed="sip",
            quality_class="genuine",
            price="100",
        )
    )
    service = SessionService(config, broker=broker, calendar=calendar, clock=lambda: NOW)
    try:
        frozen = broker.account(config.policy, NOW).model_copy(
            update={
                "id": "close-account-before-restriction",
                "equity": Decimal("20000"),
                "options_level": 3,
                "options_buying_power": Decimal("10000"),
            }
        )
        service.store.put("account_profiles", frozen, scope)
        original_report = service.suitability.assess(
            frozen, [alternative], config.data_profile, [permit]
        )
        assert original_report.outcome == "FEASIBLE"
        service.store.put("suitability_reports", original_report, scope)
        plan = original.model_copy(
            update={
                "id": "review-options-permission",
                "data_profile": config.data_profile,
                "alternative": alternative,
                "approval_id": permit.id,
                "max_buy_price": Decimal("1.10"),
                "account_profile_id": frozen.id,
                "suitability_report_id": original_report.id,
            }
        )
        service.store.put("plans", plan, scope)
        for state in ("RESEARCH_COMPLETE", "PENDING_VALIDATION"):
            service.store.transition_plan(plan.id, state, scope=scope)
        service.account_profile = frozen
        service.price_ready = service.news_ready = service.event_calendar_ready = True
        PreGauss().run(service, calendar.get("fixture-session"), NOW)
        assert service.store.projection("plan_state", plan.id)["state"] == "ELIGIBLE"

        changed = frozen.model_copy(
            update={"id": "pre-account-permission-revoked", "options_level": 2}
        )
        service.store.put("account_profiles", changed, scope)
        service.account_profile = changed
        results = PreGauss().run(service, calendar.get("fixture-session"), NOW)
        validation = next(
            service.store.get("validations", key)
            for key in results
            if service.store.get("validations", key)["plan_id"] == plan.id
        )
        assert validation["account_profile_id"] == changed.id
        assert validation["outcome"] == "DEFERRED"
        assert "OPTION_SPREAD_PERMISSION_REQUIRED" in validation["reasons"]
        assert service.store.get("suitability_reports", original_report.id)["outcome"] == "FEASIBLE"
        assert service.store.get("plans", plan.id)["account_profile_id"] == frozen.id
        intent = OrderIntent(
            account_id=config.account_id,
            environment="paper",
            purpose="open",
            plan_id=plan.id,
            plan_version=1,
            legs=opening_legs(alternative),
            limit_price="1.01",
            maximum_buy_price="1.10",
            expires_at=NOW + timedelta(seconds=10),
            reason="stale pre-restriction proposal",
        )
        final = service.risk.approve(
            intent, changed, NOW, entry_ready=True, news_ready=True, event_calendar_ready=True
        )
        assert not final.approved and "OPTION_SPREAD_PERMISSION_REQUIRED" in final.reasons
        assert service.store.db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0
        assert broker.writes == []
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
        store.close()
