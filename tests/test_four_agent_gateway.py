"""Broker-double tests for current risk reservations and restart semantics."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.runtime.calendar import SessionCalendar
from src.runtime.evidence import EvidenceService
from src.runtime.models import (AccountProfile, Alternative, Instrument, MarketEvent,
    AccountOperationalState, OrderIntent, OrderLeg, PositionGroup, PositionLeg, RiskPolicy, Rule, RuntimeConfig, SessionPlan,
    StrategyApproval, TradingSession)
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
        return AccountProfile(account_id="fixture-account", environment="paper", equity="10000",
            cash="10000", buying_power="10000", currency="USD", observed_at=now,
            mandate_id=policy.id, daily_pnl="0")

    def submit(self, intent, quantity, client_id):
        self.writes.append((intent, quantity, client_id))
        row = {"id": "broker-"+intent.id, "client_order_id": client_id, "status": "new", "filled_qty": "0"}
        self.rows.append(row)
        if self.timeout:
            raise TimeoutError("accepted then connection dropped")
        return row

    def orders(self): return self.rows
    def positions(self, account_id, now): return self.holdings
    def activities(self): return []
    def order_by_client_id(self, client_id):
        return next((row for row in self.rows if row["client_order_id"] == client_id), None)

    def replace(self, broker_id, *, limit_price, client_order_id):
        parent=next(row for row in self.rows if row['id']==broker_id)
        child={'id':'broker-'+client_order_id, 'client_order_id':client_order_id,
               'status':'new', 'filled_qty':'0', 'replaces':broker_id}
        parent.update(status='replaced',replaced_by=child['id'])
        self.rows.append(child)
        self.writes.append(('replace',limit_price,client_order_id))
        if self.timeout:
            raise TimeoutError('replacement accepted then timeout')
        return child


def setup(tmp_path, *, mode="shadow", maximum_groups=5):
    policy = RiskPolicy(max_open_position_groups=maximum_groups)
    config = RuntimeConfig(account_id="fixture-account", execution_mode=mode, policy=policy,
                           symbols=("TEST_STOCK",), strategy_allowlist=("fixture",))
    store = Store(tmp_path / "session.db")
    scope = "paper:fixture-account"
    session = TradingSession(id="fixture-session", session_date="2026-09-08", calendar_version="fixture-v1",
        open=NOW.replace(hour=13, minute=30), close=NOW.replace(hour=20, minute=0))
    evidence = EvidenceService(store, config, scope)
    calendar = SessionCalendar((session,))
    store.put('sessions',session,scope)
    broker = BrokerDouble()
    return store, config, scope, evidence, calendar, broker


def seed_plan(store, config, scope, evidence, symbol="TEST_STOCK"):
    instrument = Instrument(symbol=symbol, liquidity_capacity="3")
    alternative = Alternative(strategy_id="fixture", strategy_version="v1", instrument=instrument,
        entry_price="100", stop_price="99", expected_net_value="1", evidence_ids=("fixture-source",))
    approved = StrategyApproval(strategy_id="fixture", strategy_version="v1", account_id=config.account_id,
        environment="paper", approved_by="fixture-operator", profiles=("FREE_DELAYED",),
        execution_modes=("shadow", "paper", "replay"), requires_current_quote=False,
        maximum_signal_latency_seconds=1200, minimum_history=2, requires_news=False,
        requires_event_calendar=False, validation_reference="fixture-only", validated_net_expectancy="1",
        expiry=NOW+timedelta(days=1))
    plan = SessionPlan(target_session_id="fixture-session", snapshot_id="fixture-snapshot",
        candidate_id="fixture-candidate", account_id=config.account_id, environment="paper",
        account_profile_id="fixture-account-profile", suitability_report_id="fixture-suitability",
        data_profile=config.data_profile, data_policy_version=config.data_policy_version,
        signal_as_of=NOW-timedelta(seconds=902), strategy_id="fixture", strategy_version="v1",
        approval_id=approved.id, alternative=alternative,
        entry_rules=(Rule(rule_type="completed_bar_condition", operator="above", value="99"),),
        max_buy_price="100.10", valid_from=NOW-timedelta(hours=1),
        entry_expires_at=NOW+timedelta(hours=3), close_at=NOW+timedelta(hours=4),
        risk_policy_id=config.policy.id, exit_policy_id="fixed-exit-v1")
    store.put("strategy_approvals", approved, scope)
    store.put("plans", plan, scope)
    for state in ("RESEARCH_COMPLETE", "PENDING_VALIDATION", "ELIGIBLE"):
        store.transition_plan(plan.id, state, scope=scope)
    evidence.ingest_market(MarketEvent(symbol=symbol, source="fixture", source_version="1", event_type="bar",
        effective_event_time=NOW-timedelta(seconds=902), received_at=NOW, available_at=NOW,
        interval_start=NOW-timedelta(seconds=962), feed="sip", quality_class="genuine", price="100"))
    intent = OrderIntent(account_id=config.account_id, environment="paper", purpose="open",
        plan_id=plan.id, plan_version=1, legs=(OrderLeg(instrument=instrument, side="buy", position_intent="buy_to_open"),),
        limit_price="100", maximum_buy_price="100.10", expires_at=NOW+timedelta(minutes=1), reason="synthetic")
    return plan, intent


def approve(gate, intent, broker, config):
    return gate.approve(intent, broker.account(config.policy, NOW), NOW,
                        entry_ready=True, news_ready=True, event_calendar_ready=True)


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
    Reconciler(reopened, config, scope, broker).reconcile(NOW+timedelta(seconds=1))
    assert reopened.projection("orders", intent.id)["state"] == "OPEN"
    ExecutionGateway(reopened, config, scope, broker, evidence).submit(intent, decision, NOW+timedelta(seconds=1))
    assert len(broker.writes) == 1
    reopened.close()


def test_t14_concurrent_candidates_cannot_claim_same_position_slot(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, maximum_groups=1)
    intents = [seed_plan(store, config, scope, evidence, symbol)[1] for symbol in ("TEST_STOCK", "TEST_OTHER")]
    gate = RiskGate(store, config, scope, evidence, calendar)
    with ThreadPoolExecutor(max_workers=2) as workers:
        decisions = list(workers.map(lambda intent: approve(gate, intent, broker, config), intents))
    assert sum(decision.approved for decision in decisions) == 1
    assert "POSITION_GROUP_LIMIT" in next(decision.reasons for decision in decisions if not decision.approved)
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
    broker.rows[0].update(status="canceled", filled_qty="1", filled_avg_price="100", filled_at=NOW.isoformat())
    broker.holdings = (PositionLeg(account_id=config.account_id, environment="paper", group_id="unmanaged",
        symbol="TEST_STOCK", asset_type="stock", quantity="1", cost_basis="100", reconciled_at=NOW),)
    Reconciler(store, config, scope, broker).reconcile(NOW)
    group = store.projection("position_groups", "group-"+intent.id)
    assert group["state"] == "OPEN"
    assert Decimal(group["legs"][0]["quantity"]) == 1
    assert store.db.execute("SELECT first_fill FROM reservations WHERE intent_id=?", (intent.id,)).fetchone()[0] == 1
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
    evidence.ingest_news(NewsEvent(provider="fixture", article_id="late-announcement",
        source_version="v1", published_at=received, updated_at=received, received_at=received,
        available_at=received, symbols=("TEST_STOCK",), headline="Synthetic trading halt",
        content_hash="fixture-halt", blocking=True, verified=True))
    service = SessionService(config.model_copy(update={"database_path": str(tmp_path/"session.db")}),
        broker=broker, calendar=calendar, clock=lambda: received)
    try:
        service.run_once()
        assert service.store.projection("plan_state", plan.id)["state"] == "INVALIDATED"
        assert cancelled == [broker.rows[0]["id"]]
        assert service.store.projection("orders", intent.id)["state"] == "CANCEL_PENDING"
        assert service.evidence.news(received) == []  # current safety never becomes delayed signal
        assert service.store.db.execute("SELECT state FROM reservations").fetchone()[0] != "RELEASED"

        # The broker confirms a fill racing the cancellation on the next operational cycle.
        broker.rows[0].update(filled_qty="1", filled_avg_price="100", filled_at=received.isoformat())
        broker.holdings = (PositionLeg(account_id=config.account_id, environment="paper",
            group_id="unmanaged:TEST_STOCK", symbol="TEST_STOCK", asset_type="stock",
            quantity="1", cost_basis="100", reconciled_at=received),)
        service.run_once()
        group = service.store.projection("position_groups", "group-"+intent.id)
        assert group["state"] == "OPEN"
        assert Decimal(group["legs"][0]["quantity"]) == 1
        assert service.store.db.execute("SELECT first_fill FROM reservations").fetchone()[0] == 1
        assert len(service.store.list("fills", scope)) == 1
        assert len(broker.writes) == 1
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
        store.close()


@pytest.mark.parametrize("activity_type,option_quantity,stock_quantity", [
    ("OPEXC", "1", "100"), ("OPASN", "-1", "-100"),
])
def test_t29_assignment_or_exercise_reconciles_residual_stock(tmp_path, activity_type,
                                                            option_quantity, stock_quantity):
    store, config, scope, evidence, calendar, broker = setup(tmp_path)
    symbol = "TEST260918C00100000"
    broker.holdings = (PositionLeg(account_id=config.account_id, environment="paper",
        group_id="unmanaged:"+symbol, symbol=symbol, asset_type="option",
        quantity=option_quantity, multiplier="100", cost_basis="100", reconciled_at=NOW),)
    reconciler = Reconciler(store, config, scope, broker)
    reconciler.reconcile(NOW)
    assert store.projection("position_groups", "unmanaged:"+symbol)["state"] == "RECONCILIATION_REQUIRED"

    arrival = NOW+timedelta(seconds=1)
    activity = {"id": "fixture-"+activity_type, "activity_type": activity_type,
                "symbol": symbol, "qty": "1", "date": arrival.isoformat()}
    broker.activities = lambda: [activity]
    broker.holdings = (PositionLeg(account_id=config.account_id, environment="paper",
        group_id="unmanaged:TEST", symbol="TEST", asset_type="stock", quantity=stock_quantity,
        cost_basis="10000", market_value="10100", reconciled_at=arrival),)
    account = reconciler.reconcile(arrival)
    reconciler.reconcile(arrival)
    assert store.projection("position_groups", "unmanaged:"+symbol)["state"] == "CLOSED"
    residual = store.projection("position_groups", "unmanaged:TEST")
    assert residual["state"] == "RECONCILIATION_REQUIRED"
    assert residual["exit_policy_id"] == "manual-adoption-required"
    assert Decimal(residual["legs"][0]["quantity"]) == Decimal(stock_quantity)
    assert not PositionGroup.model_validate({key: value for key, value in residual.items()
                                           if key != "_revision"}).flat
    assert account.exposure == Decimal("10100")
    assert len(store.list("activities", scope)) == 1
    assert any(row["reason"] == "UNMANAGED_OR_ASSIGNMENT_EXPOSURE" and row["symbol"] == "TEST"
               for row in store.list("incidents", scope))
    assert broker.writes == []
    store.close()


def replacement_setup(tmp_path):
    store,config,scope,evidence,calendar,broker=setup(tmp_path,mode='paper')
    plan,intent=seed_plan(store,config,scope,evidence)
    decision=approve(RiskGate(store,config,scope,evidence,calendar),intent,broker,config)
    gateway=ExecutionGateway(store,config,scope,broker,evidence)
    gateway.submit(intent,decision,NOW)
    account=Reconciler(store,config,scope,broker).reconcile(NOW)
    return store,config,scope,evidence,broker,plan,intent,gateway,account


def replace(gateway,order_id,account,price='100.05'):
    return gateway.replace(order_id,price,NOW,account=account,entry_ready=True,
                           news_ready=True,event_calendar_ready=True)


def test_t13_unknown_replacement_retains_single_reservation_and_stable_child(tmp_path):
    store,config,scope,evidence,broker,plan,intent,gateway,account=replacement_setup(tmp_path)
    broker.timeout=True
    child=replace(gateway,intent.id,account)
    assert child['state']=='UNKNOWN'
    assert replace(gateway,intent.id,account)['id']==child['id']
    assert len(broker.writes)==2  # one submit and one replace
    reservation=store.db.execute('SELECT * FROM reservations').fetchone()
    assert reservation['intent_id']==intent.id and Decimal(reservation['capital'])==Decimal('300.15')
    assert store.db.execute('SELECT count(*) FROM reservations').fetchone()[0]==1
    Reconciler(store,config,scope,broker).reconcile(NOW)
    assert store.projection('orders',child['id'])['state']=='OPEN'
    assert store.projection('orders',intent.id)['state']=='CANCELLED'
    assert store.db.execute('SELECT state FROM reservations').fetchone()[0]!='RELEASED'
    store.close()


def test_t16_replacement_child_fill_counts_once_and_releases_only_pending_cash(tmp_path):
    store,config,scope,evidence,broker,plan,intent,gateway,account=replacement_setup(tmp_path)
    child=replace(gateway,intent.id,account)
    broker.rows[-1].update(status='canceled',filled_qty='1',filled_avg_price='100.05',filled_at=NOW.isoformat())
    broker.holdings=(PositionLeg(account_id=config.account_id,environment='paper',group_id='unmanaged:TEST_STOCK',
        symbol='TEST_STOCK',asset_type='stock',quantity='1',cost_basis='100.05',reconciled_at=NOW),)
    reconciler=Reconciler(store,config,scope,broker)
    reconciler.reconcile(NOW)
    reconciler.reconcile(NOW)
    assert len(store.list('fills'))==1
    assert store.projection('position_groups','group-'+intent.id)['state']=='OPEN'
    reservation=store.db.execute('SELECT * FROM reservations').fetchone()
    assert reservation['first_fill']==1 and reservation['state']=='EXPOSURE'
    assert Decimal(reservation['capital'])==0
    store.close()


def test_t17_late_old_fill_with_rejected_replacement_retains_managed_exposure(tmp_path):
    store,config,scope,evidence,broker,plan,intent,gateway,account=replacement_setup(tmp_path)
    child=replace(gateway,intent.id,account)
    broker.rows[0].update(status='filled',filled_qty='3',filled_avg_price='100',filled_at=NOW.isoformat())
    broker.rows[-1].update(status='rejected')
    broker.holdings=(PositionLeg(account_id=config.account_id,environment='paper',group_id='unmanaged:TEST_STOCK',
        symbol='TEST_STOCK',asset_type='stock',quantity='3',cost_basis='300',reconciled_at=NOW),)
    Reconciler(store,config,scope,broker).reconcile(NOW)
    assert store.projection('position_groups','group-'+intent.id)['state']=='OPEN'
    assert len(store.list('fills'))==1
    assert store.db.execute('SELECT first_fill FROM reservations').fetchone()[0]==1
    store.close()


def test_t15_forged_risk_decision_and_modified_intent_never_reach_broker(tmp_path):
    store,config,scope,evidence,calendar,broker=setup(tmp_path,mode='paper')
    plan,intent=seed_plan(store,config,scope,evidence)
    decision=approve(RiskGate(store,config,scope,evidence,calendar),intent,broker,config)
    gateway=ExecutionGateway(store,config,scope,broker,evidence)
    with pytest.raises(ValueError,match='durable authority'):
        gateway.submit(intent,decision.model_copy(update={'approved_quantity':Decimal('99999')}),NOW)
    with pytest.raises(ValueError,match='durable authority'):
        gateway.submit(intent.model_copy(update={'limit_price':Decimal('1000')}),decision,NOW)
    assert broker.writes==[]
    store.close()


def test_t26_closed_group_does_not_adopt_later_external_same_symbol_position(tmp_path):
    store,config,scope,evidence,broker,plan,intent,gateway,account=replacement_setup(tmp_path)
    broker.rows[0].update(status='filled',filled_qty='3',filled_avg_price='100',filled_at=NOW.isoformat())
    broker.holdings=(PositionLeg(account_id=config.account_id,environment='paper',group_id='unmanaged:TEST_STOCK',
        symbol='TEST_STOCK',asset_type='stock',quantity='3',cost_basis='300',reconciled_at=NOW),)
    reconciler=Reconciler(store,config,scope,broker)
    reconciler.reconcile(NOW)
    broker.holdings=()
    reconciler.reconcile(NOW)
    assert store.projection('position_groups','group-'+intent.id)['state']=='CLOSED'
    broker.holdings=(PositionLeg(account_id=config.account_id,environment='paper',group_id='unmanaged:TEST_STOCK',
        symbol='TEST_STOCK',asset_type='stock',quantity='1',cost_basis='100',reconciled_at=NOW),)
    reconciler.reconcile(NOW)
    assert store.projection('position_groups','group-'+intent.id)['state']=='CLOSED'
    assert store.projection('position_groups','unmanaged:TEST_STOCK')['state']=='RECONCILIATION_REQUIRED'
    store.close()


def test_t30_reconciliation_lookup_happens_outside_account_transaction(tmp_path):
    store,config,scope,evidence,broker,plan,intent,gateway,account=replacement_setup(tmp_path)
    rows=broker.rows
    broker.orders=lambda: []
    def lookup(client_id):
        assert not store.db.in_transaction
        return rows[0]
    broker.order_by_client_id=lookup
    Reconciler(store,config,scope,broker).reconcile(NOW)
    store.close()


def test_t04_invalid_financial_data_preserves_operational_holdings_and_close_authority(tmp_path):
    store,config,scope,evidence,calendar,broker=setup(tmp_path)
    def invalid(*args): raise ValueError('non-finite equity')
    broker.account=invalid
    broker.identity=lambda: {'account_id':config.account_id,'environment':'paper'}
    position=PositionLeg(account_id=config.account_id,environment='paper',group_id='unmanaged:TEST_STOCK',
        symbol='TEST_STOCK',asset_type='stock',quantity='1',cost_basis='100',reconciled_at=NOW)
    broker.holdings=(position,)
    actual=Reconciler(store,config,scope,broker).reconcile(NOW)
    assert isinstance(actual,AccountOperationalState)
    assert actual.positions[0].quantity==1
    assert 'equity' not in actual.model_dump()
    evidence.ingest_market(MarketEvent(symbol='TEST_STOCK',source='fixture',source_version='exit-quote',
        event_type='quote',effective_event_time=NOW,received_at=NOW,available_at=NOW,
        feed='sip',quality_class='genuine',bid='100',ask='100.01'))
    intent=OrderIntent(account_id=config.account_id,environment='paper',purpose='close',group_id=position.group_id,
        legs=(OrderLeg(instrument=Instrument(symbol='TEST_STOCK',liquidity_capacity='1'),side='sell',position_intent='sell_to_close'),),
        requested_quantity='1',limit_price='100',expires_at=NOW+timedelta(seconds=10),reason='confirmed protective exit')
    result=RiskGate(store,config,scope,evidence,calendar).approve(intent,actual,NOW,
        entry_ready=False,news_ready=False,event_calendar_ready=False)
    assert result.approved,result.reasons
    store.close()


def test_t13_restart_rejects_reserved_work_that_never_reached_submitting(tmp_path):
    store,config,scope,evidence,calendar,broker=setup(tmp_path,mode='paper')
    plan,intent=seed_plan(store,config,scope,evidence)
    decision=approve(RiskGate(store,config,scope,evidence,calendar),intent,broker,config)
    assert decision.approved
    store.close()
    store=Store(tmp_path/'session.db')
    Reconciler(store,config,scope,broker).reconcile(NOW)
    order=store.projection('orders',intent.id)
    assert order['state']=='REJECTED' and order['reasons']==['PRE_SUBMISSION_RECOVERY']
    assert store.db.execute('SELECT state FROM reservations').fetchone()[0]=='RELEASED'
    assert broker.writes==[]
    store.close()


def test_replacements_are_bounded(tmp_path):
    store,config,scope,evidence,broker,plan,intent,gateway,account=replacement_setup(tmp_path)
    first=replace(gateway,intent.id,account)
    account=Reconciler(store,config,scope,broker).reconcile(NOW)
    second=replace(gateway,first['id'],account,'100.06')
    account=Reconciler(store,config,scope,broker).reconcile(NOW)
    with pytest.raises(ValueError,match='ATTEMPT_LIMIT'):
        replace(gateway,second['id'],account,'100.07')
    assert store.db.execute('SELECT count(*) FROM reservations').fetchone()[0]==1
    assert len(broker.writes)==3
    store.close()


def test_cancel_failure_on_parent_keeps_cancelling_replacement_child(tmp_path):
    store,config,scope,evidence,broker,plan,intent,gateway,account=replacement_setup(tmp_path)
    child=replace(gateway,intent.id,account)
    calls=[]
    def cancel(broker_id):
        calls.append(broker_id)
        assert store.projection('orders',intent.id if broker_id==broker.rows[0]['id'] else child['id'])['state']=='CANCEL_PENDING'
        if broker_id==broker.rows[0]['id']:
            raise TimeoutError('cancellation status unknown')
    broker.cancel=cancel
    gateway.cancel_entries(NOW)
    assert len(calls)==2
    assert store.projection('orders',child['id'])['state']=='CANCEL_PENDING'
    assert store.db.execute('SELECT state FROM reservations').fetchone()[0]!='RELEASED'
    store.close()


def seed_confirmed_entry(store, config, scope, evidence, calendar, broker):
    plan, intent = seed_plan(store, config, scope, evidence)
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = approve(gate, intent, broker, config)
    assert decision.approved, decision.reasons
    ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    broker.rows[0].update(status="filled", filled_qty="1", filled_avg_price="100",
                          filled_at=NOW.isoformat())
    broker.holdings = (PositionLeg(account_id=config.account_id, environment="paper",
        group_id="unmanaged:TEST_STOCK", symbol="TEST_STOCK", asset_type="stock",
        quantity="1", cost_basis="100", reconciled_at=NOW),)
    account = Reconciler(store, config, scope, broker).reconcile(NOW)
    evidence.ingest_market(MarketEvent(symbol="TEST_STOCK", source="fixture",
        source_version="held-current-quote", event_type="quote", effective_event_time=NOW,
        received_at=NOW, available_at=NOW, feed="sip", quality_class="genuine",
        bid="100", ask="100.01"))
    return plan, intent, account


def close_confirmed_entry(config, entry):
    return OrderIntent(account_id=config.account_id, environment=config.environment,
        purpose="close", group_id="group-"+entry.id,
        legs=(OrderLeg(instrument=entry.legs[0].instrument, side="sell",
                      position_intent="sell_to_close"),),
        requested_quantity="1", limit_price="100", expires_at=NOW+timedelta(seconds=10),
        reason="synthetic confirmed holding exit")


@pytest.mark.parametrize("session_offset", [-1, 1])
def test_t09_wrong_session_plan_rejected_while_existing_group_remains_open(tmp_path,
                                                                        session_offset):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, held_entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    baseline, proposal = seed_plan(store, config, scope, evidence, "TEST_OTHER")
    other = calendar.get("fixture-session").model_copy(update={
        "id": "different-session", "session_date": (NOW+timedelta(days=session_offset)).date().isoformat(),
        "open": NOW.replace(hour=13, minute=30)+timedelta(days=session_offset),
        "close": NOW.replace(hour=20, minute=0)+timedelta(days=session_offset)})
    calendar.sessions[other.id] = other
    store.put("sessions", other, scope)
    wrong = baseline.model_copy(update={"id": "wrong-session-plan", "target_session_id": other.id})
    store.put("plans", wrong, scope)
    for state in ("RESEARCH_COMPLETE", "PENDING_VALIDATION", "ELIGIBLE"):
        store.transition_plan(wrong.id, state, scope=scope)
    proposal = proposal.model_copy(update={"plan_id": wrong.id})
    gate = RiskGate(store, config, scope, evidence, calendar)
    result = gate.approve(proposal, account, NOW, entry_ready=True, news_ready=True,
                          event_calendar_ready=True)
    assert not result.approved and "ENTRY_WINDOW_CLOSED" in result.reasons
    assert store.db.execute("SELECT count(*) FROM reservations WHERE intent_id=?",
                            (proposal.id,)).fetchone()[0] == 0
    assert store.projection("position_groups", "group-"+held_entry.id)["state"] == "OPEN"
    assert len(broker.writes) == 1
    store.close()


def test_t09_superseded_plan_revokes_pending_authority_and_preserves_holdings(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, held_entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    plan, intent = seed_plan(store, config, scope, evidence, "TEST_OTHER")
    decision = RiskGate(store, config, scope, evidence, calendar).approve(intent, account, NOW,
        entry_ready=True, news_ready=True, event_calendar_ready=True)
    assert decision.approved, decision.reasons
    store.transition_plan(plan.id, "SUPERSEDED", ("REPLACED_BY_REVIEWED_PLAN",), scope=scope)
    result = ExecutionGateway(store, config, scope, broker, evidence).submit(intent, decision, NOW)
    assert result["state"] == "REJECTED"
    assert "PLAN_NO_LONGER_ELIGIBLE" in result["reasons"]
    assert store.db.execute("SELECT state FROM reservations WHERE intent_id=?",
                            (intent.id,)).fetchone()[0] == "RELEASED"
    assert store.projection("position_groups", "group-"+held_entry.id)["state"] == "OPEN"
    assert len(broker.writes) == 1
    store.close()


def test_t18_broker_rejected_exit_preserves_open_group_and_confirmed_quantity(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    exit_intent = close_confirmed_entry(config, entry)
    gate = RiskGate(store, config, scope, evidence, calendar)
    decision = gate.approve(exit_intent, account, NOW, entry_ready=False, news_ready=False,
                             event_calendar_ready=False)
    assert decision.approved, decision.reasons
    ExecutionGateway(store, config, scope, broker, evidence).submit(exit_intent, decision, NOW)
    broker.rows[-1].update(status="rejected", filled_qty="0")
    Reconciler(store, config, scope, broker).reconcile(NOW)
    assert store.projection("orders", exit_intent.id)["state"] == "REJECTED"
    group = store.projection("position_groups", "group-"+entry.id)
    assert group["state"] == "OPEN"
    assert Decimal(group["legs"][0]["quantity"]) == 1
    assert not PositionGroup.model_validate({key: value for key, value in group.items()
                                           if key != "_revision"}).flat
    retry = close_confirmed_entry(config, entry)
    retry_decision = gate.approve(retry, account, NOW, entry_ready=False, news_ready=False,
                                  event_calendar_ready=False)
    assert retry_decision.approved, retry_decision.reasons
    assert len(broker.writes) == 2  # entry and rejected exit; no implicit resend
    store.close()


@pytest.mark.parametrize("blocker,reason", [
    ("daily_loss", "DAILY_LOSS_PAUSE"), ("entry_quota", "ENTRY_QUOTA_EXHAUSTED"),
])
def test_t25_entry_pause_limit_rejects_new_group_but_allows_confirmed_close(tmp_path,
                                                                         blocker, reason):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    if blocker == "entry_quota":
        config = config.model_copy(update={"policy": config.policy.model_copy(
            update={"max_new_entry_groups_per_session": 1})})
    _, entry, account = seed_confirmed_entry(store, config, scope, evidence, calendar, broker)
    if blocker == "daily_loss":
        account = account.model_copy(update={"daily_pnl": -account.equity*config.policy.daily_loss_pause_pct})
    _, new_intent = seed_plan(store, config, scope, evidence, "TEST_OTHER")
    gate = RiskGate(store, config, scope, evidence, calendar)
    blocked = gate.approve(new_intent, account, NOW, entry_ready=True, news_ready=True,
                           event_calendar_ready=True)
    assert not blocked.approved and reason in blocked.reasons
    assert store.db.execute("SELECT count(*) FROM reservations WHERE intent_id=?",
                            (new_intent.id,)).fetchone()[0] == 0
    closing = gate.approve(close_confirmed_entry(config, entry), account, NOW,
        entry_ready=False, news_ready=False, event_calendar_ready=False)
    assert closing.approved, closing.reasons
    assert closing.approved_quantity == 1
    assert store.projection("position_groups", "group-"+entry.id)["state"] == "OPEN"
    assert len(broker.writes) == 1
    store.close()


def test_t55_foreign_account_plan_cannot_reserve_or_reach_gateway(tmp_path):
    store, config, scope, evidence, calendar, broker = setup(tmp_path, mode="paper")
    original, intent = seed_plan(store, config, scope, evidence)
    foreign = original.model_copy(update={"id": "foreign-account-plan", "account_id": "other-account"})
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
    config = RuntimeConfig.model_validate({**base.model_dump(), "database_path": str(tmp_path/"session.db"),
        "data_profile": "SUBSCRIBED_REALTIME", "policy": RiskPolicy(options_enabled=True,
                                                                    spreads_enabled=True)})
    evidence.config = config
    evidence.clock = EvidenceClock(config.data_profile, 0)
    original, _ = seed_plan(store, base, scope, evidence)
    store.transition_plan(original.id, "SUPERSEDED", scope=scope)
    alternative, quotes = alternatives()[0][0], alternatives()[2]
    permit = approval()
    store.put("strategy_approvals", permit, scope)
    for quote in quotes.values():
        evidence.ingest_market(quote)
    evidence.ingest_market(MarketEvent(symbol="TEST_STOCK", source="fixture", source_version="current-signal",
        event_type="bar", effective_event_time=NOW-timedelta(seconds=1), received_at=NOW,
        available_at=NOW, interval_start=NOW-timedelta(seconds=61), feed="sip", quality_class="genuine",
        price="100"))
    service = SessionService(config, broker=broker, calendar=calendar, clock=lambda: NOW)
    try:
        frozen = broker.account(config.policy, NOW).model_copy(update={"id": "close-account-before-restriction",
            "equity": Decimal("20000"), "options_level": 3, "options_buying_power": Decimal("10000")})
        service.store.put("account_profiles", frozen, scope)
        original_report = service.suitability.assess(frozen, [alternative], config.data_profile, [permit])
        assert original_report.outcome == "FEASIBLE"
        service.store.put("suitability_reports", original_report, scope)
        plan = original.model_copy(update={"id": "review-options-permission", "data_profile": config.data_profile,
            "alternative": alternative, "approval_id": permit.id, "max_buy_price": Decimal("1.10"),
            "account_profile_id": frozen.id, "suitability_report_id": original_report.id})
        service.store.put("plans", plan, scope)
        for state in ("RESEARCH_COMPLETE", "PENDING_VALIDATION"):
            service.store.transition_plan(plan.id, state, scope=scope)
        service.account_profile = frozen
        service.price_ready = service.news_ready = service.event_calendar_ready = True
        PreGauss().run(service, calendar.get("fixture-session"), NOW)
        assert service.store.projection("plan_state", plan.id)["state"] == "ELIGIBLE"

        changed = frozen.model_copy(update={"id": "pre-account-permission-revoked", "options_level": 2})
        service.store.put("account_profiles", changed, scope)
        service.account_profile = changed
        results = PreGauss().run(service, calendar.get("fixture-session"), NOW)
        validation = next(service.store.get("validations", key) for key in results
                          if service.store.get("validations", key)["plan_id"] == plan.id)
        assert validation["account_profile_id"] == changed.id
        assert validation["outcome"] == "DEFERRED"
        assert "OPTION_SPREAD_PERMISSION_REQUIRED" in validation["reasons"]
        assert service.store.get("suitability_reports", original_report.id)["outcome"] == "FEASIBLE"
        assert service.store.get("plans", plan.id)["account_profile_id"] == frozen.id
        intent = OrderIntent(account_id=config.account_id, environment="paper", purpose="open",
            plan_id=plan.id, plan_version=1, legs=opening_legs(alternative), limit_price="1.01",
            maximum_buy_price="1.10", expires_at=NOW+timedelta(seconds=10), reason="stale pre-restriction proposal")
        final = service.risk.approve(intent, changed, NOW, entry_ready=True, news_ready=True,
                                     event_calendar_ready=True)
        assert not final.approved and "OPTION_SPREAD_PERMISSION_REQUIRED" in final.reasons
        assert service.store.db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0
        assert broker.writes == []
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)
        store.close()
