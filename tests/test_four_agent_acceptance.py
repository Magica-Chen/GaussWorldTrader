"""Independent synthetic acceptance checks; T numbers refer to the v1.1 plan."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.runtime.calendar import SessionCalendar
from src.runtime.evidence import EvidenceClock, EvidenceService, SnapshotDataReader
from src.runtime.models import (
    AccountProfile, Alternative, Instrument, MarketEvent, NewsEvent, PositionGroup,
    PositionLeg, RiskPolicy, RuntimeConfig, StrategyApproval, TradingSession,
)
from src.runtime.store import Store
from src.runtime.suitability import AccountSuitabilityService, compare_capital


WALL = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)


def account(**changes):
    values = dict(account_id="synthetic-account", environment="paper", hypothetical=False,
                  equity="10000", cash="10000", buying_power="10000", currency="USD",
                  observed_at=WALL, mandate_id="operator-mandate-v1")
    values.update(changes)
    return AccountProfile(**values)


def observation(**changes):
    values = dict(symbol="TEST_STOCK", source="fixture", source_version="1", event_type="bar",
                  effective_event_time=WALL-timedelta(seconds=902), received_at=WALL,
                  available_at=WALL, feed="sip", quality_class="genuine", price="100",
                  interval_start=WALL-timedelta(seconds=1202), complete=True)
    values.update(changes)
    return MarketEvent(**values)


def alternative(**changes):
    values = dict(strategy_id="fixture", strategy_version="1", instrument=Instrument(
        symbol="TEST_STOCK", liquidity_capacity="3"), entry_price="100", stop_price="99",
        expected_net_value="1", evidence_ids=("fixture-history",))
    values.update(changes)
    return Alternative(**values)


def approval(**changes):
    values = dict(strategy_id="fixture", strategy_version="1", account_id="synthetic-account",
                  environment="paper", approved_by="fixture-operator",
                  profiles=("FREE_DELAYED", "SUBSCRIBED_REALTIME"),
                  maximum_signal_latency_seconds=1200, minimum_history=2, requires_news=False,
                  requires_event_calendar=False, requires_current_quote=False,
                  validation_reference="fixture-only", validated_net_expectancy="1",
                  expiry=WALL+timedelta(days=2))
    values.update(changes)
    return StrategyApproval(**values)


def test_supported_ranking_compares_expected_value_per_unit_risk():
    safer = alternative(id='safer', entry_price='100', stop_price='99', expected_net_value='2')
    nominal = alternative(id='nominal', entry_price='100', stop_price='90', expected_net_value='5')
    report = AccountSuitabilityService(RiskPolicy()).assess(account(), [nominal, safer],
                                                          'FREE_DELAYED', [approval()])
    assert report.selected_alternative_id == safer.id
    assert report.ranking_scores[0][0] == safer.id
    assert report.objective == 'validated_net_expectancy_per_unit_planned_loss'


@pytest.mark.parametrize("offset,eligible", [(901, False), (902, True), (903, True)])
def test_t36_buffered_delayed_cutoff(offset, eligible):
    event = observation(effective_event_time=WALL-timedelta(seconds=offset))
    assert EvidenceClock("FREE_DELAYED", 2).eligible(event, WALL) is eligible


def test_t37_completed_bar_end_controls_release():
    event = observation(interval_start=WALL-timedelta(seconds=1200),
                        effective_event_time=WALL-timedelta(seconds=899))
    assert EvidenceClock("FREE_DELAYED", 0).eligible(event, WALL) is False


def test_t40_late_receipt_does_not_introduce_second_delay():
    event = observation(effective_event_time=WALL-timedelta(minutes=25),
                        interval_start=WALL-timedelta(minutes=30))
    assert EvidenceClock("FREE_DELAYED").eligible(event, WALL)
    assert not EvidenceClock("FREE_DELAYED").eligible(event, WALL-timedelta(microseconds=1))


def test_t42_indicative_quote_never_becomes_genuine_execution_quote(tmp_path):
    store = Store(tmp_path / "state.db")
    service = EvidenceService(store, RuntimeConfig(symbols=("TEST_STOCK",)), "paper:synthetic-account")
    service.ingest_market(observation(event_type="quote", feed="indicative", quality_class="indicative",
                                     price=None, bid="1", ask="1.01", interval_start=None,
                                     effective_event_time=WALL))
    assert service.current_quote("TEST_STOCK", WALL, "opra") is None
    store.close()


def test_t45_current_safety_news_is_excluded_from_delayed_signal(tmp_path):
    store = Store(tmp_path / "state.db")
    service = EvidenceService(store, RuntimeConfig(), "paper:synthetic-account")
    service.ingest_news(NewsEvent(provider="fixture", article_id="announcement", source_version="1",
                                 published_at=WALL, updated_at=WALL, received_at=WALL, available_at=WALL,
                                 headline="Synthetic material event", content_hash="source-v1",
                                 symbols=("TEST_STOCK",), verified=True, blocking=True))
    assert service.news(WALL) == []
    assert len(service.blockers("TEST_STOCK", WALL)) == 1
    store.close()


def test_t22_snapshot_preserves_original_version_after_correction(tmp_path):
    store = Store(tmp_path / "state.db")
    service = EvidenceService(store, RuntimeConfig(symbols=("TEST_STOCK",)), "paper:synthetic-account")
    service.ingest_market(observation())
    snapshot = service.freeze("fixture-session", account(), WALL)
    reader = SnapshotDataReader(store, snapshot)
    service.ingest_market(observation(source_version="2", price="120", available_at=WALL+timedelta(seconds=1)))
    assert reader.market("TEST_STOCK")[0].price == 100
    assert service.market(WALL, "TEST_STOCK")[0].price == 100
    assert service.market(WALL+timedelta(seconds=1), "TEST_STOCK")[0].price == 120
    store.close()


def test_t05_early_close_and_holiday_calendar_uses_recorded_sessions():
    early = TradingSession(id="holiday-week", session_date="2026-11-27", calendar_version="fixture",
                           open=datetime(2026, 11, 27, 14, 30, tzinfo=timezone.utc),
                           close=datetime(2026, 11, 27, 18, tzinfo=timezone.utc))
    calendar = SessionCalendar((early,))
    assert not calendar.is_open(datetime(2026, 11, 26, 16, tzinfo=timezone.utc))
    assert early.schedule(RiskPolicy())["closing_start"].hour == 17
    assert early.schedule(RiskPolicy())["closing_start"].minute == 45
    assert not calendar.permits_entry(datetime(2026, 11, 27, 17, 31, tzinfo=timezone.utc), RiskPolicy())


def test_t06_exhausted_calendar_blocks_entry():
    with pytest.raises(RuntimeError, match="CALENDAR"):
        SessionCalendar().permits_entry(WALL, RiskPolicy())


def test_t10_stale_plan_writer_fails_without_appending_event(tmp_path):
    store = Store(tmp_path / "state.db")
    store.transition_plan("plan", "RESEARCH_COMPLETE", expected_revision=0)
    with pytest.raises(ValueError, match="revision conflict"):
        store.transition_plan("plan", "PENDING_VALIDATION", expected_revision=0)
    assert store.projection("plan_state", "plan")["state"] == "RESEARCH_COMPLETE"
    assert len(store.list("plan_events")) == 1
    store.close()


def test_t30_outbox_consumer_effect_and_offset_rollback_together(tmp_path):
    store = Store(tmp_path / "state.db")
    store.emit("fixture", {"id": "source"})
    def failing(topic, payload):
        store.put("effects", {"id": "effect"})
        raise RuntimeError("fixture consumer crash")
    with pytest.raises(RuntimeError, match="consumer crash"):
        store.consume("consumer", failing)
    assert store.get("effects", "effect") is None
    calls = []
    store.consume("consumer", lambda topic, payload: calls.append(payload["id"]))
    store.consume("consumer", lambda topic, payload: calls.append(payload["id"]))
    assert calls == ["source"]
    store.close()


def test_t31_second_runtime_cannot_acquire_same_account_lease(tmp_path):
    first = Store(tmp_path / "state.db")
    second = Store(tmp_path / "state.db")
    first.acquire_lease("paper:synthetic-account", "owner-1", now=WALL)
    with pytest.raises(RuntimeError, match="another.*(runtime|process)"):
        second.acquire_lease("paper:synthetic-account", "owner-2", now=WALL)
    first.close()
    second.close()


@pytest.mark.parametrize("field,value", [("equity", "NaN"), ("cash", "Infinity"), ("buying_power", "-1")])
def test_t04_invalid_balances_reject_without_fallback(field, value):
    with pytest.raises(ValueError):
        account(**{field: value})


def test_t48_equal_equity_different_cash_changes_feasibility():
    suitability = AccountSuitabilityService(RiskPolicy())
    assert suitability.size(account(), alternative()).feasible
    blocked = suitability.size(account(cash="0", buying_power="0"), alternative())
    assert not blocked.feasible
    assert "NO_FEASIBLE_SIZE" in blocked.reasons


def test_t51_larger_capital_does_not_exceed_liquidity_capacity():
    sizing = AccountSuitabilityService(RiskPolicy()).size(
        account(equity="1000000", cash="1000000", buying_power="1000000"), alternative())
    assert sizing.quantity == 3
    assert sizing.binding_limit == "liquidity_capacity"


def test_t53_insufficient_evidence_returns_no_trade():
    report = AccountSuitabilityService(RiskPolicy()).assess(
        account(), (alternative(expected_net_value=None),), "FREE_DELAYED", (approval(),))
    assert report.outcome == "NO_TRADE"
    assert "INSUFFICIENT_EVIDENCE" in report.alternatives[0].reasons


def test_t55_comparison_rejects_actual_scope_and_preserves_input_balance():
    actual = account()
    with pytest.raises(ValueError, match="scenario scope"):
        compare_capital([actual.model_dump(), actual.model_dump()], (), "FREE_DELAYED", RiskPolicy())
    assert actual.equity == 10000


def test_t28_opposite_option_legs_are_open_even_with_zero_net_contracts():
    legs = tuple(PositionLeg(account_id="synthetic-account", environment="paper", group_id="spread",
                             symbol=symbol, asset_type="option", quantity=quantity, multiplier="100",
                             reconciled_at=WALL) for symbol, quantity in (("TEST_CALL_1", "1"), ("TEST_CALL_2", "-1")))
    group = PositionGroup(account_id="synthetic-account", environment="paper", exit_policy_id="fixed-v1",
                          legs=legs, state="OPEN")
    assert sum(leg.quantity for leg in group.legs) == 0
    assert not group.flat


def test_t18_zero_holdings_with_unknown_order_are_not_flat():
    group = PositionGroup(account_id="synthetic-account", environment="paper", exit_policy_id="fixed-v1",
                          legs=(), unresolved_orders=("unknown-intent",), state="RECONCILIATION_REQUIRED")
    assert not group.flat
