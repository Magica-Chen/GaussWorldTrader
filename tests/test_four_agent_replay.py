"""Isolated two-session causal replay across profiles and actual-account constraints."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

from src.runtime.evidence import EvidenceClock, evaluate_rules
from src.runtime.models import MarketEvent, Rule
from src.runtime.service import replay_fixture

FIXTURE = Path(__file__).parent / 'fixtures' / 'gauss_session_day.json'
CASES = {
    'below_minimum': {'equity': '100', 'cash': '100', 'volume': None, 'binding': 'capital_allocation'},
    'feasible': {'equity': '10000', 'cash': '10000', 'volume': None, 'binding': 'capital_allocation'},
    'cash_poor_equal_equity': {'equity': '10000', 'cash': '10', 'volume': None, 'binding': 'deployable_capital'},
    'capacity_limited_large': {'equity': '1000000', 'cash': '1000000', 'volume': '200', 'binding': 'liquidity_capacity'},
}


@pytest.mark.parametrize('profile', ['FREE_DELAYED', 'SUBSCRIBED_REALTIME'])
@pytest.mark.parametrize('case', CASES)
def test_two_session_profile_account_matrix(tmp_path, profile, case):
    original = json.loads(FIXTURE.read_text())
    fixture = deepcopy(original)
    constraints = CASES[case]
    from uuid import uuid4
    account_id = "matrix-fixture-" + uuid4().hex
    fixture["config"]["account_id"] = account_id
    for approval in fixture.get("approvals", []):
        approval["account_id"] = account_id
    fixture['config'].update(data_profile=profile, signal_timeframe='1Min',
                              database_path=str(tmp_path/'unused-config.sqlite3'))
    accounts = [fixture['account']] + [step['account'] for step in fixture['steps'] if 'account' in step]
    for account in accounts:
        account.update(account_id=account_id, equity=constraints['equity'], cash=constraints['cash'],
                       buying_power=constraints['cash'])
    if constraints['volume']:
        for step in fixture['steps']:
            for event in step.get('market_events', []):
                event['volume'] = constraints['volume']
    fixture_path = tmp_path/'fixture.json'
    fixture_path.write_text(json.dumps(fixture))
    report = replay_fixture(fixture_path, database_path=tmp_path/'isolated.sqlite3')

    assert original == json.loads(FIXTURE.read_text()), 'Shared fixture must remain unchanged'
    assert report['broker_writes'] == 0
    assert report['status']['data_profile'] == profile
    assert report['status']['execution_mode'] == 'shadow'
    assert report['status']['configured_delay_seconds'] == (900 if profile == 'FREE_DELAYED' else 0)
    assert {run['role'] for run in report['agent_runs']} == {'PostGauss', 'CloseGauss', 'PreGauss', 'LiveGauss'}
    assert all(run['state'] == 'COMPLETED' for run in report['agent_runs'])
    assert all(run['data_profile'] == profile for run in report['agent_runs'])
    assert all(snapshot['signal_timeframe'] == '1Min' for snapshot in report['snapshots'])
    assert all(snapshot['data_context']['data_profile'] == profile for snapshot in report['snapshots'])
    assert all(row['data_profile'] == profile for row in report['suitability_reports'])
    assert not report['fills']
    assert all(order['state'] == 'SHADOW' and Decimal(order['filled_quantity']) == 0
               for order in report['orders'])

    snapshots = {row['id'] for row in report['snapshots']}
    suitability = {row['id'] for row in report['suitability_reports']}
    for plan in report['plans']:
        assert plan['snapshot_id'] in snapshots
        assert plan['suitability_report_id'] in suitability
        assert plan['data_profile'] == profile
        assert plan['account_id'] == fixture['account']['account_id']
    sizing = [choice for row in report['suitability_reports'] for choice in row['alternatives']]
    assert sizing
    assert any(choice['binding_limit'] == constraints['binding'] for choice in sizing)
    if case in {'below_minimum', 'cash_poor_equal_equity'}:
        assert not report['orders']
        assert not report['plans']
        assert all(row['outcome'] == 'NO_TRADE' for row in report['suitability_reports'])
        assert any('NO_FEASIBLE_SIZE' in choice['reasons'] for choice in sizing)
    else:
        assert report['orders'], 'Feasible replay branch must reach a safe shadow intent'
        assert report['plans']
        assert any(event['state'] == 'INVALIDATED' and 'CURRENT_NEWS_SAFETY_VETO' in event['reasons']
                   for event in report['plan_events'])
        if case == 'capacity_limited_large':
            assert all(Decimal(order['quantity']) <= 2 for order in report['orders'])
            assert all(Decimal(choice['quantity']) <= 2 for choice in sizing)


def test_delayed_trigger_cannot_revive_expired_entry_window():
    event_time = datetime(2026, 9, 9, 19, 29, tzinfo=timezone.utc)
    observed_at = event_time + timedelta(minutes=15, seconds=5)
    expires_at = event_time + timedelta(minutes=1)
    event = MarketEvent(symbol='TEST', source='fixture', source_version='expiry',
                        event_type='bar', effective_event_time=event_time,
                        interval_start=event_time-timedelta(minutes=1),
                        received_at=observed_at, available_at=observed_at,
                        price=Decimal('101'), open_price=Decimal('100'),
                        high_price=Decimal('101'), low_price=Decimal('100'),
                        volume=Decimal('100'), complete=True, feed='sip', quality_class='genuine')
    clock = EvidenceClock('FREE_DELAYED')
    assert clock.eligible(event, observed_at), 'Delayed observation itself is causally eligible'
    price_rule = Rule(rule_type='completed_bar_condition', operator='above', value=Decimal('100'))
    assert evaluate_rules([price_rule], [event], observed_at)
    assert not evaluate_rules([price_rule, Rule(rule_type='time_in_window')], [event], observed_at,
                              valid_from=event_time-timedelta(minutes=10), expires_at=expires_at)
