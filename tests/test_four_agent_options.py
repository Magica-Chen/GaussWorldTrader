"""Option units, selectors, current premium quotes, and combined-entry authority."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.runtime.models import (AccountProfile, Alternative, Instrument, MarketEvent, OptionSelector,
    OrderIntent, PositionGroup, PositionLeg, RiskPolicy, RuntimeConfig, StrategyApproval)
from src.runtime.options import opening_legs, quote_checks, select_alternatives, structure_error
from src.runtime.suitability import AccountSuitabilityService
from src.runtime.risk import RiskGate, ExecutionGateway, liquidation_mark

NOW = datetime(2026, 9, 8, 15, tzinfo=timezone.utc)


def contract(symbol, strike, kind='call', capacity='3'):
    return Instrument(symbol=symbol, asset_type='option', multiplier='100', liquidity_capacity=capacity,
        contract_id='contract-'+symbol, expiry='2026-09-18', underlying='TEST_STOCK',
        option_type=kind, strike=strike)


def quote(symbol, bid, ask, *, age=0, feed='opra'):
    return MarketEvent(symbol=symbol, source='fixture', source_version='1', event_type='quote',
        effective_event_time=NOW-timedelta(seconds=age), received_at=NOW, available_at=NOW,
        feed=feed, quality_class='genuine' if feed=='opra' else 'indicative', bid=bid, ask=ask,
        bid_size='3', ask_size='3')


def approval(structure='debit_call_vertical'):
    return StrategyApproval(strategy_id='fixture', strategy_version='v1', account_id='fixture-account',
        environment='paper', approved_by='fixture', profiles=('SUBSCRIBED_REALTIME',),
        execution_modes=('shadow', 'paper'), asset_type='option', required_feed='opra',
        maximum_signal_latency_seconds=120, minimum_history=2, requires_news=False,
        requires_event_calendar=False, validation_reference='synthetic-test-only',
        validated_net_expectancy='1', expiry=NOW+timedelta(days=1),
        option_selector=OptionSelector(structure=structure))


def alternatives(structure='debit_call_vertical'):
    a, b = contract('TESTCALL100','100'), contract('TESTCALL105','105')
    quotes = {a.symbol: quote(a.symbol,'2.00','2.01'), b.symbol: quote(b.symbol,'1.00','1.01')}
    values, reasons = select_alternatives(approval(structure), (a,b), quotes,
                                          Decimal('100'), Decimal('99'), NOW)
    return values, reasons, quotes


def test_selects_only_standard_debit_vertical_and_keeps_signal_units():
    values, _, _ = alternatives()
    assert len(values) == 1
    item = values[0]
    assert item.entry_price == Decimal('1.01')
    assert item.signal_symbol == 'TEST_STOCK'
    assert item.underlying_entry_price == 100
    assert len(opening_legs(item)) == 2
    assert {leg.position_intent for leg in opening_legs(item)} == {'buy_to_open','sell_to_open'}


def test_indicative_or_skewed_quotes_never_pass_execution():
    assert quote_checks([quote('X','1','1.01',feed='indicative')], NOW,max_age=2)
    assert quote_checks([quote('X','1','1.01'), quote('Y','1','1.01',age=2)],
                        NOW,max_age=2,max_skew=Decimal('1')) == 'ASYNCHRONOUS_LEG_QUOTES'


def test_spread_permission_and_minimum_size_and_capacity():
    item = alternatives()[0][0]
    policy = RiskPolicy(options_enabled=True,spreads_enabled=True)
    account = AccountProfile(account_id='fixture-account',environment='paper',currency='USD',
        equity='10000',cash='10000',buying_power='10000',observed_at=NOW,
        mandate_id=policy.id,options_level=3,options_buying_power='10000')
    evaluator = AccountSuitabilityService(policy)
    assert not evaluator.size(account,item).feasible  # $101 cannot fit a $100 loss ceiling.
    funded = account.model_copy(update={'equity':Decimal('10100')})
    assert evaluator.size(funded,item).quantity == 1
    bigger = account.model_copy(update={'equity':Decimal('1000000'), 'cash':Decimal('1000000'),
                                        'buying_power':Decimal('1000000')})
    assert evaluator.size(bigger,item).quantity == 3  # actual displayed capacity, not equity tiers.
    denied = evaluator.size(funded.model_copy(update={'options_level':2}),item)
    assert 'OPTION_SPREAD_PERMISSION_REQUIRED' in denied.reasons
    restricted = evaluator.size(funded.model_copy(update={'options_buying_power':Decimal('0')}),item)
    assert not restricted.feasible
    unknown = evaluator.size(funded.model_copy(update={'options_buying_power':None}),item)
    assert 'OPTION_BUYING_POWER_UNAVAILABLE' in unknown.reasons


def test_invalid_credit_direction_or_debit_width_rejected():
    item = alternatives()[0][0]
    assert structure_error(item.model_copy(update={'entry_price':Decimal('5')})) == 'INVALID_DEBIT_WIDTH'
    inverse = item.model_copy(update={'instrument':item.short_instrument, 'short_instrument':item.instrument})
    assert structure_error(inverse) == 'INVALID_DEBIT_WIDTH'


def test_zero_net_contracts_liquidation_is_signed_premium_value():
    item, quotes = alternatives()[0][0], alternatives()[2]
    legs = tuple(PositionLeg(account_id='fixture-account',environment='paper',group_id='spread',
        symbol=instrument.symbol,asset_type='option',quantity=str(quantity),multiplier='100',
        reconciled_at=NOW) for instrument,quantity in ((item.instrument,1),(item.short_instrument,-1)))
    group = PositionGroup(id='spread',account_id='fixture-account',environment='paper',exit_policy_id='test',
                          legs=legs,state='OPEN')
    assert not group.flat
    assert liquidation_mark(group,quotes,NOW,max_age=2) == Decimal('99')


def test_full_multi_leg_shadow_intent_passes_shared_risk_gate(tmp_path):
    from test_four_agent_gateway import setup, seed_plan
    store, base, scope, evidence, calendar, broker = setup(tmp_path)
    policy = RiskPolicy(options_enabled=True,spreads_enabled=True)
    config = RuntimeConfig(account_id=base.account_id, data_profile='SUBSCRIBED_REALTIME',
                           execution_mode='shadow', policy=policy,strategy_allowlist=('fixture',))
    evidence.config = config
    from src.runtime.evidence import EvidenceClock
    evidence.clock = EvidenceClock(config.data_profile,0)
    old, _ = seed_plan(store,base,scope,evidence)
    item, quotes = alternatives()[0][0], alternatives()[2]
    permit = approval()
    plan = old.model_copy(update={'id':'multi-plan','data_profile':config.data_profile,
        'alternative':item,'approval_id':permit.id,'max_buy_price':Decimal('1.10')})
    store.put('strategy_approvals',permit,scope)
    store.put('plans',plan,scope)
    for state in ('RESEARCH_COMPLETE','PENDING_VALIDATION','ELIGIBLE'):
        store.transition_plan(plan.id,state,scope=scope)
    for value in quotes.values(): evidence.ingest_market(value)
    evidence.ingest_market(MarketEvent(symbol='TEST_STOCK',source='fixture',source_version='2',event_type='bar',
        effective_event_time=NOW-timedelta(seconds=1),received_at=NOW,available_at=NOW,
        interval_start=NOW-timedelta(seconds=61),feed='sip',quality_class='genuine',price='100'))
    account = broker.account(policy,NOW).model_copy(update={'equity':Decimal('20000'),'options_level':3,
                                                          'options_buying_power':Decimal('10000')})
    intent = OrderIntent(account_id=base.account_id,environment='paper',purpose='open',plan_id=plan.id,
        plan_version=1,legs=opening_legs(item),limit_price='1.01',maximum_buy_price='1.10',
        expires_at=NOW+timedelta(seconds=10),reason='synthetic combined order')
    decision = RiskGate(store,config,scope,evidence,calendar).approve(intent,account,NOW,
        entry_ready=True,news_ready=True,event_calendar_ready=True)
    assert decision.approved, decision.reasons
    result = ExecutionGateway(store,config,scope,broker,evidence).submit(intent,decision,NOW)
    assert result['state'] == 'SHADOW'
    assert broker.writes == []
    store.close()
