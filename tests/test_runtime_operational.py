"""Focused operational regressions: no credentials, network or broker writes."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Event
from types import SimpleNamespace

import pytest

from src.runtime.calendar import SessionCalendar
from src.runtime.evidence import EvidenceClock, EvidenceService, SnapshotDataReader
from src.runtime.models import (AccountProfile, DataProfile, Instrument, MarketEvent,
    OrderIntent, OrderLeg, PositionGroup, PositionLeg, RiskDecision, RuntimeConfig, TradingSession)
from src.runtime.risk import ExecutionGateway, liquidation_mark
from src.runtime.service import SessionClient, SessionService
from src.runtime.store import Store, account_owned

NOW=datetime(2026,9,8,14,0,tzinfo=timezone.utc)


def quote(symbol='TEST',**values):
    return MarketEvent(symbol=symbol,source='fixture',source_version='1',event_type='quote',
        effective_event_time=NOW,received_at=NOW,available_at=NOW,feed='sip',quality_class='genuine',
        bid=Decimal('100'),ask=Decimal('101'),**values)


def test_host_ownership_independent_database_and_legacy_lookup(tmp_path):
    one=Store(tmp_path/'one.db');two=Store(tmp_path/'two.db')
    scope='paper:operational-fixture'
    try:
        one.acquire_lease(scope,'first')
        assert account_owned('operational-fixture','paper',tmp_path/'missing.db')
        with pytest.raises(RuntimeError,match='owns'):
            two.acquire_lease(scope,'second')
        one.release_lease(scope,'first')
        two.acquire_lease(scope,'second')
    finally:
        one.close();two.close()


def test_read_only_ui_does_not_create_database(tmp_path):
    path=tmp_path/'absent'/'service.db'
    assert SessionClient(path).status()['runtime_state']=='NOT_STARTED'
    assert not path.exists()


def test_unknown_newer_schema_is_not_mutated(tmp_path):
    import sqlite3
    path=tmp_path/'future.db'
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE schema_migrations(version INTEGER)')
    db.execute('INSERT INTO schema_migrations VALUES(99)');db.commit();db.close()
    with pytest.raises(RuntimeError,match='newer'):
        Store(path)
    db=sqlite3.connect(path)
    assert db.execute("SELECT name FROM sqlite_master WHERE name='records'").fetchone() is None
    db.close()


def test_immutable_record_and_transactional_outbox_rollback(tmp_path):
    store=Store(tmp_path/'state.db')
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.put('plans',{'id':'p','value':'original'},'paper:test')
            store.emit('created',{'id':'p'},'paper:test')
            raise RuntimeError('write failure')
    assert store.get('plans','p') is None
    assert store.watermark('paper:test')==0
    store.put('plans',{'id':'p','value':'original'})
    with pytest.raises(ValueError,match='immutable'):
        store.put('plans',{'id':'p','value':'revised'})
    store.close()


def test_completed_bar_cutoff_and_no_double_delay():
    clock=EvidenceClock(DataProfile.FREE_DELAYED,2)
    cutoff=clock.signal_as_of(NOW)
    event=MarketEvent(symbol='TEST',source='fixture',source_version='1',event_type='bar',
        interval_start=cutoff-timedelta(minutes=5),effective_event_time=cutoff+timedelta(seconds=1),
        received_at=NOW,available_at=NOW,feed='sip',quality_class='genuine',price='100')
    assert not clock.eligible(event,NOW)
    old=event.model_copy(update={'effective_event_time':cutoff})
    assert clock.eligible(old,NOW)


def test_gateway_rejects_forged_and_mutated_authority(tmp_path):
    store=Store(tmp_path/'gateway.db')
    config=RuntimeConfig(account_id='test',database_path=str(tmp_path/'gateway.db'))
    broker=SimpleNamespace(submit=lambda *args:pytest.fail('shadow broker call'))
    gateway=ExecutionGateway(store,config,'paper:test',broker,EvidenceService(store,config,'paper:test'))
    instrument=Instrument(symbol='TEST',liquidity_capacity='10')
    intent=OrderIntent(account_id='test',environment='paper',purpose='open',plan_id='plan',plan_version=1,
        legs=(OrderLeg(instrument=instrument,side='buy',position_intent='buy_to_open'),),
        limit_price='100',expires_at=NOW+timedelta(seconds=10),reason='fixture')
    decision=RiskDecision(id=intent.id,intent_id=intent.id,approved=True,policy_id=config.policy.id,
        account_profile_id='acct',data_policy_version=config.data_policy_version,event_watermark=0,
        approved_quantity='1',reasons=(),expires_at=NOW+timedelta(seconds=5))
    with pytest.raises(ValueError,match='durable'):
        gateway.submit(intent,decision,NOW)
    store.put('order_intents',intent,'paper:test');store.put('risk_decisions',decision,'paper:test')
    with pytest.raises(ValueError,match='differs'):
        gateway.submit(intent.model_copy(update={'limit_price':Decimal('10000')}),decision,NOW)
    store.close()


def test_net_zero_option_group_is_open_and_indicative_never_values():
    legs=tuple(PositionLeg(account_id='test',environment='paper',group_id='spread',symbol=symbol,
        asset_type='option',quantity=quantity,multiplier='100',reconciled_at=NOW)
        for symbol,quantity in [('TESTCALL1','1'),('TESTCALL2','-1')])
    group=PositionGroup(account_id='test',environment='paper',exit_policy_id='test',legs=legs,state='OPEN')
    assert not group.flat
    quotes={leg.symbol:quote(leg.symbol).model_copy(update={'feed':'indicative','quality_class':'indicative'}) for leg in legs}
    with pytest.raises(ValueError,match='GENUINE'):
        liquidation_mark(group,quotes,NOW)


def test_paid_research_worker_does_not_block_broker_cycle(tmp_path):
    start=NOW.replace(hour=13,minute=30);end=NOW.replace(hour=20,minute=0)
    sessions=[TradingSession(id='US:today',session_date='2026-09-08',calendar_version='fixture',open=start,close=end)]
    cfg=RuntimeConfig(database_path=str(tmp_path/'runtime.db'),account_id='thread-fixture')
    account=AccountProfile(account_id='thread-fixture',environment='paper',equity='5000',cash='5000',
        buying_power='5000',currency='USD',observed_at=NOW,mandate_id=cfg.policy.id)
    class Broker:
        calls=0
        def account(self,policy,now):
            self.calls+=1
            from uuid import uuid4
            return account.model_copy(update={'id':uuid4().hex,'observed_at':now})
        def positions(self,*args): return ()
        def orders(self): return []
        def activities(self): return []
    broker=Broker();service=SessionService(cfg,broker=broker,calendar=SessionCalendar(sessions),clock=lambda:NOW)
    entered=Event();release=Event()
    def blocking_role(view,session,now):
        assert view.broker is None and view.gateway is None
        entered.set();release.wait(5)
        return []
    try:
        service.run_once()
        service.roles['CloseGauss']=SimpleNamespace(run=blocking_role)
        service._run_role('CloseGauss',sessions[0],NOW,'independence')
        assert entered.wait(2)
        calls=broker.calls
        service.run_once()
        assert broker.calls>calls
    finally:
        release.set()
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_timeframe_resampling_requires_every_completed_minute():
    from src.runtime.evidence import completed_bars
    start=NOW.replace(minute=0,second=0)
    bars=[]
    for i in range(1,6):
        bars.append(MarketEvent(symbol='TEST',source='fixture',source_version=str(i),event_type='bar',
            interval_start=start+timedelta(minutes=i-1),effective_event_time=start+timedelta(minutes=i),
            received_at=start+timedelta(minutes=5),available_at=start+timedelta(minutes=5),
            feed='sip',quality_class='genuine',price=str(100+i),open_price=str(99+i),
            high_price=str(101+i),low_price=str(98+i),volume='10'))
    assert completed_bars(bars[:-1],'5Min')==[]
    assert completed_bars(bars[:2]+bars[3:],'5Min')==[]
    aggregated=completed_bars(bars,'5Min')
    assert len(aggregated)==1
    assert aggregated[0].open_price==100
    assert aggregated[0].price==105
    assert aggregated[0].volume==50


def test_daily_resampling_uses_actual_early_close():
    from src.runtime.evidence import completed_bars
    start=NOW.replace(hour=13,minute=30)
    end=start+timedelta(minutes=3)
    session=TradingSession(id='short',session_date='2026-09-08',calendar_version='test',open=start,close=end)
    bars=[]
    for i in range(1,4):
        bars.append(MarketEvent(symbol='TEST',source='fixture',source_version=str(i),event_type='bar',
            interval_start=start+timedelta(minutes=i-1),effective_event_time=start+timedelta(minutes=i),
            received_at=end,available_at=end,feed='sip',quality_class='genuine',
            price='100',open_price='100',high_price='101',low_price='99',volume='10'))
    assert completed_bars(bars,'1Day')==[]
    assert completed_bars(bars[:-1],'1Day',[session])==[]
    aggregate=completed_bars(bars,'1Day',[session])[0]
    assert aggregate.effective_event_time==end
    assert aggregate.interval_start==start


def test_runtime_verifies_command_signature_not_client_claim(tmp_path,monkeypatch):
    from uuid import uuid4
    from src.runtime.models import utc_now
    cfg=RuntimeConfig(database_path=str(tmp_path/'auth.db'),account_id='auth-fixture')
    service=SessionService(cfg,clock=lambda:NOW)
    service.entries_paused=False
    service.store.project('runtime',service.scope,{'scope':service.scope,'control_token_env':'GAUSS_CONTROL_TOKEN'},service.scope)
    forged={'id':uuid4().hex,'scope':service.scope,'action':'pause_entries','operator':'attacker',
            'confirmed':False,'payload':{},'created_at':utc_now().isoformat(),'state':'QUEUED','signature':'forged'}
    service.store.emit('operator_command',forged,service.scope)
    monkeypatch.setenv('GAUSS_CONTROL_TOKEN','server-secret')
    service._commands(NOW)
    assert not service.entries_paused
    assert service.store.projection('command_results',forged['id'])['state']=='REJECTED'
    client=SessionClient(cfg.database_path)
    command=client.command('pause_entries',token='server-secret')
    service._commands(NOW)
    assert service.entries_paused
    assert service.store.projection('command_results',command['id'])['state']=='COMPLETED'
    service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_request_budget_preserves_news_capacity_without_waiting():
    from src.runtime.broker import RequestBudget
    clock=[0.0]
    budget=RequestBudget(limit=5,news_reserve=2,clock=lambda:clock[0])
    for _ in range(3): budget.acquire()
    with pytest.raises(RuntimeError,match='BUDGET_EXHAUSTED'): budget.acquire()
    budget.acquire(news=True);budget.acquire(news=True)
    with pytest.raises(RuntimeError,match='BUDGET_EXHAUSTED'): budget.acquire(news=True)
    clock[0]=61
    budget.acquire()


def test_verified_event_calendar_versions_are_stable_and_expire(tmp_path):
    import json
    from src.runtime.events import FileEventCalendar
    path=tmp_path/'events.json'
    path.write_text(json.dumps({'published_at':NOW.isoformat(),
        'valid_until':(NOW+timedelta(hours=1)).isoformat(),'verified':True,
        'events':[{'id':'earnings','occurs_at':(NOW+timedelta(minutes=30)).isoformat(),'symbols':['TEST']}]}))
    provider=FileEventCalendar(path)
    assert provider.snapshot(NOW)==provider.snapshot(NOW)
    with pytest.raises(ValueError,match='STALE'):
        provider.snapshot(NOW+timedelta(hours=2))


@pytest.mark.parametrize('day,expected_utc,expected_london',[
    ('2026-03-09',13,13),  # US daylight time begins before UK daylight time.
    ('2026-03-30',13,14),
    ('2026-10-26',13,13),  # UK returns to standard time before the US does.
    ('2026-11-02',14,14),
])
def test_calendar_preserves_exchange_open_across_uk_us_dst_mismatch(day,expected_utc,expected_london):
    from zoneinfo import ZoneInfo
    opening=datetime.fromisoformat(day+'T09:30:00').replace(tzinfo=ZoneInfo('America/New_York'))
    closing=datetime.fromisoformat(day+'T16:00:00').replace(tzinfo=ZoneInfo('America/New_York'))
    session=TradingSession(id='US:'+day,session_date=day,calendar_version='explicit-fixture',open=opening,close=closing)
    assert session.open.hour==expected_utc
    assert session.open.minute==30
    assert session.open.astimezone(ZoneInfo('Europe/London')).hour==expected_london
    next_session=TradingSession(id='US:'+day+':next',session_date=(opening+timedelta(days=1)).date().isoformat(),
        calendar_version='explicit-fixture',open=opening+timedelta(days=1),close=closing+timedelta(days=1))
    calendar=SessionCalendar([session,next_session])
    policy=RuntimeConfig().policy
    assert not calendar.permits_entry(session.open+timedelta(minutes=4),policy)
    assert calendar.permits_entry(session.open+timedelta(minutes=5),policy)
    assert not calendar.is_open(session.close)


def _held_account_service(tmp_path,*,calendar_available=True,market=None):
    from uuid import uuid4
    cfg=RuntimeConfig(database_path=str(tmp_path/'held.db'),account_id='held-fixture',
        data_profile='SUBSCRIBED_REALTIME',symbols=('TEST',))
    account=AccountProfile(account_id=cfg.account_id,environment='paper',equity='10000',cash='9000',
        buying_power='9000',currency='USD',observed_at=NOW,mandate_id=cfg.policy.id)
    held=PositionLeg(account_id=cfg.account_id,environment='paper',group_id='holding-TEST',
        symbol='TEST',asset_type='stock',quantity='2',cost_basis='200',reconciled_at=NOW)
    class Broker:
        position_calls=0
        def account(self,policy,now):
            return account.model_copy(update={'id':uuid4().hex,'observed_at':now})
        def positions(self,account_id,now):
            self.position_calls+=1
            return (held.model_copy(update={'id':uuid4().hex,'reconciled_at':now}),)
        def orders(self): return []
        def activities(self): return []
        def instrument(self,symbol,capacity): return Instrument(symbol=symbol,liquidity_capacity=capacity)
        def submit(self,*args): pytest.fail('a shadow acceptance test attempted a broker write')
        def cancel(self,*args): pytest.fail('a shadow acceptance test attempted cancellation')
    session=TradingSession(id='US:held',session_date='2026-09-08',calendar_version='fixture',
        open=NOW.replace(hour=13,minute=30),close=NOW.replace(hour=20,minute=0))
    broker=Broker()
    service=SessionService(cfg,broker=broker,market=market,
        calendar=SessionCalendar([session] if calendar_available else []),clock=lambda:NOW)
    return service,broker


def test_missing_calendar_blocks_entries_but_keeps_current_holding_reconciliation(tmp_path):
    service,broker=_held_account_service(tmp_path,calendar_available=False)
    try:
        status=service.run_once()
        assert status['runtime_state']=='MANAGE_ONLY'
        assert not status['entries_ready']
        assert 'SESSION_CALENDAR_UNAVAILABLE' in status['readiness']
        assert status['health']['reconciliation']=='HEALTHY'
        assert service.operational_profile.positions[0].quantity==2
        calls=broker.position_calls
        service.run_once()
        assert broker.position_calls>calls
        groups=service.store.projected('position_groups',service.scope)
        assert any(group['legs'][0]['quantity']=='2' for group in groups)
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_subscribed_entitlement_loss_retains_holdings_without_profile_fallback(tmp_path):
    from src.runtime.models import DataCapabilitySnapshot
    class Market:
        allowed=True
        config=None
        gaps=[]
        health={'price_data':'HEALTHY','news':'HEALTHY'}
        def probe(self,account_id,now,symbols):
            return [DataCapabilitySnapshot(account_id=account_id,environment='paper',endpoint=endpoint,
                feed=feed,outcome=outcome,quality_class='genuine',expires_at=now+timedelta(minutes=5))
                for endpoint,feed,outcome in [
                    ('stock_latest','sip','AVAILABLE' if self.allowed else 'UNAVAILABLE'),
                    ('option_latest','opra','UNAVAILABLE')]]
        def collect(self,symbols,now): return [quote()],[]
        def close(self): pass
    market=Market()
    service,broker=_held_account_service(tmp_path,market=market)
    try:
        service._start(NOW)
        service._collect(NOW)
        assert service.price_ready
        capabilities=service.store.list('data_capabilities',service.scope)
        assert next(c for c in capabilities if c['feed']=='opra')['outcome']=='UNAVAILABLE'
        market.allowed=False
        service._probe(NOW+timedelta(seconds=1))
        service._collect(NOW+timedelta(seconds=1))
        assert not service.price_ready
        assert not service.entries_ready
        assert service.config.data_profile==DataProfile.SUBSCRIBED_REALTIME
        calls=broker.position_calls
        service._reconcile(NOW+timedelta(seconds=1))
        assert broker.position_calls>calls
        assert service.operational_profile.positions[0].quantity==2
        assert service.state=='MANAGE_ONLY'
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_operator_profile_switch_revalidates_plans_and_retains_real_holdings(tmp_path,monkeypatch):
    from src.runtime.models import Alternative,Rule,SessionPlan
    service,_broker=_held_account_service(tmp_path)
    try:
        service._start(NOW)
        instrument=Instrument(symbol='CANDIDATE',liquidity_capacity='100')
        alternative=Alternative(strategy_id='momentum',strategy_version='1',instrument=instrument,
            entry_price='100',stop_price='95',expected_net_value='1',evidence_ids=('evidence',))
        plan=SessionPlan(target_session_id='US:held',snapshot_id='snapshot',candidate_id='candidate',
            account_id=service.config.account_id,environment='paper',account_profile_id=service.account_profile.id,
            suitability_report_id='suitability',data_profile='SUBSCRIBED_REALTIME',
            data_policy_version=service.config.data_policy_version,signal_as_of=NOW,
            strategy_id='momentum',strategy_version='1',approval_id='approval',alternative=alternative,
            entry_rules=(Rule(rule_type='no_blocking_event'),),max_buy_price='101',valid_from=NOW,
            entry_expires_at=NOW+timedelta(hours=1),close_at=NOW+timedelta(hours=2),
            risk_policy_id=service.config.policy.id,exit_policy_id='exit-v1')
        service.store.put('plans',plan,service.scope)
        for state in ('RESEARCH_COMPLETE','PENDING_VALIDATION','ELIGIBLE'):
            service.store.transition_plan(plan.id,state,scope=service.scope)
        before=service.store.projected('position_groups',service.scope)
        previous_policy=service.config.data_policy_version
        monkeypatch.setenv('GAUSS_CONTROL_TOKEN','profile-secret')
        command=SessionClient(service.config.database_path).command('change_profile',token='profile-secret',
            payload={'data_profile':'FREE_DELAYED','reason':'explicit delayed experiment'})
        service._commands(NOW)
        assert service.store.projection('command_results',command['id'])['state']=='COMPLETED'
        assert service.config.data_profile==DataProfile.FREE_DELAYED
        assert service.config.data_policy_version!=previous_policy
        assert service.store.projection('plan_state',plan.id)['state']=='REVIEW_REQUIRED'
        assert service.entries_paused and not service.entries_ready
        assert service.store.projected('position_groups',service.scope)==before
        assert service.operational_profile.positions[0].quantity==2
        assert service.store.get('plans',plan.id)['data_profile']=='SUBSCRIBED_REALTIME'
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_gap_recovery_requires_every_missing_minute_and_a_fresh_genuine_quote(tmp_path):
    from src.runtime.models import DataCapabilitySnapshot
    gap={'symbol':'TEST','reason':'STREAM_DISCONNECTED','at':(NOW-timedelta(minutes=3,seconds=-30)).isoformat()}
    market=SimpleNamespace(gaps=[gap],health={},close=lambda:None)
    service,_broker=_held_account_service(tmp_path,market=market)
    service.calendar_ready=True
    try:
        service.store.put('data_capabilities',DataCapabilitySnapshot(account_id=service.config.account_id,
            environment='paper',endpoint='stock_latest',feed='sip',outcome='AVAILABLE',quality_class='genuine',
            expires_at=NOW+timedelta(minutes=5)),service.scope)
        def bar(stamp):
            return MarketEvent(symbol='TEST',source='fixture',source_version=stamp.isoformat(),event_type='bar',
                interval_start=stamp-timedelta(minutes=1),effective_event_time=stamp,received_at=NOW,available_at=NOW,
                feed='sip',quality_class='genuine',price='100',open_price='100',high_price='101',low_price='99',volume='10')
        service.evidence.ingest_market(bar(NOW-timedelta(minutes=2)))
        service.evidence.ingest_market(bar(NOW))
        service.evidence.ingest_market(quote().model_copy(update={'effective_event_time':NOW-timedelta(seconds=6)}))
        service._recover_market_gaps(NOW)
        assert market.gaps
        service.evidence.ingest_market(quote())
        service._recover_market_gaps(NOW)
        assert market.gaps  # Latest timestamp cannot conceal the absent middle minute.
        service.evidence.ingest_market(bar(NOW-timedelta(minutes=1)))
        service._recover_market_gaps(NOW)
        assert not market.gaps
        assert service.price_ready
        assert service.store.list('incidents',service.scope)[0]['reason']=='FEED_GAP_RECOVERED'
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_large_universe_rotates_batched_history_and_persists_quotes_before_quota_failure():
    from queue import Queue
    from src.runtime.broker import AlpacaMarket
    market=AlpacaMarket.__new__(AlpacaMarket)
    market.config=RuntimeConfig(data_profile='SUBSCRIBED_REALTIME')
    market.events=Queue();market.gaps=[];market.cursors={};market.health={}
    market.start_stream=lambda symbols:None
    persisted=[];market.event_sink=persisted.append
    universe=tuple(f'TEST_{i:03d}' for i in range(200))
    class Stock:
        remaining=3
        requests=[]
        def get_stock_latest_quote(self,request):
            self.requests.append('quotes')
            return {'HELD':SimpleNamespace(symbol='HELD',timestamp=NOW,bid_price=100,ask_price=101,bid_size=10,ask_size=10)}
        def get_stock_bars(self,request):
            self.requests.append(tuple(request.symbol_or_symbols))
            assert len(request.symbol_or_symbols)<=16
            assert request.limit==10000
            if self.remaining==0: raise RuntimeError('MARKET_REQUEST_BUDGET_EXHAUSTED')
            self.remaining-=1
            return SimpleNamespace(data={symbol:[SimpleNamespace(symbol=symbol,timestamp=NOW-timedelta(minutes=1),
                open=100,high=101,low=99,close=100,volume=100)] for symbol in request.symbol_or_symbols})
    market.stock=Stock()
    for _ in range(5):
        market.stock.remaining=3
        start=len(persisted)
        with pytest.raises(RuntimeError,match='BUDGET_EXHAUSTED'):
            market.collect(universe,NOW)
        assert persisted[start].symbol=='HELD' and persisted[start].event_type=='quote'
    assert set(market.cursors)==set(universe)
    assert any(event.symbol=='TEST_199' and event.event_type=='bar' for event in persisted)


def test_held_stock_and_option_quotes_survive_discovery_and_history_quota_exhaustion(tmp_path):
    from src.runtime.models import AccountOperationalState
    calls=[]
    held_stock=PositionLeg(account_id='priority-fixture',environment='paper',group_id='stock',symbol='HELD',
        asset_type='stock',quantity='1',cost_basis='100',reconciled_at=NOW)
    held_option=PositionLeg(account_id='priority-fixture',environment='paper',group_id='option',symbol='HELDCALL',
        asset_type='option',quantity='1',multiplier='100',cost_basis='200',reconciled_at=NOW)
    class Market:
        gaps=[];health={'price_data':'HEALTHY','news':'HEALTHY'}
        def collect_options(self,symbols,now):
            calls.append('held-options')
            return [quote('HELDCALL').model_copy(update={'feed':'opra'})]
        def collect_quotes(self,symbols,now):
            calls.append('held-stocks');return [quote('HELD')]
        def collect(self,symbols,now):
            calls.append('history')
            raise RuntimeError('MARKET_REQUEST_BUDGET_EXHAUSTED')
        def close(self):pass
    class Broker:
        def instrument(self,symbol,capacity):
            calls.append('discovery')
            raise RuntimeError('MARKET_REQUEST_BUDGET_EXHAUSTED')
    cfg=RuntimeConfig(account_id='priority-fixture',database_path=str(tmp_path/'priority.db'),
        data_profile='SUBSCRIBED_REALTIME',symbols=('DISCOVERY',))
    service=SessionService(cfg,broker=Broker(),market=Market(),clock=lambda:NOW)
    service.operational_profile=AccountOperationalState(account_id=cfg.account_id,environment='paper',
        positions=(held_stock,held_option),observed_at=NOW)
    try:
        service._collect(NOW)
        assert calls==['held-options','held-stocks','discovery','history']
        assert service.evidence.current_quote('HELDCALL',NOW,'opra') is not None
        assert service.evidence.current_quote('HELD',NOW,'sip') is not None
        assert not service.price_ready
        reasons={row['reason'] for row in service.store.list('incidents',service.scope)}
        assert {'DISCOVERY_METADATA_DEFERRED','COLLECTION_FAILED'}<=reasons
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_confirmed_holding_instrument_lookup_bypasses_only_discovery_quota():
    from src.runtime.broker import AlpacaBroker,RequestBudget
    broker=AlpacaBroker.__new__(AlpacaBroker)
    broker.metadata_budget=RequestBudget(limit=1,news_reserve=0)
    broker.metadata_budget.acquire()
    broker.client=SimpleNamespace(get_asset=lambda symbol:SimpleNamespace(asset_class='us_equity',tradable=True))
    with pytest.raises(RuntimeError,match='BUDGET_EXHAUSTED'):
        broker.instrument('HELD',Decimal('2'))
    verified=broker.operational_instrument('HELD',Decimal('2'))
    assert verified.symbol=='HELD' and verified.asset_type=='stock' and verified.tradable
    assert verified.quantity_increment==1 and verified.multiplier==1


def test_news_disconnect_blocks_dependent_plan_then_recovers_without_losing_holdings(tmp_path):
    from src.runtime.models import Alternative,NewsEvent,Rule,SessionPlan,StrategyApproval
    service,broker=_held_account_service(tmp_path)
    service.config=service.config.model_copy(update={'strategy_allowlist':('momentum',),'signal_timeframe':'1Min'})
    service._wire()
    class NewsMarket:
        connected=False
        health={'news':'UNKNOWN'}
        def collect_news(self,symbols,now):
            if not self.connected:
                raise ConnectionError('synthetic news stream disconnected')
            self.health['news']='HEALTHY'
            return [NewsEvent(provider='verified-fixture',article_id='reconnected-news',source_version='1',
                published_at=NOW,updated_at=NOW,received_at=NOW,available_at=NOW,symbols=('CANDIDATE',),
                headline='Verified routine company disclosure',content_hash='fixture-metadata-hash',
                verified=True,blocking=False)]
        def close(self):pass
    market=NewsMarket()
    try:
        service._start(NOW)
        service.market=market
        service.price_ready=True
        evidence_id=service.evidence.ingest_market(quote('CANDIDATE'))
        approval=StrategyApproval(strategy_id='momentum',strategy_version='1',account_id=service.config.account_id,
            environment='paper',approved_by='fixture-operator',profiles=(DataProfile.SUBSCRIBED_REALTIME,),
            requires_current_quote=False,maximum_signal_latency_seconds=60,requires_news=True,
            requires_event_calendar=False,validation_reference='synthetic operational fixture',
            validated_net_expectancy='1',expiry=NOW+timedelta(hours=2))
        service.store.put('strategy_approvals',approval,service.scope)
        alternative=Alternative(strategy_id='momentum',strategy_version='1',
            instrument=Instrument(symbol='CANDIDATE',liquidity_capacity='100'),entry_price='100',stop_price='95',
            expected_net_value='1',evidence_ids=(evidence_id,))
        plan=SessionPlan(target_session_id='US:held',snapshot_id='frozen-snapshot',candidate_id='candidate',
            account_id=service.config.account_id,environment='paper',account_profile_id=service.account_profile.id,
            suitability_report_id='preliminary-suitability',data_profile='SUBSCRIBED_REALTIME',
            data_policy_version=service.config.data_policy_version,signal_as_of=NOW,strategy_id='momentum',
            strategy_version='1',approval_id=approval.id,alternative=alternative,
            entry_rules=(Rule(rule_type='no_blocking_event'),),max_buy_price='101',valid_from=NOW,
            entry_expires_at=NOW+timedelta(hours=1),close_at=NOW+timedelta(hours=2),
            risk_policy_id=service.config.policy.id,exit_policy_id='retained-exit-v1')
        service.store.put('plans',plan,service.scope)
        for state in ('RESEARCH_COMPLETE','PENDING_VALIDATION'):
            service.store.transition_plan(plan.id,state,scope=service.scope)
        session=service.calendar.get('US:held')
        before=service.operational_profile.positions[0]
        service._collect_news(NOW)
        service.roles['PreGauss'].run(service,session,NOW)
        service._publish(NOW)
        assert not service.news_ready
        assert service.status()['health']['news']=='FAILED'
        assert 'NEWS_COVERAGE_UNAVAILABLE' in service.status()['readiness']
        validation=service.store.list('validations',service.scope)[0]
        assert validation['outcome']=='DEFERRED'
        assert 'NEWS_COVERAGE_UNAVAILABLE' in validation['reasons']
        service.roles['LiveGauss'].run(service,session,NOW)
        assert service.store.projected('orders',service.scope)==[]
        calls=broker.position_calls
        service._reconcile(NOW)
        assert broker.position_calls>calls
        assert service.operational_profile.positions[0].quantity==before.quantity
        market.connected=True
        service._collect_news(NOW)
        service.roles['PreGauss'].run(service,session,NOW)
        service._publish(NOW)
        assert service.news_ready and service.status()['health']['news']=='HEALTHY'
        assert 'NEWS_COVERAGE_UNAVAILABLE' not in service.status()['readiness']
        assert service.store.projection('plan_state',plan.id)['state']=='ELIGIBLE'
        assert service.evidence.news(NOW)[0].verified
        assert service.operational_profile.positions[0].quantity==before.quantity
        assert any(row['reason']=='NEWS_COLLECTION_FAILED' for row in service.store.list('incidents',service.scope))
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_stale_database_lease_reports_expiry_and_releases_host_lock(tmp_path):
    from src.runtime.store import LeaseConflict

    path = tmp_path / 'stale.db'
    scope = 'paper:stale-startup-fixture'
    original = Store(path)
    original.acquire_lease(scope, 'dead-owner', now=NOW, seconds=120)
    original.close()  # Simulate process death: host lock released, DB lease remains.
    candidate = Store(path)
    other = Store(tmp_path / 'other.db')
    try:
        with pytest.raises(LeaseConflict, match='lease expires at'):
            candidate.acquire_lease(scope, 'new-owner', now=NOW)
        assert scope not in candidate._host_locks
        assert candidate.db.execute('SELECT owner FROM leases').fetchone()['owner'] == 'dead-owner'
        other.acquire_lease(scope, 'host-probe', now=NOW)
        other.release_lease(scope, 'host-probe')
        candidate.acquire_lease(scope, 'new-owner', now=NOW + timedelta(seconds=121))
        assert candidate.db.execute('SELECT owner FROM leases').fetchone()['owner'] == 'new-owner'
    finally:
        candidate.close()
        other.close()


def test_startup_lease_conflict_exits_without_touching_account_state(tmp_path):
    from src.runtime.runner import run_service
    from src.runtime.models import AccountOperationalState

    config = RuntimeConfig(database_path=str(tmp_path / 'startup.db'),
                           account_id='startup-conflict-fixture')
    owner = Store(config.database_path)
    scope = 'paper:startup-conflict-fixture'
    owner.acquire_lease(scope, 'running-owner')
    owner.project('runtime', scope, {'supervision': 'RUNNING'}, scope)
    before = owner.projection('runtime', scope)
    closed = []
    broker = SimpleNamespace(
        account=lambda *_: AccountOperationalState(account_id=config.account_id,
                                                   environment='paper', observed_at=NOW, positions=()),
        close=lambda: closed.append(True),
    )
    candidate = SessionService(config, broker=broker)
    records = []
    try:
        assert run_service(candidate, emit=records.append,
                           wait=lambda _: pytest.fail('Blocked startup must exit')) == 1
        assert [r['event'] for r in records] == ['startup_blocked'], records
        assert closed == [True]
        assert candidate._closed and not candidate._owns_lease
        assert owner.projection('runtime', scope) == before
        assert owner.db.execute('SELECT owner FROM leases').fetchone()['owner'] == 'running-owner'
        assert owner.projected('incidents', scope) == []
    finally:
        if not candidate._closed:
            candidate.request_shutdown(acknowledge_unmanaged_exposure=True)
        owner.close()


@pytest.mark.parametrize('allowed', [True, False])
def test_empty_configured_universe_probes_holdings_before_marking_data_ready(tmp_path, allowed):
    from src.runtime.models import DataCapabilitySnapshot

    class Market:
        gaps = []
        health = {'price_data': 'HEALTHY', 'news': 'HEALTHY'}
        probed = None
        collected = None
        def probe(self, account_id, now, symbols):
            self.probed = symbols
            return [DataCapabilitySnapshot(
                account_id=account_id, environment='paper', endpoint='stock_historical',
                feed='sip', outcome='AVAILABLE' if allowed else 'UNAVAILABLE',
                quality_class='genuine', expires_at=now + timedelta(minutes=5),
            )] if symbols else []
        def collect(self, symbols, now):
            self.collected = symbols
            return [], []
        def close(self): pass

    market = Market()
    service, _ = _held_account_service(tmp_path, market=market)
    service.config = service.config.model_copy(update={'symbols': (), 'data_profile': DataProfile.FREE_DELAYED})
    try:
        service._reconcile(NOW)
        service._probe(NOW)
        service._collect(NOW)
        assert market.probed == market.collected == ('TEST',)
        assert service.price_ready is allowed
        assert len(service.store.list('data_capabilities', service.scope)) == 1
        assert not service.approvals(NOW)
        assert not service.event_calendar_ready
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_api_event_calendar_combines_sources_without_inventing_times():
    from src.runtime.events import APIEventCalendar
    calls = []
    earnings = SimpleNamespace(get_earnings_calendar=lambda **kw: (
        calls.append(kw) or {'earningsCalendar': [{'symbol': 'TEST', 'date': '2026-09-08', 'hour': 'amc'}]}))
    macro = SimpleNamespace(get_release_dates=lambda *args: [
        {'release_id': 10, 'date': '2026-09-08', 'release_name': 'Consumer Price Index'},
        {'release_id': 18, 'date': '2026-09-08', 'release_name': 'H.15 Selected Interest Rates'},
        {'release_id': 101, 'date': '2026-09-08', 'release_name': 'FOMC Press Release'}])
    provider = APIEventCalendar(earnings, macro)
    snapshot = provider.snapshot(NOW)
    assert snapshot.verified and len(snapshot.events) == 4
    assert snapshot.coverage == {'earnings': 'AVAILABLE', 'macro': 'AVAILABLE_DATE_ONLY'}
    assert all(e.time_precision == 'date' and e.block_window_seconds == 43200 for e in snapshot.events)
    assert sum(e.blocks_entries for e in snapshot.events) == 2
    assert provider.snapshot(NOW + timedelta(minutes=1)) is snapshot
    assert len(calls) == 1
    assert provider.snapshot(NOW + timedelta(minutes=16)) is not snapshot
    assert len(calls) == 2


def test_calendar_partial_coverage_does_not_authorize_entries(tmp_path):
    from src.runtime.events import APIEventCalendar
    earnings = SimpleNamespace(get_earnings_calendar=lambda **kw: {'error': 'access denied'})
    macro = SimpleNamespace(get_release_dates=lambda *args: [
        {'release_id': 10, 'date': '2026-09-08', 'release_name': 'Consumer Price Index'}])
    provider = APIEventCalendar(earnings, macro)
    cfg = RuntimeConfig(database_path=str(tmp_path/'partial-calendar.db'))
    service = SessionService(cfg, event_calendar=provider, clock=lambda: NOW)
    try:
        service._refresh_event_calendar(NOW)
        assert not service.event_calendar_ready
        assert service.health_state['event_calendar'] == 'PARTIAL'
        assert len(service.store.list('scheduled_events', service.scope)) == 1
    finally:
        # Injected fakes have no network sessions.
        service.event_calendar = None
        service.request_shutdown()


def test_fred_calendar_paginates_and_requests_future_dates(monkeypatch):
    import requests
    from src.data.fred_provider import FREDProvider
    calls = []
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, *, params, timeout):
            calls.append(params)
            count = 1000 if params['offset'] == 0 else 1
            return SimpleNamespace(status_code=200, json=lambda: {
                'count': 1001, 'release_dates': [{'release_id': 10}]*count})
    monkeypatch.setattr(requests, 'Session', Session)
    provider = FREDProvider('synthetic')
    assert len(provider.get_release_dates('2026-09-08', '2026-09-15')) == 1001
    assert [r['offset'] for r in calls] == [0, 1000]
    assert all(r['include_release_dates_with_no_data'] == 'true' for r in calls)
