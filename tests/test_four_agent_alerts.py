"""Offline overload and durable incident channels; no external notifier is configured."""
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from queue import Queue
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.runtime.broker import AlpacaMarket
from src.runtime.models import AccountProfile, Instrument, PositionGroup, PositionLeg, RuntimeConfig
from src.runtime.service import SessionClient, SessionService

WALL = datetime(2026, 9, 9, 15, tzinfo=timezone.utc)


class FakeMarket:
    def __init__(self, gaps=(), exhausted=False):
        self.gaps = list(gaps)
        self.exhausted = exhausted
        self.health = {'news': 'HEALTHY', 'price_data': 'HEALTHY'}
        self.requested_symbols = ()

    def collect(self, symbols, now):
        self.requested_symbols = symbols
        if self.exhausted:
            raise RuntimeError('SUBSCRIPTION_CAPACITY_EXHAUSTED')
        return [], []


@pytest.fixture
def service_factory(tmp_path):
    services = []

    def make(market):
        account_id = 'alerts-fixture-' + uuid4().hex
        config = RuntimeConfig(database_path=str(tmp_path/(account_id+'.sqlite3')),
                               account_id=account_id, symbols=('SCAN_FIRST', 'SCAN_SECOND'),
                               scan_universe_limit=1)
        broker = SimpleNamespace(instrument=lambda symbol, capacity: Instrument(
            symbol=symbol, liquidity_capacity=capacity))
        service = SessionService(config, broker=broker, market=market, clock=lambda: WALL)
        services.append(service)
        leg = PositionLeg(account_id=account_id, environment='paper', group_id='held-group',
                          symbol='HELD_STOCK', asset_type='stock', quantity=Decimal('3'),
                          reconciled_at=WALL)
        service.account_profile = AccountProfile(account_id=account_id, environment='paper',
            hypothetical=False, equity='10000', cash='9000', buying_power='9000', currency='USD',
            observed_at=WALL, positions=(leg,), mandate_id=config.policy.id)
        service.operational_profile = service.account_profile
        group = PositionGroup(id='held-group', account_id=account_id, environment='paper',
                              exit_policy_id='manual-adoption-required', legs=(leg,),
                              state='RECONCILIATION_REQUIRED')
        service.store.project('position_groups', group.id, group.model_dump(mode='json'), service.scope)
        service.store.project('runtime', service.scope, {'scope': service.scope,
            'account_id': account_id, 'environment': 'paper'}, service.scope)
        service.store.put('data_capabilities', {'id':'fixture-entitlement',
            'endpoint':'stock_historical', 'outcome':'AVAILABLE',
            'expires_at':(WALL+timedelta(hours=1)).isoformat()}, service.scope)
        service.entries_paused = False
        service.state = 'READY'
        service.price_ready = True
        service.health_state['reconciliation'] = 'HEALTHY'
        return service

    yield make
    # These services never start ownership, collectors, brokers or management threads.
    for service in services:
        service._research_pool.shutdown(wait=True, cancel_futures=True)
        service._collection_pool.shutdown(wait=True, cancel_futures=True)
        service.store.close()


def test_stream_queue_overflow_is_visible_without_dropping_holdings(service_factory):
    class Stream:
        handler = None
        def subscribe_quotes(self, handler, *symbols): self.handler = handler
        def subscribe_bars(self, handler, *symbols): pass

    stream = Stream()
    adapter = AlpacaMarket.__new__(AlpacaMarket)
    adapter.stream = stream
    adapter.subscribed = {'ALREADY_SUBSCRIBED'}  # No background stream thread is started.
    adapter.events = Queue(maxsize=1)
    adapter.events.put_nowait(object())
    adapter.gaps = []
    adapter.quote = lambda *args: object()
    adapter.start_stream(('HELD_STOCK',))
    asyncio.run(stream.handler(SimpleNamespace(symbol='HELD_STOCK', bid_price=100)))
    assert adapter.events.qsize() == 1
    assert adapter.gaps[0]['reason'] == 'QUEUE_OVERFLOW'

    market = FakeMarket(adapter.gaps)
    service = service_factory(market)
    before = service.store.projection('position_groups', 'held-group')
    service._collect(WALL)
    service._collect(WALL+timedelta(seconds=1))
    assert not service.entries_ready
    assert market.requested_symbols == ('HELD_STOCK', 'SCAN_FIRST')
    assert service.store.projection('position_groups', 'held-group') == before
    incidents = service.report()['incidents']
    matching = [row for row in incidents if row['reason'] == 'QUEUE_OVERFLOW']
    assert len(matching) == 1, 'Repeated collection must not duplicate the same gap incident'
    assert matching[0]['symbol'] == 'HELD_STOCK'


def test_subscription_exhaustion_preserves_holdings_and_blocks_entries(service_factory):
    market = FakeMarket(exhausted=True)
    service = service_factory(market)
    before = service.store.projection('position_groups', 'held-group')
    service._collect(WALL)
    assert market.requested_symbols[0] == 'HELD_STOCK'
    assert not service.entries_ready
    assert service.health_state['price_data'] == 'FAILED'
    assert service.store.projection('position_groups', 'held-group') == before
    assert any(row['reason'] == 'COLLECTION_FAILED' for row in service.report()['incidents'])


def test_console_alert_failure_preserves_durable_client_visibility(service_factory, monkeypatch):
    import src.runtime.service as service_module

    class BrokenConsole:
        def write(self, text): raise OSError('fixture console transport failed')
        def flush(self): raise OSError('fixture console transport failed')

    service = service_factory(FakeMarket())
    with monkeypatch.context() as patch:
        patch.setattr(service_module.sys, 'stderr', BrokenConsole())
        service._incident('FIXTURE_ALERT_TRANSPORT_FAILURE', WALL, symbol='HELD_STOCK')
    client = SessionClient(service.config.database_path, account_id=service.config.account_id,
                           environment='paper')
    incidents = client.report()['incidents']
    assert any(row['reason'] == 'FIXTURE_ALERT_TRANSPORT_FAILURE' for row in incidents)
    assert service.store.projection('position_groups', 'held-group')['legs'][0]['quantity'] == '3'


def test_incident_is_committed_before_console_delivery(service_factory, monkeypatch):
    import src.runtime.service as service_module

    service = service_factory(FakeMarket())
    observations = []

    class ObservedConsole:
        def write(self, text):
            observations.append(bool(service.store.list('incidents', service.scope)))
        def flush(self): pass

    with monkeypatch.context() as patch:
        patch.setattr(service_module.sys, 'stderr', ObservedConsole())
        service._incident('FIXTURE_DELIVERY_ORDER', WALL)
    assert observations and all(observations)
