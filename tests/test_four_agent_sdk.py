"""Offline adapter contract tests against the installed Alpaca request models."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from alpaca.trading.enums import AccountStatus, AssetClass

from src.runtime.broker import AlpacaBroker, _standard_contract
from src.runtime.models import Instrument, OrderIntent, OrderLeg, RiskPolicy


NOW = datetime(2026, 9, 8, 15, tzinfo=timezone.utc)


def broker():
    account = SimpleNamespace(id='verified-account', equity='10000', cash='3000',
        buying_power='20000', options_buying_power='500', currency='USD', trading_blocked=False,
        account_blocked=False, trade_suspended_by_user=False, status=AccountStatus.ACTIVE,
        shorting_enabled=True, options_trading_level=3, multiplier='4')
    holdings = [SimpleNamespace(asset_class=AssetClass.US_EQUITY, symbol='AAPL', qty='2',
        cost_basis='100', market_value='200', asset_id='stock-id'),
        SimpleNamespace(asset_class=AssetClass.CRYPTO, symbol='BTCUSD', qty='.1',
        cost_basis='800', market_value='1000', asset_id='crypto-id')]
    config = dict(suspend_trade=False, fractional_trading=True, no_shorting=True,
                  closing_transactions_only=False)
    def get(path, data=None):
        assert path == '/account/configurations'
        return config
    value = AlpacaBroker.__new__(AlpacaBroker)
    value.environment='paper'
    value.client=SimpleNamespace(get_account=lambda: account, get_all_positions=lambda: holdings,
        get=get)
    return value, account


def test_order_listing_uses_the_installed_sdk_request_and_response():
    from alpaca.trading.models import Order
    from alpaca.trading.requests import GetOrdersRequest

    adapter, _ = broker()
    requests = []
    order = Order.model_validate({
        'id': 'fd24fb90-5f03-4bb6-9b2c-df779d18aa31',
        'client_order_id': 'synthetic-order', 'created_at': NOW,
        'updated_at': NOW, 'submitted_at': NOW,
        'asset_id': '0d3b4e47-3d7f-4735-b390-6a6175af74fc',
        'symbol': 'AAPL', 'asset_class': 'us_equity', 'qty': '1',
        'filled_qty': '0', 'order_class': 'simple', 'order_type': 'limit',
        'type': 'limit', 'side': 'buy', 'time_in_force': 'day',
        'limit_price': '100', 'status': 'new', 'extended_hours': False,
    })

    def get_orders(request):
        requests.append(request)
        return [order]

    adapter.client.get_orders = get_orders
    result = adapter.orders()
    assert isinstance(requests[0], GetOrdersRequest)
    assert requests[0].to_request_fields() == {
        'status': 'all', 'limit': 500, 'direction': 'desc', 'nested': True}
    assert result[0]['id'] == str(order.id)
    assert result[0]['status'] == 'new'
    assert result[0]['client_order_id'] == 'synthetic-order'


@pytest.mark.parametrize('import_failure', [False, True])
def test_service_reconciliation_exercises_sdk_adapter_and_reports_import_detail(
    tmp_path, import_failure, capsys
):
    from alpaca.trading.requests import GetOrdersRequest
    from src.runtime.calendar import SessionCalendar
    from src.runtime.models import RuntimeConfig, TradingSession
    from src.runtime.service import SessionService

    adapter, _ = broker()
    adapter._stream = None
    reads = []

    def get_orders(request):
        assert isinstance(request, GetOrdersRequest)
        reads.append('orders')
        if import_failure:
            raise ImportError("cannot import name 'MissingEnum' from 'synthetic.module'")
        return []

    configuration = adapter.client.get('/account/configurations')

    def get(path, data=None):
        if path == '/account/configurations':
            return configuration
        assert path == '/account/activities'
        assert data['page_size'] == 100
        reads.append('activities')
        return []

    adapter.client.get_orders = get_orders
    adapter.client.get = get
    config = RuntimeConfig(database_path=str(tmp_path/'session.sqlite3'),
                           account_id='verified-account')
    calendar = SessionCalendar([TradingSession(id='fixture-session', session_date='2026-09-08',
        calendar_version='fixture', open=NOW-timedelta(hours=1), close=NOW+timedelta(hours=4))])
    service = SessionService(config, broker=adapter, calendar=calendar, clock=lambda: NOW)
    try:
        service._reconcile(NOW)
        if import_failure:
            assert service.health_state['reconciliation'] == 'FAILED'
            assert service.state == 'MANAGE_ONLY'
            assert not service.entries_ready
            incident = service.store.list('incidents', service.scope)[0]
            assert incident['error'] == 'ImportError'
            assert 'MissingEnum' in incident['detail']
            assert 'synthetic.module' in capsys.readouterr().err
        else:
            assert reads == ['orders', 'activities']
            assert service.health_state['reconciliation'] == 'HEALTHY'
            assert service.last_reconciliation == NOW.isoformat()
            assert service.account_profile.equity == Decimal('10000')
            assert service.account_profile.positions[0].symbol == 'AAPL'
            assert all(row['reason'] != 'RECONCILIATION_FAILED'
                       for row in service.store.list('incidents', service.scope))
    finally:
        service.request_shutdown(acknowledge_unmanaged_exposure=True)


def test_actual_permissions_option_buying_power_and_out_of_scope_exposure():
    adapter, raw = broker()
    account = adapter.account(RiskPolicy(), NOW)
    assert account.options_buying_power == 500
    assert account.fractional_allowed and not account.shorting_allowed
    assert account.exposure == 1200
    assert account.out_of_scope_exposure == 1000
    assert account.out_of_scope_symbols == ('BTCUSD',)
    positions = adapter.positions(account.account_id, NOW)
    assert len(positions) == 1 and positions[0].asset_type == 'stock'
    raw.trade_suspended_by_user = True
    assert adapter.account(RiskPolicy(), NOW).trading_blocked


def test_current_configuration_without_retired_sdk_fields_and_close_only_restriction():
    adapter, _ = broker()
    configuration = adapter.client.get('/account/configurations')
    assert 'dtbp_check' not in configuration and 'pdt_check' not in configuration
    assert not adapter.account(RiskPolicy(), NOW).trading_blocked
    configuration['closing_transactions_only'] = True
    assert adapter.account(RiskPolicy(), NOW).trading_blocked


@pytest.mark.parametrize('invalid', [None, 'false', 0])
def test_missing_or_malformed_account_permissions_cannot_authorize_entries(invalid):
    adapter, _ = broker()
    configuration = adapter.client.get('/account/configurations')
    configuration['closing_transactions_only'] = invalid
    with pytest.raises(ValueError):
        adapter.account(RiskPolicy(), NOW)
    del configuration['closing_transactions_only']
    with pytest.raises(ValueError):
        adapter.account(RiskPolicy(), NOW)


@pytest.mark.parametrize('equity', [None, 'NaN', 'Infinity', '0'])
def test_invalid_funds_do_not_fabricate_account_but_preserve_verified_identity(equity):
    adapter, raw = broker()
    raw.equity=equity
    with pytest.raises((ValueError, TypeError)):
        adapter.account(RiskPolicy(), NOW)
    assert adapter.identity() == {'account_id':'verified-account','environment':'paper'}
    assert adapter.operational_account(NOW).account_id == 'verified-account'


@pytest.mark.parametrize('effect,expected', [('debit', 1.1), ('credit', -1.1)])
def test_multi_leg_prices_follow_sdk_signed_debit_credit_contract(effect, expected):
    adapter, _ = broker()
    captured=[]
    adapter.client.submit_order=lambda request: captured.append(request) or {'id':'fake-broker-id'}
    legs = tuple(OrderLeg(instrument=Instrument(symbol=symbol,asset_type='option',multiplier='100',
        liquidity_capacity='1'),side=side,position_intent=side+'_to_close')
        for symbol, side in [('AAPL260918C00100000','sell'),('AAPL260918C00105000','buy')])
    intent=OrderIntent(account_id='verified-account',environment='paper',purpose='close',group_id='group',
        legs=legs,limit_price='1.10',limit_effect=effect,expires_at=NOW+timedelta(seconds=5),
        reason='synthetic close')
    adapter.submit(intent,Decimal('1'),'stable-intent-id')
    assert captured[0].limit_price == expected
    assert len(captured[0].legs) == 2
    assert captured[0].client_order_id == 'stable-intent-id'


def test_adjusted_root_is_not_approved_from_multiplier_alone():
    assert _standard_contract(SimpleNamespace(size='100',root_symbol='AAPL',underlying_symbol='AAPL'))
    assert not _standard_contract(SimpleNamespace(size='100',root_symbol='AAPL1',underlying_symbol='AAPL'))


def test_crypto_instrument_cannot_be_reclassified_as_stock():
    adapter,_=broker()
    adapter.client.get_asset=lambda symbol: SimpleNamespace(asset_class=AssetClass.CRYPTO)
    with pytest.raises(ValueError,match='UNSUPPORTED_SESSION_ASSET_CLASS'):
        adapter.instrument('BTCUSD',Decimal('1'))


def test_legacy_position_api_cannot_bypass_another_database_runtime(tmp_path, monkeypatch):
    from uuid import uuid4
    import requests
    from src.account.account_manager import AccountManager, AccountAPIError
    from src.account.position_manager import PositionManager
    from src.runtime.store import Store
    from src.runtime.models import utc_now
    account_id = 'synthetic-' + uuid4().hex
    store = Store(tmp_path / 'different-session-database.sqlite3')
    store.acquire_lease('paper:' + account_id, 'fixture-owner', utc_now())
    manager = AccountManager(api_key='synthetic',secret_key='synthetic',paper=True)
    monkeypatch.setattr(manager, 'get_account', lambda: {'id': account_id})
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy broker write reached network')
    monkeypatch.setattr(requests, 'request', forbidden)
    try:
        with pytest.raises(AccountAPIError,match='owns this account'):
            PositionManager(manager).close_all_positions()
    finally:
        store.close()
