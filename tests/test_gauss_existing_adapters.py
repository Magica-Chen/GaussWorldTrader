from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data.alpaca_provider import AlpacaDataProvider
from src.data.news_provider import NewsDataProvider
from src.settings import get_gauss_config


def test_existing_stock_provider_uses_delayed_sip_without_second_history_delay():
    provider = AlpacaDataProvider.__new__(AlpacaDataProvider)
    provider.settings = SimpleNamespace(session_runtime=SimpleNamespace(
        data_profile='FREE_DELAYED', entitlement_boundary_buffer=2))
    provider.stock_historical_client = Mock()
    provider._process_stock_bars = Mock(return_value=pd.DataFrame())
    now = datetime(2026, 9, 8, 15, tzinfo=timezone.utc)
    with patch('src.data.alpaca_provider.now_et', return_value=now):
        provider.get_stock_bars('TEST', '1Min', now-timedelta(hours=1))
        request = provider.stock_historical_client.get_stock_bars.call_args.args[0]
        assert request.feed.value == 'sip'
        assert request.end == (now-timedelta(seconds=902)).replace(tzinfo=None)
        historical_end = now-timedelta(days=1)
        provider.get_stock_bars('TEST', '1Min', now-timedelta(days=2), historical_end)
        request = provider.stock_historical_client.get_stock_bars.call_args.args[0]
        assert request.end == historical_end.replace(tzinfo=None)


def test_news_corrections_and_cross_provider_provenance_survive_merge():
    provider = NewsDataProvider.__new__(NewsDataProvider)
    stamp = datetime(2026, 9, 8, 15, tzinfo=timezone.utc)
    original = {'id': 1, 'headline': 'Initial', 'url': 'https://example.test/news',
                'created_at': stamp, 'updated_at': stamp}
    first = provider._normalize_alpaca_article(original)
    correction = provider._normalize_alpaca_article({**original, 'headline': 'Corrected',
                    'updated_at': stamp+timedelta(minutes=1)})
    second_source = {**first, 'provider': 'finnhub', 'id': 'finnhub-1'}
    result = provider._merge_news([first, correction, first, second_source])
    assert len(result) == 3
    assert first['source_version'] != correction['source_version']
    assert correction['published_at'] != correction['updated_at']
    assert first['received_at']


def test_config_rejects_unknown_safety_key_and_delay_override(tmp_path):
    path = tmp_path/'config.toml'
    for config in ('[gauss.risk]\nmax_loss_typo=100\n',
                   '[gauss.data.free_delayed]\nmarket_delay_seconds=0\n'):
        path.write_text(config)
        with pytest.raises(ValueError):
            get_gauss_config(path)


def test_config_legacy_ceiling_is_stricter_and_subscription_cannot_arm(tmp_path, monkeypatch):
    monkeypatch.delenv('MAX_POSITION_SIZE', raising=False)
    monkeypatch.delenv('GAUSS_MODE', raising=False)
    path = tmp_path/'config.toml'
    path.write_text('[trading_limits]\nmax_position_size=0.03\n'
                    '[gauss.data]\nprofile="SUBSCRIBED_REALTIME"\n'
                    '[gauss.risk]\nmax_capital_allocation_pct="0.10"\n')
    config = get_gauss_config(path)
    assert str(config.policy.max_capital_allocation_pct) == '0.03'
    assert config.execution_mode == 'shadow'
    assert config.live_trading_enabled is False


def test_snapshot_analysts_never_construct_network_providers():
    from src.agent.multi_agent.agents import FundamentalAnalystAgent, SentimentAnalystAgent
    reader = Mock()
    reader.fundamentals.return_value = {'available': False}
    reader.news.return_value = []
    with patch('src.agent.multi_agent.agents.FundamentalAnalyzer', side_effect=AssertionError), \
         patch('src.agent.multi_agent.agents.NewsDataProvider', side_effect=AssertionError):
        fundamental = FundamentalAnalystAgent(Mock(), 'test', snapshot_reader=reader)
        sentiment = SentimentAnalystAgent(Mock(), snapshot_reader=reader)
    assert fundamental.analyzer is None
    assert sentiment.news_provider is None


def test_fast_momentum_is_price_proxy():
    from src.agent.multi_agent.orchestrator import MultiAgentOrchestrator
    orchestrator = MultiAgentOrchestrator('test', mode='fast')
    assert 'price_proxy' in orchestrator.fast_signal_agents
    assert 'sentiment' not in orchestrator.fast_signal_agents


@pytest.mark.parametrize('count', [None, 0, 3, '2'])
def test_shared_trading_account_accepts_nullable_daytrade_count(count):
    from src.trade.engine.stock_engine import TradingStockEngine

    engine = TradingStockEngine.__new__(TradingStockEngine)
    engine.api = SimpleNamespace(get_account=lambda: SimpleNamespace(
        id='fixture', buying_power='1000', cash='1000', portfolio_value='1000',
        equity='1000', daytrade_count=count, pattern_day_trader=None,
    ))
    info = engine.get_account_info()
    expected = None if count is None else int(count)
    assert info['daytrade_count'] == info['day_trade_count'] == expected
    assert info['cash'] == 1000


def test_live_menu_displays_unavailable_daytrade_count_and_quits(monkeypatch):
    from io import StringIO
    from rich.console import Console
    import live_script

    output = StringIO()
    monkeypatch.setattr(live_script, 'console', Console(file=output, width=100))
    monkeypatch.setattr(live_script, 'show_banner', lambda: None)
    monkeypatch.setattr(live_script, 'show_watchlist_summary', lambda: None)
    monkeypatch.setattr(live_script, 'get_alpaca_base_url', lambda: 'https://paper-api.alpaca.markets')
    context = SimpleNamespace(account_info={'daytrade_count': None, 'pattern_day_trader': None},
        account_config={}, buying_power=1000, cash=1000, portfolio_value=1000,
        shorting_enabled=False, margin_enabled=False)
    monkeypatch.setattr(live_script, 'load_account_context', lambda: context)
    monkeypatch.setattr(live_script.Prompt, 'ask', lambda *args, **kwargs: 'q')
    monkeypatch.setattr(live_script, 'run_trading', lambda *args: pytest.fail('Must not trade'))
    with pytest.raises(SystemExit) as exc:
        live_script.main()
    assert exc.value.code == 0
    text = output.getvalue()
    assert 'Trading Mode Selection' in text
    assert 'Daytrade Count' in text and 'N/A' in text
    assert 'None' not in text


@pytest.mark.parametrize('execute', [True, False])
def test_interactive_mixed_assets_keep_options_research_only(monkeypatch, execute):
    import live_script
    from io import StringIO
    from rich.console import Console

    output = StringIO()
    monkeypatch.setattr(live_script, 'console', Console(file=output, width=140))
    calls = {}
    started = []
    def factory(asset):
        def create(**kwargs):
            calls[asset] = kwargs
            return [SimpleNamespace(start=lambda: started.append(asset))]
        return create
    monkeypatch.setattr(live_script, 'create_stock_engines', factory('stock'))
    monkeypatch.setattr(live_script, 'create_crypto_engines', factory('crypto'))
    monkeypatch.setattr(live_script, 'create_option_engines', factory('option'))
    config = live_script.TradingConfig(asset_types=['stock','crypto','option'],
        symbols={'stock':['AAPL'],'crypto':['BTC/USD'],'option':['AAPL']}, execute=execute)
    live_script.show_final_config(config, None)
    live_script.run_trading(config)
    assert calls['stock']['execute'] is execute
    assert calls['crypto']['execute'] is execute
    assert calls['option']['execute'] is False
    assert calls['option']['auto_exit'] is False
    assert started == ['stock', 'crypto', 'option']
    assert 'Research only (no orders or automatic exits)' in output.getvalue()


def test_options_only_default_configuration_never_offers_execution(monkeypatch):
    import live_script
    config = live_script.TradingConfig(asset_types=['option'], symbols={'option':['AAPL']})
    prompts = []
    monkeypatch.setattr(live_script.Confirm, 'ask', lambda message, **kw: prompts.append(message) or True)
    assert live_script.configure_parameters(config, None).execute is False
    assert prompts == ['Use default parameters?']
