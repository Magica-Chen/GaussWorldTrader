import pytest
import gauss_bot


def test_bot_defaults_do_not_arm_execution(monkeypatch):
    from src.runtime.models import RuntimeConfig
    import src.settings
    monkeypatch.setattr(src.settings, 'get_gauss_config', lambda path: RuntimeConfig())
    config = gauss_bot.load_configuration(gauss_bot.parser().parse_args([]))
    assert config.execution_mode == 'shadow'
    assert config.environment == 'paper'
    assert config.data_profile == 'FREE_DELAYED'
    assert not config.account_id


def test_paid_profile_does_not_change_execution(monkeypatch):
    from src.runtime.models import RuntimeConfig
    import src.settings
    monkeypatch.setattr(src.settings, 'get_gauss_config', lambda path: RuntimeConfig())
    args = gauss_bot.parser().parse_args(['--data-profile', 'SUBSCRIBED_REALTIME'])
    config = gauss_bot.load_configuration(args)
    assert config.execution_mode == 'shadow'


def test_crypto_is_not_automatically_enabled(monkeypatch):
    from src.runtime.models import RuntimeConfig
    import src.settings
    monkeypatch.setattr(src.settings, 'get_gauss_config', lambda path: RuntimeConfig())
    args = gauss_bot.parser().parse_args(['--symbols', 'BTC/USD'])
    with pytest.raises(ValueError, match='crypto'):
        gauss_bot.load_configuration(args)


def test_live_requires_compatible_explicit_configuration(monkeypatch):
    from src.runtime.models import RuntimeConfig
    import src.settings
    monkeypatch.setattr(src.settings, 'get_gauss_config', lambda path: RuntimeConfig())
    with pytest.raises(ValueError):
        gauss_bot.load_configuration(gauss_bot.parser().parse_args(['--mode', 'live']))


def test_once_reports_clean_shutdown():
    from types import SimpleNamespace
    from src.runtime.runner import run_service
    class Service:
        config = SimpleNamespace(poll_seconds=10)
        def run_once(self): return {'entries_paused': True}
        def request_shutdown(self, **kwargs): return {'stopped': True, 'positions': []}
    records=[]
    assert run_service(Service(), once=True, emit=records.append) == 0
    assert records[-1]['event'] == 'shutdown'


def test_residual_exposure_keeps_management_running():
    from types import SimpleNamespace
    from src.runtime.runner import run_service
    class Service:
        config = SimpleNamespace(poll_seconds=100)
        cycles=0
        def run_once(self):
            self.cycles += 1
            return {'runtime_state': 'MANAGE_ONLY'}
        def request_shutdown(self, **kwargs):
            assert kwargs['acknowledge_unmanaged_exposure'] is False
            return {'stopped': self.cycles >= 2, 'positions': ['AAPL'] if self.cycles == 1 else []}
    service=Service()
    waits=[]
    records=[]
    assert run_service(service, once=True, emit=records.append, wait=waits.append) == 0
    assert service.cycles == 2
    assert waits == [60]
    assert any(row['event'] == 'management_continues' for row in records)


def test_cycle_failure_does_not_abandon_exposure():
    from types import SimpleNamespace
    from src.runtime.runner import run_service
    class Service:
        config = SimpleNamespace(poll_seconds=1)
        cycles=0
        def run_once(self):
            self.cycles += 1
            if self.cycles == 1:
                raise RuntimeError('research unavailable')
            return {'runtime_state': 'MANAGE_ONLY'}
        def request_shutdown(self, **kwargs): return {'stopped': self.cycles >= 2}
    service=Service()
    assert run_service(service, emit=lambda _: None, wait=lambda _: None) == 0
    assert service.cycles == 2


def test_explicit_acknowledgement_reaches_shutdown_gate():
    from types import SimpleNamespace
    from src.runtime.runner import run_service
    class Service:
        config = SimpleNamespace(poll_seconds=1)
        def run_once(self): return {}
        def request_shutdown(self, **kwargs):
            assert kwargs['acknowledge_unmanaged_exposure'] is True
            return {'stopped': True, 'positions': ['AAPL'], 'supervision_ending': True}
    assert run_service(Service(), once=True, acknowledge_unmanaged_exposure=True, emit=lambda _: None) == 0


def test_operator_stop_ended_service_exits_without_second_cycle():
    from types import SimpleNamespace
    from src.runtime.runner import run_service
    class Service:
        config = SimpleNamespace(poll_seconds=1)
        def run_once(self): return {'supervision': 'ENDED'}
        def request_shutdown(self, **kwargs): raise AssertionError('Already closed by operator command')
    assert run_service(Service(), emit=lambda _: None, wait=lambda _: (_ for _ in ()).throw(AssertionError('No wait'))) == 0


def test_text_console_summarizes_changes_without_repeating_configuration():
    from io import StringIO
    from src.runtime.console import ConsoleOutput

    stream = StringIO()
    emit = ConsoleOutput('text', stream)
    status = {
        'environment': 'paper', 'execution_mode': 'shadow',
        'runtime_state': 'MANAGE_ONLY', 'entries_paused': True,
        'readiness': ['NO_APPROVED_STRATEGIES'],
        'health': {'reconciliation': 'HEALTHY'},
        'account_id': 'private-account-id',
    }
    emit({'event': 'heartbeat', 'at': '2026-09-07T19:03:49+00:00', 'status': status})
    emit({'event': 'heartbeat', 'at': '2026-09-07T19:03:59+00:00', 'status': status})
    status['health']['reconciliation'] = 'FAILED'
    emit({'event': 'heartbeat', 'at': '2026-09-07T19:04:09+00:00', 'status': status})
    output = stream.getvalue()
    assert output.count('Gauss | paper / shadow') == 2
    assert output.count('heartbeat') == 3
    assert 'entries paused' in output
    assert 'no approved strategies' in output
    assert 'reconciliation: failed' in output
    assert 'private-account-id' not in output


def test_json_console_preserves_full_records_for_redirected_output():
    import json
    from io import StringIO
    from src.runtime.console import ConsoleOutput

    for mode in ('auto', 'json'):
        stream = StringIO()
        record = {'event': 'heartbeat', 'status': {'account_id': 'test', '_revision': 10}}
        ConsoleOutput(mode, stream)(record)
        assert json.loads(stream.getvalue()) == record


def test_second_interrupt_after_shutdown_notice_acknowledges_exposure(monkeypatch):
    import signal
    from types import SimpleNamespace
    from src.runtime.runner import run_service

    handlers = {}
    monkeypatch.setattr(signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))
    calls = []
    records = []

    class Service:
        config = SimpleNamespace(poll_seconds=1)
        def run_once(self):
            if not calls:
                handlers[signal.SIGINT](signal.SIGINT, None)
                # Rapid repeated interrupts before a notice do not acknowledge exposure.
                handlers[signal.SIGINT](signal.SIGINT, None)
            return {'entries_paused': True}
        def request_shutdown(self, **kwargs):
            calls.append(kwargs['acknowledge_unmanaged_exposure'])
            return {'stopped': calls[-1], 'remaining_groups': [{'id': 'held'}]}

    def wait(_):
        assert records[-1]['event'] == 'management_continues'
        handlers[signal.SIGINT](signal.SIGINT, None)

    assert run_service(Service(), emit=records.append, wait=wait) == 0
    assert calls == [False, True]
    assert 'Ctrl+C again' in records[2]['message']


def test_repeated_sigterm_does_not_acknowledge_exposure(monkeypatch):
    import signal
    from types import SimpleNamespace
    from src.runtime.runner import run_service

    handlers = {}
    monkeypatch.setattr(signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))
    calls = []

    class Service:
        config = SimpleNamespace(poll_seconds=1)
        def run_once(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return {}
        def request_shutdown(self, **kwargs):
            calls.append(kwargs['acknowledge_unmanaged_exposure'])
            return {'stopped': len(calls) == 2}

    assert run_service(Service(), emit=lambda _: None, wait=lambda _: None) == 0
    assert calls == [False, False]


def test_once_dispatches_research_without_constructing_supervisor(monkeypatch, tmp_path, capsys):
    from src.runtime.models import RuntimeConfig
    import src.runtime.screening as screening

    monkeypatch.setattr(gauss_bot, 'load_configuration', lambda _: RuntimeConfig())
    report = {'execution_eligible': False, 'candidates': []}
    monkeypatch.setattr(screening, 'run_research', lambda _: (report, tmp_path / 'report.md'))
    assert gauss_bot.main(['--once', '--output', 'json'],
        service_factory=lambda _: pytest.fail('Research must not start supervision')) == 0
    import json
    assert json.loads(capsys.readouterr().out)['execution_eligible'] is False


def test_daily_screen_uses_complete_history_and_registered_signals():
    import pandas as pd
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.runtime.screening import analyze, completed_daily

    timestamps = pd.date_range('2026-05-01', periods=90, freq='B', tz='America/New_York')
    frame = pd.DataFrame({'symbol': ['TEST']*90, 'timestamp': timestamps,
        'open': range(100,190), 'close': range(101,191), 'high': range(102,192),
        'low': range(99,189), 'volume': [1_000_000]*90})
    last = timestamps[-2].date()
    completed = completed_daily(frame, last)
    assert len(completed) == 89
    rows = analyze(completed, ('momentum', 'trend_following', 'mean_reversion'),
                   datetime.combine(last, datetime.min.time(), ZoneInfo('America/New_York')))
    assert len(rows) == 1
    row = rows[0]
    assert row['signals']['trend_following']['signal'] == 'BUY'
    assert row['stop'] < row['entry'] < row['max_entry'] < row['target']
    assert row['close'] == 189
    assert analyze(completed.iloc[:20], ('trend_following',), timestamps[19]) == []
    with pytest.raises(ValueError, match='does not support'):
        analyze(completed, ('made_up_strategy',), timestamps[-2])
