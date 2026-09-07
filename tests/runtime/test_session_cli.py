import sys
from types import SimpleNamespace

from typer.testing import CliRunner
from src.runtime import cli


def test_help_has_session_commands_without_legacy_imports():
    result = CliRunner().invoke(cli.app, ['--help'])
    assert result.exit_code == 0
    assert 'compare-capital' in result.output
    assert 'request-flatten' in result.output


def test_read_status_never_builds_runtime(monkeypatch):
    monkeypatch.setattr(cli, 'client_for', lambda *args: SimpleNamespace(status=lambda: {'runtime_state': 'STOPPED'}))
    result = CliRunner().invoke(cli.app, ['status'])
    assert result.exit_code == 0
    assert 'STOPPED' in result.output


def test_command_requires_token(monkeypatch):
    monkeypatch.setattr(cli, 'configuration', lambda *args: SimpleNamespace(control_token_env='TEST_GAUSS_TOKEN'))
    monkeypatch.delenv('TEST_GAUSS_TOKEN', raising=False)
    monkeypatch.setattr(cli, 'client_for', lambda *args: (_ for _ in ()).throw(AssertionError('Must not connect')))
    result = CliRunner().invoke(cli.app, ['pause-entries', '--reason', 'test'])
    assert result.exit_code != 0
    assert 'TEST_GAUSS_TOKEN' in result.output


def test_flatten_requires_matching_account(monkeypatch):
    monkeypatch.setattr(cli, 'client_for', lambda *args: SimpleNamespace(status=lambda: {'account_id': 'actual'}))
    result = CliRunner().invoke(cli.app, ['request-flatten', '--reason', 'test', '--confirm-account', 'wrong'])
    assert result.exit_code != 0
    assert 'actual session account ID' in result.output


def test_role_request_is_queued(monkeypatch):
    calls=[]
    monkeypatch.setattr(cli, 'send_command', lambda *args, **kwargs: calls.append((args, kwargs)))
    result = CliRunner().invoke(cli.app, ['run-task', '--task', 'post', '--reason', 'audit'])
    assert result.exit_code == 0
    assert calls[0][0][0] == 'run_task'
    assert calls[0][1]['payload'] == {'role': 'PostGauss'}


def test_pricing_preview_does_not_connect_or_submit(tmp_path, monkeypatch):
    import json
    document={'id':'fixture-pricing','model':'fixture-model','input_usd_per_million':'1',
              'output_usd_per_million':'2','verified_by':'operator','source_reference':'fixture://pricing',
              'expires_at':'2099-01-01T00:00:00Z'}
    path=tmp_path/'pricing.json';path.write_text(json.dumps(document))
    monkeypatch.setattr(cli,'client_for',lambda *args: (_ for _ in ()).throw(AssertionError('No service needed')))
    result=CliRunner().invoke(cli.app,['approve-model-pricing','--pricing',str(path),'--preview'])
    assert result.exit_code == 0, result.output
    assert 'fixture-pricing' in result.output


def test_pricing_registration_is_confirmed_and_operator_bound(tmp_path, monkeypatch):
    import json
    document={'id':'fixture-pricing','model':'fixture-model','input_usd_per_million':'1',
              'output_usd_per_million':'2','verified_by':'reviewer','source_reference':'fixture://pricing',
              'expires_at':'2099-01-01T00:00:00Z'}
    path=tmp_path/'pricing.json';path.write_text(json.dumps(document))
    monkeypatch.setattr(cli,'client_for',lambda *args: SimpleNamespace(status=lambda:{'account_id':'actual'}))
    calls=[]
    monkeypatch.setattr(cli,'send_command',lambda *args,**kwargs:calls.append((args,kwargs)))
    result=CliRunner().invoke(cli.app,['approve-model-pricing','--pricing',str(path),'--operator','reviewer',
                                     '--reason','reviewed source','--confirm-account','actual'])
    assert result.exit_code == 0,result.output
    assert calls[0][0][0]=='approve_model_pricing'
    assert calls[0][1]['confirmed'] is True
    assert calls[0][1]['payload']['pricing']['verified_by']=='reviewer'
