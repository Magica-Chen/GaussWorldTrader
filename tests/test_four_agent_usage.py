"""Usage accounting must not hide missing counts or mutate a trading session."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from src.runtime import cli
from src.runtime.service import SessionClient
from src.runtime.store import Store
from src.runtime.usage import token_usage_summary

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


@pytest.fixture
def ledger(tmp_path):
    store = Store(tmp_path / "usage.sqlite3")
    for key, scope, session, state, incoming, outgoing in (
        ("a", "paper:a", "day1", "COMPLETED", 100, 20),
        ("b", "paper:a", "day1", "FAILED", None, None),
        ("c", "paper:a", "day2", "RESERVED", None, None),
        ("d", "paper:other", "day1", "COMPLETED", 9999, 9999),
    ):
        store.project(
            "codex_usage",
            key,
            {
                "id": key,
                "session_id": session,
                "role": "PostGauss",
                "state": state,
                "input_tokens": incoming,
                "output_tokens": outgoing,
            },
            scope,
        )
    store.project(
        "automation_control",
        "paper:a",
        {
            "remaining_percent": 86,
            "monitor_only": False,
            "quota": {"observed_at": NOW.timestamp(), "windows": []},
        },
        "paper:a",
    )
    yield store
    store.close()


def test_totals_unknown_usage_and_scope_are_separate(ledger):
    before = ledger.db.total_changes
    result = token_usage_summary(ledger, "paper:a", now=NOW)
    assert result["totals"] == {
        "calls": 3,
        "completed": 1,
        "failed": 1,
        "pending": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "unknown_usage_calls": 2,
    }
    assert result["allowance"]["remaining_percent"] == 86
    assert result["allowance"]["stale"] is False
    assert result["by_session"]["day2"]["unknown_usage_calls"] == 1
    assert ledger.db.total_changes == before


def test_session_filter_does_not_filter_account_wide_allowance(ledger):
    result = token_usage_summary(ledger, "paper:a", "day2", now=NOW)
    assert result["totals"]["calls"] == 1
    assert result["totals"]["total_tokens"] == 0
    assert result["allowance"]["remaining_percent"] == 86
    assert result["totals"]["unknown_usage_calls"] == 1


def test_unknown_scope_does_not_aggregate_other_accounts(ledger):
    result = token_usage_summary(ledger, None, now=NOW)
    assert result["totals"]["calls"] == 0
    assert result["allowance"]["remaining_percent"] is None
    assert result["allowance"]["stale"] is True


@pytest.mark.parametrize("offset", [-121, 1])
def test_stale_or_future_quota_is_labelled(ledger, offset):
    ledger.project(
        "automation_control",
        "paper:a",
        {
            "remaining_percent": 86,
            "quota": {"observed_at": NOW.timestamp() + offset},
        },
        "paper:a",
    )
    assert token_usage_summary(ledger, "paper:a", now=NOW)["allowance"]["stale"]


def test_client_reads_without_starting_runtime_or_writing(ledger, monkeypatch):
    monkeypatch.setattr("src.runtime.service.build_service", lambda *a: pytest.fail("No runtime"))
    client = SessionClient(ledger.path, "a", "paper")
    assert client.usage()["totals"]["total_tokens"] == 120


def test_cli_json_and_watch_exit_never_control_trading(ledger, monkeypatch):
    result = token_usage_summary(ledger, "paper:a", now=NOW)
    calls = []

    def read(session):
        calls.append(session)
        return result

    monkeypatch.setattr(cli, "client_for", lambda *a: SimpleNamespace(usage=read))
    output = CliRunner().invoke(cli.app, ["usage", "--json", "--session", "day1"])
    assert output.exit_code == 0, output.output
    assert '"total_tokens": 120' in output.output
    assert calls == ["day1"]

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("time.sleep", interrupt)
    output = CliRunner().invoke(cli.app, ["usage", "--watch"])
    assert output.exit_code == 0, output.output
    assert "trading service is unchanged" in output.output
    output = CliRunner().invoke(cli.app, ["usage", "--watch", "--json"])
    assert output.exit_code != 0
