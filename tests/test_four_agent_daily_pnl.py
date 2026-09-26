"""Daily loss controls preserve broker history across starts and cash movements."""

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.runtime.calendar import SessionCalendar
from src.runtime.models import AccountProfile, TradingSession
from src.runtime.service import SessionService
from src.runtime.store import Store


def moment(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@pytest.fixture
def runtime(tmp_path):
    store = Store(tmp_path / "pnl.sqlite3")
    sessions = tuple(
        TradingSession(
            id=day,
            session_date=day,
            calendar_version="test",
            open=moment(day + "T13:30:00"),
            close=moment(day + "T20:00:00"),
        )
        for day in ("2026-09-25", "2026-09-28")
    )
    service = SimpleNamespace(store=store, scope="paper:test", calendar=SessionCalendar(sessions))
    yield service
    store.close()


def account(**changes):
    return AccountProfile(
        account_id="test",
        environment="paper",
        equity="9500",
        cash="5000",
        buying_power="5000",
        currency="USD",
        observed_at=moment("2026-09-28T16:00:00"),
        mandate_id="test",
        **changes,
    )


def pnl(runtime, value, when="2026-09-28T16:00:00"):
    return SessionService._daily_trading_pnl(runtime, value, moment(when))


def test_first_start_retains_earlier_daily_loss_and_immutable_reference(runtime):
    assert pnl(runtime, account(previous_close_equity="10000")) == Decimal("-500")
    assert pnl(runtime, account(previous_close_equity="9500")) == Decimal("-500")


def test_missing_broker_reference_never_invents_zero_pnl(runtime):
    assert pnl(runtime, account()) is None
    assert runtime.store.list("pnl_reference", runtime.scope) == []


def test_after_close_and_weekend_never_seed_next_session_reference(runtime):
    assert pnl(runtime, account(previous_close_equity="10000"), "2026-09-25T22:00:00") == -500
    assert pnl(runtime, account(previous_close_equity="10000"), "2026-09-26T12:00:00") is None
    references = runtime.store.list("pnl_reference", runtime.scope)
    assert len(references) == 1 and references[0]["trading_date"] == "2026-09-25"
    assert pnl(runtime, account(previous_close_equity="9600")) == -100


def test_dated_cashflows_count_even_if_seen_before_start_and_arrive_late(runtime):
    runtime.store.put(
        "activities",
        {"id": "deposit", "activity_type": "CSD", "date": "2026-09-28", "net_amount": "1000"},
        runtime.scope,
    )
    runtime.store.put(
        "activities",
        {"id": "old", "activity_type": "CSD", "date": "2026-09-25", "net_amount": "2000"},
        runtime.scope,
    )
    value = account(previous_close_equity="10000")
    assert pnl(runtime, value) == -1500
    runtime.store.put(
        "activities",
        {"id": "withdrawal", "activity_type": "CSW", "date": "2026-09-28", "net_amount": "-300"},
        runtime.scope,
    )
    assert pnl(runtime, value) == -1200


@pytest.mark.parametrize(
    "fields",
    [{}, {"date": "invalid"}, {"date": "2026-09-28"}, {"date": "2026-09-28", "net_amount": "NaN"}],
)
def test_unresolvable_cashflow_keeps_daily_loss_unknown(runtime, fields):
    runtime.store.put(
        "activities", {"id": "unknown", "activity_type": "CSD", **fields}, runtime.scope
    )
    assert pnl(runtime, account(previous_close_equity="10000")) is None


def test_explicit_replay_daily_pnl_preserved(runtime):
    assert pnl(runtime, account(daily_pnl="-321")) == -321
