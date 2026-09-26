"""Execution quotes remain exact and bounded as the durable market ledger grows."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.runtime.evidence import EvidenceService
from src.runtime.models import MarketEvent, RuntimeConfig
from src.runtime.store import Store, canonical

NOW = datetime(2026, 9, 28, 15, tzinfo=UTC)


def quote(**updates):
    values = {
        "symbol": "TEST",
        "source": "fixture",
        "source_version": "initial-poll",
        "event_type": "quote",
        "effective_event_time": NOW,
        "received_at": NOW,
        "available_at": NOW,
        "feed": "iex",
        "quality_class": "genuine",
        "bid": Decimal("100"),
        "ask": Decimal("100.01"),
        "bid_size": Decimal("2"),
        "ask_size": Decimal("3"),
    }
    return MarketEvent(**{**values, **updates})


@pytest.fixture
def evidence(tmp_path):
    store = Store(tmp_path / "quotes.sqlite3")
    service = EvidenceService(store, RuntimeConfig(), "paper:fixture")
    yield service
    store.close()


def test_repeated_poll_keeps_original_receipt_and_quote_age(evidence):
    original = evidence.ingest_market(quote())
    for second in range(1, 100):
        later = NOW + timedelta(seconds=second)
        assert (
            evidence.ingest_market(
                quote(
                    source_version=later.isoformat(),
                    received_at=later,
                    available_at=later,
                )
            )
            == original
        )
    rows = evidence.store.list("market_events", evidence.scope)
    assert len(rows) == 1
    assert MarketEvent.model_validate(rows[0]).received_at == NOW
    assert evidence.current_quote("TEST", NOW + timedelta(seconds=6), "iex") is None


def test_same_timestamp_corrections_preserve_availability_and_latest_sequence(evidence):
    first = evidence.ingest_market(quote())
    later = NOW + timedelta(microseconds=1)
    corrected = evidence.ingest_market(
        quote(
            source_version="poll-2",
            bid=Decimal("100.005"),
            available_at=later,
            received_at=later,
        )
    )
    size_correction = evidence.ingest_market(
        quote(
            source_version="poll-3",
            bid=Decimal("100.005"),
            bid_size=4,
            available_at=later,
            received_at=later,
        )
    )
    assert len({first, corrected, size_correction}) == 3
    assert evidence.current_quote("TEST", NOW, "iex").id == first
    assert evidence.current_quote("TEST", later, "iex").id == size_correction


@pytest.mark.parametrize("field", ["effective_event_time", "available_at", "received_at"])
def test_future_microsecond_never_leaks_into_current_quote(evidence, field):
    earlier = evidence.ingest_market(
        quote(
            effective_event_time=NOW - timedelta(microseconds=1),
            source_version="earlier",
        )
    )
    future = evidence.ingest_market(quote(**{field: NOW + timedelta(microseconds=1)}))
    assert future != earlier
    assert evidence.current_quote("TEST", NOW, "iex").id == earlier
    assert evidence.current_quote("TEST", NOW + timedelta(microseconds=1), "iex").id == future


def test_quote_age_boundary_is_inclusive_to_the_microsecond(evidence):
    boundary = evidence.ingest_market(quote(effective_event_time=NOW - timedelta(seconds=5)))
    assert evidence.current_quote("TEST", NOW, "iex", max_age=5).id == boundary
    assert evidence.current_quote("TEST", NOW + timedelta(microseconds=1), "iex", max_age=5) is None
    assert evidence.current_quote("TEST", NOW, "iex", max_age=-1) is None


def test_scope_and_feed_are_exact_and_quote_identity_is_account_scoped(evidence):
    first = evidence.ingest_market(quote())
    other = EvidenceService(evidence.store, evidence.config, "paper:other")
    second = other.ingest_market(quote())
    assert first != second
    assert evidence.current_quote("TEST", NOW, "iex").id == first
    assert other.current_quote("TEST", NOW, "iex").id == second
    assert evidence.current_quote("TEST", NOW, "sip") is None
    assert evidence.current_quote("OTHER", NOW, "iex") is None


@pytest.mark.parametrize(
    "updates",
    [
        {"complete": False},
        {"quality_class": "incomplete"},
        {"bid": 0},
        {"bid": None},
        {"ask": None},
    ],
)
def test_unusable_quotes_are_filtered_before_selection(evidence, updates):
    usable = evidence.ingest_market(quote())
    evidence.ingest_market(quote(**updates))
    assert evidence.current_quote("TEST", NOW, "iex").id == usable


def test_bar_revisions_keep_source_version_identity(evidence):
    first = evidence.ingest_market(quote(event_type="bar", price=100, source_version="v1"))
    second = evidence.ingest_market(quote(event_type="bar", price=100, source_version="v2"))
    assert first != second


def test_old_utc_offset_encoding_matches_new_z_encoding(evidence):
    value = quote().model_dump(mode="json")
    for field in ("effective_event_time", "available_at", "received_at"):
        value[field] = NOW.isoformat()
    evidence.store.put("market_events", value, evidence.scope)
    assert evidence.current_quote("TEST", NOW, "iex").id == value["id"]


def test_large_unrelated_ledger_does_not_expand_quote_read(evidence, monkeypatch):
    selected = evidence.ingest_market(quote())
    unrelated = quote(symbol="UNRELATED", effective_event_time=NOW - timedelta(days=60))
    body = canonical(unrelated.model_dump(mode="json"))
    with evidence.store.transaction():
        evidence.store.db.executemany(
            "INSERT INTO records(kind,id,scope,created_at,body) VALUES('market_events',?,?,?,?)",
            ((f"history-{index}", evidence.scope, NOW.isoformat(), body) for index in range(20000)),
        )
    monkeypatch.setattr(
        evidence.store, "list", lambda *args, **kwargs: pytest.fail("bulk ledger read")
    )
    executed_steps = 0

    def bound_sql_work():
        nonlocal executed_steps
        executed_steps += 1
        return int(executed_steps > 1000)

    evidence.store.db.set_progress_handler(bound_sql_work, 1)
    try:
        assert evidence.current_quote("TEST", NOW, "iex").id == selected
    finally:
        evidence.store.db.set_progress_handler(None, 0)
    assert executed_steps < 500


def test_read_only_store_uses_existing_quote_index_without_mutation(tmp_path):
    path = tmp_path / "quotes.sqlite3"
    store = Store(path)
    expected = EvidenceService(store, RuntimeConfig(), "paper:fixture").ingest_market(quote())
    store.close()
    read_only = Store(path, read_only=True)
    try:
        assert read_only.db.execute("PRAGMA query_only").fetchone()[0] == 1
        service = EvidenceService(read_only, RuntimeConfig(), "paper:fixture")
        assert service.current_quote("TEST", NOW, "iex").id == expected
    finally:
        read_only.close()
