"""Synthetic validation of historical data pagination and durable input artifacts."""

import gzip
import json
from datetime import date

import pytest

from src.data import validation_history as history


def setup_source(monkeypatch, pages):
    calls = []
    monkeypatch.setattr(history, "load_dotenv", lambda: None)
    monkeypatch.setattr(history.time, "sleep", lambda delay: None)
    monkeypatch.setenv("ALPACA_API_KEY", "synthetic-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "synthetic-secret")

    def fetch(session, url, params):
        calls.append((url, dict(params)))
        if url == history.CALENDAR_URL:
            return [
                {"date": "2025-03-03", "open": "09:30", "close": "16:00"},
                {"date": "2025-03-04", "open": "09:30", "close": "16:00"},
            ]
        return pages[params.get("page_token")]

    monkeypatch.setattr(history, "_get", fetch)
    return calls


def bar(stamp):
    return {"t": stamp, "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 20}


def test_short_pages_follow_tokens_and_resume_verified_artifacts(tmp_path, monkeypatch):
    pages = {
        None: {"bars": {"TEST": [bar("2025-03-03T14:30:00Z")]}, "next_page_token": "second"},
        "second": {"bars": {"TEST": [bar("2025-03-03T14:31:00Z")]}, "next_page_token": None},
    }
    calls = setup_source(monkeypatch, pages)
    manifest = history.download_history(
        tmp_path, date(2025, 3, 3), date(2025, 3, 3), symbols=("TEST",)
    )
    assert manifest["status"] == "COMPLETE" and manifest["pagination_exhausted"]
    assert manifest["bars"]["rows"] == 2 and len(manifest["pages"]) == 2
    assert [params.get("page_token") for url, params in calls if url == history.DATA_URL] == [
        None,
        "second",
    ]
    assert manifest["request"]["adjustment"] == "raw" and manifest["request"]["feed"] == "sip"
    assert "synthetic" not in json.dumps(manifest)
    sessions = json.loads((tmp_path / "sessions.json").read_text())
    assert sessions[0]["open"] == "2025-03-03T14:30:00+00:00"
    assert sessions[-1]["session_date"] == "2025-03-04"
    with gzip.open(tmp_path / "bars.csv.gz", "rt") as stream:
        lines = stream.read().splitlines()
    assert lines[0] == "timestamp,symbol,open,high,low,close,volume" and len(lines) == 3
    calls.clear()
    again = history.download_history(
        tmp_path, date(2025, 3, 3), date(2025, 3, 3), symbols=("TEST",)
    )
    assert again["bars"]["rows"] == 2 and not calls
    (tmp_path / "page-0001.json.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SAVED_HISTORY_HASH_MISMATCH"):
        history.download_history(tmp_path, date(2025, 3, 3), date(2025, 3, 3), symbols=("TEST",))


def test_pagination_cap_preserves_partial_state(tmp_path, monkeypatch):
    setup_source(
        monkeypatch,
        {
            None: {"bars": {"TEST": [bar("2025-03-03T14:30:00Z")]}, "next_page_token": "second"},
        },
    )
    with pytest.raises(RuntimeError, match="HISTORY_PAGE_CAP_REACHED"):
        history.download_history(
            tmp_path, date(2025, 3, 3), date(2025, 3, 3), symbols=("TEST",), max_pages=1
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "DOWNLOADING" and len(manifest["pages"]) == 1
    assert not (tmp_path / "bars.csv.gz").exists()


def test_duplicate_timestamps_are_rejected(tmp_path, monkeypatch):
    stamp = "2025-03-03T14:30:00Z"
    setup_source(
        monkeypatch,
        {
            None: {"bars": {"TEST": [bar(stamp), bar(stamp)]}, "next_page_token": None},
        },
    )
    with pytest.raises(ValueError, match="DUPLICATE_OR_UNORDERED_HISTORY_TIMESTAMP"):
        history.download_history(tmp_path, date(2025, 3, 3), date(2025, 3, 3), symbols=("TEST",))
    assert not (tmp_path / "bars.csv.gz").exists()
