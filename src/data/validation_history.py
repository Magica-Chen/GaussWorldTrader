"""Reproducible, read-only Alpaca minute history for offline session validation.

Only historical data and the paper account's exchange calendar are requested.
Raw paginated responses remain compressed beside normalized bars and a manifest;
an interrupted retrieval resumes only after verifying the saved page hashes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from datetime import time as day_time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
CALENDAR_URL = "https://paper-api.alpaca.markets/v2/calendar"
UNIVERSE = ("SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "TLT")
NY = ZoneInfo("America/New_York")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _get(session: requests.Session, url: str, params: dict) -> object:
    for attempt in range(5):
        try:
            response = session.get(url, params=params, timeout=40, allow_redirects=False)
        except requests.RequestException:
            if attempt == 4:
                raise RuntimeError("HISTORY_NETWORK_FAILED") from None
            time.sleep(min(2**attempt, 15))
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == 4:
            raise RuntimeError(f"HISTORY_HTTP_{response.status_code}")
        time.sleep(min(2 ** (attempt + 1), 15))
    raise RuntimeError("HISTORY_RETRIES_EXHAUSTED")


def _sessions(raw: list, completed_end: date) -> list[dict]:
    sessions = []
    for row in raw:
        session_date = date.fromisoformat(row["date"])
        opening = datetime.combine(session_date, day_time.fromisoformat(row["open"]), NY)
        closing = datetime.combine(session_date, day_time.fromisoformat(row["close"]), NY)
        if opening >= closing:
            raise ValueError("INVALID_EXCHANGE_SESSION")
        sessions.append(
            {
                "id": f"US_EQUITIES:{session_date}",
                "session_date": str(session_date),
                "calendar_version": "alpaca-calendar-v1",
                "open": opening.astimezone(UTC).isoformat(),
                "close": closing.astimezone(UTC).isoformat(),
            }
        )
        if session_date > completed_end:
            break
    if not sessions:
        raise ValueError("EXCHANGE_CALENDAR_EMPTY")
    return sessions


def _write_bars(directory: Path, pages: list, symbols: tuple[str, ...]) -> dict:
    destination = directory / "bars.csv.gz"
    temporary = directory / "bars.csv.gz.tmp"
    counts, first, last = Counter(), {}, {}
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp", "symbol", "open", "high", "low", "close", "volume"))
        for page in pages:
            path = directory / page["file"]
            if _sha(path) != page["sha256"]:
                raise ValueError("SAVED_HISTORY_HASH_MISMATCH")
            with gzip.open(path, "rt", encoding="utf-8") as page_stream:
                body = json.load(page_stream)
            for symbol, bars in body["bars"].items():
                if symbol not in symbols:
                    raise ValueError("UNREQUESTED_HISTORY_SYMBOL")
                for bar in bars:
                    stamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                    if stamp.utcoffset() is None or (symbol in last and stamp <= last[symbol]):
                        raise ValueError("DUPLICATE_OR_UNORDERED_HISTORY_TIMESTAMP")
                    values = [bar[key] for key in ("o", "h", "l", "c", "v")]
                    if any(
                        type(value) not in {int, float} or not math.isfinite(value)
                        for value in values
                    ):
                        raise ValueError("INVALID_HISTORY_NUMBER")
                    opening, high, low, closing, volume = values
                    if (
                        min(opening, high, low, closing) <= 0
                        or volume < 0
                        or not (low <= min(opening, closing) <= max(opening, closing) <= high)
                    ):
                        raise ValueError("INVALID_HISTORY_OHLCV")
                    first.setdefault(symbol, stamp)
                    last[symbol] = stamp
                    counts[symbol] += 1
                    writer.writerow((stamp.astimezone(UTC).isoformat(), symbol, *values))
    if set(counts) != set(symbols):
        raise ValueError("MISSING_REQUESTED_SYMBOL_HISTORY")
    temporary.replace(destination)
    return {
        "file": destination.name,
        "sha256": _sha(destination),
        "rows": sum(counts.values()),
        "counts_by_symbol": dict(counts),
        "first_by_symbol": {key: value.isoformat() for key, value in first.items()},
        "last_by_symbol": {key: value.isoformat() for key, value in last.items()},
    }


def download_history(
    directory: Path,
    start: date,
    end: date,
    *,
    symbols: tuple[str, ...] = UNIVERSE,
    max_pages: int = 256,
) -> dict:
    """Download fixed-universe raw SIP history; never infer current tradability."""
    if start > end or not symbols or len(set(symbols)) != len(symbols) or max_pages < 1:
        raise ValueError("INVALID_HISTORY_REQUEST")
    start_time = datetime.combine(start, day_time.min, NY).astimezone(UTC)
    end_time = datetime.combine(end + timedelta(days=1), day_time.min, NY).astimezone(
        UTC
    ) - timedelta(microseconds=1)
    if end_time > datetime.now(UTC) - timedelta(minutes=15, seconds=2):
        raise ValueError("COMPLETED_DELAYED_HISTORY_REQUIRED")
    request = {
        "symbols": ",".join(symbols),
        "timeframe": "1Min",
        "feed": "sip",
        "adjustment": "raw",
        "asof": "-",
        "sort": "asc",
        "limit": 10000,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["request"] != request:
            raise ValueError("EXISTING_HISTORY_REQUEST_MISMATCH")
    else:
        manifest = {
            "schema_version": 1,
            "status": "DOWNLOADING",
            "request": request,
            "data_endpoint": DATA_URL,
            "calendar_endpoint": CALENDAR_URL,
            "retrieval_started_at": datetime.now(UTC).isoformat(),
            "pages": [],
            "page_cap": max_pages,
            "interpretation": {
                "bar_timestamp": "UTC minute start; all returned market hours retained",
                "adjustment": "Raw prices; splits and other corporate actions are not normalized.",
                "universe": "Operator-selected fixed symbols, not a point-in-time or survivorship-free universe.",
                "scope": "Historical signal validation inputs only; no measured strategy performance.",
            },
        }
        _save(manifest_path, manifest)
    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("ALPACA_CREDENTIALS_REQUIRED")
    with requests.Session() as session:
        session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        calendar_path = directory / "calendar-response.json"
        if not calendar_path.exists():
            _save(
                calendar_path,
                _get(
                    session,
                    CALENDAR_URL,
                    {
                        "start": str(start),
                        "end": str(end + timedelta(days=7)),
                    },
                ),
            )
        sessions = _sessions(json.loads(calendar_path.read_text()), end)
        sessions_path = directory / "sessions.json"
        _save(sessions_path, sessions)
        manifest["calendar"] = {
            "raw_file": calendar_path.name,
            "raw_sha256": _sha(calendar_path),
            "file": sessions_path.name,
            "sha256": _sha(sessions_path),
            "session_count": len(sessions),
            "start": str(start),
            "end": str(end + timedelta(days=7)),
        }
        token = None
        complete = False
        for page in manifest["pages"]:
            path = directory / page["file"]
            if _sha(path) != page["sha256"]:
                raise ValueError("SAVED_HISTORY_HASH_MISMATCH")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                token = json.load(stream).get("next_page_token")
            complete = token is None
        seen_tokens = set()
        while not complete:
            if len(manifest["pages"]) >= max_pages:
                raise RuntimeError("HISTORY_PAGE_CAP_REACHED")
            params = {**request, **({"page_token": token} if token else {})}
            body = _get(session, DATA_URL, params)
            if (
                not isinstance(body, dict)
                or not isinstance(body.get("bars"), dict)
                or "next_page_token" not in body
                or (
                    body["next_page_token"] is not None
                    and not isinstance(body["next_page_token"], str)
                )
            ):
                raise ValueError("HISTORY_RESPONSE_INVALID")
            number = len(manifest["pages"]) + 1
            path = directory / f"page-{number:04d}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(body, stream, separators=(",", ":"))
            manifest["pages"].append(
                {
                    "file": path.name,
                    "sha256": _sha(path),
                    "rows": sum(len(rows) for rows in body["bars"].values()),
                    "retrieved_at": datetime.now(UTC).isoformat(),
                }
            )
            token = body.get("next_page_token")
            if token in seen_tokens:
                raise RuntimeError("HISTORY_PAGINATION_CYCLE")
            seen_tokens.add(token)
            complete = token is None
            _save(manifest_path, manifest)
            if number % 10 == 0 or complete:
                print(
                    json.dumps(
                        {
                            "pages": number,
                            "rows": sum(page["rows"] for page in manifest["pages"]),
                            "pagination_complete": complete,
                        }
                    ),
                    flush=True,
                )
            if not complete:
                time.sleep(0.5)
    manifest["bars"] = _write_bars(directory, manifest["pages"], symbols)
    manifest["status"] = "COMPLETE"
    manifest["pagination_exhausted"] = True
    manifest["retrieval_completed_at"] = datetime.now(UTC).isoformat()
    _save(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=256)
    args = parser.parse_args()
    manifest = download_history(args.directory, args.start, args.end, max_pages=args.max_pages)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "bars": manifest["bars"],
                "calendar": manifest["calendar"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
