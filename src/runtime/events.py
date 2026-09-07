"""Verified scheduled-event provider boundary; absent coverage is an explicit gap."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from pydantic import Field

from .models import Record


class ScheduledEvent(Record):
    occurs_at: datetime
    symbols: tuple[str, ...] = ()
    category: str = "scheduled_event"
    block_window_seconds: int = Field(default=1800, ge=0)
    source_reference: str = ""
    title: str = ""
    time_precision: str = "timestamp"
    blocks_entries: bool = True


class EventCalendarSnapshot(Record):
    published_at: datetime
    valid_until: datetime
    verified: bool
    source_reference: str = "operator-maintained calendar"
    events: tuple[ScheduledEvent, ...]
    coverage: dict[str, str] = Field(default_factory=dict)
    coverage_start: str | None = None
    coverage_end: str | None = None


class EventCalendarProvider(Protocol):
    def snapshot(self, now: datetime) -> EventCalendarSnapshot: ...


class FileEventCalendar:
    def __init__(self, path):
        self.path = Path(path)

    def snapshot(self, now):
        source = self.path.read_bytes()
        payload = json.loads(source)
        payload.setdefault("id", hashlib.sha256(source).hexdigest())
        payload.setdefault("created_at", payload["published_at"])
        for event in payload.get("events", []):
            if not event.get("id"):
                raise ValueError("calendar events require explicit source identity")
            event.setdefault("created_at", payload["published_at"])
        result = EventCalendarSnapshot.model_validate(payload)
        if not result.verified or not result.published_at <= now <= result.valid_until:
            raise ValueError("EVENT_CALENDAR_UNVERIFIED_OR_STALE")
        return result


class APIEventCalendar:
    """Combine Finnhub earnings with FRED release dates, retaining partial coverage."""

    def __init__(self, earnings=None, macro=None):
        self.earnings, self.macro = earnings, macro
        self._cached = None

    def close(self):
        if self.earnings is not None:
            self.earnings.client._session.close()

    def snapshot(self, now):
        if self._cached and self._cached.published_at <= now < self._cached.valid_until:
            return self._cached
        zone = ZoneInfo("America/New_York")
        start = now.astimezone(zone).date()
        end = start + timedelta(days=7)
        events, coverage = [], {}

        def event(day, key, title, category, reference, symbols=(), blocks=True):
            date = datetime.fromisoformat(day).date()
            if not start <= date <= end:
                raise ValueError("Calendar date outside requested range")
            # Noon is an internal anchor for the all-day window, never a reported release time.
            return ScheduledEvent(
                id=key,
                created_at=now,
                occurs_at=datetime.combine(date, time(12), zone),
                symbols=symbols,
                title=title,
                category=category,
                source_reference=reference,
                time_precision="date",
                blocks_entries=blocks,
                block_window_seconds=43200,
            )

        try:
            if self.earnings is None:
                raise ValueError("Missing Finnhub credentials")
            payload = self.earnings.get_earnings_calendar(from_date=str(start), to_date=str(end))
            rows = payload.get("earningsCalendar")
            if not isinstance(rows, list) or len(rows) >= 1500:
                raise ValueError("Malformed or potentially truncated earnings calendar")
            parsed = []
            for row in rows:
                symbol = row["symbol"]
                if not isinstance(symbol, str) or not symbol.strip():
                    raise ValueError("Missing earnings symbol")
                parsed.append(
                    event(
                        row["date"],
                        f"finnhub:{symbol}:{row['date']}",
                        f"{symbol} earnings ({row.get('hour') or 'time unspecified'})",
                        "earnings",
                        "https://finnhub.io/docs/api/earnings-calendar",
                        (symbol,),
                    )
                )
            events.extend(parsed)
            coverage["earnings"] = "AVAILABLE"
        except Exception as exc:
            coverage["earnings"] = f"UNAVAILABLE ({type(exc).__name__})"

        try:
            if self.macro is None:
                raise ValueError("Missing FRED credentials")
            rows = self.macro.get_release_dates(str(start), str(end))
            major = (
                "consumer price index",
                "producer price index",
                "employment situation",
                "gross domestic product",
                "personal income and outlays",
                "advance monthly sales for retail",
            )
            parsed = []
            for row in rows:
                title = row["release_name"]
                parsed.append(
                    event(
                        row["date"],
                        f"fred:{row['release_id']}:{row['date']}",
                        title,
                        "macro_release",
                        f"https://fred.stlouisfed.org/release?rid={int(row['release_id'])}",
                        blocks=any(term in title.lower() for term in major),
                    )
                )
            events.extend(parsed)
            coverage["macro"] = "AVAILABLE_DATE_ONLY"
        except Exception as exc:
            coverage["macro"] = f"UNAVAILABLE ({type(exc).__name__})"
        self._cached = EventCalendarSnapshot(
            created_at=now,
            published_at=now,
            valid_until=now + timedelta(minutes=15),
            verified=coverage.get("earnings") == "AVAILABLE"
            and coverage.get("macro") == "AVAILABLE_DATE_ONLY",
            source_reference="Finnhub earnings + FRED scheduled release dates; retrieved at published_at",
            coverage=coverage,
            coverage_start=str(start),
            coverage_end=str(end),
            events=tuple(sorted(events, key=lambda e: (e.occurs_at, e.id))),
        )
        return self._cached


def configured_event_calendar(config):
    if config.event_calendar_path:
        return FileEventCalendar(config.event_calendar_path)
    from src.data.finnhub_provider import FinnhubProvider
    from src.data.fred_provider import FREDProvider
    from src.settings import get_config

    settings = get_config()
    return APIEventCalendar(
        FinnhubProvider(settings.finnhub.api_key) if settings.finnhub.api_key else None,
        FREDProvider(settings.fred.api_key) if settings.fred.api_key else None,
    )
