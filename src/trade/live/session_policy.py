"""Shared verified exchange sessions for legacy stock/option observation loops."""

from datetime import timezone
from time import monotonic
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

from src.runtime.calendar import SessionCalendar
from src.runtime.models import TradingSession, utc_now


class BrokerCalendarProvider:
    def __init__(self, client):
        self.client = client

    def calendar(self, start, end):
        records = self.client.get_calendar(GetCalendarRequest(start=start, end=end))

        def utc(value):
            if value.tzinfo is None:
                value = value.replace(tzinfo=ZoneInfo("America/New_York"))
            return value.astimezone(timezone.utc)

        return tuple(
            TradingSession(
                id=f"US_EQUITIES:{item.date.isoformat()}",
                session_date=item.date.isoformat(),
                calendar_version="alpaca-calendar",
                open=utc(item.open),
                close=utc(item.close),
            )
            for item in records
        )


class SessionAwareTrading:
    """Calendar failure blocks entries and schedules a bounded retry."""

    def _session_calendar(self):
        if not hasattr(self, "_verified_calendar"):
            self._verified_calendar = SessionCalendar(
                provider=BrokerCalendarProvider(self.engine.api)
            )
            self._calendar_refreshed = 0.0
        if monotonic() - self._calendar_refreshed >= 300:
            self._verified_calendar.refresh(utc_now())
            self._calendar_refreshed = monotonic()
        return self._verified_calendar

    def _is_market_open(self):
        try:
            return self._session_calendar().is_open(utc_now())
        except Exception:
            self.logger.exception("Verified calendar unavailable; new entries paused")
            return False

    def is_market_open(self):
        return self._is_market_open()

    def _seconds_until_market_open(self):
        try:
            now = utc_now()
            session = self._session_calendar().current_or_next(now)
            return max(1.0, (session.open - now).total_seconds())
        except Exception:
            self.logger.exception("Calendar refresh failed; retrying in 60 seconds")
            return 60.0

    def seconds_until_market_open(self):
        return self._seconds_until_market_open()

    def _get_signal_interval_seconds(self):
        if not self._is_market_open():
            return self._seconds_until_market_open()
        now = utc_now()
        session = self._session_calendar().current_or_next(now)
        return max(
            1.0, min(self._seconds_until_next_interval(), (session.close - now).total_seconds())
        )
