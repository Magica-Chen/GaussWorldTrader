"""Calendar-backed sessions. No weekday or fixed-close authorization."""

from datetime import timedelta
from .models import utc_now


class SessionCalendar:
    def __init__(self, sessions=(), provider=None):
        self.sessions = {session.id: session for session in sessions}
        self.provider = provider

    def refresh(self, now=None):
        now = now or utc_now()
        if self.provider:
            sessions = self.provider.calendar(
                now.date() - timedelta(days=10), now.date() + timedelta(days=14)
            )
            self.sessions.update({session.id: session for session in sessions})
        if not self.sessions:
            raise RuntimeError("CALENDAR_UNAVAILABLE")

    def current_or_next(self, now):
        available = sorted(
            (s for s in self.sessions.values() if s.close > now), key=lambda s: s.open
        )
        if not available:
            raise RuntimeError("CALENDAR_UNAVAILABLE_OR_STALE")
        return available[0]

    def previous(self, now):
        available = sorted(
            (s for s in self.sessions.values() if s.close <= now), key=lambda s: s.close
        )
        return available[-1] if available else None

    def get(self, session_id):
        if session_id not in self.sessions:
            raise RuntimeError("CALENDAR_SESSION_UNKNOWN")
        return self.sessions[session_id]

    def permits_entry(self, now, policy):
        session = self.current_or_next(now)
        schedule = session.schedule(policy)
        return schedule["entry_start"] <= now < schedule["entry_cutoff"]

    def is_open(self, now=None):
        now = now or utc_now()
        session = self.current_or_next(now)
        return session.open <= now < session.close
