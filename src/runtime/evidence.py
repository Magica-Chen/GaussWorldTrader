"""Shared evidence clocks, durable versioned news and immutable snapshot access."""

from __future__ import annotations

from datetime import timedelta, datetime, timezone
from decimal import Decimal
from .models import DataContext, DataProfile, MarketEvent, NewsEvent, ResearchSnapshot
from .store import digest


class EvidenceClock:
    def __init__(self, profile, boundary_buffer=2):
        self.profile = DataProfile(profile)
        self.boundary_buffer = boundary_buffer

    @property
    def delay_seconds(self):
        return 900 if self.profile == DataProfile.FREE_DELAYED else 0

    def signal_as_of(self, wall_time):
        return wall_time - timedelta(
            seconds=self.delay_seconds + (self.boundary_buffer if self.delay_seconds else 0)
        )

    safe_request_end = signal_as_of

    def eligible(self, event, wall_time, signal_as_of=None):
        cutoff = min(signal_as_of or self.signal_as_of(wall_time), self.signal_as_of(wall_time))
        effective = (
            event.effective_event_time if isinstance(event, MarketEvent) else event.updated_at
        )
        return (
            effective <= cutoff
            and event.received_at <= wall_time
            and event.available_at <= wall_time
            and (not isinstance(event, MarketEvent) or event.complete)
        )

    def additional_lag(self, event, wall_time):
        return max(0, (wall_time - event.effective_event_time).total_seconds() - self.delay_seconds)


class EvidenceService:
    def __init__(self, store, config, scope):
        self.store, self.config, self.scope = store, config, scope
        self.clock = EvidenceClock(config.data_profile, config.entitlement_boundary_buffer)

    def ingest_market(self, event):
        event = MarketEvent.model_validate(event)
        key = digest(
            [
                event.source,
                event.symbol,
                event.source_version,
                event.event_type,
                event.effective_event_time.isoformat(),
                event.price,
                event.open_price,
                event.high_price,
                event.low_price,
                event.volume,
                event.bid,
                event.ask,
                event.bid_size,
                event.ask_size,
                event.complete,
                event.quality_class,
            ]
        )
        existing = self.store.get("market_events", key)
        if existing:
            return key
        event = event.model_copy(update={"id": key})
        self.store.put("market_events", event, self.scope)
        return key

    def ingest_news(self, event):
        event = NewsEvent.model_validate(event)
        key = digest([event.provider, event.article_id, event.source_version, event.content_hash])
        if self.store.get("news_versions", key):
            return key
        event = event.model_copy(update={"id": key})
        with self.store.transaction():
            self.store.put("news_versions", event, self.scope)
            self.store.emit("news_received", event.model_dump(mode="json"), self.scope, key)
        return key

    def market(self, now, symbol=None, event_type=None):
        events = [
            MarketEvent.model_validate(row)
            for row in self.store.list("market_events", self.scope, 100000)
        ]
        result = [
            e
            for e in events
            if self.clock.eligible(e, now)
            and (symbol is None or e.symbol == symbol)
            and (event_type is None or e.event_type == event_type)
        ]
        # Revisions are selected at the decision's availability time, never retroactively.
        latest = {}
        for event in sorted(result, key=lambda e: (e.available_at, e.source_version)):
            latest[(event.symbol, event.event_type, event.effective_event_time)] = event
        return sorted(latest.values(), key=lambda e: e.effective_event_time)

    def news(self, now, *, safety=False):
        events = [
            NewsEvent.model_validate(row)
            for row in self.store.list("news_versions", self.scope, 100000)
        ]
        eligible = [
            e
            for e in events
            if (e.available_at <= now and e.received_at <= now)
            and (safety or self.clock.eligible(e, now))
        ]
        latest = {}
        for event in sorted(eligible, key=lambda e: (e.updated_at, e.available_at)):
            latest[(event.provider, event.article_id)] = event
        return list(latest.values())

    def blockers(self, symbol, now):
        return tuple(
            e.id
            for e in self.news(now, safety=True)
            if e.blocking and (not e.symbols or symbol in e.symbols)
        )

    def context(self, now, symbols=()):
        rows = self.market(now)
        watermarks = {}
        for event in rows:
            watermarks[event.symbol] = max(
                watermarks.get(event.symbol, event.effective_event_time), event.effective_event_time
            )
        missing = [s for s in symbols if s not in watermarks]
        stale = [
            s
            for s in symbols
            if s in watermarks
            and (self.clock.signal_as_of(now) - watermarks[s]).total_seconds()
            > self.config.maximum_additional_lag_seconds
        ]
        blockers = tuple(
            [f"MISSING_SYMBOL:{s}" for s in missing] + [f"ADDITIONAL_LAG:{s}" for s in stale]
        )
        return DataContext(
            created_at=now,
            data_profile=self.config.data_profile,
            data_policy_version=self.config.data_policy_version,
            wall_time=now,
            signal_as_of=self.clock.signal_as_of(now),
            configured_delay_seconds=self.clock.delay_seconds,
            watermarks=tuple(sorted(watermarks.items())),
            complete=not blockers,
            execution_ready=False,
            blockers=blockers,
        )

    def freeze(self, session_id, account, now, approvals=()):
        context = self.context(now, self.config.symbols)
        evidence_ids = tuple(e.id for e in self.market(now))
        news_ids = tuple(e.id for e in self.news(now))
        manifest = [
            evidence_ids,
            news_ids,
            account.id,
            context.model_dump(mode="json"),
            tuple(a.id for a in approvals),
        ]
        from .models import TradingSession

        sessions = tuple(
            TradingSession.model_validate(row) for row in self.store.list("sessions", self.scope)
        )
        manifest.extend(
            [
                self.config.signal_timeframe,
                [session.model_dump(mode="json") for session in sessions],
            ]
        )
        snapshot = ResearchSnapshot(
            created_at=now,
            target_session_id=session_id,
            account_profile_id=account.id,
            signal_timeframe=self.config.signal_timeframe,
            trading_sessions=sessions,
            data_context=context,
            evidence_ids=evidence_ids,
            news_ids=news_ids,
            manifest_hash=digest(manifest),
            strategy_approval_ids=tuple(a.id for a in approvals),
        )
        self.store.put("snapshots", snapshot, self.scope)
        return snapshot

    def current_quote(self, symbol, now, feed, max_age=5):
        rows = [
            MarketEvent.model_validate(row)
            for row in self.store.list("market_events", self.scope, 100000)
        ]
        quotes = [
            q
            for q in rows
            if q.symbol == symbol
            and q.event_type == "quote"
            and q.feed == feed
            and q.quality_class == "genuine"
            and q.complete
            and q.bid is not None
            and q.bid > 0
            and q.ask is not None
            and q.ask >= q.bid
            and q.available_at <= now
            and q.received_at <= now
            and 0 <= (now - q.effective_event_time).total_seconds() <= max_age
        ]
        return max(quotes, key=lambda q: q.effective_event_time) if quotes else None


class SnapshotDataReader:
    """Copies a manifest's exact bytes once; exposes no network-capable object."""

    def __init__(self, store, snapshot):
        self.snapshot = ResearchSnapshot.model_validate(snapshot)
        expected = digest(
            [
                self.snapshot.evidence_ids,
                self.snapshot.news_ids,
                self.snapshot.account_profile_id,
                self.snapshot.data_context.model_dump(mode="json"),
                self.snapshot.strategy_approval_ids,
                self.snapshot.signal_timeframe,
                [session.model_dump(mode="json") for session in self.snapshot.trading_sessions],
            ]
        )
        if expected != self.snapshot.manifest_hash:
            raise ValueError("snapshot manifest hash mismatch")
        self._market = tuple(
            MarketEvent.model_validate(store.get("market_events", key))
            for key in self.snapshot.evidence_ids
        )
        self._news = tuple(
            NewsEvent.model_validate(store.get("news_versions", key))
            for key in self.snapshot.news_ids
        )
        clock = EvidenceClock(self.snapshot.data_context.data_profile, 0)
        for event in self._market + self._news:
            if not clock.eligible(
                event, self.snapshot.data_context.wall_time, self.snapshot.data_context.signal_as_of
            ):
                raise ValueError("snapshot contains ineligible future evidence")

    def market(self, symbol=None):
        return tuple(e for e in self._market if symbol is None or e.symbol == symbol)

    def news(self, symbol=None):
        return tuple(
            e for e in self._news if symbol is None or not e.symbols or symbol in e.symbols
        )

    def fundamentals(self, symbol):
        return {
            "available": False,
            "reason": "No point-in-time fundamental dataset in this snapshot.",
        }

    def bars(self, symbol):
        import pandas as pd

        observations = completed_bars(
            self.market(symbol), self.snapshot.signal_timeframe, self.snapshot.trading_sessions
        )
        rows = [
            {
                "timestamp": e.effective_event_time,
                "open": float(e.open_price),
                "high": float(e.high_price),
                "low": float(e.low_price),
                "close": float(e.price),
                "volume": float(e.volume),
            }
            for e in observations
            if e.event_type == "bar"
            and e.price
            and e.open_price is not None
            and e.high_price is not None
            and e.low_price is not None
        ]
        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()


def evaluate_rules(
    rules, observations, now, *, blocking=False, valid_from=None, expires_at=None, sessions=()
):
    for rule in rules:
        if rule.rule_type == "no_blocking_event":
            if blocking:
                return False
        elif rule.rule_type == "time_in_window":
            if valid_from is None or expires_at is None or not valid_from <= now < expires_at:
                return False
        elif rule.rule_type == "spread_within_limit":
            quotes = [e for e in observations if e.bid is not None and e.ask is not None]
            if not quotes or rule.value is None or quotes[-1].ask - quotes[-1].bid > rule.value:
                return False
        else:
            rule_observations = (
                completed_bars(observations, rule.timeframe, sessions)
                if rule.rule_type == "completed_bar_condition"
                else observations
            )
            values = [
                e.price
                for e in rule_observations
                if e.price is not None
                and (
                    rule.rule_type != "completed_bar_condition"
                    or (e.event_type == "bar" and e.complete)
                )
            ]
            if not values or rule.value is None:
                return False
            last = values[-1]
            conditions = {
                "above": last > rule.value,
                "below": last < rule.value,
                "lte": last <= rule.value,
                "crosses_above": len(values) > 1 and values[-2] <= rule.value < last,
                "crosses_below": len(values) > 1 and values[-2] >= rule.value > last,
            }
            if not conditions[rule.operator]:
                return False
    return bool(rules)


def completed_bars(events, timeframe="1Min", sessions=()):
    """Aggregate only full source intervals; exchange calendars anchor regular-session bars.

    Every required minute must be present. A final partial hour may complete at a
    verified early close; daily bars require the full actual regular session.
    """
    minutes = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60}
    bars = [e for e in events if e.event_type == "bar" and e.complete]
    if timeframe == "1Min":
        return sorted(
            (e for e in bars if e.timeframe == "1Min"), key=lambda e: e.effective_event_time
        )
    groups = {}
    for event in bars:
        if event.timeframe == timeframe:
            groups[(event.symbol, event.effective_event_time)] = ("native", event)
            continue
        if event.timeframe != "1Min":
            continue
        session = next(
            (s for s in sessions if s.open < event.effective_event_time <= s.close), None
        )
        if timeframe == "1Day":
            if session is None:
                continue
            start, end = session.open, session.close
        else:
            width = timedelta(minutes=minutes[timeframe])
            if session:
                elapsed = (event.effective_event_time - session.open).total_seconds() - 0.000001
                start = session.open + int(elapsed // width.total_seconds()) * width
                end = min(start + width, session.close)
            else:
                timestamp = event.effective_event_time.timestamp() - 0.000001
                start = datetime.fromtimestamp(
                    int(timestamp // width.total_seconds()) * width.total_seconds(), timezone.utc
                )
                end = start + width
        key = (event.symbol, end)
        if key not in groups:
            groups[key] = (start, [])
        if groups[key][0] != "native":
            groups[key][1].append(event)
    result = []
    for (symbol, end), (start, items) in groups.items():
        if start == "native":
            result.append(items)
            continue
        by_end = {event.effective_event_time: event for event in items}
        expected = int((end - start).total_seconds() // 60)
        required = [start + timedelta(minutes=i) for i in range(1, expected + 1)]
        if not expected or any(stamp not in by_end for stamp in required):
            continue
        ordered = [by_end[stamp] for stamp in required]
        if any(
            e.price is None or e.open_price is None or e.high_price is None or e.low_price is None
            for e in ordered
        ):
            continue
        first, last = ordered[0], ordered[-1]
        result.append(
            MarketEvent(
                id=digest([e.id for e in ordered]),
                symbol=symbol,
                source=first.source,
                source_version=digest([e.source_version for e in ordered]),
                event_type="bar",
                timeframe=timeframe,
                effective_event_time=end,
                interval_start=start,
                received_at=max(e.received_at for e in ordered),
                available_at=max(e.available_at for e in ordered),
                feed=first.feed,
                quality_class=first.quality_class,
                price=last.price,
                open_price=first.open_price,
                high_price=max(e.high_price for e in ordered),
                low_price=min(e.low_price for e in ordered),
                volume=sum((e.volume for e in ordered), Decimal("0")),
            )
        )
    return sorted(result, key=lambda e: e.effective_event_time)
