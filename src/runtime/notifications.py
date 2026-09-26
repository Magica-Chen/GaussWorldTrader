"""Durable, account-scoped notifications from committed trading records.

Capture only writes local queue state. A separate worker connection delivers outside
SQLite transactions. Notification state never changes the trading outbox watermark.
Transport delivery is at least once: an accepted send followed by a process crash
can be retried because SMTP/webhooks do not provide an idempotency receipt API.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from .models import utc_now
from .store import Store, digest

EVENT_KINDS = (
    "session_reviews",
    "research_reports",
    "plan_events",
    "risk_decisions",
    "decisions",
    "fills",
    "incidents",
    "ai_decisions",
)


def configured_providers():
    """Only explicitly enabled and credentialed existing channels are eligible."""
    from src.notify.notification_service import (
        EmailNotificationProvider,
        NotificationService,
        SlackNotificationProvider,
    )

    names = {EmailNotificationProvider: "email", SlackNotificationProvider: "slack"}
    return {names[type(provider)]: provider for provider in NotificationService().providers}


class RuntimeNotifications:
    def __init__(self, database_path, scope, *, channels=None, providers=None, clock=utc_now):
        self.database_path, self.scope, self.clock = str(database_path), scope, clock
        self.providers = dict(configured_providers() if providers is None else providers)
        self.requested_channels = (
            tuple(dict.fromkeys(channels)) if channels is not None else tuple(self.providers)
        )
        if channels is not None:
            self.providers = {
                name: provider for name, provider in self.providers.items() if name in channels
            }
        self.store = Store(database_path)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._closed = False
        self.last_error = None
        # Each channel starts at its own activation point. Enabling email later must
        # not replay months of Slack history; previously enabled channels resume.
        with self.store.transaction():
            sequence = self.store.db.execute(
                "SELECT coalesce(max(sequence),0) FROM records WHERE scope=?", (scope,)
            ).fetchone()[0]
            for channel in self.providers:
                cursor_id = self._cursor_id(channel)
                if self.store.projection("notification_cursor", cursor_id) is None:
                    self.store.project(
                        "notification_cursor", cursor_id, {"sequence": sequence}, scope
                    )
                    for order in self.store.projected("orders", scope):
                        if self._acknowledged(order):
                            self._mark_submission(channel, order)

    def _cursor_id(self, channel):
        return digest([self.scope, channel])

    @staticmethod
    def _acknowledged(order):
        return bool(order.get("broker_id")) and order.get("state") not in {
            "SHADOW",
            "SUBMITTING",
            "UNKNOWN",
        }

    def _mark_submission(self, channel, order):
        key = digest([self.scope, channel, "submitted", order["id"]])
        self.store.project("notification_submissions", key, {"order_id": order["id"]}, self.scope)
        return key

    def _order_details(self, order):
        intent = self.store.get("order_intents", order.get("intent_id") or order.get("id")) or {}
        legs = intent.get("legs", ())
        return {
            "symbols": ", ".join(leg["instrument"]["symbol"] for leg in legs),
            "side": ", ".join(leg.get("side", "") for leg in legs),
            "purpose": order.get("purpose", intent.get("purpose")),
            "quantity": order.get("quantity"),
            "order_type": intent.get("order_type", "limit"),
            "limit_price": order.get("limit_price") or intent.get("limit_price"),
            "reference_price_estimate": intent.get("reference_price"),
            "plan_id": order.get("plan_id", intent.get("plan_id")),
        }

    def _event(self, kind, body):
        """Return a small, explicit transport payload; never forward arbitrary records."""
        review = body.get("ai_review") or body.get("annotation") or {}
        for _ in range(3):
            if not isinstance(review, dict) or review.get("summary"):
                break
            review = review.get("annotation") or review.get("result") or {}
        summary = review.get("summary") if isinstance(review, dict) else None
        event = None
        if kind == "session_reviews" and body.get("outcome") in {"CANDIDATES", "NO_TRADE"}:
            event = {
                "role": "PostGauss",
                "outcome": body["outcome"],
                "session_id": body.get("session_id"),
                "candidate_count": len(body.get("candidates", ())),
                "summary": summary,
            }
        elif kind == "research_reports":
            event = {
                "role": "CloseGauss",
                "outcome": body.get("outcome"),
                "session_id": body.get("session_id"),
                "reason": body.get("reason"),
                "summary": summary,
            }
        elif kind == "ai_decisions":
            event = {
                "role": body.get("role"),
                "outcome": body.get("outcome"),
                "plan_id": body.get("plan_id"),
                "symbols": body.get("symbol"),
                "session_id": body.get("session_id"),
                "summary": body.get("summary"),
            }
        elif kind == "plan_events" and body.get("state") != "RESEARCH_COMPLETE":
            plan = self.store.get("plans", body.get("plan_id")) or {}
            instrument = (plan.get("alternative") or {}).get("instrument") or {}
            event = {
                "role": "CloseGauss" if body.get("state") == "PENDING_VALIDATION" else "PreGauss",
                "plan_id": body.get("plan_id"),
                "symbols": instrument.get("symbol"),
                "outcome": body.get("state"),
                "reasons": body.get("reasons"),
            }
        elif kind in {"risk_decisions", "decisions"}:
            intent = self.store.get("order_intents", body.get("intent_id")) or {}
            event = {
                **self._order_details(intent),
                "role": "LiveGauss",
                "outcome": (
                    ("APPROVED" if body["approved"] else "REJECTED")
                    if "approved" in body
                    else body.get("outcome")
                ),
                "plan_id": body.get("plan_id", intent.get("plan_id")),
                "reason": body.get("reason"),
                "reasons": body.get("reasons"),
                "quantity": body.get("approved_quantity"),
            }
        elif kind == "fills":
            order = self.store.projection("orders", body.get("local_order_id")) or {}
            filled = Decimal(str(body.get("cumulative_quantity", "0")))
            quantity = Decimal(str(order.get("quantity", "0")))
            label = "FILLED" if quantity > 0 and filled >= quantity else "PARTIAL_FILL"
            return label, {
                **self._order_details(order),
                "order_id": body.get("order_id"),
                "fill_quantity": body.get("quantity"),
                "cumulative_quantity": str(filled),
                "fill_price": body.get("price"),
                "filled_at": body.get("broker_filled_at"),
            }
        elif kind == "incidents":
            return "INCIDENT", {
                "reason": body.get("reason"),
                "role": body.get("role"),
                "symbols": body.get("symbol"),
                "group_id": body.get("group_id"),
                "summary": (
                    "Codex allowance threshold reached. Analysis and trading execution "
                    "are paused; market and account monitoring continue."
                    if body.get("reason") == "CODEX_QUOTA_MONITOR_ONLY"
                    else None
                ),
                "remaining_percent": body.get("remaining_percent"),
                "resets_at": body.get("resets_at"),
            }
        if event is not None:
            return "DECISION", event
        return None

    def _enqueue(self, channel, event_id, label, details, now, *, sequence=0):
        key = digest([self.scope, channel, event_id])
        if self.store.projection("notification_deliveries", key):
            return
        if label == "INCIDENT" or event_id.startswith("decisions:"):
            throttle_id = digest([self.scope, channel, label, details])
            prior = self.store.projection("notification_throttle", throttle_id)
            if prior and now - datetime.fromisoformat(prior["created_at"]) < timedelta(minutes=15):
                return
            self.store.project(
                "notification_throttle", throttle_id, {"created_at": now.isoformat()}, self.scope
            )
        symbol = details.get("symbols") or details.get("role") or "session"
        subject = f"[GaussWorldTrader] {label}: {symbol}".replace("\r", " ").replace("\n", " ")
        lines = [f"Environment: {self.scope.split(':', 1)[0]}", f"Event: {event_id}"]
        for name, value in details.items():
            if value is not None and value != "" and value != []:
                formatted = (
                    ", ".join(map(str, value)) if isinstance(value, list | tuple) else str(value)
                )
                lines.append(f"{name.replace('_', ' ').capitalize()}: {formatted}")
        self.store.project(
            "notification_deliveries",
            key,
            {
                "id": key,
                "channel": channel,
                "event_id": event_id,
                "event": label,
                "source_sequence": sequence,
                "priority": {"SUBMITTED": 1, "PARTIAL_FILL": 2, "FILLED": 2}.get(label, 0),
                "subject": subject[:200],
                "message": "\n".join(lines)[:3800],
                "state": "PENDING",
                "attempts": 0,
                "created_at": now.isoformat(),
                "next_attempt_at": now.isoformat(),
            },
            self.scope,
        )

    def capture(self, *, limit=1000):
        """Persist new ledger events atomically with each channel's durable cursor."""
        if self._closed or not self.providers:
            return 0
        captured, now = 0, self.clock()
        with self.store.transaction():
            for channel in self.providers:
                cursor_id = self._cursor_id(channel)
                cursor = self.store.projection("notification_cursor", cursor_id)
                placeholders = ",".join("?" for _ in EVENT_KINDS)
                rows = self.store.db.execute(
                    "SELECT sequence,kind,id,body FROM records WHERE scope=? AND sequence>? "
                    f"AND kind IN ({placeholders}) ORDER BY sequence LIMIT ?",
                    (self.scope, cursor["sequence"], *EVENT_KINDS, limit),
                ).fetchall()
                # Reconciliation can persist a fill and acknowledgement together.
                # Queue confirmed submission before the associated fill.
                for order in self.store.projected("orders", self.scope):
                    key = digest([self.scope, channel, "submitted", order["id"]])
                    if self._acknowledged(order) and not self.store.projection(
                        "notification_submissions", key
                    ):
                        self._enqueue(
                            channel,
                            "submitted:" + order["id"],
                            "SUBMITTED",
                            {
                                **self._order_details(order),
                                "order_id": order["broker_id"],
                                "status": order.get("raw_status", order.get("state")),
                            },
                            now,
                        )
                        self._mark_submission(channel, order)
                        captured += 1
                for row in rows:
                    body = json.loads(row["body"])
                    event = self._event(row["kind"], body)
                    if event:
                        self._enqueue(
                            channel,
                            row["kind"] + ":" + row["id"],
                            *event,
                            now,
                            sequence=row["sequence"],
                        )
                        captured += 1
                if rows:
                    self.store.project(
                        "notification_cursor",
                        cursor_id,
                        {"sequence": rows[-1]["sequence"]},
                        self.scope,
                    )
        self._wake.set()
        return captured

    def dispatch_once(self, *, limit=20, _store=None):
        """Send due items without holding a database transaction during transport I/O."""
        store = _store or Store(self.database_path)
        delivered = 0
        try:
            for _ in range(limit):
                if self._stop.is_set():
                    break
                now, token, pending = self.clock(), uuid4().hex, None
                with store.transaction():
                    rows = sorted(
                        store.projected("notification_deliveries", self.scope),
                        key=lambda row: (
                            row["created_at"],
                            row.get("priority", 0),
                            row.get("source_sequence", 0),
                            row["id"],
                        ),
                    )
                    for row in rows:
                        if row["channel"] not in self.providers or row["state"] == "DELIVERED":
                            continue
                        due = (
                            row.get("lease_expires_at")
                            if row["state"] == "SENDING"
                            else row["next_attempt_at"]
                        )
                        if datetime.fromisoformat(due) > now:
                            continue
                        pending = {k: v for k, v in row.items() if k != "_revision"}
                        pending.update(
                            state="SENDING",
                            token=token,
                            attempts=row["attempts"] + 1,
                            lease_expires_at=(now + timedelta(minutes=5)).isoformat(),
                        )
                        store.project("notification_deliveries", row["id"], pending, self.scope)
                        break
                if pending is None:
                    break
                error = None
                try:
                    success = (
                        self.providers[pending["channel"]].send(
                            pending["subject"], pending["message"]
                        )
                        is True
                    )
                    if not success:
                        error = "TRANSPORT_NOT_ACCEPTED"
                except Exception as exc:
                    success, error = False, type(exc).__name__
                finished = self.clock()
                with store.transaction():
                    current = store.projection("notification_deliveries", pending["id"])
                    if current.get("token") != token:
                        continue
                    pending.pop("token", None)
                    pending.pop("lease_expires_at", None)
                    if success:
                        pending.update(
                            state="DELIVERED", delivered_at=finished.isoformat(), last_error=None
                        )
                        store.put(
                            "notification_receipts",
                            {
                                "id": pending["id"],
                                "channel": pending["channel"],
                                "event_id": pending["event_id"],
                                "created_at": finished.isoformat(),
                            },
                            self.scope,
                        )
                        delivered += 1
                    else:
                        delay = min(3600, 5 * 2 ** min(pending["attempts"] - 1, 10))
                        pending.update(
                            state="RETRY",
                            last_error=error,
                            next_attempt_at=(finished + timedelta(seconds=delay)).isoformat(),
                        )
                    store.project("notification_deliveries", pending["id"], pending, self.scope)
            return delivered
        finally:
            if _store is None:
                store.close()

    def start(self):
        if self._closed or not self.providers or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gauss-notifications", daemon=True)
        self._thread.start()

    def _run(self):
        store = Store(self.database_path)
        try:
            while not self._stop.is_set():
                self._wake.clear()
                try:
                    self.dispatch_once(_store=store)
                    self.last_error = None
                except Exception as exc:
                    self.last_error = type(exc).__name__
                self._wake.wait(5)
        finally:
            store.close()

    def status(self):
        rows = self.store.projected("notification_deliveries", self.scope)
        active = [row for row in rows if row["channel"] in self.providers]
        return {
            "channels": sorted(self.providers),
            "missing_channels": sorted(set(self.requested_channels) - self.providers.keys()),
            "enabled": bool(self.providers),
            "pending": sum(row["state"] != "DELIVERED" for row in active),
            "retrying": sum(row["state"] == "RETRY" for row in active),
            "delivered": sum(row["state"] == "DELIVERED" for row in active),
            "worker_error": self.last_error,
        }

    def close(self, timeout=2):
        """Stop between sends; unfinished deliveries remain available after restart."""
        if not self._closed:
            self._stop.set()
            self._wake.set()
            if self._thread:
                self._thread.join(timeout)
            self.store.close()
            self._closed = True
        return self._thread is None or not self._thread.is_alive()
