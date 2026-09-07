"""SQLite ledger: immutable records, atomic projections/outbox and account leases."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from .models import utc_now

DEFAULT_DATABASE_PATH = "results/gauss/session.sqlite3"
TERMINAL_ORDERS = {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
PLAN_TRANSITIONS = {
    "DRAFT": {"RESEARCH_COMPLETE", "REJECTED"},
    "RESEARCH_COMPLETE": {"PENDING_VALIDATION", "REJECTED"},
    "PENDING_VALIDATION": {"ELIGIBLE", "DEFERRED", "REJECTED", "INVALIDATED", "ENTRY_EXPIRED"},
    "ELIGIBLE": {"DEFERRED", "REVIEW_REQUIRED", "INVALIDATED", "SUPERSEDED", "ENTRY_EXPIRED"},
    "DEFERRED": {"ELIGIBLE", "REJECTED", "INVALIDATED", "ENTRY_EXPIRED", "REVIEW_REQUIRED"},
    "REVIEW_REQUIRED": {"ELIGIBLE", "DEFERRED", "REJECTED", "INVALIDATED", "ENTRY_EXPIRED"},
}


def canonical(value):
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class LeaseConflict(RuntimeError):
    """Another owner or an unexpired recovery lease prevents startup."""


class Store:
    def __init__(self, path=DEFAULT_DATABASE_PATH, read_only=False):
        self.path = str(path)
        if not read_only:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        database = f"file:{Path(path).absolute()}?mode=ro" if read_only else path
        self.db = sqlite3.connect(
            database, uri=read_only, timeout=10, isolation_level=None, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._host_locks = {}
        if read_only:
            self.db.execute("PRAGMA query_only=ON")
            return
        has_version = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        if has_version and (
            self.db.execute("SELECT coalesce(max(version),0) FROM schema_migrations").fetchone()[0]
            > 1
        ):
            self.db.close()
            raise RuntimeError("database schema is newer than this deployment")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, id TEXT NOT NULL,
          scope TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL, UNIQUE(kind,id));
        CREATE INDEX IF NOT EXISTS records_scope ON records(scope,kind,sequence);
        CREATE TABLE IF NOT EXISTS projections(
          kind TEXT NOT NULL, id TEXT NOT NULL, scope TEXT NOT NULL, revision INTEGER NOT NULL,
          body TEXT NOT NULL, PRIMARY KEY(kind,id));
        CREATE INDEX IF NOT EXISTS projections_scope ON projections(scope,kind);
        CREATE TABLE IF NOT EXISTS outbox(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
          topic TEXT NOT NULL, scope TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS consumer_offsets(consumer TEXT PRIMARY KEY, sequence INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS leases(scope TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
          state TEXT NOT NULL, owner TEXT, expires_at TEXT, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reservations(intent_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
          session_id TEXT NOT NULL, capital TEXT NOT NULL, loss TEXT NOT NULL,
          symbol TEXT NOT NULL, state TEXT NOT NULL, first_fill INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS reservations_scope ON reservations(scope,session_id,state);
        """)
        existing = self.db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
        if existing and existing > 1:
            raise RuntimeError("database schema is newer than this deployment")
        self.db.execute(
            "INSERT OR IGNORE INTO schema_migrations VALUES(1,?)", (utc_now().isoformat(),)
        )
        self.db.execute("PRAGMA user_version=1")

    @contextmanager
    def transaction(self):
        with self.lock:
            nested = self.db.in_transaction
            if not nested:
                self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self
                if not nested:
                    self.db.execute("COMMIT")
            except BaseException:
                if not nested:
                    self.db.execute("ROLLBACK")
                raise

    def put(self, kind, record, scope="", record_id=None):
        body = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        record_id = record_id or body.get("id") or uuid4().hex
        stamp = body.get("created_at") or utc_now().isoformat()
        payload = canonical(body)
        with self.transaction():
            previous = self.db.execute(
                "SELECT body FROM records WHERE kind=? AND id=?", (kind, record_id)
            ).fetchone()
            if previous:
                if previous["body"] != payload:
                    raise ValueError(
                        f"immutable {kind}/{record_id} already exists with different content"
                    )
                return record_id
            self.db.execute(
                "INSERT INTO records(kind,id,scope,created_at,body) VALUES(?,?,?,?,?)",
                (kind, record_id, scope, stamp, payload),
            )
        return record_id

    def get(self, kind, record_id):
        with self.lock:
            row = self.db.execute(
                "SELECT body FROM records WHERE kind=? AND id=?", (kind, record_id)
            ).fetchone()
        return json.loads(row["body"]) if row else None

    def list(self, kind, scope=None, limit=1000):
        query = "SELECT body FROM records WHERE kind=?"
        args = [kind]
        if scope is not None:
            query += " AND scope=?"
            args.append(scope)
        query += " ORDER BY sequence DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [json.loads(row["body"]) for row in self.db.execute(query, args)]

    def project(self, kind, record_id, body, scope="", expected_revision=None):
        with self.transaction():
            old = self.db.execute(
                "SELECT revision FROM projections WHERE kind=? AND id=?", (kind, record_id)
            ).fetchone()
            version = old["revision"] if old else 0
            if expected_revision is not None and version != expected_revision:
                raise ValueError("projection revision conflict")
            self.db.execute(
                "INSERT INTO projections VALUES(?,?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET revision=excluded.revision, body=excluded.body, scope=excluded.scope",
                (kind, record_id, scope, version + 1, canonical(body)),
            )
        return version + 1

    def projection(self, kind, record_id):
        with self.lock:
            row = self.db.execute(
                "SELECT body,revision FROM projections WHERE kind=? AND id=?", (kind, record_id)
            ).fetchone()
        if not row:
            return None
        return {**json.loads(row["body"]), "_revision": row["revision"]}

    def projected(self, kind, scope=None):
        query = "SELECT body,revision FROM projections WHERE kind=?"
        args = [kind]
        if scope is not None:
            query += " AND scope=?"
            args.append(scope)
        with self.lock:
            return [
                {**json.loads(row["body"]), "_revision": row["revision"]}
                for row in self.db.execute(query, args)
            ]

    def emit(self, topic, body, scope="", event_id=None):
        event_id = event_id or uuid4().hex
        with self.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO outbox(event_id,topic,scope,body,created_at) VALUES(?,?,?,?,?)",
                (event_id, topic, scope, canonical(body), utc_now().isoformat()),
            )
        return event_id

    def consume(self, consumer, handler, *, scope=None, limit=100):
        """Apply each local consumer effect and offset in the same transaction."""
        count = 0
        with self.transaction():
            row = self.db.execute(
                "SELECT sequence FROM consumer_offsets WHERE consumer=?", (consumer,)
            ).fetchone()
            offset = row[0] if row else 0
            rows = self.db.execute(
                "SELECT * FROM outbox WHERE sequence>? ORDER BY sequence LIMIT ?", (offset, limit)
            ).fetchall()
            for row in rows:
                if scope is None or row["scope"] == scope:
                    handler(row["topic"], json.loads(row["body"]))
                self.db.execute(
                    "INSERT INTO consumer_offsets VALUES(?,?) ON CONFLICT(consumer) DO UPDATE SET sequence=excluded.sequence",
                    (consumer, row["sequence"]),
                )
                count += 1
        return count

    def watermark(self, scope):
        with self.lock:
            return self.db.execute(
                "SELECT coalesce(max(sequence),0) FROM outbox WHERE scope=?", (scope,)
            ).fetchone()[0]

    def transition_plan(self, plan_id, state, reasons=(), expected_revision=None, scope=""):
        with self.transaction():
            prior = self.projection("plan_state", plan_id)
            current = prior["state"] if prior else "DRAFT"
            if current == state:
                return
            if state not in PLAN_TRANSITIONS.get(current, set()):
                raise ValueError(f"invalid plan transition {current} -> {state}")
            event = {
                "id": uuid4().hex,
                "plan_id": plan_id,
                "from": current,
                "state": state,
                "reasons": list(reasons),
                "created_at": utc_now().isoformat(),
            }
            self.put("plan_events", event, scope)
            self.project("plan_state", plan_id, event, scope, expected_revision)
            self.emit("plan_changed", event, scope, event["id"])

    def acquire_lease(self, scope, owner, now=None, seconds=60):
        now = now or utc_now()
        acquired_host_lock = False
        if scope not in self._host_locks:
            path = _host_lock_path(scope)
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            stream = os.fdopen(fd, "r+")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                raise LeaseConflict(
                    "another process owns this account/environment on this host"
                ) from None
            stream.seek(0)
            stream.truncate()
            stream.write(
                canonical(
                    {"scope": scope, "owner": owner, "database_path": self.path, "pid": os.getpid()}
                )
            )
            stream.flush()
            self._host_locks[scope] = stream
            acquired_host_lock = True
        try:
            with self.transaction():
                row = self.db.execute("SELECT * FROM leases WHERE scope=?", (scope,)).fetchone()
                if row and row["owner"] != owner and row["expires_at"] > now.isoformat():
                    raise LeaseConflict(
                        "another session runtime owns this account/environment; "
                        f"database lease expires at {row['expires_at']}. "
                        "If the previous process has stopped, retry after that time."
                    )
                self.db.execute(
                    "INSERT INTO leases VALUES(?,?,?) ON CONFLICT(scope) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at",
                    (scope, owner, (now + timedelta(seconds=seconds)).isoformat()),
                )
        except Exception:
            if acquired_host_lock:
                stream = self._host_locks.pop(scope)
                fcntl.flock(stream, fcntl.LOCK_UN)
                stream.close()
            raise

    def release_lease(self, scope, owner):
        with self.transaction():
            self.db.execute("DELETE FROM leases WHERE scope=? AND owner=?", (scope, owner))
        stream = self._host_locks.pop(scope, None)
        if stream:
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()

    def claim_job(self, job_id, input_hash, owner, now, seconds):
        with self.transaction():
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row and (
                row["state"] == "COMPLETED"
                or (row["state"] == "RUNNING" and row["expires_at"] > now.isoformat())
            ):
                return False
            self.db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,owner=excluded.owner,expires_at=excluded.expires_at",
                (
                    job_id,
                    input_hash,
                    "RUNNING",
                    owner,
                    (now + timedelta(seconds=seconds)).isoformat(),
                    "{}",
                ),
            )
        return True

    def finish_job(self, job_id, state, body):
        with self.transaction():
            self.db.execute(
                "UPDATE jobs SET state=?,body=? WHERE id=?", (state, canonical(body), job_id)
            )

    def close(self):
        for stream in self._host_locks.values():
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
        self._host_locks.clear()
        self.db.close()


def _host_lock_path(scope):
    return (
        Path("/tmp")
        / f"gauss-account-{os.getuid()}-{hashlib.sha256(scope.encode()).hexdigest()}.lock"
    )


def account_owned(account_id, environment, database_path=None):
    host_path = _host_lock_path(f"{environment}:{account_id}")
    if host_path.exists():
        fd = os.open(host_path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            return True
        finally:
            os.close(fd)
    path = database_path or os.getenv("GAUSS_DATABASE_PATH", DEFAULT_DATABASE_PATH)
    if not Path(path).exists():
        return False
    db = sqlite3.connect(f"file:{Path(path).absolute()}?mode=ro", uri=True)
    try:
        row = db.execute(
            "SELECT owner FROM leases WHERE scope=? AND expires_at>?",
            (f"{environment}:{account_id}", utc_now().isoformat()),
        ).fetchone()
        return bool(row)
    finally:
        db.close()
