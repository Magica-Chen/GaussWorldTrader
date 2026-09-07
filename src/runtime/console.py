"""Compact terminal status with complete JSON output for log consumers."""

from __future__ import annotations

import json
import sys
from datetime import datetime


def _words(value):
    return str(value).replace("_", " ").lower()


def _time(value):
    if not value:
        return "pending"
    try:
        return (
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            .strftime("%Y-%m-%d %H:%M:%S %Z")
            .strip()
        )
    except ValueError:
        return str(value)


class ConsoleOutput:
    def __init__(self, output="auto", stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.json = output == "json" or (output == "auto" and not self.stream.isatty())
        self.previous = None

    def __call__(self, record):
        if self.json:
            print(json.dumps(record, default=str), file=self.stream, flush=True)
            return
        event = record.get("event", "event")
        if event == "heartbeat":
            status = record.get("status") or {}
            entries = (
                "paused"
                if status.get("entries_paused")
                else ("ready" if status.get("entries_ready") else "blocked")
            )
            headline = (
                f"Gauss | {status.get('environment', '?')} / "
                f"{status.get('execution_mode', '?')} | "
                f"{_words(status.get('runtime_state', 'unknown'))} | entries {entries}"
            )
            health = " | ".join(
                f"{_words(k)}: {_words(v)}" for k, v in sorted(status.get("health", {}).items())
            )
            roles = " | ".join(
                f"{k}: {_words(v.get('state', 'unknown'))}"
                for k, v in sorted(status.get("roles", {}).items())
            )
            schedule = status.get("schedule") or {}
            lines = [
                headline,
                f"  Data: {_words(status.get('data_profile', 'unknown'))} "
                f"(delay {status.get('configured_delay_seconds', 0)}s)",
                "  Entry blockers: "
                + ("; ".join(_words(v) for v in status.get("readiness", [])) or "none"),
                f"  Health: {health or 'pending'}",
                f"  Agents: {roles or 'pending'}",
                f"  Next session: {_time(schedule.get('open'))} to {_time(schedule.get('close'))}",
                f"  Preparation: {_time(schedule.get('pre_start'))} | "
                f"Entries from: {_time(schedule.get('entry_start'))}",
            ]
            stamp = _time(record.get("at"))
            if lines != self.previous:
                print("\n" + "\n".join(lines), file=self.stream)
                if self.previous is None:
                    print(
                        "  Ctrl+C: request shutdown (remaining exposure is reported first).",
                        file=self.stream,
                    )
                self.previous = lines
            print(
                f"  [{stamp}] heartbeat | reconciled {_time(status.get('last_reconciliation'))}",
                file=self.stream,
                flush=True,
            )
            return
        if event == "shutdown":
            message = (
                "Stopped; supervision ended." if record.get("stopped") else "Shutdown blocked."
            )
            message += (
                f" Remaining position groups: {len(record.get('remaining_groups', []))}; "
                f"unresolved orders: {len(record.get('unresolved_orders', []))}."
            )
        else:
            message = " | ".join(
                str(record[k])
                for k in ("reason", "message", "error", "detail", "action")
                if record.get(k)
            )
        print(f"\n{_words(event).upper()}: {message}", file=self.stream, flush=True)
