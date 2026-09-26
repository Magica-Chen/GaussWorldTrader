"""Read-only accounting for this bot's recorded Codex calls and subscription allowance."""

from datetime import datetime

from .models import utc_now


def token_usage_summary(store, scope, session_id=None, *, now=None):
    now = now or utc_now()
    rows = (
        [
            row
            for row in store.projected("codex_usage", scope)
            if session_id is None or row.get("session_id") == session_id
        ]
        if scope
        else []
    )

    def summarize(items):
        counts = {
            "calls": len(items),
            "completed": 0,
            "failed": 0,
            "pending": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "unknown_usage_calls": 0,
        }
        for row in items:
            state = row.get("state")
            counts[
                (
                    "completed"
                    if state == "COMPLETED"
                    else "failed" if state == "FAILED" else "pending"
                )
            ] += 1
            known = True
            for key in ("input_tokens", "output_tokens"):
                value = row.get(key)
                if type(value) is int and value >= 0:
                    counts[key] += value
                else:
                    known = False
            if not known:
                counts["unknown_usage_calls"] += 1
        counts["total_tokens"] = counts["input_tokens"] + counts["output_tokens"]
        return counts

    control = store.projection("automation_control", scope) or {} if scope else {}
    quota = control.get("quota") or {}
    observed = quota.get("observed_at")
    age = now.timestamp() - observed if isinstance(observed, int | float) else None
    runtime = store.projection("runtime", scope) or {} if scope else {}
    try:
        heartbeat_age = (
            now - datetime.fromisoformat(runtime["updated_at"].replace("Z", "+00:00"))
        ).total_seconds()
    except (KeyError, ValueError, TypeError):
        heartbeat_age = None
    return {
        "scope": scope,
        "session_id": session_id,
        "observed_at": now.isoformat(),
        "totals": summarize(rows),
        "by_role": {
            role: summarize([r for r in rows if r.get("role", "unknown") == role])
            for role in sorted({r.get("role", "unknown") for r in rows})
        },
        "by_session": {
            session: summarize([r for r in rows if r.get("session_id", "unknown") == session])
            for session in sorted({r.get("session_id", "unknown") for r in rows})
        },
        "allowance": {
            "remaining_percent": control.get("remaining_percent"),
            "stale": age is None or not 0 <= age <= 120,
            "age_seconds": age,
            "checked_at": control.get("checked_at"),
            "monitor_only": control.get("monitor_only"),
            "quota_latched": control.get("quota_latched", False),
            "reason": control.get("reason"),
            "stop_at_remaining_percent": 5,
            "windows": quota.get("windows", []),
        },
        "runtime": {
            "state": runtime.get("runtime_state", "NOT_STARTED"),
            "entries_paused": runtime.get("entries_paused"),
            "heartbeat_age_seconds": heartbeat_age,
            "heartbeat_stale": heartbeat_age is None or not 0 <= heartbeat_age <= 120,
        },
        "note": "Tokens cover this bot's recorded calls only. Unknown usage is not zero. "
        "The subscription allowance is account-wide and is not calculated from these tokens.",
    }


def usage_panel(summary):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    totals, quota, runtime = summary["totals"], summary["allowance"], summary["runtime"]
    remaining = quota["remaining_percent"]
    allowance = "unavailable" if remaining is None else f"{remaining:g}% remaining"
    if quota["stale"]:
        allowance += " (stale; last recorded value)"
    table = Table("Stage", "Calls", "Input", "Output", "Total", "Unknown usage")
    for role, values in summary["by_role"].items():
        table.add_row(
            role,
            *(
                f"{values[k]:,}"
                for k in (
                    "calls",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "unknown_usage_calls",
                )
            ),
        )
    state = runtime["state"] + (" (stale heartbeat)" if runtime["heartbeat_stale"] else "")
    return Panel(
        Group(
            Text(
                f"Updated: {summary['observed_at']} | Session: {summary['session_id'] or 'all recorded'}"
            ),
            Text(f"Runtime: {state} | Entries paused: {runtime['entries_paused']}"),
            Text(
                f"Recorded tokens: {totals['total_tokens']:,} "
                f"(input {totals['input_tokens']:,}, output {totals['output_tokens']:,})"
            ),
            Text(
                f"Calls: {totals['calls']} | Completed: {totals['completed']} | Failed: {totals['failed']} "
                f"| Pending/unresolved: {totals['pending']} | Unknown usage: {totals['unknown_usage_calls']}"
            ),
            Text(
                f"Codex allowance: {allowance} | Monitoring-only stop: {quota['monitor_only']} "
                "| Stop threshold: 5% remaining"
            ),
            table,
            Text(summary["note"]),
            Text("Ctrl+C closes this monitor only."),
        ),
        title="Gauss token monitor",
    )
