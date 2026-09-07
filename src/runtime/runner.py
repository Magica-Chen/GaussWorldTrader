"""Process lifecycle for the persistent session service, independent of the UI."""

from __future__ import annotations

import signal
import threading
from datetime import UTC, datetime


def run_service(
    service,
    *,
    once=False,
    acknowledge_unmanaged_exposure=False,
    stop_event=None,
    emit=None,
    wait=None,
    output="auto",
):
    """Run service cycles; an unresolved shutdown retains management and ownership.

    Broker reconciliation and independent supervision belong to SessionService.
    The process never recreates a failed service or converts unknown exposure to flat.
    """
    from .console import ConsoleOutput
    from .store import LeaseConflict

    emit = emit if emit is not None else ConsoleOutput(output)
    previous_sink = getattr(service, "event_sink", None)
    service.event_sink = emit
    stopping = stop_event or threading.Event()
    wait = wait or stopping.wait
    acknowledgement_offered = False
    acknowledged = acknowledge_unmanaged_exposure

    def handle_signal(sig, _frame):
        nonlocal acknowledged
        if sig == signal.SIGINT and acknowledgement_offered:
            acknowledged = True
        stopping.set()

    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handle_signal)
    shutdown_pending = False
    cycle_failed = False
    try:
        while True:
            try:
                result = service.run_once()
                emit(
                    {
                        "event": "heartbeat",
                        "at": datetime.now(UTC).isoformat(),
                        "status": result if result is not None else service.status(),
                    }
                )
                cycle_failed = False
                if isinstance(result, dict) and result.get("supervision") == "ENDED":
                    return 0
            except KeyboardInterrupt:
                handle_signal(signal.SIGINT, None)
            except Exception as exc:
                if isinstance(exc, LeaseConflict) and not getattr(service, "_owns_lease", False):
                    emit(
                        {
                            "event": "startup_blocked",
                            "error": str(exc),
                            "message": "This process did not start account supervision and will exit.",
                        }
                    )
                    service.request_shutdown()
                    return 1
                cycle_failed = True
                shutdown_pending = True
                emit(
                    {
                        "event": "cycle_error",
                        "error": str(exc),
                        "action": "Pause entries and reconcile before deciding whether supervision can end",
                    }
                )

            if once or stopping.is_set() or shutdown_pending:
                stopping.clear()
                shutdown_pending = True
                try:
                    # The service returns actual residual orders/positions and persists the decision.
                    outcome = service.request_shutdown(acknowledge_unmanaged_exposure=acknowledged)
                    emit({"event": "shutdown", **outcome})
                    if outcome.get("stopped") is True:
                        return 1 if cycle_failed else 0
                    emit(
                        {
                            "event": "management_continues",
                            "message": "Entries are paused. Residual or unverified exposure remains supervised. "
                            "Press Ctrl+C again to acknowledge ending supervision with remaining exposure. "
                            "This does not liquidate positions or cancel every broker order.",
                        }
                    )
                    acknowledgement_offered = True
                except Exception as exc:
                    emit(
                        {
                            "event": "shutdown_unverified",
                            "error": str(exc),
                            "message": "Retaining the service; shutdown safety could not be established.",
                        }
                    )
            interval = min(60.0, max(1.0, float(service.config.poll_seconds)))
            wait(interval)
    finally:
        service.event_sink = previous_sink
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
