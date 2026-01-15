"""Helpers to run multiple live trading engines concurrently."""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Iterable

from .live_trading_base import LiveTradingEngine


def run_live_engines(engines: Iterable[LiveTradingEngine]) -> None:
    """Run multiple live trading engines concurrently until interrupted."""
    engine_list = list(engines)
    threads: list[threading.Thread] = []

    if not engine_list:
        return

    if len(engine_list) == 1:
        engine_list[0].start()
        return

    first = engine_list[0]
    first_cls = type(first)
    if not all(isinstance(engine, first_cls) for engine in engine_list):
        raise ValueError("Mixed asset types are not supported in a shared stream")

    if hasattr(first, "crypto_loc"):
        crypto_loc = getattr(first, "crypto_loc")
        if any(getattr(engine, "crypto_loc", crypto_loc) != crypto_loc for engine in engine_list):
            raise ValueError("All crypto engines must use the same crypto_loc for shared streaming")

    stream = first._create_stream()
    engine_map = {engine.symbol: engine for engine in engine_list}

    for engine in engine_list:
        engine._stream = stream
        engine._stop_event.clear()

    async def handle_trade(data: object) -> None:
        symbol = first._get_field(data, "symbol", "S")
        if not symbol:
            return
        normalized = first._normalize_symbol(str(symbol))
        target = engine_map.get(normalized)
        if not target:
            return
        price = target._get_field(data, "price", "p")
        timestamp = target._get_field(data, "timestamp", "t")
        if price is None:
            return
        target._handle_trade_update(
            float(price), timestamp if isinstance(timestamp, datetime) else None
        )

    for engine in engine_list:
        engine._subscribe_to_stream(handle_trade, engine.symbol)

    def _run_stream() -> None:
        try:
            stream.run()
        except Exception as exc:
            first.logger.error("Stream stopped: %s", exc)

    stream_thread = threading.Thread(target=_run_stream, name="live_stream", daemon=False)
    stream_thread.start()
    first.logger.info(
        "Shared stream started for %d symbols", len(engine_list)
    )

    for engine in engine_list:
        name = f"{engine.__class__.__name__}_{engine.symbol}"
        thread = threading.Thread(target=engine.run_signal_loop, name=name, daemon=False)
        thread.start()
        threads.append(thread)

    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        for engine in engine_list:
            engine._stop_event.set()
        try:
            stream.stop()
        except Exception as exc:
            first.logger.error("Failed to stop stream: %s", exc)
    finally:
        stream_thread.join(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
