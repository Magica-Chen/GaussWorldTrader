"""Reproducible, per-share simulation of the nightly stock session contract.

Only supplied historical bars and exchange sessions are used. This is a disclosed
SIP bar execution model, not reconstructed IEX quotes, AI decisions, broker fills,
an account backtest, or an execution approval.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections import Counter
from datetime import timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .models import RuntimeConfig, TradingSession
from .store import digest

STRATEGIES = ("momentum", "trend_following", "mean_reversion")
LIMITATIONS = (
    "Historical IEX bid/ask, displayed size and queue position are unavailable; SIP bars are an execution proxy.",
    "Codex reviews, historical news/calendar vetoes and provider outages are not reconstructed.",
    "Fixed current universe has selection/survivorship bias; results apply only to these symbols and parameters.",
    "One share per independent symbol/strategy scenario; account sizing, cash, portfolio caps and tax are not simulated.",
    "Raw bar adjustments must match deployment; corporate actions may disrupt indicator continuity.",
    "Modeled fills are not broker fills or proof of executable venue liquidity; live trading remains unauthorized.",
)


def _frame(bars):
    frame = bars.copy()
    if isinstance(frame.index, pd.MultiIndex) or "timestamp" not in frame:
        frame = frame.reset_index()
    fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    if any(name not in frame for name in fields):
        raise ValueError("bars require symbol, timezone-aware minute-start timestamp and OHLCV")
    frame = frame[fields]
    timestamps = pd.to_datetime(frame["timestamp"])
    if timestamps.dt.tz is None:
        raise ValueError("bar timestamps must be timezone aware")
    frame["timestamp"] = timestamps.dt.tz_convert("UTC")
    if (frame["timestamp"].dt.second != 0).any() or (frame["timestamp"].dt.microsecond != 0).any():
        raise ValueError("source timestamps must be aligned minute starts")
    frame["symbol"] = frame["symbol"].astype(str)
    numeric = frame[["open", "high", "low", "close", "volume"]].astype(float)
    if not np.isfinite(numeric.to_numpy()).all() or (numeric.volume < 0).any():
        raise ValueError("bar values must be finite with nonnegative volume")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("bar prices must be positive")
    if (
        (numeric.high < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric.low > numeric[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("invalid OHLC bounds")
    frame[numeric.columns] = numeric
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("duplicate symbol/minute observations must be resolved upstream")
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _session_data(frame, session, timeframe):
    start, end = pd.Timestamp(session.open), pd.Timestamp(session.close)
    minute = frame.loc[(frame.index >= start) & (frame.index < end)]
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    if len(minute) != len(expected) or not minute.index.equals(expected):
        return None
    width = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "1Day": len(minute)}[timeframe]
    points = np.arange(0, len(minute), width)
    ends = np.minimum(points + width, len(minute))
    aggregated = pd.DataFrame(
        {
            "open": minute.open.to_numpy()[points],
            "high": np.maximum.reduceat(minute.high.to_numpy(), points),
            "low": np.minimum.reduceat(minute.low.to_numpy(), points),
            "close": minute.close.to_numpy()[ends - 1],
            "volume": np.add.reduceat(minute.volume.to_numpy(), points),
        },
        index=start + pd.to_timedelta(ends, unit="min"),
    )
    return minute, aggregated


def _trade(minute, signal_bars, session, plan, config, side_cost_bps):
    """One first-trigger attempt; costs are embedded in each modeled execution."""
    price, stop, target = plan
    limit = float(
        (Decimal(str(price)) * Decimal("1.001") / Decimal(".01")).to_integral_value(
            rounding=ROUND_DOWN
        )
        * Decimal(".01")
    )
    schedule = session.schedule(config.policy)
    delay = 900 + config.entitlement_boundary_buffer if config.data_profile == "FREE_DELAYED" else 0
    spread = side_cost_bps / 10000
    poll = config.poll_seconds
    output = {
        "outcome": "NO_TRIGGER",
        "trigger_price": price,
        "stop_price": stop,
        "take_profit_price": target,
        "limit_price": limit,
        "signal_as_of": None,
        "decision_at": None,
        "entry_at": None,
        "exit_at": None,
        "entry_price": None,
        "exit_price": None,
        "net_pnl_per_share": None,
    }
    decision = None
    for stamp, close in zip(signal_bars.index, signal_bars.close, strict=True):
        available = stamp + timedelta(seconds=delay)
        wall = pd.Timestamp(math.ceil(available.timestamp() / poll) * poll, unit="s", tz="UTC")
        if wall < schedule["entry_start"] or wall >= schedule["entry_cutoff"]:
            continue
        if close < stop:
            return {**output, "outcome": "DELAYED_PRICE_INVALIDATED"}
        if close > price:
            decision = wall
            output.update(signal_as_of=stamp.isoformat(), decision_at=wall.isoformat())
            break
    if decision is None:
        return output
    # The last completed *current-time* SIP minute is an explicit quote proxy.
    # It is not the delayed signal bar and cannot manufacture an IEX quote.
    index = int(minute.index.searchsorted(decision.floor("min"))) - 1
    if index < 0:
        return {**output, "outcome": "EXECUTION_PROXY_UNAVAILABLE"}
    current = float(minute.close.iloc[index])
    if current * (1 - spread) > limit:
        return {**output, "outcome": "LIMIT_BELOW_CURRENT_BID_PROXY"}
    if 2 * spread / (1 + spread) > float(config.policy.max_quote_spread_pct):
        return {**output, "outcome": "MODELED_SPREAD_TOO_WIDE"}
    expiry = min(
        decision + timedelta(seconds=config.order_cancel_seconds), schedule["entry_cutoff"]
    )
    beginning = int(minute.index.searchsorted(decision, side="left"))
    entry_index = None
    for i in range(beginning, len(minute)):
        stamp = minute.index[i]
        if stamp + timedelta(minutes=1) > expiry:
            break
        bar = minute.iloc[i]
        if float(bar.low) * (1 + spread) <= limit:
            entry_index = i
            # Paying the limit even after an opening improvement is pessimistic.
            output.update(entry_at=(stamp + timedelta(minutes=1)).isoformat(), entry_price=limit)
            break
    if entry_index is None:
        return {**output, "outcome": "UNFILLED_LIMIT_CANCELLED"}
    # The entry minute may hit a stop before or after the assumed fill. Treat
    # any such hit as adverse, then execute at the following minute's open.
    for i in range(entry_index, len(minute) - 1):
        stamp, bar = minute.index[i], minute.iloc[i]
        finish = stamp + timedelta(minutes=1)
        bid_low, bid_high = float(bar.low) * (1 - spread), float(bar.high) * (1 - spread)
        reason = None
        if bid_low <= stop:
            reason = "STOP"
        elif i > entry_index and target is not None and bid_high >= target:
            reason = "TAKE_PROFIT"
        elif finish >= schedule["closing_start"]:
            reason = "TIME_EXIT"
        if reason:
            next_open = float(minute.open.iloc[i + 1]) * (1 - spread)
            # A stop is not improved by an unknown intraminute ordering; a TP
            # gets no favorable gap windfall from this aggregate bar model.
            exit_price = (
                min(stop, next_open)
                if reason == "STOP"
                else (min(target, next_open) if reason == "TAKE_PROFIT" else next_open)
            )
            return {
                **output,
                "outcome": "TRADE",
                "exit_reason": reason,
                "exit_at": minute.index[i + 1].isoformat(),
                "exit_price": exit_price,
                "net_pnl_per_share": exit_price - limit,
            }
    return {**output, "outcome": "UNRESOLVED_SIMULATED_EXIT"}


def _summary(rows):
    pnl = np.asarray(
        [row["net_pnl_per_share"] for row in rows if row["outcome"] == "TRADE"], dtype=float
    )
    curve = np.concatenate(([0.0], np.cumsum(pnl)))
    count = len(pnl)
    mean = float(pnl.mean()) if count else None
    standard_error = float(pnl.std(ddof=1) / np.sqrt(count)) if count > 1 else None
    return {
        "sessions": len(rows),
        "trades": count,
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "net_pnl_per_share": float(pnl.sum()),
        "mean_net_pnl_per_share": mean,
        "standard_error_per_share": standard_error,
        "approximate_95pct_lower_mean_per_share": (
            mean - 1.96 * standard_error if standard_error is not None else None
        ),
        "max_drawdown_per_share": float(np.max(np.maximum.accumulate(curve) - curve)),
        "win_fraction": float((pnl > 0).mean()) if count else None,
        "uncertainty_note": "Normal approximation is descriptive; session dependence and strategy selection are not corrected.",
        "execution_eligible": False,
    }


def simulate_sessions(
    bars,
    sessions,
    config,
    *,
    strategy_params=None,
    cost_bps=(2.0, 5.0, 10.0),
    development_fraction=0.6,
    validation_fraction=0.2,
):
    """Evaluate fixed strategies with chronological, non-overlapping partitions.

    Input bars use Alpaca minute-start timestamps, flat columns or its native
    (symbol, timestamp) MultiIndex. Calendar input contains TradingSession values.
    Costs are adverse per-side bps scenarios, not measured historical spreads.
    """
    from src.strategy.registry import get_strategy_registry

    config = RuntimeConfig.model_validate(config)
    frame = _frame(bars)
    sessions = sorted((TradingSession.model_validate(s) for s in sessions), key=lambda s: s.open)
    if len(sessions) < 4 or any(
        a.close > b.open for a, b in zip(sessions, sessions[1:], strict=False)
    ):
        raise ValueError("at least four non-overlapping exchange sessions are required")
    if not (
        0 < development_fraction < 1
        and 0 < validation_fraction < 1
        and development_fraction + validation_fraction < 1
    ):
        raise ValueError("development/validation/holdout fractions must all be positive")
    costs = tuple(float(value) for value in cost_bps)
    if not costs or any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("cost scenarios must be finite nonnegative per-side bps")
    strategies = tuple(config.strategy_allowlist or STRATEGIES)
    if not set(strategies) <= set(STRATEGIES):
        raise ValueError("session simulation supports momentum, trend_following and mean_reversion")
    parameters = strategy_params or {}
    if not set(parameters) <= set(strategies):
        raise ValueError("parameters reference a strategy outside the frozen allowlist")
    registry = get_strategy_registry()
    implementations = {name: registry.create(name, parameters.get(name, {})) for name in strategies}
    frozen_params = {name: strategy.params for name, strategy in implementations.items()}
    implementation_hashes = {
        name: hashlib.sha256(Path(inspect.getfile(type(strategy))).read_bytes()).hexdigest()
        for name, strategy in implementations.items()
    }
    source_root = Path(__file__).resolve().parents[1]
    shared_hashes = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("strategy/base.py", "strategy/utils.py", "analysis/technical_analysis.py")
    }
    symbols = tuple(config.symbols or sorted(frame.symbol.unique()))
    targets = len(sessions) - 1
    development = min(targets - 2, max(1, int(targets * development_fraction)))
    validation = min(targets - development - 1, max(1, int(targets * validation_fraction)))
    partitions = ["development"] * development + ["validation"] * validation
    partitions += ["holdout"] * (targets - len(partitions))
    outcomes = []
    for symbol in symbols:
        source = frame.loc[frame.symbol == symbol].set_index("timestamp")
        source = source.drop(columns="symbol")
        prepared = [_session_data(source, session, config.signal_timeframe) for session in sessions]
        history_parts = [item[1] for item in prepared if item is not None]
        history = pd.concat(history_parts) if history_parts else pd.DataFrame()
        for j in range(1, len(sessions)):
            previous, current = sessions[j - 1], sessions[j]
            base = {
                "symbol": symbol,
                "source_session": previous.session_date,
                "session": current.session_date,
                "partition": partitions[j - 1],
            }
            source_data, target_data = prepared[j - 1], prepared[j]
            for name, strategy in implementations.items():
                item = {**base, "strategy": name}
                issue = (
                    "INCOMPLETE_SOURCE_SESSION"
                    if source_data is None
                    else ("INCOMPLETE_TARGET_SESSION" if target_data is None else None)
                )
                plan = None
                if issue is None:
                    prior = history.loc[history.index <= previous.close]
                    price = float(prior.close.iloc[-1])
                    signal = strategy.get_signal(
                        symbol=symbol,
                        current_date=previous.close,
                        current_price=price,
                        current_data={},
                        historical_data=prior,
                        portfolio=None,
                    )
                    action = (
                        strategy.get_action_plan(signal, price, previous.close) if signal else None
                    )
                    if signal is None:
                        issue = "INSUFFICIENT_HISTORY"
                    elif signal.signal != "BUY" or action is None:
                        issue = "NO_NIGHTLY_BUY_SIGNAL"
                    elif action.stop_loss is None or not 0 < float(action.stop_loss) < price:
                        issue = "INVALID_STOP"
                    else:
                        target = float(action.take_profit) if action.take_profit else None
                        plan = (
                            price,
                            float(action.stop_loss),
                            target if target and target > price else None,
                        )
                for cost in costs:
                    result = (
                        {"outcome": issue, "net_pnl_per_share": None}
                        if issue
                        else _trade(target_data[0], target_data[1], current, plan, config, cost)
                    )
                    outcomes.append({**item, "cost_bps_per_side": cost, **result})
    groups = {}
    for row in outcomes:
        key = (row["symbol"], row["strategy"], row["partition"], row["cost_bps_per_side"])
        groups.setdefault(key, []).append(row)
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    ).hexdigest()
    contract = {
        "symbols": list(symbols),
        "strategy_parameters": frozen_params,
        "strategy_implementation_hashes": implementation_hashes,
        "shared_implementation_hashes": shared_hashes,
        "simulator_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "signal_timeframe": config.signal_timeframe,
        "data_profile": str(config.data_profile),
        "execution_feed": config.stock_execution_feed,
        "source_bar_feed": "sip",
        "source_bar_timestamp": "interval_start",
        "cost_bps_per_side": costs,
        "delay_seconds": (
            900 + config.entitlement_boundary_buffer if config.data_profile == "FREE_DELAYED" else 0
        ),
        "poll_seconds": config.poll_seconds,
        "order_cancel_seconds": config.order_cancel_seconds,
        "policy": config.policy.model_dump(mode="json", exclude={"created_at"}),
        "partitions": {
            name: [
                s.session_date
                for s, part in zip(sessions[1:], partitions, strict=True)
                if part == name
            ]
            for name in ("development", "validation", "holdout")
        },
        "calendar": [
            {"date": s.session_date, "open": s.open.isoformat(), "close": s.close.isoformat()}
            for s in sessions
        ],
    }
    return {
        "schema_version": 1,
        "evidence_kind": "simulation",
        "execution_eligible": False,
        "source_digest": source_hash,
        "experiment_digest": digest([source_hash, contract]),
        "contract": contract,
        "limitations": list(LIMITATIONS),
        "assumptions": [
            "Signals use only completed regular-session bars through the prior session close.",
            "AI/news/event gates are assumed permissive to measure the technical policy, not AI selection value.",
            "Delayed completed-bar triggers are released after the configured delay and rounded up to a poll boundary.",
            "First qualifying trigger has one bounded limit attempt; a risk-proxy rejection is terminal for that plan.",
            "An entry fill requires a full future minute within the cancellation window and pays the limit.",
            "Costs model spread/slippage together; no additional double deduction from modeled fill P&L.",
            "Stops win ambiguous stop/target bars; exit uses a following-minute open with no favorable stop/target improvement.",
            "Incomplete source or target sessions are skipped, never interpolated; no artificial initial equity is assumed.",
        ],
        "summaries": [
            {
                "symbol": key[0],
                "strategy": key[1],
                "partition": key[2],
                "cost_bps_per_side": key[3],
                **_summary(rows),
            }
            for key, rows in groups.items()
        ],
        "outcomes": outcomes,
    }
