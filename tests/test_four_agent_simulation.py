"""Deterministic offline tests of the historical session execution assumptions."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.runtime.models import RuntimeConfig, TradingSession
from src.runtime.session_simulation import simulate_sessions


def inputs():
    sessions, rows = [], []
    base = 100.0
    for day in range(5):
        opening = datetime(2026, 9, 21 + day, 13, 30, tzinfo=UTC)
        session = TradingSession(
            id=str(day),
            session_date=f"2026-09-{21 + day}",
            calendar_version="synthetic",
            open=opening,
            close=opening + timedelta(minutes=390),
        )
        sessions.append(session)
        for minute in range(390):
            price = base + (minute + 1) * 0.001
            rows.append(
                {
                    "symbol": "TEST",
                    "timestamp": opening + timedelta(minutes=minute),
                    "open": price - 0.001,
                    "high": price + 0.002,
                    "low": price - 0.002,
                    "close": price,
                    "volume": 10000,
                }
            )
        base += 0.390
    config = RuntimeConfig(
        symbols=("TEST",),
        strategy_allowlist=("trend_following",),
        stock_execution_feed="iex",
        entitlement_boundary_buffer=5,
        poll_seconds=5,
    )
    params = {
        "trend_following": {"fast": 2, "slow": 3, "stop_loss_pct": 0.01, "take_profit_pct": 0.02}
    }
    return pd.DataFrame(rows), sessions, config, params


def simulate(bars, sessions, config, params, **kwargs):
    return simulate_sessions(
        bars, sessions, config, strategy_params=params, cost_bps=(2,), **kwargs
    )


def test_nightly_signal_delay_future_fill_and_actual_session_time_exit():
    bars, sessions, config, params = inputs()
    report = simulate(bars, sessions, config, params)
    trades = [r for r in report["outcomes"] if r["outcome"] == "TRADE"]
    assert trades
    for row in trades:
        target = next(s for s in sessions if s.session_date == row["session"])
        signal = datetime.fromisoformat(row["signal_as_of"])
        decision = datetime.fromisoformat(row["decision_at"])
        fill = datetime.fromisoformat(row["entry_at"])
        assert decision >= signal + timedelta(seconds=905)
        assert fill > decision
        assert target.open <= signal < decision < target.close
        assert row["exit_reason"] == "TIME_EXIT"
        assert datetime.fromisoformat(row["exit_at"]) == target.close - timedelta(minutes=15)
    assert report["execution_eligible"] is False
    assert report["evidence_kind"] == "simulation"
    assert "initial_equity" not in report


def test_holdout_data_cannot_change_earlier_outcomes_or_frozen_parameters():
    bars, sessions, config, params = inputs()
    first = simulate(bars, sessions, config, params)
    held_dates = first["contract"]["partitions"]["holdout"]
    changed = bars.copy()
    mask = changed.timestamp.dt.strftime("%Y-%m-%d").isin(held_dates)
    for field in ("open", "high", "low", "close"):
        changed.loc[mask, field] *= 1.2
    second = simulate(changed, sessions, config, params)
    assert [r for r in first["outcomes"] if r["partition"] != "holdout"] == [
        r for r in second["outcomes"] if r["partition"] != "holdout"
    ]
    assert first["contract"]["strategy_parameters"] == second["contract"]["strategy_parameters"]
    assert first["source_digest"] != second["source_digest"]


def test_missing_minute_rejects_source_and_target_without_interpolation():
    bars, sessions, config, params = inputs()
    missing = sessions[2].open + timedelta(minutes=5)
    report = simulate(bars[bars.timestamp != missing], sessions, config, params)
    rows = {r["session"]: r for r in report["outcomes"]}
    assert rows[sessions[2].session_date]["outcome"] == "INCOMPLETE_TARGET_SESSION"
    assert rows[sessions[3].session_date]["outcome"] == "INCOMPLETE_SOURCE_SESSION"
    assert rows[sessions[2].session_date]["net_pnl_per_share"] is None


def test_ambiguous_entry_minute_stop_precedes_profit_and_cannot_improve_price():
    bars, sessions, config, params = inputs()
    baseline = simulate(bars, sessions, config, params)
    trade = next(r for r in baseline["outcomes"] if r["outcome"] == "TRADE")
    entry_minute = pd.Timestamp(trade["entry_at"]) - timedelta(minutes=1)
    altered = bars.copy()
    altered.loc[altered.timestamp == entry_minute, "low"] = trade["stop_price"] - 1
    altered.loc[altered.timestamp == entry_minute, "high"] = trade["take_profit_price"] + 1
    result = simulate(altered, sessions, config, params)
    stopped = next(r for r in result["outcomes"] if r["session"] == trade["session"])
    assert stopped["exit_reason"] == "STOP"
    assert stopped["exit_price"] <= stopped["stop_price"]
    assert stopped["net_pnl_per_share"] < 0


def test_source_schema_rejects_naive_timestamps_and_duplicate_minutes():
    bars, sessions, config, params = inputs()
    bad = bars.copy()
    bad["timestamp"] = bad.timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        simulate(bad, sessions, config, params)
    with pytest.raises(ValueError, match="duplicate"):
        simulate(pd.concat([bars, bars.iloc[:1]]), sessions, config, params)


def test_partitions_scope_symbols_costs_and_reproducible_digest():
    bars, sessions, config, params = inputs()
    first = simulate(bars, sessions, config, params)
    second = simulate(bars.set_index(["symbol", "timestamp"]), sessions, config, params)
    assert first == second
    refreshed_config = config.model_copy(
        update={
            "policy": config.policy.model_copy(
                update={"created_at": config.policy.created_at + timedelta(days=1)}
            )
        }
    )
    assert (
        simulate(bars, sessions, refreshed_config, params)["experiment_digest"]
        == first["experiment_digest"]
    )
    dates = first["contract"]["partitions"]
    assert set(dates) == {"development", "validation", "holdout"}
    assert len({date for values in dates.values() for date in values}) == len(sessions) - 1
    for row in first["summaries"]:
        assert row["symbol"] == "TEST"
        assert row["strategy"] == "trend_following"
        assert row["cost_bps_per_side"] == 2
        assert row["execution_eligible"] is False
        assert row["max_drawdown_per_share"] >= 0


def test_bar_prices_above_fixed_limit_never_get_an_invented_fill():
    bars, sessions, config, params = inputs()
    changed = bars.copy()
    target_mask = changed.timestamp >= sessions[1].open
    for field in ("open", "high", "low", "close"):
        changed.loc[target_mask, field] += 5
    report = simulate(changed, sessions, config, params)
    first = report["outcomes"][0]
    assert first["outcome"] == "LIMIT_BELOW_CURRENT_BID_PROXY"
    assert first["entry_at"] is None
