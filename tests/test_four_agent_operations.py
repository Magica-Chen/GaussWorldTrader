from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3

import pytest

from src.runtime.evaluation import EvaluationSample, compare_evaluations, evaluate_sample
from src.runtime.operations import backup_state, restore_state, verify_backup


def sample(**changes):
    values = dict(
        sample_id="scope-001", account_id="explicit-test-account", hypothetical=True,
        data_profile="FREE_DELAYED", strategy_id="fixture", selection_method="fixed_watchlist",
        period_start=datetime(2026, 9, 8, tzinfo=timezone.utc),
        period_end=datetime(2026, 9, 9, tzinfo=timezone.utc), partition="holdout",
        evidence_kind="actual_fills", initial_equity="1000", gross_trading_pnl="20",
        fees="2", operating_costs=(("model", "3"), ("hosting", "1")),
        observed_lag_seconds=("905", "920"), submitted_orders=2, filled_orders=1,
    )
    values.update(changes)
    return EvaluationSample(**values)


def test_actual_fill_costs_are_deducted_once():
    report = evaluate_sample(sample())
    assert Decimal(report["net_trading_pnl"]) == 18
    assert Decimal(report["all_in_pnl"]) == 14
    assert Decimal(report["mean_additional_lag_seconds"]) == Decimal("12.5")
    assert report["execution_eligible"] is False


def test_equity_pnl_removes_external_deposit_and_does_not_charge_fees_twice():
    report = evaluate_sample(sample(gross_trading_pnl=None, ending_equity="1518", external_cashflows="500"))
    assert Decimal(report["net_trading_pnl"]) == 18
    assert Decimal(report["all_in_pnl"]) == 14


def test_decision_only_evidence_never_invents_fills_or_profit():
    with pytest.raises(ValueError, match="decision-only"):
        sample(evidence_kind="decision_only")
    with pytest.raises(ValueError, match="cannot assert fills"):
        sample(evidence_kind="decision_only", gross_trading_pnl=None)
    report = evaluate_sample(sample(evidence_kind="decision_only", gross_trading_pnl=None,
                                    filled_orders=0))
    assert report["net_trading_pnl"] is None
    assert report["all_in_return"] is None


def test_repeated_shared_cost_id_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        sample(operating_costs=(("shared-subscription", "1"), ("shared-subscription", "1")))


def test_overlapping_holdout_and_development_are_rejected():
    with pytest.raises(ValueError, match="must not overlap"):
        compare_evaluations([sample(), sample(sample_id="development", partition="development")])


def test_comparison_keeps_profiles_and_all_experiments_separate():
    rows = [sample(), sample(sample_id="realtime", data_profile="SUBSCRIBED_REALTIME")]
    report = compare_evaluations(rows)
    assert len(report["samples"]) == 2
    assert {row["data_profile"] for row in report["samples"]} == {"FREE_DELAYED", "SUBSCRIBED_REALTIME"}
    assert report["live_authorized"] is False


def test_nested_development_window_cannot_hide_holdout_leakage():
    start = datetime(2026, 9, 8, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="must not overlap"):
        compare_evaluations([
            sample(sample_id="long-training", partition="development", period_start=start,
                   period_end=start+timedelta(days=10)),
            sample(sample_id="short-training", partition="development",
                   period_start=start+timedelta(days=1), period_end=start+timedelta(days=2)),
            sample(sample_id="holdout", partition="holdout", period_start=start+timedelta(days=3),
                   period_end=start+timedelta(days=4)),
        ])


def test_report_preserves_current_wall_time_and_delayed_signal_asof():
    wall = datetime(2026, 9, 8, 14, tzinfo=timezone.utc)
    report = evaluate_sample(sample(observation_wall_time=wall,
        signal_asof=wall-timedelta(seconds=905), source_coverage_expected=3,
        source_coverage_received=2))
    assert report["observation_wall_time"] == wall.isoformat()
    assert report["signal_asof"] == (wall-timedelta(seconds=905)).isoformat()
    assert Decimal(report["source_coverage_fraction"]) == Decimal(2)/3
    assert report["fees"] == "2"
    with pytest.raises(ValueError, match="as-of"):
        sample(observation_wall_time=wall, signal_asof=wall+timedelta(seconds=1))


def test_metrics_remain_unavailable_without_observed_quote_or_equity_history():
    report = evaluate_sample(sample())
    for key in ("mean_quote_age_seconds", "coverage_fraction", "peak_backlog_events",
                "recovery_success_fraction", "validation_reversals", "cancel_rate",
                "mean_slippage_bps", "peak_exposure", "max_drawdown_fraction",
                "operating_cost_per_filled_group", "uptime_fraction"):
        assert report[key] is None


def test_drawdown_removes_deposit_before_measuring_trading_loss():
    start = datetime(2026, 9, 8, tzinfo=timezone.utc)
    report = evaluate_sample(sample(equity_observations=(
        {"observed_at": start, "equity": "1000", "cumulative_external_cashflows": "0"},
        {"observed_at": start+timedelta(hours=1), "equity": "1500", "cumulative_external_cashflows": "500"},
        {"observed_at": start+timedelta(hours=2), "equity": "1400", "cumulative_external_cashflows": "500"},
    )))
    assert Decimal(report["max_drawdown_fraction"]) == Decimal("0.10")
    assert Decimal(report["max_drawdown_dollars"]) == 100


def test_execution_diagnostics_do_not_deduct_fill_embedded_slippage_again():
    report = evaluate_sample(sample(slippage_bps=("20", "-10"),
        slippage_benchmark="recorded arrival midpoint", spread_at_decision_bps=("5", "7"),
        candidates=2, filled_position_groups=1, cancelled_orders=1, replaced_orders=2,
        quote_age_seconds=("1", "3"), coverage_expected_records=10, coverage_received_records=9,
        recovery_attempts=2, successful_recoveries=1))
    assert Decimal(report["all_in_pnl"]) == 14
    assert Decimal(report["mean_slippage_bps"]) == 5
    assert Decimal(report["coverage_fraction"]) == Decimal("0.9")
    assert Decimal(report["recovery_success_fraction"]) == Decimal("0.5")
    assert Decimal(report["operating_cost_per_filled_group"]) == 4
    assert Decimal(report["operating_cost_per_candidate"]) == 2


def test_execution_metrics_require_valid_denominators_and_benchmarks():
    with pytest.raises(ValueError, match="benchmark"):
        sample(slippage_bps=("1",))
    with pytest.raises(ValueError, match="coverage"):
        sample(coverage_received_records=2, coverage_expected_records=1)


def test_backup_restores_committed_wal_state_and_evidence(tmp_path):
    database = tmp_path / "active.db"
    evidence = tmp_path / "payloads"
    evidence.mkdir()
    (evidence / "immutable.json").write_text('{"source_version": 1}')
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA user_version=1")
    connection.execute("CREATE TABLE orders (id TEXT PRIMARY KEY, state TEXT)")
    connection.execute("INSERT INTO orders VALUES ('intent-1', 'UNKNOWN')")
    connection.commit()
    saved = tmp_path / "backup"
    backup_state(database, saved, evidence_directory=evidence)
    restored = restore_state(saved, tmp_path / "restored", expected_schema_version=1)
    with sqlite3.connect(restored["database_path"]) as reader:
        assert reader.execute("SELECT state FROM orders").fetchone()[0] == "UNKNOWN"
    assert (tmp_path / "restored/evidence/immutable.json").read_text() == '{"source_version": 1}'
    assert restored["execution_armed"] is False
    connection.close()


def test_restore_refuses_corruption_and_existing_destination(tmp_path):
    database = tmp_path / "active.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE state (value TEXT)")
    saved = tmp_path / "backup"
    backup_state(database, saved)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        restore_state(saved, existing)
    (saved / "session.db").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_backup(saved)


def test_restore_refuses_schema_mismatch_and_path_traversal(tmp_path):
    import json
    database = tmp_path / "active.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE state (value TEXT)")
    saved = tmp_path / "backup"
    backup_state(database, saved)
    with pytest.raises(ValueError, match="incompatible"):
        restore_state(saved, tmp_path / "target", expected_schema_version=999)
    manifest_path = saved / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../escape"] = "fake"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe relative path"):
        verify_backup(saved)
