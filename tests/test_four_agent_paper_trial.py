"""Paper experiment authorization remains separate from validated expectancy."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.runtime.models import (
    AccountProfile,
    Alternative,
    Instrument,
    RiskPolicy,
    StrategyApproval,
)
from src.runtime.suitability import AccountSuitabilityService

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


def approval(**changes):
    values = {
        "strategy_id": "fixture",
        "strategy_version": "v1",
        "account_id": "paper-fixture",
        "environment": "paper",
        "approved_by": "operator-paper-experiment",
        "profiles": ("FREE_DELAYED",),
        "execution_modes": ("paper", "shadow", "replay"),
        "maximum_signal_latency_seconds": 1200,
        "validation_reference": "results/fixture/fixed-contract-evaluation.json",
        "validated_net_expectancy": None,
        "evidence_level": "paper_trial",
        "symbols": ("AAA", "BBB"),
        "entry_bound_pct": ".01",
        "expiry": NOW + timedelta(days=7),
    }
    values.update(changes)
    return StrategyApproval(**values)


def policy(**changes):
    values = {
        "max_capital_allocation_pct": ".02",
        "max_portfolio_allocation_pct": ".10",
        "planned_loss_per_trade_pct": ".001",
        "daily_loss_pause_pct": ".01",
    }
    values.update(changes)
    return RiskPolicy(**values)


def account(**changes):
    values = {
        "account_id": "paper-fixture",
        "environment": "paper",
        "equity": "10000",
        "cash": "10000",
        "buying_power": "10000",
        "currency": "USD",
        "observed_at": NOW,
        "mandate_id": "operator-mandate-v1",
    }
    values.update(changes)
    return AccountProfile(**values)


def alternative(**changes):
    values = {
        "id": "candidate",
        "strategy_id": "fixture",
        "strategy_version": "v1",
        "instrument": Instrument(symbol="AAA", liquidity_capacity="10000"),
        "entry_price": "100",
        "stop_price": "99",
        "expected_net_value": None,
        "evidence_ids": ("delayed-sip-observation",),
    }
    values.update(changes)
    return Alternative(**values)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"environment": "live"}, "paper environment"),
        ({"asset_type": "option"}, "stock only"),
        ({"symbols": ()}, "explicit symbols"),
        ({"symbols": ("AAA", "AAA")}, "unique"),
        ({"symbols": (" ",)}, "nonempty"),
        ({"symbols": (" AAA",)}, "whitespace"),
        ({"execution_modes": ("live",)}, "execution modes"),
        ({"execution_modes": ("paper", "live")}, "execution modes"),
        ({"execution_modes": ()}, "execution modes"),
        ({"validated_net_expectancy": "0"}, "unknown"),
        ({"validated_net_expectancy": "1"}, "unknown"),
        ({"validation_reference": " \n"}, "validation reference"),
        ({"entry_bound_pct": "0"}, "greater than 0"),
        ({"entry_bound_pct": ".01001"}, "less than or equal"),
    ],
)
def test_trial_contract_rejects_unscoped_or_economically_mislabelled_release(changes, match):
    with pytest.raises(ValueError, match=match):
        approval(**changes)


def test_legacy_approval_defaults_remain_validated_and_unscoped():
    values = approval().model_dump()
    for field in ("evidence_level", "symbols", "entry_bound_pct"):
        values.pop(field)
    values["validated_net_expectancy"] = "1"
    restored = StrategyApproval.model_validate(values)
    assert restored.evidence_level == "validated"
    assert restored.symbols == ()
    assert restored.entry_bound_pct == Decimal(".001")


def test_trial_has_unknown_expectancy_and_ranks_lower_fractional_stop_risk():
    low_dollar_risk = alternative(id="higher-fraction", entry_price="10", stop_price="9.80")
    lower_fraction = alternative(id="lower-fraction", entry_price="100", stop_price="99")
    tied = alternative(id="z-tied", entry_price="100", stop_price="99")
    report = AccountSuitabilityService(policy()).assess(
        account(), [tied, low_dollar_risk, lower_fraction], "FREE_DELAYED", [approval()]
    )
    assert report.selected_alternative_id == "lower-fraction"
    assert report.objective == "paper_trial_lowest_fractional_planned_loss"
    assert report.ranking_scores == (
        ("lower-fraction", Decimal("-.0102")),
        ("z-tied", Decimal("-.0102")),
        ("higher-fraction", Decimal("-.022")),
    )
    assert all(row.feasible and row.evidence_level == "paper_trial" for row in report.alternatives)
    assert all(row.capital <= 200 and row.planned_loss <= 10 for row in report.alternatives)
    assert "unknown" in report.uncertainty[0]
    assert lower_fraction.expected_net_value is None


@pytest.mark.parametrize(
    "changes",
    [
        {"max_capital_allocation_pct": ".02001"},
        {"max_portfolio_allocation_pct": ".10001"},
        {"planned_loss_per_trade_pct": ".00101"},
        {"daily_loss_pause_pct": ".01001"},
        {"allow_leverage": True},
        {"options_enabled": True},
        {"spreads_enabled": True},
    ],
)
def test_trial_rejects_relaxed_policy_instead_of_silently_resizing(changes):
    report = AccountSuitabilityService(policy(**changes)).assess(
        account(), [alternative()], "FREE_DELAYED", [approval()]
    )
    assert report.outcome == "NO_TRADE"
    assert "PAPER_TRIAL_POLICY_EXCEEDS_LIMITS" in report.alternatives[0].reasons


@pytest.mark.parametrize(
    "account_changes,reason",
    [
        ({"environment": "live"}, "PAPER_TRIAL_ACCOUNT_SCOPE"),
        ({"environment": "scenario", "hypothetical": True}, "PAPER_TRIAL_ACCOUNT_SCOPE"),
        ({"account_id": "another-paper-account"}, "PAPER_TRIAL_ACCOUNT_SCOPE"),
        ({"cash": "0", "buying_power": "0"}, "NO_FEASIBLE_SIZE"),
        ({"exposure": "1000"}, "NO_FEASIBLE_SIZE"),
        ({"trading_blocked": True}, "ACCOUNT_RESTRICTED"),
        ({"mandate_id": "another-mandate"}, "MANDATE_MISMATCH"),
    ],
)
def test_trial_retains_account_scope_and_existing_capital_constraints(account_changes, reason):
    report = AccountSuitabilityService(policy()).assess(
        account(**account_changes), [alternative()], "FREE_DELAYED", [approval()]
    )
    assert report.outcome == "NO_TRADE"
    assert reason in report.alternatives[0].reasons


@pytest.mark.parametrize("level", ["validated", "paper_trial"])
def test_symbol_scope_is_required_for_both_release_classes(level):
    trial = level == "paper_trial"
    report = AccountSuitabilityService(policy()).assess(
        account(),
        [alternative(expected_net_value=None if trial else "1")],
        "FREE_DELAYED",
        [approval(evidence_level=level, symbols=("BBB",))],
    )
    assert report.outcome == "NO_TRADE"
    assert "STRATEGY_SYMBOL_NOT_APPROVED" in report.alternatives[0].reasons


def test_separate_symbol_approvals_do_not_overwrite_each_other():
    report = AccountSuitabilityService(policy()).assess(
        account(),
        [alternative()],
        "FREE_DELAYED",
        [approval(symbols=("AAA",)), approval(symbols=("BBB",))],
    )
    assert report.outcome == "FEASIBLE"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"expected_net_value": "1"}, "PAPER_TRIAL_EXPECTANCY_MUST_BE_UNKNOWN"),
        ({"evidence_ids": ()}, "INSUFFICIENT_EVIDENCE"),
        ({"stop_price": "101"}, "INVALID_STOP_DISTANCE"),
        (
            {"short_instrument": Instrument(symbol="BBB", liquidity_capacity="10")},
            "PAPER_TRIAL_STOCK_ONLY",
        ),
    ],
)
def test_trial_still_needs_provenance_valid_stops_and_single_stock_leg(changes, reason):
    report = AccountSuitabilityService(policy()).assess(
        account(), [alternative(**changes)], "FREE_DELAYED", [approval()]
    )
    assert report.outcome == "NO_TRADE"
    assert reason in report.alternatives[0].reasons


def test_trial_expiry_and_data_profile_remain_binding():
    report = AccountSuitabilityService(policy()).assess(
        account(), [alternative()], "SUBSCRIBED_REALTIME", [approval(expiry=NOW)]
    )
    assert report.outcome == "NO_TRADE"
    assert set(report.alternatives[0].reasons) == {
        "STRATEGY_APPROVAL_EXPIRED",
        "DATA_PROFILE_INCOMPATIBLE",
    }


def test_mixed_report_labels_distinct_ranking_objectives_and_prefers_validation():
    validated = alternative(id="validated", strategy_id="validated-fixture", expected_net_value="1")
    report = AccountSuitabilityService(policy()).assess(
        account(),
        [alternative(), validated],
        "FREE_DELAYED",
        [
            approval(),
            approval(
                strategy_id="validated-fixture",
                evidence_level="validated",
                validated_net_expectancy="1",
            ),
        ],
    )
    assert report.selected_alternative_id == "validated"
    assert report.objective == (
        "validated_expectancy_first_then_paper_trial_lowest_fractional_planned_loss"
    )
    assert [row.evidence_level for row in report.alternatives] == ["paper_trial", "validated"]
