"""T60: complete synthetic handovers across both clocks and explicit capital cases."""
import copy
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.runtime.service import replay_fixture


FIXTURE = Path(__file__).parent / "fixtures" / "gauss_session_day.json"


@pytest.mark.parametrize("profile", ["FREE_DELAYED", "SUBSCRIBED_REALTIME"])
@pytest.mark.parametrize("equity,cash,expect_intent", [
    ("50", "50", False),
    ("10000", "10000", True),
    ("10000", "0", False),
    ("1000000", "1000000", True),
])
def test_t60_two_session_account_profile_matrix(tmp_path, profile, equity, cash, expect_intent):
    fixture = json.loads(FIXTURE.read_text())
    fixture["config"]["database_path"] = str(tmp_path / "replay.sqlite3")
    fixture["config"]["data_profile"] = profile
    fixture["account"].update(equity=equity, cash=cash, buying_power=cash)
    for step in fixture["steps"]:
        step["account"].update(equity=equity, cash=cash, buying_power=cash)
    result = replay_fixture(fixture)
    assert result["broker_writes"] == 0
    assert {run["role"] for run in result["agent_runs"]} == {
        "PostGauss", "CloseGauss", "PreGauss", "LiveGauss"}
    assert all(run["state"] == "COMPLETED" for run in result["agent_runs"]), result["agent_runs"]
    assert all(run["data_profile"] == profile for run in result["agent_runs"])
    assert result["suitability_reports"]
    assert bool(result["orders"]) is expect_intent
    if expect_intent:
        assert len(result["orders"]) == 1
        order = result["orders"][0]
        assert order["state"] == "SHADOW"
        expected_time = "13:50:05" if profile == "FREE_DELAYED" else "13:35:05"
        assert expected_time in order["decision_time"]
        assert Decimal(order["quantity"]) <= 100  # fixture's explicit liquidity capacity
        plan = next(plan for plan in result["plans"] if plan["id"] == order["plan_id"])
        assert plan["alternative"]["instrument"]["symbol"] == "TEST_STOCK"
        assert any(event["state"] == "INVALIDATED" for event in result["plan_events"])
    else:
        assert any(report["outcome"] == "NO_TRADE" for report in result["suitability_reports"])


def test_t38_delayed_fixture_never_uses_historical_signal_as_submission_time(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    fixture["config"]["database_path"] = str(tmp_path / "delayed.sqlite3")
    result = replay_fixture(fixture)
    order = result["orders"][0]
    decision = datetime.fromisoformat(order["decision_time"])
    assert decision.hour == 13 and decision.minute == 50
    assert result["fills"] == []
    assert result["execution_model"].startswith("decision-only")
