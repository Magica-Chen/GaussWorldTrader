"""Offline regressions for the plan's existing execution corrections F01–F03/F13."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.strategy.base import ActionPlan
from src.trade.engine.execution import ExecutionContext, ExecutionEngine


@pytest.fixture
def context():
    return ExecutionContext({}, {}, 500.0, 500.0, 1000.0, False, False, False, "CASH")


@pytest.fixture
def executor():
    engine = Mock(allow_fractional=False)
    engine.api.get_option_contract.return_value = SimpleNamespace(size="100", tradable=True)
    return ExecutionEngine(engine, "stock", account_manager=Mock())


def plan(action="BUY", price=100.0, **kwargs):
    return ActionPlan("TEST", action, price, kwargs.pop("stop_loss", 90.0), 120.0, **kwargs)


@pytest.mark.parametrize("value", [None, "NaN", "Infinity", "bad", 0])
def test_missing_invalid_account_is_not_fabricated(executor, value):
    executor.trading_engine.get_account_info.return_value = {"portfolio_value": value}
    executor.account_manager.get_account.return_value = {}
    executor.account_manager.get_account_configurations.return_value = {}
    loaded = executor.load_context()
    assert loaded.portfolio_value == 0
    assert executor.build_decision(plan(), loaded, {}, 0.1, 100) is None


def test_override_still_respects_capital_and_allocation(executor, context):
    decision = executor.build_decision(plan(), context, {}, 0.1, 100, override_qty=50)
    assert decision.quantity == 1


def test_zero_balance_is_not_replaced_by_a_different_account_read(executor):
    executor.trading_engine.get_account_info.return_value = {
        "portfolio_value": 0, "buying_power": 0, "cash": 0}
    executor.account_manager.get_account.return_value = {
        "portfolio_value": "10000", "buying_power": "10000", "cash": "10000"}
    executor.account_manager.get_account_configurations.return_value = {}
    loaded = executor.load_context()
    assert loaded.portfolio_value == loaded.cash == loaded.buying_power == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_override_cannot_enter(executor, context, value):
    assert executor.build_decision(plan(), context, {}, 0.1, 100, override_qty=value) is None


def test_stop_loss_budget_separate_from_capital_allocation(executor, context):
    decision = executor.build_decision(
        plan(stop_loss=90), context, {}, 0.5, 100, planned_loss_pct=0.02
    )
    assert decision.quantity == 2


def test_option_premium_uses_verified_contract_multiplier(executor, context):
    executor.asset_type = "option"
    assert executor.build_decision(plan("BUY_TO_OPEN", 2), context, {}, 0.1, 2) is None
    decision = executor.build_decision(plan("BUY_TO_OPEN", 2), context, {}, 0.5, 2)
    assert decision.quantity == 2


def test_explicit_sell_to_close_never_opens_short(executor, context):
    executor.allow_sell_to_open = True
    permissive = replace(context, margin_enabled=True, shorting_enabled=True)
    assert executor.build_decision(plan("SELL_TO_CLOSE"), permissive, {}, 0.1, 100) is None


def test_confirmed_exit_survives_invalid_account(executor, context):
    context = replace(context, portfolio_value=0, cash=0, buying_power=0)
    decision = executor.build_decision(
        plan("SELL_TO_CLOSE"), context, {"side": "long", "qty": 2}, 0.1, 100
    )
    assert decision.quantity == 2


def test_limit_price_does_not_exceed_buy_bound(executor, context):
    decision = executor.build_decision(plan(price=100.001), context, {}, 0.2, 100)
    assert decision.limit_price <= 100.001


def test_final_rounded_price_used_for_affordability(executor, context):
    decision = executor.build_decision(plan(price=125), context, {}, 0.5, 100)
    assert decision.quantity == 4
