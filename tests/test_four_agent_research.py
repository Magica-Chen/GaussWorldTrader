from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.runtime.evidence import EvidenceService
from src.runtime.models import AccountProfile, RiskPolicy, RuntimeConfig
from src.runtime.research import ModelPricing, ResearchJobRunner, worker_environment
from src.runtime.store import Store


NOW = datetime(2026, 9, 8, 20, 45, tzinfo=timezone.utc)


def context(tmp_path, *, paid=False, **policy):
    store = Store(tmp_path / "research.db")
    cfg = RuntimeConfig(account_id="explicit-research-account", paid_model_calls_enabled=paid,
        policy=RiskPolicy(research_cost_cap="5", **policy))
    account = AccountProfile(account_id=cfg.account_id, environment="paper", equity="10000",
        cash="10000", buying_power="10000", currency="USD", mandate_id=cfg.policy.id,
        observed_at=NOW)
    evidence = EvidenceService(store, cfg, "paper:" + cfg.account_id)
    snapshot = evidence.freeze("fixture-session", account, NOW)
    return store, cfg, account, snapshot


def pricing():
    return ModelPricing(id="explicit-fixture-pricing", model="fixture-model", input_usd_per_million="1",
        output_usd_per_million="2", verified_by="fixture-operator", source_reference="synthetic-prices-only",
        expires_at=NOW+timedelta(days=1))


def success(payload, deadline, env):
    return {"annotation": {"outcome": "INSUFFICIENT_EVIDENCE", "summary": "Fixture analysis.",
        "source_ids": [], "uncertainties": ["Synthetic fixture"], "experiment_hypothesis": None},
        "input_tokens": 10, "output_tokens": 10}


def test_t23_subprocess_worker_has_no_broker_environment_and_completes(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "synthetic-parent-secret")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "synthetic-notification-secret")
    environment = worker_environment(paid=False)
    assert "ALPACA_API_KEY" not in environment and "SLACK_WEBHOOK_URL" not in environment
    assert "OPENAI_API_KEY" not in environment
    store, cfg, account, snapshot = context(tmp_path)
    outcome = ResearchJobRunner(store, cfg, "paper:" + cfg.account_id, clock=lambda:NOW).run(
        snapshot, account, "CloseGauss", "fixture-session", timeout_seconds=10)
    assert outcome.state == "COMPLETED", outcome.reasons
    assert outcome.annotation.outcome == "INSUFFICIENT_EVIDENCE"
    assert not outcome.execution_eligible
    store.close()


def test_t23_paid_disabled_blocks_before_worker(tmp_path):
    store, cfg, account, snapshot = context(tmp_path)
    calls = []
    runner = ResearchJobRunner(store, cfg, "paper:"+cfg.account_id, clock=lambda:NOW,
                              executor=lambda *args: calls.append(args))
    result = runner.run(snapshot, account, "CloseGauss", "fixture-session", model="fixture-model")
    assert result.state == "BLOCKED" and result.reasons == ("PAID_MODEL_CALLS_DISABLED",)
    assert calls == []
    store.close()


def test_t23_unknown_pricing_blocks_before_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-used")
    store, cfg, account, snapshot = context(tmp_path, paid=True)
    result = ResearchJobRunner(store, cfg, "paper:"+cfg.account_id, clock=lambda:NOW,
        executor=success).run(snapshot, account, "CloseGauss", "fixture-session", model="fixture-model")
    assert result.state == "BLOCKED"
    assert not store.projected("research_spend")
    store.close()


def test_t57_paid_budget_reserves_then_settles_actual_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-used")
    store, cfg, account, snapshot = context(tmp_path, paid=True)
    result = ResearchJobRunner(store, cfg, "paper:"+cfg.account_id, clock=lambda:NOW,
        executor=success).run(snapshot, account, "CloseGauss", "fixture-session", model="fixture-model", pricing=pricing())
    assert result.state == "COMPLETED", result.reasons
    spending = store.projected("research_spend")[0]
    assert Decimal(spending["charged_usd"]) == Decimal("0.00003")
    assert spending["state"] == "SETTLED"
    assert Decimal(spending["total_cap"]) == 1  # min($5, explicit equity * 0.0001)
    assert Decimal(spending["role_cap"]) == Decimal("0.60")
    store.close()


def test_t23_timeout_keeps_potential_paid_charge_reserved(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-used")
    store, cfg, account, snapshot = context(tmp_path, paid=True)
    def timeout(*args): raise TimeoutError("RESEARCH_DEADLINE_EXCEEDED")
    result = ResearchJobRunner(store, cfg, "paper:"+cfg.account_id, clock=lambda:NOW,
        executor=timeout).run(snapshot, account, "CloseGauss", "fixture-session", model="fixture-model", pricing=pricing())
    spending = store.projected("research_spend")[0]
    assert result.state == "FAILED"
    assert spending["state"] == "UNKNOWN_CHARGE"
    assert spending["charged_usd"] == spending["amount_reserved"]
    store.close()


def test_t50_old_snapshot_can_use_separate_current_budget_account(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-used")
    store, cfg, account, snapshot = context(tmp_path, paid=True)
    frozen = account.model_copy(update={"observed_at": NOW-timedelta(minutes=30)})
    fresh = account.model_copy(update={"id": "fresh-budget-account"})
    result = ResearchJobRunner(store, cfg, "paper:"+cfg.account_id, clock=lambda:NOW,
        executor=success).run(snapshot, frozen, "CloseGauss", "fixture-session", budget_account=fresh,
        model="fixture-model", pricing=pricing())
    assert result.state == "COMPLETED", result.reasons
    assert store.projected("research_spend")[0]["account_profile_id"] == "fresh-budget-account"
    store.close()


def test_t22_worker_cannot_cite_future_or_outside_snapshot(tmp_path):
    store, cfg, account, snapshot = context(tmp_path)
    def forged(*args):
        result=success(*args)
        result["annotation"].update(outcome="SUPPORT", source_ids=["tomorrows-news"])
        return result
    result=ResearchJobRunner(store,cfg,"paper:"+cfg.account_id,clock=lambda:NOW,executor=forged).run(
        snapshot,account,"CloseGauss","fixture-session")
    assert result.state == "FAILED" and result.reasons == ("OUT_OF_SNAPSHOT_SOURCE",)
    store.close()


def test_t11_experimental_output_is_queued_without_strategy_approval(tmp_path):
    store, cfg, account, snapshot = context(tmp_path)
    def experiment(*args):
        result=success(*args)
        result["annotation"].update(outcome="EXPERIMENT", experiment_hypothesis="Test another indicator.")
        return result
    result=ResearchJobRunner(store,cfg,"paper:"+cfg.account_id,clock=lambda:NOW,executor=experiment).run(
        snapshot,account,"CloseGauss","fixture-session")
    assert result.state == "COMPLETED"
    assert store.list("strategy_experiments")[0]["execution_eligible"] is False
    assert store.list("strategy_approvals") == []
    store.close()


def test_t58_cache_keys_include_actual_account_constraints(tmp_path):
    store, cfg, account, snapshot = context(tmp_path)
    calls=[]
    def count(*args): calls.append(1); return success(*args)
    runner=ResearchJobRunner(store,cfg,"paper:"+cfg.account_id,clock=lambda:NOW,executor=count)
    initial=runner.run(snapshot,account,"CloseGauss","fixture-session")
    cached=runner.run(snapshot,account,"CloseGauss","fixture-session")
    changed=runner.run(snapshot,account.model_copy(update={"cash":Decimal("0")}),"CloseGauss","fixture-session")
    assert initial.input_hash == cached.input_hash and cached.cached
    assert changed.input_hash != cached.input_hash
    assert len(calls)==2
    store.close()


def test_t34_unknown_model_order_field_fails_schema(tmp_path):
    store,cfg,account,snapshot=context(tmp_path)
    def injection(*args):
        result=success(*args)
        result["annotation"]["submit_order"]={"symbol":"TEST_STOCK","quantity":100000}
        return result
    result=ResearchJobRunner(store,cfg,"paper:"+cfg.account_id,clock=lambda:NOW,executor=injection).run(
        snapshot,account,"CloseGauss","fixture-session")
    assert result.state == "FAILED"
    assert store.list("order_intents") == []
    store.close()
