"""Causal operational/economic evaluation of explicitly scoped session records.

Actual-fill P&L already contains spread and slippage. These are reported execution
diagnostics, while fees and separately allocated operating costs enter the totals.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EquityObservation(BaseModel):
    """Observed equity and cumulative external cashflow since the sample began."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    observed_at: datetime
    equity: Decimal = Field(ge=0)
    cumulative_external_cashflows: Decimal = Decimal(0)

    @field_validator("observed_at")
    @classmethod
    def aware_time(cls, value):
        if value.utcoffset() is None:
            raise ValueError("equity observation timestamp must be timezone aware")
        return value


class EvaluationSample(BaseModel):
    """One account/profile/strategy/period observation; never an execution input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    sample_id: str
    account_id: str
    hypothetical: bool
    data_profile: Literal["FREE_DELAYED", "SUBSCRIBED_REALTIME"]
    strategy_id: str
    selection_method: Literal["fixed_watchlist", "close_gauss", "pre_gauss"]
    period_start: datetime
    period_end: datetime
    observation_wall_time: datetime | None = None
    signal_asof: datetime | None = None
    partition: Literal["development", "validation", "holdout", "forward"]
    evidence_kind: Literal["actual_fills", "simulation", "decision_only"]
    initial_equity: Decimal = Field(gt=0, allow_inf_nan=False)
    ending_equity: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    external_cashflows: Decimal = Field(default=Decimal(0), allow_inf_nan=False)
    gross_trading_pnl: Decimal | None = Field(default=None, allow_inf_nan=False)
    fees: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    operating_costs: tuple[tuple[str, Decimal], ...] = ()
    cost_allocation_method: str = "explicit per-account cost IDs"
    candidates: int = Field(default=0, ge=0)
    suitability_rejections: int = Field(default=0, ge=0)
    decisions: int = Field(default=0, ge=0)
    submitted_orders: int = Field(default=0, ge=0)
    filled_orders: int = Field(default=0, ge=0)
    no_trade_sessions: int = Field(default=0, ge=0)
    incidents: int = Field(default=0, ge=0)
    reconciliation_gaps: int = Field(default=0, ge=0)
    risk_limit_breaches: int = Field(default=0, ge=0)
    residual_position_incidents: int = Field(default=0, ge=0)
    uptime_seconds: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    observed_lag_seconds: tuple[Decimal, ...] = ()
    quote_age_seconds: tuple[Decimal, ...] = ()
    coverage_expected_records: int | None = Field(default=None, ge=0)
    coverage_received_records: int | None = Field(default=None, ge=0)
    source_coverage_expected: int | None = Field(default=None, ge=0)
    source_coverage_received: int | None = Field(default=None, ge=0)
    peak_backlog_events: int | None = Field(default=None, ge=0)
    recovery_attempts: int | None = Field(default=None, ge=0)
    successful_recoveries: int | None = Field(default=None, ge=0)
    alternatives_assessed: int | None = Field(default=None, ge=0)
    validation_reversals: int | None = Field(default=None, ge=0)
    cancelled_orders: int | None = Field(default=None, ge=0)
    replaced_orders: int | None = Field(default=None, ge=0)
    filled_position_groups: int | None = Field(default=None, ge=0)
    fill_latency_seconds: tuple[Decimal, ...] = ()
    spread_at_decision_bps: tuple[Decimal, ...] = ()
    slippage_bps: tuple[Decimal, ...] = ()
    slippage_benchmark: str | None = None
    peak_exposure: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    realized_pnl: Decimal | None = Field(default=None, allow_inf_nan=False)
    unrealized_pnl: Decimal | None = Field(default=None, allow_inf_nan=False)
    equity_observations: tuple[EquityObservation, ...] = ()
    minimum_feasible_quantity: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    capacity_quantity: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @field_validator("period_start", "period_end", "observation_wall_time", "signal_asof")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("evaluation timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> EvaluationSample:
        if self.period_end <= self.period_start:
            raise ValueError("evaluation period must have positive duration")
        if self.filled_orders > self.submitted_orders:
            raise ValueError("filled order count exceeds submitted order count")
        if self.evidence_kind == "decision_only" and self.gross_trading_pnl is not None:
            raise ValueError("decision-only evidence cannot claim trading P&L")
        if self.evidence_kind == "decision_only" and self.ending_equity is not None:
            raise ValueError("decision-only evidence cannot imply a trading equity curve")
        if self.evidence_kind == "decision_only" and (self.filled_orders or self.fill_latency_seconds):
            raise ValueError("decision-only evidence cannot assert fills")
        if self.signal_asof is not None and (
            self.observation_wall_time is None or self.signal_asof > self.observation_wall_time
        ):
            raise ValueError("signal as-of requires a current or later observation wall time")
        cost_ids: set[str] = set()
        for cost_id, cost in self.operating_costs:
            if not cost_id or cost_id in cost_ids:
                raise ValueError("operating cost IDs must be nonempty and unique")
            if not cost.is_finite() or cost < 0:
                raise ValueError("operating costs must be finite and nonnegative")
            cost_ids.add(cost_id)
        for value in (
            *self.observed_lag_seconds,
            *self.fill_latency_seconds,
            *self.quote_age_seconds,
            *self.spread_at_decision_bps,
        ):
            if not value.is_finite() or value < 0:
                raise ValueError("latencies must be finite and nonnegative")
        duration = Decimal(str((self.period_end - self.period_start).total_seconds()))
        if self.uptime_seconds is not None and self.uptime_seconds > duration:
            raise ValueError("uptime exceeds the observation period")
        if any(not value.is_finite() for value in self.slippage_bps):
            raise ValueError("slippage observations must be finite")
        if self.slippage_bps and not self.slippage_benchmark:
            raise ValueError("slippage observations require a declared recorded benchmark")
        if self.coverage_received_records is not None:
            if (
                self.coverage_expected_records is None
                or self.coverage_received_records > self.coverage_expected_records
            ):
                raise ValueError("coverage needs a known expected count and cannot exceed it")
        if self.successful_recoveries is not None:
            if (
                self.recovery_attempts is None
                or self.successful_recoveries > self.recovery_attempts
            ):
                raise ValueError("successful recoveries cannot exceed known recovery attempts")
        if self.source_coverage_received is not None and (
            self.source_coverage_expected is None
            or self.source_coverage_received > self.source_coverage_expected
        ):
            raise ValueError("source coverage requires a valid expected count")
        if self.cancelled_orders is not None and self.cancelled_orders > self.submitted_orders:
            raise ValueError("cancelled order count exceeds submitted orders")
        if (
            self.filled_position_groups is not None
            and self.filled_position_groups > self.filled_orders
        ):
            raise ValueError("filled groups exceed orders with fills")
        timestamps = [mark.observed_at for mark in self.equity_observations]
        if timestamps != sorted(set(timestamps)):
            raise ValueError("equity observations must have unique chronological timestamps")
        if any(not self.period_start <= stamp <= self.period_end for stamp in timestamps):
            raise ValueError("equity observations must fall inside the evaluation period")
        if self.evidence_kind == "decision_only" and (
            self.equity_observations
            or self.realized_pnl is not None
            or self.unrealized_pnl is not None
        ):
            raise ValueError("decision-only evidence cannot assert a profit/equity history")
        return self


def _mean(values: tuple[Decimal, ...]) -> str | None:
    return str(sum(values) / len(values)) if values else None


def evaluate_sample(sample: EvaluationSample) -> dict:
    """Report scope and costs without imputing missing fills or probabilities."""
    costs = sum((amount for _, amount in sample.operating_costs), Decimal(0))
    net_trading = None
    pnl_basis = "unavailable"
    if sample.gross_trading_pnl is not None:
        net_trading = sample.gross_trading_pnl - sample.fees
        pnl_basis = "gross_trading_pnl_minus_fees"
    elif sample.ending_equity is not None:
        # Broker equity already includes trading fees and external cashflows.
        net_trading = sample.ending_equity - sample.initial_equity - sample.external_cashflows
        pnl_basis = "equity_change_minus_external_cashflows_fees_already_in_equity"
    all_in = net_trading - costs if net_trading is not None else None
    intentional_delay = Decimal(900 if sample.data_profile == "FREE_DELAYED" else 0)
    additional_lag = tuple(
        max(Decimal(0), lag - intentional_delay) for lag in sample.observed_lag_seconds
    )
    duration = Decimal(str((sample.period_end - sample.period_start).total_seconds()))
    drawdown = None
    drawdown_dollars = None
    if sample.equity_observations:
        peak = sample.initial_equity
        drawdown = Decimal(0)
        drawdown_dollars = Decimal(0)
        for mark in sample.equity_observations:
            adjusted = mark.equity - mark.cumulative_external_cashflows
            peak = max(peak, adjusted)
            loss = peak - adjusted
            drawdown_dollars = max(drawdown_dollars, loss)
            drawdown = max(drawdown, loss / peak)
    return {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "account_id": sample.account_id,
        "hypothetical": sample.hypothetical,
        "execution_eligible": False,
        "data_profile": sample.data_profile,
        "strategy_id": sample.strategy_id,
        "selection_method": sample.selection_method,
        "partition": sample.partition,
        "period_start": sample.period_start.isoformat(),
        "period_end": sample.period_end.isoformat(),
        "observation_wall_time": sample.observation_wall_time.isoformat()
        if sample.observation_wall_time is not None else None,
        "signal_asof": sample.signal_asof.isoformat() if sample.signal_asof is not None else None,
        "evidence_kind": sample.evidence_kind,
        "initial_equity": str(sample.initial_equity),
        "gross_trading_pnl": str(sample.gross_trading_pnl)
        if sample.gross_trading_pnl is not None else None,
        "fees": str(sample.fees),
        "net_trading_pnl": str(net_trading) if net_trading is not None else None,
        "pnl_basis": pnl_basis,
        "operating_costs": str(costs),
        "operating_cost_fraction": str(costs / sample.initial_equity),
        "cost_allocation_method": sample.cost_allocation_method,
        "cost_ids": [cost_id for cost_id, _ in sample.operating_costs],
        "all_in_pnl": str(all_in) if all_in is not None else None,
        "all_in_return": str(all_in / sample.initial_equity) if all_in is not None else None,
        "fill_rate": str(Decimal(sample.filled_orders) / sample.submitted_orders)
        if sample.submitted_orders
        else None,
        "mean_fill_latency_seconds": _mean(sample.fill_latency_seconds),
        "intentional_delay_seconds": str(intentional_delay),
        "mean_observed_lag_seconds": _mean(sample.observed_lag_seconds),
        "mean_additional_lag_seconds": _mean(additional_lag),
        "uptime_fraction": str(sample.uptime_seconds / duration)
        if sample.uptime_seconds is not None
        else None,
        "mean_quote_age_seconds": _mean(sample.quote_age_seconds),
        "coverage_fraction": str(
            Decimal(sample.coverage_received_records) / sample.coverage_expected_records
        )
        if sample.coverage_received_records is not None and sample.coverage_expected_records
        else None,
        "peak_backlog_events": sample.peak_backlog_events,
        "source_coverage_fraction": str(
            Decimal(sample.source_coverage_received) / sample.source_coverage_expected
        )
        if sample.source_coverage_received is not None and sample.source_coverage_expected
        else None,
        "recovery_attempts": sample.recovery_attempts,
        "successful_recoveries": sample.successful_recoveries,
        "recovery_success_fraction": str(
            Decimal(sample.successful_recoveries) / sample.recovery_attempts
        )
        if sample.successful_recoveries is not None and sample.recovery_attempts
        else None,
        "alternatives_assessed": sample.alternatives_assessed,
        "validation_reversals": sample.validation_reversals,
        "cancelled_orders": sample.cancelled_orders,
        "replaced_orders": sample.replaced_orders,
        "cancel_rate": str(Decimal(sample.cancelled_orders) / sample.submitted_orders)
        if sample.cancelled_orders is not None and sample.submitted_orders
        else None,
        "replacements_per_submitted_order": str(
            Decimal(sample.replaced_orders) / sample.submitted_orders
        )
        if sample.replaced_orders is not None and sample.submitted_orders
        else None,
        "mean_spread_at_decision_bps": _mean(sample.spread_at_decision_bps),
        "mean_slippage_bps": _mean(sample.slippage_bps),
        "slippage_benchmark": sample.slippage_benchmark,
        "slippage_convention": "positive = adverse execution relative to the declared benchmark",
        "peak_exposure": str(sample.peak_exposure) if sample.peak_exposure is not None else None,
        "peak_exposure_fraction": str(sample.peak_exposure / sample.initial_equity)
        if sample.peak_exposure is not None
        else None,
        "realized_pnl": str(sample.realized_pnl) if sample.realized_pnl is not None else None,
        "unrealized_pnl": str(sample.unrealized_pnl) if sample.unrealized_pnl is not None else None,
        "max_drawdown_fraction": str(drawdown) if drawdown is not None else None,
        "max_drawdown_dollars": str(drawdown_dollars) if drawdown_dollars is not None else None,
        "drawdown_method": "observed equity less cumulative external cashflows; no interpolation",
        "operating_cost_per_candidate": str(costs / sample.candidates)
        if sample.candidates
        else None,
        "operating_cost_per_filled_group": str(costs / sample.filled_position_groups)
        if sample.filled_position_groups
        else None,
        "candidates": sample.candidates,
        "suitability_rejections": sample.suitability_rejections,
        "decisions": sample.decisions,
        "no_trade_sessions": sample.no_trade_sessions,
        "incidents": sample.incidents,
        "reconciliation_gaps": sample.reconciliation_gaps,
        "risk_limit_breaches": sample.risk_limit_breaches,
        "residual_position_incidents": sample.residual_position_incidents,
        "minimum_feasible_quantity": str(sample.minimum_feasible_quantity)
        if sample.minimum_feasible_quantity is not None
        else None,
        "capacity_quantity": str(sample.capacity_quantity)
        if sample.capacity_quantity is not None
        else None,
        "assumptions": list(sample.assumptions),
        "limitations": list(sample.limitations),
        "evidence_ids": list(sample.evidence_ids),
        "live_authorized": False,
    }


def compare_evaluations(samples: list[EvaluationSample]) -> dict:
    """Preserve every experiment and scope; do not promote the best retrospective row."""
    identifiers = [sample.sample_id for sample in samples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be unique")
    partitions: dict[tuple, list[EvaluationSample]] = defaultdict(list)
    for sample in samples:
        key = (
            sample.account_id, sample.hypothetical, sample.data_profile,
            sample.strategy_id, sample.selection_method,
        )
        partitions[key].append(sample)
    for records in partitions.values():
        ordered = sorted(records, key=lambda item: item.period_start)
        partition_ends: dict[str, datetime] = {}
        for current in ordered:
            if any(end > current.period_start for partition, end in partition_ends.items()
                   if partition != current.partition):
                raise ValueError("development, validation and holdout periods must not overlap")
            partition_ends[current.partition] = max(
                current.period_end, partition_ends.get(current.partition, current.period_end)
            )
        partition_order = {"development": 0, "validation": 1, "holdout": 2, "forward": 3}
        sequence = [partition_order[item.partition] for item in ordered]
        if sequence != sorted(sequence):
            raise ValueError("evaluation partitions must follow chronological order")
    return {
        "schema_version": 1,
        "execution_eligible": False,
        "live_authorized": False,
        "dimensions": [
            "data_profile",
            "account_id",
            "hypothetical",
            "strategy_id",
            "selection_method",
            "partition",
            "period",
        ],
        "samples": [evaluate_sample(sample) for sample in samples],
        "selection_value": "requires comparable fixed_watchlist, close_gauss and pre_gauss observations",
        "economic_viability": "requires causal fills, declared costs and untouched holdout/forward observations",
    }
