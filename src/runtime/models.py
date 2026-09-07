"""Versioned, immutable contracts for the account-scoped session service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: int = 1
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def aware_times(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                if value.utcoffset() is None:
                    raise ValueError(f"{name} must be timezone aware")
                object.__setattr__(self, name, value.astimezone(timezone.utc))
        return self


class DataProfile(StrEnum):
    FREE_DELAYED = "FREE_DELAYED"
    SUBSCRIBED_REALTIME = "SUBSCRIBED_REALTIME"


class ExecutionMode(StrEnum):
    REPLAY = "replay"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class RiskPolicy(Record):
    id: str = "operator-mandate-v1"
    max_new_entry_groups_per_session: int = Field(default=3, ge=0)
    max_open_position_groups: int = Field(default=5, ge=0)
    max_entry_order_attempts_per_session: int = Field(default=6, ge=0)
    max_replacements_per_intent: int = Field(default=2, ge=0)
    max_capital_allocation_pct: Decimal = Field(default=Decimal(".10"), gt=0, le=1)
    planned_loss_per_trade_pct: Decimal = Field(default=Decimal(".01"), gt=0, le=1)
    max_option_structural_loss_pct: Decimal = Field(default=Decimal(".01"), gt=0, le=1)
    daily_loss_pause_pct: Decimal = Field(default=Decimal(".02"), gt=0, le=1)
    max_portfolio_allocation_pct: Decimal = Field(default=Decimal(".50"), gt=0, le=1)
    cash_buffer: Decimal = Field(default=Decimal("0"), ge=0)
    execution_cost_buffer: Decimal = Field(default=Decimal(".02"), ge=0)
    operating_cost_per_session: Decimal = Field(default=Decimal("0"), ge=0)
    allow_leverage: bool = False
    options_enabled: bool = False
    spreads_enabled: bool = False
    allow_overnight_positions: bool = False
    account_max_age_seconds: int = Field(default=30, gt=0)
    quote_max_age_seconds: int = Field(default=5, gt=0)
    max_quote_spread_pct: Decimal = Field(default=Decimal(".02"), gt=0)
    entry_start_minutes: int = Field(default=5, ge=0)
    entry_cutoff_minutes: int = Field(default=30, ge=0)
    closing_minutes: int = Field(default=15, ge=0)
    pre_minutes: int = Field(default=90, ge=0)
    research_seconds: int = Field(default=300, gt=0)
    research_cost_cap: Decimal = Field(default=Decimal("0"), ge=0)
    research_equity_fraction: Decimal = Field(default=Decimal(".0001"), ge=0)


class RuntimeConfig(Record):
    enabled: bool = True
    live_trading_enabled: bool = False
    account_id_allowlist: tuple[str, ...] = ()
    strategy_allowlist: tuple[str, ...] = ()
    calendar_timezone: Literal["America/New_York"] = "America/New_York"
    display_timezone: str = "Europe/London"
    paid_model_calls_enabled: bool = False
    research_model: str | None = None
    research_pricing_id: str | None = None
    research_max_output_tokens: int = Field(default=1024, ge=16, le=8192)
    research_max_retries: int = Field(default=0, ge=0, le=2)
    research_max_parallel_jobs: int = Field(default=2, ge=1, le=4)
    research_role_budget_shares: tuple[Decimal, Decimal, Decimal, Decimal] = (
        Decimal(".10"),
        Decimal(".60"),
        Decimal(".20"),
        Decimal(".10"),
    )
    research_offset_minutes: int = Field(default=45, ge=0)
    pre_review_interval_minutes: int = Field(default=10, gt=0)
    signal_timeframe: Literal["1Min", "5Min", "15Min", "1Hour", "1Day"] = "5Min"
    scan_universe_limit: int = Field(default=200, gt=0)
    active_candidate_limit: int = Field(default=5, ge=0)
    reserve_candidate_limit: int = Field(default=5, ge=0)
    free_poll_interval_seconds: int = Field(default=60, ge=1)
    market_requests_per_minute: int = Field(default=120, ge=2)
    news_reserved_requests_per_minute: int = Field(default=20, ge=1)
    stock_quote_max_age_seconds: int = Field(default=5, gt=0)
    option_quote_max_age_seconds: int = Field(default=2, gt=0)
    capital_scenario_file: str | None = None
    payload_directory: str = "results/gauss/evidence"
    database_path: str = "results/gauss/session.sqlite3"
    account_id: str = ""
    environment: Literal["paper", "live"] = "paper"
    execution_mode: ExecutionMode = ExecutionMode.SHADOW
    data_profile: DataProfile = DataProfile.FREE_DELAYED
    data_policy_version: str = "data-policy-v1"
    deployment_version: str = "four-agents-v1"
    symbols: tuple[str, ...] = ()
    poll_seconds: float = Field(default=10, ge=1)
    entitlement_boundary_buffer: int = Field(default=2, ge=0)
    maximum_additional_lag_seconds: int = Field(default=180, ge=0)
    policy: RiskPolicy = Field(default_factory=RiskPolicy)
    event_calendar_path: str | None = None
    control_token_env: str = "GAUSS_CONTROL_TOKEN"

    @model_validator(mode="after")
    def modes(self):
        from zoneinfo import ZoneInfo

        ZoneInfo(self.display_timezone)
        if self.news_reserved_requests_per_minute >= self.market_requests_per_minute:
            raise ValueError("news request reserve must be below the total market request budget")
        if (
            any(share < 0 for share in self.research_role_budget_shares)
            or sum(self.research_role_budget_shares) != 1
        ):
            raise ValueError("research role budget shares must be nonnegative and sum to one")
        if self.execution_mode == ExecutionMode.PAPER and self.environment != "paper":
            raise ValueError("paper execution requires paper environment")
        if self.execution_mode == ExecutionMode.LIVE and self.environment != "live":
            raise ValueError("live execution requires live environment")
        if self.execution_mode == ExecutionMode.LIVE and (
            not self.live_trading_enabled or not self.account_id_allowlist
        ):
            raise ValueError("live execution requires live_trading_enabled and account allowlist")
        return self


class TradingSession(Record):
    session_date: str
    calendar_version: str
    open: datetime
    close: datetime

    @model_validator(mode="after")
    def bounds(self):
        if self.open >= self.close:
            raise ValueError("invalid calendar session")
        return self

    def schedule(self, policy: RiskPolicy) -> dict[str, datetime]:
        start = min(self.open + timedelta(minutes=policy.entry_start_minutes), self.close)
        cutoff = max(start, self.close - timedelta(minutes=policy.entry_cutoff_minutes))
        return {
            "pre_start": self.open - timedelta(minutes=policy.pre_minutes),
            "entry_start": start,
            "entry_cutoff": cutoff,
            "closing_start": max(self.open, self.close - timedelta(minutes=policy.closing_minutes)),
            "open": self.open,
            "close": self.close,
        }


class MarketEvent(Record):
    symbol: str
    source: str
    source_version: str
    event_type: Literal["bar", "quote", "trade"]
    timeframe: Literal["1Min", "5Min", "15Min", "1Hour", "1Day"] = "1Min"
    effective_event_time: datetime
    received_at: datetime
    available_at: datetime
    feed: Literal["sip", "opra", "indicative", "iex"]
    quality_class: Literal["genuine", "indicative", "incomplete"]
    price: Decimal | None = Field(default=None, gt=0)
    open_price: Decimal | None = Field(default=None, gt=0)
    high_price: Decimal | None = Field(default=None, gt=0)
    low_price: Decimal | None = Field(default=None, gt=0)
    bid: Decimal | None = Field(default=None, ge=0)
    ask: Decimal | None = Field(default=None, gt=0)
    bid_size: Decimal | None = Field(default=None, ge=0)
    ask_size: Decimal | None = Field(default=None, ge=0)
    volume: Decimal = Field(default=Decimal("0"), ge=0)
    complete: bool = True
    interval_start: datetime | None = None

    @model_validator(mode="after")
    def quality(self):
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError("crossed quote")
        if self.interval_start and self.interval_start >= self.effective_event_time:
            raise ValueError("bar end must follow interval start")
        if self.feed == "indicative" and self.quality_class != "indicative":
            raise ValueError("indicative feed cannot become genuine")
        return self


class NewsEvent(Record):
    provider: str
    article_id: str
    source_version: str
    published_at: datetime
    updated_at: datetime
    received_at: datetime
    available_at: datetime
    symbols: tuple[str, ...] = ()
    headline: str
    content_reference: str = ""
    content_hash: str
    category: str = "unclassified"
    verified: bool = False
    blocking: bool = False
    cluster_id: str | None = None


class DataCapabilitySnapshot(Record):
    account_id: str
    environment: str
    endpoint: str
    feed: str
    outcome: Literal["AVAILABLE", "UNAVAILABLE", "UNKNOWN"]
    quality_class: str
    expires_at: datetime
    details: str = ""


class DataContext(Record):
    data_profile: DataProfile
    data_policy_version: str
    wall_time: datetime
    signal_as_of: datetime
    configured_delay_seconds: int
    watermarks: tuple[tuple[str, datetime], ...] = ()
    complete: bool = False
    execution_ready: bool = False
    blockers: tuple[str, ...] = ()


class PositionLeg(Record):
    account_id: str
    environment: str
    group_id: str
    symbol: str
    asset_type: Literal["stock", "option"]
    quantity: Decimal
    multiplier: Decimal = Field(default=Decimal("1"), gt=0)
    cost_basis: Decimal = Decimal("0")
    market_value: Decimal | None = None
    reconciled_at: datetime
    contract_id: str | None = None


class PositionGroup(Record):
    account_id: str
    environment: str
    plan_id: str | None = None
    strategy_id: str | None = None
    exit_policy_id: str
    legs: tuple[PositionLeg, ...]
    state: Literal[
        "PENDING_ENTRY",
        "PARTIALLY_OPEN",
        "OPEN",
        "EXIT_PENDING",
        "CLOSED",
        "RECONCILIATION_REQUIRED",
    ]
    unresolved_orders: tuple[str, ...] = ()
    stop_price: Decimal | None = None
    take_profit: Decimal | None = None
    close_at: datetime | None = None

    @property
    def flat(self):
        return all(leg.quantity == 0 for leg in self.legs) and not self.unresolved_orders


class AccountOperationalState(Record):
    """Current confirmed holdings when financial account fields are unavailable."""

    account_id: str
    environment: Literal["paper", "live"]
    positions: tuple[PositionLeg, ...]
    observed_at: datetime
    trading_blocked: bool = True


class AccountProfile(Record):
    account_id: str
    environment: Literal["paper", "live", "scenario"]
    hypothetical: bool = False
    equity: Decimal = Field(gt=0)
    cash: Decimal = Field(ge=0)
    buying_power: Decimal = Field(ge=0)
    options_buying_power: Decimal | None = Field(default=None, ge=0)
    currency: Literal["USD"]
    observed_at: datetime
    trading_blocked: bool = False
    fractional_allowed: bool = False
    shorting_allowed: bool = False
    options_level: int = Field(default=0, ge=0)
    margin_allowed: bool = False
    reserved_capital: Decimal = Field(default=Decimal("0"), ge=0)
    exposure: Decimal = Field(default=Decimal("0"), ge=0)
    out_of_scope_exposure: Decimal = Field(default=Decimal("0"), ge=0)
    out_of_scope_symbols: tuple[str, ...] = ()
    daily_pnl: Decimal | None = None
    positions: tuple[PositionLeg, ...] = ()
    mandate_id: str

    @model_validator(mode="after")
    def scope(self):
        if self.hypothetical != (self.environment == "scenario"):
            raise ValueError("hypothetical accounts must use scenario scope")
        return self


class OptionSelector(Record):
    structure: Literal["long_call", "long_put", "debit_call_vertical", "debit_put_vertical"]
    signal_strategy_id: str = "momentum"
    minimum_dte: int = Field(default=7, ge=1)
    maximum_dte: int = Field(default=30, ge=1)
    minimum_moneyness: Decimal = Field(default=Decimal(".95"), gt=0)
    maximum_moneyness: Decimal = Field(default=Decimal("1.05"), gt=0)
    maximum_width: Decimal = Field(default=Decimal("10"), gt=0)
    maximum_quote_skew_seconds: Decimal = Field(default=Decimal("1"), ge=0)
    minimum_displayed_size: int = Field(default=1, ge=1)
    max_contracts: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def bounds(self):
        if self.minimum_dte > self.maximum_dte or self.minimum_moneyness > self.maximum_moneyness:
            raise ValueError("invalid option selector bounds")
        return self


class StrategyApproval(Record):
    strategy_id: str
    strategy_version: str
    account_id: str
    environment: str
    approved_by: str
    profiles: tuple[DataProfile, ...]
    execution_modes: tuple[ExecutionMode, ...] = (ExecutionMode.SHADOW, ExecutionMode.REPLAY)
    asset_type: Literal["stock", "option"] = "stock"
    required_feed: Literal["sip", "opra"] = "sip"
    requires_current_quote: bool = True
    maximum_signal_latency_seconds: int = Field(gt=0)
    minimum_history: int = Field(default=30, ge=2)
    requires_news: bool = True
    requires_event_calendar: bool = True
    validation_reference: str
    validated_net_expectancy: Decimal | None = None
    parameter_json: str = "{}"
    expiry: datetime
    option_selector: OptionSelector | None = None


class Instrument(Record):
    symbol: str
    asset_type: Literal["stock", "option"] = "stock"
    multiplier: Decimal = Field(default=Decimal("1"), gt=0)
    quantity_increment: Decimal = Field(default=Decimal("1"), gt=0)
    tick_size: Decimal = Field(default=Decimal(".01"), gt=0)
    liquidity_capacity: Decimal = Field(ge=0)
    contract_id: str | None = None
    expiry: str | None = None
    underlying: str | None = None
    option_type: Literal["call", "put"] | None = None
    strike: Decimal | None = None
    standard_contract: bool = True
    tradable: bool = True


class Alternative(Record):
    strategy_id: str
    strategy_version: str
    instrument: Instrument
    entry_price: Decimal = Field(gt=0)
    stop_price: Decimal = Field(gt=0)
    expected_net_value: Decimal | None = None
    evidence_ids: tuple[str, ...] = ()
    estimated_fees: Decimal = Field(default=Decimal("0"), ge=0)
    short_instrument: Instrument | None = None
    underlying_entry_price: Decimal | None = Field(default=None, gt=0)
    underlying_stop_price: Decimal | None = Field(default=None, gt=0)

    @property
    def signal_symbol(self):
        return self.instrument.underlying or self.instrument.symbol

    @property
    def instruments(self):
        return (
            (self.instrument, self.short_instrument)
            if self.short_instrument
            else (self.instrument,)
        )


class SizingResult(Record):
    alternative_id: str
    feasible: bool
    quantity: Decimal = Field(default=Decimal("0"), ge=0)
    capital: Decimal = Field(default=Decimal("0"), ge=0)
    planned_loss: Decimal = Field(default=Decimal("0"), ge=0)
    reasons: tuple[str, ...] = ()
    binding_limit: str = ""
    minimum_equity_bound: Decimal | None = None


class AccountSuitabilityReport(Record):
    account_id: str
    account_profile_id: str
    hypothetical: bool
    data_profile: DataProfile
    policy_id: str
    alternatives: tuple[SizingResult, ...]
    selected_alternative_id: str | None = None
    outcome: str
    objective: str = "validated_net_expectancy_per_unit_planned_loss"
    ranking_scores: tuple[tuple[str, Decimal], ...] = ()
    operating_cost: Decimal = Decimal("0")
    cost_fraction_equity: Decimal = Decimal("0")
    uncertainty: tuple[str, ...] = ()
    refresh_conditions: tuple[str, ...] = (
        "account",
        "reservations",
        "quotes",
        "events",
        "policy",
        "profile",
    )


class Rule(Record):
    timeframe: Literal["1Min", "5Min", "15Min", "1Hour", "1Day"] = "1Min"
    rule_type: Literal[
        "price_crosses",
        "completed_bar_condition",
        "spread_within_limit",
        "no_blocking_event",
        "time_in_window",
    ]
    indicator: Literal["close", "price", "spread"] = "close"
    operator: Literal["above", "below", "crosses_above", "crosses_below", "lte"] = "above"
    value: Decimal | None = None


class ResearchSnapshot(Record):
    signal_timeframe: Literal["1Min", "5Min", "15Min", "1Hour", "1Day"] = "1Min"
    trading_sessions: tuple[TradingSession, ...] = ()
    target_session_id: str
    account_profile_id: str
    data_context: DataContext
    evidence_ids: tuple[str, ...]
    news_ids: tuple[str, ...] = ()
    manifest_hash: str
    strategy_approval_ids: tuple[str, ...] = ()


class Candidate(Record):
    symbol: str
    snapshot_id: str
    account_profile_id: str
    suitability_report_id: str
    alternative: Alternative
    rationale: str


class SessionPlan(Record):
    version: int = 1
    target_session_id: str
    snapshot_id: str
    candidate_id: str
    account_id: str
    environment: str
    account_profile_id: str
    suitability_report_id: str
    data_profile: DataProfile
    data_policy_version: str
    signal_as_of: datetime
    strategy_id: str
    strategy_version: str
    approval_id: str
    alternative: Alternative
    entry_rules: tuple[Rule, ...]
    invalidation_rules: tuple[Rule, ...] = ()
    max_buy_price: Decimal = Field(gt=0)
    valid_from: datetime
    entry_expires_at: datetime
    close_at: datetime
    risk_policy_id: str
    exit_policy_id: str
    execution_eligible: bool = True

    @model_validator(mode="after")
    def windows(self):
        if self.valid_from >= self.entry_expires_at or self.entry_expires_at > self.close_at:
            raise ValueError("invalid plan validity window")
        if self.environment == "scenario" and self.execution_eligible:
            raise ValueError("scenario plans cannot be execution eligible")
        return self


class ValidationRecord(Record):
    plan_id: str
    plan_version: int
    account_profile_id: str
    suitability_report_id: str
    signal_as_of: datetime
    event_watermark: int
    outcome: Literal[
        "ELIGIBLE", "DEFERRED", "REJECTED", "INVALIDATED", "ENTRY_EXPIRED", "REVIEW_REQUIRED"
    ]
    reasons: tuple[str, ...]


class OrderLeg(Record):
    instrument: Instrument
    side: Literal["buy", "sell"]
    position_intent: Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]
    ratio: int = Field(default=1, ge=1)


class OrderIntent(Record):
    account_id: str
    environment: Literal["paper", "live"]
    purpose: Literal["open", "close"]
    plan_id: str | None = None
    plan_version: int | None = None
    group_id: str | None = None
    legs: tuple[OrderLeg, ...]
    requested_quantity: Decimal | None = Field(default=None, gt=0)
    limit_price: Decimal = Field(gt=0)
    limit_effect: Literal["debit", "credit"] = "debit"
    maximum_buy_price: Decimal | None = Field(default=None, gt=0)
    minimum_sell_price: Decimal | None = Field(default=None, ge=0)
    expires_at: datetime
    reason: str

    @model_validator(mode="after")
    def intent(self):
        if not self.legs or len(self.legs) > 2:
            raise ValueError("only single instruments and validated verticals supported")
        if self.purpose == "open" and not self.plan_id:
            raise ValueError("entry requires plan")
        if self.purpose == "close" and not self.group_id:
            raise ValueError("exit requires managed group")
        for leg in self.legs:
            if not leg.position_intent.startswith(leg.side + "_to_"):
                raise ValueError("side and position intent disagree")
            if not leg.position_intent.endswith("_" + self.purpose):
                raise ValueError("leg position intent differs from order purpose")
        return self


class RiskDecision(Record):
    intent_id: str
    approved: bool
    policy_id: str
    account_profile_id: str
    suitability_report_id: str | None = None
    data_policy_version: str
    event_watermark: int
    approved_quantity: Decimal = Field(default=Decimal("0"), ge=0)
    capital_reserved: Decimal = Field(default=Decimal("0"), ge=0)
    loss_reserved: Decimal = Field(default=Decimal("0"), ge=0)
    reasons: tuple[str, ...]
    expires_at: datetime


class AgentRun(Record):
    role: Literal["PostGauss", "CloseGauss", "PreGauss", "LiveGauss"]
    job_id: str
    input_hash: str
    account_profile_id: str | None
    data_profile: DataProfile
    state: Literal["IDLE", "RUNNING", "DEGRADED", "FAILED", "COMPLETED"]
    completed_at: datetime | None = None
    output_ids: tuple[str, ...] = ()
    failure: str | None = None
    cost: Decimal = Decimal("0")
    tokens: int = 0
