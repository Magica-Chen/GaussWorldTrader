"""Shared deterministic capital feasibility; research never grants execution authority."""

from decimal import Decimal, ROUND_FLOOR
from .models import AccountProfile, AccountSuitabilityReport, SizingResult


def units(amount, increment):
    return (amount / increment).to_integral_value(rounding=ROUND_FLOOR) * increment


class AccountSuitabilityService:
    def __init__(self, policy):
        self.policy = policy

    def size(self, account, alternative, *, reserved=Decimal("0")):
        policy, instrument = self.policy, alternative.instrument
        reasons = []
        if account.trading_blocked:
            reasons.append("ACCOUNT_RESTRICTED")
        if not instrument.tradable:
            reasons.append("INSTRUMENT_NOT_TRADABLE")
        if account.mandate_id != policy.id:
            reasons.append("MANDATE_MISMATCH")
        increment = instrument.quantity_increment
        if instrument.asset_type == "stock":
            if increment < 1 and not account.fractional_allowed:
                reasons.append("FRACTIONAL_PERMISSION_REQUIRED")
            if instrument.multiplier != 1:
                reasons.append("INVALID_STOCK_MULTIPLIER")
            loss = alternative.entry_price - alternative.stop_price + policy.execution_cost_buffer
            if alternative.stop_price >= alternative.entry_price:
                reasons.append("INVALID_STOP_DISTANCE")
            loss_budget = account.equity * policy.planned_loss_per_trade_pct
        else:
            if not policy.options_enabled:
                reasons.append("OPTIONS_LIFECYCLE_DISABLED")
            if account.options_level < 2:
                reasons.append("OPTION_PERMISSION_REQUIRED")
            from .options import structure_error

            error = structure_error(alternative)
            if error:
                reasons.append(error)
            if alternative.short_instrument:
                if not policy.spreads_enabled:
                    reasons.append("SPREAD_LIFECYCLE_DISABLED")
                if account.options_level < 3:
                    reasons.append("OPTION_SPREAD_PERMISSION_REQUIRED")
            if (
                not instrument.standard_contract
                or instrument.multiplier != 100
                or not instrument.contract_id
            ):
                reasons.append("UNSUPPORTED_OPTION_CONTRACT")
            if increment != 1:
                reasons.append("WHOLE_CONTRACT_REQUIRED")
            loss = alternative.entry_price * instrument.multiplier + alternative.estimated_fees
            loss_budget = account.equity * policy.max_option_structural_loss_pct
        cost = alternative.entry_price * instrument.multiplier + alternative.estimated_fees
        deployable = (
            account.buying_power
            if policy.allow_leverage and account.margin_allowed
            else min(account.cash, account.buying_power)
        )
        if instrument.asset_type == "option":
            if account.options_buying_power is None:
                reasons.append("OPTION_BUYING_POWER_UNAVAILABLE")
                deployable = Decimal("0")
            else:
                deployable = min(deployable, account.options_buying_power)
        available = max(
            Decimal("0"), deployable - account.reserved_capital - reserved - policy.cash_buffer
        )
        portfolio = max(
            Decimal("0"),
            account.equity * policy.max_portfolio_allocation_pct
            - account.exposure
            - account.reserved_capital
            - reserved,
        )
        bounds = {
            "capital_allocation": account.equity * policy.max_capital_allocation_pct / cost,
            "deployable_capital": available / cost,
            "portfolio_exposure": portfolio / cost,
            "planned_loss": loss_budget / loss if loss > 0 else Decimal("0"),
            "liquidity_capacity": min(item.liquidity_capacity for item in alternative.instruments),
        }
        binding = min(bounds, key=bounds.get)
        quantity = max(Decimal("0"), units(bounds[binding], increment))
        if quantity < increment:
            reasons.append("NO_FEASIBLE_SIZE")
        minimum = max(
            cost * increment / policy.max_capital_allocation_pct,
            loss
            * increment
            / (
                policy.max_option_structural_loss_pct
                if instrument.asset_type == "option"
                else policy.planned_loss_per_trade_pct
            ),
        )
        return SizingResult(
            alternative_id=alternative.id,
            feasible=not reasons,
            quantity=quantity if not reasons else Decimal("0"),
            capital=cost * quantity,
            planned_loss=loss * quantity,
            reasons=tuple(reasons),
            binding_limit=binding,
            minimum_equity_bound=minimum,
        )

    def assess(self, account, alternatives, profile, approvals=(), *, reserved=Decimal("0")):
        approval_map = {(a.strategy_id, a.strategy_version): a for a in approvals}
        results = []
        eligible = []
        for alternative in alternatives:
            result = self.size(account, alternative, reserved=reserved)
            approval = approval_map.get((alternative.strategy_id, alternative.strategy_version))
            reasons = list(result.reasons)
            if approval is None:
                reasons.append("STRATEGY_NOT_APPROVED")
            elif profile not in approval.profiles:
                reasons.append("DATA_PROFILE_INCOMPATIBLE")
            if alternative.expected_net_value is None or not alternative.evidence_ids:
                reasons.append("INSUFFICIENT_EVIDENCE")
            elif alternative.expected_net_value <= self.policy.operating_cost_per_session:
                reasons.append("COST_NOT_JUSTIFIED")
            result = result.model_copy(update={"reasons": tuple(reasons), "feasible": not reasons})
            results.append(result)
            if result.feasible:
                eligible.append(alternative)
        scores = {}
        for alternative in eligible:
            sized = next(result for result in results if result.alternative_id == alternative.id)
            unit_loss = sized.planned_loss / sized.quantity
            scores[alternative.id] = alternative.expected_net_value / unit_loss
        ranking = tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
        selected = ranking[0][0] if ranking else None
        return AccountSuitabilityReport(
            created_at=account.observed_at,
            account_id=account.account_id,
            account_profile_id=account.id,
            hypothetical=account.hypothetical,
            data_profile=profile,
            policy_id=self.policy.id,
            alternatives=tuple(results),
            selected_alternative_id=selected,
            outcome="FEASIBLE" if selected else "NO_TRADE",
            ranking_scores=ranking,
            operating_cost=self.policy.operating_cost_per_session,
            cost_fraction_equity=self.policy.operating_cost_per_session / account.equity,
            uncertainty=("Validated expectancy is research evidence, not a guaranteed outcome.",),
        )


def compare_capital(scenarios, alternatives, profile, policy, approvals=()):
    if len(scenarios) < 2:
        raise ValueError("comparison requires at least two explicit hypothetical accounts")
    reports = []
    identities = set()
    for value in scenarios:
        account = AccountProfile.model_validate(value)
        if not account.hypothetical or account.environment != "scenario":
            raise ValueError("capital comparisons require non-executable scenario scope")
        if account.account_id in identities:
            raise ValueError("scenario identities must be distinct")
        identities.add(account.account_id)
        reports.append(
            AccountSuitabilityService(policy).assess(account, alternatives, profile, approvals)
        )
    return {
        "execution_eligible": False,
        "data_profile": str(profile),
        "cost_allocation": "Each scenario is an alternative account experiment; costs are not summed.",
        "reports": [r.model_dump(mode="json") for r in reports],
    }
