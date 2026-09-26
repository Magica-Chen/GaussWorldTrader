"""Shared deterministic capital feasibility; research never grants execution authority."""

from decimal import ROUND_FLOOR, Decimal

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
        approval_map = {}
        for approval in approvals:
            approval_map.setdefault((approval.strategy_id, approval.strategy_version), []).append(
                approval
            )
        results = []
        eligible = []
        levels = {}
        for alternative in alternatives:
            result = self.size(account, alternative, reserved=reserved)
            candidates = approval_map.get(
                (alternative.strategy_id, alternative.strategy_version), []
            )
            scoped = [
                candidate
                for candidate in candidates
                if not candidate.symbols or alternative.signal_symbol in candidate.symbols
            ]
            # Prefer economic validation when both release classes cover this symbol.
            validated = [
                candidate for candidate in scoped if candidate.evidence_level == "validated"
            ]
            approval = (validated or scoped or candidates or [None])[-1]
            reasons = list(result.reasons)
            if approval is None:
                reasons.append("STRATEGY_NOT_APPROVED")
            else:
                if profile not in approval.profiles:
                    reasons.append("DATA_PROFILE_INCOMPATIBLE")
                if approval.symbols and alternative.signal_symbol not in approval.symbols:
                    reasons.append("STRATEGY_SYMBOL_NOT_APPROVED")
            level = approval.evidence_level if approval else None
            if level == "paper_trial":
                reasons.extend(self._paper_trial_reasons(account, alternative, approval))
                if not alternative.evidence_ids:
                    reasons.append("INSUFFICIENT_EVIDENCE")
                if alternative.expected_net_value is not None:
                    reasons.append("PAPER_TRIAL_EXPECTANCY_MUST_BE_UNKNOWN")
            else:
                if alternative.expected_net_value is None or not alternative.evidence_ids:
                    reasons.append("INSUFFICIENT_EVIDENCE")
                elif alternative.expected_net_value <= self.policy.operating_cost_per_session:
                    reasons.append("COST_NOT_JUSTIFIED")
            result = result.model_copy(
                update={"reasons": tuple(reasons), "feasible": not reasons, "evidence_level": level}
            )
            results.append(result)
            levels[alternative.id] = level
            if result.feasible:
                eligible.append(alternative)
        scores = {}
        sized_by_id = {result.alternative_id: result for result in results}
        for alternative in eligible:
            sized = sized_by_id[alternative.id]
            unit_loss = sized.planned_loss / sized.quantity
            scores[alternative.id] = (
                -unit_loss / alternative.entry_price
                if levels[alternative.id] == "paper_trial"
                else alternative.expected_net_value / unit_loss
            )
        ranking = tuple(
            sorted(
                scores.items(),
                key=lambda item: (levels[item[0]] == "paper_trial", -item[1], item[0]),
            )
        )
        selected = ranking[0][0] if ranking else None
        trial = "paper_trial" in levels.values()
        validated = "validated" in levels.values()
        objective = "validated_net_expectancy_per_unit_planned_loss"
        uncertainty = ("Validated expectancy is research evidence, not a guaranteed outcome.",)
        if trial:
            objective = (
                "validated_expectancy_first_then_paper_trial_lowest_fractional_planned_loss"
                if validated
                else "paper_trial_lowest_fractional_planned_loss"
            )
            trial_uncertainty = (
                "Paper trial ranks buffered stop distance divided by entry price; "
                "net expectancy is unknown and profitability is unvalidated.",
                "Planned loss is a sizing assumption; gaps and execution can exceed the stop.",
            )
            uncertainty = uncertainty + trial_uncertainty if validated else trial_uncertainty
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
            objective=objective,
            ranking_scores=ranking,
            operating_cost=self.policy.operating_cost_per_session,
            cost_fraction_equity=self.policy.operating_cost_per_session / account.equity,
            uncertainty=uncertainty,
        )

    def _paper_trial_reasons(self, account, alternative, approval):
        reasons = []
        if (
            account.environment != "paper"
            or account.hypothetical
            or approval.account_id != account.account_id
            or approval.environment != account.environment
        ):
            reasons.append("PAPER_TRIAL_ACCOUNT_SCOPE")
        if alternative.instrument.asset_type != "stock" or alternative.short_instrument is not None:
            reasons.append("PAPER_TRIAL_STOCK_ONLY")
        if approval.expiry <= account.observed_at:
            reasons.append("STRATEGY_APPROVAL_EXPIRED")
        policy = self.policy
        if (
            policy.max_capital_allocation_pct > Decimal(".02")
            or policy.max_portfolio_allocation_pct > Decimal(".10")
            or policy.planned_loss_per_trade_pct > Decimal(".001")
            or policy.daily_loss_pause_pct > Decimal(".01")
            or policy.allow_leverage
            or policy.options_enabled
            or policy.spreads_enabled
        ):
            reasons.append("PAPER_TRIAL_POLICY_EXCEEDS_LIMITS")
        return reasons


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
