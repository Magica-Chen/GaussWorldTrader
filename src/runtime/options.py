"""Account-independent option selection over verified contracts and timestamped quotes.

Selection produces research alternatives. The risk gate repeats mechanics and sizing
against current account state; no selector can submit an order or change its mandate.
"""

from datetime import date
from decimal import Decimal, ROUND_CEILING

from .models import Alternative, Instrument, OrderLeg


def opening_legs(alternative):
    legs = [OrderLeg(instrument=alternative.instrument, side="buy", position_intent="buy_to_open")]
    if alternative.short_instrument:
        legs.append(
            OrderLeg(
                instrument=alternative.short_instrument, side="sell", position_intent="sell_to_open"
            )
        )
    return tuple(legs)


def structure_error(alternative):
    """Verify standard debit economics before any premium*multiplier calculation."""
    if alternative.instrument.asset_type != "option":
        return "STOCK_CANNOT_HAVE_OPTION_LEG" if alternative.short_instrument else None
    for instrument in alternative.instruments:
        if (
            instrument.asset_type != "option"
            or not instrument.standard_contract
            or instrument.multiplier != 100
            or instrument.quantity_increment != 1
            or not instrument.contract_id
            or not instrument.expiry
            or not instrument.underlying
            or not instrument.option_type
            or not instrument.strike
            or not instrument.tradable
        ):
            return "UNSUPPORTED_OPTION_CONTRACT"
    bought, sold = alternative.instrument, alternative.short_instrument
    if sold is None:
        return None
    if (
        bought.symbol == sold.symbol
        or bought.underlying != sold.underlying
        or bought.expiry != sold.expiry
        or bought.option_type != sold.option_type
        or bought.multiplier != sold.multiplier
    ):
        return "UNSUPPORTED_OPTION_STRUCTURE"
    width = (
        (sold.strike - bought.strike)
        if bought.option_type == "call"
        else (bought.strike - sold.strike)
    )
    if width <= 0 or alternative.entry_price >= width:
        return "INVALID_DEBIT_WIDTH"
    return None


def quote_checks(
    quotes, now, *, max_age, max_skew=Decimal("1"), max_spread=Decimal(".02"), available_at=None
):
    available_at = available_at or now
    if not quotes or any(q is None for q in quotes):
        return "CURRENT_GENUINE_QUOTE_REQUIRED"
    for quote in quotes:
        if (
            quote.feed != "opra"
            or quote.quality_class != "genuine"
            or not quote.complete
            or quote.bid is None
            or quote.ask is None
            or quote.bid <= 0
            or quote.ask < quote.bid
            or not 0 <= (now - quote.effective_event_time).total_seconds() <= max_age
            or quote.received_at > available_at
            or quote.available_at > available_at
        ):
            return "CURRENT_GENUINE_QUOTE_REQUIRED"
        if (quote.ask - quote.bid) / quote.ask > max_spread:
            return "QUOTE_SPREAD_TOO_WIDE"
    skew = (
        max(q.effective_event_time for q in quotes) - min(q.effective_event_time for q in quotes)
    ).total_seconds()
    return "ASYNCHRONOUS_LEG_QUOTES" if Decimal(str(skew)) > max_skew else None


def select_alternatives(
    approval,
    contracts,
    quote_by_symbol,
    underlying_price,
    underlying_stop,
    now,
    *,
    quote_max_age=2,
    max_spread=Decimal(".02"),
    available_at=None,
):
    """Rank eligible unit-cost alternatives deterministically; account ranking follows."""
    selector = approval.option_selector
    if selector is None:
        return (), ("OPTION_SELECTOR_REQUIRED",)
    desired_type = "call" if "call" in selector.structure else "put"
    candidates, reasons = [], set()
    for item in contracts:
        item = Instrument.model_validate(item)
        if not item.expiry:
            reasons.add("CONTRACT_EXPIRY_MISSING")
            continue
        dte = (date.fromisoformat(item.expiry) - now.date()).days
        if (
            item.asset_type != "option"
            or item.option_type != desired_type
            or not selector.minimum_dte <= dte <= selector.maximum_dte
            or not item.strike
            or not selector.minimum_moneyness
            <= item.strike / underlying_price
            <= selector.maximum_moneyness
        ):
            continue
        quote = quote_by_symbol.get(item.symbol)
        error = quote_checks(
            [quote], now, max_age=quote_max_age, max_spread=max_spread, available_at=available_at
        )
        if error:
            reasons.add(error)
            continue
        if item.liquidity_capacity < selector.minimum_displayed_size:
            reasons.add("INSUFFICIENT_DISPLAYED_SIZE")
            continue
        candidates.append((item, quote))
    alternatives = []
    for bought, ask_quote in candidates:
        pairs = [(None, None)] if selector.structure.startswith("long_") else candidates
        for sold, bid_quote in pairs:
            if sold is not None:
                if (
                    bought.expiry != sold.expiry
                    or bought.symbol == sold.symbol
                    or abs(bought.strike - sold.strike) > selector.maximum_width
                ):
                    continue
                error = quote_checks(
                    [ask_quote, bid_quote],
                    now,
                    max_age=quote_max_age,
                    max_skew=selector.maximum_quote_skew_seconds,
                    max_spread=max_spread,
                    available_at=available_at,
                )
                if error:
                    reasons.add(error)
                    continue
            debit = ask_quote.ask - (bid_quote.bid if bid_quote else Decimal("0"))
            if debit <= 0:
                continue
            # Conservative synthetic entry bound; never presented as a complex-order quote.
            tick = max(bought.tick_size, sold.tick_size if sold else bought.tick_size)
            debit = (debit / tick).to_integral_value(rounding=ROUND_CEILING) * tick
            alternative = Alternative(
                created_at=available_at or now,
                strategy_id=approval.strategy_id,
                strategy_version=approval.strategy_version,
                instrument=bought,
                short_instrument=sold,
                entry_price=debit,
                stop_price=max(tick, debit / 2),
                underlying_entry_price=underlying_price,
                underlying_stop_price=underlying_stop,
                expected_net_value=approval.validated_net_expectancy,
                evidence_ids=tuple(q.id for q in (ask_quote, bid_quote) if q is not None),
            )
            error = structure_error(alternative)
            if error:
                reasons.add(error)
            else:
                alternatives.append(alternative)
    alternatives.sort(
        key=lambda a: (
            a.entry_price,
            a.instrument.expiry,
            a.instrument.symbol,
            a.short_instrument.symbol if a.short_instrument else "",
        )
    )
    return tuple(alternatives), tuple(sorted(reasons))
