"""Black-Scholes pricing, implied volatility, and greeks helpers."""
from __future__ import annotations

import math
from typing import Dict, Optional


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Return the Black-Scholes price for a European option."""
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return 0.0

    option = option_type.lower()
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    if option == "call":
        return (
            spot * math.exp(-dividend_yield * time_to_expiry) * _norm_cdf(d1)
            - strike * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(d2)
        )

    if option == "put":
        return (
            strike * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(-d2)
            - spot * math.exp(-dividend_yield * time_to_expiry) * _norm_cdf(-d1)
        )

    raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    """Return delta, gamma, theta, vega for a European option (per-year theta, per-1.0 vega)."""
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    option = option_type.lower()
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    exp_qt = math.exp(-dividend_yield * time_to_expiry)
    exp_rt = math.exp(-risk_free_rate * time_to_expiry)
    pdf_d1 = _norm_pdf(d1)

    if option == "call":
        delta = exp_qt * _norm_cdf(d1)
        theta = (
            -spot * exp_qt * pdf_d1 * volatility / (2.0 * sqrt_t)
            - risk_free_rate * strike * exp_rt * _norm_cdf(d2)
            + dividend_yield * spot * exp_qt * _norm_cdf(d1)
        )
    elif option == "put":
        delta = exp_qt * (_norm_cdf(d1) - 1.0)
        theta = (
            -spot * exp_qt * pdf_d1 * volatility / (2.0 * sqrt_t)
            + risk_free_rate * strike * exp_rt * _norm_cdf(-d2)
            - dividend_yield * spot * exp_qt * _norm_cdf(-d1)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = exp_qt * pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * exp_qt * pdf_d1 * sqrt_t

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Optional[float]:
    """Solve for implied volatility using bisection."""
    if price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return None

    option = option_type.lower()
    exp_qt = math.exp(-dividend_yield * time_to_expiry)
    exp_rt = math.exp(-risk_free_rate * time_to_expiry)

    if option == "call":
        intrinsic = max(0.0, spot * exp_qt - strike * exp_rt)
        upper = spot * exp_qt
    elif option == "put":
        intrinsic = max(0.0, strike * exp_rt - spot * exp_qt)
        upper = strike * exp_rt
    else:
        return None

    if price < intrinsic - 1e-6 or price > upper + 1e-6:
        return None

    low, high = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = bs_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=mid,
            option_type=option,
            dividend_yield=dividend_yield,
        )
        diff = mid_price - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid

    return None
