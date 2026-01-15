"""Shared helpers for strategy implementations."""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


def latest_price(data: Optional[pd.DataFrame], price_column: str = "close") -> float:
    """Return the latest price from a dataframe, falling back safely."""
    if data is None or data.empty:
        return 0.0
    if price_column in data.columns:
        series = data[price_column]
    else:
        series = data.iloc[:, -1]
    if series.empty:
        return 0.0
    return float(series.iloc[-1])


def safe_series(series: Optional[pd.Series], default: float = 0.0) -> float:
    """Return the last value from a series or a default when missing."""
    if series is None or series.empty:
        return default
    return float(series.iloc[-1])


def rate_of_change(prices: List[float], period: int) -> List[Optional[float]]:
    """Calculate Rate of Change (ROC) momentum indicator.

    ROC = (Current Price - Price n periods ago) / Price n periods ago

    Args:
        prices: List of closing prices
        period: Lookback period

    Returns:
        List of ROC values (None for insufficient data points)
    """
    roc: List[Optional[float]] = []
    for i in range(len(prices)):
        if i < period:
            roc.append(None)
        else:
            prev_price = prices[i - period]
            if prev_price != 0:
                roc.append((prices[i] - prev_price) / prev_price)
            else:
                roc.append(None)
    return roc


def detect_momentum_crossover(
    short_mom: List[Optional[float]],
    long_mom: List[Optional[float]],
    threshold: float = 0.005,
) -> str:
    """Detect momentum crossover signal.

    Args:
        short_mom: Short-term momentum values (ROC)
        long_mom: Long-term momentum values (ROC)
        threshold: Minimum momentum difference to trigger signal

    Returns:
        'BUY' for bullish crossover, 'SELL' for bearish crossover, 'HOLD' otherwise
    """
    if len(short_mom) < 2 or len(long_mom) < 2:
        return "HOLD"

    curr_short = short_mom[-1]
    curr_long = long_mom[-1]
    prev_short = short_mom[-2]
    prev_long = long_mom[-2]

    if any(v is None for v in [curr_short, curr_long, prev_short, prev_long]):
        return "HOLD"

    curr_diff = curr_short - curr_long
    prev_diff = prev_short - prev_long

    if prev_diff <= 0 and curr_diff > threshold:
        return "BUY"
    elif prev_diff >= 0 and curr_diff < -threshold:
        return "SELL"
    return "HOLD"
