"""Shared helpers for strategy implementations."""
from __future__ import annotations

from typing import Optional

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
