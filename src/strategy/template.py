"""
Strategy template and shared primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import logging


@dataclass(frozen=True)
class StrategyMeta:
    """Metadata that describes a strategy."""

    name: str
    label: str
    category: str
    description: str
    visible_in_dashboard: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategySignal:
    """Normalized signal output for backtests and live runs."""

    symbol: str
    action: str  # BUY or SELL
    quantity: int
    price: Optional[float] = None
    reason: str = ""
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class StrategyBase:
    """Base class for all strategies.

    Subclasses should implement generate_signals and update the meta field.
    """

    meta: StrategyMeta

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = {**self.meta.default_params, **(params or {})}
        self.parameters = self.params
        self.positions: Dict[str, Any] = {}
        self.signals: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.meta.name)
        self.name = self.meta.label

    def generate_signals(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def supports_dashboard(self) -> bool:
        return self.meta.visible_in_dashboard

    def _position_size(self, price: float, portfolio_value: float, risk_pct: float) -> int:
        if price <= 0 or portfolio_value <= 0:
            return 0
        return max(1, int((portfolio_value * risk_pct) / price))

    def _normalize(self, signals: Iterable[StrategySignal]) -> List[Dict[str, Any]]:
        return [signal.to_dict() for signal in signals]

    def log_signal(self, signal: Dict[str, Any]) -> None:
        signal_with_timestamp = {**signal, "timestamp": signal.get("timestamp") or datetime.now()}
        self.signals.append(signal_with_timestamp)

    def reset_strategy_state(self) -> None:
        self.positions.clear()
        self.signals.clear()

    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            "name": self.meta.label,
            "type": self.meta.category,
            "description": self.meta.description,
            "parameters": self.params,
            "signals_generated": len(self.signals),
        }
