"""Momentum strategy using dual momentum crossover.

Generates BUY signals when short-term momentum crosses above long-term momentum,
and SELL signals when short-term momentum crosses below long-term momentum.
Includes stop-loss and take-profit levels for risk management.

Supports both stock and crypto asset types via asset_type parameter.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.strategy.base import ActionPlan, SignalSnapshot, StrategyBase, StrategyMeta, StrategySignal
from src.strategy.utils import detect_momentum_crossover, latest_price, rate_of_change


# Default parameters by asset type
_STOCK_DEFAULTS = {
    "short_period": 12,
    "long_period": 26,
    "threshold": 0.005,
    "risk_pct": 0.05,
    "stop_loss_pct": 0.03,
    "take_profit_pct": 0.06,
}

_CRYPTO_DEFAULTS = {
    "short_period": 12,
    "long_period": 26,
    "threshold": 0.005,
    "risk_pct": 0.10,
    "stop_loss_pct": 0.03,
    "take_profit_pct": 0.06,
    "qty_precision": 6,
    "min_qty": 0.000001,
}


class MomentumStrategy(StrategyBase):
    """Dual momentum crossover strategy supporting stock and crypto.

    Uses short-term and long-term Rate of Change momentum to detect crossovers.
    Position sizing behavior depends on `asset_type` parameter:
    - "stock": whole shares (default)
    - "crypto": decimal precision with min_qty support
    """

    meta = StrategyMeta(
        name="momentum",
        label="Momentum",
        category="signal",
        description="Dual momentum crossover strategy with stop-loss and take-profit.",
        asset_type="stock",
        default_params=_STOCK_DEFAULTS,
        visible_in_dashboard=True,
    )
    summary = (
        "Dual momentum crossover strategy. Calculates short-term (12-period) and "
        "long-term (26-period) Rate of Change momentum. Generates BUY when short "
        "momentum crosses above long momentum, SELL when it crosses below. "
        "Includes stop-loss (3%) and take-profit (6%) for risk management."
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        # Determine asset type from params before calling super().__init__
        self._asset_type = (params or {}).get("asset_type", "stock")

        # Select defaults based on asset type
        defaults = _CRYPTO_DEFAULTS if self._asset_type == "crypto" else _STOCK_DEFAULTS
        merged_params = {**defaults, **(params or {})}

        super().__init__(merged_params)

    def _position_size(self, price: float, portfolio_value: float, risk_pct: float) -> float:
        """Position sizing based on asset type."""
        if price <= 0 or portfolio_value <= 0:
            return 0.0

        if self._asset_type == "crypto":
            # Crypto: decimal precision with min_qty
            quantity = (portfolio_value * risk_pct) / price
            precision = int(self.params.get("qty_precision", 6))
            min_qty = float(self.params.get("min_qty", 0.0))
            quantity = round(quantity, precision)
            if min_qty > 0 and quantity < min_qty:
                return 0.0
            return quantity
        else:
            # Stock: whole shares
            return float(max(1, int((portfolio_value * risk_pct) / price)))

    def generate_signals(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> List[Dict[str, Any]]:
        signals: List[StrategySignal] = []
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            price = current_prices.get(symbol, latest_price(data))
            snapshot = self.get_signal(
                symbol=symbol,
                current_date=current_date,
                current_price=price,
                current_data=current_data.get(symbol, {}),
                historical_data=data,
                portfolio=portfolio,
            )
            if snapshot is None:
                continue
            plan = self.get_action_plan(snapshot, price, current_date)
            if not plan or plan.action == "HOLD":
                continue

            portfolio_value = getattr(
                portfolio, "get_portfolio_value", lambda *_: 100000
            )(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)
            if quantity <= 0:
                continue
            signals.append(self._plan_to_signal(plan, quantity, price))

        return self._normalize(signals)

    def get_signal(
        self,
        symbol: str,
        current_date: datetime,
        current_price: float,
        current_data: Dict[str, Any],
        historical_data: pd.DataFrame,
        portfolio: Any = None,
    ) -> Optional[SignalSnapshot]:
        short_period = int(self.params["short_period"])
        long_period = int(self.params["long_period"])
        threshold = float(self.params["threshold"])

        min_bars = long_period + 2
        if len(historical_data) < min_bars:
            return None

        prices = historical_data["close"].tolist()
        short_mom = rate_of_change(prices, short_period)
        long_mom = rate_of_change(prices, long_period)

        signal_type = detect_momentum_crossover(short_mom, long_mom, threshold)
        curr_short = short_mom[-1] if short_mom[-1] is not None else 0.0
        curr_long = long_mom[-1] if long_mom[-1] is not None else 0.0
        diff = curr_short - curr_long

        reason = f"momentum crossover (short: {curr_short:.2%}, long: {curr_long:.2%})"
        return SignalSnapshot(
            symbol=symbol,
            signal=signal_type,
            indicators={
                "short_momentum": float(curr_short),
                "long_momentum": float(curr_long),
                "momentum_diff": float(diff),
                "threshold": float(threshold),
            },
            signal_strength=float(diff),
            reason=reason,
            timestamp=current_date,
        )

    def get_action_plan(
        self,
        signal: SignalSnapshot,
        current_price: float,
        current_date: datetime,
    ) -> Optional[ActionPlan]:
        if signal.signal == "HOLD":
            return None

        side = "long" if signal.signal == "BUY" else "short"
        stop_loss = self.calculate_stop_loss(current_price, side)
        take_profit = self.calculate_take_profit(current_price, side)

        return ActionPlan(
            symbol=signal.symbol,
            action=signal.signal,
            target_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=signal.reason,
            strength=abs(signal.signal_strength),
            timestamp=signal.timestamp or current_date,
        )
