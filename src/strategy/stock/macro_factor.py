"""Macro factor strategy using FRED economic regime signals."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis
from src.data import FREDProvider
from src.strategy.base import (
    ActionPlan,
    SignalSnapshot,
    StrategyBase,
    StrategyMeta,
    StrategySignal,
)
from src.strategy.utils import latest_price, safe_series


ta = TechnicalAnalysis()


class MacroFactorStrategy(StrategyBase):
    """Trade a simple macro regime with price-trend confirmation."""

    meta = StrategyMeta(
        name="macro_factor",
        label="Macro Factor",
        category="signal",
        description="FRED macro regime model with moving-average confirmation.",
        asset_type="stock",
        default_params={
            "economic_lookback_days": 365,
            "fed_funds_high": 4.0,
            "unemployment_high": 5.0,
            "inflation_high": 3.0,
            "sma_period": 50,
            "risk_pct": 0.03,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
        },
        visible_in_dashboard=True,
    )
    summary = (
        "Uses Federal Funds, unemployment, and CPI trends to classify the macro "
        "regime, then confirms with a price trend filter."
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.fred = FREDProvider()
        self._macro_cache: dict[str, dict[str, float | None]] = {}

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
            if plan is None or plan.action == "HOLD":
                continue

            portfolio_value = getattr(
                portfolio,
                "get_portfolio_value",
                lambda *_: 100000,
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
        sma_period = int(self.params["sma_period"])
        if len(historical_data) < sma_period + 1:
            return None

        macro = self._macro_snapshot(current_date)
        sma_value = safe_series(ta.sma(historical_data["close"], sma_period))
        trend_up = current_price >= sma_value

        bull_signals = 0
        bear_signals = 0
        fed_funds = macro["fed_funds_rate"]
        unemployment = macro["unemployment_rate"]
        inflation = macro["inflation_yoy"]

        if fed_funds is not None:
            if fed_funds < float(self.params["fed_funds_high"]):
                bull_signals += 1
            else:
                bear_signals += 1
        if unemployment is not None:
            if unemployment < float(self.params["unemployment_high"]):
                bull_signals += 1
            else:
                bear_signals += 1
        if inflation is not None:
            if inflation < float(self.params["inflation_high"]):
                bull_signals += 1
            else:
                bear_signals += 1

        if trend_up and bull_signals >= 2:
            signal = "BUY"
            reason = "supportive macro regime with positive price trend"
        elif not trend_up and bear_signals >= 2:
            signal = "SELL"
            reason = "restrictive macro regime with negative price trend"
        else:
            signal = "HOLD"
            reason = "macro regime not aligned with trend"

        strength = abs(bull_signals - bear_signals) / 3.0
        return SignalSnapshot(
            symbol=symbol,
            signal=signal,
            indicators={
                "fed_funds_rate": float(fed_funds or 0.0),
                "unemployment_rate": float(unemployment or 0.0),
                "inflation_yoy": float(inflation or 0.0),
                "sma": float(sma_value),
            },
            signal_strength=float(strength),
            reason=reason,
            timestamp=current_date,
            metadata={"macro_snapshot": macro},
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

    def _macro_snapshot(self, current_date: datetime) -> dict[str, float | None]:
        cache_key = current_date.date().isoformat()
        if cache_key in self._macro_cache:
            return self._macro_cache[cache_key]

        lookback_days = int(self.params["economic_lookback_days"])
        start_date = (current_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = current_date.strftime("%Y-%m-%d")
        indicators = self.fred.get_economic_indicators(start_date, end_date)
        cpi_data = indicators.get("Inflation", pd.DataFrame())
        snapshot = {
            "fed_funds_rate": self._latest_value(
                indicators.get("Federal_Funds_Rate", pd.DataFrame())
            ),
            "unemployment_rate": self._latest_value(
                indicators.get("Unemployment", pd.DataFrame())
            ),
            "inflation_yoy": self._inflation_yoy(cpi_data),
        }
        self._macro_cache[cache_key] = snapshot
        return snapshot

    def _latest_value(self, data: pd.DataFrame) -> float | None:
        if data.empty or "value" not in data.columns:
            return None
        value = data["value"].iloc[-1]
        if pd.isna(value):
            return None
        return float(value)

    def _inflation_yoy(self, data: pd.DataFrame) -> float | None:
        if data.empty or "value" not in data.columns or len(data) < 13:
            return None
        latest = data["value"].iloc[-1]
        prior = data["value"].iloc[-13]
        if pd.isna(latest) or pd.isna(prior) or prior == 0:
            return None
        return float(((latest / prior) - 1) * 100)
