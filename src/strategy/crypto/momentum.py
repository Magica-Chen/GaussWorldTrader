"""Crypto momentum strategy using dual momentum crossover.

Uses short-term and long-term Rate of Change momentum to detect crossovers
and generate trading signals for crypto pairs. Includes stop-loss and
take-profit levels for risk management.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from src.strategy.base import BaseCryptoStrategy, StrategyMeta, StrategySignal
from src.strategy.utils import detect_momentum_crossover, latest_price, rate_of_change


class CryptoMomentumStrategy(BaseCryptoStrategy):
    meta = StrategyMeta(
        name="crypto_momentum",
        label="Crypto Momentum",
        category="signal",
        description="Dual momentum crossover strategy for crypto with risk management.",
        asset_type="crypto",
        default_params={
            "short_period": 12,
            "long_period": 26,
            "threshold": 0.005,
            "risk_pct": 0.10,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
        },
        visible_in_dashboard=True,
    )
    summary = (
        "Dual momentum crossover strategy for crypto. Calculates short-term (12-period) "
        "and long-term (26-period) Rate of Change momentum. Generates BUY when short "
        "momentum crosses above long momentum, SELL when it crosses below. "
        "Includes stop-loss (3%) and take-profit (6%) for risk management."
    )

    def generate_signals(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> List[Dict[str, Any]]:
        signals: List[StrategySignal] = []
        short_period = int(self.params["short_period"])
        long_period = int(self.params["long_period"])
        threshold = float(self.params["threshold"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            min_bars = long_period + 2
            if len(data) < min_bars:
                continue

            prices = data["close"].tolist()
            short_mom = rate_of_change(prices, short_period)
            long_mom = rate_of_change(prices, long_period)

            signal_type = detect_momentum_crossover(short_mom, long_mom, threshold)
            if signal_type == "HOLD":
                continue

            price = current_prices.get(symbol, latest_price(data))
            portfolio_value = getattr(
                portfolio, "get_portfolio_value", lambda *_: 100000
            )(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            side = "long" if signal_type == "BUY" else "short"
            stop_loss = self.calculate_stop_loss(price, side)
            take_profit = self.calculate_take_profit(price, side)

            curr_short = short_mom[-1] if short_mom[-1] is not None else 0
            curr_long = long_mom[-1] if long_mom[-1] is not None else 0
            reason = f"momentum crossover (short: {curr_short:.2%}, long: {curr_long:.2%})"

            signals.append(
                StrategySignal(
                    symbol=symbol,
                    action=signal_type,
                    quantity=quantity,
                    price=price,
                    reason=reason,
                    timestamp=current_date,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
            )

        return self._normalize(signals)
