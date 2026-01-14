"""
Built-in strategies implemented using the shared template.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from src.analysis.technical_analysis import TechnicalAnalysis

from .template import StrategyBase, StrategyMeta, StrategySignal


ta = TechnicalAnalysis()


def _latest_price(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    return float(data["close"].iloc[-1])


def _safe_series(series: pd.Series, default: float = 0.0) -> float:
    if series.empty:
        return default
    return float(series.iloc[-1])


class MomentumStrategy(StrategyBase):
    meta = StrategyMeta(
        name="momentum",
        label="Momentum",
        category="signal",
        description="Buys strength and sells weakness using recent returns.",
        default_params={"lookback": 20, "threshold": 0.02, "risk_pct": 0.05},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        signals: List[StrategySignal] = []
        risk_pct = float(self.params["risk_pct"])
        lookback = int(self.params["lookback"])
        threshold = float(self.params["threshold"])

        for symbol, data in historical_data.items():
            if len(data) < lookback + 1:
                continue
            window = data["close"].iloc[-lookback:]
            returns = (window.iloc[-1] - window.iloc[0]) / window.iloc[0]
            price = current_prices.get(symbol, _latest_price(data))
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if returns >= threshold:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason=f"{returns:.2%} momentum",
                        timestamp=current_date,
                    )
                )
            elif returns <= -threshold:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason=f"{returns:.2%} drawdown",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)


class ValueStrategy(StrategyBase):
    meta = StrategyMeta(
        name="value",
        label="Value",
        category="signal",
        description="Looks for price discounts to a long-term average.",
        default_params={"sma_period": 50, "discount_pct": 0.03, "risk_pct": 0.04},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        signals: List[StrategySignal] = []
        period = int(self.params["sma_period"])
        discount = float(self.params["discount_pct"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < period + 1:
                continue
            sma = ta.sma(data["close"], period)
            sma_value = _safe_series(sma)
            price = current_prices.get(symbol, _latest_price(data))
            if sma_value <= 0:
                continue
            deviation = (price - sma_value) / sma_value
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if deviation <= -discount:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} below SMA{period}",
                        timestamp=current_date,
                    )
                )
            elif deviation >= discount:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} above SMA{period}",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)


class TrendFollowingStrategy(StrategyBase):
    meta = StrategyMeta(
        name="trend_following",
        label="Trend Following",
        category="signal",
        description="Trades on moving average crossovers.",
        default_params={"fast": 20, "slow": 50, "risk_pct": 0.05},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        signals: List[StrategySignal] = []
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < slow + 1:
                continue
            fast_sma = ta.sma(data["close"], fast)
            slow_sma = ta.sma(data["close"], slow)
            fast_val = _safe_series(fast_sma)
            slow_val = _safe_series(slow_sma)
            price = current_prices.get(symbol, _latest_price(data))
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if fast_val > slow_val:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason="fast SMA above slow SMA",
                        timestamp=current_date,
                    )
                )
            elif fast_val < slow_val:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason="fast SMA below slow SMA",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)


class ScalpingStrategy(StrategyBase):
    meta = StrategyMeta(
        name="scalping",
        label="Scalping",
        category="signal",
        description="Short-term mean reversion around a short EMA.",
        default_params={"ema_period": 5, "band_pct": 0.003, "risk_pct": 0.02},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        signals: List[StrategySignal] = []
        period = int(self.params["ema_period"])
        band = float(self.params["band_pct"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < period + 1:
                continue
            ema = ta.ema(data["close"], period)
            ema_val = _safe_series(ema)
            price = current_prices.get(symbol, _latest_price(data))
            if ema_val <= 0:
                continue
            deviation = (price - ema_val) / ema_val
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if deviation <= -band:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} below EMA{period}",
                        timestamp=current_date,
                    )
                )
            elif deviation >= band:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason=f"{deviation:.2%} above EMA{period}",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)


class StatisticalArbitrageStrategy(StrategyBase):
    meta = StrategyMeta(
        name="statistical_arbitrage",
        label="Statistical Arbitrage",
        category="signal",
        description="Mean reversion using z-score of recent returns.",
        default_params={"window": 20, "zscore": 1.5, "risk_pct": 0.03},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        signals: List[StrategySignal] = []
        window = int(self.params["window"])
        zscore = float(self.params["zscore"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < window + 2:
                continue
            returns = data["close"].pct_change().dropna().iloc[-window:]
            if returns.empty:
                continue
            mean = float(returns.mean())
            std = float(returns.std()) or 1e-6
            score = (returns.iloc[-1] - mean) / std
            price = current_prices.get(symbol, _latest_price(data))
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if score <= -zscore:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason=f"z-score {score:.2f} oversold",
                        timestamp=current_date,
                    )
                )
            elif score >= zscore:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason=f"z-score {score:.2f} overbought",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)


class GaussianProcessStrategy(StrategyBase):
    meta = StrategyMeta(
        name="gaussian_process",
        label="Gaussian Process",
        category="signal",
        description="Proxy uncertainty-aware mean reversion strategy.",
        default_params={"window": 30, "zscore": 1.0, "risk_pct": 0.03},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        signals: List[StrategySignal] = []
        window = int(self.params["window"])
        zscore = float(self.params["zscore"])
        risk_pct = float(self.params["risk_pct"])

        for symbol, data in historical_data.items():
            if len(data) < window + 2:
                continue
            returns = data["close"].pct_change().dropna().iloc[-window:]
            if returns.empty:
                continue
            mean = float(returns.mean())
            std = float(returns.std()) or 1e-6
            score = (returns.iloc[-1] - mean) / std
            price = current_prices.get(symbol, _latest_price(data))
            portfolio_value = getattr(portfolio, "get_portfolio_value", lambda *_: 100000)(current_prices)
            quantity = self._position_size(price, portfolio_value, risk_pct)

            if score <= -zscore:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        reason="uncertainty signal BUY",
                        timestamp=current_date,
                    )
                )
            elif score >= zscore:
                signals.append(
                    StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=price,
                        reason="uncertainty signal SELL",
                        timestamp=current_date,
                    )
                )

        return self._normalize(signals)


class XGBoostStrategy(StrategyBase):
    meta = StrategyMeta(
        name="xgboost",
        label="XGBoost",
        category="ml",
        description="Template ML strategy (rule-based placeholder).",
        default_params={"lookback": 10, "threshold": 0.015, "risk_pct": 0.04},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        base = MomentumStrategy({
            "lookback": self.params["lookback"],
            "threshold": self.params["threshold"],
            "risk_pct": self.params["risk_pct"],
        })
        return base.generate_signals(current_date, current_prices, current_data, historical_data, portfolio)


class DeepLearningStrategy(StrategyBase):
    meta = StrategyMeta(
        name="deep_learning",
        label="Deep Learning",
        category="ml",
        description="Template deep learning strategy (rule-based placeholder).",
        default_params={"lookback": 15, "threshold": 0.02, "risk_pct": 0.04},
        visible_in_dashboard=True,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        base = MomentumStrategy({
            "lookback": self.params["lookback"],
            "threshold": self.params["threshold"],
            "risk_pct": self.params["risk_pct"],
        })
        return base.generate_signals(current_date, current_prices, current_data, historical_data, portfolio)


class WheelStrategy(StrategyBase):
    meta = StrategyMeta(
        name="wheel",
        label="Wheel",
        category="options",
        description="Options wheel template (managed outside the dashboard).",
        default_params={},
        visible_in_dashboard=False,
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        return []
