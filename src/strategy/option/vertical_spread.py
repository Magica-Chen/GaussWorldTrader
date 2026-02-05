"""Vertical spread options strategy (bull/bear, call/put)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.analysis.option_greeks import bs_greeks, implied_volatility
from src.analysis.technical_analysis import TechnicalAnalysis
from src.data.alpaca_provider import AlpacaDataProvider
from src.data.fred_provider import FREDProvider
from src.strategy.base import ActionPlan, BaseOptionStrategy, SignalSnapshot, StrategyMeta
from src.strategy.utils import safe_series
from src.utils.timezone_utils import now_et


ta = TechnicalAnalysis()


@dataclass(frozen=True)
class _SpreadCandidate:
    spread_type: str
    action: str
    limit_price: float
    score: float
    reason: str
    metadata: Dict[str, Any]
    max_loss: float


class VerticalSpreadStrategy(BaseOptionStrategy):
    """Vertical spread strategy using trend + RSI with IV filters."""

    meta = StrategyMeta(
        name="vertical_spread",
        label="Vertical Spread",
        category="option",
        description="Vertical spread strategy using trend + RSI with IV/greeks filters.",
        asset_type="option",
        visible_in_dashboard=False,
        default_params={
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "rsi_bull": 55,
            "rsi_bear": 45,
            "dte_min": 7,
            "dte_max": 21,
            "otm_pct": 0.03,
            "width_pct": 0.02,
            "iv_min_credit": 0.60,
            "iv_max_debit": 0.30,
            "short_delta_min": 0.30,
            "short_delta_max": 0.40,
            "credit_min_pct": 0.25,
            "credit_max_pct": 0.60,
            "debit_min_pct": 0.20,
            "debit_max_pct": 0.60,
            "risk_pct": 0.05,
            "risk_free_rate_maturity": "3M",
            "risk_free_rate_fallback": 0.02,
        },
    )

    summary = (
        "Vertical spread strategy (bull put/call, bear put/call) using EMA trend + RSI, "
        "DTE filters, fixed OTM/width strikes, and IV/greeks computed via Black-Scholes."
    )

    def __init__(self, parameters: Dict[str, Any] | None = None) -> None:
        super().__init__(parameters)
        self.name = "VerticalSpreadStrategy"
        self.provider = AlpacaDataProvider()
        self.fred = FREDProvider()
        self._risk_free_cache: Optional[Tuple[date, float]] = None

    def filter_underlying_stocks(self, client: Any) -> List[str]:
        return self.symbol_list

    def filter_options(
        self, client: Any, underlying: str, option_type: str = "put"
    ) -> List[Dict[str, Any]]:
        return []

    def score_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return options

    def select_best_options(
        self, scored_options: List[Dict[str, Any]], limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if limit is None:
            return scored_options
        return scored_options[:limit]

    def generate_signals(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []

        for symbol, data in historical_data.items():
            price = float(current_prices.get(symbol, 0.0))
            if price <= 0 and data is not None and not data.empty:
                price = float(data["close"].iloc[-1])
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
            qty = float(plan.metadata.get("override_qty") or 0)
            if qty <= 0:
                continue
            signals.append(
                {
                    "symbol": plan.symbol,
                    "action": plan.action,
                    "quantity": qty,
                    "price": plan.target_price,
                    "reason": plan.reason,
                    "timestamp": plan.timestamp or current_date,
                }
            )

        return signals

    def get_signal(
        self,
        symbol: str,
        current_date: datetime,
        current_price: float,
        current_data: Dict[str, Any],
        historical_data: pd.DataFrame,
        portfolio: Any = None,
    ) -> Optional[SignalSnapshot]:
        if historical_data is None or historical_data.empty:
            return None

        ema_fast = ta.ema(historical_data["close"], int(self.params["ema_fast"]))
        ema_slow = ta.ema(historical_data["close"], int(self.params["ema_slow"]))
        rsi = ta.rsi(historical_data["close"], int(self.params["rsi_period"]))

        fast_val = safe_series(ema_fast)
        slow_val = safe_series(ema_slow)
        rsi_val = safe_series(rsi)

        bias = None
        reason = "no trend signal"
        if fast_val > slow_val and rsi_val >= float(self.params["rsi_bull"]):
            bias = "bull"
            reason = "bullish trend + RSI"
        elif fast_val < slow_val and rsi_val <= float(self.params["rsi_bear"]):
            bias = "bear"
            reason = "bearish trend + RSI"

        if bias is None:
            return SignalSnapshot(
                symbol=symbol,
                signal="HOLD",
                indicators={
                    "ema_fast": float(fast_val),
                    "ema_slow": float(slow_val),
                    "rsi": float(rsi_val),
                },
                signal_strength=0.0,
                reason=reason,
                timestamp=current_date,
                metadata={},
            )

        spread = self._select_spread(symbol, current_price, bias)
        if spread is None:
            return SignalSnapshot(
                symbol=symbol,
                signal="HOLD",
                indicators={
                    "ema_fast": float(fast_val),
                    "ema_slow": float(slow_val),
                    "rsi": float(rsi_val),
                },
                signal_strength=0.0,
                reason="no valid spread found",
                timestamp=current_date,
                metadata={},
            )

        override_qty = self._size_spread(spread.max_loss, portfolio)
        if override_qty <= 0:
            return SignalSnapshot(
                symbol=symbol,
                signal="HOLD",
                indicators={
                    "ema_fast": float(fast_val),
                    "ema_slow": float(slow_val),
                    "rsi": float(rsi_val),
                },
                signal_strength=0.0,
                reason="risk sizing produced zero quantity",
                timestamp=current_date,
                metadata={},
            )

        metadata = {**spread.metadata, "override_qty": override_qty}
        return SignalSnapshot(
            symbol=symbol,
            signal=spread.action,
            indicators={
                "ema_fast": float(fast_val),
                "ema_slow": float(slow_val),
                "rsi": float(rsi_val),
            },
            signal_strength=float(spread.score),
            reason=spread.reason,
            timestamp=current_date,
            metadata={
                "order_metadata": metadata,
                "limit_price": spread.limit_price,
            },
        )

    def get_action_plan(
        self,
        signal: SignalSnapshot,
        current_price: float,
        current_date: datetime,
    ) -> Optional[ActionPlan]:
        if signal.signal == "HOLD":
            return ActionPlan(
                symbol=signal.symbol,
                action="HOLD",
                target_price=None,
                stop_loss=None,
                take_profit=None,
                reason=signal.reason,
                strength=signal.signal_strength,
                timestamp=signal.timestamp or current_date,
                metadata=signal.metadata,
            )

        order_metadata = (signal.metadata or {}).get("order_metadata")
        if not order_metadata:
            return None
        limit_price = (signal.metadata or {}).get("limit_price")
        if limit_price is None:
            return None
        return ActionPlan(
            symbol=signal.symbol,
            action=signal.signal,
            target_price=limit_price,
            stop_loss=None,
            take_profit=None,
            reason=signal.reason,
            strength=signal.signal_strength,
            timestamp=signal.timestamp or current_date,
            metadata=order_metadata,
        )

    def _size_spread(self, max_loss: float, portfolio: Any) -> int:
        if max_loss <= 0:
            return 0
        portfolio_value = 100000.0
        if portfolio is not None and hasattr(portfolio, "get_portfolio_value"):
            portfolio_value = float(portfolio.get_portfolio_value({}))
        risk_pct = float(self.params.get("risk_pct", 0.05))
        qty = int((portfolio_value * risk_pct) / max_loss)
        return max(0, qty)

    def _get_risk_free_rate(self) -> float:
        fallback = float(self.params.get("risk_free_rate_fallback", 0.02))
        maturity = str(self.params.get("risk_free_rate_maturity", "3M")).upper()

        today = now_et().date()
        if self._risk_free_cache and self._risk_free_cache[0] == today:
            return self._risk_free_cache[1]

        rate = None
        try:
            data = self.fred.get_treasury_yield(maturity)
            if isinstance(data, pd.DataFrame) and "value" in data.columns:
                if "error" not in data.columns and not data["value"].empty:
                    value = float(data["value"].iloc[-1])
                    rate = value / 100.0
        except Exception:
            rate = None

        if rate is None or rate <= 0:
            rate = fallback

        self._risk_free_cache = (today, rate)
        return rate

    @staticmethod
    def _mid_price(row: pd.Series) -> Optional[float]:
        bid = float(row.get("bid_price") or 0.0)
        ask = float(row.get("ask_price") or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        last = float(row.get("last_price") or 0.0)
        if last > 0:
            return last
        return None

    @staticmethod
    def _parse_expiration(value: Any) -> Optional[date]:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d").date()
                except ValueError:
                    return None
        return None

    @staticmethod
    def _pick_strike(strikes: List[float], target: float, direction: str) -> Optional[float]:
        if not strikes:
            return None
        if direction == "lte":
            candidates = [s for s in strikes if s <= target]
            return max(candidates) if candidates else None
        if direction == "gte":
            candidates = [s for s in strikes if s >= target]
            return min(candidates) if candidates else None
        return None

    def _select_spread(self, symbol: str, spot: float, bias: str) -> Optional[_SpreadCandidate]:
        if spot <= 0:
            return None
        chain = self.provider.get_options_chain(symbol)
        if chain is None or chain.empty:
            return None
        required_cols = {"expiration_date", "strike_price", "option_type"}
        if not required_cols.issubset(chain.columns):
            return None

        chain = chain.copy()
        chain["expiration"] = chain["expiration_date"].apply(self._parse_expiration)
        chain = chain[chain["expiration"].notna()]
        if chain.empty:
            return None

        dte_min = int(self.params["dte_min"])
        dte_max = int(self.params["dte_max"])
        today = now_et().date()
        chain["dte"] = chain["expiration"].apply(lambda d: (d - today).days)
        chain = chain[(chain["dte"] >= dte_min) & (chain["dte"] <= dte_max)]
        if chain.empty:
            return None

        chain["option_type"] = chain["option_type"].astype(str).str.upper()
        expirations = sorted(chain["expiration"].unique())

        candidates: List[_SpreadCandidate] = []
        if bias == "bull":
            for spread_type in ("bull_put", "bull_call"):
                candidate = self._best_candidate_for_type(
                    chain, expirations, spot, symbol, spread_type
                )
                if candidate:
                    candidates.append(candidate)
        else:
            for spread_type in ("bear_call", "bear_put"):
                candidate = self._best_candidate_for_type(
                    chain, expirations, spot, symbol, spread_type
                )
                if candidate:
                    candidates.append(candidate)

        if not candidates:
            return None
        return max(candidates, key=lambda c: c.score)

    def _best_candidate_for_type(
        self,
        chain: pd.DataFrame,
        expirations: List[date],
        spot: float,
        symbol: str,
        spread_type: str,
    ) -> Optional[_SpreadCandidate]:
        option_type = "P" if "put" in spread_type else "C"
        width_target = spot * float(self.params["width_pct"])
        otm_pct = float(self.params["otm_pct"])
        r = self._get_risk_free_rate()

        best: Optional[_SpreadCandidate] = None
        for exp in expirations:
            df_exp = chain[(chain["expiration"] == exp) & (chain["option_type"] == option_type)]
            if df_exp.empty:
                continue
            strikes = sorted(float(s) for s in df_exp["strike_price"].unique())

            short_strike, long_strike, short_side, long_side, is_credit = self._spread_strikes(
                spread_type, strikes, spot, otm_pct, width_target
            )
            if short_strike is None or long_strike is None:
                continue

            short_row = df_exp.loc[(df_exp["strike_price"] - short_strike).abs() < 1e-6]
            long_row = df_exp.loc[(df_exp["strike_price"] - long_strike).abs() < 1e-6]
            if short_row.empty or long_row.empty:
                continue
            short_row = short_row.iloc[0]
            long_row = long_row.iloc[0]

            short_mid = self._mid_price(short_row)
            long_mid = self._mid_price(long_row)
            if short_mid is None or long_mid is None:
                continue

            dte = int((exp - now_et().date()).days)
            if dte <= 0:
                continue
            t_years = dte / 365.0

            iv_short = implied_volatility(
                price=short_mid,
                spot=spot,
                strike=short_strike,
                time_to_expiry=t_years,
                risk_free_rate=r,
                option_type="put" if option_type == "P" else "call",
            )
            if iv_short is None:
                continue

            greeks_short = bs_greeks(
                spot=spot,
                strike=short_strike,
                time_to_expiry=t_years,
                risk_free_rate=r,
                volatility=iv_short,
                option_type="put" if option_type == "P" else "call",
            )
            delta_short = abs(float(greeks_short.get("delta", 0.0)))

            if not self._delta_in_range(delta_short):
                continue

            iv_long = implied_volatility(
                price=long_mid,
                spot=spot,
                strike=long_strike,
                time_to_expiry=t_years,
                risk_free_rate=r,
                option_type="put" if option_type == "P" else "call",
            )
            if iv_long is None:
                continue

            width = abs(short_strike - long_strike)
            if width <= 0:
                continue

            if is_credit:
                credit = short_mid - long_mid
                if credit <= 0:
                    continue
                credit_pct = credit / width
                if not self._credit_in_range(credit_pct):
                    continue
                if iv_short < float(self.params["iv_min_credit"]):
                    continue
                max_loss = (width - credit) * 100.0
                if max_loss <= 0:
                    continue
                score = credit / max_loss
                limit_price = -round(credit, 2)
                reason = f"{spread_type} credit {credit:.2f} ({credit_pct:.0%} width), IV {iv_short:.0%}"
                net_price = credit
            else:
                debit = long_mid - short_mid
                if debit <= 0:
                    continue
                debit_pct = debit / width
                if not self._debit_in_range(debit_pct):
                    continue
                if iv_long > float(self.params["iv_max_debit"]):
                    continue
                max_loss = debit * 100.0
                score = (width - debit) / width
                limit_price = round(debit, 2)
                reason = f"{spread_type} debit {debit:.2f} ({debit_pct:.0%} width), IV {iv_long:.0%}"
                net_price = debit

            metadata = {
                "order_class": "mleg",
                "spread_type": spread_type,
                "underlying_symbol": symbol,
                "legs": [
                    {
                        "option_type": "put" if option_type == "P" else "call",
                        "strike": short_strike,
                        "expiration": exp.isoformat(),
                        "side": short_side,
                        "ratio": 1,
                    },
                    {
                        "option_type": "put" if option_type == "P" else "call",
                        "strike": long_strike,
                        "expiration": exp.isoformat(),
                        "side": long_side,
                        "ratio": 1,
                    },
                ],
                "net_price": net_price,
                "width": width,
                "max_loss": max_loss,
                "iv_short": iv_short,
                "iv_long": iv_long,
                "delta_short": delta_short,
                "time_in_force": "day",
                "dte": dte,
            }

            action = "SELL_TO_OPEN" if is_credit else "BUY_TO_OPEN"
            candidate = _SpreadCandidate(
                spread_type=spread_type,
                action=action,
                limit_price=limit_price,
                score=score,
                reason=reason,
                metadata=metadata,
                max_loss=max_loss,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        return best

    def _spread_strikes(
        self,
        spread_type: str,
        strikes: List[float],
        spot: float,
        otm_pct: float,
        width_target: float,
    ) -> Tuple[Optional[float], Optional[float], str, str, bool]:
        if spread_type == "bull_put":
            short_target = spot * (1 - otm_pct)
            short_strike = self._pick_strike(strikes, short_target, "lte")
            if short_strike is None:
                return None, None, "", "", True
            long_target = short_strike - width_target
            long_strike = self._pick_strike(strikes, long_target, "lte")
            return short_strike, long_strike, "sell", "buy", True

        if spread_type == "bull_call":
            long_target = spot * (1 + otm_pct)
            long_strike = self._pick_strike(strikes, long_target, "gte")
            if long_strike is None:
                return None, None, "", "", False
            short_target = long_strike + width_target
            short_strike = self._pick_strike(strikes, short_target, "gte")
            return short_strike, long_strike, "sell", "buy", False

        if spread_type == "bear_call":
            short_target = spot * (1 + otm_pct)
            short_strike = self._pick_strike(strikes, short_target, "gte")
            if short_strike is None:
                return None, None, "", "", True
            long_target = short_strike + width_target
            long_strike = self._pick_strike(strikes, long_target, "gte")
            return short_strike, long_strike, "sell", "buy", True

        if spread_type == "bear_put":
            long_target = spot * (1 - otm_pct)
            long_strike = self._pick_strike(strikes, long_target, "lte")
            if long_strike is None:
                return None, None, "", "", False
            short_target = long_strike - width_target
            short_strike = self._pick_strike(strikes, short_target, "lte")
            return short_strike, long_strike, "sell", "buy", False

        return None, None, "", "", False

    def _delta_in_range(self, delta: float) -> bool:
        return float(self.params["short_delta_min"]) <= delta <= float(self.params["short_delta_max"])

    def _credit_in_range(self, credit_pct: float) -> bool:
        return float(self.params["credit_min_pct"]) <= credit_pct <= float(self.params["credit_max_pct"])

    def _debit_in_range(self, debit_pct: float) -> bool:
        return float(self.params["debit_min_pct"]) <= debit_pct <= float(self.params["debit_max_pct"])
