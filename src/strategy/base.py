"""
Shared strategy base classes and metadata.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration for strategies."""

    stop_loss_pct: float = 0.03      # 3% stop-loss
    take_profit_pct: float = 0.06    # 6% take-profit (2:1 risk/reward)
    position_size_pct: float = 0.10  # 10% of portfolio per trade


@dataclass(frozen=True)
class StrategyMeta:
    """Metadata that describes a strategy."""

    name: str
    label: str
    category: str
    description: str
    asset_type: str = "stock"
    visible_in_dashboard: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategySignal:
    """Normalized signal output for backtests and live runs.

    Core fields work for all asset types. Optional fields support options trading
    without breaking existing stock/crypto strategies.
    """

    # Core fields (all asset types)
    symbol: str
    action: str  # BUY|SELL|SELL_TO_OPEN|BUY_TO_CLOSE|ROLL|HOLD
    quantity: float
    price: Optional[float] = None
    reason: str = ""
    timestamp: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Option-specific fields (None for stocks/crypto)
    underlying_symbol: Optional[str] = None
    option_type: Optional[str] = None  # "put" or "call"
    strike_price: Optional[float] = None
    expiration_date: Optional[datetime] = None
    delta: Optional[float] = None
    premium: Optional[float] = None
    strategy_stage: Optional[str] = None  # e.g., "cash_secured_put", "covered_call"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }
        # Include option fields only if present
        if self.underlying_symbol is not None:
            result["underlying_symbol"] = self.underlying_symbol
        if self.option_type is not None:
            result["option_type"] = self.option_type
        if self.strike_price is not None:
            result["strike_price"] = self.strike_price
        if self.expiration_date is not None:
            result["expiration_date"] = self.expiration_date
        if self.delta is not None:
            result["delta"] = self.delta
        if self.premium is not None:
            result["premium"] = self.premium
        if self.strategy_stage is not None:
            result["strategy_stage"] = self.strategy_stage
        return result


@dataclass
class MarketDataContext:
    """Container for all market data passed to strategy.generate_signals().

    This provides a unified interface for data injection, enabling pure strategies
    that don't need to fetch their own data. Supports stocks, crypto, and options.
    """

    current_prices: Dict[str, float]
    historical_bars: Dict[str, pd.DataFrame]
    portfolio_value: float = 100000.0
    available_cash: float = 100000.0
    current_positions: Optional[Dict[str, Any]] = None
    # Option-specific data injection (Phase 2)
    options_chains: Optional[Dict[str, pd.DataFrame]] = None
    option_positions: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class TradingPlanItem:
    """Structured trading plan item derived from strategy signals."""

    symbol: str
    action: str
    quantity: float
    price: Optional[float] = None
    reason: str = ""
    timestamp: Optional[datetime] = None
    asset_type: str = "stock"
    strategy: str = ""
    plan_type: str = "signal"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "asset_type": self.asset_type,
            "strategy": self.strategy,
            "plan_type": self.plan_type,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


class StrategyBase:
    """Base class for all strategies.

    Subclasses should implement generate_signals and update the meta field.
    """

    meta: StrategyMeta
    summary: str = ""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.meta.description.strip():
            raise ValueError(f"{self.__class__.__name__} must define a non-empty meta.description.")
        if not self._skip_summary_check() and not self.summary.strip():
            raise ValueError(f"{self.__class__.__name__} must define a non-empty summary.")
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

    def generate_trading_plan(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_data: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any = None,
    ) -> List[Dict[str, Any]]:
        """Return a structured trading plan for the current market snapshot."""
        signals = self.generate_signals(
            current_date,
            current_prices,
            current_data,
            historical_data,
            portfolio,
        )
        return self._plan_from_signals(signals, timestamp=current_date)

    def supports_dashboard(self) -> bool:
        return self.meta.visible_in_dashboard

    def _position_size(self, price: float, portfolio_value: float, risk_pct: float) -> float:
        if price <= 0 or portfolio_value <= 0:
            return 0.0
        return float(max(1, int((portfolio_value * risk_pct) / price)))

    def _normalize(self, signals: Iterable[StrategySignal]) -> List[Dict[str, Any]]:
        return [signal.to_dict() for signal in signals]

    def _plan_from_signals(
        self, signals: Iterable[Dict[str, Any]], timestamp: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        plan: List[TradingPlanItem] = []
        for signal in signals:
            quantity = signal.get("quantity", 0)
            quantity_value = float(quantity) if quantity is not None else 0.0
            plan.append(
                TradingPlanItem(
                    symbol=signal.get("symbol", ""),
                    action=signal.get("action", ""),
                    quantity=quantity_value,
                    price=signal.get("price"),
                    reason=signal.get("reason", ""),
                    timestamp=signal.get("timestamp") or timestamp,
                    asset_type=self.meta.asset_type,
                    strategy=self.meta.name,
                    plan_type="signal",
                    stop_loss=signal.get("stop_loss"),
                    take_profit=signal.get("take_profit"),
                )
            )
        return [item.to_dict() for item in plan]

    def get_risk_config(self) -> RiskConfig:
        """Get risk configuration from strategy params or defaults."""
        return RiskConfig(
            stop_loss_pct=float(self.params.get("stop_loss_pct", 0.03)),
            take_profit_pct=float(self.params.get("take_profit_pct", 0.06)),
            position_size_pct=float(self.params.get("risk_pct", 0.10)),
        )

    def calculate_stop_loss(self, price: float, side: str) -> float:
        """Calculate stop-loss price based on entry price and side.

        Args:
            price: Entry price
            side: 'long' or 'short'

        Returns:
            Stop-loss price
        """
        risk = self.get_risk_config()
        if side.lower() == "long":
            return price * (1 - risk.stop_loss_pct)
        return price * (1 + risk.stop_loss_pct)

    def calculate_take_profit(self, price: float, side: str) -> float:
        """Calculate take-profit price based on entry price and side.

        Args:
            price: Entry price
            side: 'long' or 'short'

        Returns:
            Take-profit price
        """
        risk = self.get_risk_config()
        if side.lower() == "long":
            return price * (1 + risk.take_profit_pct)
        return price * (1 - risk.take_profit_pct)

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
            "asset_type": self.meta.asset_type,
            "description": self.meta.description,
            "summary": self.summary,
            "parameters": self.params,
            "signals_generated": len(self.signals),
        }

    def _skip_summary_check(self) -> bool:
        return isinstance(self, BaseOptionStrategy) or self.__class__ is StrategyBase


class BaseOptionStrategy(StrategyBase, ABC):
    """
    Abstract base class for option trading strategies.

    This class extends BaseStrategy to include option-specific functionality
    such as option filtering, scoring, and management of option positions
    alongside stock positions.
    """

    meta = StrategyMeta(
        name="option_base",
        label="Option Base",
        category="option",
        description="Base class for option strategies.",
        asset_type="option",
        visible_in_dashboard=False,
        default_params={},
    )

    def __init__(self, parameters: Dict[str, Any] | None = None) -> None:
        """
        Initialize the base option strategy.

        Args:
            parameters: Strategy configuration parameters
        """
        super().__init__(parameters)

        # Option-specific state
        self.option_positions: Dict[str, Dict[str, Any]] = {}
        self.option_signals: List[Dict[str, Any]] = []

        # Load watchlist symbols instead of symbol_list.txt
        self.symbol_list = self._load_watchlist_symbols()

        # Default option strategy parameters
        self.default_params = {
            "max_risk": 80000,  # Maximum risk in dollars
            "delta_min": 0.15,  # Minimum delta (absolute value)
            "delta_max": 0.30,  # Maximum delta (absolute value)
            "yield_min": 0.04,  # Minimum yield (4%)
            "yield_max": 1.00,  # Maximum yield (100%)
            "dte_min": 0,  # Minimum days to expiration
            "dte_max": 21,  # Maximum days to expiration
            "min_open_interest": 100,  # Minimum open interest
            "min_score": 0.05,  # Minimum option score
            "position_size_pct": 0.1,  # Position size as % of portfolio
            "assignment_tolerance": 0.95,  # Tolerance for assignment risk
        }

        # Merge default params with provided params
        self.parameters = {**self.default_params, **self.parameters}

    def _load_watchlist_symbols(self) -> List[str]:
        """
        Load symbols from watchlist.json filtered to stock entries.

        Returns:
            List of symbols from the watchlist
        """
        from src.agent.watchlist_manager import WatchlistManager

        manager = WatchlistManager()
        symbols = manager.get_watchlist(asset_type="stock")
        self.logger.info("Loaded %s symbols from watchlist.json", len(symbols))
        return symbols

    @abstractmethod
    def filter_underlying_stocks(self, client: Any) -> List[str]:
        """
        Filter underlying stocks based on strategy criteria.

        Args:
            client: Alpaca trading client

        Returns:
            List of filtered stock symbols
        """

    @abstractmethod
    def filter_options(
        self, client: Any, underlying: str, option_type: str = "put"
    ) -> List[Dict[str, Any]]:
        """
        Filter options based on strategy criteria.

        Args:
            client: Alpaca trading client
            underlying: Stock symbol
            option_type: 'put' or 'call'

        Returns:
            List of filtered option contracts
        """

    @abstractmethod
    def score_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score options based on strategy-specific criteria.

        Args:
            options: List of option contracts

        Returns:
            List of options with scores added
        """

    @abstractmethod
    def select_best_options(
        self, scored_options: List[Dict[str, Any]], limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Select the best options based on scores.

        Args:
            scored_options: List of scored option contracts
            limit: Maximum number of options to select

        Returns:
            List of selected option contracts
        """

    def calculate_option_yield(self, option: Dict[str, Any]) -> float:
        """
        Calculate the yield of an option.

        Args:
            option: Option contract data

        Returns:
            Option yield as a percentage
        """
        try:
            bid_price = option.get("bid", 0)
            strike_price = option.get("strike_price", 0)

            if strike_price == 0:
                return 0.0

            # For puts: yield = bid / strike
            # For calls: yield = bid / (current_stock_price - strike) for ITM calls
            option_type = option.get("type", "put").lower()

            if option_type == "put":
                yield_pct = (bid_price / strike_price) * 100
            else:  # call
                current_price = option.get("underlying_price", strike_price)
                if current_price > strike_price:  # ITM call
                    yield_pct = (bid_price / (current_price - strike_price)) * 100
                else:  # OTM call
                    yield_pct = (bid_price / current_price) * 100

            return round(yield_pct, 2)

        except Exception as exc:
            self.logger.error("Error calculating option yield: %s", exc)
            return 0.0

    def calculate_option_score(self, option: Dict[str, Any]) -> float:
        """
        Calculate option score using the wheel strategy scoring formula.

        The scoring formula is:
        score = (1 - |Δ|) * (250 / (DTE + 5)) * (bid price / strike price)

        Args:
            option: Option contract data

        Returns:
            Option score
        """
        try:
            delta = abs(option.get("delta", 0))
            dte = option.get("days_to_expiration", 1)
            bid_price = option.get("bid", 0)
            strike_price = option.get("strike_price", 1)

            # Avoid division by zero
            if strike_price == 0 or dte < 0:
                return 0.0

            # Wheel strategy scoring formula
            delta_component = 1 - delta
            time_component = 250 / (dte + 5)
            yield_component = bid_price / strike_price

            score = delta_component * time_component * yield_component

            return round(score, 4)

        except Exception as exc:
            self.logger.error("Error calculating option score: %s", exc)
            return 0.0

    def check_option_assignment_risk(self, option: Dict[str, Any], underlying_price: float) -> Dict[str, Any]:
        """
        Check assignment risk for an option position.

        Args:
            option: Option contract data
            underlying_price: Current price of underlying stock

        Returns:
            Assignment risk analysis
        """
        try:
            option_type = option.get("type", "put").lower()
            strike_price = option.get("strike_price", 0)
            expiration_date = option.get("expiration_date")

            # Calculate days to expiration
            if isinstance(expiration_date, str):
                exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            else:
                exp_date = expiration_date

            days_to_exp = (exp_date - datetime.now()).days

            # Assignment probability estimation
            if option_type == "put":
                # Put is ITM if underlying < strike
                is_itm = underlying_price < strike_price
                distance_from_strike = abs(underlying_price - strike_price) / strike_price
            else:  # call
                # Call is ITM if underlying > strike
                is_itm = underlying_price > strike_price
                distance_from_strike = abs(underlying_price - strike_price) / strike_price

            # Simple assignment risk estimation
            if is_itm:
                assignment_prob = min(0.9, 0.5 + (0.4 / max(1, days_to_exp)))
            else:
                assignment_prob = max(0.1, distance_from_strike * 0.3)

            return {
                "assignment_probability": round(assignment_prob, 2),
                "days_to_expiration": days_to_exp,
                "is_itm": is_itm,
                "distance_from_strike": round(distance_from_strike, 4),
            }
        except Exception as exc:
            self.logger.error("Error checking assignment risk: %s", exc)
            return {
                "assignment_probability": 0.0,
                "days_to_expiration": 0,
                "is_itm": False,
                "distance_from_strike": 0.0,
            }
