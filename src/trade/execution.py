"""Execution layer for converting action plans into concrete orders."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import logging
from typing import Any, Dict, Optional

from alpaca.trading.requests import OptionLegRequest
from alpaca.trading.enums import OrderSide

from config import Config
from src.account.account_manager import AccountManager
from src.strategy.base import ActionPlan

from .trading_engine import TradingEngine
from .option_engine import TradingOptionEngine


@dataclass(frozen=True)
class ExecutionContext:
    account_info: Dict[str, Any]
    account_config: Dict[str, Any]
    buying_power: float
    cash: float
    portfolio_value: float
    margin_enabled: bool
    fractional_enabled: bool
    shorting_enabled: bool
    account_type: str


@dataclass(frozen=True)
class ExecutionDecision:
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: Optional[float]
    action: str
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """Turn action plans into executable orders with account-aware sizing."""

    def __init__(
        self,
        trading_engine: TradingEngine,
        asset_type: str,
        allow_sell_to_open: bool = False,
        order_type: str = "auto",
        execute: bool = True,
        account_manager: Optional[AccountManager] = None,
    ) -> None:
        self.trading_engine = trading_engine
        self.asset_type = asset_type
        self.allow_sell_to_open = allow_sell_to_open
        self.order_type = order_type
        self.execute = execute
        self.logger = logging.getLogger(f"ExecutionEngine.{asset_type}")

        if account_manager is None:
            base_url = getattr(Config, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            account_manager = AccountManager(base_url=base_url)
        self.account_manager = account_manager

    def load_context(self) -> ExecutionContext:
        account_info = self.trading_engine.get_account_info() or {}
        account_raw = self.account_manager.get_account() or {}
        account_config = self.account_manager.get_account_configurations() or {}
        if "error" in account_config:
            account_config = {}
        if "error" in account_raw:
            account_raw = {}

        account_type = str(
            account_raw.get("account_type") or account_info.get("account_type") or ""
        ).upper()
        multiplier = self._safe_float(account_raw.get("multiplier"), default=1.0)
        margin_enabled = account_type == "MARGIN" or multiplier > 1.0

        shorting_enabled = not bool(account_config.get("no_shorting", False))
        fractional_enabled = bool(account_config.get("fractional_trading", False))

        buying_power = self._safe_float(
            account_info.get("buying_power") or account_raw.get("buying_power")
        )
        cash = self._safe_float(account_info.get("cash") or account_raw.get("cash"))
        portfolio_value = self._safe_float(
            account_info.get("portfolio_value")
            or account_raw.get("portfolio_value")
            or account_info.get("equity")
            or account_raw.get("equity"),
            default=100000.0,
        )

        return ExecutionContext(
            account_info=account_info,
            account_config=account_config,
            buying_power=buying_power,
            cash=cash,
            portfolio_value=portfolio_value,
            margin_enabled=margin_enabled,
            fractional_enabled=fractional_enabled,
            shorting_enabled=shorting_enabled,
            account_type=account_type,
        )

    def build_decision(
        self,
        action_plan: ActionPlan,
        context: ExecutionContext,
        position: Any,
        risk_pct: float,
        current_price: float,
        override_qty: Optional[float] = None,
        order_pref: Optional[str] = None,
    ) -> Optional[ExecutionDecision]:
        action = (action_plan.action or "HOLD").upper()
        if action == "HOLD":
            return None

        metadata = action_plan.metadata or {}

        pos_side = self._position_side(position)
        pos_qty = abs(self._position_qty(position))

        intent, side = self._resolve_intent(action, pos_side)
        if intent is None:
            return None

        if intent == "open_short":
            blocked = self._short_block_reason(context)
            if blocked:
                self.logger.info("Short blocked for %s: %s", action_plan.symbol, blocked)
                return None

        if intent == "close_long" and pos_qty == 0:
            return None
        if intent == "close_short" and pos_qty == 0:
            return None
        if intent == "open_long" and pos_side == "long":
            return None
        if intent == "open_short" and pos_side == "short":
            return None

        if override_qty is None and metadata:
            meta_override = metadata.get("override_qty")
            if meta_override is not None:
                override_qty = float(meta_override)

        quantity = self._resolve_quantity(
            action_plan=action_plan,
            intent=intent,
            position_qty=pos_qty,
            context=context,
            risk_pct=risk_pct,
            current_price=current_price,
            override_qty=override_qty,
        )
        if quantity <= 0:
            return None

        if metadata.get("order_class") == "mleg":
            order_type = "limit"
            limit_price = action_plan.target_price
            if limit_price is None:
                return None
        else:
            order_type, limit_price = self._resolve_order_type(
                action_plan, side, current_price, order_pref
            )

        return ExecutionDecision(
            symbol=action_plan.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            action=action,
            reason=action_plan.reason,
            stop_loss=action_plan.stop_loss,
            take_profit=action_plan.take_profit,
            metadata=metadata,
        )

    def execute_decision(self, decision: ExecutionDecision) -> bool:
        if decision is None:
            return False
        if not self.execute:
            if decision.metadata.get("order_class") == "mleg":
                self.logger.info(
                    "DRY RUN: would submit multi-leg order %s qty %s at %s",
                    decision.symbol,
                    decision.quantity,
                    decision.limit_price,
                )
            else:
                self.logger.info(
                    "DRY RUN: would %s %s %s (%s)",
                    decision.side,
                    decision.quantity,
                    decision.symbol,
                    decision.order_type,
                )
            return False

        if isinstance(self.trading_engine, TradingOptionEngine):
            if decision.metadata.get("order_class") == "mleg":
                return self._execute_option_mleg(decision)
            return self._execute_option_decision(decision)

        if decision.order_type == "limit" and decision.limit_price is not None:
            self.trading_engine.place_limit_order(
                decision.symbol,
                decision.quantity,
                decision.limit_price,
                side=decision.side,
            )
        else:
            self.trading_engine.place_market_order(
                decision.symbol,
                decision.quantity,
                side=decision.side,
            )
        return True

    def _execute_option_mleg(self, decision: ExecutionDecision) -> bool:
        engine = self.trading_engine
        if not isinstance(engine, TradingOptionEngine):
            return False
        if decision.limit_price is None:
            self.logger.error("Multi-leg order missing limit price.")
            return False

        metadata = decision.metadata or {}
        legs_meta = metadata.get("legs") or []
        if not legs_meta:
            self.logger.error("Multi-leg order missing legs metadata.")
            return False

        underlying = metadata.get("underlying_symbol") or metadata.get("underlying") or decision.symbol
        legs: list[OptionLegRequest] = []
        for leg in legs_meta:
            symbol = leg.get("symbol")
            if not symbol:
                expiration = leg.get("expiration")
                strike = leg.get("strike")
                option_type = leg.get("option_type")
                if expiration is None or strike is None or option_type is None:
                    self.logger.error("Leg missing expiration/strike/option_type: %s", leg)
                    return False
                exp_date = self._parse_expiration(expiration)
                if exp_date is None:
                    self.logger.error("Could not parse expiration %s", expiration)
                    return False
                symbol = engine.find_option_contract_symbol(
                    underlying=underlying,
                    expiration=exp_date,
                    strike=float(strike),
                    contract_type=str(option_type),
                )

            ratio = int(leg.get("ratio") or leg.get("ratio_qty") or 1)
            side = str(leg.get("side") or "buy").lower()
            side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

            legs.append(
                OptionLegRequest(
                    symbol=symbol,
                    ratio_qty=ratio,
                    side=side_enum,
                )
            )

        time_in_force = metadata.get("time_in_force", "day")
        engine.submit_mleg_limit_order(
            legs=legs,
            qty=int(decision.quantity),
            limit_price=float(decision.limit_price),
            time_in_force=time_in_force,
        )
        return True

    @staticmethod
    def _parse_expiration(value: Any) -> Optional[date]:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d").date()
                except ValueError:
                    return None
        return None

    def _execute_option_decision(self, decision: ExecutionDecision) -> bool:
        engine = self.trading_engine
        if not isinstance(engine, TradingOptionEngine):
            return False

        limit_price = decision.limit_price if decision.order_type == "limit" else None
        action = decision.action
        if action == "SELL_TO_OPEN":
            engine.sell_to_open(decision.symbol, int(decision.quantity), limit_price)
        elif action == "BUY_TO_CLOSE":
            engine.buy_to_close(decision.symbol, int(decision.quantity), limit_price)
        elif action == "BUY_TO_OPEN":
            engine.buy_to_open(decision.symbol, int(decision.quantity), limit_price)
        elif action == "SELL_TO_CLOSE":
            engine.sell_to_close(decision.symbol, int(decision.quantity), limit_price)
        else:
            if decision.order_type == "limit" and limit_price is not None:
                engine.place_limit_order(
                    decision.symbol, int(decision.quantity), limit_price, side=decision.side
                )
            else:
                engine.place_market_order(
                    decision.symbol, int(decision.quantity), side=decision.side
                )
        return True

    def _resolve_intent(self, action: str, pos_side: str) -> tuple[Optional[str], Optional[str]]:
        if action in {"BUY", "BUY_TO_OPEN"}:
            if pos_side == "short":
                return "close_short", "buy"
            return "open_long", "buy"
        if action == "BUY_TO_CLOSE":
            return "close_short", "buy"
        if action in {"SELL", "SELL_TO_CLOSE"}:
            if pos_side == "long":
                return "close_long", "sell"
            return "open_short", "sell"
        if action == "SELL_TO_OPEN":
            return "open_short", "sell"
        return None, None

    def _short_block_reason(self, context: ExecutionContext) -> Optional[str]:
        if self.asset_type == "crypto":
            return "crypto does not support short selling"
        if not self.allow_sell_to_open:
            return "sell-to-open disabled by user"
        if not context.margin_enabled:
            return "account is not margin-enabled"
        if not context.shorting_enabled:
            return "shorting disabled by account configuration"
        return None

    def _resolve_quantity(
        self,
        action_plan: ActionPlan,
        intent: str,
        position_qty: float,
        context: ExecutionContext,
        risk_pct: float,
        current_price: float,
        override_qty: Optional[float],
    ) -> float:
        if intent in {"close_long", "close_short"}:
            if override_qty is None:
                return position_qty
            return min(position_qty, abs(float(override_qty)))

        if override_qty is not None:
            return self._adjust_for_fractional(abs(float(override_qty)), context)

        if current_price <= 0:
            return 0.0
        raw_qty = (context.portfolio_value * risk_pct) / current_price
        raw_qty = self._adjust_for_fractional(raw_qty, context)
        if raw_qty <= 0:
            return 0.0

        max_qty = self._max_affordable_qty(raw_qty, context, current_price)
        return min(raw_qty, max_qty)

    def _resolve_order_type(
        self,
        action_plan: ActionPlan,
        side: str,
        current_price: float,
        order_pref: Optional[str],
    ) -> tuple[str, Optional[float]]:
        preference = (order_pref or self.order_type or "auto").lower()
        if preference in {"market", "limit"}:
            if preference == "market":
                return "market", None
            limit_price = action_plan.target_price or current_price
            return "limit", self._improve_limit_price(limit_price, side)

        if action_plan.target_price is None:
            return "market", None

        return "limit", self._improve_limit_price(action_plan.target_price, side)

    def _improve_limit_price(self, price: float, side: str) -> float:
        increment = self._min_price_increment()
        if side == "buy":
            return price + increment
        return max(0.0, price - increment)

    def _min_price_increment(self) -> float:
        if self.asset_type == "crypto":
            return 0.0001
        return 0.01

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _max_affordable_qty(self, desired_qty: float, context: ExecutionContext, price: float) -> float:
        buying_power = context.buying_power if self.asset_type != "crypto" else context.cash
        if buying_power <= 0 or price <= 0:
            return 0.0
        max_qty = buying_power / price
        if context.fractional_enabled or self.asset_type == "crypto":
            return max_qty
        return float(int(max_qty))

    def _adjust_for_fractional(self, quantity: float, context: ExecutionContext) -> float:
        if quantity <= 0:
            return 0.0
        if self.asset_type == "crypto":
            return quantity
        if context.fractional_enabled:
            return quantity
        return float(int(quantity))

    @staticmethod
    def _position_side(position: Any) -> str:
        if hasattr(position, "side"):
            return getattr(position, "side") or "flat"
        if isinstance(position, dict):
            return position.get("side", "flat")
        return "flat"

    @staticmethod
    def _position_qty(position: Any) -> float:
        if hasattr(position, "qty"):
            return float(getattr(position, "qty") or 0.0)
        if isinstance(position, dict):
            return float(position.get("qty") or 0.0)
        return 0.0
