"""Abstract base trading engine with common functionality for all asset types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.agent.notification_service import NotificationService

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import Config
from .portfolio import Portfolio


class TradingEngine(ABC):
    """Abstract base trading engine for Alpaca API integration."""

    def __init__(self, paper_trading: bool = True,
                 notification_service: "NotificationService" = None) -> None:
        if not Config.validate_alpaca_config():
            raise ValueError("Alpaca API credentials not configured")

        self.api = TradingClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY,
            paper=Config.ALPACA_BASE_URL != "https://api.alpaca.markets"
        )

        self.paper_trading = paper_trading
        self.portfolio = Portfolio()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._notification_service = notification_service

        if paper_trading:
            self.logger.info("Trading engine initialized in PAPER TRADING mode")
        else:
            self.logger.warning("Trading engine initialized in LIVE TRADING mode")

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format. Override in subclasses for asset-specific handling."""
        return symbol.strip().upper()

    def _notify_order(self, order_dict: Dict[str, Any]) -> None:
        """Send notification for order submission if notification service is configured."""
        if self._notification_service:
            self._notification_service.notify_order_submitted(order_dict)

    def validate_order(self, symbol: str, qty: float, side: str) -> None:
        """Validate order before submission. Override in subclasses for asset-specific rules."""
        if qty <= 0:
            raise ValueError(f"Order quantity must be positive, got {qty}")

    @abstractmethod
    def place_market_order(self, symbol: str, qty: float, side: str = 'buy',
                          time_in_force: str = 'gtc') -> Dict[str, Any]:
        """Place a market order. Implementation varies by asset type."""
        pass

    @abstractmethod
    def place_limit_order(self, symbol: str, qty: float, limit_price: float,
                         side: str = 'buy', time_in_force: str = 'gtc') -> Dict[str, Any]:
        """Place a limit order. Implementation varies by asset type."""
        pass

    def place_stop_loss_order(self, symbol: str, qty: float, stop_price: float,
                             side: str = 'sell', time_in_force: str = 'gtc') -> Dict[str, Any]:
        """Place a stop loss order."""
        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, qty, side)

        order_request = StopOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if time_in_force == 'gtc' else TimeInForce.DAY,
            stop_price=stop_price
        )
        order = self.api.submit_order(order_request)

        order_dict = {
            'id': order.id,
            'symbol': order.symbol,
            'qty': float(order.qty),
            'side': order.side,
            'type': order.type,
            'stop_price': float(order.stop_price),
            'status': order.status,
            'submitted_at': order.submitted_at
        }

        self.logger.info(f"Stop loss order placed: {side} {qty} of {symbol} at ${stop_price}")
        self._notify_order(order_dict)
        return order_dict

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self.api.cancel_order_by_id(order_id)
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of an order."""
        try:
            order = self.api.get_order_by_id(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return {}

    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get all open orders, optionally filtered by symbol."""
        try:
            orders = self.api.get_orders(status='open')
            result = []
            for order in orders:
                if symbol and order.symbol != self.normalize_symbol(symbol):
                    continue
                result.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'submitted_at': order.submitted_at,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None
                })
            return result
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'multiplier': float(getattr(account, 'multiplier', 1) or 1),
                'buying_power': float(account.buying_power),
                'daytrading_buying_power': float(getattr(account, 'daytrading_buying_power', 0) or 0),
                'non_marginable_buying_power': float(
                    getattr(account, 'non_marginable_buying_power', 0) or 0
                ),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'daytrade_count': int(getattr(account, 'daytrade_count', 0)),
                'day_trade_count': int(getattr(account, 'daytrade_count', 0)),
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False),
                'trading_blocked': getattr(account, 'trading_blocked', False),
                'transfers_blocked': getattr(account, 'transfers_blocked', False),
                'account_blocked': getattr(account, 'account_blocked', False),
                'status': getattr(account, 'status', 'UNKNOWN')
            }
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {}

    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        try:
            from src.account.position_manager import convert_crypto_symbol_for_display
            positions = self.api.get_all_positions()
            return [{
                'symbol': convert_crypto_symbol_for_display(pos.symbol),
                'qty': float(pos.qty),
                'side': pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price) if pos.current_price else None
            } for pos in positions]
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []

    def close_position(self, symbol: str, percentage: float = 1.0) -> Dict[str, Any]:
        """Close a position (fully or partially)."""
        positions = self.get_current_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)

        if not position:
            raise ValueError(f"No position found for symbol {symbol}")

        qty_to_close = abs(float(position['qty'])) * percentage
        side = 'sell' if float(position['qty']) > 0 else 'buy'

        return self.place_market_order(symbol, qty_to_close, side)

    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all open positions."""
        results = []
        positions = self.get_current_positions()

        for position in positions:
            try:
                result = self.close_position(position['symbol'])
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to close position for {position['symbol']}: {e}")

        return results

    def _has_position(self, symbol: str) -> bool:
        """Check if there is an existing position for the symbol."""
        positions = self.get_current_positions()
        return any(p['symbol'] == symbol and float(p['qty']) != 0 for p in positions)

    def _build_order_dict(self, order: Any) -> Dict[str, Any]:
        """Build standardized order dictionary from Alpaca order object."""
        return {
            'id': order.id,
            'symbol': order.symbol,
            'qty': float(order.qty),
            'side': order.side,
            'type': order.type,
            'status': order.status,
            'submitted_at': order.submitted_at,
            'filled_at': order.filled_at,
            'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'stop_price': float(order.stop_price) if order.stop_price else None
        }
