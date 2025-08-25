import alpaca_trade_api as tradeapi
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from config import Config
from .portfolio import Portfolio

class TradingEngine:
    def __init__(self, paper_trading: bool = True):
        if not Config.validate_alpaca_config():
            raise ValueError("Alpaca API credentials not configured")
        
        self.api = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        self.paper_trading = paper_trading
        self.portfolio = Portfolio()
        self.logger = logging.getLogger(__name__)
        
        if paper_trading:
            self.logger.info("Trading engine initialized in PAPER TRADING mode")
        else:
            self.logger.warning("Trading engine initialized in LIVE TRADING mode")
    
    def place_market_order(self, symbol: str, qty: int, side: str = 'buy', 
                          time_in_force: str = 'gtc') -> Dict[str, Any]:
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side.lower(),
                type='market',
                time_in_force=time_in_force
            )
            
            order_dict = {
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
            
            self.logger.info(f"Market order placed: {side} {qty} shares of {symbol}")
            return order_dict
            
        except Exception as e:
            self.logger.error(f"Failed to place market order: {e}")
            raise
    
    def place_limit_order(self, symbol: str, qty: int, limit_price: float, 
                         side: str = 'buy', time_in_force: str = 'gtc') -> Dict[str, Any]:
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side.lower(),
                type='limit',
                time_in_force=time_in_force,
                limit_price=limit_price
            )
            
            order_dict = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'limit_price': float(order.limit_price),
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
            self.logger.info(f"Limit order placed: {side} {qty} shares of {symbol} at ${limit_price}")
            return order_dict
            
        except Exception as e:
            self.logger.error(f"Failed to place limit order: {e}")
            raise
    
    def place_stop_loss_order(self, symbol: str, qty: int, stop_price: float,
                             side: str = 'sell', time_in_force: str = 'gtc') -> Dict[str, Any]:
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side.lower(),
                type='stop',
                time_in_force=time_in_force,
                stop_price=stop_price
            )
            
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
            
            self.logger.info(f"Stop loss order placed: {side} {qty} shares of {symbol} at ${stop_price}")
            return order_dict
            
        except Exception as e:
            self.logger.error(f"Failed to place stop loss order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        try:
            order = self.api.get_order(order_id)
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
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        try:
            orders = self.api.list_orders(status='open')
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None
            } for order in orders]
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'day_trade_count': int(getattr(account, 'day_trade_count', 0)),
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
        try:
            from src.utils.validators import convert_crypto_symbol_for_display
            positions = self.api.list_positions()
            return [{
                'symbol': convert_crypto_symbol_for_display(pos.symbol),
                'qty': float(pos.qty),
                'side': pos.side,
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
        try:
            positions = self.get_current_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position:
                raise ValueError(f"No position found for symbol {symbol}")
            
            qty_to_close = int(abs(float(position['qty'])) * percentage)
            side = 'sell' if float(position['qty']) > 0 else 'buy'
            
            return self.place_market_order(symbol, qty_to_close, side)
            
        except Exception as e:
            self.logger.error(f"Failed to close position for {symbol}: {e}")
            raise
    
    def close_all_positions(self) -> List[Dict[str, Any]]:
        results = []
        positions = self.get_current_positions()
        
        for position in positions:
            try:
                result = self.close_position(position['symbol'])
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to close position for {position['symbol']}: {e}")
        
        return results