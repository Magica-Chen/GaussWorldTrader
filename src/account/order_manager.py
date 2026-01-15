"""
Order Management for Alpaca Trading

Handles order placement, tracking, and management
"""

import requests
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

class OrderManager:
    """Manages trading orders"""
    
    def __init__(self, account_manager):
        self.account_manager = account_manager
        self.logger = logging.getLogger(__name__)
    
    def get_orders(self, status: str = 'all', symbols: List[str] = None,
                   start_date: str = None, end_date: str = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get orders with optional filters"""
        params = {
            'status': status,
            'limit': limit
        }
        
        if symbols:
            params['symbols'] = ','.join(symbols)
        if start_date:
            params['after'] = start_date
        if end_date:
            params['until'] = end_date
        
        try:
            response = requests.get(
                f"{self.account_manager.base_url}/v2/orders",
                headers=self.account_manager.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            orders = response.json()
            self.logger.info(f"Retrieved {len(orders)} orders")
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error retrieving orders: {e}")
            return [{"error": str(e)}]
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get specific order by ID"""
        try:
            response = requests.get(
                f"{self.account_manager.base_url}/v2/orders/{order_id}",
                headers=self.account_manager.headers,
                timeout=10
            )
            response.raise_for_status()
            
            order = response.json()
            self.logger.info(f"Retrieved order {order_id}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error retrieving order {order_id}: {e}")
            return {"error": str(e)}
    
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = 'market',
                   time_in_force: str = 'day', limit_price: float = None,
                   stop_price: float = None, trail_price: float = None,
                   trail_percent: float = None, extended_hours: bool = False) -> Dict[str, Any]:
        """Place a trading order"""
        
        order_data = {
            'symbol': symbol,
            'qty': str(abs(qty)),
            'side': side.lower(),
            'type': order_type.lower(),
            'time_in_force': time_in_force.lower(),
            'extended_hours': extended_hours
        }
        
        # Add price parameters based on order type
        if order_type.lower() in ['limit', 'stop_limit'] and limit_price:
            order_data['limit_price'] = str(limit_price)
        
        if order_type.lower() in ['stop', 'stop_limit'] and stop_price:
            order_data['stop_price'] = str(stop_price)
        
        if trail_price:
            order_data['trail_price'] = str(trail_price)
        
        if trail_percent:
            order_data['trail_percent'] = str(trail_percent)
        
        try:
            response = requests.post(
                f"{self.account_manager.base_url}/v2/orders",
                headers=self.account_manager.headers,
                json=order_data,
                timeout=10
            )
            response.raise_for_status()
            
            order = response.json()
            self.logger.info(f"Order placed: {order.get('id')} for {symbol}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            return {"error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a specific order"""
        try:
            response = requests.delete(
                f"{self.account_manager.base_url}/v2/orders/{order_id}",
                headers=self.account_manager.headers,
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info(f"Order {order_id} cancelled")
            return {"success": True, "order_id": order_id}
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return {"error": str(e)}
    
    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all open orders"""
        try:
            response = requests.delete(
                f"{self.account_manager.base_url}/v2/orders",
                headers=self.account_manager.headers,
                timeout=10
            )
            response.raise_for_status()
            
            results = response.json()
            self.logger.info("All orders cancelled")
            
            return {"success": True, "cancelled_orders": results}
            
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return {"error": str(e)}
    
    def replace_order(self, order_id: str, qty: int = None, time_in_force: str = None,
                     limit_price: float = None, stop_price: float = None,
                     trail_price: float = None, trail_percent: float = None) -> Dict[str, Any]:
        """Replace/modify an existing order"""

        replace_data = {}

        if qty is not None:
            replace_data['qty'] = str(abs(qty))
        if time_in_force:
            replace_data['time_in_force'] = time_in_force.lower()
        if limit_price is not None:
            replace_data['limit_price'] = str(limit_price)
        if stop_price is not None:
            replace_data['stop_price'] = str(stop_price)
        if trail_price is not None:
            replace_data['trail_price'] = str(trail_price)
        if trail_percent is not None:
            replace_data['trail_percent'] = str(trail_percent)

        try:
            response = requests.patch(
                f"{self.account_manager.base_url}/v2/orders/{order_id}",
                headers=self.account_manager.headers,
                json=replace_data,
                timeout=10
            )
            response.raise_for_status()

            order = response.json()
            self.logger.info(f"Order {order_id} replaced")

            return order

        except Exception as e:
            self.logger.error(f"Error replacing order {order_id}: {e}")
            return {"error": str(e)}

    def place_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_loss: float,
        take_profit: float,
        time_in_force: str = "gtc",
    ) -> Dict[str, Any]:
        """Place a bracket order (market entry with stop-loss and take-profit).

        This places a market order with attached OTO (one-triggers-other) orders
        for stop-loss and take-profit.

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            time_in_force: Time in force for the order (default 'gtc')

        Returns:
            Order response dict or error dict
        """
        order_data = {
            "symbol": symbol,
            "qty": str(abs(qty)),
            "side": side.lower(),
            "type": "market",
            "time_in_force": time_in_force.lower(),
            "order_class": "bracket",
            "stop_loss": {"stop_price": str(stop_loss)},
            "take_profit": {"limit_price": str(take_profit)},
        }

        try:
            response = requests.post(
                f"{self.account_manager.base_url}/v2/orders",
                headers=self.account_manager.headers,
                json=order_data,
                timeout=10,
            )
            response.raise_for_status()

            order = response.json()
            self.logger.info(
                f"Bracket order placed: {order.get('id')} for {symbol} "
                f"(SL: {stop_loss}, TP: {take_profit})"
            )

            return order

        except Exception as e:
            self.logger.error(f"Error placing bracket order for {symbol}: {e}")
            return {"error": str(e)}
    
    def analyze_orders(self, days: int = 30) -> Dict[str, Any]:
        """Analyze order history"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        orders = self.get_orders(
            status='all',
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            limit=500
        )
        
        if not orders or (len(orders) == 1 and 'error' in orders[0]):
            return {"error": "No orders found or error retrieving orders"}
        
        analysis = {
            'total_orders': len(orders),
            'filled_orders': 0,
            'cancelled_orders': 0,
            'pending_orders': 0,
            'rejected_orders': 0,
            'buy_orders': 0,
            'sell_orders': 0,
            'market_orders': 0,
            'limit_orders': 0,
            'order_types': {},
            'order_statuses': {},
            'symbols_traded': set(),
            'total_volume': 0,
            'avg_order_size': 0,
            'largest_order': 0,
            'most_active_symbols': {}
        }
        
        valid_orders = []
        
        for order in orders:
            try:
                status = order.get('status', '').lower()
                side = order.get('side', '').lower()
                order_type = order.get('type', '').lower()
                symbol = order.get('symbol', '')
                qty = float(order.get('qty', 0))
                
                # Count by status
                if status == 'filled':
                    analysis['filled_orders'] += 1
                elif status == 'cancelled':
                    analysis['cancelled_orders'] += 1
                elif status in ['new', 'partially_filled', 'pending_new']:
                    analysis['pending_orders'] += 1
                elif status in ['rejected', 'canceled']:
                    analysis['rejected_orders'] += 1
                
                # Count by side
                if side == 'buy':
                    analysis['buy_orders'] += 1
                elif side == 'sell':
                    analysis['sell_orders'] += 1
                
                # Count by type
                if order_type == 'market':
                    analysis['market_orders'] += 1
                elif order_type == 'limit':
                    analysis['limit_orders'] += 1
                
                # Track order types and statuses
                analysis['order_types'][order_type] = analysis['order_types'].get(order_type, 0) + 1
                analysis['order_statuses'][status] = analysis['order_statuses'].get(status, 0) + 1
                
                # Track symbols
                if symbol:
                    analysis['symbols_traded'].add(symbol)
                    analysis['most_active_symbols'][symbol] = analysis['most_active_symbols'].get(symbol, 0) + 1
                
                # Volume analysis
                if qty > 0:
                    analysis['total_volume'] += qty
                    if qty > analysis['largest_order']:
                        analysis['largest_order'] = qty
                    valid_orders.append(qty)
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error processing order {order.get('id', 'unknown')}: {e}")
                continue
        
        # Calculate averages
        if valid_orders:
            analysis['avg_order_size'] = sum(valid_orders) / len(valid_orders)
        
        # Convert set to count
        analysis['unique_symbols'] = len(analysis['symbols_traded'])
        analysis['symbols_traded'] = list(analysis['symbols_traded'])
        
        # Sort most active symbols
        analysis['most_active_symbols'] = dict(
            sorted(analysis['most_active_symbols'].items(), 
                   key=lambda x: x[1], reverse=True)[:10]
        )
        
        return analysis
    
    def get_recent_orders_summary(self, limit: int = 10) -> str:
        """Get summary of recent orders"""
        orders = self.get_orders(status='all', limit=limit)
        
        if not orders or (len(orders) == 1 and 'error' in orders[0]):
            return "No recent orders found or error retrieving orders"
        
        summary = f"""
🌍 GAUSS WORLD TRADER - RECENT ORDERS
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Showing {min(len(orders), limit)} most recent orders

"""
        
        for i, order in enumerate(orders[:limit], 1):
            try:
                order_id = order.get('id', 'N/A')[:8] + '...'  # Truncate ID
                symbol = order.get('symbol', 'N/A')
                side = order.get('side', 'N/A').upper()
                qty = float(order.get('qty', 0))
                order_type = order.get('type', 'N/A').upper()
                status = order.get('status', 'N/A').upper()
                submitted_at = order.get('submitted_at', '')
                
                # Format timestamp
                if submitted_at:
                    try:
                        dt = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except:
                        time_str = submitted_at[:16]
                else:
                    time_str = 'N/A'
                
                summary += f"{i:2d}. {order_id} | {symbol:>6} | {side:>4} {qty:>8.0f} | {order_type:>7} | {status:>10} | {time_str}\n"
                
            except (ValueError, TypeError) as e:
                summary += f"{i:2d}. Error processing order: {e}\n"
        
        return summary
    
    def get_orders_analysis_summary(self, days: int = 30) -> str:
        """Get formatted order analysis summary"""
        analysis = self.analyze_orders(days)
        
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        summary = f"""
🌍 GAUSS WORLD TRADER - ORDER ANALYSIS
=====================================
Analysis Period: Last {days} days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ORDER OVERVIEW:
--------------
• Total Orders: {analysis['total_orders']}
• Filled Orders: {analysis['filled_orders']} ({analysis['filled_orders']/analysis['total_orders']*100:.1f}%)
• Cancelled Orders: {analysis['cancelled_orders']} ({analysis['cancelled_orders']/analysis['total_orders']*100:.1f}%)
• Pending Orders: {analysis['pending_orders']}
• Rejected Orders: {analysis['rejected_orders']}

ORDER BREAKDOWN:
---------------
• Buy Orders: {analysis['buy_orders']} ({analysis['buy_orders']/analysis['total_orders']*100:.1f}%)
• Sell Orders: {analysis['sell_orders']} ({analysis['sell_orders']/analysis['total_orders']*100:.1f}%)
• Market Orders: {analysis['market_orders']} ({analysis['market_orders']/analysis['total_orders']*100:.1f}%)
• Limit Orders: {analysis['limit_orders']} ({analysis['limit_orders']/analysis['total_orders']*100:.1f}%)

TRADING ACTIVITY:
----------------
• Unique Symbols: {analysis['unique_symbols']}
• Total Volume: {analysis['total_volume']:,.0f} shares
• Average Order Size: {analysis['avg_order_size']:,.0f} shares
• Largest Order: {analysis['largest_order']:,.0f} shares
"""
        
        # Most active symbols
        if analysis['most_active_symbols']:
            summary += """
MOST ACTIVE SYMBOLS:
-------------------
"""
            for symbol, count in list(analysis['most_active_symbols'].items())[:5]:
                summary += f"• {symbol}: {count} orders\n"
        
        return summary