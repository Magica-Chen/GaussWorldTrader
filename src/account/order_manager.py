"""
Order Management for Alpaca Trading

Handles order placement, tracking, and management
"""

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List

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

        orders = self.account_manager._request_json(
            "GET",
            "/v2/orders",
            action="Retrieve orders",
            params=params,
        )
        self.logger.info("Retrieved %s orders", len(orders))
        return orders
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get specific order by ID"""
        order = self.account_manager._request_json(
            "GET",
            f"/v2/orders/{order_id}",
            action=f"Retrieve order {order_id}",
        )
        self.logger.info("Retrieved order %s", order_id)
        return order
    
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
        
        order = self.account_manager._request_json(
            "POST",
            "/v2/orders",
            action=f"Place order for {symbol}",
            json=order_data,
        )
        self.logger.info("Order placed: %s for %s", order.get("id"), symbol)
        return order
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a specific order"""
        self.account_manager._request_json(
            "DELETE",
            f"/v2/orders/{order_id}",
            action=f"Cancel order {order_id}",
            allow_empty=True,
        )
        self.logger.info("Order %s cancelled", order_id)
        return {"success": True, "order_id": order_id}
    
    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all open orders"""
        results = self.account_manager._request_json(
            "DELETE",
            "/v2/orders",
            action="Cancel all orders",
        )
        self.logger.info("All orders cancelled")
        return {"success": True, "cancelled_orders": results}
    
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

        order = self.account_manager._request_json(
            "PATCH",
            f"/v2/orders/{order_id}",
            action=f"Replace order {order_id}",
            json=replace_data,
        )
        self.logger.info("Order %s replaced", order_id)
        return order

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
            Order response dict
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

        order = self.account_manager._request_json(
            "POST",
            "/v2/orders",
            action=f"Place bracket order for {symbol}",
            json=order_data,
        )
        self.logger.info(
            "Bracket order placed: %s for %s (SL: %s, TP: %s)",
            order.get("id"),
            symbol,
            stop_loss,
            take_profit,
        )
        return order
    
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
                
            except (ValueError, TypeError) as exc:
                order_id = order.get("id", "unknown")
                raise ValueError(f"Invalid order data for {order_id}: {exc}") from exc
        
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
        if not orders:
            return "No recent orders found"
        
        summary = f"""
🌍 GAUSS WORLD TRADER - RECENT ORDERS
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Showing {min(len(orders), limit)} most recent orders

"""
        
        for i, order in enumerate(orders[:limit], 1):
            order_id = order.get('id', 'N/A')[:8] + '...'  # Truncate ID
            symbol = order.get('symbol', 'N/A')
            side = order.get('side', 'N/A').upper()
            qty = float(order.get('qty', 0))
            order_type = order.get('type', 'N/A').upper()
            status = order.get('status', 'N/A').upper()
            submitted_at = order.get('submitted_at', '')

            # Keep the raw timestamp if formatting fails, but do not suppress missing order data.
            if submitted_at:
                try:
                    dt = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
                    time_str = dt.strftime('%m/%d %H:%M')
                except (ValueError, TypeError):
                    time_str = str(submitted_at)[:16]
            else:
                time_str = 'N/A'

            summary += f"{i:2d}. {order_id} | {symbol:>6} | {side:>4} {qty:>8.0f} | {order_type:>7} | {status:>10} | {time_str}\n"
        
        return summary
    
    def get_orders_analysis_summary(self, days: int = 30) -> str:
        """Get formatted order analysis summary"""
        analysis = self.analyze_orders(days)
        total_orders = analysis["total_orders"]
        pct = lambda count: (count / total_orders * 100) if total_orders else 0
        summary = f"""
🌍 GAUSS WORLD TRADER - ORDER ANALYSIS
=====================================
Analysis Period: Last {days} days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ORDER OVERVIEW:
--------------
• Total Orders: {analysis['total_orders']}
• Filled Orders: {analysis['filled_orders']} ({pct(analysis['filled_orders']):.1f}%)
• Cancelled Orders: {analysis['cancelled_orders']} ({pct(analysis['cancelled_orders']):.1f}%)
• Pending Orders: {analysis['pending_orders']}
• Rejected Orders: {analysis['rejected_orders']}

ORDER BREAKDOWN:
---------------
• Buy Orders: {analysis['buy_orders']} ({pct(analysis['buy_orders']):.1f}%)
• Sell Orders: {analysis['sell_orders']} ({pct(analysis['sell_orders']):.1f}%)
• Market Orders: {analysis['market_orders']} ({pct(analysis['market_orders']):.1f}%)
• Limit Orders: {analysis['limit_orders']} ({pct(analysis['limit_orders']):.1f}%)

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
