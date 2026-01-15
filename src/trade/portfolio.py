from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import logging

class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
    def add_position(self, symbol: str, quantity: float, price: float, 
                    timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol in self.positions:
            current_qty = self.positions[symbol]['quantity']
            current_cost = self.positions[symbol]['cost_basis'] * current_qty
            new_cost = price * quantity
            total_qty = current_qty + quantity
            
            if total_qty != 0:
                new_avg_cost = (current_cost + new_cost) / total_qty
                self.positions[symbol] = {
                    'quantity': total_qty,
                    'cost_basis': new_avg_cost,
                    'last_price': price,
                    'last_updated': timestamp
                }
            else:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {
                'quantity': quantity,
                'cost_basis': price,
                'last_price': price,
                'last_updated': timestamp
            }
        
        cost = quantity * price
        self.cash -= cost
        
        self.transactions.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'timestamp': timestamp,
            'type': 'BUY' if quantity > 0 else 'SELL'
        })
    
    def remove_position(self, symbol: str, quantity: float, price: float,
                       timestamp: Optional[datetime] = None):
        if symbol not in self.positions:
            raise ValueError(f"No position found for symbol {symbol}")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        current_qty = self.positions[symbol]['quantity']
        if abs(quantity) > abs(current_qty):
            raise ValueError(f"Cannot sell {quantity} shares, only {current_qty} available")
        
        new_qty = current_qty - quantity
        proceeds = quantity * price
        self.cash += proceeds
        
        if new_qty == 0:
            del self.positions[symbol]
        else:
            self.positions[symbol]['quantity'] = new_qty
            self.positions[symbol]['last_price'] = price
            self.positions[symbol]['last_updated'] = timestamp
        
        self.transactions.append({
            'symbol': symbol,
            'quantity': -quantity,
            'price': price,
            'cost': -proceeds,
            'timestamp': timestamp,
            'type': 'SELL'
        })
    
    def update_prices(self, price_data: Dict[str, float], timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol]['last_price'] = price
                self.positions[symbol]['last_updated'] = timestamp
    
    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        if current_prices:
            self.update_prices(current_prices)
        
        portfolio_value = self.cash
        for symbol, position in self.positions.items():
            portfolio_value += position['quantity'] * position['last_price']
        
        return portfolio_value
    
    def get_position_value(self, symbol: str) -> float:
        if symbol not in self.positions:
            return 0.0
        position = self.positions[symbol]
        return position['quantity'] * position['last_price']
    
    def get_unrealized_pnl(self, symbol: str) -> float:
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        current_value = position['quantity'] * position['last_price']
        cost_basis = position['quantity'] * position['cost_basis']
        return current_value - cost_basis
    
    def get_total_unrealized_pnl(self) -> float:
        total_pnl = 0.0
        for symbol in self.positions:
            total_pnl += self.get_unrealized_pnl(symbol)
        return total_pnl
    
    def get_realized_pnl(self) -> float:
        realized_pnl = 0.0
        position_lots: Dict[str, List[Dict[str, float]]] = {}

        for transaction in self.transactions:
            symbol = transaction['symbol']
            quantity = float(transaction['quantity'])
            price = float(transaction['price'])

            lots = position_lots.setdefault(symbol, [])

            if quantity > 0:
                lots.append({"qty": quantity, "price": price})
                continue

            sell_qty = abs(quantity)
            while sell_qty > 0 and lots:
                lot = lots[0]
                lot_qty = lot["qty"]
                take_qty = min(lot_qty, sell_qty)
                realized_pnl += take_qty * (price - lot["price"])
                lot["qty"] = lot_qty - take_qty
                sell_qty -= take_qty

                if lot["qty"] <= 0:
                    lots.pop(0)

        return realized_pnl
    
    def get_performance_metrics(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        current_value = self.get_portfolio_value(current_prices)
        total_return = current_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'current_portfolio_value': current_value,
            'total_return': total_return,
            'total_return_percentage': total_return_pct,
            'unrealized_pnl': self.get_total_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'number_of_positions': len(self.positions),
            'number_of_transactions': len(self.transactions)
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'cost_basis': position['cost_basis'],
                'last_price': position['last_price'],
                'market_value': position['quantity'] * position['last_price'],
                'unrealized_pnl': self.get_unrealized_pnl(symbol),
                'last_updated': position['last_updated']
            })
        return pd.DataFrame(data)
    
    def get_transactions_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.transactions)
    
    def record_performance(self, timestamp: Optional[datetime] = None, 
                         current_prices: Optional[Dict[str, float]] = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics = self.get_performance_metrics(current_prices)
        metrics['timestamp'] = timestamp
        self.performance_history.append(metrics)
    
    def get_performance_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.performance_history)
