from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

class BaseStrategy(ABC):
    def __init__(self, parameters: Dict[str, Any] = None) -> None:
        self.parameters = parameters or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Strategy state
        self.positions: Dict[str, Any] = {}
        self.signals: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    @abstractmethod
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], 
                        historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, price: float, 
                              portfolio_value: float, volatility: float = None) -> int:
        pass
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        required_fields = ['symbol', 'action', 'quantity']
        
        if not all(field in signal for field in required_fields):
            self.logger.warning(f"Signal missing required fields: {signal}")
            return False
        
        if signal['action'].upper() not in ['BUY', 'SELL']:
            self.logger.warning(f"Invalid action in signal: {signal['action']}")
            return False
        
        if not isinstance(signal['quantity'], (int, float)) or signal['quantity'] <= 0:
            self.logger.warning(f"Invalid quantity in signal: {signal['quantity']}")
            return False
        
        return True
    
    def update_position(self, symbol: str, quantity: int, price: float, 
                       action: str, timestamp: datetime):
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'total_cost': 0,
                'last_updated': timestamp
            }
        
        position = self.positions[symbol]
        
        if action.upper() == 'BUY':
            new_quantity = position['quantity'] + quantity
            new_total_cost = position['total_cost'] + (quantity * price)
            
            if new_quantity > 0:
                new_avg_price = new_total_cost / new_quantity
                self.positions[symbol].update({
                    'quantity': new_quantity,
                    'avg_price': new_avg_price,
                    'total_cost': new_total_cost,
                    'last_updated': timestamp
                })
            else:
                self.positions[symbol].update({
                    'quantity': new_quantity,
                    'avg_price': price,
                    'total_cost': new_total_cost,
                    'last_updated': timestamp
                })
        
        elif action.upper() == 'SELL':
            new_quantity = position['quantity'] - quantity
            cost_per_share = position['avg_price']
            cost_reduction = quantity * cost_per_share
            
            self.positions[symbol].update({
                'quantity': new_quantity,
                'total_cost': position['total_cost'] - cost_reduction,
                'last_updated': timestamp
            })
            
            if new_quantity == 0:
                self.positions[symbol]['avg_price'] = 0
                self.positions[symbol]['total_cost'] = 0
    
    def get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = {}
        
        # Simple Moving Averages
        if len(data) >= 20:
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
        if len(data) >= 50:
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
        if len(data) >= 200:
            indicators['sma_200'] = data['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        if len(data) >= 12:
            indicators['ema_12'] = data['close'].ewm(span=12).mean()
        if len(data) >= 26:
            indicators['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        if len(data) >= 14:
            indicators['rsi'] = self._calculate_rsi(data['close'], 14)
        
        # MACD
        if len(data) >= 26:
            indicators['macd'], indicators['macd_signal'], indicators['macd_histogram'] = self._calculate_macd(data['close'])
        
        # Bollinger Bands
        if len(data) >= 20:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'], 20, 2)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
        
        # Volume indicators
        if 'volume' in data.columns and len(data) >= 20:
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def risk_management_check(self, signal: Dict[str, Any], portfolio,
                            max_position_size: float = 0.1) -> Dict[str, Any]:
        symbol = signal['symbol']
        quantity = signal['quantity']
        action = signal['action'].upper()
        
        if action == 'BUY':
            portfolio_value = portfolio.get_portfolio_value()
            position_value = quantity * signal.get('price', 0)
            position_ratio = position_value / portfolio_value
            
            if position_ratio > max_position_size:
                new_quantity = int((max_position_size * portfolio_value) / signal.get('price', 1))
                self.logger.info(f"Reducing position size for {symbol} from {quantity} to {new_quantity}")
                signal['quantity'] = max(1, new_quantity)
        
        elif action == 'SELL':
            current_position = portfolio.positions.get(symbol, {}).get('quantity', 0)
            if quantity > current_position:
                self.logger.info(f"Reducing sell quantity for {symbol} from {quantity} to {current_position}")
                signal['quantity'] = current_position
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': self.__class__.__name__,
            'type': 'Unknown',
            'timeframe': 'Unknown',
            'risk_level': 'Unknown', 
            'expected_trades_per_day': 'Unknown',
            'holding_period': 'Unknown',
            'description': 'No description',
            'parameters': self.parameters,
            'current_positions': self.positions,
            'total_signals_generated': len(self.signals),
            'performance_metrics': self.performance_metrics
        }
    
    def log_signal(self, signal: Dict[str, Any]):
        self.signals.append({
            **signal,
            'timestamp': datetime.now(),
            'strategy': self.name
        })
        self.logger.info(f"Generated signal: {signal}")
    
    def reset_strategy_state(self):
        self.positions.clear()
        self.signals.clear()
        self.performance_metrics.clear()
        self.logger.info(f"Strategy {self.name} state reset")