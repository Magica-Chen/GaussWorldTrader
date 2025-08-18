from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import numpy as np
from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 20,  # Period for momentum calculation
            'rsi_period': 14,       # RSI period
            'rsi_oversold': 30,     # RSI oversold threshold
            'rsi_overbought': 70,   # RSI overbought threshold
            'volume_threshold': 1.5, # Volume threshold multiplier
            'price_change_threshold': 0.02,  # Minimum price change for signal
            'stop_loss_pct': 0.05,  # Stop loss percentage
            'take_profit_pct': 0.15, # Take profit percentage
            'max_positions': 5,     # Maximum number of concurrent positions
            'position_size_pct': 0.1 # Position size as percentage of portfolio
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Momentum Strategy", default_params)
        
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Dict[str, Any]], 
                        historical_data: Dict[str, pd.DataFrame],
                        portfolio) -> List[Dict[str, Any]]:
        
        signals = []
        
        for symbol, price in current_prices.items():
            if symbol not in historical_data:
                continue
            
            data = historical_data[symbol]
            if len(data) < max(self.parameters['lookback_period'], self.parameters['rsi_period']):
                continue
            
            # Calculate technical indicators
            indicators = self.get_technical_indicators(data)
            
            # Get current values
            current_rsi = indicators.get('rsi', pd.Series()).iloc[-1] if 'rsi' in indicators else None
            current_volume = current_data.get(symbol, {}).get('volume', 0)
            
            # Calculate momentum
            momentum = self._calculate_momentum(data, self.parameters['lookback_period'])
            volume_ratio = self._calculate_volume_ratio(data, 20)
            
            # Check for buy signal
            buy_signal = self._check_buy_conditions(
                symbol, price, momentum, current_rsi, volume_ratio, current_volume, portfolio
            )
            
            if buy_signal:
                position_size = self.calculate_position_size(
                    symbol, price, portfolio.get_portfolio_value()
                )
                if position_size > 0:
                    signal = {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': position_size,
                        'price': price,
                        'reason': buy_signal['reason'],
                        'confidence': buy_signal['confidence'],
                        'stop_loss': price * (1 - self.parameters['stop_loss_pct']),
                        'take_profit': price * (1 + self.parameters['take_profit_pct'])
                    }
                    
                    # Apply risk management
                    signal = self.risk_management_check(signal, portfolio)
                    
                    if self.validate_signal(signal):
                        signals.append(signal)
                        self.log_signal(signal)
            
            # Check for sell signal
            sell_signal = self._check_sell_conditions(
                symbol, price, momentum, current_rsi, portfolio
            )
            
            if sell_signal and symbol in portfolio.positions:
                current_position = portfolio.positions[symbol]['quantity']
                if current_position > 0:
                    signal = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': current_position,
                        'price': price,
                        'reason': sell_signal['reason'],
                        'confidence': sell_signal['confidence']
                    }
                    
                    if self.validate_signal(signal):
                        signals.append(signal)
                        self.log_signal(signal)
        
        return signals
    
    def _calculate_momentum(self, data: pd.DataFrame, period: int) -> float:
        if len(data) < period + 1:
            return 0.0
        
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-period-1]
        
        return (current_price - past_price) / past_price
    
    def _calculate_volume_ratio(self, data: pd.DataFrame, period: int = 20) -> float:
        if len(data) < period or 'volume' not in data.columns:
            return 1.0
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=period).mean().iloc[-1]
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _check_buy_conditions(self, symbol: str, price: float, momentum: float,
                             rsi: float, volume_ratio: float, current_volume: float,
                             portfolio) -> Dict[str, Any]:
        
        reasons = []
        confidence = 0.0
        
        # Check if we already have max positions
        current_positions = len([p for p in portfolio.positions.values() if p['quantity'] > 0])
        if current_positions >= self.parameters['max_positions']:
            return None
        
        # Don't buy if we already have a position in this symbol
        if symbol in portfolio.positions and portfolio.positions[symbol]['quantity'] > 0:
            return None
        
        # Momentum condition
        if momentum > self.parameters['price_change_threshold']:
            reasons.append(f"Strong momentum: {momentum:.2%}")
            confidence += 0.3
        
        # RSI condition (not overbought)
        if rsi is not None and rsi < self.parameters['rsi_overbought']:
            reasons.append(f"RSI not overbought: {rsi:.1f}")
            confidence += 0.2
            
            # Extra confidence if RSI is recovering from oversold
            if rsi > self.parameters['rsi_oversold'] and rsi < self.parameters['rsi_oversold'] + 10:
                reasons.append("RSI recovering from oversold")
                confidence += 0.2
        
        # Volume condition
        if volume_ratio > self.parameters['volume_threshold']:
            reasons.append(f"High volume: {volume_ratio:.1f}x average")
            confidence += 0.3
        
        # Additional momentum confirmation
        if momentum > self.parameters['price_change_threshold'] * 2:
            reasons.append("Very strong momentum")
            confidence += 0.2
        
        # Minimum confidence threshold
        if confidence >= 0.5 and len(reasons) >= 2:
            return {
                'reason': '; '.join(reasons),
                'confidence': min(confidence, 1.0)
            }
        
        return None
    
    def _check_sell_conditions(self, symbol: str, price: float, momentum: float,
                              rsi: float, portfolio) -> Dict[str, Any]:
        
        if symbol not in portfolio.positions:
            return None
        
        position = portfolio.positions[symbol]
        if position['quantity'] <= 0:
            return None
        
        reasons = []
        confidence = 0.0
        
        # Calculate current P&L
        cost_basis = position['cost_basis']
        current_pnl_pct = (price - cost_basis) / cost_basis
        
        # Stop loss condition
        if current_pnl_pct <= -self.parameters['stop_loss_pct']:
            reasons.append(f"Stop loss triggered: {current_pnl_pct:.2%}")
            confidence = 1.0
        
        # Take profit condition
        elif current_pnl_pct >= self.parameters['take_profit_pct']:
            reasons.append(f"Take profit triggered: {current_pnl_pct:.2%}")
            confidence = 0.9
        
        # Momentum reversal
        elif momentum < -self.parameters['price_change_threshold']:
            reasons.append(f"Negative momentum: {momentum:.2%}")
            confidence += 0.4
        
        # RSI overbought condition
        if rsi is not None and rsi > self.parameters['rsi_overbought']:
            reasons.append(f"RSI overbought: {rsi:.1f}")
            confidence += 0.3
        
        # Additional sell conditions based on position holding time or other factors
        # This could be enhanced with more sophisticated exit rules
        
        if confidence >= 0.6 and len(reasons) >= 1:
            return {
                'reason': '; '.join(reasons),
                'confidence': min(confidence, 1.0)
            }
        
        return None
    
    def calculate_position_size(self, symbol: str, price: float, 
                              portfolio_value: float, volatility: float = None) -> int:
        
        # Calculate position size based on percentage of portfolio
        target_position_value = portfolio_value * self.parameters['position_size_pct']
        position_size = int(target_position_value / price)
        
        # Minimum position size
        if position_size < 1:
            return 0
        
        # Adjust for volatility if provided
        if volatility is not None and volatility > 0.02:  # High volatility
            position_size = int(position_size * 0.7)  # Reduce position size
        
        return max(1, position_size)
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': 'Momentum-based trading strategy using RSI and volume confirmation',
            'parameters': self.parameters,
            'risk_management': {
                'stop_loss': f"{self.parameters['stop_loss_pct']:.1%}",
                'take_profit': f"{self.parameters['take_profit_pct']:.1%}",
                'max_positions': self.parameters['max_positions'],
                'position_size': f"{self.parameters['position_size_pct']:.1%}"
            }
        }
    
    def optimize_parameters(self, historical_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple parameter optimization based on historical performance
        This is a basic implementation - more sophisticated optimization could be added
        """
        current_sharpe = historical_results.get('sharpe_ratio', 0)
        current_return = historical_results.get('total_return_percentage', 0)
        
        suggestions = {}
        
        # Suggest adjustments based on performance
        if current_sharpe < 1.0:
            suggestions['stop_loss_pct'] = self.parameters['stop_loss_pct'] * 0.8  # Tighter stop loss
            suggestions['take_profit_pct'] = self.parameters['take_profit_pct'] * 1.2  # Higher take profit
        
        if current_return < 10:  # Less than 10% return
            suggestions['price_change_threshold'] = self.parameters['price_change_threshold'] * 0.8  # Lower threshold
            suggestions['position_size_pct'] = min(0.15, self.parameters['position_size_pct'] * 1.2)  # Larger positions
        
        return suggestions