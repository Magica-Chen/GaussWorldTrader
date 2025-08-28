"""
Scalping Strategy - High Frequency Trading

Captures small price movements with high frequency trades (minute to hour timeframes)
Focuses on market microstructure and order book dynamics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseStrategy
from src.utils.timezone_utils import now_et

class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping strategy based on:
    - VWAP deviation signals
    - Order flow imbalance
    - Bid-ask spread analysis
    - Volume spikes
    - Mean reversion on micro timeframes
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'vwap_period': 20,
            'vwap_threshold': 0.002,  # 0.2% deviation threshold
            'volume_spike_multiplier': 2.0,
            'spread_threshold': 0.001,  # 0.1% spread threshold
            'position_size_pct': 0.05,  # 5% of portfolio per trade
            'stop_loss_pct': 0.005,    # 0.5% stop loss
            'take_profit_pct': 0.008,  # 0.8% take profit
            'max_holding_minutes': 60,  # Maximum holding period
            'min_volume': 10000,       # Minimum daily volume
            'risk_limit_pct': 0.10    # 10% total risk limit
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """Generate scalping signals based on high-frequency patterns"""
        
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_prices or data.empty:
                continue
            
            try:
                signal = self._analyze_scalping_opportunity(
                    symbol, data, current_prices[symbol], 
                    current_data.get(symbol, {}), portfolio
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} for scalping: {e}")
                continue
        
        return signals
    
    def _analyze_scalping_opportunity(self, symbol: str, data: pd.DataFrame, 
                                    current_price: float, current_bar: Dict[str, Any],
                                    portfolio: Any = None) -> Optional[Dict[str, Any]]:
        """Analyze scalping opportunity for a single symbol"""
        current_date = now_et()

        if len(data) < self.parameters['vwap_period'] + 10:
            return None
        
        # Calculate technical indicators for scalping
        indicators = self._calculate_scalping_indicators(data)
        
        # Check volume requirements
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = current_bar.get('volume', 0)
        
        if avg_volume < self.parameters['min_volume']:
            return None
        
        # VWAP deviation signal
        vwap_signal = self._check_vwap_deviation(indicators, current_price)
        
        # Volume spike detection
        volume_signal = self._check_volume_spike(current_volume, avg_volume)
        
        # Bid-ask spread analysis (simulated from OHLC)
        spread_signal = self._check_spread_conditions(current_bar)
        
        # Order flow imbalance (approximated)
        flow_signal = self._check_order_flow(data.tail(5))
        
        # Mean reversion signal
        reversion_signal = self._check_mean_reversion(indicators, current_price)
        
        # Combine signals
        buy_score = 0
        sell_score = 0
        
        # VWAP signals
        if vwap_signal == 'buy':
            buy_score += 3
        elif vwap_signal == 'sell':
            sell_score += 3
        
        # Volume confirmation
        if volume_signal:
            buy_score += 1
            sell_score += 1
        
        # Spread conditions
        if spread_signal:
            buy_score += 1
            sell_score += 1
        
        # Order flow
        if flow_signal == 'buy':
            buy_score += 2
        elif flow_signal == 'sell':
            sell_score += 2
        
        # Mean reversion
        if reversion_signal == 'buy':
            buy_score += 2
        elif reversion_signal == 'sell':
            sell_score += 2
        
        # Generate signal if score threshold is met
        signal_threshold = 4
        
        if buy_score >= signal_threshold and buy_score > sell_score:
            action = 'buy'
        elif sell_score >= signal_threshold and sell_score > buy_score:
            action = 'sell'
        else:
            return None
        
        # Calculate position size
        portfolio_value = portfolio.get_portfolio_value({symbol: current_price}) if portfolio else 100000
        position_size = self.calculate_position_size(symbol, current_price, portfolio_value)
        
        # Risk management check
        if not self._check_risk_limits(symbol, action, position_size, current_price, portfolio):
            return None
        
        return {
            'symbol': symbol,
            'action': action,
            'quantity': position_size,
            'price': current_price,
            'strategy': 'scalping',
            'confidence': min(max(buy_score, sell_score) / 8.0, 1.0),
            'stop_loss': current_price * (1 - self.parameters['stop_loss_pct']) if action == 'buy' 
                        else current_price * (1 + self.parameters['stop_loss_pct']),
            'take_profit': current_price * (1 + self.parameters['take_profit_pct']) if action == 'buy'
                          else current_price * (1 - self.parameters['take_profit_pct']),
            'max_holding_time': current_date + timedelta(minutes=self.parameters['max_holding_minutes']),
            'signals': {
                'vwap': vwap_signal,
                'volume_spike': volume_signal,
                'spread_ok': spread_signal,
                'order_flow': flow_signal,
                'mean_reversion': reversion_signal,
                'buy_score': buy_score,
                'sell_score': sell_score
            }
        }
    
    def _calculate_scalping_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators specific to scalping"""
        
        # VWAP (Volume Weighted Average Price)
        vwap = self._calculate_vwap(data, self.parameters['vwap_period'])
        
        # RSI for overbought/oversold on short timeframes
        rsi = self._calculate_rsi(data['close'], 7)
        
        # Bollinger Bands for volatility
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['close'], 10, 2)
        
        # EMA for trend
        ema_fast = data['close'].ewm(span=5).mean()
        ema_slow = data['close'].ewm(span=15).mean()
        
        # Price momentum
        momentum_1 = data['close'].pct_change(1)
        momentum_3 = data['close'].pct_change(3)
        
        # Volume momentum
        volume_ma = data['volume'].rolling(10).mean()
        volume_ratio = data['volume'] / volume_ma
        
        return {
            'vwap': vwap,
            'rsi': rsi,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'momentum_1': momentum_1,
            'momentum_3': momentum_3,
            'volume_ratio': volume_ratio
        }
    
    def _calculate_vwap(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        vwap = []
        for i in range(len(data)):
            start_idx = max(0, i - period + 1)
            period_data = data.iloc[start_idx:i+1]
            typical_period = typical_price.iloc[start_idx:i+1]
            
            if len(period_data) > 0:
                total_volume = period_data['volume'].sum()
                if total_volume > 0:
                    vwap_val = (typical_period * period_data['volume']).sum() / total_volume
                else:
                    vwap_val = typical_period.iloc[-1]
            else:
                vwap_val = typical_price.iloc[i]
            
            vwap.append(vwap_val)
        
        return pd.Series(vwap, index=data.index)
    
    def _check_vwap_deviation(self, indicators: Dict[str, Any], current_price: float) -> str:
        """Check for VWAP deviation signals"""
        if len(indicators['vwap']) == 0:
            return 'neutral'
        
        current_vwap = indicators['vwap'].iloc[-1]
        deviation = (current_price - current_vwap) / current_vwap
        
        threshold = self.parameters['vwap_threshold']
        
        if deviation < -threshold:  # Price below VWAP - potential buy
            return 'buy'
        elif deviation > threshold:  # Price above VWAP - potential sell
            return 'sell'
        else:
            return 'neutral'
    
    def _check_volume_spike(self, current_volume: float, avg_volume: float) -> bool:
        """Check for volume spike indicating interest"""
        if avg_volume <= 0:
            return False
        
        volume_ratio = current_volume / avg_volume
        return volume_ratio >= self.parameters['volume_spike_multiplier']
    
    def _check_spread_conditions(self, current_bar: Dict[str, Any]) -> bool:
        """Check if bid-ask spread is favorable (simulated from OHLC)"""
        high = current_bar.get('high', 0)
        low = current_bar.get('low', 0)
        close = current_bar.get('close', 0)
        
        if close <= 0:
            return False
        
        # Approximate spread as high-low range relative to close
        spread_pct = (high - low) / close
        
        return spread_pct <= self.parameters['spread_threshold']
    
    def _check_order_flow(self, recent_data: pd.DataFrame) -> str:
        """Approximate order flow analysis"""
        if len(recent_data) < 3:
            return 'neutral'
        
        # Approximate buying/selling pressure using price and volume
        buy_pressure = 0
        sell_pressure = 0
        
        for i in range(1, len(recent_data)):
            prev_close = recent_data['close'].iloc[i-1]
            curr_close = recent_data['close'].iloc[i]
            curr_volume = recent_data['volume'].iloc[i]
            
            if curr_close > prev_close:
                buy_pressure += curr_volume * (curr_close - prev_close) / prev_close
            else:
                sell_pressure += curr_volume * (prev_close - curr_close) / prev_close
        
        if buy_pressure > sell_pressure * 1.5:
            return 'buy'
        elif sell_pressure > buy_pressure * 1.5:
            return 'sell'
        else:
            return 'neutral'
    
    def _check_mean_reversion(self, indicators: Dict[str, Any], current_price: float) -> str:
        """Check for mean reversion opportunities"""
        if len(indicators['bb_upper']) == 0:
            return 'neutral'
        
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        rsi = indicators['rsi'].iloc[-1] if len(indicators['rsi']) > 0 else 50
        
        # Oversold conditions
        if current_price <= bb_lower and rsi < 30:
            return 'buy'
        # Overbought conditions
        elif current_price >= bb_upper and rsi > 70:
            return 'sell'
        else:
            return 'neutral'
    
    def _check_risk_limits(self, symbol: str, action: str, position_size: int,
                          current_price: float, portfolio: Any) -> bool:
        """Check if trade respects risk management rules"""
        if portfolio is None:
            return True
        
        # Check maximum risk exposure
        portfolio_value = portfolio.get_portfolio_value({symbol: current_price})
        trade_value = position_size * current_price
        risk_pct = trade_value / portfolio_value
        
        if risk_pct > self.parameters['risk_limit_pct']:
            return False
        
        # Check position concentration
        current_positions = portfolio.positions
        if symbol in current_positions:
            current_qty = current_positions[symbol]['quantity']
            new_total_qty = abs(current_qty) + position_size
            new_total_value = new_total_qty * current_price
            
            if new_total_value / portfolio_value > 0.15:  # Max 15% per symbol
                return False
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = None) -> int:
        """Calculate position size for scalping"""
        # Base position size
        target_value = portfolio_value * self.parameters['position_size_pct']
        base_size = int(target_value / price)
        
        # Adjust for volatility if available
        if volatility and volatility > 0:
            # Reduce size for high volatility
            volatility_adjustment = min(0.02 / volatility, 2.0)  # Target 2% volatility
            base_size = int(base_size * volatility_adjustment)
        
        # Minimum viable size
        min_size = max(1, int(1000 / price))  # At least $1000 trade
        
        return max(base_size, min_size)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and parameters"""
        return {
            'name': 'Scalping Strategy',
            'type': 'High Frequency',
            'timeframe': '1min - 1hour',
            'description': 'High-frequency scalping strategy based on VWAP, volume, and mean reversion',
            'parameters': self.parameters,
            'risk_level': 'High',
            'expected_trades_per_day': '10-50',
            'holding_period': 'Minutes to 1 hour',
            'best_markets': ['High volume stocks', 'ETFs', 'Liquid markets']
        }