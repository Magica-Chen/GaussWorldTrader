"""
Trend Following Strategy - Low Frequency

Multi-timeframe trend following using various technical indicators
Designed for daily to weekly holding periods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    Comprehensive trend following strategy using:
    - Moving average crossovers
    - Donchian channels
    - ATR-based position sizing
    - Multiple timeframe confirmation
    - Momentum indicators
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'fast_ma_period': 10,
            'slow_ma_period': 30,
            'donchian_period': 20,
            'atr_period': 14,
            'rsi_period': 14,
            'momentum_period': 12,
            'min_trend_strength': 0.02,  # Minimum 2% trend strength
            'position_size_pct': 0.08,   # 8% of portfolio per position
            'atr_multiplier': 2.0,       # ATR multiplier for stops
            'max_positions': 5,          # Maximum concurrent positions
            'min_volume': 50000,         # Minimum average volume
            'trend_confirmation_days': 3  # Days of trend confirmation
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """Generate trend following signals"""
        
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_prices or data.empty:
                continue
            
            try:
                signal = self._analyze_trend_opportunity(
                    symbol, data, current_prices[symbol], 
                    current_data.get(symbol, {}), portfolio
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} for trend following: {e}")
                continue
        
        return signals
    
    def _analyze_trend_opportunity(self, symbol: str, data: pd.DataFrame,
                                  current_price: float, current_bar: Dict[str, Any],
                                  portfolio: Any = None) -> Optional[Dict[str, Any]]:
        """Analyze trend following opportunity for a single symbol"""
        
        if len(data) < max(self.parameters['slow_ma_period'], 
                          self.parameters['donchian_period']) + 10:
            return None
        
        # Check volume requirement
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        if avg_volume < self.parameters['min_volume']:
            return None
        
        # Calculate trend indicators
        indicators = self._calculate_trend_indicators(data)
        
        # Analyze trend signals
        ma_signal = self._analyze_ma_crossover(indicators)
        donchian_signal = self._analyze_donchian_breakout(indicators, current_price)
        momentum_signal = self._analyze_momentum(indicators)
        trend_strength = self._calculate_trend_strength(indicators)
        
        # Multi-timeframe confirmation
        timeframe_confirmation = self._check_timeframe_alignment(indicators)
        
        # Combine signals
        buy_score = 0
        sell_score = 0
        
        # Moving average signals
        if ma_signal == 'buy':
            buy_score += 2
        elif ma_signal == 'sell':
            sell_score += 2
        
        # Donchian breakout signals
        if donchian_signal == 'buy':
            buy_score += 3
        elif donchian_signal == 'sell':
            sell_score += 3
        
        # Momentum confirmation
        if momentum_signal == 'buy':
            buy_score += 2
        elif momentum_signal == 'sell':
            sell_score += 2
        
        # Trend strength requirement
        if trend_strength >= self.parameters['min_trend_strength']:
            buy_score += 1
            sell_score += 1
        
        # Timeframe alignment
        if timeframe_confirmation:
            buy_score += 2
            sell_score += 2
        
        # Determine action
        signal_threshold = 5
        
        if buy_score >= signal_threshold and buy_score > sell_score:
            action = 'buy'
            confidence = buy_score
        elif sell_score >= signal_threshold and sell_score > buy_score:
            action = 'sell'
            confidence = sell_score
        else:
            return None
        
        # Portfolio and risk checks
        if not self._check_portfolio_constraints(symbol, action, portfolio):
            return None
        
        # Calculate position size using ATR
        portfolio_value = portfolio.get_portfolio_value({symbol: current_price}) if portfolio else 100000
        atr = indicators['atr'].iloc[-1]
        position_size = self._calculate_atr_position_size(current_price, atr, portfolio_value)
        
        # Calculate stops and targets
        stop_loss, take_profit = self._calculate_stops_and_targets(
            current_price, atr, action
        )
        
        return {
            'symbol': symbol,
            'action': action,
            'quantity': position_size,
            'price': current_price,
            'strategy': 'trend_following',
            'confidence': min(confidence / 10.0, 1.0),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'holding_period_target': timedelta(days=7),  # Target 1 week hold
            'trend_strength': trend_strength,
            'signals': {
                'ma_crossover': ma_signal,
                'donchian_breakout': donchian_signal,
                'momentum': momentum_signal,
                'trend_strength': trend_strength,
                'timeframe_aligned': timeframe_confirmation,
                'buy_score': buy_score,
                'sell_score': sell_score
            },
            'risk_metrics': {
                'atr': atr,
                'atr_pct': atr / current_price,
                'position_risk': (abs(current_price - stop_loss) / current_price) * 100
            }
        }
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all trend following indicators"""
        
        # Moving averages
        fast_ma = data['close'].rolling(self.parameters['fast_ma_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_ma_period']).mean()
        
        # Exponential moving averages
        ema_fast = data['close'].ewm(span=self.parameters['fast_ma_period']).mean()
        ema_slow = data['close'].ewm(span=self.parameters['slow_ma_period']).mean()
        
        # Donchian channels
        donchian_high = data['high'].rolling(self.parameters['donchian_period']).max()
        donchian_low = data['low'].rolling(self.parameters['donchian_period']).min()
        donchian_mid = (donchian_high + donchian_low) / 2
        
        # ATR for volatility
        atr = self._calculate_atr(data, self.parameters['atr_period'])
        
        # RSI for momentum
        rsi = self._calculate_rsi(data['close'], self.parameters['rsi_period'])
        
        # Rate of Change (momentum)
        roc = data['close'].pct_change(self.parameters['momentum_period'])
        
        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(data['close'])
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['close'], 20, 2)
        
        # Average Directional Index (ADX) - simplified
        adx = self._calculate_simplified_adx(data)
        
        return {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'donchian_high': donchian_high,
            'donchian_low': donchian_low,
            'donchian_mid': donchian_mid,
            'atr': atr,
            'rsi': rsi,
            'roc': roc,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'adx': adx
        }
    
    def _analyze_ma_crossover(self, indicators: Dict[str, Any]) -> str:
        """Analyze moving average crossover signals"""
        fast_ma = indicators['fast_ma']
        slow_ma = indicators['slow_ma']
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return 'neutral'
        
        # Current and previous values
        fast_curr, fast_prev = fast_ma.iloc[-1], fast_ma.iloc[-2]
        slow_curr, slow_prev = slow_ma.iloc[-1], slow_ma.iloc[-2]
        ema_fast_curr = ema_fast.iloc[-1]
        ema_slow_curr = ema_slow.iloc[-1]
        
        # Check for bullish crossover
        bullish_crossover = (fast_prev <= slow_prev and fast_curr > slow_curr)
        bullish_ema = ema_fast_curr > ema_slow_curr
        
        # Check for bearish crossover
        bearish_crossover = (fast_prev >= slow_prev and fast_curr < slow_curr)
        bearish_ema = ema_fast_curr < ema_slow_curr
        
        if bullish_crossover and bullish_ema:
            return 'buy'
        elif bearish_crossover and bearish_ema:
            return 'sell'
        else:
            return 'neutral'
    
    def _analyze_donchian_breakout(self, indicators: Dict[str, Any], current_price: float) -> str:
        """Analyze Donchian channel breakout signals"""
        donchian_high = indicators['donchian_high']
        donchian_low = indicators['donchian_low']
        
        if len(donchian_high) == 0 or len(donchian_low) == 0:
            return 'neutral'
        
        recent_high = donchian_high.iloc[-1]
        recent_low = donchian_low.iloc[-1]
        
        # Breakout signals
        if current_price > recent_high:
            return 'buy'
        elif current_price < recent_low:
            return 'sell'
        else:
            return 'neutral'
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> str:
        """Analyze momentum indicators"""
        rsi = indicators['rsi']
        roc = indicators['roc']
        macd_histogram = indicators['macd_histogram']
        
        if len(rsi) == 0 or len(roc) == 0 or len(macd_histogram) == 0:
            return 'neutral'
        
        current_rsi = rsi.iloc[-1]
        current_roc = roc.iloc[-1]
        current_macd_hist = macd_histogram.iloc[-1]
        
        # Momentum signals
        rsi_bullish = 30 < current_rsi < 70  # Not overbought/oversold
        roc_bullish = current_roc > 0.01  # Positive momentum > 1%
        macd_bullish = current_macd_hist > 0
        
        rsi_bearish = 30 < current_rsi < 70  # Not overbought/oversold
        roc_bearish = current_roc < -0.01  # Negative momentum < -1%
        macd_bearish = current_macd_hist < 0
        
        bullish_count = sum([rsi_bullish, roc_bullish, macd_bullish])
        bearish_count = sum([rsi_bearish, roc_bearish, macd_bearish])
        
        if bullish_count >= 2:
            return 'buy'
        elif bearish_count >= 2:
            return 'sell'
        else:
            return 'neutral'
    
    def _calculate_trend_strength(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall trend strength"""
        fast_ma = indicators['fast_ma']
        slow_ma = indicators['slow_ma']
        adx = indicators['adx']
        
        if len(fast_ma) == 0 or len(slow_ma) == 0:
            return 0.0
        
        # MA spread as percentage
        ma_spread = abs(fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]
        
        # ADX strength (normalized)
        adx_strength = min(adx.iloc[-1] / 50.0, 1.0) if len(adx) > 0 else 0.5
        
        # Combine trend strength measures
        trend_strength = (ma_spread * 0.7) + (adx_strength * 0.3)
        
        return trend_strength
    
    def _check_timeframe_alignment(self, indicators: Dict[str, Any]) -> bool:
        """Check if multiple timeframes are aligned"""
        fast_ma = indicators['fast_ma']
        slow_ma = indicators['slow_ma']
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        
        if len(fast_ma) < 5 or len(slow_ma) < 5:
            return False
        
        # Check trend consistency over last few days
        confirmation_days = self.parameters['trend_confirmation_days']
        
        # Check if trend has been consistent
        fast_trend_up = all(fast_ma.iloc[-i] > slow_ma.iloc[-i] for i in range(1, confirmation_days + 1))
        fast_trend_down = all(fast_ma.iloc[-i] < slow_ma.iloc[-i] for i in range(1, confirmation_days + 1))
        
        ema_trend_up = all(ema_fast.iloc[-i] > ema_slow.iloc[-i] for i in range(1, confirmation_days + 1))
        ema_trend_down = all(ema_fast.iloc[-i] < ema_slow.iloc[-i] for i in range(1, confirmation_days + 1))
        
        # Both timeframes should agree
        return (fast_trend_up and ema_trend_up) or (fast_trend_down and ema_trend_down)
    
    def _calculate_simplified_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate simplified ADX (Average Directional Index)"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                           np.maximum(low.shift(1) - low, 0), 0)
        
        # Smooth the values
        tr_smooth = pd.Series(tr).rolling(period).mean()
        dm_plus_smooth = pd.Series(dm_plus).rolling(period).mean()
        dm_minus_smooth = pd.Series(dm_minus).rolling(period).mean()
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(25)  # Default neutral ADX
    
    def _calculate_atr_position_size(self, price: float, atr: float, 
                                   portfolio_value: float) -> int:
        """Calculate position size based on ATR volatility"""
        # Risk per trade as percentage of portfolio
        risk_per_trade = 0.02  # 2% risk per trade
        
        # ATR stop distance
        stop_distance = atr * self.parameters['atr_multiplier']
        
        # Position size based on risk
        risk_amount = portfolio_value * risk_per_trade
        position_value = risk_amount / (stop_distance / price)
        
        # Apply position size percentage limit
        max_position_value = portfolio_value * self.parameters['position_size_pct']
        position_value = min(position_value, max_position_value)
        
        position_size = int(position_value / price)
        
        # Minimum viable size
        min_size = max(1, int(2000 / price))  # At least $2000 position
        
        return max(position_size, min_size)
    
    def _calculate_stops_and_targets(self, price: float, atr: float, 
                                   action: str) -> tuple:
        """Calculate stop loss and take profit levels"""
        atr_multiplier = self.parameters['atr_multiplier']
        
        if action == 'buy':
            stop_loss = price - (atr * atr_multiplier)
            take_profit = price + (atr * atr_multiplier * 2)  # 2:1 R:R
        else:  # sell
            stop_loss = price + (atr * atr_multiplier)
            take_profit = price - (atr * atr_multiplier * 2)  # 2:1 R:R
        
        return stop_loss, take_profit
    
    def _check_portfolio_constraints(self, symbol: str, action: str, 
                                   portfolio: Any) -> bool:
        """Check portfolio-level constraints"""
        if portfolio is None:
            return True
        
        # Check maximum positions limit
        current_positions = len([pos for pos in portfolio.positions.values() 
                               if pos['quantity'] != 0])
        
        if current_positions >= self.parameters['max_positions']:
            return False
        
        # Check if we already have a position in this symbol
        if symbol in portfolio.positions:
            current_qty = portfolio.positions[symbol]['quantity']
            if current_qty != 0:
                return False  # Don't add to existing positions
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = None) -> int:
        """Calculate position size for trend following"""
        target_value = portfolio_value * self.parameters['position_size_pct']
        position_size = int(target_value / price)
        
        # Adjust for volatility if provided
        if volatility and volatility > 0:
            # Reduce size for high volatility
            volatility_adjustment = min(0.02 / volatility, 2.0)
            position_size = int(position_size * volatility_adjustment)
        
        # Minimum viable size
        min_size = max(1, int(2000 / price))
        
        return max(position_size, min_size)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Trend Following Strategy',
            'type': 'Low Frequency Trend',
            'timeframe': 'Daily to Weekly',
            'description': 'Multi-timeframe trend following with ATR-based risk management',
            'parameters': self.parameters,
            'risk_level': 'Medium',
            'expected_trades_per_day': '1-3',
            'holding_period': 'Days to weeks',
            'best_markets': ['Trending stocks', 'ETFs', 'Commodities', 'Currencies']
        }