import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

class TechnicalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = TechnicalAnalysis.atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.abs().rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def identify_patterns(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        patterns = {
            'hammer': [],
            'doji': [],
            'engulfing_bullish': [],
            'engulfing_bearish': [],
            'morning_star': [],
            'evening_star': []
        }
        
        for i in range(len(data)):
            # Hammer pattern
            if self._is_hammer(data.iloc[i]):
                patterns['hammer'].append(i)
            
            # Doji pattern
            if self._is_doji(data.iloc[i]):
                patterns['doji'].append(i)
            
            # Engulfing patterns (need at least 2 candles)
            if i > 0:
                if self._is_bullish_engulfing(data.iloc[i-1], data.iloc[i]):
                    patterns['engulfing_bullish'].append(i)
                
                if self._is_bearish_engulfing(data.iloc[i-1], data.iloc[i]):
                    patterns['engulfing_bearish'].append(i)
            
            # Star patterns (need at least 3 candles)
            if i > 1:
                if self._is_morning_star(data.iloc[i-2:i+1]):
                    patterns['morning_star'].append(i)
                
                if self._is_evening_star(data.iloc[i-2:i+1]):
                    patterns['evening_star'].append(i)
        
        return patterns
    
    def _is_hammer(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        
        return (lower_shadow > 2 * body and upper_shadow < 0.3 * body)
    
    def _is_doji(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        range_size = candle['high'] - candle['low']
        
        return body < 0.1 * range_size
    
    def _is_bullish_engulfing(self, prev_candle, curr_candle) -> bool:
        prev_bearish = prev_candle['close'] < prev_candle['open']
        curr_bullish = curr_candle['close'] > curr_candle['open']
        
        engulfs = (curr_candle['open'] < prev_candle['close'] and 
                  curr_candle['close'] > prev_candle['open'])
        
        return prev_bearish and curr_bullish and engulfs
    
    def _is_bearish_engulfing(self, prev_candle, curr_candle) -> bool:
        prev_bullish = prev_candle['close'] > prev_candle['open']
        curr_bearish = curr_candle['close'] < curr_candle['open']
        
        engulfs = (curr_candle['open'] > prev_candle['close'] and 
                  curr_candle['close'] < prev_candle['open'])
        
        return prev_bullish and curr_bearish and engulfs
    
    def _is_morning_star(self, candles) -> bool:
        if len(candles) < 3:
            return False
        
        first = candles.iloc[0]
        second = candles.iloc[1]
        third = candles.iloc[2]
        
        first_bearish = first['close'] < first['open']
        third_bullish = third['close'] > third['open']
        
        second_small_body = abs(second['close'] - second['open']) < 0.3 * abs(first['close'] - first['open'])
        gap_down = second['high'] < first['close']
        gap_up = third['open'] > second['high']
        
        return first_bearish and third_bullish and second_small_body and gap_down and gap_up
    
    def _is_evening_star(self, candles) -> bool:
        if len(candles) < 3:
            return False
        
        first = candles.iloc[0]
        second = candles.iloc[1]
        third = candles.iloc[2]
        
        first_bullish = first['close'] > first['open']
        third_bearish = third['close'] < third['open']
        
        second_small_body = abs(second['close'] - second['open']) < 0.3 * abs(first['close'] - first['open'])
        gap_up = second['low'] > first['close']
        gap_down = third['open'] < second['low']
        
        return first_bullish and third_bearish and second_small_body and gap_up and gap_down
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        highs = data['high'].rolling(window=window).max()
        lows = data['low'].rolling(window=window).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data)):
            if data['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['high'].iloc[i])
            
            if data['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['low'].iloc[i])
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
        support_levels = sorted(list(set(support_levels)))[:5]
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def trend_analysis(self, data: pd.DataFrame) -> Dict[str, any]:
        close_prices = data['close']
        
        # Calculate various moving averages
        sma_20 = self.sma(close_prices, 20)
        sma_50 = self.sma(close_prices, 50)
        sma_200 = self.sma(close_prices, 200)
        
        current_price = close_prices.iloc[-1]
        
        # Determine trend based on moving averages
        short_term_trend = 'bullish' if current_price > sma_20.iloc[-1] else 'bearish'
        medium_term_trend = 'bullish' if current_price > sma_50.iloc[-1] else 'bearish'
        long_term_trend = 'bullish' if current_price > sma_200.iloc[-1] else 'bearish'
        
        # Calculate trend strength
        rsi_val = self.rsi(close_prices).iloc[-1]
        macd_line, _, _ = self.macd(close_prices)
        
        trend_strength = 'strong' if abs(rsi_val - 50) > 20 else 'weak'
        
        return {
            'current_price': current_price,
            'short_term_trend': short_term_trend,
            'medium_term_trend': medium_term_trend,
            'long_term_trend': long_term_trend,
            'trend_strength': trend_strength,
            'rsi': rsi_val,
            'macd': macd_line.iloc[-1],
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1],
            'sma_200': sma_200.iloc[-1]
        }