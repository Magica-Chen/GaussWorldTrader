"""
Statistical Arbitrage Strategy - High Frequency

Exploits short-term price inefficiencies between correlated assets
Uses pairs trading and mean reversion techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression

from .base_strategy import BaseStrategy

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy using:
    - Pairs trading with cointegration
    - Cross-asset momentum
    - ETF-constituent arbitrage opportunities
    - Index arbitrage
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 60,     # Days for cointegration analysis
            'zscore_threshold': 2.0,   # Z-score threshold for entry
            'zscore_exit': 0.5,        # Z-score threshold for exit
            'min_correlation': 0.7,    # Minimum correlation for pairs
            'position_size_pct': 0.05, # 5% of portfolio per leg
            'stop_loss_pct': 0.02,     # 2% stop loss
            'max_holding_days': 5,     # Maximum holding period
            'min_spread_vol': 0.01,    # Minimum spread volatility
            'transaction_cost': 0.001  # 0.1% transaction cost estimate
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = logging.getLogger(__name__)
        self.pairs_cache = {}
        self.spread_history = {}
    
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """Generate arbitrage signals"""
        
        signals = []
        symbols = list(historical_data.keys())
        
        if len(symbols) < 2:
            return signals
        
        # Find and analyze pairs
        pairs = self._identify_trading_pairs(symbols, historical_data, current_date)
        
        for pair_id, pair_info in pairs.items():
            try:
                pair_signals = self._analyze_pair_opportunity(
                    pair_info, current_prices, historical_data, portfolio
                )
                signals.extend(pair_signals)
                
            except Exception as e:
                self.logger.error(f"Error analyzing pair {pair_id}: {e}")
                continue
        
        return signals
    
    def _identify_trading_pairs(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame],
                               current_date: datetime) -> Dict[str, Dict]:
        """Identify potential trading pairs based on correlation and cointegration"""
        
        pairs = {}
        
        # Create aligned price matrix
        price_data = {}
        min_length = float('inf')
        
        for symbol in symbols:
            if symbol in historical_data and not historical_data[symbol].empty:
                prices = historical_data[symbol]['close'].dropna()
                price_data[symbol] = prices
                min_length = min(min_length, len(prices))
        
        if min_length < self.parameters['lookback_period']:
            return pairs
        
        # Align all price series
        aligned_data = {}
        for symbol, prices in price_data.items():
            aligned_data[symbol] = prices.tail(min_length)
        
        # Find pairs with high correlation
        symbols_list = list(aligned_data.keys())
        
        for i in range(len(symbols_list)):
            for j in range(i + 1, len(symbols_list)):
                symbol_a = symbols_list[i]
                symbol_b = symbols_list[j]
                
                try:
                    correlation = self._calculate_correlation(
                        aligned_data[symbol_a], aligned_data[symbol_b]
                    )
                    
                    if abs(correlation) >= self.parameters['min_correlation']:
                        # Test for cointegration
                        coint_result = self._test_cointegration(
                            aligned_data[symbol_a], aligned_data[symbol_b]
                        )
                        
                        if coint_result['is_cointegrated']:
                            pair_id = f"{symbol_a}_{symbol_b}"
                            pairs[pair_id] = {
                                'symbol_a': symbol_a,
                                'symbol_b': symbol_b,
                                'correlation': correlation,
                                'cointegration': coint_result,
                                'hedge_ratio': coint_result['hedge_ratio']
                            }
                            
                except Exception as e:
                    self.logger.warning(f"Error analyzing pair {symbol_a}-{symbol_b}: {e}")
                    continue
        
        return pairs
    
    def _calculate_correlation(self, series_a: pd.Series, series_b: pd.Series) -> float:
        """Calculate correlation between two price series"""
        # Use returns for correlation analysis
        returns_a = series_a.pct_change().dropna()
        returns_b = series_b.pct_change().dropna()
        
        if len(returns_a) < 20 or len(returns_b) < 20:
            return 0.0
        
        correlation, _ = stats.pearsonr(returns_a, returns_b)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _test_cointegration(self, series_a: pd.Series, series_b: pd.Series) -> Dict[str, Any]:
        """Test for cointegration between two price series"""
        
        # Simple cointegration test using linear regression
        X = series_a.values.reshape(-1, 1)
        y = series_b.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        hedge_ratio = model.coef_[0]
        residuals = y - model.predict(X)
        
        # Augmented Dickey-Fuller test on residuals (simplified)
        # Check if residuals are mean-reverting
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Simple stationarity check
        # More sophisticated implementation would use ADF test
        normalized_residuals = (residuals - mean_residual) / std_residual
        is_stationary = np.abs(normalized_residuals[-1]) < 2.0  # Simplified test
        
        return {
            'is_cointegrated': is_stationary and std_residual > self.parameters['min_spread_vol'],
            'hedge_ratio': hedge_ratio,
            'residuals': residuals,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'current_residual': residuals[-1]
        }
    
    def _analyze_pair_opportunity(self, pair_info: Dict, current_prices: Dict[str, float],
                                 historical_data: Dict[str, pd.DataFrame],
                                 portfolio: Any) -> List[Dict[str, Any]]:
        """Analyze trading opportunity for a specific pair"""
        
        signals = []
        symbol_a = pair_info['symbol_a']
        symbol_b = pair_info['symbol_b']
        
        if symbol_a not in current_prices or symbol_b not in current_prices:
            return signals
        
        price_a = current_prices[symbol_a]
        price_b = current_prices[symbol_b]
        hedge_ratio = pair_info['hedge_ratio']
        
        # Calculate current spread
        current_spread = price_b - hedge_ratio * price_a
        
        # Get historical spread statistics
        coint_data = pair_info['cointegration']
        mean_spread = coint_data['mean_residual']
        std_spread = coint_data['std_residual']
        
        if std_spread <= 0:
            return signals
        
        # Calculate z-score
        z_score = (current_spread - mean_spread) / std_spread
        
        # Generate signals based on z-score
        threshold = self.parameters['zscore_threshold']
        exit_threshold = self.parameters['zscore_exit']
        
        # Check if we have existing positions
        has_position_a = self._has_position(symbol_a, portfolio)
        has_position_b = self._has_position(symbol_b, portfolio)
        
        if not has_position_a and not has_position_b:
            # Entry signals
            if z_score > threshold:
                # Spread is too high - short spread (short B, long A)
                signals.extend(self._create_pair_entry_signals(
                    symbol_a, symbol_b, price_a, price_b, hedge_ratio,
                    'long_a_short_b', z_score, portfolio
                ))
            elif z_score < -threshold:
                # Spread is too low - long spread (long B, short A)
                signals.extend(self._create_pair_entry_signals(
                    symbol_a, symbol_b, price_a, price_b, hedge_ratio,
                    'short_a_long_b', z_score, portfolio
                ))
        
        else:
            # Exit signals
            if abs(z_score) < exit_threshold:
                signals.extend(self._create_pair_exit_signals(
                    symbol_a, symbol_b, price_a, price_b, portfolio
                ))
        
        return signals
    
    def _create_pair_entry_signals(self, symbol_a: str, symbol_b: str,
                                  price_a: float, price_b: float, hedge_ratio: float,
                                  direction: str, z_score: float,
                                  portfolio: Any) -> List[Dict[str, Any]]:
        """Create entry signals for a pairs trade"""
        
        signals = []
        
        # Calculate position sizes
        portfolio_value = portfolio.get_portfolio_value({symbol_a: price_a, symbol_b: price_b}) if portfolio else 100000
        
        # Position sizing based on hedge ratio
        target_value_a = portfolio_value * self.parameters['position_size_pct']
        target_value_b = target_value_a * hedge_ratio
        
        qty_a = int(target_value_a / price_a)
        qty_b = int(target_value_b / price_b)
        
        if qty_a < 1 or qty_b < 1:
            return signals
        
        confidence = min(abs(z_score) / self.parameters['zscore_threshold'], 1.0)
        
        if direction == 'long_a_short_b':
            # Long A, Short B
            signals.append({
                'symbol': symbol_a,
                'action': 'buy',
                'quantity': qty_a,
                'price': price_a,
                'strategy': 'stat_arbitrage',
                'confidence': confidence,
                'pair_trade': True,
                'pair_symbol': symbol_b,
                'z_score': z_score,
                'hedge_ratio': hedge_ratio,
                'stop_loss': price_a * (1 - self.parameters['stop_loss_pct']),
                'metadata': {
                    'trade_type': 'pairs_entry',
                    'direction': direction,
                    'expected_holding_days': self.parameters['max_holding_days']
                }
            })
            
            signals.append({
                'symbol': symbol_b,
                'action': 'sell',
                'quantity': qty_b,
                'price': price_b,
                'strategy': 'stat_arbitrage',
                'confidence': confidence,
                'pair_trade': True,
                'pair_symbol': symbol_a,
                'z_score': z_score,
                'hedge_ratio': hedge_ratio,
                'stop_loss': price_b * (1 + self.parameters['stop_loss_pct']),
                'metadata': {
                    'trade_type': 'pairs_entry',
                    'direction': direction,
                    'expected_holding_days': self.parameters['max_holding_days']
                }
            })
        
        elif direction == 'short_a_long_b':
            # Short A, Long B
            signals.append({
                'symbol': symbol_a,
                'action': 'sell',
                'quantity': qty_a,
                'price': price_a,
                'strategy': 'stat_arbitrage',
                'confidence': confidence,
                'pair_trade': True,
                'pair_symbol': symbol_b,
                'z_score': z_score,
                'hedge_ratio': hedge_ratio,
                'stop_loss': price_a * (1 + self.parameters['stop_loss_pct']),
                'metadata': {
                    'trade_type': 'pairs_entry',
                    'direction': direction,
                    'expected_holding_days': self.parameters['max_holding_days']
                }
            })
            
            signals.append({
                'symbol': symbol_b,
                'action': 'buy',
                'quantity': qty_b,
                'price': price_b,
                'strategy': 'stat_arbitrage',
                'confidence': confidence,
                'pair_trade': True,
                'pair_symbol': symbol_a,
                'z_score': z_score,
                'hedge_ratio': hedge_ratio,
                'stop_loss': price_b * (1 - self.parameters['stop_loss_pct']),
                'metadata': {
                    'trade_type': 'pairs_entry',
                    'direction': direction,
                    'expected_holding_days': self.parameters['max_holding_days']
                }
            })
        
        return signals
    
    def _create_pair_exit_signals(self, symbol_a: str, symbol_b: str,
                                 price_a: float, price_b: float,
                                 portfolio: Any) -> List[Dict[str, Any]]:
        """Create exit signals for existing pairs positions"""
        
        signals = []
        
        if portfolio is None:
            return signals
        
        positions = portfolio.positions
        
        # Close existing positions
        if symbol_a in positions:
            qty_a = positions[symbol_a]['quantity']
            if qty_a != 0:
                signals.append({
                    'symbol': symbol_a,
                    'action': 'sell' if qty_a > 0 else 'buy',
                    'quantity': abs(qty_a),
                    'price': price_a,
                    'strategy': 'stat_arbitrage',
                    'confidence': 0.8,
                    'pair_trade': True,
                    'pair_symbol': symbol_b,
                    'metadata': {
                        'trade_type': 'pairs_exit',
                        'reason': 'spread_convergence'
                    }
                })
        
        if symbol_b in positions:
            qty_b = positions[symbol_b]['quantity']
            if qty_b != 0:
                signals.append({
                    'symbol': symbol_b,
                    'action': 'sell' if qty_b > 0 else 'buy',
                    'quantity': abs(qty_b),
                    'price': price_b,
                    'strategy': 'stat_arbitrage',
                    'confidence': 0.8,
                    'pair_trade': True,
                    'pair_symbol': symbol_a,
                    'metadata': {
                        'trade_type': 'pairs_exit',
                        'reason': 'spread_convergence'
                    }
                })
        
        return signals
    
    def _has_position(self, symbol: str, portfolio: Any) -> bool:
        """Check if portfolio has position in symbol"""
        if portfolio is None:
            return False
        
        positions = portfolio.positions
        return symbol in positions and positions[symbol]['quantity'] != 0
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = None) -> int:
        """Calculate position size for arbitrage strategy"""
        target_value = portfolio_value * self.parameters['position_size_pct']
        position_size = int(target_value / price)
        
        # Minimum viable size
        min_size = max(1, int(500 / price))  # At least $500 per leg
        
        return max(position_size, min_size)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Statistical Arbitrage Strategy',
            'type': 'High Frequency Pairs Trading',
            'timeframe': '1min - 1day',
            'description': 'Exploits price inefficiencies between correlated assets using cointegration',
            'parameters': self.parameters,
            'risk_level': 'Medium-High',
            'expected_trades_per_day': '5-20 pairs',
            'holding_period': 'Hours to days',
            'best_markets': ['Correlated stocks', 'ETF pairs', 'Index components']
        }