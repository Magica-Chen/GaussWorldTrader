"""
Value Investment Strategy - Low Frequency

Fundamental analysis-based value investing strategy
Focuses on undervalued securities with strong fundamentals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseStrategy

class ValueInvestmentStrategy(BaseStrategy):
    """
    Value investment strategy based on:
    - Price-to-earnings ratios
    - Price-to-book ratios
    - Dividend yield analysis
    - Revenue growth
    - Technical confirmation signals
    - Relative strength vs market
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'max_pe_ratio': 20,           # Maximum acceptable P/E ratio
            'min_dividend_yield': 0.02,   # Minimum 2% dividend yield
            'max_pb_ratio': 3.0,          # Maximum price-to-book ratio
            'min_revenue_growth': 0.05,   # Minimum 5% revenue growth
            'position_size_pct': 0.10,    # 10% of portfolio per position
            'rebalance_frequency': 90,    # Rebalance every 90 days
            'min_market_cap': 1e9,        # Minimum $1B market cap
            'max_debt_to_equity': 0.6,    # Maximum debt-to-equity ratio
            'min_current_ratio': 1.2,     # Minimum current ratio
            'technical_confirmation': True, # Require technical confirmation
            'relative_strength_period': 252, # 1 year relative strength
            'value_score_threshold': 0.6   # Minimum value score
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """Generate value investment signals"""
        
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_prices or data.empty:
                continue
            
            try:
                signal = self._analyze_value_opportunity(
                    symbol, data, current_prices[symbol], 
                    current_data.get(symbol, {}), portfolio
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} for value investing: {e}")
                continue
        
        return signals
    
    def _analyze_value_opportunity(self, symbol: str, data: pd.DataFrame,
                                  current_price: float, current_bar: Dict[str, Any],
                                  portfolio: Any = None) -> Optional[Dict[str, Any]]:
        """Analyze value investment opportunity for a single symbol"""
        
        if len(data) < 252:  # Need at least 1 year of data
            return None
        
        # Get fundamental data (would typically come from external API)
        fundamental_data = self._get_fundamental_data(symbol)
        
        if not fundamental_data:
            return None
        
        # Calculate value score
        value_score = self._calculate_value_score(fundamental_data, current_price)
        
        if value_score < self.parameters['value_score_threshold']:
            return None
        
        # Technical confirmation if required
        if self.parameters['technical_confirmation']:
            technical_signal = self._check_technical_confirmation(data, current_price)
            if technical_signal != 'buy':
                return None
        
        # Portfolio constraints
        if not self._check_portfolio_constraints(symbol, portfolio):
            return None
        
        # Calculate position size
        portfolio_value = portfolio.get_portfolio_value({symbol: current_price}) if portfolio else 100000
        position_size = self.calculate_position_size(symbol, current_price, portfolio_value)
        
        # Calculate target price and holding period
        target_price = self._calculate_target_price(fundamental_data, current_price)
        expected_return = (target_price - current_price) / current_price
        
        return {
            'symbol': symbol,
            'action': 'buy',
            'quantity': position_size,
            'price': current_price,
            'strategy': 'value_investment',
            'confidence': value_score,
            'target_price': target_price,
            'expected_return': expected_return,
            'holding_period_target': timedelta(days=180),  # 6 months minimum
            'value_metrics': {
                'value_score': value_score,
                'pe_ratio': fundamental_data.get('pe_ratio'),
                'pb_ratio': fundamental_data.get('pb_ratio'),
                'dividend_yield': fundamental_data.get('dividend_yield'),
                'roe': fundamental_data.get('roe'),
                'debt_to_equity': fundamental_data.get('debt_to_equity')
            },
            'fundamental_data': fundamental_data,
            'rebalance_date': datetime.now() + timedelta(days=self.parameters['rebalance_frequency'])
        }
    
    def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for the symbol
        In a real implementation, this would fetch from financial APIs
        For demonstration, we'll simulate some fundamental data
        """
        
        # This is simulated data - in practice, you'd fetch from:
        # - Alpha Vantage, Yahoo Finance, IEX Cloud, etc.
        # - Or use the Finnhub integration from the agent module
        
        np.random.seed(hash(symbol) % 2**32)  # Deterministic "random" for demo
        
        # Simulate fundamental ratios
        pe_ratio = np.random.uniform(8, 25)
        pb_ratio = np.random.uniform(0.8, 4.0)
        dividend_yield = np.random.uniform(0, 0.08)  # 0-8%
        roe = np.random.uniform(0.05, 0.25)  # 5-25%
        debt_to_equity = np.random.uniform(0.1, 1.0)
        current_ratio = np.random.uniform(0.8, 3.0)
        revenue_growth = np.random.uniform(-0.1, 0.3)  # -10% to 30%
        market_cap = np.random.uniform(5e8, 1e12)  # $500M to $1T
        
        return {
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'dividend_yield': dividend_yield,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'revenue_growth': revenue_growth,
            'market_cap': market_cap,
            'book_value_per_share': np.random.uniform(10, 100),
            'earnings_per_share': np.random.uniform(1, 10),
            'free_cash_flow': np.random.uniform(1e6, 1e10)
        }
    
    def _calculate_value_score(self, fundamental_data: Dict[str, Any], 
                              current_price: float) -> float:
        """Calculate comprehensive value score (0-1)"""
        
        score = 0.0
        max_score = 0.0
        
        # P/E Ratio scoring (lower is better)
        pe_ratio = fundamental_data.get('pe_ratio', 999)
        if pe_ratio <= self.parameters['max_pe_ratio']:
            pe_score = max(0, (self.parameters['max_pe_ratio'] - pe_ratio) / self.parameters['max_pe_ratio'])
            score += pe_score * 0.25
        max_score += 0.25
        
        # P/B Ratio scoring (lower is better)
        pb_ratio = fundamental_data.get('pb_ratio', 999)
        if pb_ratio <= self.parameters['max_pb_ratio']:
            pb_score = max(0, (self.parameters['max_pb_ratio'] - pb_ratio) / self.parameters['max_pb_ratio'])
            score += pb_score * 0.20
        max_score += 0.20
        
        # Dividend Yield scoring (higher is better)
        dividend_yield = fundamental_data.get('dividend_yield', 0)
        if dividend_yield >= self.parameters['min_dividend_yield']:
            div_score = min(1.0, dividend_yield / 0.08)  # Cap at 8%
            score += div_score * 0.15
        max_score += 0.15
        
        # ROE scoring (higher is better)
        roe = fundamental_data.get('roe', 0)
        roe_score = min(1.0, roe / 0.20)  # Cap at 20%
        score += roe_score * 0.20
        max_score += 0.20
        
        # Debt-to-Equity scoring (lower is better)
        debt_to_equity = fundamental_data.get('debt_to_equity', 999)
        if debt_to_equity <= self.parameters['max_debt_to_equity']:
            debt_score = max(0, (self.parameters['max_debt_to_equity'] - debt_to_equity) / self.parameters['max_debt_to_equity'])
            score += debt_score * 0.10
        max_score += 0.10
        
        # Current Ratio scoring (higher is better, but diminishing returns)
        current_ratio = fundamental_data.get('current_ratio', 0)
        if current_ratio >= self.parameters['min_current_ratio']:
            current_score = min(1.0, (current_ratio - 1.0) / 2.0)  # Optimal around 3.0
            score += current_score * 0.10
        max_score += 0.10
        
        # Revenue Growth scoring
        revenue_growth = fundamental_data.get('revenue_growth', -1)
        if revenue_growth >= self.parameters['min_revenue_growth']:
            growth_score = min(1.0, revenue_growth / 0.20)  # Cap at 20%
            score += growth_score * 0.10
        max_score += 0.10
        
        # Market Cap requirement (binary)
        market_cap = fundamental_data.get('market_cap', 0)
        if market_cap >= self.parameters['min_market_cap']:
            max_score += 0.0  # No additional score, just a filter
        else:
            return 0.0  # Fail if below minimum market cap
        
        # Normalize score
        final_score = score / max_score if max_score > 0 else 0.0
        
        return min(1.0, final_score)
    
    def _check_technical_confirmation(self, data: pd.DataFrame, current_price: float) -> str:
        """Check for technical confirmation of value opportunity"""
        
        # Calculate technical indicators
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        rsi = self._calculate_rsi(data['close'], 14)
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(20).mean()
        recent_volume = data['volume'].rolling(5).mean()
        
        if len(sma_20) == 0 or len(sma_50) == 0 or len(rsi) == 0:
            return 'neutral'
        
        # Technical conditions for value confirmation
        conditions = []
        
        # Price above short-term moving average
        conditions.append(current_price > sma_20.iloc[-1])
        
        # Short MA above long MA (uptrend)
        conditions.append(sma_20.iloc[-1] > sma_50.iloc[-1])
        
        # RSI not overbought
        conditions.append(rsi.iloc[-1] < 70)
        
        # RSI not oversold (some momentum)
        conditions.append(rsi.iloc[-1] > 35)
        
        # Volume confirmation
        if len(avg_volume) > 0 and avg_volume.iloc[-1] > 0:
            volume_ratio = recent_volume.iloc[-1] / avg_volume.iloc[-1]
            conditions.append(volume_ratio > 0.8)  # Reasonable volume
        
        # Relative strength vs recent lows
        recent_low = data['low'].rolling(50).min().iloc[-1]
        conditions.append(current_price > recent_low * 1.05)  # 5% above recent low
        
        # At least 4 out of 6 conditions must be met
        if sum(conditions) >= 4:
            return 'buy'
        else:
            return 'neutral'
    
    def _calculate_target_price(self, fundamental_data: Dict[str, Any], 
                               current_price: float) -> float:
        """Calculate fair value target price"""
        
        # Multiple valuation approaches
        
        # 1. P/E based valuation
        eps = fundamental_data.get('earnings_per_share', 0)
        fair_pe = min(15, self.parameters['max_pe_ratio'] * 0.8)  # Conservative P/E
        pe_target = eps * fair_pe
        
        # 2. P/B based valuation
        book_value = fundamental_data.get('book_value_per_share', current_price)
        fair_pb = min(2.0, self.parameters['max_pb_ratio'] * 0.7)  # Conservative P/B
        pb_target = book_value * fair_pb
        
        # 3. Dividend discount model (simplified)
        dividend_per_share = current_price * fundamental_data.get('dividend_yield', 0)
        required_return = 0.10  # 10% required return
        growth_rate = min(0.05, fundamental_data.get('revenue_growth', 0))  # Conservative growth
        
        if required_return > growth_rate and dividend_per_share > 0:
            ddm_target = dividend_per_share * (1 + growth_rate) / (required_return - growth_rate)
        else:
            ddm_target = current_price
        
        # 4. Asset-based valuation
        asset_target = book_value * 1.2  # 20% premium to book value
        
        # Weight the different approaches
        targets = [pe_target, pb_target, ddm_target, asset_target]
        valid_targets = [t for t in targets if t > 0 and t < current_price * 3]  # Reasonable range
        
        if valid_targets:
            # Use median to reduce impact of outliers
            target_price = np.median(valid_targets)
        else:
            # Fallback to conservative estimate
            target_price = current_price * 1.2
        
        # Conservative cap
        return min(target_price, current_price * 2.0)
    
    def _check_portfolio_constraints(self, symbol: str, portfolio: Any) -> bool:
        """Check portfolio-level constraints for value investing"""
        if portfolio is None:
            return True
        
        # Check if we already have a large position
        if symbol in portfolio.positions:
            current_qty = portfolio.positions[symbol]['quantity']
            current_value = abs(current_qty) * portfolio.positions[symbol].get('average_cost', 0)
            portfolio_value = portfolio.get_portfolio_value({})
            
            position_pct = current_value / portfolio_value
            if position_pct > self.parameters['position_size_pct'] * 0.5:
                return False  # Already have significant position
        
        # Check sector concentration (would need sector data)
        # This is a placeholder for sector diversification logic
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = None) -> int:
        """Calculate position size for value investing"""
        
        # Value investing typically uses equal weighting
        target_value = portfolio_value * self.parameters['position_size_pct']
        position_size = int(target_value / price)
        
        # Minimum viable size for value investing
        min_size = max(1, int(5000 / price))  # At least $5000 position
        
        return max(position_size, min_size)
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """Check if portfolio should be rebalanced"""
        days_since_rebalance = (current_date - last_rebalance).days
        return days_since_rebalance >= self.parameters['rebalance_frequency']
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Value Investment Strategy',
            'type': 'Low Frequency Value',
            'timeframe': 'Quarterly to Yearly',
            'description': 'Fundamental analysis-based value investing with technical confirmation',
            'parameters': self.parameters,
            'risk_level': 'Low-Medium',
            'expected_trades_per_day': '0.1-0.5',
            'holding_period': 'Months to years',
            'best_markets': ['Large cap stocks', 'Dividend stocks', 'REITs', 'Value ETFs'],
            'rebalance_frequency': f"{self.parameters['rebalance_frequency']} days"
        }