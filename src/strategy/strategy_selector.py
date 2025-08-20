"""
Strategy Selector and Manager

Provides utilities for selecting, configuring, and managing trading strategies
"""

from typing import Dict, List, Any, Optional, Type
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from .momentum_strategy import MomentumStrategy
from .scalping_strategy import ScalpingStrategy
from .arbitrage_strategy import StatisticalArbitrageStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .value_strategy import ValueInvestmentStrategy
from .xgboost_strategy import XGBoostStrategy
from .deep_learning_strategy import DeepLearningStrategy
from .gaussian_process_strategy import GaussianProcessStrategy

class StrategySelector:
    """
    Strategy selection and management utility
    
    Provides methods to:
    - List available strategies
    - Create strategy instances with parameters
    - Get strategy recommendations based on market conditions
    - Manage strategy portfolios
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Registry of all available strategies
        self.strategy_registry = {
            # Classical Strategies
            'momentum': MomentumStrategy,
            'scalping': ScalpingStrategy,
            'statistical_arbitrage': StatisticalArbitrageStrategy,
            'trend_following': TrendFollowingStrategy,
            'value_investment': ValueInvestmentStrategy,
            
            # ML Strategies
            'xgboost': XGBoostStrategy,
            'deep_learning': DeepLearningStrategy,
            'gaussian_process': GaussianProcessStrategy
        }
        
        # Strategy categories
        self.strategy_categories = {
            'high_frequency': ['scalping', 'statistical_arbitrage'],
            'low_frequency': ['trend_following', 'value_investment'],
            'medium_frequency': ['momentum'],
            'machine_learning': ['xgboost', 'deep_learning', 'gaussian_process'],
            'classical': ['momentum', 'scalping', 'statistical_arbitrage', 'trend_following', 'value_investment'],
            'modern': ['xgboost', 'deep_learning', 'gaussian_process']
        }
    
    def list_strategies(self, category: Optional[str] = None) -> List[str]:
        """List available strategies, optionally filtered by category"""
        
        if category is None:
            return list(self.strategy_registry.keys())
        
        if category in self.strategy_categories:
            return self.strategy_categories[category]
        else:
            self.logger.warning(f"Unknown category: {category}")
            return []
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed information about a strategy"""
        
        if strategy_name not in self.strategy_registry:
            return {"error": f"Strategy '{strategy_name}' not found"}
        
        try:
            # Create temporary instance to get info
            strategy_class = self.strategy_registry[strategy_name]
            temp_strategy = strategy_class()
            
            info = temp_strategy.get_strategy_info()
            info['strategy_name'] = strategy_name
            info['category'] = self._get_strategy_category(strategy_name)
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting info for strategy {strategy_name}: {e}")
            return {"error": str(e)}
    
    def create_strategy(self, strategy_name: str, parameters: Dict[str, Any] = None) -> Optional[BaseStrategy]:
        """Create a strategy instance with optional parameters"""
        
        if strategy_name not in self.strategy_registry:
            self.logger.error(f"Strategy '{strategy_name}' not found")
            return None
        
        try:
            strategy_class = self.strategy_registry[strategy_name]
            
            if parameters:
                strategy = strategy_class(parameters)
            else:
                strategy = strategy_class()
            
            self.logger.info(f"Created strategy: {strategy_name}")
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error creating strategy {strategy_name}: {e}")
            return None
    
    def get_strategy_recommendations(self, market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend strategies based on market conditions"""
        
        recommendations = []
        
        # Extract market condition indicators
        volatility = market_conditions.get('volatility', 'medium')
        trend_strength = market_conditions.get('trend_strength', 'medium')
        market_regime = market_conditions.get('regime', 'normal')
        trading_session = market_conditions.get('session', 'regular')
        liquidity = market_conditions.get('liquidity', 'high')
        
        # High volatility conditions
        if volatility == 'high':
            recommendations.append({
                'strategy': 'scalping',
                'reason': 'High volatility provides scalping opportunities',
                'confidence': 0.8,
                'suggested_params': {'vwap_threshold': 0.003, 'position_size_pct': 0.03}
            })
            
            recommendations.append({
                'strategy': 'statistical_arbitrage',
                'reason': 'Volatility creates pricing inefficiencies',
                'confidence': 0.7,
                'suggested_params': {'zscore_threshold': 1.5}
            })
        
        # Strong trend conditions
        if trend_strength == 'strong':
            recommendations.append({
                'strategy': 'trend_following',
                'reason': 'Strong trend favors trend following strategies',
                'confidence': 0.9,
                'suggested_params': {'min_trend_strength': 0.015}
            })
            
            recommendations.append({
                'strategy': 'momentum',
                'reason': 'Momentum strategies work well in trending markets',
                'confidence': 0.8,
                'suggested_params': {'momentum_threshold': 0.02}
            })
        
        # Low volatility, ranging market
        if volatility == 'low' and trend_strength == 'weak':
            recommendations.append({
                'strategy': 'value_investment',
                'reason': 'Low volatility suitable for fundamental analysis',
                'confidence': 0.7,
                'suggested_params': {'rebalance_frequency': 60}
            })
            
            recommendations.append({
                'strategy': 'statistical_arbitrage',
                'reason': 'Mean reversion opportunities in ranging markets',
                'confidence': 0.6,
                'suggested_params': {'zscore_threshold': 2.5}
            })
        
        # Machine learning recommendations
        if market_regime == 'uncertain' or market_regime == 'complex':
            recommendations.append({
                'strategy': 'gaussian_process',
                'reason': 'GP handles uncertainty and provides confidence intervals',
                'confidence': 0.6,
                'suggested_params': {'uncertainty_threshold': 0.03}
            })
            
            recommendations.append({
                'strategy': 'xgboost',
                'reason': 'ML can capture complex non-linear patterns',
                'confidence': 0.7,
                'suggested_params': {'retrain_frequency': 21}
            })
        
        # High-frequency trading during active sessions
        if trading_session == 'active' and liquidity == 'high':
            recommendations.append({
                'strategy': 'scalping',
                'reason': 'Active session with high liquidity ideal for scalping',
                'confidence': 0.8,
                'suggested_params': {'max_holding_minutes': 30}
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations
    
    def create_strategy_portfolio(self, strategy_configs: List[Dict[str, Any]]) -> Dict[str, BaseStrategy]:
        """Create a portfolio of strategies with different allocations"""
        
        portfolio = {}
        
        for config in strategy_configs:
            strategy_name = config.get('strategy')
            parameters = config.get('parameters', {})
            allocation = config.get('allocation', 1.0)
            
            if strategy_name not in self.strategy_registry:
                self.logger.warning(f"Unknown strategy: {strategy_name}")
                continue
            
            # Adjust position sizing based on allocation
            if 'position_size_pct' in parameters:
                parameters['position_size_pct'] *= allocation
            elif allocation != 1.0:
                # Add allocation-adjusted position sizing
                parameters['position_size_pct'] = 0.05 * allocation  # Default 5% * allocation
            
            strategy = self.create_strategy(strategy_name, parameters)
            if strategy:
                portfolio[f"{strategy_name}_{allocation}"] = strategy
        
        return portfolio
    
    def get_strategy_compatibility(self, strategy1: str, strategy2: str) -> Dict[str, Any]:
        """Check compatibility between two strategies for portfolio use"""
        
        if strategy1 not in self.strategy_registry or strategy2 not in self.strategy_registry:
            return {"compatible": False, "reason": "One or both strategies not found"}
        
        # Get strategy info
        info1 = self.get_strategy_info(strategy1)
        info2 = self.get_strategy_info(strategy2)
        
        compatibility = {
            "compatible": True,
            "conflicts": [],
            "recommendations": []
        }
        
        # Check timeframe compatibility
        timeframe1 = info1.get('timeframe', '')
        timeframe2 = info2.get('timeframe', '')
        
        if 'hour' in timeframe1.lower() and 'day' in timeframe2.lower():
            compatibility["recommendations"].append("Different timeframes - good diversification")
        elif timeframe1 == timeframe2:
            compatibility["conflicts"].append("Same timeframe may lead to signal conflicts")
        
        # Check market compatibility
        risk1 = info1.get('risk_level', 'Medium')
        risk2 = info2.get('risk_level', 'Medium')
        
        if risk1 == 'High' and risk2 == 'High':
            compatibility["conflicts"].append("Both high-risk strategies may increase portfolio risk")
        
        # Check strategy types
        cat1 = self._get_strategy_category(strategy1)
        cat2 = self._get_strategy_category(strategy2)
        
        if 'machine_learning' in [cat1, cat2] and 'classical' in [cat1, cat2]:
            compatibility["recommendations"].append("Good mix of classical and ML approaches")
        
        # Set overall compatibility
        compatibility["compatible"] = len(compatibility["conflicts"]) <= 1
        
        return compatibility
    
    def optimize_strategy_parameters(self, strategy_name: str, 
                                   historical_data: Dict[str, Any],
                                   optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize strategy parameters using historical data
        This is a placeholder for more sophisticated parameter optimization
        """
        
        if strategy_name not in self.strategy_registry:
            return {"error": f"Strategy '{strategy_name}' not found"}
        
        # This would implement parameter optimization using techniques like:
        # - Grid search
        # - Bayesian optimization
        # - Genetic algorithms
        # - Walk-forward optimization
        
        # For now, return default parameters with optimization notes
        strategy_info = self.get_strategy_info(strategy_name)
        default_params = strategy_info.get('parameters', {})
        
        return {
            "optimized_parameters": default_params,
            "optimization_method": "placeholder",
            "metric_achieved": 0.0,
            "note": "Parameter optimization not yet implemented - using defaults"
        }
    
    def _get_strategy_category(self, strategy_name: str) -> str:
        """Get the primary category for a strategy"""
        
        for category, strategies in self.strategy_categories.items():
            if strategy_name in strategies:
                return category
        
        return "unknown"
    
    def get_all_strategies_overview(self) -> Dict[str, Any]:
        """Get overview of all available strategies"""
        
        overview = {
            "total_strategies": len(self.strategy_registry),
            "categories": {},
            "strategies": {}
        }
        
        # Count by category
        for category, strategies in self.strategy_categories.items():
            overview["categories"][category] = len(strategies)
        
        # Get info for each strategy
        for strategy_name in self.strategy_registry.keys():
            info = self.get_strategy_info(strategy_name)
            overview["strategies"][strategy_name] = {
                "type": info.get("type", "Unknown"),
                "timeframe": info.get("timeframe", "Unknown"),
                "risk_level": info.get("risk_level", "Unknown"),
                "description": info.get("description", "No description")
            }
        
        return overview

# Global strategy selector instance
strategy_selector = StrategySelector()

def get_strategy_selector() -> StrategySelector:
    """Get the global strategy selector instance"""
    return strategy_selector

def list_available_strategies() -> List[str]:
    """Convenience function to list all available strategies"""
    return strategy_selector.list_strategies()

def create_strategy(strategy_name: str, parameters: Dict[str, Any] = None) -> Optional[BaseStrategy]:
    """Convenience function to create a strategy"""
    return strategy_selector.create_strategy(strategy_name, parameters)