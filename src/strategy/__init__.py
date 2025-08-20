"""
Trading Strategy Framework

This module contains all trading strategy implementations
Includes classical and modern ML-based strategies
"""

from .base_strategy import BaseStrategy
from .momentum_strategy import MomentumStrategy

# Classical High-Frequency Strategies
from .scalping_strategy import ScalpingStrategy
from .arbitrage_strategy import StatisticalArbitrageStrategy

# Classical Low-Frequency Strategies  
from .trend_following_strategy import TrendFollowingStrategy
from .value_strategy import ValueInvestmentStrategy

# Modern ML-Based Strategies
from .xgboost_strategy import XGBoostStrategy
from .deep_learning_strategy import DeepLearningStrategy
from .gaussian_process_strategy import GaussianProcessStrategy

__all__ = [
    'BaseStrategy', 
    'MomentumStrategy',
    # High Frequency
    'ScalpingStrategy',
    'StatisticalArbitrageStrategy', 
    # Low Frequency
    'TrendFollowingStrategy',
    'ValueInvestmentStrategy',
    # Machine Learning
    'XGBoostStrategy',
    'DeepLearningStrategy', 
    'GaussianProcessStrategy'
]