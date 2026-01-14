from .momentum import MomentumStrategy
from .value import ValueStrategy
from .trend_following import TrendFollowingStrategy
from .scalping import ScalpingStrategy
from .statistical_arbitrage import StatisticalArbitrageStrategy
from .gaussian_process import GaussianProcessStrategy
from .xgboost import XGBoostStrategy
from .deep_learning import DeepLearningStrategy

__all__ = [
    "MomentumStrategy",
    "ValueStrategy",
    "TrendFollowingStrategy",
    "ScalpingStrategy",
    "StatisticalArbitrageStrategy",
    "GaussianProcessStrategy",
    "XGBoostStrategy",
    "DeepLearningStrategy",
]
