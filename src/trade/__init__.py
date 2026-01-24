from .trading_engine import TradingEngine
from .crypto_engine import TradingCryptoEngine
from .stock_engine import TradingStockEngine
from .option_engine import TradingOptionEngine
from .portfolio import Portfolio, FinancialMetrics, PerformanceAnalyzer, PortfolioTracker
from .backtester import Backtester
from .live_trading_base import LiveTradingEngine, PositionState

__all__ = [
    'TradingEngine',
    'TradingCryptoEngine',
    'TradingStockEngine',
    'TradingOptionEngine',
    'Portfolio',
    'FinancialMetrics',
    'PerformanceAnalyzer',
    'PortfolioTracker',
    'Backtester',
    'LiveTradingEngine',
    'PositionState',
    'LiveTradingCrypto',
    'LiveTradingStock',
    'LiveTradingOption',
]


def __getattr__(name: str):
    if name == 'LiveTradingCrypto':
        from .live_trading_crypto import LiveTradingCrypto
        return LiveTradingCrypto
    if name == 'LiveTradingStock':
        from .live_trading_stock import LiveTradingStock
        return LiveTradingStock
    if name == 'LiveTradingOption':
        from .live_trading_option import LiveTradingOption
        return LiveTradingOption
    raise AttributeError(f"module 'src.trade' has no attribute {name!r}")
