from .engine import (
    TradingEngine,
    TradingCryptoEngine,
    TradingStockEngine,
    TradingOptionEngine,
    ExecutionEngine,
    ExecutionContext,
    ExecutionDecision,
)
from .portfolio import (
    Portfolio,
    FinancialMetrics,
    PerformanceAnalyzer,
    PortfolioTracker,
)
from .live import LiveTradingEngine, PositionState

__all__ = [
    "TradingEngine",
    "TradingCryptoEngine",
    "TradingStockEngine",
    "TradingOptionEngine",
    "ExecutionEngine",
    "ExecutionContext",
    "ExecutionDecision",
    "Portfolio",
    "FinancialMetrics",
    "PerformanceAnalyzer",
    "PortfolioTracker",
    "Backtester",
    "LiveTradingEngine",
    "PositionState",
    "LiveTradingCrypto",
    "LiveTradingStock",
    "LiveTradingOption",
]


def __getattr__(name: str):
    if name == "Backtester":
        from src.backtest import Backtester
        return Backtester
    if name == "LiveTradingCrypto":
        from .live.live_trading_crypto import LiveTradingCrypto
        return LiveTradingCrypto
    if name == "LiveTradingStock":
        from .live.live_trading_stock import LiveTradingStock
        return LiveTradingStock
    if name == "LiveTradingOption":
        from .live.live_trading_option import LiveTradingOption
        return LiveTradingOption
    raise AttributeError(
        f"module 'src.trade' has no attribute {name!r}"
    )
