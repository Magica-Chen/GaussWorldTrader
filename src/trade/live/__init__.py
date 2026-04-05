from .live_trading_base import LiveTradingEngine, PositionState
from .live_runner import run_live_engines

__all__ = [
    "LiveTradingEngine",
    "PositionState",
    "run_live_engines",
]


def __getattr__(name: str):
    if name == "LiveTradingCrypto":
        from .live_trading_crypto import LiveTradingCrypto
        return LiveTradingCrypto
    if name == "LiveTradingStock":
        from .live_trading_stock import LiveTradingStock
        return LiveTradingStock
    if name == "LiveTradingOption":
        from .live_trading_option import LiveTradingOption
        return LiveTradingOption
    raise AttributeError(
        f"module 'src.trade.live' has no attribute {name!r}"
    )
