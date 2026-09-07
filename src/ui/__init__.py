"""UI exports loaded on demand so session views do not initialize legacy trading code."""

from importlib import import_module

_EXPORTS = {
    "Dashboard": "dashboard",
    "UIComponents": "ui_components",
    "MarketViewsMixin": "market_views",
    "AccountViewsMixin": "account_views",
    "TradingViewsMixin": "trading_views",
    "AnalysisViewsMixin": "analysis_views",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
