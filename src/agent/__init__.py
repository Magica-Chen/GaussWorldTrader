"""
AI Agent Module for Gauss World Trader

Provides intelligent analysis using LLM providers and financial
data sources including Finnhub and FRED APIs.
"""

from src.data.finnhub_provider import FinnhubProvider
from src.data.fred_provider import FREDProvider
from src.llm import (
    ClaudeProvider,
    DeepSeekProvider,
    MoonshotProvider,
    OpenAIProvider,
)
from src.notify import NotificationService, TradeStreamHandler
from src.utils.asset_utils import (
    merge_symbol_sources,
    parse_symbol_args,
    positions_for_asset_type,
)
from src.watchlist import (
    WatchlistManager,
    get_default_watchlist,
    get_watchlist_manager,
)

from .agent_manager import AgentManager
from .fundamental_analyzer import FundamentalAnalyzer

__all__ = [
    "OpenAIProvider",
    "DeepSeekProvider",
    "ClaudeProvider",
    "MoonshotProvider",
    "FinnhubProvider",
    "FREDProvider",
    "FundamentalAnalyzer",
    "AgentManager",
    "NotificationService",
    "TradeStreamHandler",
    "WatchlistManager",
    "get_watchlist_manager",
    "get_default_watchlist",
    "parse_symbol_args",
    "positions_for_asset_type",
    "merge_symbol_sources",
]
