"""
AI Agent Module for Gauss World Trader

Provides intelligent analysis using LLM providers and financial
data sources including Finnhub and FRED APIs.
"""

from src.llm import (
    OpenAIProvider,
    DeepSeekProvider,
    ClaudeProvider,
    MoonshotProvider,
)
from src.data.finnhub_provider import FinnhubProvider
from src.data.fred_provider import FREDProvider
from .fundamental_analyzer import FundamentalAnalyzer
from .agent_manager import AgentManager
from src.notify import NotificationService, TradeStreamHandler
from src.watchlist import (
    WatchlistManager,
    get_watchlist_manager,
    get_default_watchlist,
)
from src.utils.asset_utils import (
    parse_symbol_args,
    positions_for_asset_type,
    merge_symbol_sources,
)

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
