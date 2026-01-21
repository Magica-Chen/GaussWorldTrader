"""
AI Agent Module for Gauss World Trader

This module provides intelligent analysis capabilities using multiple LLM providers
and financial data sources including Finnhub and FRED APIs.
"""

from .llm_providers import OpenAIProvider, DeepSeekProvider, ClaudeProvider, MoonshotProvider
from src.data.finnhub_provider import FinnhubProvider
from src.data.fred_provider import FREDProvider
from .fundamental_analyzer import FundamentalAnalyzer
from .agent_manager import AgentManager
from .notification_service import NotificationService, TradeStreamHandler

__all__ = [
    'OpenAIProvider', 'DeepSeekProvider', 'ClaudeProvider', 'MoonshotProvider',
    'FinnhubProvider', 'FREDProvider', 'FundamentalAnalyzer', 'AgentManager',
    'NotificationService', 'TradeStreamHandler'
]