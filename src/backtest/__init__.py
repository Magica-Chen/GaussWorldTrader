"""
Backtesting Module for Gauss World Trader

This module contains all backtesting functionality moved from the trade module
to provide better organization and separation of concerns.
"""

from .backtester import Backtester
from .portfolio import Portfolio
from .performance_analyzer import PerformanceAnalyzer

__all__ = ['Backtester', 'Portfolio', 'PerformanceAnalyzer']