"""
Base Option Strategy Class

This module provides the abstract base class for all option trading strategies
in the Gauss World Trader system. It extends the concept of the base strategy
to include option-specific functionality.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple

if TYPE_CHECKING:
    from collections.abc import Sequence

from src.stock_strategy.base_strategy import BaseStrategy


class BaseOptionStrategy(BaseStrategy):
    """
    Abstract base class for option trading strategies.

    This class extends BaseStrategy to include option-specific functionality
    such as option filtering, scoring, and management of option positions
    alongside stock positions.
    """

    def __init__(self, parameters: Dict[str, Any] = None) -> None:
        """
        Initialize the base option strategy.

        Args:
            parameters: Strategy configuration parameters
        """
        super().__init__(parameters)

        # Option-specific state
        self.option_positions: Dict[str, Dict[str, Any]] = {}
        self.option_signals: List[Dict[str, Any]] = []

        # Load watchlist symbols instead of symbol_list.txt
        self.symbol_list = self._load_watchlist_symbols()

        # Default option strategy parameters
        self.default_params = {
            'max_risk': 80000,              # Maximum risk in dollars
            'delta_min': 0.15,              # Minimum delta (absolute value)
            'delta_max': 0.30,              # Maximum delta (absolute value)
            'yield_min': 0.04,              # Minimum yield (4%)
            'yield_max': 1.00,              # Maximum yield (100%)
            'dte_min': 0,                   # Minimum days to expiration
            'dte_max': 21,                  # Maximum days to expiration
            'min_open_interest': 100,       # Minimum open interest
            'min_score': 0.05,              # Minimum option score
            'position_size_pct': 0.1,       # Position size as % of portfolio
            'assignment_tolerance': 0.95,   # Tolerance for assignment risk
        }

        # Merge default params with provided params
        self.parameters = {**self.default_params, **self.parameters}

    def _load_watchlist_symbols(self) -> List[str]:
        """
        Load symbols from watchlist.json instead of config/symbol_list.txt.

        Returns:
            List of symbols from the watchlist
        """
        try:
            with open('watchlist.json', 'r') as f:
                watchlist_data = json.load(f)
                symbols = watchlist_data.get('watchlist', [])
                self.logger.info(f"Loaded {len(symbols)} symbols from watchlist.json")
                return symbols
        except Exception as e:
            self.logger.error(f"Failed to load watchlist.json: {e}")
            # Fallback to a few default symbols if watchlist fails
            default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
            self.logger.warning(f"Using default symbols: {default_symbols}")
            return default_symbols

    @abstractmethod
    def filter_underlying_stocks(self, client: Any) -> List[str]:
        """
        Filter underlying stocks based on strategy criteria.

        Args:
            client: Alpaca trading client

        Returns:
            List of filtered stock symbols
        """
        pass

    @abstractmethod
    def filter_options(self, client: Any, underlying: str,
                      option_type: str = 'put') -> List[Dict[str, Any]]:
        """
        Filter options based on strategy criteria.

        Args:
            client: Alpaca trading client
            underlying: Stock symbol
            option_type: 'put' or 'call'

        Returns:
            List of filtered option contracts
        """
        pass

    @abstractmethod
    def score_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score options based on strategy-specific criteria.

        Args:
            options: List of option contracts

        Returns:
            List of options with scores added
        """
        pass

    @abstractmethod
    def select_best_options(self, scored_options: List[Dict[str, Any]],
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select the best options based on scores.

        Args:
            scored_options: List of scored option contracts
            limit: Maximum number of options to select

        Returns:
            List of selected option contracts
        """
        pass

    def calculate_option_yield(self, option: Dict[str, Any]) -> float:
        """
        Calculate the yield of an option.

        Args:
            option: Option contract data

        Returns:
            Option yield as a percentage
        """
        try:
            bid_price = option.get('bid', 0)
            strike_price = option.get('strike_price', 0)

            if strike_price == 0:
                return 0.0

            # For puts: yield = bid / strike
            # For calls: yield = bid / (current_stock_price - strike) for ITM calls
            option_type = option.get('type', 'put').lower()

            if option_type == 'put':
                yield_pct = (bid_price / strike_price) * 100
            else:  # call
                current_price = option.get('underlying_price', strike_price)
                if current_price > strike_price:  # ITM call
                    yield_pct = (bid_price / (current_price - strike_price)) * 100
                else:  # OTM call
                    yield_pct = (bid_price / current_price) * 100

            return round(yield_pct, 2)

        except Exception as e:
            self.logger.error(f"Error calculating option yield: {e}")
            return 0.0

    def calculate_option_score(self, option: Dict[str, Any]) -> float:
        """
        Calculate option score using the wheel strategy scoring formula.

        The scoring formula is:
        score = (1 - |Δ|) × (250 / (DTE + 5)) × (bid price / strike price)

        Args:
            option: Option contract data

        Returns:
            Option score
        """
        try:
            delta = abs(option.get('delta', 0))
            dte = option.get('days_to_expiration', 1)
            bid_price = option.get('bid', 0)
            strike_price = option.get('strike_price', 1)

            # Avoid division by zero
            if strike_price == 0 or dte < 0:
                return 0.0

            # Wheel strategy scoring formula
            delta_component = 1 - delta
            time_component = 250 / (dte + 5)
            yield_component = bid_price / strike_price

            score = delta_component * time_component * yield_component

            return round(score, 4)

        except Exception as e:
            self.logger.error(f"Error calculating option score: {e}")
            return 0.0

    def check_option_assignment_risk(self, option: Dict[str, Any],
                                   underlying_price: float) -> Dict[str, Any]:
        """
        Check assignment risk for an option position.

        Args:
            option: Option contract data
            underlying_price: Current price of underlying stock

        Returns:
            Assignment risk analysis
        """
        try:
            option_type = option.get('type', 'put').lower()
            strike_price = option.get('strike_price', 0)
            expiration_date = option.get('expiration_date')

            # Calculate days to expiration
            if isinstance(expiration_date, str):
                exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            else:
                exp_date = expiration_date

            days_to_exp = (exp_date - datetime.now()).days

            # Assignment probability estimation
            if option_type == 'put':
                # Put is ITM if underlying < strike
                is_itm = underlying_price < strike_price
                distance_from_strike = abs(underlying_price - strike_price) / strike_price
            else:  # call
                # Call is ITM if underlying > strike
                is_itm = underlying_price > strike_price
                distance_from_strike = abs(underlying_price - strike_price) / strike_price

            # Simple assignment risk estimation
            if is_itm:
                assignment_prob = min(0.9, 0.5 + (0.4 / max(1, days_to_exp)))
            else:
                assignment_prob = max(0.1, distance_from_strike * 0.3)

            return {
                'is_itm': is_itm,
                'days_to_expiration': days_to_exp,
                'distance_from_strike_pct': distance_from_strike * 100,
                'assignment_probability': assignment_prob,
                'assignment_risk': 'HIGH' if assignment_prob > 0.7 else 'MEDIUM' if assignment_prob > 0.3 else 'LOW'
            }

        except Exception as e:
            self.logger.error(f"Error checking assignment risk: {e}")
            return {'assignment_risk': 'UNKNOWN', 'assignment_probability': 0.5}

    def manage_option_position(self, position: Dict[str, Any],
                             current_market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Manage an existing option position (roll, close, or hold).

        Args:
            position: Current option position
            current_market_data: Current market data

        Returns:
            List of management signals (roll, close, etc.)
        """
        signals = []

        try:
            option_symbol = position.get('symbol')
            underlying = position.get('underlying_symbol')
            current_price = current_market_data.get(underlying, {}).get('price', 0)

            # Get assignment risk
            assignment_risk = self.check_option_assignment_risk(position, current_price)

            # Management rules based on assignment risk and days to expiration
            days_to_exp = assignment_risk.get('days_to_expiration', 0)
            assignment_prob = assignment_risk.get('assignment_probability', 0)

            # Rule 1: Close if assignment probability is too high and close to expiration
            if days_to_exp <= 2 and assignment_prob > self.parameters['assignment_tolerance']:
                signals.append({
                    'symbol': option_symbol,
                    'action': 'BUY_TO_CLOSE',
                    'quantity': position.get('quantity', 1),
                    'reason': f'High assignment risk ({assignment_prob:.2%}) with {days_to_exp} days to expiration',
                    'priority': 'HIGH'
                })

            # Rule 2: Consider rolling if moderate assignment risk with time remaining
            elif days_to_exp <= 7 and assignment_prob > 0.5:
                signals.append({
                    'symbol': option_symbol,
                    'action': 'ROLL',
                    'quantity': position.get('quantity', 1),
                    'reason': f'Consider rolling - assignment risk {assignment_prob:.2%}',
                    'priority': 'MEDIUM'
                })

            # Rule 3: Profit taking if option has lost significant value
            current_option_price = current_market_data.get(option_symbol, {}).get('price', 0)
            entry_price = position.get('entry_price', 0)

            if entry_price > 0:
                profit_pct = (entry_price - current_option_price) / entry_price
                if profit_pct > 0.5:  # 50% profit
                    signals.append({
                        'symbol': option_symbol,
                        'action': 'BUY_TO_CLOSE',
                        'quantity': position.get('quantity', 1),
                        'reason': f'Profit taking - {profit_pct:.1%} profit achieved',
                        'priority': 'MEDIUM'
                    })

        except Exception as e:
            self.logger.error(f"Error managing option position: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information including option-specific metrics.

        Returns:
            Strategy information dictionary
        """
        base_info = super().get_strategy_info()

        option_info = {
            'option_positions': len(self.option_positions),
            'option_signals_generated': len(self.option_signals),
            'watchlist_symbols': len(self.symbol_list),
            'option_parameters': {
                'max_risk': self.parameters.get('max_risk'),
                'delta_range': f"{self.parameters.get('delta_min'):.2f} - {self.parameters.get('delta_max'):.2f}",
                'yield_range': f"{self.parameters.get('yield_min'):.1%} - {self.parameters.get('yield_max'):.1%}",
                'dte_range': f"{self.parameters.get('dte_min')} - {self.parameters.get('dte_max')} days",
                'min_open_interest': self.parameters.get('min_open_interest'),
                'min_score': self.parameters.get('min_score')
            }
        }

        # Merge base info with option-specific info
        return {**base_info, **option_info}

    def reset_strategy_state(self):
        """Reset strategy state including option positions."""
        super().reset_strategy_state()
        self.option_positions.clear()
        self.option_signals.clear()
        self.logger.info(f"Option strategy {self.name} state reset")