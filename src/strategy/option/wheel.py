"""
Wheel Options Strategy Implementation

The Wheel Strategy is a systematic options trading approach designed to generate
consistent income while potentially acquiring quality stocks at favorable prices.

Strategy Overview:
1. CASH-SECURED PUTS: Sell put options on stocks you're willing to own
   - Collect premium income
   - If assigned, purchase shares at predetermined strike price

2. COVERED CALLS: After assignment, sell covered calls on owned shares
   - Generate additional premium income
   - If called away, sell shares at strike price and restart cycle

3. WHEEL CONTINUATION: Repeat the process with new put options

This implementation integrates with the GaussWorldTrader system and uses
symbols from watchlist.json instead of a separate configuration file.
"""

from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

from src.strategy.base import BaseOptionStrategy, StrategyMeta


class WheelStrategy(BaseOptionStrategy):
    """
    Implementation of the Wheel Options Strategy.

    The Wheel Strategy systematically sells cash-secured puts and covered calls
    to generate income while managing stock positions through assignment cycles.
    """

    meta = StrategyMeta(
        name="wheel",
        label="Wheel",
        category="option",
        description="Options wheel strategy for income-focused trading.",
        asset_type="option",
        visible_in_dashboard=False,
        default_params={},
    )

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Wheel Strategy.

        Args:
            parameters: Strategy configuration parameters
        """
        super().__init__(parameters)

        self.name = "WheelStrategy"

        # Wheel-specific parameters with detailed explanations
        wheel_defaults = {
            # RISK MANAGEMENT PARAMETERS
            'max_risk': 80000,                    # Maximum total risk exposure in dollars
            'max_positions': 10,                  # Maximum number of concurrent positions
            'position_size_pct': 0.08,           # Position size as % of portfolio (8%)

            # DELTA PARAMETERS (Assignment Probability)
            'put_delta_min': 0.15,               # Min delta for puts (15% assignment prob)
            'put_delta_max': 0.30,               # Max delta for puts (30% assignment prob)
            'call_delta_min': 0.15,              # Min delta for calls (15% assignment prob)
            'call_delta_max': 0.30,              # Max delta for calls (30% assignment prob)

            # YIELD REQUIREMENTS
            'min_yield': 0.04,                   # Minimum yield (4% return)
            'max_yield': 1.00,                   # Maximum yield (100% - filter out suspiciously high)

            # TIME TO EXPIRATION
            'dte_min': 7,                        # Minimum days to expiration
            'dte_max': 45,                       # Maximum days to expiration
            'preferred_dte': 21,                 # Preferred days to expiration (3 weeks)

            # LIQUIDITY REQUIREMENTS
            'min_open_interest': 100,            # Minimum open interest for liquidity
            'min_daily_volume': 50,              # Minimum daily volume

            # SCORING AND SELECTION
            'min_score': 0.05,                   # Minimum option score threshold
            'max_options_per_underlying': 1,     # Max options per stock symbol

            # ASSIGNMENT AND MANAGEMENT
            'assignment_tolerance': 0.80,        # Close position if assignment prob > 80%
            'profit_target': 0.50,               # Close at 50% profit
            'management_dte': 7,                 # Start managing when DTE <= 7

            # STOCK SELECTION CRITERIA
            'min_stock_price': 10.0,             # Minimum stock price
            'max_stock_price': 500.0,            # Maximum stock price
            'min_market_cap': 1_000_000_000,     # Minimum market cap ($1B)
        }

        # Merge wheel-specific defaults with provided parameters
        self.parameters = {**wheel_defaults, **self.parameters}

        # Track strategy state
        self.current_cycle_stage = {}  # Track which stage each symbol is in
        self.assignment_history = []   # Track assignment events
        self.profit_history = []       # Track profit/loss from closed positions

        self.logger.info(f"Wheel Strategy initialized with {len(self.symbol_list)} watchlist symbols")

    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any],
                        historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """
        Generate wheel strategy trading signals.

        The wheel strategy operates in cycles:
        1. Analyze current positions and market conditions
        2. Manage existing option positions
        3. Look for new cash-secured put opportunities
        4. Look for covered call opportunities on owned stocks

        Args:
            current_date: Current trading date
            current_prices: Current stock prices
            current_data: Current market data
            historical_data: Historical price data
            portfolio: Portfolio object

        Returns:
            List of trading signals
        """
        signals = []

        try:
            self.logger.info(f"Generating wheel strategy signals for {current_date}")

            # STEP 1: Manage existing option positions
            management_signals = self._manage_existing_positions(current_prices, portfolio)
            signals.extend(management_signals)

            # STEP 2: Identify cash-secured put opportunities
            put_signals = self._find_cash_secured_put_opportunities(
                current_prices, current_data, portfolio
            )
            signals.extend(put_signals)

            # STEP 3: Identify covered call opportunities
            call_signals = self._find_covered_call_opportunities(
                current_prices, current_data, portfolio
            )
            signals.extend(call_signals)

            # STEP 4: Apply risk management and position limits
            filtered_signals = self._apply_risk_management(signals, portfolio)

            # Log signal generation summary
            self.logger.info(
                f"Generated {len(filtered_signals)} wheel strategy signals: "
                f"{len(management_signals)} management, "
                f"{len(put_signals)} puts, "
                f"{len(call_signals)} calls"
            )

            # Record signals for tracking
            for signal in filtered_signals:
                self.log_signal(signal)

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error generating wheel strategy signals: {e}")
            return []

    def _manage_existing_positions(self, current_prices: Dict[str, float],
                                 portfolio: Any) -> List[Dict[str, Any]]:
        """
        Manage existing option positions based on assignment risk and profit targets.

        Args:
            current_prices: Current stock prices
            portfolio: Portfolio object

        Returns:
            List of management signals
        """
        management_signals = []

        if not portfolio:
            return management_signals

        try:
            # Get current option positions from portfolio
            option_positions = getattr(portfolio, 'option_positions', {})

            for symbol, position in option_positions.items():
                # Analyze position for management opportunities
                underlying = position.get('underlying_symbol')
                current_stock_price = current_prices.get(underlying, 0)

                if current_stock_price == 0:
                    continue

                # Check if position needs management
                management_signal = self._evaluate_position_management(
                    position, current_stock_price
                )

                if management_signal:
                    management_signals.append(management_signal)

        except Exception as e:
            self.logger.error(f"Error managing existing positions: {e}")

        return management_signals

    def _evaluate_position_management(self, position: Dict[str, Any],
                                    current_stock_price: float) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether a position needs management action.

        Args:
            position: Option position data
            current_stock_price: Current price of underlying stock

        Returns:
            Management signal if action is needed, None otherwise
        """
        try:
            option_type = position.get('type', '').lower()
            strike_price = position.get('strike_price', 0)
            expiration_date = position.get('expiration_date')
            entry_price = position.get('entry_price', 0)
            quantity = position.get('quantity', 0)

            # Calculate days to expiration
            if isinstance(expiration_date, str):
                exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            else:
                exp_date = expiration_date

            days_to_exp = (exp_date - datetime.now()).days

            # Calculate current profit/loss
            current_option_price = self._estimate_option_price(position, current_stock_price)
            profit_loss = (entry_price - current_option_price) / entry_price if entry_price > 0 else 0

            # MANAGEMENT RULE 1: Profit Target
            if profit_loss >= self.parameters['profit_target']:
                return {
                    'symbol': position.get('symbol'),
                    'action': 'BUY_TO_CLOSE',
                    'quantity': quantity,
                    'price': current_option_price,
                    'reason': f'Profit target reached: {profit_loss:.1%} profit',
                    'priority': 'HIGH',
                    'strategy_stage': 'profit_taking'
                }

            # MANAGEMENT RULE 2: Assignment Risk Management
            if days_to_exp <= self.parameters['management_dte']:
                assignment_risk = self._calculate_assignment_probability(
                    option_type, strike_price, current_stock_price, days_to_exp
                )

                if assignment_risk > self.parameters['assignment_tolerance']:
                    return {
                        'symbol': position.get('symbol'),
                        'action': 'BUY_TO_CLOSE',
                        'quantity': quantity,
                        'price': current_option_price,
                        'reason': f'High assignment risk: {assignment_risk:.1%} with {days_to_exp} DTE',
                        'priority': 'HIGH',
                        'strategy_stage': 'risk_management'
                    }

            # MANAGEMENT RULE 3: Rolling Opportunity
            if days_to_exp <= 7 and 0.3 < assignment_risk < 0.7:
                return {
                    'symbol': position.get('symbol'),
                    'action': 'ROLL',
                    'quantity': quantity,
                    'reason': f'Rolling opportunity - {assignment_risk:.1%} assignment risk',
                    'priority': 'MEDIUM',
                    'strategy_stage': 'rolling',
                    'roll_dte': self.parameters['preferred_dte']
                }

        except Exception as e:
            self.logger.error(f"Error evaluating position management: {e}")

        return None

    def _find_cash_secured_put_opportunities(self, current_prices: Dict[str, float],
                                           current_data: Dict[str, Any],
                                           portfolio: Any) -> List[Dict[str, Any]]:
        """
        Find opportunities to sell cash-secured puts.

        Cash-secured puts are sold on stocks we're willing to own at the strike price.
        This is typically done when we don't currently own the stock.

        Args:
            current_prices: Current stock prices
            current_data: Current market data
            portfolio: Portfolio object

        Returns:
            List of cash-secured put signals
        """
        put_signals = []

        try:
            # Filter stocks suitable for cash-secured puts
            suitable_stocks = self._filter_stocks_for_puts(current_prices, portfolio)

            for symbol in suitable_stocks:
                current_price = current_prices.get(symbol, 0)
                if current_price == 0:
                    continue

                # Find suitable put options for this stock
                put_options = self._get_put_options_for_symbol(symbol, current_price)

                # Score and select best put option
                if put_options:
                    scored_puts = self.score_options(put_options)
                    best_puts = self.select_best_options(
                        scored_puts,
                        limit=self.parameters['max_options_per_underlying']
                    )

                    for put_option in best_puts:
                        # Calculate position size
                        position_size = self._calculate_put_position_size(
                            put_option, portfolio
                        )

                        if position_size > 0:
                            put_signals.append({
                                'symbol': put_option['symbol'],
                                'underlying_symbol': symbol,
                                'action': 'SELL_TO_OPEN',
                                'option_type': 'put',
                                'quantity': position_size,
                                'strike_price': put_option['strike_price'],
                                'expiration_date': put_option['expiration_date'],
                                'premium': put_option['bid'],
                                'delta': put_option['delta'],
                                'yield': self.calculate_option_yield(put_option),
                                'score': put_option['score'],
                                'reason': f'Cash-secured put - {put_option["yield"]:.1f}% yield, score: {put_option["score"]:.3f}',
                                'strategy_stage': 'cash_secured_put',
                                'confidence': min(0.95, put_option['score'] * 10)
                            })

        except Exception as e:
            self.logger.error(f"Error finding cash-secured put opportunities: {e}")

        return put_signals

    def _find_covered_call_opportunities(self, current_prices: Dict[str, float],
                                       current_data: Dict[str, Any],
                                       portfolio: Any) -> List[Dict[str, Any]]:
        """
        Find opportunities to sell covered calls on owned stocks.

        Covered calls are sold on stocks we already own to generate additional income.
        We're willing to sell the stock at the strike price.

        Args:
            current_prices: Current stock prices
            current_data: Current market data
            portfolio: Portfolio object

        Returns:
            List of covered call signals
        """
        call_signals = []

        try:
            if not portfolio:
                return call_signals

            # Get stocks we currently own
            stock_positions = getattr(portfolio, 'positions', {})

            for symbol, position in stock_positions.items():
                quantity = position.get('quantity', 0)
                if quantity < 100:  # Need at least 100 shares for covered call
                    continue

                current_price = current_prices.get(symbol, 0)
                if current_price == 0:
                    continue

                # Find suitable call options for this stock
                call_options = self._get_call_options_for_symbol(symbol, current_price)

                # Score and select best call option
                if call_options:
                    scored_calls = self.score_options(call_options)
                    best_calls = self.select_best_options(
                        scored_calls,
                        limit=self.parameters['max_options_per_underlying']
                    )

                    for call_option in best_calls:
                        # Calculate how many contracts we can sell (100 shares per contract)
                        max_contracts = quantity // 100

                        if max_contracts > 0:
                            call_signals.append({
                                'symbol': call_option['symbol'],
                                'underlying_symbol': symbol,
                                'action': 'SELL_TO_OPEN',
                                'option_type': 'call',
                                'quantity': min(max_contracts, 1),  # Start with 1 contract
                                'strike_price': call_option['strike_price'],
                                'expiration_date': call_option['expiration_date'],
                                'premium': call_option['bid'],
                                'delta': call_option['delta'],
                                'yield': self.calculate_option_yield(call_option),
                                'score': call_option['score'],
                                'reason': f'Covered call - {call_option["yield"]:.1f}% yield, score: {call_option["score"]:.3f}',
                                'strategy_stage': 'covered_call',
                                'confidence': min(0.95, call_option['score'] * 10)
                            })

        except Exception as e:
            self.logger.error(f"Error finding covered call opportunities: {e}")

        return call_signals

    def _filter_stocks_for_puts(self, current_prices: Dict[str, float],
                               portfolio: Any) -> List[str]:
        """
        Filter stocks suitable for selling cash-secured puts.

        Args:
            current_prices: Current stock prices
            portfolio: Portfolio object

        Returns:
            List of suitable stock symbols
        """
        suitable_stocks = []

        try:
            for symbol in self.symbol_list:
                current_price = current_prices.get(symbol, 0)

                # Basic price filters
                if not (self.parameters['min_stock_price'] <=
                       current_price <=
                       self.parameters['max_stock_price']):
                    continue

                # Check if we already have a position (covered call candidates)
                if portfolio and hasattr(portfolio, 'positions'):
                    position = portfolio.positions.get(symbol, {})
                    if position.get('quantity', 0) > 0:
                        continue  # Skip if we already own the stock

                # Check if we already have an option position on this symbol
                if portfolio and hasattr(portfolio, 'option_positions'):
                    existing_options = [
                        pos for pos_symbol, pos in portfolio.option_positions.items()
                        if pos.get('underlying_symbol') == symbol and pos.get('type') == 'put'
                    ]
                    if existing_options:
                        continue  # Skip if we already have put options

                suitable_stocks.append(symbol)

        except Exception as e:
            self.logger.error(f"Error filtering stocks for puts: {e}")

        return suitable_stocks

    def _get_put_options_for_symbol(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """
        Get suitable put options for a given symbol from the live options chain.

        Args:
            symbol: Stock symbol
            current_price: Current stock price

        Returns:
            List of suitable put options
        """
        put_options = []

        try:
            from src.data import AlpacaDataProvider

            provider = AlpacaDataProvider()
            chain = provider.get_options_chain(symbol)
            if chain is None or chain.empty:
                self.logger.warning(f"Options chain unavailable for {symbol}.")
                return []

            chain = chain.copy()
            chain['expiration_date'] = pd.to_datetime(chain['expiration_date'], errors='coerce')
            base_date = datetime.now()

            for _, row in chain.iterrows():
                option_type = str(row.get('option_type', '')).upper()
                if option_type != 'P':
                    continue

                exp_date = row.get('expiration_date')
                if pd.isna(exp_date):
                    continue

                strike = float(row.get('strike_price') or 0)
                if strike < self.parameters['min_stock_price']:
                    continue

                dte = (exp_date - base_date).days
                if not (self.parameters['dte_min'] <= dte <= self.parameters['dte_max']):
                    continue

                delta = row.get('delta')
                if delta is not None:
                    if not (self.parameters['put_delta_min'] <= abs(delta) <= self.parameters['put_delta_max']):
                        continue

                volume = row.get('volume')
                open_interest = row.get('open_interest')

                if volume is not None and volume < self.parameters['min_daily_volume']:
                    continue
                if open_interest is not None and open_interest < self.parameters['min_open_interest']:
                    continue

                put_options.append({
                    'symbol': row.get('symbol'),
                    'underlying_symbol': symbol,
                    'type': 'put',
                    'strike_price': strike,
                    'expiration_date': exp_date,
                    'days_to_expiration': dte,
                    'bid': row.get('bid_price'),
                    'ask': row.get('ask_price'),
                    'delta': delta,
                    'open_interest': open_interest,
                    'volume': volume
                })

        except Exception as e:
            self.logger.error(f"Error getting put options for {symbol}: {e}")

        return put_options

    def _get_call_options_for_symbol(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """
        Get suitable call options for a given symbol from the live options chain.

        Args:
            symbol: Stock symbol
            current_price: Current stock price

        Returns:
            List of suitable call options
        """
        call_options = []

        try:
            from src.data import AlpacaDataProvider

            provider = AlpacaDataProvider()
            chain = provider.get_options_chain(symbol)
            if chain is None or chain.empty:
                self.logger.warning(f"Options chain unavailable for {symbol}.")
                return []

            chain = chain.copy()
            chain['expiration_date'] = pd.to_datetime(chain['expiration_date'], errors='coerce')
            base_date = datetime.now()

            for _, row in chain.iterrows():
                option_type = str(row.get('option_type', '')).upper()
                if option_type != 'C':
                    continue

                exp_date = row.get('expiration_date')
                if pd.isna(exp_date):
                    continue

                strike = float(row.get('strike_price') or 0)
                dte = (exp_date - base_date).days
                if not (self.parameters['dte_min'] <= dte <= self.parameters['dte_max']):
                    continue

                delta = row.get('delta')
                if delta is not None:
                    if not (self.parameters['call_delta_min'] <= delta <= self.parameters['call_delta_max']):
                        continue

                volume = row.get('volume')
                open_interest = row.get('open_interest')

                if volume is not None and volume < self.parameters['min_daily_volume']:
                    continue
                if open_interest is not None and open_interest < self.parameters['min_open_interest']:
                    continue

                call_options.append({
                    'symbol': row.get('symbol'),
                    'underlying_symbol': symbol,
                    'type': 'call',
                    'strike_price': strike,
                    'expiration_date': exp_date,
                    'days_to_expiration': dte,
                    'bid': row.get('bid_price'),
                    'ask': row.get('ask_price'),
                    'delta': delta,
                    'open_interest': open_interest,
                    'volume': volume
                })

        except Exception as e:
            self.logger.error(f"Error getting call options for {symbol}: {e}")

        return call_options

    def _calculate_assignment_probability(self, option_type: str, strike_price: float,
                                        current_price: float, days_to_exp: int) -> float:
        """
        Calculate assignment probability for an option.

        Args:
            option_type: 'put' or 'call'
            strike_price: Option strike price
            current_price: Current stock price
            days_to_exp: Days to expiration

        Returns:
            Assignment probability (0.0 to 1.0)
        """
        try:
            if option_type.lower() == 'put':
                # Put is ITM when stock price < strike price
                moneyness = current_price / strike_price
                if moneyness < 1.0:  # ITM
                    base_prob = 0.6 + (1.0 - moneyness) * 0.3
                else:  # OTM
                    base_prob = max(0.1, (1.0 - moneyness) * 0.5)
            else:  # call
                # Call is ITM when stock price > strike price
                moneyness = current_price / strike_price
                if moneyness > 1.0:  # ITM
                    base_prob = 0.6 + (moneyness - 1.0) * 0.3
                else:  # OTM
                    base_prob = max(0.1, (moneyness - 1.0) * 0.5)

            # Adjust for time to expiration
            time_factor = max(0.1, min(1.0, days_to_exp / 21.0))
            assignment_prob = base_prob * time_factor

            return max(0.0, min(1.0, assignment_prob))

        except Exception as e:
            self.logger.error(f"Error calculating assignment probability: {e}")
            return 0.5  # Default to 50% if calculation fails

    def _estimate_option_price(self, position: Dict[str, Any], current_stock_price: float) -> float:
        """
        Estimate current option price based on stock price movement.

        This is a simplified estimation. In production, use real-time option prices.

        Args:
            position: Option position data
            current_stock_price: Current stock price

        Returns:
            Estimated option price
        """
        try:
            option_type = position.get('type', '').lower()
            strike_price = position.get('strike_price', 0)
            delta = position.get('delta', 0)
            original_stock_price = position.get('underlying_price', current_stock_price)

            # Calculate intrinsic value
            if option_type == 'put':
                intrinsic_value = max(0, strike_price - current_stock_price)
            else:  # call
                intrinsic_value = max(0, current_stock_price - strike_price)

            # Estimate price change based on delta
            stock_price_change = current_stock_price - original_stock_price
            option_price_change = delta * stock_price_change

            # Simplified time decay (theta effect)
            days_passed = 1  # Assume 1 day has passed
            time_decay = position.get('entry_price', 0) * 0.02 * (days_passed / 365.0)

            estimated_price = max(intrinsic_value,
                                position.get('entry_price', 0) + option_price_change - time_decay)

            return round(estimated_price, 2)

        except Exception as e:
            self.logger.error(f"Error estimating option price: {e}")
            return position.get('entry_price', 0)

    def _calculate_put_position_size(self, put_option: Dict[str, Any], portfolio: Any) -> int:
        """
        Calculate position size for a cash-secured put.

        Args:
            put_option: Put option data
            portfolio: Portfolio object

        Returns:
            Number of contracts to sell
        """
        try:
            if not portfolio:
                return 0

            portfolio_value = portfolio.get_portfolio_value()
            available_cash = portfolio.get_available_cash()
            strike_price = put_option['strike_price']

            # Cash required per contract (100 shares)
            cash_required_per_contract = strike_price * 100

            # Maximum contracts based on available cash
            max_contracts_by_cash = int(available_cash / cash_required_per_contract)

            # Maximum contracts based on position size limit
            max_position_value = portfolio_value * self.parameters['position_size_pct']
            max_contracts_by_size = int(max_position_value / cash_required_per_contract)

            # Take the minimum
            max_contracts = min(max_contracts_by_cash, max_contracts_by_size)

            # Ensure at least 1 contract if we have sufficient cash
            if max_contracts >= 1:
                return min(max_contracts, 3)  # Limit to 3 contracts initially
            else:
                return 0

        except Exception as e:
            self.logger.error(f"Error calculating put position size: {e}")
            return 0

    def _apply_risk_management(self, signals: List[Dict[str, Any]],
                             portfolio: Any) -> List[Dict[str, Any]]:
        """
        Apply risk management rules to filter and limit signals.

        Args:
            signals: List of trading signals
            portfolio: Portfolio object

        Returns:
            Filtered list of signals
        """
        filtered_signals = []

        try:
            # Risk management checks
            total_risk_exposure = 0
            position_count = 0

            for signal in signals:
                # Skip management signals (they don't add new risk)
                if signal.get('action') in ['BUY_TO_CLOSE', 'ROLL']:
                    filtered_signals.append(signal)
                    continue

                # Calculate risk exposure for new positions
                if signal.get('action') == 'SELL_TO_OPEN':
                    strike_price = signal.get('strike_price', 0)
                    quantity = signal.get('quantity', 0)
                    risk_per_position = strike_price * quantity * 100  # 100 shares per contract

                    # Check maximum risk limit
                    if total_risk_exposure + risk_per_position > self.parameters['max_risk']:
                        self.logger.warning(f"Skipping {signal['symbol']} - would exceed max risk")
                        continue

                    # Check maximum position count
                    if position_count >= self.parameters['max_positions']:
                        self.logger.warning(f"Skipping {signal['symbol']} - max positions reached")
                        continue

                    total_risk_exposure += risk_per_position
                    position_count += 1

                # Apply minimum score filter
                if signal.get('score', 0) < self.parameters['min_score']:
                    self.logger.debug(f"Skipping {signal['symbol']} - score too low")
                    continue

                filtered_signals.append(signal)

            self.logger.info(f"Risk management: {len(filtered_signals)}/{len(signals)} signals passed")

        except Exception as e:
            self.logger.error(f"Error applying risk management: {e}")
            filtered_signals = signals  # Return original signals if error

        return filtered_signals

    # Implementation of abstract methods from BaseOptionStrategy

    def filter_underlying_stocks(self, client: Any) -> List[str]:
        """Filter underlying stocks based on wheel strategy criteria."""
        return self.symbol_list

    def filter_options(self, client: Any, underlying: str,
                      option_type: str = 'put') -> List[Dict[str, Any]]:
        """Filter options based on wheel strategy criteria."""
        # This would be implemented with real Alpaca API calls
        return []

    def score_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score options using wheel strategy scoring formula."""
        scored_options = []

        for option in options:
            try:
                # Calculate yield
                option['yield'] = self.calculate_option_yield(option)

                # Calculate score using wheel formula
                option['score'] = self.calculate_option_score(option)

                # Apply yield filters
                if (self.parameters['min_yield'] <= option['yield'] / 100 <= self.parameters['max_yield']):
                    scored_options.append(option)

            except Exception as e:
                self.logger.error(f"Error scoring option {option.get('symbol', 'unknown')}: {e}")

        return scored_options

    def select_best_options(self, scored_options: List[Dict[str, Any]],
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Select best options based on score."""
        try:
            # Filter by minimum score
            qualified_options = [
                opt for opt in scored_options
                if opt.get('score', 0) >= self.parameters['min_score']
            ]

            # Sort by score (descending)
            qualified_options.sort(key=lambda x: x.get('score', 0), reverse=True)

            # Apply limit
            if limit:
                qualified_options = qualified_options[:limit]

            return qualified_options

        except Exception as e:
            self.logger.error(f"Error selecting best options: {e}")
            return []

    def calculate_position_size(self, symbol: str, price: float,
                              portfolio_value: float, volatility: float = None) -> int:
        """Calculate position size for wheel strategy."""
        try:
            # For options, position size is calculated differently
            # This method is primarily for stock positions
            max_position_value = portfolio_value * self.parameters['position_size_pct']
            position_size = int(max_position_value / price)

            # Ensure we can buy in increments of 100 (for covered calls)
            position_size = (position_size // 100) * 100

            return max(0, position_size)

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive wheel strategy information."""
        base_info = super().get_strategy_info()

        wheel_info = {
            'type': 'Options Trading',
            'timeframe': 'Medium-term (weeks to months)',
            'risk_level': 'Medium',
            'expected_trades_per_day': '1-3',
            'holding_period': '2-8 weeks per cycle',
            'description': 'Systematic wheel strategy selling cash-secured puts and covered calls',
            'strategy_stages': ['cash_secured_put', 'assignment', 'covered_call', 'call_away'],
            'current_cycle_stages': self.current_cycle_stage,
            'assignment_history': len(self.assignment_history),
            'total_wheel_cycles': len([p for p in self.profit_history if p.get('cycle_complete', False)])
        }

        return {**base_info, **wheel_info}
