"""Options-specific trading engine with multi-leg support and expiration handling."""
from __future__ import annotations

import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce

from .trading_engine import TradingEngine


class TradingOptionEngine(TradingEngine):
    """Trading engine for options with Alpaca-specific rules.

    Options trading on Alpaca:
    - No fractional contracts (whole numbers only)
    - OCC symbol format: AAPL240119C00150000
    - Market hours: 9:30 AM - 4:00 PM ET
    - Expiration dates must be valid

    See: https://docs.alpaca.markets/docs/options-trading
    """

    def __init__(self, paper_trading: bool = True) -> None:
        super().__init__(paper_trading)

    def validate_order(self, symbol: str, qty: float, side: str) -> None:
        """Validate option order - whole contracts, valid expiration."""
        super().validate_order(symbol, qty, side)

        if qty != int(qty):
            raise ValueError(f"Option contracts must be whole numbers, got {qty}")

        days_to_exp = self.check_expiration(symbol)
        if days_to_exp is not None:
            if days_to_exp < 0:
                raise ValueError(f"Option {symbol} has expired")
            if days_to_exp == 0:
                raise ValueError(f"Option {symbol} expires today - exercise caution")

    def parse_option_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Parse OCC option symbol format.

        Format: UNDERLYING(1-6) + DATE(6) + TYPE(1) + STRIKE(8)
        Example: AAPL240119C00150000
                 -> AAPL, 2024-01-19, Call, $150.00
        """
        symbol = symbol.strip().upper()
        pattern = r'^([A-Z]{1,6})(\d{6})([CP])(\d{8})$'
        match = re.match(pattern, symbol)

        if not match:
            return None

        underlying, date_str, option_type, strike_str = match.groups()

        try:
            exp_date = datetime.strptime(date_str, '%y%m%d').date()
        except ValueError:
            return None

        return {
            'underlying': underlying,
            'expiration': exp_date,
            'type': 'call' if option_type == 'C' else 'put',
            'strike': int(strike_str) / 1000,
            'raw_symbol': symbol
        }

    def build_option_symbol(self, underlying: str, expiration: date,
                           option_type: str, strike: float) -> str:
        """Build OCC option symbol from components.

        Args:
            underlying: Stock ticker (e.g., 'AAPL')
            expiration: Expiration date
            option_type: 'call' or 'put'
            strike: Strike price (e.g., 150.00)
        """
        underlying = underlying.upper().ljust(6)[:6]
        date_str = expiration.strftime('%y%m%d')
        type_char = 'C' if option_type.lower() == 'call' else 'P'
        strike_int = int(strike * 1000)
        strike_str = f"{strike_int:08d}"

        return f"{underlying.strip()}{date_str}{type_char}{strike_str}"

    def check_expiration(self, symbol: str) -> Optional[int]:
        """Check days until expiration for an option symbol.

        Returns:
            Days to expiration, or None if symbol cannot be parsed
        """
        parsed = self.parse_option_symbol(symbol)
        if not parsed:
            return None

        exp_date = parsed['expiration']
        today = date.today()
        return (exp_date - today).days

    def place_market_order(self, symbol: str, qty: float, side: str = 'buy',
                          time_in_force: str = 'day') -> Dict[str, Any]:
        """Place a market order for options.

        Args:
            symbol: OCC option symbol (e.g., AAPL240119C00150000)
            qty: Number of contracts (whole numbers only)
            side: 'buy' or 'sell'
            time_in_force: 'day' (default) or 'gtc'
        """
        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, int(qty), side)

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=int(abs(qty)),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
        )
        order = self.api.submit_order(order_request)

        order_dict = self._build_order_dict(order)
        self.logger.info(f"Option market order placed: {side} {int(qty)} contracts of {symbol}")
        return order_dict

    def place_limit_order(self, symbol: str, qty: float, limit_price: float,
                         side: str = 'buy', time_in_force: str = 'day') -> Dict[str, Any]:
        """Place a limit order for options.

        Args:
            symbol: OCC option symbol
            qty: Number of contracts (whole numbers only)
            limit_price: Limit price per contract
            side: 'buy' or 'sell'
            time_in_force: 'day' (default) or 'gtc'
        """
        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, int(qty), side)

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=int(abs(qty)),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC,
            limit_price=limit_price
        )
        order = self.api.submit_order(order_request)

        order_dict = self._build_order_dict(order)
        self.logger.info(
            f"Option limit order placed: {side} {int(qty)} contracts of {symbol} at ${limit_price}"
        )
        return order_dict

    def buy_to_open(self, symbol: str, qty: int,
                    limit_price: float = None) -> Dict[str, Any]:
        """Buy to open a new option position."""
        if limit_price:
            return self.place_limit_order(symbol, qty, limit_price, side='buy')
        return self.place_market_order(symbol, qty, side='buy')

    def sell_to_close(self, symbol: str, qty: int = None,
                      limit_price: float = None) -> Dict[str, Any]:
        """Sell to close an existing long option position."""
        if qty is None:
            positions = self.get_current_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            if not position:
                raise ValueError(f"No position found for {symbol}")
            qty = int(abs(float(position['qty'])))

        if limit_price:
            return self.place_limit_order(symbol, qty, limit_price, side='sell')
        return self.place_market_order(symbol, qty, side='sell')

    def sell_to_open(self, symbol: str, qty: int,
                     limit_price: float = None) -> Dict[str, Any]:
        """Sell to open a new short option position (writing options)."""
        if limit_price:
            return self.place_limit_order(symbol, qty, limit_price, side='sell')
        return self.place_market_order(symbol, qty, side='sell')

    def buy_to_close(self, symbol: str, qty: int = None,
                     limit_price: float = None) -> Dict[str, Any]:
        """Buy to close an existing short option position."""
        if qty is None:
            positions = self.get_current_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            if not position:
                raise ValueError(f"No short position found for {symbol}")
            qty = int(abs(float(position['qty'])))

        if limit_price:
            return self.place_limit_order(symbol, qty, limit_price, side='buy')
        return self.place_market_order(symbol, qty, side='buy')

    def get_option_positions(self) -> List[Dict[str, Any]]:
        """Get only option positions with parsed symbol data."""
        positions = self.get_current_positions()
        option_positions = []

        for pos in positions:
            parsed = self.parse_option_symbol(pos['symbol'])
            if parsed:
                pos_with_details = {**pos, **parsed}
                pos_with_details['days_to_expiration'] = self.check_expiration(pos['symbol'])
                option_positions.append(pos_with_details)

        return option_positions

    def get_expiring_positions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get option positions expiring within specified days."""
        positions = self.get_option_positions()
        return [p for p in positions if p.get('days_to_expiration', 999) <= days]

    def roll_position(self, current_symbol: str, new_symbol: str,
                      qty: int = None, limit_price: float = None) -> Dict[str, Any]:
        """Roll an option position to a new expiration/strike.

        This closes the current position and opens a new one.
        """
        close_result = self.sell_to_close(current_symbol, qty, limit_price)
        open_result = self.buy_to_open(new_symbol, qty or int(close_result.get('qty', 1)))

        return {
            'closed': close_result,
            'opened': open_result,
            'rolled_from': current_symbol,
            'rolled_to': new_symbol
        }
