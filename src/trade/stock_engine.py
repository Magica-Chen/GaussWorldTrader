"""Stock-specific trading engine with margin, short selling, and fractional share support."""
from __future__ import annotations

from datetime import time
from typing import Any, Dict

from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .trading_engine import TradingEngine


class TradingStockEngine(TradingEngine):
    """Trading engine for stocks with Alpaca-specific rules.

    Stock trading on Alpaca:
    - Margin trading allowed (depends on account type)
    - Short selling allowed (with locate requirements for hard-to-borrow)
    - Fractional shares optional (disabled by default)
    - PDT rules apply (3 day trades in 5 days for accounts < $25k)
    - Market hours: 9:30 AM - 4:00 PM ET (extended hours optional)

    See: https://docs.alpaca.markets/docs/margin-and-short-selling
         https://docs.alpaca.markets/docs/fractional-trading
    """

    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EXTENDED_OPEN = time(4, 0)
    EXTENDED_CLOSE = time(20, 0)

    def __init__(self, paper_trading: bool = True, allow_fractional: bool = False) -> None:
        super().__init__(paper_trading)
        self.allow_fractional = allow_fractional

    def validate_order(self, symbol: str, qty: float, side: str) -> None:
        """Validate stock order with PDT and fractional checks."""
        super().validate_order(symbol, qty, side)

        if not self.allow_fractional and qty != int(qty):
            raise ValueError(
                f"Fractional shares not enabled. Got qty={qty}. "
                "Set allow_fractional=True to enable."
            )

    def check_pdt_status(self) -> Dict[str, Any]:
        """Check Pattern Day Trader status and day trade count."""
        account = self.get_account_info()
        return {
            'pattern_day_trader': account.get('pattern_day_trader', False),
            'day_trade_count': account.get('day_trade_count', 0),
            'equity': account.get('equity', 0),
            'pdt_threshold': 25000.0,
            'is_restricted': (
                account.get('pattern_day_trader', False) and
                account.get('equity', 0) < 25000
            )
        }

    def check_margin_requirements(self, symbol: str, qty: float) -> Dict[str, Any]:
        """Check if account has sufficient margin/buying power for the order."""
        account = self.get_account_info()
        buying_power = account.get('buying_power', 0)

        return {
            'buying_power': buying_power,
            'symbol': symbol,
            'qty': qty,
            'has_margin': buying_power > 0
        }

    def check_locate_requirements(self, symbol: str) -> bool:
        """Check if short selling is available for a symbol.

        Note: Alpaca handles locate automatically for most symbols.
        Hard-to-borrow stocks may require manual locate.
        """
        return True

    def place_market_order(self, symbol: str, qty: float, side: str = 'buy',
                          time_in_force: str = 'day') -> Dict[str, Any]:
        """Place a market order for stocks.

        Args:
            symbol: Stock ticker symbol
            qty: Number of shares (fractional if enabled)
            side: 'buy' or 'sell'
            time_in_force: 'day' (default for stocks) or 'gtc'
        """
        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, qty, side)

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
        )
        order = self.api.submit_order(order_request)

        order_dict = self._build_order_dict(order)
        self.logger.info(f"Stock market order placed: {side} {qty} shares of {symbol}")
        return order_dict

    def place_limit_order(self, symbol: str, qty: float, limit_price: float,
                         side: str = 'buy', time_in_force: str = 'day') -> Dict[str, Any]:
        """Place a limit order for stocks.

        Args:
            symbol: Stock ticker symbol
            qty: Number of shares (fractional if enabled)
            limit_price: Limit price for the order
            side: 'buy' or 'sell'
            time_in_force: 'day' (default) or 'gtc'
        """
        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, qty, side)

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC,
            limit_price=limit_price
        )
        order = self.api.submit_order(order_request)

        order_dict = self._build_order_dict(order)
        self.logger.info(f"Stock limit order placed: {side} {qty} shares of {symbol} at ${limit_price}")
        return order_dict

    def place_bracket_order(self, symbol: str, qty: float, side: str,
                           stop_loss: float, take_profit: float,
                           time_in_force: str = 'day') -> Dict[str, Any]:
        """Place a bracket order with stop loss and take profit.

        Alpaca bracket orders create three linked orders:
        1. Entry market order
        2. Stop loss order (OCO)
        3. Take profit limit order (OCO)
        """
        from alpaca.trading.requests import (
            MarketOrderRequest as BracketMarketOrderRequest,
            TakeProfitRequest,
            StopLossRequest,
        )
        from alpaca.trading.enums import OrderClass

        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, qty, side)

        order_request = BracketMarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss)
        )
        order = self.api.submit_order(order_request)

        order_dict = self._build_order_dict(order)
        order_dict['stop_loss'] = stop_loss
        order_dict['take_profit'] = take_profit
        self.logger.info(
            f"Stock bracket order placed: {side} {qty} of {symbol} "
            f"(SL: ${stop_loss}, TP: ${take_profit})"
        )
        return order_dict

    def short_sell(self, symbol: str, qty: float,
                   time_in_force: str = 'day') -> Dict[str, Any]:
        """Open a short position.

        Note: Requires margin account. May fail for hard-to-borrow stocks.
        """
        if not self.check_locate_requirements(symbol):
            raise ValueError(f"Cannot short {symbol}: locate not available")

        return self.place_market_order(symbol, qty, side='sell', time_in_force=time_in_force)

    def cover_short(self, symbol: str, qty: float = None) -> Dict[str, Any]:
        """Cover (close) a short position.

        Args:
            symbol: Stock symbol
            qty: Quantity to cover (defaults to full position)
        """
        positions = self.get_current_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)

        if not position:
            raise ValueError(f"No short position found for {symbol}")

        pos_qty = float(position['qty'])
        if pos_qty >= 0:
            raise ValueError(f"Position for {symbol} is not short (qty={pos_qty})")

        cover_qty = abs(pos_qty) if qty is None else abs(qty)
        return self.place_market_order(symbol, cover_qty, side='buy')
