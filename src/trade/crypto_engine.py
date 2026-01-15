"""Crypto-specific trading engine. NO margin, NO short selling, 24/7 trading."""
from __future__ import annotations

from typing import Any, Dict

from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .trading_engine import TradingEngine


class TradingCryptoEngine(TradingEngine):
    """Trading engine for cryptocurrency with Alpaca-specific rules.

    Crypto trading on Alpaca:
    - No margin trading (must have cash)
    - No short selling (long only)
    - Fractional quantities always supported
    - 24/7 trading availability
    - Symbol format: BTC/USD or BTCUSD (normalized internally)
    """

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol to Alpaca format (BTC/USD)."""
        symbol = symbol.strip().upper()
        if "/" in symbol:
            return symbol
        if symbol.endswith("USD") and len(symbol) > 3:
            return f"{symbol[:-3]}/USD"
        return symbol

    def validate_order(self, symbol: str, qty: float, side: str) -> None:
        """Validate crypto order - no shorts allowed."""
        super().validate_order(symbol, qty, side)

        if side.lower() == 'sell' and not self._has_position(symbol):
            raise ValueError("Crypto does not support short selling")

        if side.lower() == 'buy':
            account = self.get_account_info()
            cash = account.get('cash', 0)
            if cash <= 0:
                raise ValueError("Insufficient cash for crypto purchase (no margin available)")

    def place_market_order(self, symbol: str, qty: float, side: str = 'buy',
                          time_in_force: str = 'gtc') -> Dict[str, Any]:
        """Place a market order for crypto."""
        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, qty, side)

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if time_in_force == 'gtc' else TimeInForce.DAY
        )
        order = self.api.submit_order(order_request)

        order_dict = self._build_order_dict(order)
        self.logger.info(f"Crypto market order placed: {side} {qty} of {symbol}")
        return order_dict

    def place_limit_order(self, symbol: str, qty: float, limit_price: float,
                         side: str = 'buy', time_in_force: str = 'gtc') -> Dict[str, Any]:
        """Place a limit order for crypto."""
        symbol = self.normalize_symbol(symbol)
        self.validate_order(symbol, qty, side)

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if time_in_force == 'gtc' else TimeInForce.DAY,
            limit_price=limit_price
        )
        order = self.api.submit_order(order_request)

        order_dict = self._build_order_dict(order)
        self.logger.info(f"Crypto limit order placed: {side} {qty} of {symbol} at ${limit_price}")
        return order_dict

    def get_crypto_positions(self) -> list[Dict[str, Any]]:
        """Get only crypto positions (filter by symbol format)."""
        positions = self.get_current_positions()
        return [p for p in positions if '/' in p['symbol'] or p['symbol'].endswith('USD')]
