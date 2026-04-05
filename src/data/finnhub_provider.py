"""
Finnhub API provider for financial market data and news
"""

from datetime import datetime, timedelta
import logging
import os
from typing import Any, Dict, List

import finnhub


class FinnhubProviderError(RuntimeError):
    """Raised when Finnhub returns an error payload."""


class FinnhubProvider:
    """Finnhub API provider for market data and news"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError("Finnhub API key not provided")
        self.client = finnhub.Client(api_key=self.api_key)

    def _unwrap(self, payload: Any, action: str) -> Any:
        if isinstance(payload, dict) and "error" in payload:
            raise FinnhubProviderError(f"{action} failed: {payload['error']}")
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and "error" in first:
                raise FinnhubProviderError(f"{action} failed: {first['error']}")
        return payload

    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile information"""
        payload = self.client.company_profile2(symbol=symbol)
        return self._unwrap(payload, f"Fetch company profile for {symbol}")

    def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Get basic financial metrics"""
        payload = self.client.company_basic_financials(symbol, 'all')
        return self._unwrap(payload, f"Fetch basic financials for {symbol}")

    def get_earnings_calendar(
        self, symbol: str = None,
        from_date: str = None,
        to_date: str = None
    ) -> Dict[str, Any]:
        """Get earnings calendar"""
        payload = self.client.earnings_calendar(
            _from=from_date, to=to_date, symbol=symbol
        )
        return self._unwrap(payload, "Fetch earnings calendar")

    def get_company_news(
        self, symbol: str,
        from_date: str = None,
        to_date: str = None
    ) -> List[Dict[str, Any]]:
        """Get company news"""
        if not from_date:
            from_date = (
                datetime.now() - timedelta(days=7)
            ).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        payload = self.client.company_news(
            symbol, _from=from_date, to=to_date
        )
        return self._unwrap(payload, f"Fetch company news for {symbol}")

    def get_market_news(
        self, category: str = 'general'
    ) -> List[Dict[str, Any]]:
        """Get general market news"""
        payload = self.client.general_news(category, min_id=0)
        return self._unwrap(payload, f"Fetch market news for {category}")

    def get_recommendation_trends(
        self, symbol: str
    ) -> Dict[str, Any]:
        """Get analyst recommendation trends"""
        payload = self.client.recommendation_trends(symbol)
        return self._unwrap(payload, f"Fetch recommendation trends for {symbol}")

    def get_price_target(self, symbol: str) -> Dict[str, Any]:
        """Get analyst price targets"""
        payload = self.client.price_target(symbol)
        return self._unwrap(payload, f"Fetch price target for {symbol}")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote"""
        payload = self.client.quote(symbol)
        return self._unwrap(payload, f"Fetch quote for {symbol}")

    def get_stock_candles(
        self, symbol: str, resolution: str = 'D',
        from_timestamp: int = None,
        to_timestamp: int = None
    ) -> Dict[str, Any]:
        """Get stock price candles"""
        if not from_timestamp:
            from_timestamp = int(
                (datetime.now() - timedelta(days=30)).timestamp()
            )
        if not to_timestamp:
            to_timestamp = int(datetime.now().timestamp())
        payload = self.client.stock_candles(
            symbol, resolution, from_timestamp, to_timestamp
        )
        return self._unwrap(payload, f"Fetch stock candles for {symbol}")

    def get_earnings_surprises(
        self, symbol: str, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """Get earnings surprises"""
        payload = self.client.company_earnings(symbol, limit)
        return self._unwrap(payload, f"Fetch earnings surprises for {symbol}")

    def get_insider_transactions(
        self, symbol: str
    ) -> List[Dict[str, Any]]:
        """Get insider transactions"""
        payload = self.client.stock_insider_transactions(symbol)
        return self._unwrap(payload, f"Fetch insider transactions for {symbol}")

    def get_insider_sentiment(
        self, symbol: str,
        from_date: str = None,
        to_date: str = None
    ) -> Dict[str, Any]:
        """Get insider sentiment"""
        if not from_date:
            from_date = (
                datetime.now() - timedelta(days=90)
            ).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        payload = self.client.stock_insider_sentiment(
            symbol, from_date, to_date
        )
        return self._unwrap(payload, f"Fetch insider sentiment for {symbol}")
