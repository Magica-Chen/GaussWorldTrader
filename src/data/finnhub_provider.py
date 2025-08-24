"""
Finnhub API provider for financial market data and news
"""

import os
import finnhub
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging


class FinnhubProvider:
    """Finnhub API provider for market data and news"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("Finnhub API key not provided")
            self.client = None
        else:
            self.client = finnhub.Client(api_key=self.api_key)
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile information"""
        if not self.client:
            return {"error": "API key not configured"}
        
        try:
            return self.client.company_profile2(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error fetching company profile for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Get basic financial metrics"""
        if not self.client:
            return {"error": "API key not configured"}
        
        try:
            return self.client.company_basic_financials(symbol, 'all')
        except Exception as e:
            self.logger.error(f"Error fetching financials for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_earnings_calendar(self, symbol: str = None, 
                            from_date: str = None, 
                            to_date: str = None) -> Dict[str, Any]:
        """Get earnings calendar"""
        if not self.client:
            return {"error": "API key not configured"}
        
        try:
            return self.client.earnings_calendar(_from=from_date, to=to_date, symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error fetching earnings calendar: {e}")
            return {"error": str(e)}
    
    def get_company_news(self, symbol: str, 
                        from_date: str = None, 
                        to_date: str = None) -> List[Dict[str, Any]]:
        """Get company news"""
        if not self.client:
            return [{"error": "API key not configured"}]
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            return self.client.company_news(symbol, _from=from_date, to=to_date)
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return [{"error": str(e)}]
    
    def get_market_news(self, category: str = 'general') -> List[Dict[str, Any]]:
        """Get general market news"""
        if not self.client:
            return [{"error": "API key not configured"}]
        
        try:
            return self.client.general_news(category, min_id=0)
        except Exception as e:
            self.logger.error(f"Error fetching market news: {e}")
            return [{"error": str(e)}]
    
    def get_recommendation_trends(self, symbol: str) -> Dict[str, Any]:
        """Get analyst recommendation trends"""
        if not self.client:
            return {"error": "API key not configured"}
        
        try:
            return self.client.recommendation_trends(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching recommendations for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_price_target(self, symbol: str) -> Dict[str, Any]:
        """Get analyst price targets"""
        if not self.client:
            return {"error": "API key not configured"}
        
        try:
            return self.client.price_target(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching price target for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote"""
        if not self.client:
            return {"error": "API key not configured"}
        
        try:
            return self.client.quote(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_stock_candles(self, symbol: str, resolution: str = 'D', 
                         from_timestamp: int = None, 
                         to_timestamp: int = None) -> Dict[str, Any]:
        """Get stock price candles"""
        if not self.client:
            return {"error": "API key not configured"}
        
        if not from_timestamp:
            from_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())
        if not to_timestamp:
            to_timestamp = int(datetime.now().timestamp())
        
        try:
            return self.client.stock_candles(symbol, resolution, from_timestamp, 
                                           to_timestamp)
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_earnings_surprises(self, symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
        """Get earnings surprises"""
        if not self.client:
            return [{"error": "API key not configured"}]
        
        try:
            return self.client.company_earnings(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching earnings for {symbol}: {e}")
            return [{"error": str(e)}]
    
    def get_insider_transactions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get insider transactions"""
        if not self.client:
            return [{"error": "API key not configured"}]
        
        try:
            return self.client.stock_insider_transactions(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching insider transactions for {symbol}: {e}")
            return [{"error": str(e)}]
    
    def get_insider_sentiment(self, symbol: str, 
                             from_date: str = None, 
                             to_date: str = None) -> Dict[str, Any]:
        """Get insider sentiment"""
        if not self.client:
            return {"error": "API key not configured"}
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            return self.client.stock_insider_sentiment(symbol, from_date, to_date)
        except Exception as e:
            self.logger.error(f"Error fetching insider sentiment for {symbol}: {e}")
            return {"error": str(e)}