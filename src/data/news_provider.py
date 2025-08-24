import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .finnhub_provider import FinnhubProvider


class NewsDataProvider:
    def __init__(self, api_key: str = None):
        self.finnhub = FinnhubProvider(api_key)
        self.logger = logging.getLogger(__name__)
    
    def get_company_news(self, symbol: str, from_date: Optional[datetime] = None, 
                        to_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        result = self.finnhub.get_company_news(symbol, from_str, to_str)
        
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            self.logger.error(f"Error fetching company news: {result[0]['error']}")
            return []
        
        return result if isinstance(result, list) else []
    
    def get_market_news(self, category: str = "general") -> List[Dict[str, Any]]:
        result = self.finnhub.get_market_news(category)
        
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            self.logger.error(f"Error fetching market news: {result[0]['error']}")
            return []
        
        return result if isinstance(result, list) else []
    
    def get_insider_transactions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get insider transactions"""
        result = self.finnhub.get_insider_transactions(symbol)
        
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            self.logger.error(f"Error fetching insider transactions: {result[0]['error']}")
            return []
        
        return result if isinstance(result, list) else []
    
    def get_insider_sentiment(self, symbol: str, 
                            from_date: Optional[datetime] = None,
                            to_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get insider sentiment"""
        if from_date is None:
            from_date = datetime.now() - timedelta(days=90)
        if to_date is None:
            to_date = datetime.now()
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        result = self.finnhub.get_insider_sentiment(symbol, from_str, to_str)
        
        if "error" in result:
            self.logger.error(f"Error fetching insider sentiment: {result['error']}")
            return {}
        
        return result
    
    def search_news(self, query: str, from_date: Optional[datetime] = None,
                   to_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)
        if to_date is None:
            to_date = datetime.now()
        
        try:
            news_data = self.get_market_news("general")
            
            filtered_news = []
            for article in news_data:
                if (query.lower() in article.get('headline', '').lower() or 
                    query.lower() in article.get('summary', '').lower()):
                    
                    article_date = datetime.fromtimestamp(article.get('datetime', 0))
                    if from_date <= article_date <= to_date:
                        filtered_news.append(article)
            
            return filtered_news
        except Exception as e:
            print(f"Error searching news: {e}")
            return []