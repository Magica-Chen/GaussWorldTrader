import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from config import Config

class NewsDataProvider:
    def __init__(self):
        if not Config.validate_finhub_config():
            raise ValueError("Finhub API key not configured")
        
        self.api_key = Config.FINHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_company_news(self, symbol: str, from_date: Optional[datetime] = None, 
                        to_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()
        
        url = f"{self.base_url}/company-news"
        params = {
            'symbol': symbol,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching company news: {e}")
            return []
    
    def get_market_news(self, category: str = "general") -> List[Dict[str, Any]]:
        url = f"{self.base_url}/news"
        params = {
            'category': category,
            'token': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching market news: {e}")
            return []
    
    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.base_url}/news-sentiment"
        params = {
            'symbol': symbol,
            'token': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': symbol,
                'buzz_articles_in_last_week': data.get('buzz', {}).get('articlesInLastWeek', 0),
                'buzz_buzz': data.get('buzz', {}).get('buzz', 0),
                'company_news_score': data.get('companyNewsScore', 0),
                'sector_average_bullishness': data.get('sectorAverageBullishness', 0),
                'sector_average_news_score': data.get('sectorAverageNewsScore', 0),
                'sentiment_bearish_percent': data.get('sentiment', {}).get('bearishPercent', 0),
                'sentiment_bullish_percent': data.get('sentiment', {}).get('bullishPercent', 0)
            }
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return {}
    
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