import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from config import Config
from .finnhub_provider import FinnhubProvider

try:
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest
    ALPACA_PY_AVAILABLE = True
except ImportError:
    ALPACA_PY_AVAILABLE = False


class NewsDataProvider:
    def __init__(self, api_key: str = None):
        self.finnhub = FinnhubProvider(api_key)
        self.logger = logging.getLogger(__name__)
        self.alpaca_client = None

        if ALPACA_PY_AVAILABLE and Config.validate_alpaca_config():
            try:
                self.alpaca_client = NewsClient(
                    api_key=Config.ALPACA_API_KEY,
                    secret_key=Config.ALPACA_SECRET_KEY
                )
            except Exception as exc:
                self.logger.warning(f"Alpaca news client unavailable: {exc}")

    def _normalize_finnhub_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = article.get("datetime")
        published = "Unknown"
        sort_ts = 0
        if isinstance(timestamp, (int, float)):
            published_dt = datetime.fromtimestamp(timestamp)
            published = published_dt.strftime("%Y-%m-%d %H:%M:%S")
            sort_ts = int(published_dt.timestamp())
        elif isinstance(timestamp, str):
            published = timestamp

        return {
            "provider": "finnhub",
            "id": f"finnhub-{article.get('id', timestamp)}",
            "headline": article.get("headline"),
            "summary": article.get("summary"),
            "source": article.get("source"),
            "url": article.get("url"),
            "datetime": published,
            "symbols": article.get("related"),
            "_sort_ts": sort_ts,
        }

    def _normalize_alpaca_article(self, article: Any) -> Dict[str, Any]:
        if hasattr(article, "model_dump"):
            payload = article.model_dump()
        elif isinstance(article, dict):
            payload = article
        else:
            payload = {}

        created_at = payload.get("created_at") or payload.get("updated_at")
        published = "Unknown"
        sort_ts = 0
        if isinstance(created_at, datetime):
            published = created_at.strftime("%Y-%m-%d %H:%M:%S")
            sort_ts = int(created_at.timestamp())
        elif isinstance(created_at, str):
            published = created_at

        summary = payload.get("summary") or payload.get("content")

        return {
            "provider": "alpaca",
            "id": f"alpaca-{payload.get('id')}",
            "headline": payload.get("headline"),
            "summary": summary,
            "source": payload.get("source", "Benzinga"),
            "url": payload.get("url"),
            "datetime": published,
            "symbols": payload.get("symbols"),
            "_sort_ts": sort_ts,
        }

    def _get_alpaca_news(self, symbols: Optional[List[str]] = None,
                         start: Optional[datetime] = None,
                         end: Optional[datetime] = None,
                         limit: int = 50) -> List[Dict[str, Any]]:
        if not self.alpaca_client:
            return []

        symbols_param = ",".join(symbols) if symbols else None
        request = NewsRequest(
            symbols=symbols_param,
            start=start,
            end=end,
            limit=limit,
            sort="desc"
        )

        try:
            result = self.alpaca_client.get_news(request)
        except Exception as exc:
            self.logger.error(f"Error fetching Alpaca news: {exc}")
            return []

        items: List[Any] = []
        if hasattr(result, "data"):
            for news_list in result.data.values():
                items.extend(news_list)
        elif isinstance(result, dict):
            items = result.get("data", [])

        return [self._normalize_alpaca_article(item) for item in items]

    def _merge_news(self, *news_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = []
        seen = set()

        for news in news_lists:
            for article in news:
                key = article.get("url") or article.get("id") or article.get("headline")
                if key in seen:
                    continue
                seen.add(key)
                combined.append(article)

        combined.sort(key=lambda item: item.get("_sort_ts", 0), reverse=True)
        for article in combined:
            article.pop("_sort_ts", None)
        return combined
    
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
            result = []

        finnhub_news = [
            self._normalize_finnhub_article(article)
            for article in result
            if isinstance(article, dict)
        ]
        alpaca_news = self._get_alpaca_news(
            symbols=[symbol],
            start=from_date,
            end=to_date
        )

        return self._merge_news(finnhub_news, alpaca_news)
    
    def get_market_news(self, category: str = "general") -> List[Dict[str, Any]]:
        result = self.finnhub.get_market_news(category)
        
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            self.logger.error(f"Error fetching market news: {result[0]['error']}")
            result = []

        finnhub_news = [
            self._normalize_finnhub_article(article)
            for article in result
            if isinstance(article, dict)
        ]
        alpaca_news = self._get_alpaca_news()

        return self._merge_news(finnhub_news, alpaca_news)
    
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
