"""
Comprehensive market data aggregator
"""

from typing import Dict, Any
from datetime import datetime, timedelta

from .finnhub_provider import FinnhubProvider
from .fred_provider import FREDProvider


def get_comprehensive_market_data(symbol: str, 
                                finnhub_key: str = None, 
                                fred_key: str = None) -> Dict[str, Any]:
    """Get comprehensive market data from multiple sources"""
    data = {}
    
    # Finnhub data
    finnhub = FinnhubProvider(finnhub_key)
    data['company_profile'] = finnhub.get_company_profile(symbol)
    data['basic_financials'] = finnhub.get_basic_financials(symbol)
    data['company_news'] = finnhub.get_company_news(symbol)
    data['recommendations'] = finnhub.get_recommendation_trends(symbol)
    data['price_target'] = finnhub.get_price_target(symbol)
    data['quote'] = finnhub.get_quote(symbol)
    data['earnings_surprises'] = finnhub.get_earnings_surprises(symbol)
    data['insider_transactions'] = finnhub.get_insider_transactions(symbol)
    data['insider_sentiment'] = finnhub.get_insider_sentiment(symbol)
    
    # FRED economic data
    fred = FREDProvider(fred_key)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data['economic_indicators'] = fred.get_economic_indicators(start_date)
    
    return data