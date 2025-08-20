"""
External Data Source Providers for Financial Analysis

Integrates with Finnhub and FRED APIs for comprehensive market data
"""

import os
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

class FinnhubProvider:
    """Finnhub API provider for market data and news"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        self.base_url = "https://finnhub.io/api/v1"
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("Finnhub API key not provided")
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile information"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            response = requests.get(
                f"{self.base_url}/stock/profile2",
                params={'symbol': symbol, 'token': self.api_key},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching company profile for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Get basic financial metrics"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            response = requests.get(
                f"{self.base_url}/stock/metric",
                params={'symbol': symbol, 'metric': 'all', 'token': self.api_key},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching financials for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_earnings_calendar(self, symbol: str = None, 
                            from_date: str = None, 
                            to_date: str = None) -> Dict[str, Any]:
        """Get earnings calendar"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        params = {'token': self.api_key}
        if symbol:
            params['symbol'] = symbol
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        try:
            response = requests.get(
                f"{self.base_url}/calendar/earnings",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching earnings calendar: {e}")
            return {"error": str(e)}
    
    def get_company_news(self, symbol: str, 
                        from_date: str = None, 
                        to_date: str = None) -> List[Dict[str, Any]]:
        """Get company news"""
        if not self.api_key:
            return [{"error": "API key not configured"}]
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            response = requests.get(
                f"{self.base_url}/company-news",
                params={
                    'symbol': symbol,
                    'from': from_date,
                    'to': to_date,
                    'token': self.api_key
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return [{"error": str(e)}]
    
    def get_market_news(self, category: str = 'general') -> List[Dict[str, Any]]:
        """Get general market news"""
        if not self.api_key:
            return [{"error": "API key not configured"}]
        
        try:
            response = requests.get(
                f"{self.base_url}/news",
                params={'category': category, 'token': self.api_key},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching market news: {e}")
            return [{"error": str(e)}]
    
    def get_recommendation_trends(self, symbol: str) -> Dict[str, Any]:
        """Get analyst recommendation trends"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            response = requests.get(
                f"{self.base_url}/stock/recommendation",
                params={'symbol': symbol, 'token': self.api_key},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching recommendations for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_price_target(self, symbol: str) -> Dict[str, Any]:
        """Get analyst price targets"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            response = requests.get(
                f"{self.base_url}/stock/price-target",
                params={'symbol': symbol, 'token': self.api_key},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching price target for {symbol}: {e}")
            return {"error": str(e)}

class FREDProvider:
    """Federal Reserve Economic Data (FRED) API provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred"
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("FRED API key not provided")
    
    def get_series_data(self, series_id: str, 
                       start_date: str = None, 
                       end_date: str = None) -> pd.DataFrame:
        """Get economic data series from FRED"""
        if not self.api_key:
            return pd.DataFrame({"error": ["API key not configured"]})
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        try:
            response = requests.get(
                f"{self.base_url}/series/observations",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            df = pd.DataFrame(observations)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.set_index('date')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching FRED series {series_id}: {e}")
            return pd.DataFrame({"error": [str(e)]})
    
    def get_gdp_data(self, start_date: str = None) -> pd.DataFrame:
        """Get GDP data"""
        return self.get_series_data('GDP', start_date)
    
    def get_unemployment_rate(self, start_date: str = None) -> pd.DataFrame:
        """Get unemployment rate"""
        return self.get_series_data('UNRATE', start_date)
    
    def get_inflation_rate(self, start_date: str = None) -> pd.DataFrame:
        """Get CPI inflation rate"""
        return self.get_series_data('CPIAUCSL', start_date)
    
    def get_federal_funds_rate(self, start_date: str = None) -> pd.DataFrame:
        """Get Federal Funds Rate"""
        return self.get_series_data('FEDFUNDS', start_date)
    
    def get_treasury_yield(self, maturity: str = '10Y', start_date: str = None) -> pd.DataFrame:
        """Get Treasury yield rates"""
        series_mapping = {
            '3M': 'TB3MS',
            '6M': 'TB6MS',
            '1Y': 'GS1',
            '2Y': 'GS2',
            '5Y': 'GS5',
            '10Y': 'GS10',
            '30Y': 'GS30'
        }
        
        series_id = series_mapping.get(maturity, 'GS10')
        return self.get_series_data(series_id, start_date)
    
    def get_economic_indicators(self, start_date: str = None) -> Dict[str, pd.DataFrame]:
        """Get key economic indicators"""
        indicators = {
            'GDP': self.get_gdp_data(start_date),
            'Unemployment': self.get_unemployment_rate(start_date),
            'Inflation': self.get_inflation_rate(start_date),
            'Federal_Funds_Rate': self.get_federal_funds_rate(start_date),
            'Treasury_10Y': self.get_treasury_yield('10Y', start_date)
        }
        
        return indicators
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for economic data series"""
        if not self.api_key:
            return [{"error": "API key not configured"}]
        
        try:
            response = requests.get(
                f"{self.base_url}/series/search",
                params={
                    'search_text': search_text,
                    'api_key': self.api_key,
                    'file_type': 'json',
                    'limit': limit
                },
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('seriess', [])
            
        except Exception as e:
            self.logger.error(f"Error searching FRED series: {e}")
            return [{"error": str(e)}]

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
    
    # FRED economic data
    fred = FREDProvider(fred_key)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data['economic_indicators'] = fred.get_economic_indicators(start_date)
    
    return data