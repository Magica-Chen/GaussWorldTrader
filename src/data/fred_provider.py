"""
Federal Reserve Economic Data (FRED) API provider
"""

import os
import pandas as pd
from typing import Dict, List, Any
import logging

try:
    from fredapi import Fred
except ImportError:
    Fred = None


class FREDProvider:
    """Federal Reserve Economic Data (FRED) API provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("FRED API key not provided")
            self.client = None
        elif Fred is None:
            self.logger.error("fredapi library not installed. Install with: pip install fredapi")
            self.client = None
        else:
            try:
                self.client = Fred(api_key=self.api_key)
            except Exception as e:
                self.logger.error(f"Error initializing FRED client: {e}")
                self.client = None
    
    def get_series_data(self, series_id: str, 
                       start_date: str = None, 
                       end_date: str = None) -> pd.DataFrame:
        """Get economic data series from FRED"""
        if not self.client:
            return pd.DataFrame({"error": ["API client not configured"]})
        
        try:
            data = self.client.get_series(
                series_id, 
                observation_start=start_date, 
                observation_end=end_date
            )
            
            # Convert to DataFrame with consistent format
            df = pd.DataFrame({'value': data})
            df.index.name = 'date'
            
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
    
    def get_treasury_yield(self, maturity: str = '10Y', 
                          start_date: str = None) -> pd.DataFrame:
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
        if not self.client:
            return [{"error": "API client not configured"}]
        
        try:
            # Use fredapi's search functionality
            search_results = self.client.search(search_text, limit=limit)
            
            # Convert to list of dictionaries for consistency
            result_list = []
            for idx, row in search_results.iterrows():
                result_list.append({
                    'id': row.get('id', ''),
                    'title': row.get('title', ''),
                    'observation_start': row.get('observation_start', ''),
                    'observation_end': row.get('observation_end', ''),
                    'frequency': row.get('frequency', ''),
                    'units': row.get('units', ''),
                    'seasonal_adjustment': row.get('seasonal_adjustment', ''),
                    'notes': row.get('notes', '')
                })
            
            return result_list
            
        except Exception as e:
            self.logger.error(f"Error searching FRED series: {e}")
            return [{"error": str(e)}]