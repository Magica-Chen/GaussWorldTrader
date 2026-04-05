"""
Federal Reserve Economic Data (FRED) API provider
"""

import logging
import os
from typing import Any, Dict, List

import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None


class FREDProviderError(RuntimeError):
    """Raised when FRED data cannot be loaded."""


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
            self.client = Fred(api_key=self.api_key)

    def _require_client(self) -> Fred:
        if not self.api_key:
            raise FREDProviderError("FRED API key not provided")
        if Fred is None:
            raise FREDProviderError("fredapi library is not installed")
        if self.client is None:
            raise FREDProviderError("FRED client is not initialized")
        return self.client
    
    def get_series_data(
        self,
        series_id: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Get economic data series from FRED"""
        client = self._require_client()
        try:
            data = client.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
        except Exception as exc:
            raise FREDProviderError(f"Failed to load FRED series {series_id}: {exc}") from exc

        df = pd.DataFrame({'value': data})
        df.index.name = 'date'
        return df
    
    def get_gdp_data(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Get GDP data"""
        return self.get_series_data('GDP', start_date, end_date)
    
    def get_unemployment_rate(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Get unemployment rate"""
        return self.get_series_data('UNRATE', start_date, end_date)
    
    def get_inflation_rate(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Get CPI inflation rate"""
        return self.get_series_data('CPIAUCSL', start_date, end_date)
    
    def get_federal_funds_rate(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Get Federal Funds Rate"""
        return self.get_series_data('FEDFUNDS', start_date, end_date)
    
    def get_treasury_yield(
        self,
        maturity: str = '10Y',
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
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
        return self.get_series_data(series_id, start_date, end_date)
    
    def get_economic_indicators(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """Get key economic indicators"""
        indicators = {
            'GDP': self.get_gdp_data(start_date, end_date),
            'Unemployment': self.get_unemployment_rate(start_date, end_date),
            'Inflation': self.get_inflation_rate(start_date, end_date),
            'Federal_Funds_Rate': self.get_federal_funds_rate(start_date, end_date),
            'Treasury_10Y': self.get_treasury_yield('10Y', start_date, end_date)
        }
        
        return indicators
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for economic data series"""
        client = self._require_client()
        try:
            search_results = client.search(search_text, limit=limit)
        except Exception as exc:
            raise FREDProviderError(f"Failed to search FRED series for {search_text}: {exc}") from exc

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
