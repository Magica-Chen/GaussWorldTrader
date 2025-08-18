import requests
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from config import Config

class MacroDataProvider:
    def __init__(self):
        if not Config.validate_fred_config():
            raise ValueError("FRED API key not configured")
        
        self.api_key = Config.FRED_API_KEY
        self.base_url = "https://api.stlouisfed.org/fred"
    
    def get_series_data(self, series_id: str, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        url = f"{self.base_url}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date.strftime('%Y-%m-%d'),
            'observation_end': end_date.strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            observations = data.get('observations', [])
            df_data = []
            
            for obs in observations:
                if obs['value'] != '.':
                    df_data.append({
                        'date': pd.to_datetime(obs['date']),
                        'value': float(obs['value'])
                    })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error fetching FRED data for {series_id}: {e}")
            return pd.DataFrame()
    
    def get_gdp_data(self) -> pd.DataFrame:
        return self.get_series_data('GDP')
    
    def get_unemployment_rate(self) -> pd.DataFrame:
        return self.get_series_data('UNRATE')
    
    def get_inflation_rate(self) -> pd.DataFrame:
        return self.get_series_data('CPIAUCSL')
    
    def get_federal_funds_rate(self) -> pd.DataFrame:
        return self.get_series_data('FEDFUNDS')
    
    def get_10_year_treasury(self) -> pd.DataFrame:
        return self.get_series_data('GS10')
    
    def get_sp500_data(self) -> pd.DataFrame:
        return self.get_series_data('SP500')
    
    def get_vix_data(self) -> pd.DataFrame:
        return self.get_series_data('VIXCLS')
    
    def get_dollar_index(self) -> pd.DataFrame:
        return self.get_series_data('DTWEXBGS')
    
    def get_economic_indicators(self) -> Dict[str, pd.DataFrame]:
        indicators = {
            'GDP': self.get_gdp_data(),
            'Unemployment_Rate': self.get_unemployment_rate(),
            'Inflation_Rate': self.get_inflation_rate(),
            'Federal_Funds_Rate': self.get_federal_funds_rate(),
            'Treasury_10Y': self.get_10_year_treasury(),
            'SP500': self.get_sp500_data(),
            'VIX': self.get_vix_data(),
            'Dollar_Index': self.get_dollar_index()
        }
        return indicators
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/series"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'seriess' in data and len(data['seriess']) > 0:
                series_info = data['seriess'][0]
                return {
                    'id': series_info.get('id'),
                    'title': series_info.get('title'),
                    'units': series_info.get('units'),
                    'frequency': series_info.get('frequency'),
                    'seasonal_adjustment': series_info.get('seasonal_adjustment'),
                    'last_updated': series_info.get('last_updated'),
                    'observation_start': series_info.get('observation_start'),
                    'observation_end': series_info.get('observation_end')
                }
            return {}
            
        except Exception as e:
            print(f"Error fetching series info for {series_id}: {e}")
            return {}