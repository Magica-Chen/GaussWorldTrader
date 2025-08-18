import requests
import pandas as pd
from typing import Dict, Any
from config import Config

class CryptoDataProvider:
    def __init__(self):
        self.coindesk_url = Config.COINDESK_API_URL
    
    def get_bitcoin_price(self) -> Dict[str, Any]:
        try:
            response = requests.get(self.coindesk_url)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': 'BTC',
                'price_usd': float(data['bpi']['USD']['rate_float']),
                'price_eur': float(data['bpi']['EUR']['rate_float']),
                'price_gbp': float(data['bpi']['GBP']['rate_float']),
                'last_updated': data['time']['updated'],
                'disclaimer': data['disclaimer']
            }
        except Exception as e:
            raise Exception(f"Failed to fetch Bitcoin price: {e}")
    
    def get_crypto_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            prices = data['prices']
            volumes = data['total_volumes']
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp, unit='ms'),
                    'price': price,
                    'volume': volumes[i][1] if i < len(volumes) else 0
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df
        
        except Exception as e:
            print(f"Error fetching crypto historical data: {e}")
            return pd.DataFrame()
    
    def get_crypto_market_data(self, symbol: str) -> Dict[str, Any]:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'].upper(),
                'name': data['name'],
                'current_price': data['market_data']['current_price']['usd'],
                'market_cap': data['market_data']['market_cap']['usd'],
                'total_volume': data['market_data']['total_volume']['usd'],
                'price_change_24h': data['market_data']['price_change_24h'],
                'price_change_percentage_24h': data['market_data']['price_change_percentage_24h'],
                'market_cap_rank': data['market_cap_rank'],
                'last_updated': data['last_updated']
            }
        except Exception as e:
            print(f"Error fetching crypto market data: {e}")
            return {}