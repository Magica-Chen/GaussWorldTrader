import alpaca_trade_api as tradeapi
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
from config import Config

class AlpacaDataProvider:
    def __init__(self):
        if not Config.validate_alpaca_config():
            raise ValueError("Alpaca API credentials not configured")
        
        self.api = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL,
            api_version='v2'
        )
    
    def get_bars(self, symbol: str, timeframe: str = '1Day', 
                 start: Optional[datetime] = None, 
                 end: Optional[datetime] = None,
                 limit: int = 1000) -> pd.DataFrame:
        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()
        
        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            limit=limit,
            adjustment='raw'
        )
        
        data = []
        for bar in bars:
            # Handle different bar object formats
            timestamp = getattr(bar, 'timestamp', None) or getattr(bar, 't', None)
            data.append({
                'timestamp': timestamp,
                'open': float(getattr(bar, 'open', 0) or getattr(bar, 'o', 0)),
                'high': float(getattr(bar, 'high', 0) or getattr(bar, 'h', 0)),
                'low': float(getattr(bar, 'low', 0) or getattr(bar, 'l', 0)),
                'close': float(getattr(bar, 'close', 0) or getattr(bar, 'c', 0)),
                'volume': int(getattr(bar, 'volume', 0) or getattr(bar, 'v', 0))
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        quote = self.api.get_latest_quote(symbol)
        return {
            'symbol': symbol,
            'bid_price': float(quote.bid_price),
            'bid_size': int(quote.bid_size),
            'ask_price': float(quote.ask_price),
            'ask_size': int(quote.ask_size),
            'timestamp': quote.timestamp
        }
    
    def get_account(self) -> Dict[str, Any]:
        account = self.api.get_account()
        return {
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'day_trade_count': int(account.day_trade_count),
            'pattern_day_trader': account.pattern_day_trader
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        positions = self.api.list_positions()
        return [{
            'symbol': pos.symbol,
            'qty': float(pos.qty),
            'side': pos.side,
            'market_value': float(pos.market_value),
            'cost_basis': float(pos.cost_basis),
            'unrealized_pl': float(pos.unrealized_pl),
            'unrealized_plpc': float(pos.unrealized_plpc)
        } for pos in positions]
    
    def get_options_chain(self, underlying_symbol: str) -> pd.DataFrame:
        try:
            options = self.api.list_options_contracts(underlying_symbol)
            data = []
            for option in options:
                data.append({
                    'symbol': option.symbol,
                    'underlying_symbol': option.underlying_symbol,
                    'option_type': option.type,
                    'strike_price': float(option.strike_price),
                    'expiration_date': option.expiration_date,
                    'size': option.size
                })
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return pd.DataFrame()