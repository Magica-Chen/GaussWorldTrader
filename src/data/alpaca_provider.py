import alpaca_trade_api as tradeapi
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import Config
from src.utils.timezone_utils import EASTERN, now_et

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
        
        # Check VIP status using get_latest_bar with SPY
        self.vip = self._check_vip_status()
        self.default_feed = 'sip'
        self.using_iex = False
        
        if not self.vip:
            logging.info("Free tier: Using fallback logic - SIP historical + IEX for today's data")
        else:
            logging.info("Pro tier: Using SIP feed for all data")
    
    def get_bars(self, symbol: str, timeframe: str = '1Day', 
                 start: Optional[datetime] = None, 
                 end: Optional[datetime] = None,
                 limit: int = 1000, feed: str = 'sip') -> pd.DataFrame:
        # Use ET time for all trading logic
        if start is None:
            start = now_et() - timedelta(days=365)
        if end is None:
            end = now_et()
        
        # Reset IEX usage flag for each call
        self.using_iex = False
        
        # VIP accounts can use SIP directly
        if self.vip:
            try:
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    limit=limit,
                    adjustment='raw',
                    feed=feed
                )
                return self._process_bars(bars)
            except Exception as e:
                logging.error(f"SIP feed failed for VIP account: {e}")
                # Fallback to free tier logic
        
        # Free tier or fallback logic: try SIP first, then use historical SIP + today's IEX
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=limit,
                adjustment='raw',
                feed='sip'
            )
            return self._process_bars(bars)
        except Exception as sip_error:
            logging.info(f"SIP feed failed, using fallback logic: {sip_error}")
            
            # Fallback: combine historical SIP data + today's IEX data
            today = now_et().date()
            yesterday = end - timedelta(days=1)
            
            if end.date() >= today:
                # Need to combine historical + latest data
                historical_end = yesterday
                
                try:
                    # Get historical data until yesterday
                    historical_bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=start.strftime('%Y-%m-%d'),
                        end=historical_end.strftime('%Y-%m-%d'),
                        limit=limit,
                        adjustment='raw',
                        feed='sip'  # Historical data is free
                    )
                    
                    # Get today's data with IEX feed
                    today_bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=today.strftime('%Y-%m-%d'),
                        end=today.strftime('%Y-%m-%d'),
                        limit=1,
                        adjustment='raw',
                        feed='iex'  # Real-time IEX data for today
                    )
                    
                    # Combine historical and today's bars
                    all_bars = list(historical_bars) + list(today_bars)
                    self.using_iex = True
                    logging.info(f"Combined historical SIP data + today's IEX data for {symbol}")
                    return self._process_bars(all_bars)
                    
                except Exception as e:
                    logging.warning(f"Could not get today's IEX data for {symbol}: {e}")
                    # Return historical data only
                    try:
                        historical_bars = self.api.get_bars(
                            symbol,
                            timeframe,
                            start=start.strftime('%Y-%m-%d'),
                            end=historical_end.strftime('%Y-%m-%d'),
                            limit=limit,
                            adjustment='raw',
                            feed='sip'
                        )
                        return self._process_bars(historical_bars)
                    except Exception as final_error:
                        logging.error(f"All data retrieval methods failed for {symbol}: {final_error}")
                        return pd.DataFrame()
            else:
                # Requesting only historical data - try SIP again
                try:
                    bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'),
                        limit=limit,
                        adjustment='raw',
                        feed='sip'
                    )
                    return self._process_bars(bars)
                except Exception as e:
                    logging.error(f"Failed to get historical data for {symbol}: {e}")
                    return pd.DataFrame()
        
    def _process_bars(self, bars) -> pd.DataFrame:
        """Process bars data into DataFrame"""
        data = []
        for bar in bars:
            # Handle different bar object formats
            timestamp = getattr(bar, 'timestamp', None) or getattr(bar, 't', None)
            
            # Skip bars with invalid timestamps
            if timestamp is None:
                continue
                
            data.append({
                'timestamp': timestamp,
                'open': float(getattr(bar, 'open', 0) or getattr(bar, 'o', 0)),
                'high': float(getattr(bar, 'high', 0) or getattr(bar, 'h', 0)),
                'low': float(getattr(bar, 'low', 0) or getattr(bar, 'l', 0)),
                'close': float(getattr(bar, 'close', 0) or getattr(bar, 'c', 0)),
                'volume': int(getattr(bar, 'volume', 0) or getattr(bar, 'v', 0))
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            # Ensure index is datetime and drop any NaT values
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
        
        return df
    
    def _check_vip_status(self) -> bool:
        """Check if account is VIP by testing get_latest_bar with SPY and SIP feed"""
        try:
            # Test VIP status by attempting to get latest bar for SPY with SIP feed
            latest_bar = self.api.get_latest_bar('SPY', feed='sip')
            
            # If we get here without exception, it's a VIP account
            logging.info("VIP account detected: Successfully accessed SIP feed")
            return True
            
        except Exception as e:
            logging.info(f"Free tier account detected: {e}")
            return False
    
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        feed_type = 'sip' if self.vip else 'iex'
        
        try:
            quote = self.api.get_latest_quote(symbol, feed=feed_type)
            
            result = {
                'symbol': symbol,
                'bid_price': float(quote.bid_price),
                'bid_size': int(quote.bid_size),
                'ask_price': float(quote.ask_price),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp,
                'feed_type': feed_type,
                'is_delayed': False  # IEX is real-time, SIP is also real-time
            }
            
            if not self.vip:
                result['data_source'] = "Real-time IEX data"
            else:
                result['data_source'] = "Real-time SIP data"
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting latest quote for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'feed_type': feed_type,
                'is_delayed': False
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
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including VIP status and data feed info"""
        try:
            account = self.api.get_account()
            return {
                'vip': self.vip,
                'using_iex': self.using_iex,
                'account_equity': float(account.equity) if account.equity else 0,
                'account_status': account.status if hasattr(account, 'status') else 'unknown',
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False),
                'default_feed': self.default_feed,
                'has_real_time_data': True,
                'data_delay': 'Real-time',
                'feed_description': 'Securities Information Processor (SIP)' if self.vip else 'IEX Real-time + SIP Historical'
            }
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
            return {
                'vip': self.vip,
                'using_iex': self.using_iex,
                'account_equity': 0,
                'account_status': 'error',
                'pattern_day_trader': False,
                'error': str(e)
            }
    
    
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
    
