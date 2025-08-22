import alpaca_trade_api as tradeapi
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import pytz
EASTERN = pytz.timezone('US/Eastern')
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
        
        # Check subscription status and determine feed type
        self.subscription_status = self._check_subscription_status()
        self.default_feed = 'sip'
        
        if not self.subscription_status['has_sip_subscription']:
            logging.info("Free tier: Using SIP for historical data + IEX for today's real-time data")
    
    def get_bars(self, symbol: str, timeframe: str = '1Day', 
                 start: Optional[datetime] = None, 
                 end: Optional[datetime] = None,
                 limit: int = 1000) -> pd.DataFrame:
        # Use ET time for all trading logic
        if start is None:
            start = datetime.now(EASTERN) - timedelta(days=365)
        if end is None:
            end = datetime.now(EASTERN)
        
        # Use conditional logic based on subscription status
        if self.subscription_status['has_sip_subscription']:
            # Full SIP subscription - use real-time feed
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=limit,
                adjustment='raw',
                feed='sip'
            )
        else:
            # Free tier - combine historical data (until yesterday) + latest data
            today = datetime.now(EASTERN).date()
            yesterday = end- timedelta(days=1)

            if end.date() >= today:
                # Need to combine historical + latest data
                historical_end = yesterday

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

                # Get today's data with IEX feed (real-time, not delayed)
                try:
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
                    bars = all_bars
                    logging.info(f"Combined historical SIP data + today's IEX data for {symbol}")

                except Exception as e:
                    logging.warning(f"Could not get today's IEX data for {symbol}: {e}")
                    bars = historical_bars

            else:
                # Requesting only historical data - use SIP feed
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    limit=limit,
                    adjustment='raw',
                    feed='sip'
                )
        
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
        df.set_index('timestamp', inplace=True)
        
        # Ensure index is datetime and drop any NaT values
        df.index = pd.to_datetime(df.index)
        df = df.dropna()
        
        return df
    
    def _check_subscription_status(self) -> Dict[str, Any]:
        """Check the account's market data subscription status by testing SIP feed access"""
        try:
            account = self.api.get_account()
            equity = float(account.equity) if account.equity else 0
            
            # Test SIP subscription by attempting to get latest bar with SIP feed
            has_sip_subscription = False
            test_error = None
            
            try:
                # Simple test: get latest bar for SPY with SIP feed
                latest_bar = self.api.get_latest_bar('SPY', feed='sip')
                
                # If we get here without exception, SIP subscription is active
                has_sip_subscription = True
                logging.info("Pro Tier detected: Successfully accessed SIP feed")
                
            except Exception as sip_error:
                has_sip_subscription = False
                test_error = f"Free Tier: {sip_error}"
                logging.info("Free Tier detected: SIP subscription not available")
            
            return {
                'has_sip_subscription': has_sip_subscription,
                'account_equity': equity,
                'account_status': account.status if hasattr(account, 'status') else 'unknown',
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False),
                'sip_test_error': test_error
            }
            
        except Exception as e:
            logging.error(f"Error checking subscription status: {e}")
            return {
                'has_sip_subscription': False,
                'account_equity': 0,
                'account_status': 'error',
                'pattern_day_trader': False,
                'error': str(e)
            }
    
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        feed_type = 'sip' if self.subscription_status['has_sip_subscription'] else 'iex'
        
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
            
            if not self.subscription_status['has_sip_subscription']:
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
    
    def get_subscription_info(self) -> Dict[str, Any]:
        """Get detailed subscription and data feed information for display"""
        info = self.subscription_status.copy()
        info.update({
            'default_feed': self.default_feed,
            'has_real_time_data': True,  # Both SIP and IEX provide real-time data
            'data_delay': 'Real-time',
            'feed_description': 'Securities Information Processor (SIP)' if self.subscription_status['has_sip_subscription'] else 'IEX Real-time + SIP Historical'
        })
        return info
    
    
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
    
