import alpaca_trade_api as tradeapi
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import requests
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
        
        # Set up headers for direct API calls
        self.headers = {
            'APCA-API-KEY-ID': Config.ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': Config.ALPACA_SECRET_KEY,
            'Content-Type': 'application/json'
        }
        self.data_base_url = "https://data.alpaca.markets"
        
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
        # Check if this is an options symbol and route to appropriate method
        if self.is_option_symbol(symbol):
            return self.get_option_bars(symbol, timeframe, start, end, limit)
        
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
        """Get options chain using the snapshots endpoint"""
        try:
            url = f"{self.data_base_url}/v1beta1/options/snapshots/{underlying_symbol}"
            feed = 'opra' if self.vip else 'indicative'
            
            params = {'feed': feed}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_options_chain_json(data, underlying_symbol)
            
        except Exception as e:
            logging.error(f"Error fetching options chain for {underlying_symbol}: {e}")
            return pd.DataFrame()
    
    def _process_options_chain_json(self, data: Dict, underlying_symbol: str) -> pd.DataFrame:
        """Process options chain JSON response into DataFrame"""
        options_data = []
        snapshots = data.get('snapshots', {})
        
        for symbol, snapshot in snapshots.items():
            try:
                # Extract option details from symbol (e.g., AAPL250117C00150000)
                option_type = 'C' if 'C' in symbol else 'P' if 'P' in symbol else 'Unknown'
                
                latest_quote = snapshot.get('latestQuote', {})
                latest_trade = snapshot.get('latestTrade', {})
                greeks = snapshot.get('greeks', {})
                
                options_data.append({
                    'symbol': symbol,
                    'underlying_symbol': underlying_symbol,
                    'option_type': option_type,
                    'bid_price': float(latest_quote.get('bp', 0)),
                    'ask_price': float(latest_quote.get('ap', 0)),
                    'last_price': float(latest_trade.get('p', 0)),
                    'volume': int(latest_trade.get('s', 0)),
                    'delta': float(greeks.get('delta', 0)),
                    'gamma': float(greeks.get('gamma', 0)),
                    'theta': float(greeks.get('theta', 0)),
                    'vega': float(greeks.get('vega', 0))
                })
            except Exception as e:
                logging.warning(f"Error processing option {symbol}: {e}")
                continue
        
        return pd.DataFrame(options_data)
    
    def is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an options contract"""
        # Options symbols typically have format: AAPL250117C00150000
        # Or legacy format with spaces/special chars
        return (
            len(symbol) > 10 and 
            ('C' in symbol[-9:] or 'P' in symbol[-9:]) and 
            any(char.isdigit() for char in symbol[-8:])
        ) or 'C00' in symbol or 'P00' in symbol
    
    def get_option_bars(self, symbol: str, timeframe: str = '1Day',
                       start: Optional[datetime] = None,
                       end: Optional[datetime] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Get historical option bars using direct API call"""
        if start is None:
            start = now_et() - timedelta(days=30)  # Options have shorter history
        if end is None:
            end = now_et()
        
        try:
            # Use direct HTTP request to options bars endpoint
            url = f"{self.data_base_url}/v1beta1/options/bars"
            params = {
                'symbols': symbol,
                'timeframe': timeframe,  # Keep original format like '1Day'
                'start': start.strftime('%Y-%m-%dT%H:%M:%S') + 'Z',  # ISO format with timezone
                'end': end.strftime('%Y-%m-%dT%H:%M:%S') + 'Z',
                'limit': limit,
                'sort': 'asc'
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_option_bars_json(data, symbol)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.warning(f"Options bars not available for free tier account: {symbol}")
                return self._create_synthetic_option_bars_from_quotes(symbol, start, end)
            elif e.response.status_code == 400:
                logging.warning(f"Options bars not available or invalid symbol: {symbol}")
                return self._create_synthetic_option_bars_from_quotes(symbol, start, end)
            else:
                logging.error(f"HTTP error getting option bars for {symbol}: {e}")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error getting option bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_option_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest quotes for option symbols using direct API call"""
        feed = 'opra' if self.vip else 'indicative'
        results = {}
        
        try:
            url = f"{self.data_base_url}/v1beta1/options/quotes/latest"
            params = {
                'symbols': ','.join(symbols),  # Comma-separated list of symbols
                'feed': feed
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            quotes = data.get('quotes', {})
            
            for symbol in symbols:
                if symbol in quotes:
                    quote = quotes[symbol]
                    results[symbol] = {
                        'symbol': symbol,
                        'bid_price': float(quote.get('bp', 0)),
                        'bid_size': int(quote.get('bs', 0)),
                        'ask_price': float(quote.get('ap', 0)),
                        'ask_size': int(quote.get('as', 0)),
                        'timestamp': quote.get('t', ''),
                        'feed_type': feed
                    }
                else:
                    results[symbol] = {'symbol': symbol, 'error': 'No quote data available'}
                    
        except Exception as e:
            logging.error(f"Error getting option quotes: {e}")
            for symbol in symbols:
                results[symbol] = {'symbol': symbol, 'error': str(e)}
        
        return results
    
    def get_option_latest_trades(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest trades for option symbols using direct API call"""
        feed = 'opra' if self.vip else 'indicative'
        results = {}
        
        try:
            url = f"{self.data_base_url}/v1beta1/options/trades/latest"
            params = {
                'symbols': ','.join(symbols),  # Comma-separated list of symbols
                'feed': feed
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            trades = data.get('trades', {})
            
            for symbol in symbols:
                if symbol in trades:
                    trade = trades[symbol]
                    results[symbol] = {
                        'symbol': symbol,
                        'price': float(trade.get('p', 0)),
                        'size': int(trade.get('s', 0)),
                        'timestamp': trade.get('t', ''),
                        'exchange': trade.get('x', 'N/A'),
                        'feed_type': feed
                    }
                else:
                    results[symbol] = {'symbol': symbol, 'error': 'No trade data available'}
                    
        except Exception as e:
            logging.error(f"Error getting option trades: {e}")
            for symbol in symbols:
                results[symbol] = {'symbol': symbol, 'error': str(e)}
        
        return results
    
    def get_option_trades(self, symbol: str, 
                         start: Optional[datetime] = None,
                         end: Optional[datetime] = None,
                         limit: int = 1000) -> pd.DataFrame:
        """Get historical option trades using direct API call"""
        if start is None:
            start = now_et() - timedelta(days=7)  # Options trades have limited history
        if end is None:
            end = now_et()
            
        feed = 'opra' if self.vip else 'indicative'
        
        try:
            url = f"{self.data_base_url}/v1beta1/options/trades"
            params = {
                'symbols': symbol,
                'start': start.strftime('%Y-%m-%dT%H:%M:%S') + 'Z',  # ISO format with timezone
                'end': end.strftime('%Y-%m-%dT%H:%M:%S') + 'Z',
                'limit': limit,
                'feed': feed,
                'sort': 'asc'
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_option_trades_json(data, symbol)
            
        except Exception as e:
            logging.error(f"Error getting option trades for {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_option_bars_json(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Process option bars JSON response into DataFrame"""
        bars_data = []
        bars = data.get('bars', {}).get(symbol, [])
        
        for bar in bars:
            bars_data.append({
                'timestamp': bar.get('t'),
                'open': float(bar.get('o', 0)),
                'high': float(bar.get('h', 0)),
                'low': float(bar.get('l', 0)),
                'close': float(bar.get('c', 0)),
                'volume': int(bar.get('v', 0)),
                'trade_count': int(bar.get('n', 0)),
                'vwap': float(bar.get('vw', 0))
            })
        
        df = pd.DataFrame(bars_data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
        
        return df
    
    def _process_option_trades_json(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Process option trades JSON response into DataFrame"""
        trades_data = []
        trades = data.get('trades', {}).get(symbol, [])
        
        for trade in trades:
            trades_data.append({
                'timestamp': trade.get('t'),
                'price': float(trade.get('p', 0)),
                'size': int(trade.get('s', 0)),
                'exchange': trade.get('x', ''),
                'conditions': trade.get('c', [])
            })
        
        df = pd.DataFrame(trades_data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
        
        return df
    
    def _create_synthetic_option_bars_from_quotes(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Create synthetic option bars using latest quotes when bars are not available"""
        try:
            quotes = self.get_option_latest_quotes([symbol])
            quote = quotes.get(symbol, {})
            
            if 'error' in quote or not quote.get('bid_price') or not quote.get('ask_price'):
                return pd.DataFrame()
            
            # Use mid-price as synthetic price
            mid_price = (quote['bid_price'] + quote['ask_price']) / 2.0
            
            if mid_price <= 0:
                return pd.DataFrame()
            
            # Create a single synthetic bar for today
            synthetic_data = [{
                'timestamp': end.replace(hour=16, minute=0, second=0, microsecond=0),  # Market close
                'open': mid_price,
                'high': mid_price,
                'low': mid_price,
                'close': mid_price,
                'volume': 0,
                'trade_count': 0,
                'vwap': mid_price
            }]
            
            df = pd.DataFrame(synthetic_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            logging.info(f"Created synthetic option bar for {symbol} using mid-price ${mid_price:.2f}")
            return df
            
        except Exception as e:
            logging.error(f"Error creating synthetic option bars for {symbol}: {e}")
            return pd.DataFrame()
    
