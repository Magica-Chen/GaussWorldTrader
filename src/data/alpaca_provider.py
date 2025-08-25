from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import Config
from src.utils.timezone_utils import EASTERN, now_et

try:
    from alpaca.data.historical import (
        StockHistoricalDataClient, 
        CryptoHistoricalDataClient, 
        OptionHistoricalDataClient,
        NewsClient
    )
    from alpaca.data.live import (
        StockDataStream, 
        CryptoDataStream, 
        OptionDataStream
    )
    from alpaca.data.requests import (
        StockBarsRequest, 
        StockLatestQuoteRequest,
        StockLatestTradeRequest,
        CryptoBarsRequest,
        CryptoLatestQuoteRequest, 
        CryptoLatestTradeRequest,
        OptionBarsRequest,
        OptionLatestQuoteRequest,
        OptionLatestTradeRequest,
        OptionSnapshotRequest,
        OptionChainRequest
    )
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.common.exceptions import APIError
    ALPACA_PY_AVAILABLE = True
except ImportError:
    logging.warning("alpaca-py not installed, using fallback mode")
    ALPACA_PY_AVAILABLE = False


class AlpacaProvider:
    """
    Modern Alpaca data provider using alpaca-py SDK with separate clients
    for stocks, options, and crypto data.
    """
    
    def __init__(self):
        if not Config.validate_alpaca_config():
            raise ValueError("Alpaca API credentials not configured")
        
        if not ALPACA_PY_AVAILABLE:
            raise ImportError(
                "alpaca-py is required. Install with: pip install alpaca-py"
            )
        
        # Initialize clients with API credentials
        self._init_clients()
        
        # Check account tier and available feeds
        self.account_info = self._get_account_info()
        self.is_pro_tier = self._check_pro_tier()
        
        logging.info(f"Alpaca Provider initialized - Pro tier: {self.is_pro_tier}")
    
    def _init_clients(self):
        """Initialize all Alpaca clients"""
        # Stock data clients
        self.stock_historical_client = StockHistoricalDataClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY
        )
        
        # Option data clients  
        self.option_historical_client = OptionHistoricalDataClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY
        )
        
        # Crypto data clients
        self.crypto_historical_client = CryptoHistoricalDataClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY
        )
        
        # Trading client for account/positions
        self.trading_client = TradingClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY,
            paper=Config.ALPACA_BASE_URL != "https://api.alpaca.markets"
        )
        
        # News client
        self.news_client = NewsClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY
        )
    
    def _check_pro_tier(self) -> bool:
        """Check if account has pro-tier data access"""
        try:
            # Test SIP feed access with a simple stock quote
            request = StockLatestQuoteRequest(
                symbol_or_symbols="SPY",
                feed="sip"
            )
            self.stock_historical_client.get_stock_latest_quote(request)
            return True
        except APIError as e:
            if "subscription" in str(e).lower() or "upgrade" in str(e).lower():
                return False
            # Re-raise if it's a different error
            raise
        except Exception:
            return False
    
    def _get_account_info(self) -> Dict[str, Any]:
        """Get basic account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'account_number': account.account_number,
                'status': account.status,
                'equity': float(account.equity) if account.equity else 0,
                'buying_power': float(account.buying_power) if account.buying_power else 0,
                'cash': float(account.cash) if account.cash else 0,
                'portfolio_value': float(account.portfolio_value) if account.portfolio_value else 0
            }
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
            return {}
    
    def get_stock_bars(self, symbol: str, timeframe: str = '1Day',
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None,
                      limit: int = 1000) -> pd.DataFrame:
        """Get historical stock bars using StockHistoricalDataClient"""
        if start is None:
            start = now_et() - timedelta(days=365)
        if end is None:
            end = now_et()
        
        try:
            # Convert timeframe string to TimeFrame enum
            tf = self._parse_timeframe(timeframe)
            
            # Choose appropriate feed based on account tier
            feed = "sip" if self.is_pro_tier else "iex"
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
                feed=feed
            )
            
            bars = self.stock_historical_client.get_stock_bars(request)
            return self._process_stock_bars(bars, symbol)
            
        except Exception as e:
            logging.error(f"Error getting stock bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest quote for a stock"""
        try:
            feed = "sip" if self.is_pro_tier else "iex"
            
            request = StockLatestQuoteRequest(
                symbol_or_symbols=symbol,
                feed=feed
            )
            
            quotes = self.stock_historical_client.get_stock_latest_quote(request)
            quote = quotes.get(symbol)
            
            if quote:
                return {
                    'symbol': symbol,
                    'bid_price': float(quote.bid_price) if quote.bid_price else 0,
                    'bid_size': int(quote.bid_size) if quote.bid_size else 0,
                    'ask_price': float(quote.ask_price) if quote.ask_price else 0,
                    'ask_size': int(quote.ask_size) if quote.ask_size else 0,
                    'timestamp': quote.timestamp,
                    'feed_type': feed
                }
            else:
                return {'symbol': symbol, 'error': 'No quote data available'}
                
        except Exception as e:
            logging.error(f"Error getting stock quote for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_option_bars(self, symbol: str, timeframe: str = '1Day',
                       start: Optional[datetime] = None,
                       end: Optional[datetime] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Get historical option bars using OptionHistoricalDataClient"""
        if start is None:
            start = now_et() - timedelta(days=30)  # Options have shorter history

        # no need to add end becuase get_option_bars can set it as default.
        # if end is None:
        #     end = now_et()
        
        try:
            tf = self._parse_timeframe(timeframe)

            feed = "opra" if self.is_pro_tier else "indicative"
            
            request = OptionBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
                feed=feed
            )

            bars = self.option_historical_client.get_option_bars(request)
            return self._process_option_bars(bars, symbol)
            
        except Exception as e:
            logging.error(f"Error getting option bars for {symbol}: {e}")
            # Try to create synthetic bar from latest quote as fallback
            return self._create_synthetic_option_bar(symbol, end)
    
    def get_option_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest quote for an option"""
        try:
            feed = "opra" if self.is_pro_tier else "indicative"
            
            request = OptionLatestQuoteRequest(
                symbol_or_symbols=symbol,
                feed=feed
            )
            
            quotes = self.option_historical_client.get_option_latest_quote(request)
            quote = quotes.get(symbol)
            
            if quote:
                return {
                    'symbol': symbol,
                    'bid_price': float(quote.bid_price) if quote.bid_price else 0,
                    'bid_size': int(quote.bid_size) if quote.bid_size else 0,
                    'ask_price': float(quote.ask_price) if quote.ask_price else 0,
                    'ask_size': int(quote.ask_size) if quote.ask_size else 0,
                    'timestamp': quote.timestamp,
                    'feed_type': feed
                }
            else:
                return {'symbol': symbol, 'error': 'No quote data available'}
                
        except Exception as e:
            logging.error(f"Error getting option quote for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_options_chain(self, underlying_symbol: str) -> pd.DataFrame:
        """Get options chain for an underlying symbol"""
        try:
            feed = "opra" if self.is_pro_tier else "indicative"
            
            request = OptionChainRequest(
                underlying_symbols=underlying_symbol,
                feed=feed
            )
            
            chain = self.option_historical_client.get_option_chain(request)
            return self._process_options_chain(chain, underlying_symbol)
            
        except Exception as e:
            logging.error(f"Error getting options chain for {underlying_symbol}: {e}")
            return pd.DataFrame()
    
    def get_crypto_bars(self, symbol: str, timeframe: str = '1Day',
                       start: Optional[datetime] = None,
                       end: Optional[datetime] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Get historical crypto bars using CryptoHistoricalDataClient"""
        if start is None:
            start = now_et() - timedelta(days=365)
        if end is None:
            end = now_et()
        
        try:
            tf = self._parse_timeframe(timeframe)
            
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit
            )
            
            bars = self.crypto_historical_client.get_crypto_bars(request)
            return self._process_crypto_bars(bars, symbol)
            
        except Exception as e:
            logging.error(f"Error getting crypto bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_crypto_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest quote for a crypto pair"""
        try:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.crypto_historical_client.get_crypto_latest_quote(request)
            quote = quotes.get(symbol)
            
            if quote:
                return {
                    'symbol': symbol,
                    'bid_price': float(quote.bid_price) if quote.bid_price else 0,
                    'bid_size': float(quote.bid_size) if quote.bid_size else 0,
                    'ask_price': float(quote.ask_price) if quote.ask_price else 0,
                    'ask_size': float(quote.ask_size) if quote.ask_size else 0,
                    'timestamp': quote.timestamp
                }
            else:
                return {'symbol': symbol, 'error': 'No quote data available'}
                
        except Exception as e:
            logging.error(f"Error getting crypto quote for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        return self.account_info.copy()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get account positions"""
        try:
            positions = self.trading_client.get_all_positions()
            return [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                'market_value': float(pos.market_value) if pos.market_value else 0,
                'cost_basis': float(pos.cost_basis) if pos.cost_basis else 0,
                'unrealized_pl': float(pos.unrealized_pl) if pos.unrealized_pl else 0,
                'unrealized_plpc': float(pos.unrealized_plpc) if pos.unrealized_plpc else 0
            } for pos in positions]
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return []
    
    def is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an options contract"""
        return (
            len(symbol) > 10 and 
            ('C' in symbol[-9:] or 'P' in symbol[-9:]) and 
            any(char.isdigit() for char in symbol[-8:])
        ) or 'C00' in symbol or 'P00' in symbol
    
    def is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto pair"""
        return '/' in symbol or symbol.endswith('USD') and len(symbol) > 3
    
    def get_bars(self, symbol: str, timeframe: str = '1Day',
                start: Optional[datetime] = None,
                end: Optional[datetime] = None,
                limit: int = 1000) -> pd.DataFrame:
        """Universal method to get bars for any asset type"""
        if self.is_option_symbol(symbol):
            return self.get_option_bars(symbol, timeframe, start, end, limit)
        elif self.is_crypto_symbol(symbol):
            return self.get_crypto_bars(symbol, timeframe, start, end, limit)
        else:
            return self.get_stock_bars(symbol, timeframe, start, end, limit)
    
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Universal method to get latest quote for any asset type"""
        if self.is_option_symbol(symbol):
            return self.get_option_latest_quote(symbol)
        elif self.is_crypto_symbol(symbol):
            return self.get_crypto_latest_quote(symbol)
        else:
            return self.get_stock_latest_quote(symbol)
    
    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert timeframe string to alpaca-py TimeFrame enum"""
        timeframe_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, 'Minute'),
            '15Min': TimeFrame(15, 'Minute'),
            '30Min': TimeFrame(30, 'Minute'),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day,
            '1Week': TimeFrame.Week,
            '1Month': TimeFrame.Month
        }
        
        return timeframe_map.get(timeframe, TimeFrame.Day)
    
    def _process_stock_bars(self, bars_response, symbol: str) -> pd.DataFrame:
        """Process stock bars response into DataFrame"""
        if not bars_response or not hasattr(bars_response, 'data') or symbol not in bars_response.data:
            return pd.DataFrame()
        
        bars = bars_response.data[symbol]
        data = []
        
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'trade_count': int(bar.trade_count) if bar.trade_count else 0,
                'vwap': float(bar.vwap) if bar.vwap else 0
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
        
        return df
    
    def _process_option_bars(self, bars_response, symbol: str) -> pd.DataFrame:
        """Process option bars response into DataFrame"""
        if not bars_response or not hasattr(bars_response, 'data') or symbol not in bars_response.data:
            return pd.DataFrame()
        
        bars = bars_response.data[symbol]
        data = []
        
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'trade_count': int(bar.trade_count) if bar.trade_count else 0,
                'vwap': float(bar.vwap) if bar.vwap else 0
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
        
        return df
    
    def _process_crypto_bars(self, bars_response, symbol: str) -> pd.DataFrame:
        """Process crypto bars response into DataFrame"""
        if not bars_response or not hasattr(bars_response, 'data') or symbol not in bars_response.data:
            return pd.DataFrame()
        
        bars = bars_response.data[symbol]
        data = []
        
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume),
                'trade_count': int(bar.trade_count) if bar.trade_count else 0,
                'vwap': float(bar.vwap) if bar.vwap else 0
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
        
        return df
    
    def _process_options_chain(self, chain_response, underlying_symbol: str) -> pd.DataFrame:
        """Process options chain response into DataFrame"""
        data = []
        
        for option_data in chain_response:
            data.append({
                'symbol': option_data.symbol,
                'underlying_symbol': underlying_symbol,
                'option_type': 'C' if 'C' in option_data.symbol else 'P',
                'strike_price': getattr(option_data, 'strike_price', 0),
                'expiration_date': getattr(option_data, 'expiration_date', None),
                'bid_price': getattr(option_data, 'bid_price', 0),
                'ask_price': getattr(option_data, 'ask_price', 0),
                'last_price': getattr(option_data, 'last_price', 0),
                'volume': getattr(option_data, 'volume', 0)
            })
        
        return pd.DataFrame(data)
    
    def _create_synthetic_option_bar(self, symbol: str, end_time: datetime) -> pd.DataFrame:
        """Create synthetic option bar from latest quote when bars unavailable"""
        try:
            quote_data = self.get_option_latest_quote(symbol)
            
            if 'error' in quote_data:
                return pd.DataFrame()
            
            bid = quote_data.get('bid_price', 0)
            ask = quote_data.get('ask_price', 0)
            
            if bid <= 0 or ask <= 0:
                return pd.DataFrame()
            
            mid_price = (bid + ask) / 2.0
            
            synthetic_data = [{
                'timestamp': end_time.replace(hour=16, minute=0, second=0, microsecond=0),
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
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating synthetic option bar for {symbol}: {e}")
            return pd.DataFrame()


    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including VIP status and data feed info (legacy method)"""
        account_data = self.get_account()
        return {
            'vip': self.is_pro_tier,
            'using_iex': not self.is_pro_tier,  # Free tier uses IEX
            'account_equity': account_data.get('equity', 0),
            'account_status': account_data.get('status', 'unknown'),
            'pattern_day_trader': False,  # Would need to be fetched from trading client
            'default_feed': 'sip' if self.is_pro_tier else 'iex',
            'has_real_time_data': True,
            'data_delay': 'Real-time',
            'feed_description': 'Securities Information Processor (SIP)' if self.is_pro_tier else 'IEX Real-time + SIP Historical'
        }
    
    @property
    def vip(self) -> bool:
        """Legacy property for VIP status"""
        return self.is_pro_tier
    
    @property  
    def using_iex(self) -> bool:
        """Legacy property for IEX usage"""
        return not self.is_pro_tier


# Maintain backward compatibility
AlpacaDataProvider = AlpacaProvider