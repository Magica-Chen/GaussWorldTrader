"""
Async data provider optimized for Python 3.12
Uses modern async/await patterns and type hints
"""
from __future__ import annotations

import asyncio
import aiohttp
import aiofiles
from datetime import datetime, timedelta
import pandas as pd
from typing import Any, AsyncGenerator
from collections.abc import Sequence
import logging

from config import Config

class AsyncDataProvider:
    """High-performance async data provider for Python 3.12+"""
    
    def __init__(self, max_concurrent: int = 10) -> None:
        self.session: aiohttp.ClientSession | None = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self) -> AsyncDataProvider:
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Trading-System-Python-3.12'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_multiple_symbols(
        self, 
        symbols: Sequence[str],
        timeframe: str = '1Day',
        days_back: int = 30
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently
        Uses Python 3.12's improved asyncio performance
        """
        tasks = [
            self._fetch_symbol_data(symbol, timeframe, days_back) 
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to fetch {symbol}: {result}")
                data_dict[symbol] = pd.DataFrame()
            else:
                data_dict[symbol] = result
                
        return data_dict
    
    async def _fetch_symbol_data(
        self,
        symbol: str,
        timeframe: str,
        days_back: int
    ) -> pd.DataFrame:
        """Fetch data for a single symbol with rate limiting"""
        async with self.semaphore:
            if not self.session:
                raise RuntimeError("Session not initialized - use async context manager")
            
            # Simulate API call - replace with actual Alpaca async API
            await asyncio.sleep(0.1)  # Rate limiting
            
            # Mock data - replace with real API call
            dates = pd.date_range(
                end=datetime.now(),
                periods=days_back,
                freq='D'
            )
            
            # Generate mock OHLCV data
            import numpy as np
            np.random.seed(hash(symbol) % 2**32)
            base_price = 100
            
            data = {
                'open': base_price + np.random.randn(len(dates)).cumsum(),
                'high': base_price + np.random.randn(len(dates)).cumsum() + 2,
                'low': base_price + np.random.randn(len(dates)).cumsum() - 2,
                'close': base_price + np.random.randn(len(dates)).cumsum(),
                'volume': np.random.randint(1000, 10000, len(dates))
            }
            
            df = pd.DataFrame(data, index=dates)
            df['high'] = df[['open', 'close', 'high']].max(axis=1)
            df['low'] = df[['open', 'close', 'low']].min(axis=1)
            
            return df
    
    async def stream_market_data(
        self, 
        symbols: Sequence[str]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream real-time market data using async generator
        Python 3.12 optimized async iteration
        """
        while True:
            try:
                # Simulate streaming data
                for symbol in symbols:
                    price_data = {
                        'symbol': symbol,
                        'price': 100 + (hash(symbol + str(datetime.now())) % 100),
                        'timestamp': datetime.now().isoformat(),
                        'volume': (hash(symbol) % 10000) + 1000
                    }
                    yield price_data
                
                await asyncio.sleep(1)  # Stream delay
                
            except asyncio.CancelledError:
                self.logger.info("Market data stream cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in market data stream: {e}")
                await asyncio.sleep(5)  # Error recovery delay
    
    async def batch_process_data(
        self,
        data: dict[str, pd.DataFrame],
        processing_func: callable
    ) -> dict[str, Any]:
        """
        Process data in batches using async tasks
        Optimized for Python 3.12's improved task performance
        """
        async def process_symbol(symbol: str, df: pd.DataFrame) -> tuple[str, Any]:
            # Run CPU-bound work in thread pool for better concurrency
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, processing_func, df)
            return symbol, result
        
        tasks = [
            process_symbol(symbol, df)
            for symbol, df in data.items()
            if not df.empty
        ]
        
        results = await asyncio.gather(*tasks)
        return dict(results)

# Usage example function
async def example_usage():
    """Example of using the async data provider"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    async with AsyncDataProvider() as provider:
        # Fetch multiple symbols concurrently
        data = await provider.fetch_multiple_symbols(symbols, days_back=100)
        
        print(f"Fetched data for {len(data)} symbols")
        for symbol, df in data.items():
            print(f"{symbol}: {len(df)} rows")
        
        # Stream market data for a short time
        async for market_update in provider.stream_market_data(['AAPL', 'GOOGL']):
            print(f"Market update: {market_update}")
            break  # Just show one update for example

if __name__ == '__main__':
    # Python 3.12's improved asyncio.run performance
    asyncio.run(example_usage())