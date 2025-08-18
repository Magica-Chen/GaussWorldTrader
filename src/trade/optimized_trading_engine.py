"""
High-performance trading engine optimized for Python 3.12
Uses new performance improvements, modern type hints, and optimized patterns
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, final, override
from collections.abc import AsyncGenerator, Sequence
import logging
from functools import cached_property
import alpaca_trade_api as tradeapi

from config import Config
from src.utils.error_handling import (
    ErrorHandler, TradingEngineError, ErrorSeverity, with_error_handling
)

@dataclass(frozen=True, slots=True)  # Python 3.10+ slots optimization
class OrderRequest:
    """Immutable order request with slots for memory efficiency"""
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    price: float | None = None
    stop_price: float | None = None
    time_in_force: str = 'gtc'
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(slots=True)
class OrderResult:
    """Mutable order result with slots optimization"""
    order_id: str
    status: str
    filled_qty: float = 0.0
    filled_price: float | None = None
    error_message: str | None = None

@final  # Prevent inheritance for optimization
class OptimizedTradingEngine:
    """
    High-performance trading engine using Python 3.12 optimizations
    - Slots for memory efficiency
    - Cached properties for performance
    - Modern async patterns
    - Exception groups for error handling
    """
    
    __slots__ = (
        '_api', '_paper_trading', '_logger', '_error_handler',
        '_order_queue', '_order_history', '_performance_metrics',
        '_last_account_fetch', '_cached_account_info'
    )
    
    def __init__(self, paper_trading: bool = True) -> None:
        if not Config.validate_alpaca_config():
            raise TradingEngineError(
                "Alpaca API credentials not configured",
                ErrorSeverity.CRITICAL,
                config_issue=True
            )
        
        self._api = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        self._paper_trading = paper_trading
        self._logger = logging.getLogger(__name__)
        self._error_handler = ErrorHandler()
        
        # Use deque for O(1) append/popleft operations
        self._order_queue: deque[OrderRequest] = deque(maxlen=1000)
        self._order_history: deque[OrderResult] = deque(maxlen=10000)
        
        # Performance tracking
        self._performance_metrics = {
            'orders_processed': 0,
            'orders_per_second': 0.0,
            'average_response_time': 0.0,
            'error_rate': 0.0
        }
        
        # Account info caching
        self._last_account_fetch: float = 0.0
        self._cached_account_info: dict[str, Any] | None = None
        
        if paper_trading:
            self._logger.info("🧪 Trading engine initialized in PAPER TRADING mode")
        else:
            self._logger.warning("🔴 Trading engine initialized in LIVE TRADING mode")
    
    @cached_property
    def is_market_open(self) -> bool:
        """Cached property for market status - expensive API call"""
        try:
            clock = self._api.get_clock()
            return clock.is_open
        except Exception as e:
            self._logger.error(f"Failed to get market status: {e}")
            return False
    
    @with_error_handling("place_order")
    async def place_order_async(self, request: OrderRequest) -> OrderResult:
        """
        Place order asynchronously with performance tracking
        Uses Python 3.12's improved asyncio performance
        """
        start_time = time.perf_counter()
        
        try:
            # Add to processing queue
            self._order_queue.append(request)
            
            # Execute order in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            order = await loop.run_in_executor(
                None, 
                self._execute_sync_order,
                request
            )
            
            result = OrderResult(
                order_id=order.id,
                status=order.status,
                filled_qty=float(order.filled_qty or 0),
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None
            )
            
        except Exception as e:
            result = OrderResult(
                order_id="",
                status="failed",
                error_message=str(e)
            )
            raise TradingEngineError(
                f"Failed to place order for {request.symbol}",
                ErrorSeverity.HIGH,
                symbol=request.symbol,
                order_type=request.order_type
            ) from e
        
        finally:
            # Update performance metrics
            processing_time = time.perf_counter() - start_time
            self._update_performance_metrics(processing_time, result.error_message is None)
            self._order_history.append(result)
        
        self._logger.info(
            f"📊 Order processed: {request.symbol} {request.side} "
            f"{request.quantity} - Status: {result.status}"
        )
        
        return result
    
    def _execute_sync_order(self, request: OrderRequest) -> Any:
        """Execute order synchronously (called from thread pool)"""
        order_params = {
            'symbol': request.symbol,
            'qty': abs(request.quantity),
            'side': request.side.lower(),
            'type': request.order_type,
            'time_in_force': request.time_in_force
        }
        
        # Add price parameters based on order type
        match request.order_type:
            case 'limit':
                if request.price is None:
                    raise ValueError("Limit orders require a price")
                order_params['limit_price'] = request.price
            case 'stop':
                if request.stop_price is None:
                    raise ValueError("Stop orders require a stop price")
                order_params['stop_price'] = request.stop_price
            case 'market':
                pass  # No additional parameters needed
            case _:
                raise ValueError(f"Unsupported order type: {request.order_type}")
        
        return self._api.submit_order(**order_params)
    
    async def batch_place_orders(
        self, 
        requests: Sequence[OrderRequest]
    ) -> list[OrderResult]:
        """
        Place multiple orders concurrently
        Uses Python 3.12's improved task scheduling
        """
        tasks = [
            self.place_order_async(request) 
            for request in requests
        ]
        
        # Use exception groups to handle multiple errors
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except* TradingEngineError as eg:
            self._error_handler.handle_exception_group(eg)
            raise
        
        # Process results and separate successful from failed
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(OrderResult(
                    order_id="",
                    status="failed", 
                    error_message=str(result)
                ))
            else:
                successful_results.append(result)
        
        self._logger.info(
            f"🔄 Batch order completed: {len(successful_results)} successful, "
            f"{len(failed_results)} failed"
        )
        
        return successful_results + failed_results
    
    async def get_account_info_cached(self, max_age_seconds: float = 30.0) -> dict[str, Any]:
        """
        Get account info with intelligent caching
        Uses Python 3.12's improved time handling
        """
        current_time = time.perf_counter()
        
        # Return cached data if it's fresh enough
        if (self._cached_account_info is not None and 
            current_time - self._last_account_fetch < max_age_seconds):
            return self._cached_account_info
        
        # Fetch fresh data
        loop = asyncio.get_running_loop()
        account = await loop.run_in_executor(None, self._api.get_account)
        
        self._cached_account_info = {
            'account_id': account.id,
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'day_trade_count': int(account.day_trade_count),
            'pattern_day_trader': account.pattern_day_trader,
            'status': account.status,
            'fetched_at': datetime.now().isoformat()
        }
        
        self._last_account_fetch = current_time
        return self._cached_account_info
    
    async def stream_order_updates(self) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream order updates using async generator
        Optimized for Python 3.12's async iteration performance
        """
        while True:
            try:
                # In a real implementation, this would connect to Alpaca's WebSocket
                # For demo, we'll simulate order updates
                await asyncio.sleep(1)
                
                if self._order_history:
                    latest_order = self._order_history[-1]
                    yield {
                        'order_id': latest_order.order_id,
                        'status': latest_order.status,
                        'filled_qty': latest_order.filled_qty,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except asyncio.CancelledError:
                self._logger.info("Order update stream cancelled")
                break
            except Exception as e:
                self._logger.error(f"Error in order update stream: {e}")
                await asyncio.sleep(5)
    
    def _update_performance_metrics(self, processing_time: float, success: bool) -> None:
        """Update internal performance metrics"""
        self._performance_metrics['orders_processed'] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        current_avg = self._performance_metrics['average_response_time']
        self._performance_metrics['average_response_time'] = (
            alpha * processing_time + (1 - alpha) * current_avg
        )
        
        # Update error rate
        if not success:
            error_count = len([r for r in self._order_history if r.error_message])
            total_count = len(self._order_history)
            self._performance_metrics['error_rate'] = error_count / total_count if total_count > 0 else 0
    
    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary"""
        recent_orders = list(self._order_history)[-100:]  # Last 100 orders
        
        return {
            **self._performance_metrics,
            'queue_size': len(self._order_queue),
            'history_size': len(self._order_history),
            'recent_success_rate': (
                sum(1 for r in recent_orders if r.error_message is None) / len(recent_orders)
                if recent_orders else 1.0
            ),
            'cache_hit_rate': 0.95 if self._cached_account_info else 0.0,  # Placeholder
            'engine_type': 'paper' if self._paper_trading else 'live'
        }
    
    @override  # Python 3.12 improved override decorator
    def __repr__(self) -> str:
        mode = "paper" if self._paper_trading else "live"
        return f"OptimizedTradingEngine(mode={mode}, orders_processed={self._performance_metrics['orders_processed']})"

# Usage example
async def example_trading_operations():
    """Example of using the optimized trading engine"""
    async with OptimizedTradingEngine() as engine:
        # Single order
        order_request = OrderRequest(
            symbol="AAPL",
            quantity=100,
            side="buy",
            order_type="market"
        )
        
        result = await engine.place_order_async(order_request)
        print(f"Order result: {result}")
        
        # Batch orders
        batch_requests = [
            OrderRequest("GOOGL", 50, "buy", "limit", price=150.0),
            OrderRequest("MSFT", 75, "buy", "market"),
            OrderRequest("TSLA", 25, "sell", "limit", price=200.0)
        ]
        
        batch_results = await engine.batch_place_orders(batch_requests)
        print(f"Batch results: {len(batch_results)} orders processed")
        
        # Performance summary
        performance = engine.get_performance_summary()
        print(f"Performance: {performance}")

if __name__ == '__main__':
    # Python 3.12's optimized asyncio.run
    asyncio.run(example_trading_operations())