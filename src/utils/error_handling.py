"""
Modern error handling using Python 3.12 features
Includes exception groups, improved error messages, and structured error handling
"""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from types import TracebackType

class ErrorSeverity(Enum):
    """Error severity levels for better categorization"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(frozen=True)
class TradingError:
    """Structured error information for trading operations"""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    context: dict[str, Any]
    traceback_info: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/storage"""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'traceback': self.traceback_info
        }

class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **context) -> None:
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context
        self.timestamp = datetime.now()

class DataProviderError(TradingSystemError):
    """Errors related to data providers"""
    pass

class TradingEngineError(TradingSystemError):
    """Errors related to trading engine operations"""
    pass

class StrategyError(TradingSystemError):
    """Errors related to strategy execution"""
    pass

class RiskManagementError(TradingSystemError):
    """Errors related to risk management"""
    pass

class ErrorHandler:
    """
    Centralized error handler using Python 3.12 exception groups
    and improved error handling patterns
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.error_history: list[TradingError] = []
        
    def handle_exception_group(self, exc_group: BaseExceptionGroup) -> None:
        """
        Handle exception groups (Python 3.11+ feature, optimized for 3.12)
        """
        errors = []
        
        for exc in exc_group.exceptions:
            if isinstance(exc, TradingSystemError):
                trading_error = TradingError(
                    error_type=type(exc).__name__,
                    message=exc.message,
                    severity=exc.severity,
                    timestamp=exc.timestamp,
                    context=exc.context,
                    traceback_info=traceback.format_exc()
                )
                errors.append(trading_error)
                self._log_error(trading_error)
            else:
                # Handle non-trading system errors
                generic_error = TradingError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    severity=ErrorSeverity.HIGH,
                    timestamp=datetime.now(),
                    context={},
                    traceback_info=traceback.format_exc()
                )
                errors.append(generic_error)
                self._log_error(generic_error)
        
        self.error_history.extend(errors)
    
    def handle_single_exception(self, exc: Exception, context: dict[str, Any] | None = None) -> TradingError:
        """Handle a single exception with context"""
        context = context or {}
        
        if isinstance(exc, TradingSystemError):
            trading_error = TradingError(
                error_type=type(exc).__name__,
                message=exc.message,
                severity=exc.severity,
                timestamp=exc.timestamp,
                context={**exc.context, **context},
                traceback_info=traceback.format_exc()
            )
        else:
            trading_error = TradingError(
                error_type=type(exc).__name__,
                message=str(exc),
                severity=ErrorSeverity.MEDIUM,
                timestamp=datetime.now(),
                context=context,
                traceback_info=traceback.format_exc()
            )
        
        self._log_error(trading_error)
        self.error_history.append(trading_error)
        return trading_error
    
    def _log_error(self, error: TradingError) -> None:
        """Log error with appropriate level based on severity"""
        log_message = f"{error.error_type}: {error.message}"
        if error.context:
            log_message += f" | Context: {error.context}"
        
        match error.severity:
            case ErrorSeverity.LOW:
                self.logger.info(log_message)
            case ErrorSeverity.MEDIUM:
                self.logger.warning(log_message)
            case ErrorSeverity.HIGH:
                self.logger.error(log_message)
            case ErrorSeverity.CRITICAL:
                self.logger.critical(log_message)
    
    def get_error_summary(self, hours_back: int = 24) -> dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp >= cutoff_time
        ]
        
        # Count by severity and type
        severity_counts = {}
        type_counts = {}
        
        for error in recent_errors:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            type_counts[error.error_type] = type_counts.get(error.error_type, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'severity_breakdown': severity_counts,
            'error_type_breakdown': type_counts,
            'time_period_hours': hours_back
        }
    
    @contextmanager
    def error_context(self, operation: str, **context):
        """
        Context manager for operations that may raise exceptions
        Uses Python 3.12's improved exception handling
        """
        try:
            yield
        except Exception as e:
            # Handle all exceptions (including groups)
            enhanced_context = {
                'operation': operation,
                **context
            }
            
            # Check if it's an exception group
            if isinstance(e, BaseExceptionGroup):
                self.logger.error(f"Trading system errors in operation '{operation}'")
                self.handle_exception_group(e)
            else:
                # Handle single exceptions
                self.handle_single_exception(e, enhanced_context)
            raise

# Global error handler instance
error_handler = ErrorHandler()

def safe_execute(func: callable, *args, **kwargs) -> tuple[Any, TradingError | None]:
    """
    Safely execute a function and return result with any error
    Uses Python 3.12's improved exception handling
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error = error_handler.handle_single_exception(e, {'function': func.__name__})
        return None, error

async def safe_execute_async(func: callable, *args, **kwargs) -> tuple[Any, TradingError | None]:
    """Async version of safe_execute"""
    try:
        result = await func(*args, **kwargs)
        return result, None
    except Exception as e:
        error = error_handler.handle_single_exception(e, {'async_function': func.__name__})
        return None, error

# Example usage with Python 3.12 pattern matching
def handle_trading_operation_result(result: Any, error: TradingError | None) -> bool:
    """Handle the result of a trading operation using pattern matching"""
    match (result, error):
        case (None, TradingError(severity=ErrorSeverity.CRITICAL)):
            # Critical error - stop all operations
            logging.critical("Critical error occurred - halting operations")
            return False
        case (None, TradingError(severity=ErrorSeverity.HIGH)):
            # High severity - retry with caution
            logging.error("High severity error - implementing fallback")
            return False
        case (None, TradingError()):
            # Other errors - log and continue
            logging.warning("Operation failed but continuing")
            return True
        case (result, None):
            # Success
            logging.info("Operation completed successfully")
            return True
        case _:
            # Unexpected case
            logging.warning("Unexpected result/error combination")
            return True

# Decorator for automatic error handling
def with_error_handling(operation_name: str | None = None):
    """Decorator to add automatic error handling to functions"""
    def decorator(func: callable):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with error_handler.error_context(op_name, function=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator