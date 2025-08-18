"""
Optimized configuration system for Python 3.12
Uses modern features like dataclasses, pattern matching, and performance optimizations
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, final
from collections.abc import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from loguru import logger

# Load environment variables
load_dotenv()

@final
@dataclass(frozen=True, slots=True)
class APICredentials:
    """Immutable API credentials with slots for memory efficiency"""
    api_key: str
    secret_key: str | None = None
    base_url: str | None = None
    
    def is_valid(self) -> bool:
        """Check if credentials are valid"""
        return bool(self.api_key and len(self.api_key.strip()) > 10)

@final  
@dataclass(frozen=True, slots=True)
class TradingLimits:
    """Trading risk limits with validation"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_trades: int = 50
    max_open_positions: int = 10
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.15  # 15%
    
    def __post_init__(self) -> None:
        """Validate limits after initialization"""
        if not (0 < self.max_position_size <= 1):
            raise ValueError("max_position_size must be between 0 and 1")
        if self.max_daily_trades <= 0:
            raise ValueError("max_daily_trades must be positive")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")

class PerformanceConfig(BaseModel):
    """Performance configuration using Pydantic for validation"""
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    cache_ttl_seconds: int = Field(default=30, ge=1, le=3600)
    batch_size: int = Field(default=50, ge=1, le=1000)
    connection_pool_size: int = Field(default=20, ge=5, le=100)
    request_timeout: float = Field(default=30.0, ge=1.0, le=120.0)
    
    @validator('max_concurrent_requests')
    def validate_concurrent_requests(cls, v):
        """Ensure reasonable concurrency limits"""
        if v > 50:
            logger.warning(f"High concurrency ({v}) may cause rate limiting")
        return v

@final
class OptimizedConfig:
    """
    High-performance configuration system for Python 3.12
    Uses caching, slots, and modern Python features
    """
    
    __slots__ = (
        '_alpaca_credentials', '_finhub_credentials', '_fred_credentials',
        '_trading_limits', '_performance_config', '_database_url',
        '_log_level', '_config_file_path', '_last_reload'
    )
    
    def __init__(self, config_file: Path | None = None) -> None:
        self._config_file_path = config_file or Path("config.toml")
        self._last_reload: float = 0.0
        
        # Initialize from environment and config file
        self._load_configuration()
        
        logger.info(f"✅ Configuration loaded (Python {os.sys.version_info.major}.{os.sys.version_info.minor})")
    
    def _load_configuration(self) -> None:
        """Load configuration from environment and files"""
        
        # Load from TOML config file if it exists
        config_data = {}
        if self._config_file_path.exists():
            try:
                with open(self._config_file_path, 'rb') as f:
                    config_data = tomllib.load(f)
                logger.info(f"📄 Loaded config from {self._config_file_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # API Credentials
        self._alpaca_credentials = APICredentials(
            api_key=os.getenv('ALPACA_API_KEY', ''),
            secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        self._finhub_credentials = APICredentials(
            api_key=os.getenv('FINHUB_API_KEY', ''),
            base_url='https://finnhub.io/api/v1'
        )
        
        self._fred_credentials = APICredentials(
            api_key=os.getenv('FRED_API_KEY', ''),
            base_url='https://api.stlouisfed.org/fred'
        )
        
        # Trading limits from config or environment
        limits_config = config_data.get('trading_limits', {})
        self._trading_limits = TradingLimits(
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', limits_config.get('max_position_size', 0.1))),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', limits_config.get('max_daily_trades', 50))),
            max_open_positions=int(os.getenv('MAX_OPEN_POSITIONS', limits_config.get('max_open_positions', 10))),
            stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', limits_config.get('stop_loss_pct', 0.05))),
            take_profit_pct=float(os.getenv('TAKE_PROFIT_PCT', limits_config.get('take_profit_pct', 0.15)))
        )
        
        # Performance configuration
        perf_config = config_data.get('performance', {})
        self._performance_config = PerformanceConfig(**perf_config)
        
        # Other settings
        self._database_url = os.getenv('DATABASE_URL', 'sqlite:///trading_system.db')
        self._log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        self._last_reload = datetime.now().timestamp()
    
    @property
    def alpaca(self) -> APICredentials:
        """Alpaca trading API credentials"""
        return self._alpaca_credentials
    
    @property
    def finhub(self) -> APICredentials:
        """Finhub news API credentials"""
        return self._finhub_credentials
    
    @property
    def fred(self) -> APICredentials:
        """FRED economic data API credentials"""
        return self._fred_credentials
    
    @property
    def trading_limits(self) -> TradingLimits:
        """Trading risk limits"""
        return self._trading_limits
    
    @property
    def performance(self) -> PerformanceConfig:
        """Performance configuration"""
        return self._performance_config
    
    @property
    def database_url(self) -> str:
        """Database connection URL"""
        return self._database_url
    
    @property
    def log_level(self) -> str:
        """Logging level"""
        return self._log_level
    
    def validate_all_credentials(self) -> dict[str, bool]:
        """Validate all API credentials"""
        return {
            'alpaca': self.alpaca.is_valid() and bool(self.alpaca.secret_key),
            'finhub': self.finhub.is_valid(),
            'fred': self.fred.is_valid()
        }
    
    def get_validation_summary(self) -> str:
        """Get human-readable validation summary"""
        validations = self.validate_all_credentials()
        
        status_emojis = {True: "✅", False: "❌"}
        lines = ["🔧 Configuration Status:"]
        
        for service, is_valid in validations.items():
            emoji = status_emojis[is_valid]
            lines.append(f"  {emoji} {service.capitalize()}: {'Valid' if is_valid else 'Invalid/Missing'}")
        
        return "\n".join(lines)
    
    def reload_if_changed(self, force: bool = False) -> bool:
        """Reload configuration if file has changed"""
        if not self._config_file_path.exists():
            return False
        
        file_mtime = self._config_file_path.stat().st_mtime
        
        if force or file_mtime > self._last_reload:
            # Reload configuration (properties will automatically use new values)
            self._load_configuration()
            logger.info("🔄 Configuration reloaded")
            return True
        
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """Export configuration to dictionary (for debugging)"""
        validations = self.validate_all_credentials()
        
        return {
            'credentials_status': validations,
            'trading_limits': {
                'max_position_size': self.trading_limits.max_position_size,
                'max_daily_trades': self.trading_limits.max_daily_trades,
                'max_open_positions': self.trading_limits.max_open_positions,
                'stop_loss_pct': self.trading_limits.stop_loss_pct,
                'take_profit_pct': self.trading_limits.take_profit_pct
            },
            'performance': self.performance.dict(),
            'database_url': self.database_url,
            'log_level': self.log_level,
            'config_file': str(self._config_file_path),
            'last_reload': datetime.fromtimestamp(self._last_reload).isoformat()
        }
    
    def export_template(self, output_path: Path) -> None:
        """Export a configuration template file"""
        template_content = '''# Trading System Configuration (Python 3.12+)
# This file uses TOML format for better type safety and readability

[trading_limits]
max_position_size = 0.1      # Maximum position size as fraction of portfolio (10%)
max_daily_trades = 50        # Maximum trades per day
max_open_positions = 10      # Maximum concurrent positions
stop_loss_pct = 0.05        # Stop loss percentage (5%)
take_profit_pct = 0.15      # Take profit percentage (15%)

[performance]
max_concurrent_requests = 10     # Maximum concurrent API requests
cache_ttl_seconds = 30          # Cache time-to-live in seconds  
batch_size = 50                 # Batch size for bulk operations
connection_pool_size = 20       # HTTP connection pool size
request_timeout = 30.0          # Request timeout in seconds

[logging]
level = "INFO"                  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
format = "{time} | {level} | {name}:{function}:{line} | {message}"

[database]
url = "sqlite:///trading_system.db"  # Database connection URL

# Environment variables still take precedence for sensitive data:
# ALPACA_API_KEY, ALPACA_SECRET_KEY, FINHUB_API_KEY, FRED_API_KEY
'''
        
        output_path.write_text(template_content)
        logger.info(f"📄 Configuration template exported to {output_path}")

# Global configuration instance with lazy loading
_config_instance: OptimizedConfig | None = None

def get_config() -> OptimizedConfig:
    """Get global configuration instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = OptimizedConfig()
    return _config_instance

def reload_config(force: bool = False) -> bool:
    """Reload global configuration"""
    return get_config().reload_if_changed(force)

# Legacy compatibility (for existing code)
class Config:
    """Legacy config class for backward compatibility"""
    
    @staticmethod
    def validate_alpaca_config() -> bool:
        return get_config().validate_all_credentials()['alpaca']
    
    @staticmethod
    def validate_finhub_config() -> bool:
        return get_config().validate_all_credentials()['finhub']
    
    @staticmethod
    def validate_fred_config() -> bool:
        return get_config().validate_all_credentials()['fred']
    
    # Dynamic property access for backward compatibility
    def __getattr__(self, name: str) -> Any:
        config = get_config()
        
        # Map legacy attribute names to new structure
        attribute_mapping = {
            'ALPACA_API_KEY': lambda: config.alpaca.api_key,
            'ALPACA_SECRET_KEY': lambda: config.alpaca.secret_key,
            'ALPACA_BASE_URL': lambda: config.alpaca.base_url,
            'FINHUB_API_KEY': lambda: config.finhub.api_key,
            'FRED_API_KEY': lambda: config.fred.api_key,
            'DATABASE_URL': lambda: config.database_url,
            'LOG_LEVEL': lambda: config.log_level
        }
        
        if name in attribute_mapping:
            return attribute_mapping[name]()
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# Example usage and testing
if __name__ == '__main__':
    # Example of using the optimized config
    config = get_config()
    
    print(config.get_validation_summary())
    print(f"\nTrading limits: {config.trading_limits}")
    print(f"Performance config: {config.performance.dict()}")
    
    # Export template
    template_path = Path("config_template.toml")
    config.export_template(template_path)
    print(f"\nTemplate exported to {template_path}")