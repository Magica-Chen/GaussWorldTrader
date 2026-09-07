"""
Optimized configuration system for Python 3.12
Uses modern features like dataclasses, pattern matching, and performance optimizations
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, final

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()

DEFAULT_ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_ALPACA_BASE_URL = "https://api.alpaca.markets"

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
    
    @field_validator('max_concurrent_requests')
    @classmethod
    def validate_concurrent_requests(cls, v: int) -> int:
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
        '_alpaca_credentials', '_finnhub_credentials', '_fred_credentials',
        '_trading_limits', '_performance_config', '_database_url',
        '_log_level', '_config_file_path', '_last_reload'
    )
    
    def __init__(self, config_file: Path | None = None) -> None:
        self._config_file_path = config_file or Path("config.toml")
        self._last_reload: float = 0.0
        
        # Initialize from environment and config file
        self._load_configuration()
        
        version = f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
        logger.info(f"✅ Configuration loaded (Python {version})")
    
    def _load_configuration(self) -> None:
        """Load configuration from environment and files"""
        
        # Load from TOML config file if it exists
        config_data = {}
        if self._config_file_path.exists():
            with open(self._config_file_path, 'rb') as f:
                config_data = tomllib.load(f)
            logger.info(f"Loaded config from {self._config_file_path}")
        
        # API Credentials
        self._alpaca_credentials = APICredentials(
            api_key=os.getenv('ALPACA_API_KEY', ''),
            secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        self._finnhub_credentials = APICredentials(
            api_key=os.getenv('FINNHUB_API_KEY', ''),
            base_url='https://finnhub.io/api/v1'
        )
        
        self._fred_credentials = APICredentials(
            api_key=os.getenv('FRED_API_KEY', ''),
            base_url='https://api.stlouisfed.org/fred'
        )
        
        # Trading limits from config or environment
        limits_config = config_data.get('trading_limits', {})
        self._trading_limits = TradingLimits(
            max_position_size=float(
                os.getenv('MAX_POSITION_SIZE', limits_config.get('max_position_size', 0.1))
            ),
            max_daily_trades=int(
                os.getenv('MAX_DAILY_TRADES', limits_config.get('max_daily_trades', 50))
            ),
            max_open_positions=int(
                os.getenv('MAX_OPEN_POSITIONS', limits_config.get('max_open_positions', 10))
            ),
            stop_loss_pct=float(
                os.getenv('STOP_LOSS_PCT', limits_config.get('stop_loss_pct', 0.05))
            ),
            take_profit_pct=float(
                os.getenv('TAKE_PROFIT_PCT', limits_config.get('take_profit_pct', 0.15))
            )
        )
        
        # Performance configuration
        perf_config = config_data.get('performance', {})
        self._performance_config = PerformanceConfig(**perf_config)
        
        # Other settings
        database_config = config_data.get('database', {})
        logging_config = config_data.get('logging', {})
        self._database_url = os.getenv(
            'DATABASE_URL',
            database_config.get('url', 'sqlite:///trading_system.db'),
        )
        self._log_level = os.getenv(
            'LOG_LEVEL',
            logging_config.get('level', 'INFO'),
        ).upper()
        
        self._last_reload = datetime.now().timestamp()
    
    @property
    def alpaca(self) -> APICredentials:
        """Alpaca trading API credentials"""
        return self._alpaca_credentials
    
    @property
    def finnhub(self) -> APICredentials:
        """Finnhub news API credentials"""
        return self._finnhub_credentials
    
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
    def session_runtime(self):
        """Validated Gauss configuration from the same TOML and environment loader."""
        return get_gauss_config(self._config_file_path)
    
    @property
    def log_level(self) -> str:
        """Logging level"""
        return self._log_level
    
    def validate_all_credentials(self) -> dict[str, bool]:
        """Validate all API credentials"""
        return {
            'alpaca': self.alpaca.is_valid() and bool(self.alpaca.secret_key),
            'finnhub': self.finnhub.is_valid(),
            'fred': self.fred.is_valid()
        }
    
    def get_validation_summary(self) -> str:
        """Get human-readable validation summary"""
        validations = self.validate_all_credentials()
        
        status_emojis = {True: "✅", False: "❌"}
        lines = ["🔧 Configuration Status:"]
        
        for service, is_valid in validations.items():
            emoji = status_emojis[is_valid]
            status = 'Valid' if is_valid else 'Invalid/Missing'
            lines.append(f"  {emoji} {service.capitalize()}: {status}")
        
        return "\n".join(lines)
    
    def reload_if_changed(self, force: bool = False) -> bool:
        """Reload configuration if file has changed"""
        if force:
            self._load_configuration()
            logger.info("🔄 Configuration reloaded")
            return True

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
# ALPACA_API_KEY, ALPACA_SECRET_KEY, FINNHUB_API_KEY, FRED_API_KEY
'''
        
        output_path.write_text(template_content)
        logger.info(f"📄 Configuration template exported to {output_path}")

# Global configuration instance with lazy loading
_config_instance: OptimizedConfig | None = None

def get_gauss_config(config_file: str | Path | None = None):
    """Load the plan's [gauss] TOML contract; reject unknown or unsafe combinations.

    GAUSS_* environment keys override TOML. General trading ceilings apply as an
    additional bound; broker URLs never select the session's execution mode.
    """
    from decimal import Decimal
    from src.runtime.models import RuntimeConfig, RiskPolicy

    path = Path(config_file or os.getenv('GAUSS_CONFIG', 'config.toml'))
    if config_file is not None and not path.exists() and str(path) != 'config.toml':
        raise ValueError(f'Gauss configuration file does not exist: {path}')
    document = {}
    if path.exists():
        with path.open('rb') as stream:
            document = tomllib.load(stream)
    source = document.get('gauss', {})
    if not isinstance(source, dict):
        raise ValueError('[gauss] must be a TOML table')
    values, policy = {}, {}
    aliases = {
        'mode': 'execution_mode', 'account_environment': 'environment',
        'data.profile': 'data_profile', 'data.policy_version': 'data_policy_version',
        'storage.payload_directory': 'payload_directory',
        'storage.database_path': 'database_path',
        'account.id': 'account_id', 'news.event_calendar_path': 'event_calendar_path',
        'data.free_delayed.entitlement_boundary_buffer_seconds': 'entitlement_boundary_buffer',
        'data.free_delayed.poll_interval_seconds': 'free_poll_interval_seconds',
        'data.free_delayed.max_additional_lag_seconds': 'maximum_additional_lag_seconds',
        'data.subscribed_realtime.stock_execution_quote_max_age_seconds': 'stock_quote_max_age_seconds',
        'data.subscribed_realtime.option_execution_quote_max_age_seconds': 'option_quote_max_age_seconds',
        'universe.scan_universe_limit': 'scan_universe_limit',
        'universe.active_candidate_limit': 'active_candidate_limit',
        'universe.reserve_candidate_limit': 'reserve_candidate_limit',
        'universe.strategy_allowlist': 'strategy_allowlist',
        'schedule.close_research_offset_minutes': 'research_offset_minutes',
        'schedule.pre_review_interval_minutes': 'pre_review_interval_minutes',
        'schedule.signal_timeframe': 'signal_timeframe',
        'evaluation.capital_scenario_file': 'capital_scenario_file',
        'research.paid_model_calls_enabled': 'paid_model_calls_enabled',
        'research.max_parallel_jobs': 'research_max_parallel_jobs',
        'research.model': 'research_model', 'research.pricing_id': 'research_pricing_id',
        'research.max_output_tokens': 'research_max_output_tokens',
        'research.max_retries': 'research_max_retries',
    }
    risk_aliases = {
        'account.snapshot_max_age_seconds': 'account_max_age_seconds',
        'schedule.pre_start_minutes_before_open': 'pre_minutes',
        'schedule.entry_start_minutes_after_open': 'entry_start_minutes',
        'schedule.entry_stop_minutes_before_close': 'entry_cutoff_minutes',
        'schedule.close_attempt_minutes_before_close': 'closing_minutes',
        'research.max_job_seconds': 'research_seconds',
        'research.session_llm_budget_equity_fraction': 'research_equity_fraction',
        'research.session_llm_absolute_cap_usd': 'research_cost_cap',
        'risk.policy_id': 'id',
    }
    # These are fixed invariants, not switches an agent or a loose TOML key can disable.
    invariants = {
        'account.profile_source': 'broker_snapshot', 'account.require_currency_match': True,
        'data.allow_silent_feed_fallback': False,
        'data.require_endpoint_entitlement_checks': True,
        'data.free_delayed.market_delay_seconds': 900,
        'data.free_delayed.stock_history_feed': 'sip',
        'data.free_delayed.options_policy': 'verified_history_or_indicative_research',
        'data.free_delayed.allow_current_iex_in_signals': False,
        'data.subscribed_realtime.stock_feed': 'sip',
        'data.subscribed_realtime.option_feed': 'opra',
        'data.subscribed_realtime.intentional_market_delay_seconds': 0,
        'news.signal_alignment': 'profile_market_cutoff',
        'news.current_safety_alerts_enabled': True,
        'news.news_required_for_event_strategies': True,
        'suitability.required_before_plan_publication': True,
        'suitability.required_before_pre_validation': True,
        'suitability.required_before_entry': True,
        'suitability.allow_no_trade': True, 'suitability.allow_capital_band_shortcuts': False,
        'suitability.ranking_objective': 'validated_net_expectancy_subject_to_risk',
        'suitability.include_operating_costs': True, 'suitability.include_liquidity_capacity': True,
        'evaluation.require_explicit_scenario_capital': True,
        'evaluation.scenario_outputs_execution_eligible': False,
        'risk.allow_uncovered_options': False, 'risk.allow_expiry_day_entries': False,
        'risk.allow_free_delayed_live_entries': False,
    }
    def leaves(table, prefix=''):
        for key, value in table.items():
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                yield from leaves(value, name)
            else:
                yield name, value
    shares = {}
    for key, value in leaves(source):
        if key in invariants:
            if value != invariants[key] or isinstance(value, bool) != isinstance(invariants[key], bool):
                raise ValueError(f'gauss.{key} must be {invariants[key]!r}')
        elif key in aliases:
            values[aliases[key]] = value
        elif key in risk_aliases:
            policy[risk_aliases[key]] = value
        elif key == 'storage.database_url':
            if not isinstance(value, str) or not value.startswith('sqlite:///'):
                raise ValueError('Gauss storage requires a local sqlite:/// URL')
            values['database_path'] = value.removeprefix('sqlite:///')
        elif key.startswith(('risk.', 'policy.')) and key.split('.', 1)[1] in RiskPolicy.model_fields:
            policy[key.split('.', 1)[1]] = value
        elif key in {f'research.{role}_budget_share' for role in ('post', 'close', 'pre', 'live')}:
            shares[key.split('.')[1].split('_')[0]] = Decimal(str(value))
        elif key in RuntimeConfig.model_fields and key not in {'id', 'created_at', 'schema_version'}:
            values[key] = value
        else:
            raise ValueError(f'Unknown Gauss configuration key: {key}')
    if shares:
        defaults = dict(zip(('post', 'close', 'pre', 'live'), ('.10', '.60', '.20', '.10')))
        values['research_role_budget_shares'] = tuple(
            shares.get(role, Decimal(defaults[role])) for role in defaults
        )
    env_fields = {
        'GAUSS_ENABLED': 'enabled', 'GAUSS_ACCOUNT_ID': 'account_id',
        'GAUSS_ENVIRONMENT': 'environment', 'GAUSS_MODE': 'execution_mode',
        'GAUSS_DATA_PROFILE': 'data_profile', 'GAUSS_DATABASE_PATH': 'database_path',
        'GAUSS_LIVE_TRADING_ENABLED': 'live_trading_enabled',
        'GAUSS_EVENT_CALENDAR': 'event_calendar_path', 'GAUSS_POLL_SECONDS': 'poll_seconds',
    }
    for env_key, key in env_fields.items():
        if env_key in os.environ:
            values[key] = os.environ[env_key]
    for env_key, key in (('GAUSS_ACCOUNT_ID_ALLOWLIST', 'account_id_allowlist'),
                         ('GAUSS_STRATEGY_ALLOWLIST', 'strategy_allowlist'),
                         ('GAUSS_SYMBOLS', 'symbols')):
        if env_key in os.environ:
            values[key] = tuple(item.strip() for item in os.environ[env_key].split(',') if item.strip())
    limits = document.get('trading_limits', {})
    ceilings = (
        ('max_capital_allocation_pct', 'max_position_size', 'MAX_POSITION_SIZE', Decimal('.1')),
        ('max_new_entry_groups_per_session', 'max_daily_trades', 'MAX_DAILY_TRADES', 50),
        ('max_open_position_groups', 'max_open_positions', 'MAX_OPEN_POSITIONS', 10),
    )
    defaults = RiskPolicy()
    for key, legacy, env_key, default in ceilings:
        ceiling = Decimal(str(os.getenv(env_key, limits.get(legacy, default))))
        if not ceiling.is_finite() or ceiling < 0:
            raise ValueError(f'Invalid legacy trading ceiling: {legacy}')
        strict = min(Decimal(str(policy.get(key, getattr(defaults, key)))), ceiling)
        policy[key] = strict if key.endswith('_pct') else int(strict)
    values['policy'] = RiskPolicy.model_validate(policy)
    return RuntimeConfig.model_validate(values)

def get_config() -> OptimizedConfig:
    """Get global configuration instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = OptimizedConfig()
    return _config_instance


def has_alpaca_credentials() -> bool:
    """Return whether Alpaca credentials are configured."""
    return get_config().validate_all_credentials()["alpaca"]


def get_alpaca_base_url() -> str:
    """Return the configured Alpaca base URL with a safe default."""
    return get_config().alpaca.base_url or DEFAULT_ALPACA_BASE_URL


def is_paper_trading() -> bool:
    """Return whether the configured base URL targets Alpaca paper trading."""
    return get_alpaca_base_url() != LIVE_ALPACA_BASE_URL

def reload_config(force: bool = False) -> bool:
    """Reload global configuration"""
    return get_config().reload_if_changed(force)


__all__ = [
    "APICredentials",
    "TradingLimits",
    "PerformanceConfig",
    "OptimizedConfig",
    "DEFAULT_ALPACA_BASE_URL",
    "LIVE_ALPACA_BASE_URL",
    "get_config",
    "get_gauss_config",
    "has_alpaca_credentials",
    "get_alpaca_base_url",
    "is_paper_trading",
    "reload_config",
]

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
