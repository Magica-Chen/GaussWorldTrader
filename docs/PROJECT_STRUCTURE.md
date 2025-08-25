# Gauss World Trader - Project Structure

## Overview

This document describes the organized project structure following clean architecture principles with high cohesion and low coupling.

## Directory Structure

```
GaussWorldTrader/
├── main.py                 # Main entry point (CLI interface selection)
├── dashboard.py            # Dashboard launcher (Web UI interface)
├── CLAUDE.md              # Code style guidelines
├── README.md              # Project documentation
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Dependencies
├── watchlist.json         # Default watchlist configuration
│
├── config/                # Configuration management
│   ├── __init__.py
│   ├── config.py          # Legacy configuration (backward compatibility)
│   └── optimized_config.py # Modern Python 3.12+ configuration
│
├── src/                   # Main source code
│   ├── __init__.py
│   │
│   ├── account/           # Account and portfolio management
│   │   ├── __init__.py
│   │   ├── account_config.py      # Account configuration
│   │   ├── account_manager.py     # Main account management
│   │   ├── order_manager.py       # Order handling
│   │   ├── portfolio_tracker.py   # Portfolio tracking
│   │   └── position_manager.py    # Position management
│   │
│   ├── agent/             # AI/ML agents and analysis
│   │   ├── __init__.py
│   │   ├── agent_manager.py       # Agent coordination
│   │   ├── data_sources.py        # [REMOVED] Data source management
│   │   ├── fundamental_analyzer.py # Fundamental analysis
│   │   └── llm_providers.py       # LLM integration
│   │
│   ├── analysis/          # Financial analysis tools
│   │   ├── __init__.py
│   │   ├── financial_metrics.py   # Financial calculations
│   │   ├── technical_analysis.py  # Technical indicators
│   │   └── performance_analyzer.py # Performance metrics & analysis
│   │
│   ├── data/              # Data providers and handlers
│   │   ├── __init__.py
│   │   ├── alpaca_provider.py     # Unified data provider (stocks, options, crypto)
│   │   ├── macro_provider.py      # Macroeconomic data
│   │   └── news_provider.py       # News and sentiment data
│   │
│   ├── strategy/          # Trading strategies
│   │   ├── __init__.py
│   │   ├── base_strategy.py       # Base strategy interface
│   │   ├── strategy_selector.py   # Strategy selection system
│   │   ├── momentum_strategy.py   # Momentum-based trading
│   │   ├── arbitrage_strategy.py  # Arbitrage opportunities
│   │   ├── scalping_strategy.py   # High-frequency scalping
│   │   ├── trend_following_strategy.py # Trend following
│   │   ├── value_strategy.py      # Value investing
│   │   ├── deep_learning_strategy.py # Neural networks
│   │   ├── gaussian_process_strategy.py # Gaussian processes
│   │   └── xgboost_strategy.py    # XGBoost ML strategy
│   │
│   ├── trade/             # Trading execution and backtesting
│   │   ├── __init__.py
│   │   ├── backtester.py          # Backtesting engine
│   │   ├── portfolio.py           # Portfolio management
│   │   ├── trading_engine.py      # Basic trading engine
│   │   └── optimized_trading_engine.py # High-performance engine
│   │
│   ├── ui/                # User interfaces
│   │   ├── __init__.py
│   │   ├── core_cli.py            # Base CLI abstraction (shared functionality)
│   │   ├── dashboard.py           # Dashboard launcher
│   │   ├── simple_dashboard.py    # Basic Streamlit dashboard
│   │   ├── advanced_dashboard.py  # Advanced Streamlit dashboard
│   │   ├── modern_dashboard.py    # Unified modern dashboard
│   │   ├── simple_cli.py          # Basic CLI interface (uses core_cli)
│   │   ├── modern_cli.py          # Modern CLI with Rich (uses core_cli, primary)
│   │   └── portfolio_commands.py  # Portfolio CLI commands
│   │
│   └── utils/             # Shared utilities
│       ├── __init__.py
│       ├── dashboard_utils.py     # Shared dashboard functions
│       ├── error_handling.py     # Error management
│       ├── logger.py             # Logging configuration
│       ├── timezone_utils.py     # Timezone handling
│       ├── validators.py         # Data validation
│       └── watchlist_manager.py  # Watchlist management
│
├── examples/              # Example scripts and tutorials
│   ├── README.md
│   ├── simple_example.py         # Basic usage example
│   ├── momentum_backtest_example.py # Momentum strategy demo
│   ├── advanced_strategies_example.py # All strategies demo
│   └── run_backtest_example.py   # CLI backtest example
│
├── tests/                 # Test suite
│   └── test_dashboard.py         # Dashboard tests
│
├── results/               # Generated results and outputs
│   ├── backtest_results_*.png    # Chart outputs
│   └── transactions_*.csv        # Transaction logs
│
└── docs/                  # Documentation
    └── PROJECT_STRUCTURE.md      # This file
```

## Architecture Principles

### 1. High Cohesion, Low Coupling
- **Modules**: Each directory contains related functionality
- **Interfaces**: Clear boundaries between components
- **Dependencies**: Minimal cross-module dependencies

### 2. Separation of Concerns
- **Data Layer**: `src/data/` - External data sources
- **Business Logic**: `src/strategy/`, `src/trade/` - Core trading logic
- **Presentation**: `src/ui/` - User interfaces
- **Infrastructure**: `src/utils/`, `config/` - Supporting services

### 3. Code Reuse and DRY
- **Shared Utilities**: Common functions in `src/utils/`
- **Base Classes**: Abstract interfaces in `src/strategy/base_strategy.py`
- **Configuration**: Centralized in `config/`

## Module Responsibilities

### Core Trading Modules
- **`src/strategy/`**: Trading algorithms and decision-making logic
- **`src/trade/`**: Order execution, backtesting, and portfolio management
- **`src/data/`**: Market data acquisition and processing

### Supporting Modules
- **`src/account/`**: Account state management and position tracking
- **`src/analysis/`**: Financial calculations, technical indicators, and performance analysis
- **`src/agent/`**: AI-powered analysis and decision support

### Interface Modules
- **`src/ui/`**: All user interface implementations
  - **`core_cli.py`**: Base CLI abstraction providing shared functionality
  - **`simple_cli.py`**: Basic CLI interface inheriting from core
  - **`modern_cli.py`**: Advanced CLI interface using core utilities
- **`src/utils/`**: Shared functionality across modules

### CLI Architecture (New)
The CLI system now uses an abstract base class to eliminate code duplication:

```
core_cli.py (BaseCLI class)
├── Shared command implementations (account info, config validation, etc.)
├── Common error handling and display utilities
├── Base table creation and formatting functions
└── Abstract methods for custom command setup

simple_cli.py (SimpleCLI class inheriting from BaseCLI)
├── Basic command registration
├── Simple momentum strategy runner
└── Fallback mode for systems without rich/typer

modern_cli.py (Uses BaseCLI utilities)
├── Advanced async operations
├── Sub-command organization 
├── Progress bars and live displays
└── Comprehensive trading features
```

### Entry Point Selection (New)
The `main.py` entry point now supports CLI interface selection:

```bash
# Use modern CLI (default)
python main.py [commands]

# Explicitly choose CLI interface
python main.py --cli modern [commands]    # Rich CLI with sub-commands
python main.py --cli simple [commands]    # Simple flat command structure

# Examples
python main.py account info               # Modern: sub-command syntax
python main.py --cli simple account-info  # Simple: flat command syntax
```

**Benefits:**
- **Modern CLI**: Rich interface, sub-commands, async operations, advanced features
- **Simple CLI**: Lightweight, flat commands, basic functionality, fallback compatibility
- **Unified Entry**: Single entry point with automatic CLI selection and argument forwarding

## Import Guidelines

### Correct Import Patterns
```python
# Trading functionality
from src.trade import Backtester, Portfolio
from src.strategy import MomentumStrategy

# Data providers
from src.data import AlpacaDataProvider

# Utilities
from src.utils import timezone_utils, dashboard_utils
```

### Updated Import Patterns
```python
# Performance analysis now in analysis module
from src.analysis import PerformanceAnalyzer

# Backtesting functionality in trade module
from src.trade import Backtester, Portfolio
```

## File Organization Rules

1. **Entry Points**: Root level (`main.py`)
2. **Examples**: `examples/` directory with README
3. **Source Code**: `src/` with logical module separation
4. **Configuration**: `config/` directory
5. **Tests**: `tests/` directory
6. **Results**: `results/` for generated outputs
7. **Documentation**: `docs/` for project documentation

## Benefits of This Structure

1. **Maintainability**: Clear module boundaries make changes easier
2. **Testability**: Isolated components are easier to test
3. **Scalability**: New features fit into existing structure
4. **Code Reuse**: Shared utilities eliminate duplication
5. **Team Collaboration**: Clear ownership of different modules

This structure follows modern Python project conventions and supports the project's growth while maintaining code quality and organization.