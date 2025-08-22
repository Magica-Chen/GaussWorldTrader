# 🌍 Gauss World Trader

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/issues)

A high-performance, **Python 3.12+ optimized** quantitative trading system featuring modern async operations, rich CLI interfaces, and advanced financial analysis capabilities. Named after Carl Friedrich Gauss, the mathematical genius who revolutionized statistics and probability theory - foundations of modern quantitative finance.

**🔗 Repository**: [https://github.com/Magica-Chen/GaussWorldTrader](https://github.com/Magica-Chen/GaussWorldTrader)  
**👤 Author**: Magica Chen  
**📊 Version**: 2.0.0 - AI-Powered Trading with Advanced Portfolio Management

## ✨ Features

### 🏎️ Python 3.12+ Optimizations
- **Modern Type Hints**: Uses `dict[str, Any]` and `list[Type]` syntax
- **Exception Groups**: Advanced error handling with `except*` syntax
- **Async Performance**: Optimized asyncio patterns and concurrency
- **Memory Efficiency**: Dataclasses with `slots=True` for reduced memory usage
- **Pattern Matching**: Uses `match/case` statements for clean logic
- **Cached Properties**: `@cached_property` for expensive computations

### 📊 Trading & Data Features  
- **Smart Data Feeds**: Intelligent subscription detection with SIP historical + IEX real-time data
- **Real-time Trading**: Integration with Alpaca Markets API for stocks and options
- **Subscription Management**: Automatic Pro/Free tier detection with appropriate data feeds
- **Watchlist System**: JSON-based watchlist management with CLI and dashboard editing
- **Cryptocurrency Data**: Real-time crypto prices via CoinDesk and CoinGecko APIs  
- **News & Sentiment**: Financial news and sentiment analysis via Finhub API
- **Macro Economics**: Economic indicators via FRED API
- **Backtesting Framework**: Comprehensive backtesting with performance metrics
- **Technical Analysis**: Full suite of technical indicators and pattern recognition
- **Strategy Framework**: Modular strategy system with momentum strategy example

### 🤖 AI-Powered Analysis (NEW v2.0)
- **Multi-LLM Integration**: Support for OpenAI, DeepSeek, Claude, and Moonshot APIs
- **Fundamental Analysis**: AI-powered company analysis with financial ratios
- **Economic Context**: Integration with FRED economic data for macro analysis
- **News Sentiment**: Intelligent news analysis and sentiment scoring
- **Comparative Analysis**: Multi-provider AI analysis for comprehensive insights

### 📈 Advanced Portfolio Management (NEW v2.0)
- **Real-time Positions**: Live position tracking with P&L analysis
- **Order Management**: Complete order lifecycle management and analysis
- **Portfolio Analytics**: Advanced performance metrics and risk assessment
- **Account Configuration**: Full Alpaca account settings management
- **Risk Metrics**: Value at Risk, concentration analysis, and drawdown tracking

### 🖥️ Modern Interfaces
- **Rich CLI**: Beautiful command-line interface with progress bars and tables
- **Advanced Dashboard**: Modern Streamlit web dashboard with tabbed interface
- **Async Operations**: Non-blocking data fetching and order execution
- **Performance Monitoring**: Real-time metrics and error tracking

### 🎯 Advanced Trading Strategies (NEW v2.0)
- **Classical Strategies**: Momentum, Scalping, Statistical Arbitrage, Trend Following, Value Investment
- **ML-Based Strategies**: XGBoost, Deep Learning (LSTM/CNN/Attention), Gaussian Process
- **Strategy Selector**: Intelligent strategy recommendation based on market conditions
- **Backtesting Engine**: Comprehensive strategy backtesting with performance comparison
- **Portfolio Management**: Multi-strategy portfolio optimization and risk management  
- **Configuration Management**: TOML-based config with validation
- **Risk Management**: Built-in position sizing and risk controls

## Project Structure

```
GaussWorldTrader/
├── src/
│   ├── account/           # 🆕 Account management and portfolio tracking
│   ├── agent/             # 🆕 AI-powered analysis with multi-LLM support
│   ├── analysis/          # Technical analysis and financial metrics
│   ├── backtest/          # 🆕 Enhanced backtesting framework (moved from trade)
│   ├── data/              # Data providers (Alpaca, Crypto, News, Macro)  
│   ├── strategy/          # Trading strategy implementations
│   ├── trade/             # Trading engine and execution
│   ├── ui/                # User interfaces (CLI and dashboard)
│   └── utils/             # Utility functions and validators
├── config/                # Configuration management
├── examples/              # Example scripts and usage
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🛠️ Installation

### Prerequisites
- **Python 3.12+** (Required for optimal performance)
- Git for version control

### Quick Setup

1. **Verify Python version**:
   ```bash
   python --version  # Should show 3.12+
   ```
2. **Clone the repository**:
   ```bash
   git clone https://github.com/Magica-Chen/GaussWorldTrader.git
   cd GaussWorldTrader
   ```
3. **Create a virtual environment**:
   ```bash
   conda create -n gaussworldtrader python=3.12
   conda activate gaussworldtrader
   ```

4. **Install dependencies** (Python 3.12 optimized):
   ```bash
   # Option 1: Using pip
   pip install -r requirements.txt
   
   # Option 2: Using the modern pyproject.toml
   pip install -e .
   
   # Option 3: Development installation with all extras
   pip install -e ".[dev,performance,all]"
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

6. **Generate configuration template** (optional):
   ```bash
   python -c "from config.optimized_config import get_config; get_config().export_template(Path('config.toml'))"
   ```

### 🔑 Required API Keys
- **Alpaca**: Get free paper trading keys from [alpaca.markets](https://alpaca.markets)
  - **Free Tier**: Access to SIP historical data + IEX real-time data for today
  - **Pro Tier**: Full SIP real-time data access
- **Finhub**: Get free API key from [finnhub.io](https://finnhub.io)
- **FRED**: Get free API key from [fred.stlouisfed.org](https://fred.stlouisfed.org)

### 🤖 Optional AI Provider API Keys (NEW v2.0)
- **OpenAI**: Get API key from [platform.openai.com](https://platform.openai.com)
- **DeepSeek**: Get API key from [platform.deepseek.com](https://platform.deepseek.com)
- **Claude (Anthropic)**: Get API key from [console.anthropic.com](https://console.anthropic.com)
- **Moonshot**: Get API key from [platform.moonshot.cn](https://platform.moonshot.cn)

### 🚀 Performance Optimizations
The system includes Python 3.12 specific optimizations:
- Faster startup times with optimized imports
- Reduced memory usage with `slots=True` dataclasses  
- Improved async performance with updated asyncio
- Better error handling with exception groups

## 🚀 Quick Start

### Modern CLI Interface (Python 3.12 Powered)

#### 🆕 Quick CLI Reference for v2.0 Features

```bash
# Portfolio Management Commands
python main.py check-positions          # Check positions and recent orders  
python main.py watchlist-trade          # Analyze watchlist and execute trades

# Using the new modules programmatically
python -c "
from src.account import AccountManager
manager = AccountManager()
print(manager.get_account_summary())
"

# AI Analysis (requires API keys)
python -c "
from src.agent import AgentManager
agent = AgentManager()
analysis = agent.analyze_symbol('AAPL', 'openai')
print(analysis)
"
```

```bash
# 🆕 Rich, beautiful CLI with async operations

# Account Operations
python main.py account info --refresh                    # Show account information
python main.py account performance --days 30             # Performance metrics & analysis

# Data Operations  
python main.py data fetch AAPL GOOGL MSFT --days 30      # Fetch historical market data
python main.py data stream AAPL GOOGL --interval 5       # Live data streaming

# Technical Analysis
python main.py analysis technical AAPL --indicators rsi macd bb sma --days 100

# Trading Operations
python main.py trade place AAPL buy 100 --type market --dry-run

# Strategy Operations
python main.py strategy run AAPL GOOGL TSLA --strategy momentum --watch

# 🆕 NEW v2.0 Commands - Advanced Portfolio Management
python main.py check-positions                              # Check positions and recent orders
python main.py watchlist-trade                              # Analyze watchlist and execute trades

# Get help for any command
python main.py --help
python main.py account --help

# 🌟 NEW Advanced Dashboard (v2.0)
python run_dashboard.py                                    # Launch advanced web dashboard
# Opens in browser at http://localhost:3721
python main.py analysis technical --help
```

## 🌟 Advanced Web Dashboard

The new Advanced Dashboard provides a comprehensive web-based interface for trading management, analysis, and strategy execution.

### 🚀 Getting Started with Dashboard

```bash
# Install dashboard dependencies
pip install -r requirements_dashboard.txt

# Test dashboard functionality
python test_dashboard.py

# Launch the dashboard
python run_dashboard.py
```

The dashboard will open automatically at **http://localhost:3721**

### 📊 Dashboard Features

#### 💼 Account Tab
- **Portfolio Overview**: Real-time account metrics, equity, buying power, and P&L
- **Position Management**: Detailed position tracking with real-time P&L and risk metrics
- **Order Management**: Active orders, order history, and new order placement
- **Account Configuration**: Trading permissions, risk settings, and account preferences
- **P&L Analysis**: Interactive charts showing portfolio performance over time

#### 📈 Live Analysis Tab
- **Current Positions Analysis**: Technical and fundamental analysis for your active positions
- **Symbol Analysis**: Comprehensive analysis for any stock symbol
- **Market Overview**: Market indices, sentiment indicators, and sector performance
- **Watchlist Management**: Track and analyze your favorite symbols

#### 🧪 Backtesting Tab
- **Position Backtesting**: Test strategies on your current positions
- **Strategy Comparison**: Compare performance of multiple trading strategies
- **Custom Backtesting**: Advanced backtesting with custom parameters
- **Results Analysis**: Detailed performance metrics and visualizations

#### ⚡ Trading Tab
- **Quick Trade**: Fast order execution with risk management tools
- **Strategy Trading**: Execute trades based on algorithmic strategies
- **Options Trading**: Options chain analysis and trading (coming soon)
- **Trade History**: Complete trading history with performance analysis

### 🎯 8 Built-in Trading Strategies

The dashboard integrates all advanced trading strategies:

#### Classical Strategies
1. **Momentum Strategy**: RSI and volume-based momentum trading
2. **Scalping Strategy**: High-frequency VWAP-based scalping
3. **Statistical Arbitrage**: Mean reversion and correlation trading
4. **Trend Following**: Multi-timeframe trend analysis
5. **Value Investment**: Fundamental analysis-based value investing

#### ML-Based Strategies  
6. **XGBoost Strategy**: Gradient boosting with feature engineering
7. **Deep Learning Strategy**: LSTM, CNN, and Attention ensemble models
8. **Gaussian Process Strategy**: Bayesian approach with uncertainty quantification

### 🔧 Dashboard Architecture

- **Modern UI**: Clean Streamlit interface with responsive design
- **Smart Data Feeds**: Automatic subscription detection and optimal data routing
- **Real-time Data**: Live market data with subscription-aware messaging
- **Interactive Charts**: Advanced Plotly visualizations
- **Risk Management**: Built-in position sizing and risk controls
- **Strategy Integration**: Seamless integration with all trading strategies
- **Export Capabilities**: Download reports and analysis results
- **Watchlist Management**: JSON-based watchlist with real-time editing

### ⚡ CLI Features

#### 💼 Account Management
- **Account Info**: Real-time portfolio value, cash, equity, and buying power
- **Performance Metrics**: Returns, volatility, Sharpe ratio, and risk analysis
- **Trading Activity**: Recent orders and transaction history
- **Position Tracking**: Current holdings and allocation analysis

#### 📊 Data Operations
- **Historical Data**: Fetch OHLCV data for any timeframe
- **Live Streaming**: Real-time price monitoring with change tracking
- **Multiple Symbols**: Concurrent data fetching for portfolio analysis
- **Export Options**: Save data to CSV for external analysis

#### 📈 Technical Analysis
- **RSI**: Relative Strength Index for overbought/oversold signals
- **MACD**: Moving Average Convergence Divergence with crossover detection
- **Bollinger Bands**: Volatility bands for support/resistance analysis
- **Moving Averages**: SMA trends for momentum identification
- **Visual Signals**: Color-coded BUY/SELL/HOLD recommendations

#### 💰 Trading Operations
- **Order Placement**: Market and limit orders with confirmation
- **Risk Management**: Dry-run mode for testing strategies
- **Position Sizing**: Automated position calculation based on risk
- **Order Validation**: Pre-trade validation and error checking

#### 🧠 Strategy Framework
- **Momentum Strategy**: Built-in momentum-based trading algorithm
- **Strategy Runner**: Live execution with monitoring and alerts
- **Backtesting**: Historical performance testing and metrics
- **Watch Mode**: Continuous monitoring with real-time updates

### ⚡ Performance Features
- **Concurrent Operations**: Fetch multiple symbols simultaneously
- **Intelligent Caching**: Reduce API calls with smart caching
- **Progress Indicators**: Beautiful progress bars for long operations
- **Error Recovery**: Graceful handling with detailed error messages
- **Configuration Validation**: Automatic API credential validation

## 🆕 NEW v2.0 Features Usage

### 🤖 AI-Powered Analysis

The new AI agent system provides comprehensive fundamental analysis using multiple LLM providers:

```python
from src.agent import AgentManager, FundamentalAnalyzer

# Initialize AI agent manager
agent_manager = AgentManager({
    'finnhub_api_key': 'your_finnhub_key',
    'fred_api_key': 'your_fred_key'
})

# Analyze a company using OpenAI
analysis = agent_manager.analyze_symbol('AAPL', provider='openai')

# Compare analysis from multiple AI providers
comparison = agent_manager.compare_providers('AAPL', ['openai', 'claude', 'deepseek'])

# Generate comprehensive report
analyzer = FundamentalAnalyzer()
company_analysis = analyzer.analyze_company('AAPL')
report = analyzer.generate_report(company_analysis)
print(report)
```

#### Supported AI Providers:
- **OpenAI**: GPT-4 for comprehensive financial analysis
- **DeepSeek**: Quantitative analysis with mathematical rigor
- **Claude**: Balanced, nuanced investment insights
- **Moonshot**: Bilingual analysis with Chinese market perspective

#### AI Analysis Features:
- Company profile and financial metrics
- News sentiment analysis
- Economic context integration
- Analyst recommendations synthesis
- Risk assessment and valuation

### 📈 Advanced Portfolio Management

Complete account and portfolio management with real-time tracking:

```python
from src.account import AccountManager, PositionManager, OrderManager, PortfolioTracker

# Initialize account management
account_manager = AccountManager()
position_manager = PositionManager(account_manager)
order_manager = OrderManager(account_manager)
portfolio_tracker = PortfolioTracker(account_manager)

# Account operations
account_summary = account_manager.get_account_summary()
trading_status = account_manager.get_trading_account_status()

# Position management
all_positions = position_manager.get_all_positions()
positions_analysis = position_manager.analyze_positions()
position_details = position_manager.get_position_details('AAPL')

# Order management
recent_orders = order_manager.get_recent_orders_summary()
order_analysis = order_manager.analyze_orders(days=30)

# Place new orders
order_result = order_manager.place_order(
    symbol='AAPL',
    qty=100,
    side='buy',
    order_type='market'
)

# Portfolio analytics
portfolio_performance = portfolio_tracker.get_portfolio_performance('1D')
asset_allocation = portfolio_tracker.get_asset_allocation()
risk_metrics = portfolio_tracker.calculate_risk_metrics()
portfolio_report = portfolio_tracker.generate_portfolio_report()
```

#### Portfolio Features:
- Real-time position tracking with P&L
- Order history and analysis
- Advanced performance metrics
- Risk assessment and concentration analysis
- Asset allocation visualization
- Account configuration management

### 🎯 Watchlist Trading

Automated watchlist analysis and strategy-based trading:

```python
# From command line
python main.py watchlist-trade

# The system will:
# 1. Display current watchlist symbols
# 2. Check market status and account buying power
# 3. Analyze each symbol using momentum strategy
# 4. Generate trading signals
# 5. Ask for confirmation before executing trades
# 6. Place orders and provide execution feedback
```

#### Watchlist Features:
- Pre-configured symbol watchlist (SPY, QQQ, AAPL, etc.)
- Real-time market status checking
- Strategy-based signal generation
- Risk-aware position sizing
- Confirmation prompts for safety
- Execution tracking and reporting

### 📊 Enhanced Backtesting

Moved to dedicated module with advanced analytics:

```python
from src.backtest import Backtester, PerformanceAnalyzer

# Enhanced backtesting
backtester = Backtester(initial_cash=100000)
backtester.add_data('AAPL', historical_data)

# Run backtest with strategy
results = backtester.run_backtest(strategy.generate_signals)

# Advanced performance analysis
analyzer = PerformanceAnalyzer(results)
advanced_metrics = analyzer.calculate_advanced_metrics()
performance_report = analyzer.generate_performance_report()

# Generate visualization charts
analyzer.plot_performance_charts('performance_charts.png')
```

#### Enhanced Metrics:
- Value at Risk (VaR) and Conditional VaR
- Sortino and Calmar ratios
- Rolling Sharpe ratio analysis
- Skewness and kurtosis analysis
- Tail ratio and risk distribution
- Interactive performance charts

### 🔧 Account Configuration

Full Alpaca account settings management:

```python
from src.account import AccountConfigurator

configurator = AccountConfigurator(account_manager)

# View current configurations
current_config = configurator.get_account_configurations()
config_summary = configurator.get_configuration_summary()

# Enable/disable features
configurator.enable_extended_hours_trading()
configurator.enable_fractional_trading()
configurator.set_pdt_check('BOTH')
configurator.set_margin_multiplier(2.0)

# Apply preset configurations
configurator.apply_conservative_settings()  # Safe settings
configurator.apply_aggressive_settings()    # Advanced trading
```

#### Configuration Options:
- Trading hours (standard/extended)
- Fractional share trading
- Pattern Day Trader (PDT) checks
- Margin multiplier settings
- Short selling permissions
- Trade confirmation emails

### Web Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard.py

# Or using Python directly
python dashboard.py
```

### Example Backtest

```bash
# Run the momentum strategy example
python examples/momentum_backtest_example.py
```

## Strategy Framework

### Base Strategy Class

All strategies inherit from `BaseStrategy` and implement:

- `generate_signals()`: Generate buy/sell signals
- `calculate_position_size()`: Determine position sizes
- Built-in technical indicators and risk management

### Momentum Strategy Example

The included momentum strategy uses:
- **Price momentum**: Identifies trending stocks
- **RSI filtering**: Avoids overbought/oversold conditions  
- **Volume confirmation**: Requires above-average volume
- **Risk management**: Stop loss and take profit levels

```python
from src.strategy import MomentumStrategy

# Create strategy with custom parameters
strategy = MomentumStrategy({
    'lookback_period': 20,
    'rsi_period': 14,
    'position_size_pct': 0.1,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.15
})
```

## Data Providers

### Alpaca Markets
- Real-time and historical stock data
- Options chains
- Account and position information
- Order execution

### Cryptocurrency  
- Bitcoin prices via CoinDesk
- Historical crypto data via CoinGecko
- Multiple currency pairs

### News & Sentiment
- Company-specific news via Finhub
- Market news and sentiment scores
- News search and filtering

### Economic Data
- Fed economic data via FRED API
- GDP, unemployment, inflation rates
- Treasury yields and market indicators

## Backtesting Framework

🆕 **Enhanced in v2.0** - Moved to dedicated module with advanced analytics:

- **Performance Metrics**: Returns, Sharpe ratio, max drawdown
- **Advanced Risk Analysis**: VaR, CVaR, Sortino ratio, Calmar ratio
- **Trade Analysis**: Win rate, profit factor, trade distribution
- **Portfolio Tracking**: Full position and cash flow history
- **Rolling Metrics**: Rolling Sharpe, volatility analysis
- **Visualization**: Performance charts and drawdown analysis

```python
from src.backtest import Backtester, PerformanceAnalyzer
from src.strategy import MomentumStrategy

backtester = Backtester(initial_cash=100000)
backtester.add_data('AAPL', historical_data)

strategy = MomentumStrategy()
results = backtester.run_backtest(strategy.generate_signals)

# Basic results
print(backtester.get_results_summary())

# Advanced analysis
analyzer = PerformanceAnalyzer(results)
advanced_metrics = analyzer.calculate_advanced_metrics()
performance_report = analyzer.generate_performance_report()
analyzer.plot_performance_charts('charts.png')
```

## API Configuration

### Environment Variables

```bash
# Alpaca Trading (Paper Trading)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret  
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Financial Data
FINHUB_API_KEY=your_finhub_key
FRED_API_KEY=your_fred_key

# 🆕 AI Provider APIs (Optional - for fundamental analysis)
OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key
ANTHROPIC_API_KEY=your_claude_key
MOONSHOT_API_KEY=your_moonshot_key

# Optional
DATABASE_URL=sqlite:///trading_system.db
LOG_LEVEL=INFO
```

## Safety & Risk Management

**⚠️ IMPORTANT SAFETY NOTES:**

1. **Paper Trading Default**: System defaults to paper trading mode
2. **Risk Controls**: Built-in position sizing and stop losses
3. **API Validation**: All API credentials are validated before use
4. **No Live Trading**: Live trading requires explicit configuration changes

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Adding New Strategies

1. Inherit from `BaseStrategy`
2. Implement required methods
3. Add to strategy module `__init__.py`
4. Create example usage

```python
from src.strategy.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio):
        # Your strategy logic here
        return signals
    
    def calculate_position_size(self, symbol, price, portfolio_value, volatility=None):
        # Position sizing logic
        return size
```

## Performance Optimization

- **Vectorized Operations**: Uses pandas/numpy for fast calculations
- **Caching**: API responses cached to reduce calls
- **Async Support**: Async data fetching for multiple symbols
- **Memory Management**: Efficient data structures for large datasets

## Troubleshooting

### Common Issues

1. **API Key Errors**: Verify keys in `.env` file
2. **Data Issues**: Check internet connection and API status
3. **Module Import Errors**: Ensure project root in Python path
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Logging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py --help
```

## 🤝 Contributing

We welcome contributions to Gauss World Trader! Here's how to get started:

1. **Fork the repository**: Visit [https://github.com/Magica-Chen/GaussWorldTrader](https://github.com/Magica-Chen/GaussWorldTrader)
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests for new functionality**: Ensure your code is well-tested
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Submit a pull request**: Open a PR with a clear description

### 📋 Issue Reporting
- **Bug Reports**: [Report bugs here](https://github.com/Magica-Chen/GaussWorldTrader/issues)
- **Feature Requests**: [Suggest new features](https://github.com/Magica-Chen/GaussWorldTrader/issues)
- **Questions**: [Ask questions in Discussions](https://github.com/Magica-Chen/GaussWorldTrader/discussions)

## License

This project is licensed under the MIT License - see LICENSE file for details.

## 📈 Changelog

### v2.1.0 - Smart Data Feeds & Subscription Management (Latest)
**🆕 Major New Features:**
- **Smart Subscription Detection**: Automatic Pro/Free tier detection using SIP feed testing
- **Optimized Data Feeds**: SIP historical + IEX real-time for Free tier, full SIP for Pro tier
- **JSON Watchlist System**: Flexible watchlist management with CLI and dashboard editing
- **Subscription-Aware UI**: Context-aware messaging showing data sources and real-time status
- **Enhanced Strategy Execution**: Improved get_watchlists_and_trade with strategy parameters

**🔧 Technical Improvements:**
- Simplified subscription status checking with single function approach
- Removed unused imports and functions for cleaner codebase
- Enhanced error handling for subscription status detection
- Improved timezone handling with consistent ET time for trading logic
- Better data source routing based on subscription tier

**🎯 Data Feed Optimizations:**
- Free Tier: SIP historical data + IEX real-time data for current day
- Pro Tier: Full SIP real-time data access
- No more delayed data - all tiers get real-time information

### v2.0.0 - AI-Powered Trading with Advanced Portfolio Management
**🆕 Major New Features:**
- **AI Agent System**: Multi-LLM support (OpenAI, DeepSeek, Claude, Moonshot)
- **Advanced Portfolio Management**: Real-time tracking, analytics, and risk assessment
- **Enhanced Backtesting**: Dedicated module with advanced performance metrics
- **Account Configuration**: Full Alpaca account settings management
- **Watchlist Trading**: Automated analysis and strategy-based execution
- **Position Management**: Comprehensive position tracking and analysis
- **Order Management**: Complete order lifecycle management

**🔧 Technical Improvements:**
- Modular architecture with dedicated account, agent, and backtest modules
- Improved timezone handling for US market hours
- Enhanced error handling and validation
- Better API integration and data sources
- Advanced risk metrics and analysis

**🎯 New CLI Commands:**
- `check-positions`: Quick portfolio status check
- `watchlist-trade`: Automated watchlist analysis and trading

### v1.1.0 - Complete CLI with Technical Analysis
- Rich CLI interface with technical indicators
- Modern Python 3.12+ optimizations
- Streamlit dashboard
- Basic trading and backtesting framework

## Disclaimer

**This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Use at your own risk.**