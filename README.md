# 🌍 Gauss World Trader

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/issues)

A high-performance, **Python 3.12+ optimized** quantitative trading system featuring modern async operations, rich CLI interfaces, and advanced financial analysis capabilities. Named after Carl Friedrich Gauss, the mathematical genius who revolutionized statistics and probability theory - foundations of modern quantitative finance.

**🔗 Repository**: [https://github.com/Magica-Chen/GaussWorldTrader](https://github.com/Magica-Chen/GaussWorldTrader)  
**👤 Author**: Magica Chen  
**📊 Version**: 1.1.0 - Complete CLI with Technical Analysis

## ✨ Features

### 🏎️ Python 3.12+ Optimizations
- **Modern Type Hints**: Uses `dict[str, Any]` and `list[Type]` syntax
- **Exception Groups**: Advanced error handling with `except*` syntax
- **Async Performance**: Optimized asyncio patterns and concurrency
- **Memory Efficiency**: Dataclasses with `slots=True` for reduced memory usage
- **Pattern Matching**: Uses `match/case` statements for clean logic
- **Cached Properties**: `@cached_property` for expensive computations

### 📊 Trading & Data Features  
- **Real-time Trading**: Integration with Alpaca Markets API for stocks and options
- **Cryptocurrency Data**: Real-time crypto prices via CoinDesk and CoinGecko APIs  
- **News & Sentiment**: Financial news and sentiment analysis via Finhub API
- **Macro Economics**: Economic indicators via FRED API
- **Backtesting Framework**: Comprehensive backtesting with performance metrics
- **Technical Analysis**: Full suite of technical indicators and pattern recognition
- **Strategy Framework**: Modular strategy system with momentum strategy example

### 🖥️ Modern Interfaces
- **Rich CLI**: Beautiful command-line interface with progress bars and tables
- **Async Operations**: Non-blocking data fetching and order execution
- **Performance Monitoring**: Real-time metrics and error tracking  
- **Configuration Management**: TOML-based config with validation
- **Risk Management**: Built-in position sizing and risk controls

## Project Structure

```
GaussWorldTrader/
├── src/
│   ├── analysis/           # Technical analysis and financial metrics
│   ├── data/              # Data providers (Alpaca, Crypto, News, Macro)  
│   ├── strategy/          # Trading strategy implementations
│   ├── trade/             # Trading engine, portfolio, and backtesting
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

3. **Install dependencies** (Python 3.12 optimized):
   ```bash
   # Option 1: Using pip
   pip install -r requirements.txt
   
   # Option 2: Using the modern pyproject.toml
   pip install -e .
   
   # Option 3: Development installation with all extras
   pip install -e ".[dev,performance,all]"
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

5. **Generate configuration template** (optional):
   ```bash
   python -c "from config.optimized_config import get_config; get_config().export_template(Path('config.toml'))"
   ```

### 🔑 Required API Keys
- **Alpaca**: Get free paper trading keys from [alpaca.markets](https://alpaca.markets)
- **Finhub**: Get free API key from [finnhub.io](https://finnhub.io)
- **FRED**: Get free API key from [fred.stlouisfed.org](https://fred.stlouisfed.org)

### 🚀 Performance Optimizations
The system includes Python 3.12 specific optimizations:
- Faster startup times with optimized imports
- Reduced memory usage with `slots=True` dataclasses  
- Improved async performance with updated asyncio
- Better error handling with exception groups

## 🚀 Quick Start

### Modern CLI Interface (Python 3.12 Powered)

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

# Get help for any command
python main.py --help
python main.py account --help
python main.py analysis technical --help
```

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

Comprehensive backtesting with:
- **Performance Metrics**: Returns, Sharpe ratio, max drawdown
- **Risk Analysis**: VaR, volatility, correlation analysis
- **Trade Analysis**: Win rate, profit factor, trade distribution
- **Portfolio Tracking**: Full position and cash flow history

```python
from src.trade import Backtester
from src.strategy import MomentumStrategy

backtester = Backtester(initial_cash=100000)
backtester.add_data('AAPL', historical_data)

strategy = MomentumStrategy()
results = backtester.run_backtest(strategy.generate_signals)

print(backtester.get_results_summary())
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

## Disclaimer

**This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Use at your own risk.**