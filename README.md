# 🌍 Gauss World Trader

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/issues)

A high-performance, **Python 3.12+ optimized** quantitative trading system featuring modern async operations, intelligent data feeds, and advanced portfolio management. Named after Carl Friedrich Gauss, who revolutionized statistics and probability theory - the foundations of modern quantitative finance.

**🔗 Repository**: [https://github.com/Magica-Chen/GaussWorldTrader](https://github.com/Magica-Chen/GaussWorldTrader)  
**👤 Author**: Magica Chen  
**📊 Version**: 1.1.0 - API Optimization & Crypto Integration

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Magica-Chen/GaussWorldTrader.git
cd GaussWorldTrader

# Create environment (Python 3.12+ required)
conda create -n gaussworldtrader python=3.12
conda activate gaussworldtrader

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
cp .env.template .env
# Edit .env with your Alpaca API keys
```

### 3. Launch Dashboard (Web UI)
```bash
# Modern unified dashboard (default)
python dashboard.py

# Or choose specific dashboard
python dashboard.py --modern    # Full-featured modern UI
python dashboard.py --simple    # Basic analysis tools

# Opens at http://localhost:3721
```

### 4. CLI Commands (Terminal Interface)
```bash
# Main CLI entry point (Modern CLI by default)
python main.py

# Choose CLI interface (NEW)
python main.py --cli modern    # Rich CLI with sub-commands (default)
python main.py --cli simple    # Simple CLI with flat commands

# Check account status
python main.py account info                    # Modern CLI
python main.py --cli simple account-info      # Simple CLI

# Analyze and trade watchlist (NEW)
python main.py watchlist-trade --days 30 --strategy momentum

# Real-time technical analysis (Modern CLI only)
python main.py analysis technical AAPL --indicators rsi macd sma
```

---

## ✨ Key Features

### 🎯 **Streamlined Data Integration (v1.1.0)**
- **Alpaca API Integration**: Unified data provider for stocks, options, and crypto via alpaca-py
- **Automatic Data Updates**: Real-time data fetching with intelligent caching
- **Eastern Time Consistency**: All market operations use ET timezone for accuracy
- **Clean Data Architecture**: Streamlined data sources with focused Alpaca integration

### 🧠 **Advanced Trading Strategies**
- **8 Built-in Strategies**: Momentum, Value, Trend Following, Scalping, Arbitrage, XGBoost, Deep Learning, Gaussian Process
- **Machine Learning**: Advanced ML models with feature engineering
- **Strategy Selector**: Intelligent recommendations based on market conditions
- **Market Hours Awareness**: Automatic market/limit order selection based on trading hours

### 💼 **Portfolio Management**
- **Real-time Tracking**: Live positions, P&L, and risk metrics
- **Order Management**: Complete order lifecycle with execution tracking
- **Account Analytics**: Performance metrics, Sharpe ratio, drawdown analysis
- **Risk Assessment**: VaR, concentration analysis, and position sizing

### 🖥️ **Modern Interfaces**
- **Rich CLI**: Beautiful command-line with progress bars and real-time updates
- **Web Dashboard**: Modern Streamlit interface with tabbed navigation
- **Smart Notifications**: Context-aware messages showing market status and data sources
- **Interactive Charts**: Real-time price charts with technical indicators

---

## 📊 Dashboard Features

Launch with `python dashboard.py` to access:

### 📈 **Live Market Analysis**
- **Restructured Dashboard**: New left-right panel layout for better organization
- **Real-time Price Charts**: Live charts with automatic data updates
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Multi-Asset Support**: Stocks, options, and crypto with unified display
- **Enhanced Data Integration**: Finnhub and FRED API for comprehensive market data

### 💰 **Trading Operations**
- **Watchlist Trading**: Automated analysis and strategy-based execution
- **Risk Management**: Built-in position sizing and validation
- **Order Execution**: Market orders during trading hours, limit orders when closed
- **Real-time Feedback**: Live updates on order status and execution

### 📊 **Portfolio Analytics**
- **Performance Metrics**: Returns, volatility, and risk-adjusted ratios
- **Position Analysis**: Holdings breakdown with P&L tracking
- **Asset Allocation**: Visual representation of portfolio distribution
- **Historical Performance**: Charts showing portfolio evolution over time

---

## 🛠️ CLI Commands

### **CLI Interface Selection (NEW)**
```bash
# Choose your preferred CLI interface
python main.py --cli modern     # Modern Rich CLI with sub-commands (default)
python main.py --cli simple     # Simple CLI with flat command structure

# All commands work with both interfaces, but syntax differs:
python main.py account info              # Modern CLI (sub-command)
python main.py --cli simple account-info # Simple CLI (flat command)
```

### **Account Operations**
```bash
python main.py account info --refresh           # Account overview
python main.py account performance --days 30    # Performance analysis
```

### **Data & Analysis**
```bash
python main.py data fetch AAPL GOOGL --days 30        # Historical data
python main.py data stream AAPL --interval 5          # Live streaming
python main.py analysis technical AAPL --indicators rsi macd bb
```

### **Trading (NEW)**
```bash
python main.py watchlist-trade                         # Default: 30 days, momentum
python main.py watchlist-trade --days 60 --strategy value
python main.py watchlist-trade -d 14 -s scalping

# Available strategies: momentum, value, trend_following, scalping, arbitrage, gaussian_process, xgboost, deep_learning
```

### **Portfolio Management**
```bash
python main.py check-positions                         # Current positions & orders
python main.py trade place AAPL buy 100 --dry-run     # Test order placement
```

---

## 🔧 Configuration

### **Required API Keys**
```bash
# Alpaca Trading (get free paper trading keys)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Financial Data (free tiers available)
FINNHUB_API_KEY=your_finnhub_key    # finnhub.io
FRED_API_KEY=your_fred_key          # fred.stlouisfed.org

# Optional: Additional data providers
ALPHA_VANTAGE_API_KEY=your_key      # alphavantage.co
POLYGON_API_KEY=your_key            # polygon.io
```

### **Data Integration**
- **Real-time Updates**: Alpaca API automatically provides current market data
- **No Manual Configuration**: System automatically fetches up-to-date information
- **Timezone Aware**: All operations use Eastern Time for market consistency

---

## 🏗️ Architecture

```
GaussWorldTrader/
├── src/
│   ├── account/          # Portfolio & order management
│   │   ├── account_config.py    # Account configuration
│   │   ├── account_manager.py   # Account operations
│   │   ├── order_manager.py     # Order lifecycle management
│   │   ├── portfolio_tracker.py # Portfolio tracking
│   │   └── position_manager.py  # Position management
│   ├── agent/            # AI-powered analysis (multi-LLM)
│   │   ├── agent_manager.py     # AI agent coordination
│   │   ├── fundamental_analyzer.py # Fundamental analysis
│   │   └── llm_providers.py     # Multi-provider LLM integration
│   ├── analysis/         # Technical analysis & metrics
│   │   ├── financial_metrics.py # Financial calculations
│   │   ├── performance_analyzer.py # Performance analysis
│   │   └── technical_analysis.py # Technical indicators
│   ├── data/             # Smart data providers
│   │   ├── alpaca_provider.py   # Primary data provider (stocks, options, crypto)
│   │   ├── finnhub_provider.py  # Finnhub financial data
│   │   ├── fred_provider.py     # Federal Reserve economic data
│   │   ├── macro_provider.py    # Macroeconomic data
│   │   ├── market_info_provider.py # Market information
│   │   └── news_provider.py     # News data integration
│   ├── strategy/         # 10 trading strategies
│   │   ├── base_strategy.py     # Abstract base class
│   │   ├── momentum_strategy.py # Momentum trading
│   │   ├── value_strategy.py    # Value investing
│   │   ├── trend_following_strategy.py # Trend following
│   │   ├── scalping_strategy.py # High-frequency scalping
│   │   ├── arbitrage_strategy.py # Statistical arbitrage
│   │   ├── xgboost_strategy.py  # XGBoost ML strategy
│   │   ├── deep_learning_strategy.py # Neural networks
│   │   ├── gaussian_process_strategy.py # Bayesian approach
│   │   └── strategy_selector.py # Strategy recommendation
│   ├── trade/            # Trading engine & execution
│   │   ├── backtester.py        # Backtesting framework
│   │   ├── trading_engine.py    # Core trading engine
│   │   ├── optimized_trading_engine.py # Performance optimized
│   │   └── portfolio.py         # Portfolio management
│   ├── ui/               # Multiple interface options
│   │   ├── core_cli.py          # Base CLI functionality (shared)
│   │   ├── core_dashboard.py    # Core dashboard functionality
│   │   ├── modern_dashboard.py  # Modern web interface
│   │   ├── simple_dashboard.py  # Simple interface
│   │   ├── modern_cli.py        # Rich CLI interface (primary)
│   │   ├── simple_cli.py        # Basic CLI (fallback)
│   │   └── portfolio_commands.py # Portfolio CLI commands
│   └── utils/            # Utilities and helpers
│       ├── dashboard_utils.py   # Dashboard utilities
│       ├── error_handling.py    # Error management
│       ├── logger.py            # Logging system
│       ├── timezone_utils.py    # Timezone handling
│       ├── validators.py        # Data validation
│       └── watchlist_manager.py # Watchlist management
├── config/               # Configuration management
│   └── config.py                # Performance optimized config (Python 3.12+)
├── examples/             # Usage examples and tutorials
│   ├── simple_example.py        # Basic usage
│   ├── run_backtest_example.py  # Backtesting example
│   ├── momentum_backtest_example.py # Momentum strategy
│   └── advanced_strategies_example.py # Advanced usage
├── results/              # Backtesting and analysis results
├── tests/                # Test suite
├── docs/                 # Project documentation
├── main.py               # CLI entry point
├── dashboard.py          # Dashboard launcher
├── pyproject.toml        # Project configuration
├── requirements.txt      # Python dependencies
└── watchlist.json        # Trading watchlist
```

---

## 🎯 Trading Strategies

### **Classical Strategies**
1. **Momentum**: RSI & volume-based trending
2. **Value**: Fundamental analysis-based investing
3. **Trend Following**: Multi-timeframe trend analysis
4. **Scalping**: High-frequency VWAP-based trading
5. **Arbitrage**: Statistical arbitrage & mean reversion

### **Machine Learning**
6. **XGBoost**: Gradient boosting with feature engineering
7. **Deep Learning**: LSTM, CNN, and Attention models
8. **Gaussian Process**: Bayesian approach with uncertainty quantification

### **Strategy Components**
- **Base Strategy**: Abstract foundation with signal validation
- **Strategy Selector**: Intelligent strategy recommendation system
- **8 Core Strategies**: Comprehensive suite for various market conditions

### **Usage Example**
```python
from src.strategy import MomentumStrategy

strategy = MomentumStrategy({
    'lookback_period': 20,
    'rsi_period': 14,
    'position_size_pct': 0.1
})

# Generate signals
signals = strategy.generate_signals(current_date, prices, data, historical_data, portfolio)
```

---

## 🛡️ Safety & Risk Management

**🔒 Built-in Safety Features:**
- **Paper Trading Default**: All trading starts in paper mode
- **Market Hours Awareness**: Automatic order type selection
- **Position Sizing**: Risk-based position calculation
- **Confirmation Prompts**: Manual approval for live trades
- **Data Validation**: Comprehensive input validation
- **Error Recovery**: Graceful handling of API failures

**⚠️ Important**: This is educational software. Real trading involves significant risk.

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Test** your changes thoroughly
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Submit** a pull request

**Issues & Support**: [GitHub Issues](https://github.com/Magica-Chen/GaussWorldTrader/issues)

---

## 📈 Recent Updates

### **v1.1.0 - API Migration & Crypto Integration**
- ✅ **Alpaca API Migration**: Complete migration from alpaca-trade-api to alpaca-py
- ✅ **Cryptocurrency Support**: Full integration for crypto trading and analysis
- ✅ **Options Trading**: Enhanced options support with better data handling
- ✅ **Watchlist Trading**: Automated analysis and strategy execution
- ✅ **Data Source Optimization**: Streamlined to focus on Alpaca as primary provider
- ✅ **Performance Improvements**: Reduced complexity and improved execution speed

### **v1.0.0 - AI & Portfolio Management**
- 🤖 Multi-LLM AI analysis (OpenAI, DeepSeek, Claude, Moonshot)
- 💼 Complete portfolio management system
- 📊 Advanced backtesting with risk metrics
- 🎯 Automated watchlist trading

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This software is for educational purposes only. Trading involves substantial risk of loss. Use at your own risk and never invest more than you can afford to lose.
