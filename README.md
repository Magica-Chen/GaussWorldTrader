# 🌍 Gauss World Trader

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Magica-Chen/GaussWorldTrader.svg)](https://github.com/Magica-Chen/GaussWorldTrader/issues)

A high-performance, **Python 3.12+ optimized** quantitative trading system featuring modern async operations, intelligent data feeds, and advanced portfolio management. Named after Carl Friedrich Gauss, who revolutionized statistics and probability theory - the foundations of modern quantitative finance.

**🔗 Repository**: [https://github.com/Magica-Chen/GaussWorldTrader](https://github.com/Magica-Chen/GaussWorldTrader)  
**👤 Author**: Magica Chen  
**📊 Version**: 1.1.1 - Simplified Alpaca Integration

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
python dashboard.py --advanced  # Advanced trading features  
python dashboard.py --simple    # Basic analysis tools

# Opens at http://localhost:3721
```

### 4. CLI Commands (Terminal Interface)
```bash
# Main CLI entry point
python main.py

# Check account status
python main.py account info

# Analyze and trade watchlist (NEW)
python main.py watchlist-trade --days 30 --strategy momentum

# Real-time technical analysis
python main.py analysis technical AAPL --indicators rsi macd sma
```

---

## ✨ Key Features

### 🎯 **Streamlined Data Integration (NEW v1.1.1)**
- **Simplified Alpaca API**: Unified data provider for stocks, options, and crypto
- **Automatic Data Updates**: Real-time data fetching without manual end_date parameters
- **Eastern Time Consistency**: All market operations use ET timezone for accuracy
- **Clean Data Context**: Removed unnecessary market status calculations for better performance

### 🧠 **Advanced Trading Strategies**
- **8 Built-in Strategies**: Momentum, Value, Trend Following, Scalping, Arbitrage
- **Machine Learning**: XGBoost, Deep Learning (LSTM/CNN), Gaussian Process
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
- **Real-time Price Charts**: Live charts with automatic data updates
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Multi-Symbol Analysis**: Compare multiple stocks simultaneously
- **Streamlined Interface**: Clean, focused market data presentation

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

# Available strategies: momentum, value, trend, scalping, arbitrage, gaussian, xgboost, deep
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
FRED_API_KEY=your_fred_key        # fred.stlouisfed.org
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
│   │   └── llm_providers.py     # LLM integration
│   ├── analysis/         # Technical analysis & metrics
│   │   ├── financial_metrics.py # Financial calculations
│   │   ├── performance_analyzer.py # Performance analysis
│   │   └── technical_analysis.py # Technical indicators
│   ├── data/             # Smart data providers
│   │   ├── alpaca_provider.py   # Unified data provider (stocks, options, crypto)
│   │   ├── macro_provider.py    # Macroeconomic data
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
│   │   ├── modern_dashboard.py  # Modern web interface
│   │   ├── advanced_dashboard.py # Advanced features
│   │   ├── simple_dashboard.py  # Simple interface
│   │   ├── modern_cli.py        # Rich CLI interface
│   │   ├── cli_interface.py     # Command-line interface
│   │   ├── simple_cli.py        # Basic CLI
│   │   └── portfolio_commands.py # Portfolio CLI commands
│   └── utils/            # Utilities and helpers
│       ├── dashboard_utils.py   # Dashboard utilities
│       ├── error_handling.py    # Error management
│       ├── logger.py            # Logging system
│       ├── timezone_utils.py    # Timezone handling
│       ├── validators.py        # Data validation
│       └── watchlist_manager.py # Watchlist management
├── config/               # Configuration management
│   ├── config.py                # Base configuration
│   └── optimized_config.py      # Performance optimized config
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
- **Base Strategy**: Abstract foundation for all trading strategies
- **Strategy Selector**: Intelligent strategy recommendation system
- **10 Total Strategies**: Complete suite for different market conditions

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

### **v1.1.1 - Simplified Alpaca Integration**
- ✅ **Simplified Data Fetching**: Removed unnecessary end_date parameters from all Alpaca API calls
- ✅ **Clean Dashboard Logic**: Removed complex data_context calculations from all dashboards
- ✅ **Centralized Timezone**: Consistent Eastern Time for all market operations
- ✅ **Enhanced Watchlist Trading**: Added `--days` and `--strategy` parameters
- ✅ **Market Hours Awareness**: Smart order type selection (market/limit)
- ✅ **Clean Codebase**: Streamlined data provider implementation

### **v1.1.0 - Smart Data Feeds & Enhanced Trading**
- ✅ **Automatic VIP Detection**: Uses SPY SIP feed test for tier detection
- ✅ **Data Source Notifications**: Clear indicators for SIP vs IEX usage
- ✅ **Fallback Logic**: Robust data retrieval with intelligent fallbacks

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