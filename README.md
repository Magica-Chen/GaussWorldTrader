<div align="center">
  <img src="src/pic/logo3.png" width="600" alt="Gauss World Trader logo">
  <p>
    <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
    <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=for-the-badge" alt="Platform">
    <img src="https://img.shields.io/badge/Trading-Alpaca-yellow?style=for-the-badge" alt="Alpaca">
  </p>
  <p>
    <strong>Gauss World Trader</strong> — <em>A high-performance, Python 3.12+ optimized quantitative trading system featuring modern async operations, intelligent data feeds, and advanced portfolio management.</em>
  </p>
  <p>
    <strong>Named after Carl Friedrich Gauss</strong>, who revolutionized statistics and probability theory — the foundations of modern quantitative finance.
  </p>
</div>

---

## ✨ Features

- **🚀 Modern Async Architecture** — Built for Python 3.12+ with async/await patterns
- **📊 Multiple Trading Strategies** — Momentum, Value, Trend Following, Statistical Arbitrage, and more
- **📈 Real-time Dashboard** — Interactive Streamlit interface for monitoring and analysis
- **💼 Portfolio Management** — Advanced position tracking and risk management
- **🔌 Multi-source Data Feeds** — Alpaca, Finnhub, FRED, and News integrations

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Magica-Chen/GaussWorldTrader.git
cd GaussWorldTrader

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env

# Run the dashboard
python dashboard.py

# Or use the CLI
python main_cli.py list-strategies
```

---

## 🎯 Two Entry Points

| Entry Point | Command | Description |
|-------------|---------|-------------|
| **CLI** | `python main_cli.py` | Command-line interface for scripting and automation |
| **Dashboard** | `python dashboard.py` | Interactive Streamlit web interface at `http://localhost:3721` |

 ![Trading Dashboard](/src/pic/screenshot2.png)

### CLI Examples

```bash
python main_cli.py list-strategies              # List all available strategies
python main_cli.py run-strategy momentum AAPL MSFT --days 90
python main_cli.py backtest momentum AAPL --days 365
python main_cli.py account-info                 # View account details
python main_cli.py stream-market --asset-type crypto --crypto-loc eu-1 --symbols BTC/USD,ETH/USD
```

---

## 🛰️ Live Trading

```bash
# Launch unified interactive CLI
python live_script.py
```
 ![live Trading CLI](/src/pic/screenshot1.png)

The unified CLI provides:
- **Quick Start** — Trade all asset types with watchlist defaults
- **Custom Configuration** — Select asset types, symbols, strategies, and parameters interactively

**Strategy Selection by Asset Type:**
| Asset Type | Available Strategies |
|------------|---------------------|
| Stock | momentum, value, trend_following, scalping, statistical_arbitrage |
| Crypto | crypto_momentum |
| Option | wheel |

Notes:
- Multi-symbol runs share a single websocket per asset type to stay within Alpaca connection limits.
- Due to Alpaca connection limits, multiple asset types run sequentially (press Ctrl+C to advance).
- Stock and option engines check market hours before trading.
- Defaults are pulled from `watchlist.json` + current positions for each asset type.

---

## 📊 Built-in Strategies

| Strategy | Category | Dashboard |
|----------|----------|-----------|
| 📈 Momentum | Signal | ✅ |
| 💰 Value | Signal | ✅ |
| 📉 Trend Following | Signal | ✅ |
| ⚡ Scalping | Signal | ✅ |
| 📐 Statistical Arbitrage | Signal | ✅ |
| 🎡 Wheel (Options) | Options | ❌ |

---

## 🏗️ Project Structure

```
GaussWorldTrader/
├── 📄 main_cli.py          # CLI entry point
├── 📄 dashboard.py         # Streamlit dashboard entry
├── 📄 live_script.py       # Unified live trading CLI
├── 📄 watchlist.json       # Watchlist entries with asset_type
├── 📁 src/
│   ├── 📁 strategy/        # Trading strategies & templates
│   ├── 📁 script/          # Live trading modules (stock, crypto, option)
│   ├── 📁 ui/              # Dashboard components
│   ├── 📁 trade/           # Trading engine & backtester
│   ├── 📁 data/            # Market data providers
│   └── 📁 account/         # Portfolio & position tracking
└── 📁 docs/                # Documentation
```

---

## 🧩 Adding a Strategy

```python
from src.strategy.base import StrategyBase, StrategyMeta, StrategySignal

class MyStrategy(StrategyBase):
    meta = StrategyMeta(
        name="my_strategy",
        label="My Strategy",
        category="signal",
        description="Your strategy description here.",
        visible_in_dashboard=True,
        default_params={"lookback": 20}
    )
    summary = "Brief intro + formulas/logic for this strategy."

    def generate_signals(self, current_date, current_prices, current_data,
                         historical_data, portfolio=None):
        return self._normalize([
            StrategySignal(
                symbol="AAPL",
                action="BUY",
                quantity=1,
                price=current_prices.get("AAPL"),
                reason="example signal",
                timestamp=current_date,
            )
        ])
```

---

## ⚙️ Configuration

Create a `.env` file with the following API keys:

| Key | Required | Description |
|-----|----------|-------------|
| `ALPACA_API_KEY` | ✅ | Alpaca trading API key |
| `ALPACA_SECRET_KEY` | ✅ | Alpaca secret key |
| `ALPACA_BASE_URL` | ✅ | Alpaca API endpoint |
| `FINNHUB_API_KEY` | ✅ | Finnhub market data |
| `FRED_API_KEY` | ✅ | Federal Reserve economic data |

---

## 👁️ Watchlist

Watchlist entries are typed by asset so the dashboard and live scripts can filter symbols correctly.

```json
{
  "watchlist": [
    {"symbol": "AAPL", "asset_type": "stock"},
    {"symbol": "BTC/USD", "asset_type": "crypto"}
  ],
  "metadata": {
    "created": "2025-08-21",
    "last_updated": "2026-01-16 00:11:10",
    "description": "Gauss World Trader Default Watchlist",
    "version": "2.0"
  }
}
```

- Supported `asset_type`: `stock`, `crypto`, `option`.
- The dashboard Watchlist tab lets you add/remove symbols with an asset type.

---

## 📚 Documentation

- [Wheel Options Strategy](docs/wheel_strategy.md) — Detailed guide for the wheel options strategy

---

## ⚠️ Disclaimer

> **This project is for educational purposes only.** Live trading carries significant financial risk. Always use paper trading mode first and never trade with money you cannot afford to lose.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Magica-Chen/GaussWorldTrader&type=date&legend=top-left)](https://www.star-history.com/#Magica-Chen/GaussWorldTrader&type=date&legend=top-left)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Magica-Chen">Magica-Chen</a>
</p>
