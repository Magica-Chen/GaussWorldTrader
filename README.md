# Gauss World Trader

A streamlined Python 3.12+ quantitative trading system with a unified dashboard
and a single, simple CLI entry point.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env

# Run the dashboard
python dashboard.py

# Run the CLI
python main_cli.py list-strategies
```

## Two Entry Points

- **CLI**: `main_cli.py`
- **Dashboard**: `dashboard.py` (Streamlit)

### CLI Examples

```bash
python main_cli.py list-strategies
python main_cli.py run-strategy momentum AAPL MSFT --days 90
python main_cli.py backtest momentum AAPL --days 365
python main_cli.py account-info
```

### Dashboard

```bash
python dashboard.py
# Opens at http://localhost:3721
```

## Strategy Template

All strategies share a single template in `src/strategy/template.py` and are
implemented together in `src/strategy/strategies.py`.

Template highlights:
- `StrategyMeta`: name, label, category, description, dashboard visibility
- `StrategyBase.generate_signals(...)`: single output path for backtests and live runs
- `StrategyRegistry`: list + factory methods

### Adding a Strategy (Template)

```python
from src.strategy.template import StrategyBase, StrategyMeta, StrategySignal

class MyStrategy(StrategyBase):
    meta = StrategyMeta(
        name="my_strategy",
        label="My Strategy",
        category="signal",
        description="Explain what this does.",
        visible_in_dashboard=True,
        default_params={"lookback": 20}
    )

    def generate_signals(self, current_date, current_prices, current_data, historical_data, portfolio=None):
        # Use the same signal structure everywhere
        return self._normalize([
            StrategySignal(
                symbol="AAPL",
                action="BUY",
                quantity=1,
                price=current_prices.get("AAPL"),
                reason="example",
                timestamp=current_date,
            )
        ])
```

## Built-in Strategies

Dashboard-visible:
- Momentum
- Value
- Trend Following
- Scalping
- Statistical Arbitrage
- Gaussian Process
- XGBoost (template placeholder)
- Deep Learning (template placeholder)

Non-dashboard:
- Wheel (options template)

## Project Layout

```
GaussWorldTrader/
├── main_cli.py
├── dashboard.py
├── src/
│   ├── strategy/          # Template + built-ins
│   ├── ui/                # Streamlit dashboard
│   ├── trade/             # Trading engine + backtester
│   ├── data/              # Market data providers
│   └── ...
```

## Configuration

Environment variables are loaded from `.env`.

Required keys:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL`
- `FINNHUB_API_KEY`
- `FRED_API_KEY`

## Notes

This project is educational. Live trading carries risk—use paper mode first.
