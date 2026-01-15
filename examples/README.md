# Examples Directory

This directory contains example scripts demonstrating various features of Gauss World Trader.

## Available Examples

### 1. `simple_example.py`
Basic usage example showing:
- Stock momentum signal generation
- Trading plan output

### 2. `momentum_backtest_example.py`
Momentum strategy backtesting example showing:
- Advanced backtesting with performance analytics
- Chart generation and visualization
- CSV transaction export

### 3. `crypto_momentum_example.py`
Crypto momentum strategy example showing:
- BTC/USD data fetch with Alpaca
- Strategy signal generation
- Trading plan output

### 4. `wheel_strategy_example.py`
Wheel options strategy example showing:
- Live options chain usage (Alpaca entitlement required)
- Cash-secured put/covered call signals

## Running Examples

From the project root directory:

```bash
# Simple trading example
python examples/simple_example.py

# Momentum strategy backtest
python examples/momentum_backtest_example.py

# Crypto momentum example (requires Alpaca API keys)
python examples/crypto_momentum_example.py

# Wheel options example (requires Alpaca option data access)
python examples/wheel_strategy_example.py
```

## Requirements

All examples require the same dependencies as the main application. Ensure you have:
- Python 3.12+
- All packages from `requirements.txt`
- Valid API credentials in `.env` file

## Example Data

Examples use live market data. If required credentials or data entitlements are missing, the script exits with a brief message.
