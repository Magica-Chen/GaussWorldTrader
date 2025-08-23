# Examples Directory

This directory contains example scripts demonstrating various features of Gauss World Trader.

## Available Examples

### 1. `simple_example.py`
Basic usage example showing:
- Simple trading strategy implementation
- Market data fetching
- Basic backtesting

### 2. `momentum_backtest_example.py`
Momentum strategy backtesting example showing:
- Advanced backtesting with performance analytics
- Chart generation and visualization
- CSV transaction export

### 3. `advanced_strategies_example.py`
Comprehensive strategy demonstration showing:
- All available trading strategies
- Strategy comparison and selection
- Performance metrics analysis

### 4. `run_backtest_example.py`
Command-line backtesting example showing:
- CLI-based backtesting
- Multiple symbol analysis
- Results visualization

## Running Examples

From the project root directory:

```bash
# Simple trading example
python examples/simple_example.py

# Momentum strategy backtest
python examples/momentum_backtest_example.py

# Advanced strategies demonstration
python examples/advanced_strategies_example.py

# CLI backtest example
python examples/run_backtest_example.py
```

## Requirements

All examples require the same dependencies as the main application. Ensure you have:
- Python 3.12+
- All packages from `requirements.txt`
- Valid API credentials in `.env` file

## Example Data

Examples use real market data when API credentials are available, or demonstrate with sample data otherwise.