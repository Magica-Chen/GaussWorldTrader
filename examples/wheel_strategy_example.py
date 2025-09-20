#!/usr/bin/env python3
"""
Wheel Strategy Example

This example demonstrates how to use the Wheel Options Strategy
in the Gauss World Trader system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.option_strategy import WheelStrategy
from src.trade import Portfolio

def create_mock_portfolio():
    """Create a mock portfolio for testing"""
    portfolio = Portfolio(initial_cash=100000)

    # Add some mock stock positions for covered call testing
    portfolio.positions = {
        'AAPL': {'quantity': 200, 'avg_price': 150.0, 'total_cost': 30000},
        'MSFT': {'quantity': 100, 'avg_price': 280.0, 'total_cost': 28000}
    }

    # Mock option positions
    portfolio.option_positions = {}

    return portfolio

def create_mock_market_data():
    """Create mock market data"""
    current_prices = {
        'AAPL': 155.50,
        'MSFT': 285.75,
        'GOOGL': 138.25,
        'TSLA': 185.30,
        'AMZN': 145.80,
        'RGTI': 12.50,
        'AFRM': 45.60,
        'UPST': 32.40
    }

    current_data = {}
    for symbol, price in current_prices.items():
        current_data[symbol] = {
            'open': price * 0.99,
            'high': price * 1.02,
            'low': price * 0.98,
            'close': price,
            'volume': np.random.randint(1000000, 5000000)
        }

    # Create some historical data
    historical_data = {}
    for symbol, price in current_prices.items():
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        prices = price * (1 + np.random.randn(len(dates)).cumsum() * 0.01)

        historical_data[symbol] = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, len(dates)),
            'high': prices * np.random.uniform(1.00, 1.03, len(dates)),
            'low': prices * np.random.uniform(0.97, 1.00, len(dates)),
            'close': prices,
            'volume': np.random.randint(500000, 2000000, len(dates))
        }, index=dates)

    return current_prices, current_data, historical_data

def test_wheel_strategy():
    """Test the wheel strategy with mock data"""

    print("🎯 Wheel Strategy Test")
    print("=" * 50)

    # Initialize the strategy
    wheel_params = {
        'max_risk': 50000,           # Reduce for testing
        'position_size_pct': 0.05,   # 5% position sizes
        'max_positions': 5,          # Max 5 positions
        'put_delta_min': 0.20,       # 20% assignment probability
        'put_delta_max': 0.35,       # 35% assignment probability
        'min_yield': 0.03,           # 3% minimum yield
        'dte_min': 7,                # 7 days minimum
        'dte_max': 35,               # 35 days maximum
    }

    strategy = WheelStrategy(wheel_params)

    # Create mock data
    portfolio = create_mock_portfolio()
    current_prices, current_data, historical_data = create_mock_market_data()

    print(f"📊 Strategy Info:")
    info = strategy.get_strategy_info()
    print(f"  Name: {info['name']}")
    print(f"  Type: {info['type']}")
    print(f"  Risk Level: {info['risk_level']}")
    print(f"  Timeframe: {info['timeframe']}")
    print(f"  Watchlist Symbols: {info['watchlist_symbols']}")

    print(f"\n💰 Portfolio Status:")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Stock Positions: {len(portfolio.positions)}")
    print(f"  Option Positions: {len(getattr(portfolio, 'option_positions', {}))}")

    # Generate signals
    print(f"\n🔄 Generating Wheel Strategy Signals...")
    current_date = datetime.now()

    signals = strategy.generate_signals(
        current_date=current_date,
        current_prices=current_prices,
        current_data=current_data,
        historical_data=historical_data,
        portfolio=portfolio
    )

    print(f"\n📈 Generated {len(signals)} signals:")

    if signals:
        for i, signal in enumerate(signals, 1):
            print(f"\n  Signal {i}:")
            print(f"    Symbol: {signal['symbol']}")
            print(f"    Underlying: {signal.get('underlying_symbol', 'N/A')}")
            print(f"    Action: {signal['action']}")
            print(f"    Type: {signal.get('option_type', 'stock')}")
            print(f"    Quantity: {signal['quantity']}")
            print(f"    Strategy Stage: {signal.get('strategy_stage', 'N/A')}")
            print(f"    Reason: {signal.get('reason', 'N/A')}")
            print(f"    Confidence: {signal.get('confidence', 0):.1%}")

            if 'strike_price' in signal:
                print(f"    Strike Price: ${signal['strike_price']:.2f}")
            if 'premium' in signal:
                print(f"    Premium: ${signal['premium']:.2f}")
            if 'yield' in signal:
                print(f"    Yield: {signal['yield']:.2f}%")
            if 'score' in signal:
                print(f"    Score: {signal['score']:.4f}")
    else:
        print("  No signals generated (normal in current market conditions)")

    print(f"\n📊 Strategy Parameters:")
    params = strategy.parameters
    print(f"  Max Risk: ${params['max_risk']:,}")
    print(f"  Position Size: {params['position_size_pct']:.1%}")
    print(f"  Put Delta Range: {params['put_delta_min']:.2f} - {params['put_delta_max']:.2f}")
    print(f"  Min Yield: {params['min_yield']:.1%}")
    print(f"  DTE Range: {params['dte_min']} - {params['dte_max']} days")

    return strategy, signals

def demonstrate_strategy_cycle():
    """Demonstrate the complete wheel strategy cycle"""

    print(f"\n🔄 Wheel Strategy Cycle Demonstration")
    print("=" * 50)

    print("""
The Wheel Strategy operates in a systematic cycle:

1. 🎯 CASH-SECURED PUTS
   - Sell put options on stocks you want to own
   - Collect premium income
   - Wait for expiration or assignment

2. 📦 ASSIGNMENT (if put expires ITM)
   - Receive 100 shares per contract
   - Pay the strike price
   - Now own the underlying stock

3. 📞 COVERED CALLS
   - Sell call options on owned shares
   - Collect additional premium
   - Wait for expiration or assignment

4. 🔄 CALL AWAY (if call expires ITM)
   - Shares are sold at strike price
   - Collect the call premium
   - Return to step 1 with cash

💡 Key Benefits:
   - Generate income at every step
   - Potentially acquire stocks at lower prices
   - Systematic approach to options trading
   - Built-in risk management

⚠️ Key Risks:
   - Assignment at unfavorable prices
   - Opportunity cost if stock rises significantly
   - Requires active management
   - Capital intensive (cash-secured puts)
    """)

if __name__ == '__main__':
    try:
        # Test the strategy
        strategy, signals = test_wheel_strategy()

        # Demonstrate the cycle
        demonstrate_strategy_cycle()

        print(f"\n✅ Wheel Strategy Test Complete!")
        print(f"\n📚 Next Steps:")
        print(f"  1. Review generated signals above")
        print(f"  2. Integrate with live Alpaca API for real option data")
        print(f"  3. Run backtests with historical data")
        print(f"  4. Paper trade to validate strategy")
        print(f"  5. Deploy with real capital (start small!)")

    except Exception as e:
        print(f"❌ Error running wheel strategy test: {e}")
        import traceback
        traceback.print_exc()