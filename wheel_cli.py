#!/usr/bin/env python3
"""
Wheel Strategy Command Line Interface

Simple CLI for running and managing the Wheel Options Strategy.
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.option_strategy import WheelStrategy
from src.trade import Portfolio
import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample market data for testing"""
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

    return current_prices, current_data


def run_wheel_strategy(config_file=None, verbose=False):
    """Run the wheel strategy with given configuration"""

    print("🎯 Gauss World Trader - Wheel Options Strategy")
    print("=" * 50)

    # Load configuration
    if config_file:
        config_path = Path(config_file)
    else:
        # Default to config/wheel_config.json
        config_path = Path("config/wheel_config.json")

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"📄 Loaded configuration from {config_path}")
    else:
        # Default configuration
        config = {
            'max_risk': 50000,
            'position_size_pct': 0.08,
            'put_delta_min': 0.20,
            'put_delta_max': 0.35,
            'min_yield': 0.04,
            'dte_min': 14,
            'dte_max': 35,
            'max_positions': 5
        }
        print("📄 Using default configuration")

    if verbose:
        print(f"📊 Configuration: {json.dumps(config, indent=2)}")

    # Initialize strategy
    print("\n🔧 Initializing Wheel Strategy...")
    strategy = WheelStrategy(config)

    # Create mock portfolio and data
    portfolio = Portfolio(initial_cash=100000)
    current_prices, current_data = create_sample_data()

    print(f"💰 Portfolio: ${portfolio.cash:,.2f} cash")
    print(f"📈 Market data: {len(current_prices)} symbols")

    # Generate signals
    print("\n🔄 Generating wheel strategy signals...")
    signals = strategy.generate_signals(
        current_date=datetime.now(),
        current_prices=current_prices,
        current_data=current_data,
        historical_data={},
        portfolio=portfolio
    )

    # Display results
    print(f"\n📊 Generated {len(signals)} signals:")

    if signals:
        for i, signal in enumerate(signals, 1):
            print(f"\n  [{i}] {signal['symbol']}")
            print(f"      Action: {signal['action']}")
            print(f"      Type: {signal.get('option_type', 'N/A')}")
            print(f"      Quantity: {signal['quantity']}")
            print(f"      Stage: {signal.get('strategy_stage', 'N/A')}")
            print(f"      Reason: {signal.get('reason', 'N/A')}")

            if 'strike_price' in signal:
                print(f"      Strike: ${signal['strike_price']:.2f}")
            if 'premium' in signal:
                print(f"      Premium: ${signal['premium']:.2f}")
            if 'yield' in signal:
                print(f"      Yield: {signal['yield']:.2f}%")
            if 'confidence' in signal:
                print(f"      Confidence: {signal['confidence']:.1%}")
    else:
        print("  No signals generated (normal in current conditions)")

    # Strategy info
    if verbose:
        print(f"\n📋 Strategy Information:")
        info = strategy.get_strategy_info()
        print(f"  Name: {info['name']}")
        print(f"  Type: {info['type']}")
        print(f"  Risk Level: {info['risk_level']}")
        print(f"  Watchlist Symbols: {info['watchlist_symbols']}")

    return signals


def generate_config_template():
    """Generate a configuration template file"""
    config_template = {
        "max_risk": 80000,
        "position_size_pct": 0.08,
        "put_delta_min": 0.15,
        "put_delta_max": 0.30,
        "call_delta_min": 0.15,
        "call_delta_max": 0.30,
        "min_yield": 0.04,
        "max_yield": 1.00,
        "dte_min": 7,
        "dte_max": 45,
        "preferred_dte": 21,
        "min_open_interest": 100,
        "min_daily_volume": 50,
        "min_score": 0.05,
        "max_options_per_underlying": 1,
        "assignment_tolerance": 0.80,
        "profit_target": 0.50,
        "management_dte": 7,
        "max_positions": 10,
        "min_stock_price": 10.0,
        "max_stock_price": 500.0
    }

    # Ensure config directory exists
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "wheel_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_template, f, indent=2)

    print(f"✅ Configuration template created: {config_file}")
    print("Edit this file to customize your wheel strategy parameters.")


def show_strategy_info():
    """Show detailed strategy information"""
    print("🎯 Wheel Options Strategy Information")
    print("=" * 50)

    print("""
The Wheel Strategy is a systematic options trading approach that:

1. 🎯 SELLS CASH-SECURED PUTS
   - Target stocks you want to own
   - Collect premium income
   - Be willing to be assigned

2. 📦 MANAGES ASSIGNMENT
   - Purchase shares at strike price
   - Move to covered call phase
   - Maintain disciplined approach

3. 📞 SELLS COVERED CALLS
   - Generate additional income
   - Target profitable exit prices
   - Complete the wheel cycle

KEY BENEFITS:
✅ Systematic income generation
✅ Disciplined stock acquisition
✅ Built-in risk management
✅ Suitable for stable stocks

KEY RISKS:
⚠️  Assignment at unfavorable prices
⚠️  Opportunity cost in bull markets
⚠️  Requires significant capital
⚠️  Active management needed

IDEAL CONDITIONS:
📈 Moderate volatility
📈 Stable, quality stocks
📈 Neutral to slightly bullish outlook
📈 Sufficient capital for assignment
    """)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Wheel Options Strategy CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (default: config/wheel_config.json)',
        type=str
    )

    parser.add_argument(
        '--verbose', '-v',
        help='Verbose output',
        action='store_true'
    )

    parser.add_argument(
        '--generate-config',
        help='Generate configuration template',
        action='store_true'
    )

    parser.add_argument(
        '--info',
        help='Show strategy information',
        action='store_true'
    )

    parser.add_argument(
        '--run',
        help='Run the wheel strategy',
        action='store_true'
    )

    args = parser.parse_args()

    try:
        if args.generate_config:
            generate_config_template()
        elif args.info:
            show_strategy_info()
        elif args.run:
            run_wheel_strategy(args.config, args.verbose)
        else:
            # Default action - run the strategy
            run_wheel_strategy(args.config, args.verbose)

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()