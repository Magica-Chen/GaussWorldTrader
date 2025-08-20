#!/usr/bin/env python3
"""
Advanced Trading Strategies Example

Demonstrates the usage of all classical and modern ML-based trading strategies
in the Gauss World Trader system.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.strategy_selector import StrategySelector, get_strategy_selector
from src.data import AlpacaDataProvider
from src.backtest import Backtester

def demonstrate_strategy_selection():
    """Demonstrate the strategy selection system"""
    
    print("🌍 Gauss World Trader - Advanced Strategies Demonstration")
    print("=" * 60)
    
    # Get strategy selector
    selector = get_strategy_selector()
    
    # List all available strategies
    print("\n📋 Available Strategies:")
    all_strategies = selector.list_strategies()
    for i, strategy in enumerate(all_strategies, 1):
        print(f"{i:2d}. {strategy}")
    
    # Show strategies by category
    print("\n📊 Strategies by Category:")
    categories = ['high_frequency', 'low_frequency', 'machine_learning', 'classical']
    
    for category in categories:
        strategies = selector.list_strategies(category)
        print(f"\n{category.replace('_', ' ').title()}:")
        for strategy in strategies:
            info = selector.get_strategy_info(strategy)
            print(f"  • {strategy}: {info.get('description', 'No description')[:60]}...")
    
    return selector

def demonstrate_strategy_creation_and_info():
    """Demonstrate creating strategies and getting their information"""
    
    print("\n🔧 Strategy Creation and Information")
    print("-" * 40)
    
    selector = get_strategy_selector()
    
    # Create different types of strategies
    strategies_to_demo = [
        ('momentum', {}),
        ('scalping', {'position_size_pct': 0.03}),
        ('trend_following', {'max_positions': 3}),
        ('xgboost', {'prediction_horizon': 3}),
        ('gaussian_process', {'uncertainty_threshold': 0.015})
    ]
    
    for strategy_name, params in strategies_to_demo:
        print(f"\n📈 {strategy_name.replace('_', ' ').title()} Strategy:")
        
        # Get strategy info
        info = selector.get_strategy_info(strategy_name)
        
        print(f"  Type: {info.get('type', 'Unknown')}")
        print(f"  Timeframe: {info.get('timeframe', 'Unknown')}")
        print(f"  Risk Level: {info.get('risk_level', 'Unknown')}")
        print(f"  Expected Trades/Day: {info.get('expected_trades_per_day', 'Unknown')}")
        print(f"  Holding Period: {info.get('holding_period', 'Unknown')}")
        
        # Create strategy instance
        strategy = selector.create_strategy(strategy_name, params)
        if strategy:
            print(f"  ✅ Strategy created successfully")
            if hasattr(strategy, 'parameters'):
                print(f"  Parameters: {len(strategy.parameters)} configured")
        else:
            print(f"  ❌ Failed to create strategy")

def demonstrate_market_condition_recommendations():
    """Demonstrate strategy recommendations based on market conditions"""
    
    print("\n🎯 Market-Based Strategy Recommendations")
    print("-" * 45)
    
    selector = get_strategy_selector()
    
    # Different market scenarios
    market_scenarios = [
        {
            'name': 'High Volatility Bull Market',
            'conditions': {
                'volatility': 'high',
                'trend_strength': 'strong',
                'regime': 'trending',
                'session': 'active',
                'liquidity': 'high'
            }
        },
        {
            'name': 'Low Volatility Ranging Market',
            'conditions': {
                'volatility': 'low',
                'trend_strength': 'weak',
                'regime': 'ranging',
                'session': 'regular',
                'liquidity': 'medium'
            }
        },
        {
            'name': 'Uncertain Market Conditions',
            'conditions': {
                'volatility': 'medium',
                'trend_strength': 'medium',
                'regime': 'uncertain',
                'session': 'regular',
                'liquidity': 'high'
            }
        }
    ]
    
    for scenario in market_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        recommendations = selector.get_strategy_recommendations(scenario['conditions'])
        
        print(f"  Recommended Strategies ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
            print(f"    {i}. {rec['strategy']} (Confidence: {rec['confidence']:.1%})")
            print(f"       Reason: {rec['reason']}")
            if rec.get('suggested_params'):
                print(f"       Suggested params: {rec['suggested_params']}")

def demonstrate_strategy_portfolio():
    """Demonstrate creating a portfolio of strategies"""
    
    print("\n📂 Strategy Portfolio Creation")
    print("-" * 35)
    
    selector = get_strategy_selector()
    
    # Define a diversified strategy portfolio
    portfolio_config = [
        {
            'strategy': 'trend_following',
            'allocation': 0.4,  # 40% allocation
            'parameters': {'position_size_pct': 0.08}
        },
        {
            'strategy': 'momentum',
            'allocation': 0.3,  # 30% allocation
            'parameters': {'position_size_pct': 0.06}
        },
        {
            'strategy': 'scalping',
            'allocation': 0.2,  # 20% allocation
            'parameters': {'position_size_pct': 0.04}
        },
        {
            'strategy': 'gaussian_process',
            'allocation': 0.1,  # 10% allocation
            'parameters': {'position_size_pct': 0.02}
        }
    ]
    
    print("Creating diversified strategy portfolio:")
    portfolio = selector.create_strategy_portfolio(portfolio_config)
    
    for strategy_id, strategy in portfolio.items():
        strategy_name = strategy_id.split('_')[0]
        allocation = strategy_id.split('_')[-1]
        print(f"  ✅ {strategy_name}: {float(allocation)*100:.0f}% allocation")
    
    print(f"\nTotal strategies in portfolio: {len(portfolio)}")

def demonstrate_strategy_compatibility():
    """Demonstrate strategy compatibility checking"""
    
    print("\n🔗 Strategy Compatibility Analysis")
    print("-" * 40)
    
    selector = get_strategy_selector()
    
    # Test different strategy combinations
    combinations = [
        ('trend_following', 'scalping'),
        ('momentum', 'statistical_arbitrage'),
        ('xgboost', 'value_investment'),
        ('gaussian_process', 'deep_learning')
    ]
    
    for strategy1, strategy2 in combinations:
        print(f"\n🔍 Analyzing: {strategy1} + {strategy2}")
        compatibility = selector.get_strategy_compatibility(strategy1, strategy2)
        
        if compatibility['compatible']:
            print("  ✅ Compatible")
        else:
            print("  ⚠️  Potential conflicts")
        
        if compatibility['conflicts']:
            print("  Conflicts:")
            for conflict in compatibility['conflicts']:
                print(f"    • {conflict}")
        
        if compatibility['recommendations']:
            print("  Recommendations:")
            for rec in compatibility['recommendations']:
                print(f"    • {rec}")

def run_mini_backtest_comparison():
    """Run a mini backtest comparing different strategies"""
    
    print("\n🧪 Mini Backtest Comparison")
    print("-" * 35)
    
    try:
        # Create sample data (in real usage, this would come from AlpacaDataProvider)
        print("Generating sample data for backtesting...")
        
        # Create synthetic data for demonstration
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        volumes = np.random.lognormal(10, 0.5, len(dates))
        
        data = pd.DataFrame({
            'open': prices * np.random.uniform(0.98, 1.02, len(dates)),
            'high': prices * np.random.uniform(1.00, 1.05, len(dates)),
            'low': prices * np.random.uniform(0.95, 1.00, len(dates)),
            'close': prices,
            'volume': volumes.astype(int)
        }, index=dates)
        
        # Ensure OHLC logic
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
        
        # Test strategies
        selector = get_strategy_selector()
        strategies_to_test = ['momentum', 'trend_following']
        
        print(f"\nTesting {len(strategies_to_test)} strategies on sample data:")
        
        for strategy_name in strategies_to_test:
            print(f"\n📊 Testing {strategy_name} strategy:")
            
            # Create strategy
            strategy = selector.create_strategy(strategy_name)
            if not strategy:
                print(f"  ❌ Failed to create {strategy_name} strategy")
                continue
            
            # Generate sample signals
            current_date = datetime.now()
            current_prices = {'TEST': data['close'].iloc[-1]}
            current_data = {'TEST': {
                'open': data['open'].iloc[-1],
                'high': data['high'].iloc[-1],
                'low': data['low'].iloc[-1],
                'close': data['close'].iloc[-1],
                'volume': data['volume'].iloc[-1]
            }}
            historical_data = {'TEST': data}
            
            # Generate signals (pass None for portfolio since this is just a demo)
            signals = strategy.generate_signals(
                current_date, current_prices, current_data, historical_data, None
            )
            
            print(f"  Generated {len(signals)} signals")
            if signals:
                for signal in signals[:3]:  # Show first 3 signals
                    print(f"    • {signal.get('action', 'N/A')} {signal.get('quantity', 0)} shares")
                    print(f"      Confidence: {signal.get('confidence', 0):.2%}")
            else:
                print("    No signals generated")
                
    except Exception as e:
        print(f"  ⚠️  Backtest demonstration failed: {e}")
        print("     This is normal if required ML libraries are not installed")

def main():
    """Main demonstration function"""
    
    print("🚀 Starting Advanced Strategies Demonstration")
    print()
    
    try:
        # Run demonstrations
        demonstrate_strategy_selection()
        demonstrate_strategy_creation_and_info()
        demonstrate_market_condition_recommendations()
        demonstrate_strategy_portfolio()
        demonstrate_strategy_compatibility()
        run_mini_backtest_comparison()
        
        print("\n" + "=" * 60)
        print("✅ Advanced Strategies Demonstration Complete!")
        print("\nStrategy Summary:")
        
        selector = get_strategy_selector()
        overview = selector.get_all_strategies_overview()
        
        print(f"  • Total Strategies: {overview['total_strategies']}")
        print(f"  • Classical Strategies: {overview['categories'].get('classical', 0)}")
        print(f"  • ML Strategies: {overview['categories'].get('machine_learning', 0)}")
        print(f"  • High Frequency: {overview['categories'].get('high_frequency', 0)}")
        print(f"  • Low Frequency: {overview['categories'].get('low_frequency', 0)}")
        
        print("\n📚 Next Steps:")
        print("  1. Install optional ML libraries: pip install xgboost tensorflow scikit-learn")
        print("  2. Configure API keys for live data")
        print("  3. Run backtests with real market data")
        print("  4. Optimize strategy parameters")
        print("  5. Create custom strategy portfolios")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()