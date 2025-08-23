#!/usr/bin/env python3
"""
Simple working example of the trading system
Tests core functionality without complex dependencies
"""

import sys
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, '.')

def test_config():
    """Test configuration loading"""
    print("🔧 Testing configuration...")
    try:
        from config import Config
        
        # Test basic config
        alpaca_valid = Config.validate_alpaca_config()
        finhub_valid = Config.validate_finhub_config() 
        fred_valid = Config.validate_fred_config()
        
        print(f"  Alpaca API: {'✅ Valid' if alpaca_valid else '❌ Missing/Invalid'}")
        print(f"  Finhub API: {'✅ Valid' if finhub_valid else '❌ Missing/Invalid'}")
        print(f"  FRED API: {'✅ Valid' if fred_valid else '❌ Missing/Invalid'}")
        
        return alpaca_valid
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False

def test_data_provider(use_alpaca=True):
    """Test data provider"""
    print("\n📊 Testing data provider...")
    
    try:
        if use_alpaca:
            from src.data import AlpacaDataProvider
            provider = AlpacaDataProvider()
        else:
            # Mock data provider for testing
            print("  Using mock data (no API keys needed)")
            import pandas as pd
            import numpy as np
            
            class MockProvider:
                def get_bars(self, symbol, timeframe, start_date, end_date):
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    np.random.seed(42)  # For reproducible results
                    
                    data = {
                        'open': 100 + np.random.randn(len(dates)).cumsum(),
                        'high': 102 + np.random.randn(len(dates)).cumsum(),
                        'low': 98 + np.random.randn(len(dates)).cumsum(), 
                        'close': 101 + np.random.randn(len(dates)).cumsum(),
                        'volume': np.random.randint(10000, 100000, len(dates))
                    }
                    
                    df = pd.DataFrame(data, index=dates)
                    # Ensure high >= max(open, close) and low <= min(open, close)
                    df['high'] = df[['open', 'close', 'high']].max(axis=1)
                    df['low'] = df[['open', 'close', 'low']].min(axis=1)
                    return df
            
            provider = MockProvider()
        
        # Test data fetching
        symbol = 'AAPL'
        end_date = datetime.now() - timedelta(days=1)  # Avoid weekend issues
        start_date = end_date - timedelta(days=30)
        
        print(f"  Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
        
        data = provider.get_bars(symbol, '1Day', start_date, end_date)
        
        if not data.empty:
            print(f"  ✅ Retrieved {len(data)} bars")
            print(f"  📈 Latest price: ${data['close'].iloc[-1]:.2f}")
            print(f"  📊 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
            return data
        else:
            print("  ❌ No data retrieved")
            return None
            
    except Exception as e:
        print(f"  ❌ Data provider error: {e}")
        print("  💡 Trying with mock data instead...")
        return test_data_provider(use_alpaca=False)

def test_strategy(data):
    """Test trading strategy"""
    print("\n🧠 Testing momentum strategy...")
    
    try:
        from src.strategy import MomentumStrategy
        from src.trade import Portfolio
        
        strategy = MomentumStrategy()
        portfolio = Portfolio()
        
        # Prepare data
        symbol = 'AAPL'
        current_prices = {symbol: data['close'].iloc[-1]}
        historical_data = {symbol: data}
        current_data = {
            symbol: {
                'open': data['open'].iloc[-1],
                'high': data['high'].iloc[-1],
                'low': data['low'].iloc[-1],
                'close': data['close'].iloc[-1],
                'volume': data['volume'].iloc[-1]
            }
        }
        
        # Generate signals
        signals = strategy.generate_signals(
            current_date=datetime.now(),
            current_prices=current_prices,
            current_data=current_data,
            historical_data=historical_data,
            portfolio=portfolio
        )
        
        if signals:
            print(f"  ✅ Generated {len(signals)} signals:")
            for signal in signals:
                print(f"    📊 {signal['symbol']}: {signal['action'].upper()} {signal['quantity']} shares")
                print(f"       Reason: {signal.get('reason', 'N/A')}")
                print(f"       Confidence: {signal.get('confidence', 0):.1%}")
        else:
            print("  📭 No trading signals generated (normal for current market conditions)")
            
        return signals
        
    except Exception as e:
        print(f"  ❌ Strategy error: {e}")
        return None

def test_backtesting(data):
    """Test simple backtesting"""
    print("\n🔄 Testing backtesting framework...")
    
    try:
        from src.trade import Backtester
        from src.strategy import MomentumStrategy
        
        # Create backtester
        backtester = Backtester(initial_cash=10000, commission=0.01)
        backtester.add_data('AAPL', data)
        
        # Create strategy
        strategy = MomentumStrategy()
        
        def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
            return strategy.generate_signals(
                current_date, current_prices, current_data, historical_data, portfolio
            )
        
        # Run backtest on subset of data
        start_date = data.index[20]  # Skip first 20 days for indicators
        end_date = data.index[-1]
        
        print(f"  Running backtest from {start_date.date()} to {end_date.date()}")
        
        results = backtester.run_backtest(
            strategy_func,
            start_date=start_date,
            end_date=end_date,
            symbols=['AAPL']
        )
        
        if results:
            print("  ✅ Backtest completed successfully!")
            print(f"    💰 Initial Value: ${results['initial_value']:,.2f}")
            print(f"    💰 Final Value: ${results['final_value']:,.2f}")
            print(f"    📈 Total Return: {results['total_return_percentage']:.2f}%")
            print(f"    📊 Total Trades: {results['total_trades']}")
            print(f"    🎯 Win Rate: {results['win_rate']:.1f}%")
        else:
            print("  ❌ Backtest failed")
            
        return results
        
    except Exception as e:
        print(f"  ❌ Backtesting error: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 Trading System Test Suite")
    print("=" * 50)
    
    # Test configuration
    has_api_keys = test_config()
    
    # Test data provider
    data = test_data_provider(use_alpaca=has_api_keys)
    
    if data is not None:
        # Test strategy
        test_strategy(data)
        
        # Test backtesting
        test_backtesting(data)
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")
    print("\n💡 To run individual components:")
    print("  python simple_example.py")
    print("  python examples/momentum_backtest_example.py  # (needs API keys)")

if __name__ == '__main__':
    main()