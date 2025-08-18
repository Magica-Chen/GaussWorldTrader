#!/usr/bin/env python3
"""
Enhanced Backtest Runner with CSV Export
Runs momentum strategy backtest and generates transaction log
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.momentum_backtest_example import momentum_backtest_example

def main():
    print("🚀 Enhanced Momentum Strategy Backtest")
    print("=" * 60)
    print("📊 Features:")
    print("  ✅ Full backtest with performance metrics")
    print("  ✅ P&L plot generation and display")
    print("  ✅ Detailed transaction CSV export")
    print("  ✅ Trading summary and position analysis")
    print("=" * 60)
    
    try:
        momentum_backtest_example()
    except KeyboardInterrupt:
        print("\\n⏹️ Backtest interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error running backtest: {e}")
        print("\\nTroubleshooting:")
        print("1. Check your .env file has valid Alpaca API keys")
        print("2. Ensure all packages are installed: pip install matplotlib pandas")
        print("3. Make sure you're in the correct directory")

if __name__ == '__main__':
    main()