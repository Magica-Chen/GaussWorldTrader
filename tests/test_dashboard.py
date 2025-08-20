#!/usr/bin/env python3
"""
Test script for the Advanced Dashboard

Verify that all dashboard components work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all required imports for the dashboard"""
    print("🧪 Testing Dashboard Imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError:
        print("❌ Plotly not found. Install with: pip install plotly")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
    except ImportError:
        print("❌ Pandas/NumPy not found. Install with: pip install pandas numpy")
        return False
    
    try:
        import pytz
        print("✅ PyTZ imported successfully")
    except ImportError:
        print("❌ PyTZ not found. Install with: pip install pytz")
        return False
    
    return True

def test_project_modules():
    """Test project-specific module imports"""
    print("\n🔧 Testing Project Module Imports...")
    
    try:
        from src.account.account_manager import AccountManager
        print("✅ AccountManager imported successfully")
    except ImportError as e:
        print(f"⚠️  AccountManager import warning: {e}")
    
    try:
        from src.account.position_manager import PositionManager
        print("✅ PositionManager imported successfully")
    except ImportError as e:
        print(f"⚠️  PositionManager import warning: {e}")
    
    try:
        from src.account.order_manager import OrderManager
        print("✅ OrderManager imported successfully")
    except ImportError as e:
        print(f"⚠️  OrderManager import warning: {e}")
    
    try:
        from src.strategy.strategy_selector import get_strategy_selector
        print("✅ StrategySelector imported successfully")
    except ImportError as e:
        print(f"⚠️  StrategySelector import warning: {e}")
    
    try:
        from src.agent.fundamental_analyzer import FundamentalAnalyzer
        print("✅ FundamentalAnalyzer imported successfully")
    except ImportError as e:
        print(f"⚠️  FundamentalAnalyzer import warning: {e}")
    
    return True

def test_strategy_framework():
    """Test the strategy framework functionality"""
    print("\n🎯 Testing Strategy Framework...")
    
    try:
        from src.strategy.strategy_selector import get_strategy_selector
        
        selector = get_strategy_selector()
        strategies = selector.list_strategies()
        
        print(f"✅ Found {len(strategies)} available strategies:")
        for strategy in strategies:
            print(f"   • {strategy}")
        
        # Test strategy creation
        if strategies:
            test_strategy = strategies[0]
            strategy_instance = selector.create_strategy(test_strategy)
            if strategy_instance:
                print(f"✅ Successfully created {test_strategy} strategy instance")
                
                # Test strategy info
                info = strategy_instance.get_strategy_info()
                print(f"   Strategy type: {info.get('type', 'Unknown')}")
                print(f"   Risk level: {info.get('risk_level', 'Unknown')}")
            else:
                print(f"⚠️  Failed to create {test_strategy} strategy instance")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy framework test failed: {e}")
        return False

def test_dashboard_file():
    """Test that the dashboard file exists and is valid"""
    print("\n📊 Testing Dashboard File...")
    
    dashboard_path = project_root / "src" / "ui" / "advanced_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    print(f"✅ Dashboard file found: {dashboard_path}")
    
    # Try to import the dashboard module (basic syntax check)
    try:
        sys.path.insert(0, str(dashboard_path.parent))
        import advanced_dashboard
        print("✅ Dashboard module imports correctly")
        
        # Check for main functions
        required_functions = [
            'main',
            'render_account_tab',
            'render_live_analysis_tab',
            'render_backtesting_tab',
            'render_trading_tab'
        ]
        
        for func_name in required_functions:
            if hasattr(advanced_dashboard, func_name):
                print(f"✅ Found function: {func_name}")
            else:
                print(f"⚠️  Missing function: {func_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard module import failed: {e}")
        return False

def main():
    """Run all dashboard tests"""
    print("🌍 Gauss World Trader - Dashboard Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_project_modules()
    all_passed &= test_strategy_framework()
    all_passed &= test_dashboard_file()
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ All dashboard tests passed!")
        print("\n🚀 You can now run the dashboard with:")
        print("   python run_dashboard.py")
        print("\n📋 Or manually with:")
        print("   streamlit run src/ui/advanced_dashboard.py --server.port=3721")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        print("\n📥 Install missing dependencies with:")
        print("   pip install -r requirements_dashboard.txt")
    
    print("\n🌟 Dashboard Features Available:")
    print("   • Account Management (Positions, Orders, P&L)")
    print("   • Live Market Analysis (Technical & Fundamental)")
    print("   • Strategy Backtesting (8+ Trading Strategies)")
    print("   • Active Trading Interface")
    print("   • Market Overview and Watchlists")

if __name__ == "__main__":
    main()