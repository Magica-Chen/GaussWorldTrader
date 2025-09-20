#!/usr/bin/env python3
"""
Gauss World Trader - Dashboard Launcher

Root-level dashboard launcher for easy access alongside main.py
Supports simple and modern dashboard interfaces
All dashboard implementations are in src/ui/ directory
"""

import sys
import os
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_dashboard(mode="modern"):
    """Launch dashboard using streamlit run command"""
    # Set the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Add project to Python path
    sys.path.insert(0, str(project_root))

    # Mode-specific configurations
    if mode == "simple":
        dashboard_file = "src/ui/simple_dashboard.py"
        print("🔹 Starting Simple Dashboard via Streamlit...")
    elif mode == "wheel":
        dashboard_file = "src/ui/wheel_dashboard.py"
        print("🎯 Starting Wheel Strategy Dashboard via Streamlit...")
    else:  # modern (default)
        dashboard_file = "src/ui/modern_dashboard.py"
        print("🌍 Starting Modern Dashboard via Streamlit...")
    
    try:
        # Launch using streamlit run command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            dashboard_file,
            "--server.port=3721",
            "--server.address=localhost",
            "--theme.base=light",
            "--theme.primaryColor=#1f77b4",
            "--theme.backgroundColor=#ffffff", 
            "--theme.secondaryBackgroundColor=#f0f2f6",
            "--theme.textColor=#262730"
        ]
        
        print(f"🚀 Dashboard will open at http://localhost:3721")
        print("🔄 Press Ctrl+C to stop the dashboard")
        print()
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

def launch_dashboard(mode="modern"):
    """Launch the dashboard with proper configuration"""
    # Set the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Add project to Python path
    sys.path.insert(0, str(project_root))

    # Mode-specific configurations
    if mode == "simple":
        dashboard_file = "src/ui/simple_dashboard.py"
        print("🔹 Starting Gauss World Trader - Simple Dashboard")
        print("=" * 60)
        print("Dashboard Features:")
        print("• 📈 Market Analysis with Technical Indicators")
        print("• 📊 Strategy Backtesting")
        print("• 💼 Account Overview")
        print("• 🔄 Trading Interface")
        print("• 📰 News & Sentiment Analysis")
        print("• ₿ Cryptocurrency Data")
        print("=" * 60)
    elif mode == "wheel":
        dashboard_file = "src/ui/wheel_dashboard.py"
        print("🎯 Starting Gauss World Trader - Wheel Strategy Dashboard")
        print("=" * 60)
        print("Dashboard Features:")
        print("• 🎯 Wheel Strategy Overview & Cycle Monitoring")
        print("• 📊 Real-time Signal Generation & Analysis")
        print("• 📈 Option Position Management & Risk Assessment")
        print("• ⚙️ Strategy Configuration & Parameter Tuning")
        print("• 📚 Educational Content & Strategy Explanation")
        print("• 🛡️ Risk Management & Assignment Monitoring")
        print("=" * 60)
    else:  # modern (default)
        dashboard_file = "src/ui/modern_dashboard.py"
        print("🌍 Starting Gauss World Trader - Modern Dashboard")
        print("=" * 60)
        print("Dashboard Features:")
        print("• 📊 Market Overview (Indices, VIX, Sectors, Calendar, Crypto)")
        print("• 💼 Account Info (Account, Positions, Portfolio, Performance, Config)")
        print("• 🔍 Live Analysis (Symbol Analysis, Watchlist)")
        print("• 📈 Strategy Backtest (Quick Backtest, Strategy Comparison)")
        print("• ⚡ Trade & Order (Quick Trade, Active Orders, Order History)")
        print("• 📰 News & Report (Company News, Insider Activity, AI Reports)")
        print("=" * 60)
    
    try:
        # Launch Streamlit dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            dashboard_file,
            "--server.port=3721",
            "--server.address=localhost",  # Fixed to localhost only
            "--theme.base=light",
            "--theme.primaryColor=#1f77b4",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f0f2f6",
            "--theme.textColor=#262730"
        ]
        
        print(f"🚀 Launching dashboard on http://localhost:3721")
        print("📱 Open your browser and navigate to the URL above")
        print("🔄 Press Ctrl+C to stop the dashboard")
        print()
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Gauss World Trader Dashboard',
        epilog="""
Examples:
  python dashboard.py                    # Launch modern dashboard (default)
  python dashboard.py --simple          # Launch simple dashboard
  python dashboard.py --wheel           # Launch wheel strategy dashboard
  python dashboard.py launch --wheel    # Launch wheel dashboard with enhanced config
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('command', nargs='?', default='run', choices=['launch', 'run'],
                       help='Command to execute (launch or run)')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple dashboard interface (includes crypto, news, and technical analysis)')
    parser.add_argument('--modern', action='store_true',
                       help='Use modern dashboard interface (default - redesigned navigation structure)')
    parser.add_argument('--wheel', action='store_true',
                       help='Use wheel strategy dashboard interface (options trading focus)')

    args = parser.parse_args()

    # Determine dashboard mode
    if args.simple:
        mode = "simple"
    elif args.wheel:
        mode = "wheel"
    else:
        mode = "modern"  # default
    
    # Execute command
    if args.command == "launch":
        # Launch mode - start streamlit with enhanced configuration
        launch_dashboard(mode)
    else:
        # Normal streamlit run mode - run dashboard directly
        run_dashboard(mode)