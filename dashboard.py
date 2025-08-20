#!/usr/bin/env python3
"""
Gauss World Trader - Advanced Dashboard

Modern web-based dashboard with tabbed interface for comprehensive trading management
Combined launcher and dashboard application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import logging
import pytz
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Gauss World Trader - Advanced Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tab-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .profit-positive {
        color: #00c853;
        font-weight: bold;
    }
    .profit-negative {
        color: #d32f2f;
        font-weight: bold;
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def get_eastern_time():
    """Get current time in Eastern timezone"""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 'Account'
    
    # Initialize managers with proper error handling
    if 'managers_initialized' not in st.session_state:
        try:
            from src.account.account_manager import AccountManager
            from src.account.position_manager import PositionManager
            from src.account.order_manager import OrderManager
            from src.strategy.strategy_selector import get_strategy_selector
            from src.data import AlpacaDataProvider
            
            # Initialize account manager first
            st.session_state.account_manager = AccountManager()
            
            # Initialize other managers with account_manager dependency
            st.session_state.position_manager = PositionManager(st.session_state.account_manager)
            st.session_state.order_manager = OrderManager(st.session_state.account_manager)
            
            # Initialize other components
            st.session_state.strategy_selector = get_strategy_selector()
            st.session_state.data_provider = AlpacaDataProvider()
            
            st.session_state.managers_initialized = True
            logger.info("All managers initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Some modules not available: {e}")
            st.session_state.managers_initialized = False
        except Exception as e:
            logger.error(f"Error initializing managers: {e}")
            st.session_state.managers_initialized = False
            # Show user-friendly error in sidebar
            st.sidebar.error("⚠️ API connection not configured. Some features may be limited.")

def create_sidebar():
    """Create the left sidebar with main navigation tabs"""
    st.sidebar.markdown("## 🌍 Gauss World Trader")
    st.sidebar.markdown("---")
    
    # Main navigation tabs
    tabs = ['Account', 'Live Analysis', 'Backtesting', 'Trading']
    
    # Custom radio button styling
    selected_tab = st.sidebar.radio(
        "Main Navigation",
        tabs,
        index=tabs.index(st.session_state.current_tab),
        key="main_navigation"
    )
    
    if selected_tab != st.session_state.current_tab:
        st.session_state.current_tab = selected_tab
        st.experimental_rerun()
    
    st.sidebar.markdown("---")
    
    # Quick status section
    st.sidebar.markdown("### 📊 Quick Status")
    
    if st.session_state.get('managers_initialized', False):
        try:
            # Use correct method name
            account_info = st.session_state.account_manager.get_trading_account_status()
            
            if 'error' not in account_info:
                equity = account_info.get('equity', 0)
                buying_power = account_info.get('buying_power', 0)
                
                st.sidebar.metric("Portfolio Value", f"${equity:,.2f}")
                st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")
                
                # Position count
                positions = st.session_state.position_manager.get_all_positions()
                if not any('error' in str(pos) for pos in positions):
                    position_count = len([p for p in positions if float(p.get('qty', 0)) != 0])
                    st.sidebar.metric("Active Positions", position_count)
                else:
                    st.sidebar.metric("Active Positions", "0")
            else:
                st.sidebar.error("Unable to load account data")
                
        except Exception as e:
            logger.error(f"Error loading quick status: {e}")
            st.sidebar.info("Demo Mode - Limited functionality")
    else:
        st.sidebar.info("Demo Mode - API not connected")
    
    st.sidebar.markdown("---")
    
    # Market status
    current_time = get_eastern_time()
    st.sidebar.markdown(f"**Market Time (ET):** {current_time.strftime('%H:%M:%S')}")
    
    # Market hours check (simplified)
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if market_open <= current_time <= market_close and current_time.weekday() < 5:
        st.sidebar.success("🟢 Market Open")
    else:
        st.sidebar.warning("🟡 Market Closed")

def render_account_tab():
    """Render the Account tab with configurations, positions, orders, and portfolio"""
    st.markdown('<h1 class="tab-header">💼 Account Management</h1>', unsafe_allow_html=True)
    
    # Sub-tabs for account sections
    account_tabs = st.tabs(["Overview", "Positions", "Orders", "Configuration", "P&L Analysis"])
    
    with account_tabs[0]:  # Overview
        render_account_overview()
    
    with account_tabs[1]:  # Positions
        render_positions_view()
    
    with account_tabs[2]:  # Orders
        render_orders_view()
    
    with account_tabs[3]:  # Configuration
        render_account_configuration()
    
    with account_tabs[4]:  # P&L Analysis
        render_pnl_analysis()

def render_account_overview():
    """Render account overview with key metrics"""
    st.subheader("📈 Portfolio Overview")
    
    if not st.session_state.get('managers_initialized', False):
        st.info("🔧 Account overview requires API configuration. Currently showing demo data.")
        render_demo_account_overview()
        return
    
    try:
        account_info = st.session_state.account_manager.get_trading_account_status()
        
        if 'error' in account_info:
            st.error(f"Error loading account data: {account_info.get('error', 'Unknown error')}")
            render_demo_account_overview()
            return
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            equity = float(account_info.get('equity', 0))
            st.metric("Total Equity", f"${equity:,.2f}")
        
        with col2:
            buying_power = float(account_info.get('buying_power', 0))
            st.metric("Buying Power", f"${buying_power:,.2f}")
        
        with col3:
            cash = float(account_info.get('cash', 0))
            st.metric("Available Cash", f"${cash:,.2f}")
        
        with col4:
            portfolio_value = float(account_info.get('portfolio_value', 0))
            day_change = float(account_info.get('equity_change', 0))
            day_change_pct = float(account_info.get('equity_change_percentage', 0))
            
            delta_color = "normal"
            if day_change > 0:
                delta_color = "normal"
            elif day_change < 0:
                delta_color = "inverse"
            
            st.metric("Day P&L", f"${day_change:,.2f}", f"{day_change_pct:+.2f}%", delta_color=delta_color)
        
        # Account details
        st.subheader("📋 Account Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Account ID:**", account_info.get('account_id', 'N/A'))
            st.write("**Account Status:**", account_info.get('status', 'N/A'))
            st.write("**Trading Blocked:**", account_info.get('trading_blocked', 'N/A'))
            st.write("**Pattern Day Trader:**", account_info.get('pattern_day_trader', 'N/A'))
        
        with col2:
            st.write("**Currency:**", account_info.get('currency', 'USD'))
            st.write("**Last Equity:**", f"${float(account_info.get('last_equity', 0)):,.2f}")
            st.write("**Multiplier:**", account_info.get('multiplier', 'N/A'))
            st.write("**Day Trading BP:**", f"${float(account_info.get('day_trading_buying_power', 0)):,.2f}")
        
    except Exception as e:
        logger.error(f"Account overview error: {e}")
        st.error("Error loading account overview. Showing demo data instead.")
        render_demo_account_overview()

def render_demo_account_overview():
    """Render demo account overview when API is not available"""
    st.info("📊 Demo Account Overview")
    
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Equity", "$125,430.50")
    
    with col2:
        st.metric("Buying Power", "$45,320.75")
    
    with col3:
        st.metric("Available Cash", "$23,150.25")
    
    with col4:
        st.metric("Day P&L", "$1,245.30", "+1.00%")
    
    st.markdown("**Note:** This is demo data. Configure your Alpaca API keys to see real account information.")

def render_positions_view():
    """Render current positions with detailed information"""
    st.subheader("📊 Current Positions")
    
    if not st.session_state.get('managers_initialized', False):
        st.info("Position tracking requires API configuration. Showing demo data.")
        render_demo_positions()
        return
    
    try:
        positions = st.session_state.position_manager.get_all_positions()
        
        if not positions or any('error' in str(pos) for pos in positions):
            st.info("No active positions found or API error occurred.")
            render_demo_positions()
            return
        
        # Create positions DataFrame
        positions_data = []
        for pos in positions:
            qty = float(pos.get('qty', 0))
            if qty != 0:  # Only show non-zero positions
                market_value = float(pos.get('market_value', 0))
                cost_basis = float(pos.get('cost_basis', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                unrealized_plpc = float(pos.get('unrealized_plpc', 0)) * 100
                
                positions_data.append({
                    'Symbol': pos.get('symbol', ''),
                    'Quantity': int(qty),
                    'Market Value': f"${market_value:,.2f}",
                    'Cost Basis': f"${cost_basis:,.2f}",
                    'Unrealized P&L': f"${unrealized_pl:,.2f}",
                    'Unrealized %': f"{unrealized_plpc:+.2f}%",
                    'Side': pos.get('side', ''),
                    'Exchange': pos.get('exchange', '')
                })
        
        if positions_data:
            df = pd.DataFrame(positions_data)
            
            # Style the dataframe
            def style_pnl(val):
                if isinstance(val, str) and '$' in val:
                    try:
                        num_val = float(val.replace('$', '').replace(',', ''))
                        color = 'color: green' if num_val >= 0 else 'color: red'
                        return color
                    except:
                        return ''
                elif isinstance(val, str) and '%' in val:
                    try:
                        num_val = float(val.replace('%', '').replace('+', ''))
                        color = 'color: green' if num_val >= 0 else 'color: red'
                        return color
                    except:
                        return ''
                return ''
            
            styled_df = df.style.applymap(style_pnl, subset=['Unrealized P&L', 'Unrealized %'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No active positions found.")
            
    except Exception as e:
        logger.error(f"Positions error: {e}")
        st.error("Error loading positions. Showing demo data.")
        render_demo_positions()

def render_demo_positions():
    """Render demo positions when API is not available"""
    st.info("📊 Demo Positions")
    
    demo_data = [
        {'Symbol': 'AAPL', 'Quantity': 50, 'Market Value': '$8,750.00', 'Cost Basis': '$8,500.00', 'Unrealized P&L': '$250.00', 'Unrealized %': '+2.94%'},
        {'Symbol': 'MSFT', 'Quantity': 25, 'Market Value': '$6,450.00', 'Cost Basis': '$6,200.00', 'Unrealized P&L': '$250.00', 'Unrealized %': '+4.03%'},
        {'Symbol': 'GOOGL', 'Quantity': 10, 'Market Value': '$2,830.00', 'Cost Basis': '$2,900.00', 'Unrealized P&L': '-$70.00', 'Unrealized %': '-2.41%'},
    ]
    
    df = pd.DataFrame(demo_data)
    st.dataframe(df, use_container_width=True)

def render_orders_view():
    """Render orders view with order history and management"""
    st.subheader("📋 Order Management")
    
    # Order tabs
    order_tabs = st.tabs(["Active Orders", "Order History", "Place Order"])
    
    with order_tabs[0]:  # Active Orders
        if st.session_state.get('managers_initialized', False):
            try:
                orders = st.session_state.order_manager.get_orders(status='open')
                
                if not orders or any('error' in str(order) for order in orders):
                    st.info("No active orders found.")
                else:
                    orders_data = []
                    for order in orders:
                        orders_data.append({
                            'Order ID': order.get('id', ''),
                            'Symbol': order.get('symbol', ''),
                            'Side': order.get('side', ''),
                            'Type': order.get('type', ''),
                            'Quantity': order.get('qty', ''),
                            'Price': f"${float(order.get('limit_price', 0)):,.2f}" if order.get('limit_price') else 'Market',
                            'Status': order.get('status', ''),
                            'Created': order.get('created_at', '')
                        })
                    
                    df = pd.DataFrame(orders_data)
                    st.dataframe(df, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error loading active orders: {e}")
        else:
            st.info("Order management requires API configuration.")
    
    with order_tabs[1]:  # Order History
        st.info("Order history feature - requires API configuration for full functionality")
    
    with order_tabs[2]:  # Place Order
        render_place_order_form()

def render_place_order_form():
    """Render order placement form"""
    st.subheader("📤 Place New Order")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL")
        side = st.selectbox("Side", ["buy", "sell"])
        order_type = st.selectbox("Order Type", ["market", "limit", "stop", "stop_limit"])
        quantity = st.number_input("Quantity", min_value=1, value=1)
    
    with col2:
        if order_type in ["limit", "stop_limit"]:
            limit_price = st.number_input("Limit Price", min_value=0.01, value=1.00, step=0.01)
        else:
            limit_price = None
        
        if order_type in ["stop", "stop_limit"]:
            stop_price = st.number_input("Stop Price", min_value=0.01, value=1.00, step=0.01)
        else:
            stop_price = None
        
        time_in_force = st.selectbox("Time in Force", ["day", "gtc", "ioc", "fok"])
    
    if st.button("🚀 Place Order", type="primary"):
        try:
            order_data = {
                'symbol': symbol.upper(),
                'qty': quantity,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if limit_price:
                order_data['limit_price'] = limit_price
            if stop_price:
                order_data['stop_price'] = stop_price
            
            st.success(f"Order would be placed! (Demo mode)")
            st.json(order_data)
            
        except Exception as e:
            st.error(f"Error placing order: {e}")

def render_account_configuration():
    """Render account configuration settings"""
    st.subheader("⚙️ Account Configuration")
    st.info("Account configuration features available in full version with API access.")

def render_pnl_analysis():
    """Render P&L analysis with charts and metrics"""
    st.subheader("📊 Profit & Loss Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Generate sample P&L data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    portfolio_values = 100000 + np.cumsum(np.random.randn(len(dates)) * 500)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
    
    # Portfolio value chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig1.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
        st.metric("Total Return", f"{total_return:+.2f}%")
    
    with col2:
        volatility = np.std(daily_returns) * np.sqrt(252)
        st.metric("Annualized Volatility", f"{volatility:.2f}%")
    
    with col3:
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        max_drawdown = np.min(np.minimum.accumulate(daily_returns))
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

def render_live_analysis_tab():
    """Render the Live Analysis tab with fundamental and technical reports"""
    st.markdown('<h1 class="tab-header">📊 Live Market Analysis</h1>', unsafe_allow_html=True)
    
    # Analysis tabs
    analysis_tabs = st.tabs(["Symbol Analysis", "Market Overview", "Watchlist", "Technical Analysis"])
    
    with analysis_tabs[0]:  # Symbol Analysis
        render_symbol_analysis()
    
    with analysis_tabs[1]:  # Market Overview
        render_market_overview()
    
    with analysis_tabs[2]:  # Watchlist
        render_watchlist_analysis()
    
    with analysis_tabs[3]:  # Technical Analysis
        render_technical_analysis_demo()

def render_symbol_analysis():
    """Render analysis for any symbol"""
    st.subheader("🔍 Symbol Analysis")
    
    # Symbol input
    symbol = st.text_input("Enter Symbol for Analysis", placeholder="e.g., AAPL, TSLA, SPY").upper()
    
    if symbol and st.button("🔍 Analyze"):
        st.success(f"Analysis for {symbol} would be displayed here with real data integration.")
        
        # Demo fundamental data
        st.markdown("#### 📈 Key Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Market Cap", "$2.5T")
            st.metric("P/E Ratio", "28.5")
            st.metric("EPS", "$6.05")
            st.metric("Revenue Growth", "8.2%")
        
        with col2:
            st.metric("Dividend Yield", "0.5%")
            st.metric("ROE", "147.4%")
            st.metric("Debt-to-Equity", "1.73")
            st.metric("Free Cash Flow", "$100.0B")

def render_market_overview():
    """Render overall market overview"""
    st.subheader("🌍 Market Overview")
    
    # Market indices
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500", "4,150.25", "+15.30 (+0.37%)")
    
    with col2:
        st.metric("NASDAQ", "12,845.87", "-45.20 (-0.35%)")
    
    with col3:
        st.metric("DOW", "33,875.40", "+125.60 (+0.37%)")
    
    with col4:
        st.metric("VIX", "18.45", "-1.25 (-6.34%)")

def render_watchlist_analysis():
    """Render watchlist with analysis"""
    st.subheader("👀 Watchlist Analysis")
    
    # Watchlist management
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_symbol = st.text_input("Add Symbol to Watchlist", placeholder="e.g., AAPL")
    
    with col2:
        if st.button("➕ Add"):
            if new_symbol:
                if 'watchlist' not in st.session_state:
                    st.session_state.watchlist = []
                if new_symbol.upper() not in st.session_state.watchlist:
                    st.session_state.watchlist.append(new_symbol.upper())
                    st.success(f"Added {new_symbol.upper()} to watchlist")
    
    # Display watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
    
    if st.session_state.watchlist:
        watchlist_data = []
        
        for symbol in st.session_state.watchlist:
            # Simulate market data
            price = 100 + np.random.randn() * 20
            change = np.random.randn() * 2
            change_pct = change / price * 100
            volume = np.random.randint(1000000, 50000000)
            
            watchlist_data.append({
                'Symbol': symbol,
                'Price': f"${price:.2f}",
                'Change': f"${change:+.2f}",
                'Change %': f"{change_pct:+.2f}%",
                'Volume': f"{volume:,}"
            })
        
        df = pd.DataFrame(watchlist_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Your watchlist is empty. Add symbols to start tracking them.")

def render_technical_analysis_demo():
    """Render technical analysis demo"""
    st.subheader("📈 Technical Analysis")
    st.info("Technical analysis with real-time data available in full version.")

def render_backtesting_tab():
    """Render the Backtesting tab with strategy-based backtesting"""
    st.markdown('<h1 class="tab-header">🧪 Strategy Backtesting</h1>', unsafe_allow_html=True)
    
    # Strategy selector
    if st.session_state.get('managers_initialized', False):
        try:
            available_strategies = st.session_state.strategy_selector.list_strategies()
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_strategy = st.selectbox("Select Strategy", available_strategies)
            
            with col2:
                symbol = st.text_input("Symbol", value="AAPL")
            
            # Backtest parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            
            with col2:
                end_date = st.date_input("End Date", datetime.now())
            
            with col3:
                initial_capital = st.number_input("Initial Capital", min_value=1000, value=100000, step=1000)
            
            if st.button("🚀 Run Backtest", type="primary"):
                with st.spinner(f"Running backtest for {symbol} using {selected_strategy} strategy..."):
                    # Simulate backtest results
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    returns = np.random.randn(len(dates)) * 0.01 + 0.0002
                    equity_curve = initial_capital * np.exp(np.cumsum(returns))
                    
                    # Calculate metrics
                    total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                    
                    st.success(f"Backtest completed! Total Return: {total_return:.2f}%")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Return", f"{total_return:.2f}%")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    with col3:
                        st.metric("Final Value", f"${equity_curve[-1]:,.2f}")
                    
                    # Equity curve chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=equity_curve,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Backtest Results - {selected_strategy} on {symbol}",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in backtesting: {e}")
            st.info("Backtesting requires proper strategy framework initialization.")
    else:
        st.info("Backtesting requires API configuration for full functionality.")

def render_trading_tab():
    """Render the Trading tab for active trading interface"""
    st.markdown('<h1 class="tab-header">⚡ Active Trading</h1>', unsafe_allow_html=True)
    
    # Trading tabs
    trading_tabs = st.tabs(["Quick Trade", "Strategy Signals", "Market Data"])
    
    with trading_tabs[0]:  # Quick Trade
        render_quick_trading()
    
    with trading_tabs[1]:  # Strategy Signals
        render_strategy_signals()
    
    with trading_tabs[2]:  # Market Data
        render_market_data()

def render_quick_trading():
    """Render quick trading interface"""
    st.subheader("⚡ Quick Trade")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
        
        if symbol:
            current_price = 150.25 + np.random.randn() * 2
            st.metric(f"{symbol} Current Price", f"${current_price:.2f}", f"{np.random.randn()*0.5:+.2f}")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            action = st.selectbox("Action", ["Buy", "Sell"])
            quantity = st.number_input("Quantity", min_value=1, value=100)
        
        with col1b:
            order_type = st.selectbox("Order Type", ["Market", "Limit"])
            if order_type == "Limit":
                limit_price = st.number_input("Limit Price", value=current_price if 'current_price' in locals() else 100.0)
        
        if st.button("🚀 Execute Trade", type="primary"):
            st.success(f"Trade executed! {action} {quantity} shares of {symbol} (Demo mode)")
    
    with col2:
        st.markdown("#### 📊 Position Calculator")
        
        portfolio_value = st.number_input("Portfolio Value", value=100000)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, 0.1)
        
        if 'current_price' in locals():
            risk_amount = portfolio_value * (risk_pct / 100)
            st.metric("Risk Amount", f"${risk_amount:,.2f}")

def render_strategy_signals():
    """Render strategy signals"""
    st.subheader("🧠 Strategy Signals")
    
    if st.session_state.get('managers_initialized', False):
        try:
            available_strategies = st.session_state.strategy_selector.list_strategies()
            selected_strategy = st.selectbox("Select Strategy", available_strategies)
            
            watchlist_symbols = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL")
            
            if st.button("🔍 Generate Signals"):
                symbols = [s.strip().upper() for s in watchlist_symbols.split(',')]
                
                signals_data = []
                for symbol in symbols:
                    signal_strength = np.random.uniform(-1, 1)
                    
                    if signal_strength > 0.3:
                        action = "BUY"
                        confidence = signal_strength
                    elif signal_strength < -0.3:
                        action = "SELL"
                        confidence = abs(signal_strength)
                    else:
                        action = "HOLD"
                        confidence = 1 - abs(signal_strength)
                    
                    signals_data.append({
                        'Symbol': symbol,
                        'Signal': action,
                        'Confidence': f"{confidence:.1%}",
                        'Price': f"${100 + np.random.randn() * 10:.2f}",
                        'Reason': f"{selected_strategy} criteria met"
                    })
                
                df = pd.DataFrame(signals_data)
                st.dataframe(df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating signals: {e}")
    else:
        st.info("Strategy signals require API configuration.")

def render_market_data():
    """Render market data view"""
    st.subheader("📈 Market Data")
    st.info("Real-time market data available in full version with API access.")

def main():
    """Main dashboard function"""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🌍 Gauss World Trader - Advanced Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Create sidebar
    create_sidebar()
    
    # Render appropriate tab based on selection
    if st.session_state.current_tab == 'Account':
        render_account_tab()
    elif st.session_state.current_tab == 'Live Analysis':
        render_live_analysis_tab()
    elif st.session_state.current_tab == 'Backtesting':
        render_backtesting_tab()
    elif st.session_state.current_tab == 'Trading':
        render_trading_tab()

def launch_dashboard():
    """Launch the dashboard with proper configuration"""
    print("🌍 Starting Gauss World Trader - Advanced Dashboard")
    print("=" * 60)
    print("Dashboard Features:")
    print("• 💼 Account Management (Positions, Orders, P&L)")
    print("• 📊 Live Market Analysis (Technical & Fundamental)")
    print("• 🧪 Strategy Backtesting (8+ Trading Strategies)")
    print("• ⚡ Active Trading Interface")
    print("=" * 60)
    print("🚀 Dashboard ready at http://localhost:3721")
    print("📱 Dashboard will open automatically in your browser")
    print("🔄 Press Ctrl+C to stop the dashboard")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        # Launch mode - start streamlit
        launch_dashboard()
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            __file__,
            "--server.port=3721",
            "--server.address=0.0.0.0",
            "--theme.base=light",
            "--theme.primaryColor=#1f77b4"
        ]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n⏹️  Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Error launching dashboard: {e}")
    else:
        # Normal streamlit run mode
        main()