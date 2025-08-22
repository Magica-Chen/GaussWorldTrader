#!/usr/bin/env python3
"""
Advanced Trading Dashboard

Modern web-based dashboard with tabbed interface for comprehensive trading management
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
EASTERN = pytz.timezone('US/Eastern')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.account.account_manager import AccountManager
from src.account.position_manager import PositionManager
from src.account.order_manager import OrderManager
from src.agent.fundamental_analyzer import FundamentalAnalyzer
from src.strategy.strategy_selector import get_strategy_selector
from src.backtest import Backtester, Portfolio
from src.data import AlpacaDataProvider
from src.utils.watchlist_manager import WatchlistManager

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

def get_local_time():
    """Get current time in local timezone for display purposes"""
    return datetime.now()

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 'Account'
    if 'account_manager' not in st.session_state:
        try:
            st.session_state.account_manager = AccountManager()
            st.session_state.position_manager = PositionManager(st.session_state.account_manager)
            st.session_state.order_manager = OrderManager(st.session_state.account_manager)
            st.session_state.fundamental_analyzer = FundamentalAnalyzer()
            st.session_state.strategy_selector = get_strategy_selector()
            st.session_state.data_provider = AlpacaDataProvider()
            st.session_state.watchlist_manager = WatchlistManager()
            
            # Get subscription info for account tier display
            st.session_state.subscription_info = st.session_state.data_provider.get_subscription_info()
            
        except Exception as e:
            logger.error(f"Error initializing managers: {e}")
            st.error("Error initializing trading modules. Please check API configuration.")

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
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Quick status section
    st.sidebar.markdown("### 📊 Quick Status")
    
    try:
        # Account status
        account_info = st.session_state.account_manager.get_trading_account_status()
        equity = account_info.get('equity', 0)
        buying_power = account_info.get('buying_power', 0)
        
        st.sidebar.metric("Portfolio Value", f"${equity:,.2f}")
        st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")
        
        # Position count
        positions = st.session_state.position_manager.get_all_positions()
        position_count = len([p for p in positions if p.get('qty', 0) != 0])
        st.sidebar.metric("Active Positions", position_count)
        
    except Exception as e:
        st.sidebar.error("Unable to load account status")
        logger.error(f"Error loading quick status: {e}")
    
    st.sidebar.markdown("---")
    
    # Market status
    current_time = datetime.now(EASTERN)
    local_time = get_local_time()
    st.sidebar.markdown(f"**Local Time:** {local_time.strftime('%H:%M:%S')}")
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
    
    try:
        account_info = st.session_state.account_manager.get_trading_account_status()
        
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
            day_change = float(account_info.get('unrealized_pl', 0))
            day_change_pct = (day_change / portfolio_value * 100) if portfolio_value > 0 else 0
            
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
            st.write("**Account Number:**", account_info.get('account_number', 'N/A'))
            st.write("**Account Status:**", account_info.get('status', 'N/A'))
            st.write("**Trading Status:**", account_info.get('trading_blocked', 'N/A'))
            st.write("**Pattern Day Trader:**", account_info.get('pattern_day_trader', 'N/A'))
        
        with col2:
            st.write("**Currency:**", account_info.get('currency', 'USD'))
            st.write("**Last Equity:**", f"${float(account_info.get('last_equity', 0)):,.2f}")
            st.write("**Initial Margin:**", f"${float(account_info.get('initial_margin', 0)):,.2f}")
            st.write("**Maintenance Margin:**", f"${float(account_info.get('maintenance_margin', 0)):,.2f}")
        
    except Exception as e:
        st.error(f"Error loading account overview: {e}")
        logger.error(f"Account overview error: {e}")

def render_positions_view():
    """Render current positions with detailed information"""
    st.subheader("📊 Current Positions")
    
    try:
        positions = st.session_state.position_manager.get_all_positions()
        
        if not positions:
            st.info("No active positions found.")
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
                    num_val = float(val.replace('$', '').replace(',', ''))
                    color = 'color: green' if num_val >= 0 else 'color: red'
                    return color
                elif isinstance(val, str) and '%' in val:
                    num_val = float(val.replace('%', '').replace('+', ''))
                    color = 'color: green' if num_val >= 0 else 'color: red'
                    return color
                return ''
            
            styled_df = df.style.map(style_pnl, subset=['Unrealized P&L', 'Unrealized %'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No active positions found.")
            
    except Exception as e:
        st.error(f"Error loading positions: {e}")
        logger.error(f"Positions error: {e}")

def render_orders_view():
    """Render orders view with order history and management"""
    st.subheader("📋 Order Management")
    
    # Order tabs
    order_tabs = st.tabs(["Active Orders", "Order History", "Place Order"])
    
    with order_tabs[0]:  # Active Orders
        try:
            orders = st.session_state.order_manager.get_orders(status='open')
            
            if not orders:
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
                
                # Cancel order functionality
                if st.button("🗑️ Cancel Selected Orders"):
                    st.info("Order cancellation functionality would be implemented here")
        
        except Exception as e:
            st.error(f"Error loading active orders: {e}")
    
    with order_tabs[1]:  # Order History
        try:
            # Date range selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30), key="order_history_start_date")
            with col2:
                end_date = st.date_input("End Date", datetime.now(), key="order_history_end_date")
            
            if st.button("Load Order History"):
                orders = st.session_state.order_manager.get_orders(
                    status='all',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if orders:
                    orders_data = []
                    for order in orders:
                        orders_data.append({
                            'Date': order.get('created_at', ''),
                            'Symbol': order.get('symbol', ''),
                            'Side': order.get('side', ''),
                            'Quantity': order.get('qty', ''),
                            'Price': f"${float(order.get('filled_avg_price', 0)):,.2f}" if order.get('filled_avg_price') else 'N/A',
                            'Status': order.get('status', ''),
                            'Type': order.get('type', '')
                        })
                    
                    df = pd.DataFrame(orders_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No orders found for the selected date range.")
        
        except Exception as e:
            st.error(f"Error loading order history: {e}")
    
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
        quantity = st.number_input("Quantity", min_value=1, value=1, key="order_quantity")
    
    with col2:
        if order_type in ["limit", "stop_limit"]:
            limit_price = st.number_input("Limit Price", min_value=0.01, value=1.00, step=0.01, key="order_limit_price")
        else:
            limit_price = None
        
        if order_type in ["stop", "stop_limit"]:
            stop_price = st.number_input("Stop Price", min_value=0.01, value=1.00, step=0.01, key="order_stop_price")
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
            
            # Here you would place the actual order
            st.success(f"Order placed successfully! (Demo mode)")
            st.json(order_data)
            
        except Exception as e:
            st.error(f"Error placing order: {e}")

def render_account_configuration():
    """Render account configuration settings"""
    st.subheader("⚙️ Account Configuration")
    
    try:
        config = st.session_state.account_manager.get_account_configurations()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trading Configuration**")
            
            day_trading = st.checkbox("Day Trading", value=config.get('day_trading_enabled', False))
            options_trading = st.checkbox("Options Trading", value=config.get('options_trading_enabled', False))
            crypto_trading = st.checkbox("Crypto Trading", value=config.get('crypto_trading_enabled', False))
            
            max_margin = st.number_input("Max Margin Multiplier", 
                                       min_value=1.0, max_value=4.0, 
                                       value=float(config.get('max_margin_multiplier', 2.0)),
                                       step=0.1)
        
        with col2:
            st.markdown("**Risk Management**")
            
            day_trade_limit = st.number_input("Day Trade Limit", 
                                            min_value=0, 
                                            value=int(config.get('day_trade_limit', 3)))
            
            position_limit = st.number_input("Max Position Size (%)", 
                                           min_value=1, max_value=100,
                                           value=int(config.get('max_position_size_pct', 25)))
            
            stop_loss_default = st.number_input("Default Stop Loss (%)", 
                                              min_value=0.1, max_value=50.0,
                                              value=float(config.get('default_stop_loss_pct', 5.0)),
                                              step=0.1)
        
        if st.button("💾 Save Configuration"):
            new_config = {
                'day_trading_enabled': day_trading,
                'options_trading_enabled': options_trading,
                'crypto_trading_enabled': crypto_trading,
                'max_margin_multiplier': max_margin,
                'day_trade_limit': day_trade_limit,
                'max_position_size_pct': position_limit,
                'default_stop_loss_pct': stop_loss_default
            }
            
            # Here you would save the configuration
            st.success("Configuration saved successfully! (Demo mode)")
            st.json(new_config)
    
    except Exception as e:
        st.error(f"Error loading account configuration: {e}")

def render_pnl_analysis():
    """Render P&L analysis with charts and metrics"""
    st.subheader("📊 Profit & Loss Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90), key="pnl_start_date")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), key="pnl_end_date")
    
    try:
        # Generate sample P&L data (in a real implementation, this would come from the account manager)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
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
        
        # Daily returns chart
        fig2 = go.Figure()
        colors = ['green' if x >= 0 else 'red' for x in daily_returns]
        
        fig2.add_trace(go.Bar(
            x=dates[1:],
            y=daily_returns,
            marker_color=colors,
            name='Daily Returns'
        ))
        
        fig2.update_layout(
            title="Daily Returns (%)",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
            st.metric("Total Return", f"{total_return:+.2f}%")
        
        with col2:
            volatility = np.std(daily_returns) * np.sqrt(252)
            st.metric("Annualized Volatility", f"{volatility:.2f}%")
        
        with col3:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with col4:
            max_drawdown = np.min(np.minimum.accumulate(daily_returns))
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    except Exception as e:
        st.error(f"Error generating P&L analysis: {e}")

def render_live_analysis_tab():
    """Render the Live Analysis tab with fundamental and technical reports"""
    st.markdown('<h1 class="tab-header">📊 Live Market Analysis</h1>', unsafe_allow_html=True)
    
    # Analysis tabs
    analysis_tabs = st.tabs(["Current Positions", "Symbol Analysis", "Market Overview", "Watchlist"])
    
    with analysis_tabs[0]:  # Current Positions Analysis
        render_current_positions_analysis()
    
    with analysis_tabs[1]:  # Symbol Analysis
        render_symbol_analysis()
    
    with analysis_tabs[2]:  # Market Overview
        render_market_overview()
    
    with analysis_tabs[3]:  # Watchlist
        render_watchlist_analysis()

def render_current_positions_analysis():
    """Render analysis for current positions"""
    st.subheader("🎯 Current Positions Analysis")
    
    try:
        positions = st.session_state.position_manager.get_all_positions()
        active_positions = [p for p in positions if float(p.get('qty', 0)) != 0]
        
        if not active_positions:
            st.info("No active positions found. Analysis will be available when you have open positions.")
            return
        
        # Position selector
        position_symbols = [p.get('symbol', '') for p in active_positions]
        selected_symbol = st.selectbox("Select Position for Analysis", position_symbols)
        
        if selected_symbol:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                render_technical_analysis(selected_symbol)
            
            with col2:
                render_fundamental_analysis(selected_symbol)
    
    except Exception as e:
        st.error(f"Error loading position analysis: {e}")

def render_symbol_analysis():
    """Render analysis for any symbol"""
    st.subheader("🔍 Symbol Analysis")
    
    # Symbol input
    symbol = st.text_input("Enter Symbol for Analysis", placeholder="e.g., AAPL, TSLA, SPY").upper()
    
    if symbol and st.button("🔍 Analyze"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            render_technical_analysis(symbol)
        
        with col2:
            render_fundamental_analysis(symbol)

def render_technical_analysis(symbol):
    """Render technical analysis for a symbol"""
    st.markdown(f"### 📈 Technical Analysis - {symbol}")
    
    try:
        # Get historical data
        data_provider = st.session_state.data_provider
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # In a real implementation, this would fetch actual data
        # For demo, we'll create sample data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # For consistent demo data
        
        # Generate realistic price data
        returns = np.random.randn(len(dates)) * 0.02
        price_base = 100
        prices = price_base * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        high_prices = prices * (1 + np.random.rand(len(dates)) * 0.03)
        low_prices = prices * (1 - np.random.rand(len(dates)) * 0.03)
        open_prices = np.roll(prices, 1)
        open_prices[0] = prices[0]
        volumes = np.random.randint(1000000, 10000000, len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'volume': volumes
        })
        
        # Calculate technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = calculate_rsi(df['close'], 14)
        
        # Price chart with moving averages
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f'{symbol} Price & Moving Averages', 'Volume', 'RSI'],
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_50'], name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['rsi'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f"Technical Analysis - {symbol}",
            template="plotly_white",
            height=800,
            xaxis3_title="Date"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators summary
        current_price = df['close'].iloc[-1]
        sma20 = df['sma_20'].iloc[-1]
        sma50 = df['sma_50'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        
        st.markdown("#### 📊 Technical Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
            trend = "Bullish" if current_price > sma20 > sma50 else "Bearish" if current_price < sma20 < sma50 else "Neutral"
            st.metric("Trend", trend)
        
        with col2:
            st.metric("SMA 20", f"${sma20:.2f}")
            st.metric("SMA 50", f"${sma50:.2f}")
        
        with col3:
            st.metric("RSI (14)", f"{current_rsi:.1f}")
            rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            st.metric("RSI Signal", rsi_signal)
    
    except Exception as e:
        st.error(f"Error in technical analysis: {e}")

def render_fundamental_analysis(symbol):
    """Render fundamental analysis for a symbol"""
    st.markdown(f"### 📚 Fundamental Analysis - {symbol}")
    
    try:
        # In a real implementation, this would use the FundamentalAnalyzer
        # For demo, we'll show sample fundamental data
        
        st.markdown("#### 📈 Key Metrics")
        
        # Sample fundamental data
        fundamental_data = {
            'Market Cap': '$2.5T',
            'P/E Ratio': '28.5',
            'EPS': '$6.05',
            'Revenue Growth': '8.2%',
            'Dividend Yield': '0.5%',
            'ROE': '147.4%',
            'Debt-to-Equity': '1.73',
            'Free Cash Flow': '$100.0B'
        }
        
        col1, col2 = st.columns(2)
        
        for i, (metric, value) in enumerate(fundamental_data.items()):
            if i % 2 == 0:
                col1.metric(metric, value)
            else:
                col2.metric(metric, value)
        
        # Analyst recommendations
        st.markdown("#### 🎯 Analyst Recommendations")
        
        recommendation_data = {
            'Strong Buy': 15,
            'Buy': 12,
            'Hold': 8,
            'Sell': 2,
            'Strong Sell': 1
        }
        
        fig = px.bar(
            x=list(recommendation_data.keys()),
            y=list(recommendation_data.values()),
            title="Analyst Recommendations",
            color=list(recommendation_data.values()),
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(
            template="plotly_white",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI-generated analysis (placeholder)
        st.markdown("#### 🤖 AI Analysis Summary")
        st.info(f"""
        **{symbol} Analysis Summary:**
        
        • **Valuation**: Currently trading at a premium but justified by growth prospects
        • **Growth**: Strong revenue growth with expanding margins
        • **Risk**: Market volatility and regulatory concerns remain key risks
        • **Recommendation**: HOLD with positive outlook for long-term investors
        
        *This is a sample analysis. In production, this would be generated by the AI analysis engine.*
        """)
    
    except Exception as e:
        st.error(f"Error in fundamental analysis: {e}")

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
    
    # Market sentiment indicators
    st.markdown("### 📊 Market Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fear & Greed Index (simulated)
        fear_greed = 65
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = fear_greed,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fear & Greed Index"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector performance
        sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer', 'Industrial']
        performance = [2.1, 1.5, 0.8, -0.5, -1.2, 0.3]
        
        fig = px.bar(
            x=sectors,
            y=performance,
            title="Sector Performance (%)",
            color=performance,
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_watchlist_analysis():
    """Render watchlist with analysis using JSON persistence"""
    st.subheader("👀 Watchlist Analysis")
    
    # Watchlist management
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        new_symbol = st.text_input("Add Symbol to Watchlist", placeholder="e.g., AAPL", key="watchlist_add_symbol")
    
    with col2:
        if st.button("➕ Add", key="watchlist_add_btn"):
            if new_symbol:
                try:
                    if st.session_state.watchlist_manager.add_symbol(new_symbol):
                        st.success(f"✅ Added {new_symbol.upper()} to watchlist")
                        st.rerun()
                    else:
                        st.warning(f"ℹ️ {new_symbol.upper()} is already in watchlist")
                except Exception as e:
                    st.error(f"Error adding symbol: {e}")
    
    with col3:
        if st.button("🔄 Refresh", key="watchlist_refresh_btn"):
            st.rerun()
    
    # Display current watchlist from JSON
    try:
        watchlist = st.session_state.watchlist_manager.get_watchlist()
        
        if watchlist:
            st.write(f"**Current Watchlist ({len(watchlist)} symbols):**")
            
            watchlist_data = []
            
            for symbol in watchlist:
                # Simulate market data (in production, fetch real data)
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
            
            # Style the dataframe
            def style_change(val):
                if isinstance(val, str) and ('+' in val or '-' in val):
                    color = 'color: green' if '+' in val else 'color: red'
                    return color
                return ''
            
            styled_df = df.style.map(style_change, subset=['Change', 'Change %'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Symbol removal section
            st.subheader("🗑️ Remove Symbols")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                symbol_to_remove = st.selectbox("Select symbol to remove", watchlist, key="watchlist_remove_select")
            
            with col2:
                if st.button("🗑️ Remove", key="watchlist_remove_btn"):
                    try:
                        if st.session_state.watchlist_manager.remove_symbol(symbol_to_remove):
                            st.success(f"✅ Removed {symbol_to_remove} from watchlist")
                            st.rerun()
                        else:
                            st.error(f"❌ {symbol_to_remove} not found in watchlist")
                    except Exception as e:
                        st.error(f"Error removing symbol: {e}")
            
            # Bulk operations
            st.subheader("⚙️ Bulk Operations")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear All", key="watchlist_clear_btn"):
                    try:
                        st.session_state.watchlist_manager.clear_watchlist()
                        st.success("✅ Watchlist cleared")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing watchlist: {e}")
            
            with col2:
                backup_filename = st.text_input("Backup filename (optional)", placeholder="watchlist_backup.json", key="watchlist_backup_name")
                if st.button("💾 Backup", key="watchlist_backup_btn"):
                    try:
                        backup_file = st.session_state.watchlist_manager.backup_watchlist(backup_filename if backup_filename else None)
                        st.success(f"✅ Watchlist backed up to {backup_file}")
                    except Exception as e:
                        st.error(f"Error backing up watchlist: {e}")
        else:
            st.info("📭 Your watchlist is empty. Add symbols to start tracking them.")
            
        # Watchlist info
        with st.expander("📊 Watchlist Information"):
            try:
                info = st.session_state.watchlist_manager.get_watchlist_info()
                metadata = info.get('metadata', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Symbols:** {len(watchlist)}")
                    st.write(f"**Created:** {metadata.get('created', 'Unknown')}")
                
                with col2:
                    st.write(f"**Last Updated:** {metadata.get('last_updated', 'Unknown')}")
                    st.write(f"**Version:** {metadata.get('version', 'N/A')}")
                
                st.write(f"**Description:** {metadata.get('description', 'N/A')}")
                
            except Exception as e:
                st.error(f"Error loading watchlist info: {e}")
    
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        st.info("Using fallback watchlist")
        # Fallback to hardcoded list if JSON fails
        fallback_watchlist = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
        st.write("**Fallback Watchlist:**")
        st.write(", ".join(fallback_watchlist))

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def render_backtesting_tab():
    """Render the Backtesting tab with strategy-based backtesting"""
    st.markdown('<h1 class="tab-header">🧪 Strategy Backtesting</h1>', unsafe_allow_html=True)
    
    # Backtesting tabs
    backtest_tabs = st.tabs(["Current Positions", "Strategy Comparison", "Custom Backtest", "Results Analysis"])
    
    with backtest_tabs[0]:  # Current Positions Backtesting
        render_position_backtesting()
    
    with backtest_tabs[1]:  # Strategy Comparison
        render_strategy_comparison()
    
    with backtest_tabs[2]:  # Custom Backtest
        render_custom_backtest()
    
    with backtest_tabs[3]:  # Results Analysis
        render_backtest_results()

def render_position_backtesting():
    """Render backtesting for current positions"""
    st.subheader("📊 Current Position Backtesting")
    
    try:
        positions = st.session_state.position_manager.get_all_positions()
        active_positions = [p for p in positions if float(p.get('qty', 0)) != 0]
        
        if not active_positions:
            st.info("No active positions found. Backtesting will be available when you have open positions.")
            return
        
        # Position and strategy selector
        col1, col2 = st.columns(2)
        
        with col1:
            position_symbols = [p.get('symbol', '') for p in active_positions]
            selected_symbol = st.selectbox("Select Position", position_symbols)
        
        with col2:
            strategy_selector = st.session_state.strategy_selector
            available_strategies = strategy_selector.list_strategies()
            selected_strategy = st.selectbox("Select Strategy", available_strategies)
        
        # Backtest parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365), key="backtest1_start_date")
        
        with col2:
            end_date = st.date_input("End Date", datetime.now(), key="backtest1_end_date")
        
        with col3:
            initial_capital = st.number_input("Initial Capital", min_value=1000, value=100000, step=1000, key="backtest1_initial_capital")
        
        if st.button("🚀 Run Backtest", type="primary"):
            run_backtest_analysis(selected_symbol, selected_strategy, start_date, end_date, initial_capital)
    
    except Exception as e:
        st.error(f"Error in position backtesting: {e}")

def render_strategy_comparison():
    """Render strategy comparison backtesting"""
    st.subheader("⚔️ Strategy Comparison")
    
    # Strategy selection
    strategy_selector = st.session_state.strategy_selector
    available_strategies = strategy_selector.list_strategies()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Select Strategies to Compare:**")
        selected_strategies = st.multiselect(
            "Strategies", 
            available_strategies, 
            default=available_strategies[:3] if len(available_strategies) >= 3 else available_strategies
        )
    
    with col2:
        symbol = st.text_input("Symbol for Comparison", value="SPY", placeholder="e.g., SPY, AAPL")
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=100000, step=1000, key="backtest2_initial_capital")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365), key="backtest2_start_date")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), key="backtest2_end_date")
    
    if st.button("📊 Compare Strategies") and selected_strategies:
        render_strategy_comparison_results(selected_strategies, symbol, start_date, end_date, initial_capital)

def render_custom_backtest():
    """Render custom backtesting interface"""
    st.subheader("🔧 Custom Backtest Configuration")
    
    # Advanced backtest settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Settings**")
        symbols = st.text_area(
            "Symbols (one per line)", 
            value="AAPL\nMSFT\nGOOGL\nTSLA\nAMZN",
            height=100
        )
        strategy = st.selectbox("Strategy", st.session_state.strategy_selector.list_strategies())
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=100000, step=1000, key="backtest3_initial_capital")
    
    with col2:
        st.markdown("**Advanced Settings**")
        commission = st.number_input("Commission per Trade", min_value=0.0, value=1.0, step=0.1, key="backtest3_commission")
        slippage = st.number_input("Slippage (%)", min_value=0.0, value=0.1, step=0.01, key="backtest3_slippage")
        position_sizing = st.selectbox("Position Sizing", ["Equal Weight", "Risk Parity", "Kelly Criterion"])
        rebalance_freq = st.selectbox("Rebalance Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"])
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730), key="backtest3_start_date")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), key="backtest3_end_date")
    
    if st.button("🧪 Run Custom Backtest", type="primary"):
        symbol_list = [s.strip().upper() for s in symbols.split('\n') if s.strip()]
        run_custom_backtest(symbol_list, strategy, start_date, end_date, initial_capital, commission, slippage)

def render_backtest_results():
    """Render backtest results analysis"""
    st.subheader("📈 Backtest Results Analysis")
    
    # Placeholder for saved backtest results
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        results = st.session_state.backtest_results
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{results.get('total_return', 0):.2f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2f}%")
        
        with col4:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.1f}%")
        
        # Performance chart
        if 'equity_curve' in results:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['dates'],
                y=results['equity_curve'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Backtest Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No backtest results available. Run a backtest to see results here.")

def render_trading_tab():
    """Render the Trading tab for active trading interface"""
    st.markdown('<h1 class="tab-header">⚡ Active Trading</h1>', unsafe_allow_html=True)
    
    # Trading tabs
    trading_tabs = st.tabs(["Quick Trade", "Strategy Trading", "Options Trading", "Trade History"])
    
    with trading_tabs[0]:  # Quick Trade
        render_quick_trading()
    
    with trading_tabs[1]:  # Strategy Trading
        render_strategy_trading()
    
    with trading_tabs[2]:  # Options Trading
        render_options_trading()
    
    with trading_tabs[3]:  # Trade History
        render_trade_history()

def render_quick_trading():
    """Render quick trading interface"""
    st.subheader("⚡ Quick Trade")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Trading form
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
        
        if symbol:
            # Show current price (simulated)
            current_price = 150.25 + np.random.randn() * 2
            st.metric(f"{symbol} Current Price", f"${current_price:.2f}", f"{np.random.randn()*0.5:+.2f}")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            action = st.selectbox("Action", ["Buy", "Sell"])
            quantity = st.number_input("Quantity", min_value=1, value=100, key="trading_quantity")
        
        with col1b:
            order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop Loss", "Stop Limit"])
            if order_type in ["Limit", "Stop Limit"]:
                limit_price = st.number_input("Limit Price", value=current_price if 'current_price' in locals() else 100.0, key="trading_limit_price")
        
        # Risk management
        st.markdown("#### 🛡️ Risk Management")
        col1a, col1b = st.columns(2)
        
        with col1a:
            stop_loss = st.number_input("Stop Loss (%)", min_value=0.0, value=5.0, step=0.1, key="trading_stop_loss")
        
        with col1b:
            take_profit = st.number_input("Take Profit (%)", min_value=0.0, value=10.0, step=0.1, key="trading_take_profit")
        
        if st.button("🚀 Execute Trade", type="primary"):
            execute_trade_simulation(symbol, action, quantity, order_type, locals())
    
    with col2:
        # Position size calculator
        st.markdown("#### 📊 Position Calculator")
        
        try:
            account_info = st.session_state.account_manager.get_trading_account_status()
            portfolio_value = float(account_info.get('equity', 100000))
            
            risk_pct = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, 0.1)
            
            if 'current_price' in locals() and stop_loss > 0:
                risk_amount = portfolio_value * (risk_pct / 100)
                price_risk = current_price * (stop_loss / 100)
                suggested_qty = int(risk_amount / price_risk) if price_risk > 0 else 0
                
                st.metric("Suggested Quantity", suggested_qty)
                st.metric("Risk Amount", f"${risk_amount:,.2f}")
                st.metric("Max Loss", f"${suggested_qty * price_risk:,.2f}")
        
        except Exception as e:
            st.error("Unable to calculate position size")

def render_strategy_trading():
    """Render strategy-based trading interface"""
    st.subheader("🧠 Strategy Trading")
    
    # Strategy selector
    strategy_selector = st.session_state.strategy_selector
    available_strategies = strategy_selector.list_strategies()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_strategy = st.selectbox("Select Trading Strategy", available_strategies)
        
        if selected_strategy:
            strategy_info = strategy_selector.get_strategy_info(selected_strategy)
            
            st.markdown(f"**Strategy Type:** {strategy_info.get('type', 'Unknown')}")
            st.markdown(f"**Timeframe:** {strategy_info.get('timeframe', 'Unknown')}")
            st.markdown(f"**Risk Level:** {strategy_info.get('risk_level', 'Unknown')}")
    
    with col2:
        # Strategy parameters
        st.markdown("**Strategy Parameters:**")
        
        if selected_strategy:
            # Get default parameters
            strategy = strategy_selector.create_strategy(selected_strategy)
            if strategy:
                params = strategy.parameters
                
                # Show key parameters (simplified)
                for key, value in list(params.items())[:5]:  # Show first 5 parameters
                    if isinstance(value, (int, float)):
                        new_value = st.number_input(f"{key.replace('_', ' ').title()}", value=value)
                        params[key] = new_value
    
    # Watchlist for strategy
    st.markdown("#### 👀 Strategy Watchlist")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        watchlist_symbols = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL,TSLA,AMZN")
    with col2:
        auto_trade = st.checkbox("Auto-execute signals")
    
    if st.button("🔍 Generate Strategy Signals"):
        symbols = [s.strip().upper() for s in watchlist_symbols.split(',')]
        generate_strategy_signals(selected_strategy, symbols, auto_trade)

def render_options_trading():
    """Render options trading interface"""
    st.subheader("📋 Options Trading")
    
    st.info("🚧 Options trading interface coming soon! This will include:")
    st.markdown("""
    - Options chain analysis
    - Greeks calculation and visualization
    - Options strategies (covered calls, protective puts, spreads)
    - Volatility analysis
    - Risk/reward visualization
    """)

def render_trade_history():
    """Render trade history and analysis"""
    st.subheader("📚 Trade History")
    
    # Date filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("From", datetime.now() - timedelta(days=30), key="analysis_start_date")
    
    with col2:
        end_date = st.date_input("To", datetime.now(), key="analysis_end_date")
    
    with col3:
        symbol_filter = st.text_input("Symbol Filter", placeholder="Optional")
    
    # Simulated trade history
    if st.button("📊 Load Trade History"):
        trade_data = generate_sample_trade_history(start_date, end_date, symbol_filter)
        
        if trade_data:
            df = pd.DataFrame(trade_data)
            
            # Style profitable trades
            def style_pnl(val):
                if isinstance(val, str) and '$' in val:
                    try:
                        num_val = float(val.replace('$', '').replace(',', ''))
                        return 'color: green' if num_val >= 0 else 'color: red'
                    except:
                        return ''
                return ''
            
            styled_df = df.style.map(style_pnl, subset=['P&L'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Trade statistics
            total_pnl = sum([float(row['P&L'].replace('$', '').replace(',', '')) for row in trade_data])
            winning_trades = len([row for row in trade_data if float(row['P&L'].replace('$', '').replace(',', '')) > 0])
            total_trades = len(trade_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            
            with col2:
                st.metric("Win Rate", f"{winning_trades/total_trades*100:.1f}%" if total_trades > 0 else "0%")
            
            with col3:
                st.metric("Total Trades", total_trades)

# Helper functions for the trading functionality

def run_backtest_analysis(symbol, strategy, start_date, end_date, initial_capital):
    """Run backtest analysis and display results"""
    with st.spinner(f"Running backtest for {symbol} using {strategy} strategy..."):
        # Simulate backtest results
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        returns = np.random.randn(len(dates)) * 0.01 + 0.0002  # Slight positive bias
        equity_curve = initial_capital * np.exp(np.cumsum(returns))
        
        # Calculate metrics
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
        volatility = np.std(returns) * np.sqrt(252) * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_drawdown = np.min(np.minimum.accumulate(returns)) * 100
        
        # Store results
        st.session_state.backtest_results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.random.uniform(45, 65),
            'equity_curve': equity_curve,
            'dates': dates
        }
        
        st.success(f"Backtest completed! Total Return: {total_return:.2f}%")

def render_strategy_comparison_results(strategies, symbol, start_date, end_date, initial_capital):
    """Render strategy comparison results"""
    st.markdown("### 📊 Strategy Comparison Results")
    
    # Generate comparison data
    comparison_data = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = go.Figure()
    
    for i, strategy in enumerate(strategies):
        # Simulate performance for each strategy
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        returns = np.random.randn(len(dates)) * 0.01 + np.random.uniform(-0.0001, 0.0005)
        equity_curve = initial_capital * np.exp(np.cumsum(returns))
        
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = np.min(np.minimum.accumulate(returns)) * 100
        
        comparison_data.append({
            'Strategy': strategy,
            'Total Return (%)': f"{total_return:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown (%)': f"{max_drawdown:.2f}%"
        })
        
        # Add to chart
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name=strategy,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Display comparison table
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Display comparison chart
    fig.update_layout(
        title=f"Strategy Comparison - {symbol}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_custom_backtest(symbols, strategy, start_date, end_date, initial_capital, commission, slippage):
    """Run custom backtest with advanced settings"""
    st.success(f"Custom backtest initiated for {len(symbols)} symbols using {strategy} strategy!")
    
    # Display backtest configuration
    st.markdown("#### 🔧 Backtest Configuration")
    config_data = {
        'Parameter': ['Symbols', 'Strategy', 'Initial Capital', 'Commission', 'Slippage', 'Date Range'],
        'Value': [
            ', '.join(symbols[:5]) + ('...' if len(symbols) > 5 else ''),
            strategy,
            f"${initial_capital:,}",
            f"${commission}",
            f"{slippage}%",
            f"{start_date} to {end_date}"
        ]
    }
    
    st.dataframe(pd.DataFrame(config_data), use_container_width=True)

def execute_trade_simulation(symbol, action, quantity, order_type, params):
    """Execute a simulated trade"""
    st.success(f"Trade executed! {action} {quantity} shares of {symbol}")
    
    # Display order confirmation
    st.markdown("#### 📋 Order Confirmation")
    st.json({
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'order_type': order_type,
        'estimated_value': f"${quantity * params.get('current_price', 100):.2f}",
        'status': 'Filled (Simulated)'
    })

def generate_strategy_signals(strategy, symbols, auto_trade):
    """Generate strategy signals for symbols"""
    st.markdown(f"### 🎯 {strategy} Strategy Signals")
    
    signals_data = []
    
    for symbol in symbols:
        # Simulate signal generation
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
            'Reason': f"{strategy} criteria met"
        })
    
    df = pd.DataFrame(signals_data)
    
    # Style signals
    def style_signals(val):
        if val == 'BUY':
            return 'color: green; font-weight: bold'
        elif val == 'SELL':
            return 'color: red; font-weight: bold'
        else:
            return 'color: orange; font-weight: bold'
    
    styled_df = df.style.map(style_signals, subset=['Signal'])
    st.dataframe(styled_df, use_container_width=True)
    
    if auto_trade:
        buy_signals = len([s for s in signals_data if s['Signal'] == 'BUY'])
        sell_signals = len([s for s in signals_data if s['Signal'] == 'SELL'])
        st.info(f"Auto-trading enabled: Would execute {buy_signals} buy orders and {sell_signals} sell orders")

def generate_sample_trade_history(start_date, end_date, symbol_filter):
    """Generate sample trade history data"""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA']
    
    if symbol_filter:
        symbols = [s for s in symbols if symbol_filter.upper() in s]
    
    trade_data = []
    current_date = start_date
    
    while current_date <= end_date:
        if np.random.random() > 0.7:  # 30% chance of trade per day
            symbol = np.random.choice(symbols)
            action = np.random.choice(['BUY', 'SELL'])
            quantity = np.random.randint(1, 500)
            price = 100 + np.random.randn() * 20
            pnl = np.random.randn() * 500
            
            trade_data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Action': action,
                'Quantity': quantity,
                'Price': f"${price:.2f}",
                'P&L': f"${pnl:,.2f}"
            })
        
        current_date += timedelta(days=1)
    
    return trade_data

def main():
    """Main dashboard function"""
    # Initialize session state
    initialize_session_state()
    
    # Main header with account tier info
    st.markdown('<h1 class="main-header">🌍 Gauss World Trader - Advanced Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Account tier and delay notice header
    current_time = datetime.now(EASTERN)
    
    try:
        subscription_info = st.session_state.subscription_info
        is_subscribed = subscription_info.get('has_sip_subscription', False)
        account_tier = "Pro Tier" if is_subscribed else "Free Tier"
        
        # Check if today is a trading day
        is_trading_day = current_time.weekday() < 5  # Monday=0, Friday=4
        
        # Create header info row
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            local_time = get_local_time()
            st.markdown(f"**📅 Local Time:** {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            # Account tier display
            if is_subscribed:
                st.success(f"✨ {account_tier}")
            else:
                st.info(f"🆓 {account_tier}")
        
        with col3:
            # Data source notice for free tier on trading days
            if not is_subscribed and is_trading_day:
                st.info("📊 Today's data: Real-time IEX")
                
    except Exception as e:
        # Fallback if subscription info not available
        col1, col2 = st.columns([2, 2])
        
        with col1:
            local_time = get_local_time()
            st.markdown(f"**📅 Local Time:** {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            is_trading_day = current_time.weekday() < 5
            if is_trading_day:
                st.info("🆓 Free Tier")
                st.info("📊 Today's data: Real-time IEX")
    
    st.markdown("---")
    
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

if __name__ == "__main__":
    main()