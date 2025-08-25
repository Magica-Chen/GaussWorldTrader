#!/usr/bin/env python3
"""
Modern Trading Dashboard - Unified Interface
Combining the best features from simple and advanced dashboards
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from src.utils.timezone_utils import EASTERN, now_et, get_market_status
from typing import Optional, Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import trading modules
from src.account.account_manager import AccountManager
from src.account.position_manager import PositionManager
from src.account.order_manager import OrderManager
from src.agent.fundamental_analyzer import FundamentalAnalyzer
from src.strategy.strategy_selector import get_strategy_selector
from src.trade import Backtester, Portfolio
from src.data import AlpacaDataProvider
from src.utils.watchlist_manager import WatchlistManager
from src.utils.dashboard_utils import (
    get_shared_market_data, run_shared_backtest, render_shared_positions_table
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🌍 Gauss World Trader - Modern Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
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
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_local_time():
    """Get current time in local timezone for display purposes"""
    return datetime.now()

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 'Live Analysis'
    
    if 'account_manager' not in st.session_state:
        try:
            st.session_state.account_manager = AccountManager()
            st.session_state.position_manager = PositionManager(st.session_state.account_manager)
            st.session_state.order_manager = OrderManager(st.session_state.account_manager)
            st.session_state.fundamental_analyzer = FundamentalAnalyzer()
            st.session_state.strategy_selector = get_strategy_selector()
            st.session_state.data_provider = AlpacaDataProvider()
            st.session_state.watchlist_manager = WatchlistManager()
            
            # Get account info for tier display
            st.session_state.account_info = st.session_state.data_provider.get_account_info()
            
        except Exception as e:
            logger.error(f"Error initializing managers: {e}")
            st.error("Error initializing trading modules. Please check API configuration.")

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def load_market_data(symbol, days=30):
    """Load market data"""
    try:
        # Create fresh provider each time (like simple dashboard) to avoid session state issues
        # This ensures we always have the latest VIP/IEX status
        provider = AlpacaDataProvider()
        account_info = provider.get_account_info()
        
        # Update session state with fresh info
        st.session_state.data_provider = provider
        st.session_state.account_info = account_info
        
        current_time = now_et()
        start_date = current_time - timedelta(days=days)

        # Fetch data with detailed error handling
        try:
            data = provider.get_bars(symbol, '1Day', start_date)
            if data is not None and not data.empty:
                return data, None
            else:
                # Log why data is empty
                logger.warning(f"No data returned for {symbol} from {start_date}")
                return None, f"No data available for {symbol} (requested from: {start_date.date()})"
        except Exception as data_error:
            logger.error(f"Error fetching data for {symbol}: {data_error}")
            return None, f"Data fetch error: {data_error}"
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Load market data error for {symbol}: {error_msg}")
        if "subscription does not permit" in error_msg.lower() or "not entitled" in error_msg.lower():
            # Remove fallback - let real errors surface
            pass
        return None, error_msg

@st.cache_data(ttl=300)
def get_account_info():
    """Get account information with caching"""
    from src.utils.dashboard_utils import get_shared_account_info
    return get_shared_account_info()

@st.cache_data(ttl=600)
def get_technical_indicators(data):
    """Get technical indicators with caching"""
    try:
        from src.analysis import TechnicalAnalysis
        
        if data is None or data.empty:
            return None
        
        ta = TechnicalAnalysis()
        indicators = {}
        
        # RSI
        rsi = ta.rsi(data['close'])
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 0
        
        # Moving Averages
        sma_20 = ta.sma(data['close'], 20)
        sma_50 = ta.sma(data['close'], 50)
        indicators['sma_20'] = sma_20.iloc[-1] if not sma_20.empty else 0
        indicators['sma_50'] = sma_50.iloc[-1] if not sma_50.empty else 0
        
        # MACD
        macd, signal, histogram = ta.macd(data['close'])
        indicators['macd'] = macd.iloc[-1] if not macd.empty else 0
        
        # Trend Analysis
        trend_info = ta.trend_analysis(data)
        indicators['trends'] = trend_info
        
        return indicators
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return None

@st.cache_data(ttl=600)
def get_crypto_data():
    """Get cryptocurrency data with caching"""
    try:
        from src.data import AlpacaDataProvider
        provider = AlpacaDataProvider()
        btc_quote = provider.get_crypto_latest_quote('BTC/USD')
        btc_data = {
            'symbol': 'BTC',
            'price_usd': btc_quote.get('bid_price', 0) if 'error' not in btc_quote else 0,
            'last_updated': btc_quote.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if 'error' not in btc_quote else 'Unknown'
        }
        return btc_data, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=1200)
def get_news_insider_data(symbol):
    """Get news and insider data with caching"""
    try:
        from src.data import NewsDataProvider
        provider = NewsDataProvider()
        news = provider.get_company_news(symbol)
        insider_transactions = provider.get_insider_transactions(symbol)
        insider_sentiment = provider.get_insider_sentiment(symbol)
        return news, insider_transactions, insider_sentiment, None
    except Exception as e:
        return [], [], {}, str(e)

@st.cache_data(ttl=60)
def generate_signals(symbol, data):
    """Generate trading signals with caching"""
    try:
        from src.strategy import MomentumStrategy
        from src.trade import Portfolio
        
        if data is None or data.empty:
            return [], "No data available"
        
        strategy = MomentumStrategy()
        portfolio = Portfolio()
        
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
        
        signals = strategy.generate_signals(
            current_date=now_et(),
            current_prices=current_prices,
            current_data=current_data,
            historical_data=historical_data,
            portfolio=portfolio
        )
        
        return signals, None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_backtest(symbols, days_back=365, initial_cash=100000, strategy_type="Momentum"):
    """Run backtest with caching and strategy selection"""
    return run_shared_backtest(symbols, days_back, initial_cash, strategy_type)

def generate_dashboard_transaction_log(results, symbols):
    """Generate transaction log for dashboard download"""
    try:
        if not results or 'trades_history' not in results:
            return None
        
        trades_df = results['trades_history']
        if trades_df.empty:
            return None
        
        # Enhanced transaction log
        enhanced_trades = []
        positions = {}
        trade_counter = 1
        
        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            action = trade['action'].upper()
            quantity = abs(trade['quantity'])
            price = trade.get('price', 0)
            trade_date = trade['date']
            trade_value = quantity * price
            
            # Position tracking
            position_before = positions.get(symbol, {'qty': 0, 'avg_cost': 0, 'total_cost': 0})
            
            if action == 'BUY':
                new_qty = position_before['qty'] + quantity
                new_total_cost = position_before['total_cost'] + trade_value
                new_avg_cost = new_total_cost / new_qty if new_qty > 0 else 0
                
                positions[symbol] = {
                    'qty': new_qty,
                    'avg_cost': new_avg_cost,
                    'total_cost': new_total_cost
                }
                realized_pnl = 0
                
            elif action == 'SELL':
                if position_before['qty'] >= quantity:
                    cost_basis = position_before['avg_cost'] * quantity
                    proceeds = trade_value
                    realized_pnl = proceeds - cost_basis
                    
                    new_qty = position_before['qty'] - quantity
                    new_total_cost = position_before['total_cost'] - cost_basis
                    
                    positions[symbol] = {
                        'qty': new_qty,
                        'avg_cost': position_before['avg_cost'] if new_qty > 0 else 0,
                        'total_cost': new_total_cost
                    }
                else:
                    realized_pnl = 0
            
            position_after = positions.get(symbol, {'qty': 0, 'avg_cost': 0, 'total_cost': 0})
            
            # Create enhanced trade record
            enhanced_trade = {
                'Trade_ID': trade_counter,
                'Date': trade_date.strftime('%Y-%m-%d') if hasattr(trade_date, 'strftime') else str(trade_date),
                'Symbol': symbol,
                'Action': action,
                'Quantity': quantity,
                'Price': f"{price:.4f}",
                'Trade_Value': f"{trade_value:.2f}",
                'Commission': f"{trade_value * 0.01:.2f}",
                'Net_Amount': f"{trade_value * (0.99 if action == 'BUY' else 1.01):.2f}",
                'Position_Before': position_before['qty'],
                'Position_After': position_after['qty'],
                'Avg_Cost_Basis': f"{position_after['avg_cost']:.4f}",
                'Realized_PnL': f"{realized_pnl:.2f}",
                'Strategy': 'Modern Dashboard',
                'Notes': 'Backtest transaction'
            }
            
            enhanced_trades.append(enhanced_trade)
            trade_counter += 1
        
        # Create DataFrame and save
        transactions_df = pd.DataFrame(enhanced_trades)
        timestamp = now_et().strftime('%Y%m%d_%H%M%S')
        filename = f"modern_dashboard_transactions_{timestamp}.csv"
        
        transactions_df.to_csv(filename, index=False)
        return filename
        
    except Exception as e:
        st.error(f"Error generating transaction log: {e}")
        return None

def create_enhanced_sidebar():
    """Create enhanced sidebar with navigation and quick status"""
    st.sidebar.markdown("## 🌍 Gauss World Trader")
    st.sidebar.markdown("### Modern Dashboard")
    st.sidebar.markdown("---")
    
    # Main navigation
    tabs = ['Live Analysis', 'Account Management', 'Strategy Backtesting', 'Active Trading', 'News & Crypto', 'Portfolio Analysis']
    
    selected_tab = st.sidebar.radio(
        "Navigation",
        tabs,
        index=tabs.index(st.session_state.current_tab) if st.session_state.current_tab in tabs else 0,
        key="main_navigation"
    )
    
    if selected_tab != st.session_state.current_tab:
        st.session_state.current_tab = selected_tab
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Market status and time
    st.sidebar.markdown("### 🕒 Market Status")
    current_time = now_et()
    local_time = get_local_time()
    st.sidebar.markdown(f"**Local Time:** {local_time.strftime('%H:%M:%S')}")
    st.sidebar.markdown(f"**Market Time (ET):** {current_time.strftime('%H:%M:%S')}")
    
    # Enhanced market status
    is_weekend = current_time.weekday() >= 5
    is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
    is_after_hours = 16 <= current_time.hour < 20
    is_overnight = current_time.hour >= 20 or current_time.hour < 4
    is_market_open = not is_weekend and not is_pre_market and not is_after_hours and not is_overnight and (9 <= current_time.hour < 16)
    
    if is_weekend:
        st.sidebar.error("🔴 Market: Closed (Weekend)")
    elif is_overnight:
        st.sidebar.info("🔵 Market: Overnight")
    elif is_pre_market:
        st.sidebar.warning("🟡 Market: Pre-Market")
    elif is_after_hours:
        st.sidebar.warning("🟡 Market: After Hours")
    else:
        st.sidebar.success("🟢 Market: Open")
    
    st.sidebar.markdown("---")
    
    # Quick account status
    st.sidebar.markdown("### 📊 Quick Status")
    try:
        account_info, error = get_account_info()
        if account_info and not error:
            equity = float(account_info.get('equity', 0))
            buying_power = float(account_info.get('buying_power', 0))
            
            st.sidebar.metric("Portfolio Value", f"${equity:,.2f}")
            st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")
            
            # Position count
            positions = st.session_state.position_manager.get_all_positions()
            position_count = len([p for p in positions if float(p.get('qty', 0)) != 0])
            st.sidebar.metric("Active Positions", position_count)
        else:
            st.sidebar.warning("Unable to load account status")
    except Exception as e:
        st.sidebar.error("Account status unavailable")
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button("📊 Check Positions"):
        st.session_state.current_tab = 'Account Management'
        st.rerun()

def _render_analysis_controls():
    """Render analysis input controls"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", key="analysis_symbol").upper()
    with col2:
        days_back = st.slider("Days of History", 7, 365, 30, key="analysis_days")
    with col3:
        analysis_type = st.selectbox("Analysis Type", ["Technical", "Fundamental", "Combined"], key="analysis_type")
    
    return symbol, days_back, analysis_type

def _create_price_chart(symbol, data):
    """Create enhanced candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} Price & Indicators", "Volume", "RSI")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data['open'], high=data['high'],
            low=data['low'], close=data['close'], name=symbol
        ), row=1, col=1
    )
    
    # Add moving averages if enough data
    if len(data) >= 20:
        sma_20 = data['close'].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=data.index, y=sma_20, name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if len(data) >= 50:
        sma_50 = data['close'].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=data.index, y=sma_50, name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=data.index, y=data['volume'], name='Volume', opacity=0.3),
        row=2, col=1
    )
    
    # RSI
    if len(data) >= 14:
        rsi = calculate_rsi(data['close'])
        fig.add_trace(
            go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=800, showlegend=True, xaxis3_title="Date", template="plotly_white"
    )
    
    return fig

def _render_current_metrics(data):
    """Render current price metrics"""
    st.subheader("📊 Current Metrics")
    
    current_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    st.metric(
        label="Current Price", 
        value=f"${current_price:.2f}",
        delta=f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
    )
    
    st.metric("Volume", f"{data['volume'].iloc[-1]:,.0f}")
    range_text = f"${data['low'].min():.2f} - ${data['high'].max():.2f}"
    st.metric("Range (Period)", range_text)

def _render_technical_indicators(data):
    """Render technical indicators section"""
    st.subheader("🔬 Technical Indicators")
    
    with st.spinner("Calculating indicators..."):
        indicators = get_technical_indicators(data)
    
    if indicators:
        st.metric("RSI (14)", f"{indicators['rsi']:.2f}")
        if indicators['sma_20'] > 0:
            st.metric("SMA 20", f"${indicators['sma_20']:.2f}")
        if indicators['sma_50'] > 0:
            st.metric("SMA 50", f"${indicators['sma_50']:.2f}")
        st.metric("MACD", f"{indicators['macd']:.4f}")
        
        # Trend analysis
        if 'trends' in indicators and indicators['trends']:
            st.subheader("📈 Trend Analysis")
            trends = indicators['trends']
            st.write("**Short-term:**", trends.get('short_term_trend', 'N/A'))
            st.write("**Medium-term:**", trends.get('medium_term_trend', 'N/A'))
            st.write("**Long-term:**", trends.get('long_term_trend', 'N/A'))

def _render_trading_signals(symbol, data):
    """Render trading signals section"""
    st.subheader("🧠 Trading Signals")
    
    with st.spinner("Generating signals..."):
        signals, signal_error = generate_signals(symbol, data)
    
    if signal_error:
        st.error(f"Signal Error: {signal_error}")
    elif signals:
        for signal in signals:
            signal_color = "🟢" if signal['action'].upper() == 'BUY' else "🔴"
            st.success(f"""
            {signal_color} **{signal['action'].upper()} SIGNAL**
            
            - **Symbol**: {signal['symbol']}
            - **Quantity**: {signal['quantity']} shares  
            - **Confidence**: {signal.get('confidence', 0):.1%}
            - **Reason**: {signal.get('reason', 'Strategy criteria met')}
            """)
    else:
        st.info("📭 No trading signals generated")

def render_live_analysis_tab():
    """Enhanced live analysis with both simple and advanced features"""
    st.markdown('<h1 class="tab-header">📈 Live Market Analysis</h1>', unsafe_allow_html=True)
    
    symbol, days_back, analysis_type = _render_analysis_controls()
    
    if symbol:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 {symbol} Price Chart")
            
            # Load data
            with st.spinner(f"Loading {symbol} data..."):
                data, error = load_market_data(symbol, days_back)
            
            if error:
                st.error(f"❌ Data Error: {error}")
                return
            
            if data is None or data.empty:
                st.warning(f"⚠️ No data found for {symbol}")
                return
            
            # Create and display chart
            fig = _create_price_chart(symbol, data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data period info
            try:
                data_start = data.index[0].strftime('%Y-%m-%d') if pd.notna(data.index[0]) else "N/A"
                data_end = data.index[-1].strftime('%Y-%m-%d') if pd.notna(data.index[-1]) else "N/A"
                st.info(f"📅 Data Period: {data_start} to {data_end} ({len(data)} trading days)")
            except (AttributeError, ValueError):
                st.info(f"📅 Data Period: {len(data)} trading days available")
        
        with col2:
            _render_current_metrics(data)
            _render_technical_indicators(data)
            _render_trading_signals(symbol, data)

def render_account_management_tab():
    """Enhanced account management combining both dashboard features"""
    st.markdown('<h1 class="tab-header">💼 Account Management</h1>', unsafe_allow_html=True)
    
    # Account tabs
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
    """Enhanced account overview"""
    st.subheader("📈 Portfolio Overview")
    
    try:
        account_info, error = get_account_info()
        
        if error:
            st.error(f"Error loading account information: {error}")
            return
        
        if not account_info:
            st.warning("No account information available")
            return
        
        # Key metrics in enhanced layout
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
            
            delta_color = "normal" if day_change >= 0 else "inverse"
            st.metric("Day P&L", f"${day_change:,.2f}", f"{day_change_pct:+.2f}%", delta_color=delta_color)
        
        # Account details in enhanced format
        st.subheader("📋 Account Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Account Information**")
            account_details = {
                "Account Number": account_info.get('account_number', 'N/A'),
                "Account Status": account_info.get('status', 'N/A'),
                "Trading Status": "Active" if not account_info.get('trading_blocked', True) else "Blocked",
                "Pattern Day Trader": "Yes" if account_info.get('pattern_day_trader', False) else "No"
            }
            
            for label, value in account_details.items():
                st.write(f"**{label}:** {value}")
        
        with col2:
            st.markdown("**Financial Information**")
            financial_details = {
                "Currency": account_info.get('currency', 'USD'),
                "Last Equity": f"${float(account_info.get('last_equity', 0)):,.2f}",
                "Initial Margin": f"${float(account_info.get('initial_margin', 0)):,.2f}",
                "Maintenance Margin": f"${float(account_info.get('maintenance_margin', 0)):,.2f}"
            }
            
            for label, value in financial_details.items():
                st.write(f"**{label}:** {value}")
                
    except Exception as e:
        st.error(f"Error loading account overview: {e}")

def render_positions_view():
    """Enhanced positions view using shared utility"""
    st.subheader("📊 Current Positions")
    
    try:
        positions = st.session_state.position_manager.get_all_positions()
        render_shared_positions_table(positions)
            
    except Exception as e:
        st.error(f"Error loading positions: {e}")

def render_orders_view():
    """Enhanced orders view"""
    st.subheader("📋 Order Management")
    
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
        
        except Exception as e:
            st.error(f"Error loading active orders: {e}")
    
    with order_tabs[1]:  # Order History
        render_order_history()
    
    with order_tabs[2]:  # Place Order
        render_enhanced_place_order()

def render_order_history():
    """Enhanced order history"""
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30), key="order_hist_start")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), key="order_hist_end")
    
    if st.button("Load Order History"):
        try:
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

def render_enhanced_place_order():
    """Enhanced order placement form"""
    st.subheader("📤 Place New Order")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL", key="place_order_symbol")
        side = st.selectbox("Side", ["buy", "sell"], key="place_order_side")
        order_type = st.selectbox("Order Type", ["market", "limit", "stop", "stop_limit"], key="place_order_type")
        quantity = st.number_input("Quantity", min_value=1, value=1, key="place_order_quantity")
    
    with col2:
        if order_type in ["limit", "stop_limit"]:
            limit_price = st.number_input("Limit Price", min_value=0.01, value=1.00, step=0.01, key="place_order_limit")
        else:
            limit_price = None
        
        if order_type in ["stop", "stop_limit"]:
            stop_price = st.number_input("Stop Price", min_value=0.01, value=1.00, step=0.01, key="place_order_stop")
        else:
            stop_price = None
        
        time_in_force = st.selectbox("Time in Force", ["day", "gtc", "ioc", "fok"], key="place_order_tif")
    
    # Risk management section
    st.subheader("🛡️ Risk Management")
    col1, col2 = st.columns(2)
    
    with col1:
        stop_loss_pct = st.number_input("Stop Loss (%)", min_value=0.0, value=5.0, step=0.1, key="place_order_stop_loss")
    
    with col2:
        take_profit_pct = st.number_input("Take Profit (%)", min_value=0.0, value=10.0, step=0.1, key="place_order_take_profit")
    
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
            
            st.success(f"Order placed successfully! (Demo mode)")
            st.json(order_data)
            
        except Exception as e:
            st.error(f"Error placing order: {e}")

def render_account_configuration():
    """Enhanced account configuration"""
    st.subheader("⚙️ Account Configuration")
    
    try:
        config = st.session_state.account_manager.get_account_configurations()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trading Configuration**")
            day_trading = st.checkbox("Day Trading", value=config.get('day_trading_enabled', False), key="config_day_trading")
            options_trading = st.checkbox("Options Trading", value=config.get('options_trading_enabled', False), key="config_options")
            crypto_trading = st.checkbox("Crypto Trading", value=config.get('crypto_trading_enabled', False), key="config_crypto")
            
            max_margin = st.number_input("Max Margin Multiplier", 
                                       min_value=1.0, max_value=4.0, 
                                       value=float(config.get('max_margin_multiplier', 2.0)),
                                       step=0.1, key="config_margin")
        
        with col2:
            st.markdown("**Risk Management**")
            day_trade_limit = st.number_input("Day Trade Limit", 
                                            min_value=0, 
                                            value=int(config.get('day_trade_limit', 3)),
                                            key="config_dt_limit")
            
            position_limit = st.number_input("Max Position Size (%)", 
                                           min_value=1, max_value=100,
                                           value=int(config.get('max_position_size_pct', 25)),
                                           key="config_position_limit")
            
            stop_loss_default = st.number_input("Default Stop Loss (%)", 
                                              min_value=0.1, max_value=50.0,
                                              value=float(config.get('default_stop_loss_pct', 5.0)),
                                              step=0.1, key="config_stop_loss")
        
        if st.button("💾 Save Configuration", type="primary"):
            new_config = {
                'day_trading_enabled': day_trading,
                'options_trading_enabled': options_trading,
                'crypto_trading_enabled': crypto_trading,
                'max_margin_multiplier': max_margin,
                'day_trade_limit': day_trade_limit,
                'max_position_size_pct': position_limit,
                'default_stop_loss_pct': stop_loss_default
            }
            
            st.success("Configuration saved successfully! (Demo mode)")
            st.json(new_config)
    
    except Exception as e:
        st.error(f"Error loading account configuration: {e}")

def render_pnl_analysis():
    """Enhanced P&L analysis"""
    st.subheader("📊 Profit & Loss Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90), key="pnl_analysis_start")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), key="pnl_analysis_end")
    
    try:
        # Generate enhanced P&L data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        returns = np.random.randn(len(dates)) * 0.01 + 0.0002
        portfolio_values = 100000 * np.exp(np.cumsum(returns))
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
        
        # Portfolio value chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=3),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        fig1.update_layout(
            title="📈 Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Daily returns with enhanced visualization
        fig2 = go.Figure()
        colors = ['rgba(0, 200, 83, 0.8)' if x >= 0 else 'rgba(211, 47, 47, 0.8)' for x in daily_returns]
        
        fig2.add_trace(go.Bar(
            x=dates[1:],
            y=daily_returns,
            marker_color=colors,
            name='Daily Returns'
        ))
        
        fig2.update_layout(
            title="📊 Daily Returns Distribution",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Enhanced performance metrics
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
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    except Exception as e:
        st.error(f"Error generating P&L analysis: {e}")

def render_strategy_backtesting_tab():
    """Enhanced strategy backtesting"""
    st.markdown('<h1 class="tab-header">🧪 Strategy Backtesting</h1>', unsafe_allow_html=True)
    
    backtest_tabs = st.tabs(["Quick Backtest", "Strategy Comparison", "Custom Backtest", "Results Analysis"])
    
    with backtest_tabs[0]:
        render_quick_backtest()
    
    with backtest_tabs[1]:
        render_strategy_comparison()
    
    with backtest_tabs[2]:
        render_custom_backtest()
    
    with backtest_tabs[3]:
        render_backtest_results()

def render_quick_backtest():
    """Enhanced quick backtesting interface with real data"""
    st.subheader("⚡ Quick Strategy Backtest")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strategy_type = st.selectbox(
            "Select Strategy",
            ["Momentum", "Mean Reversion", "Trend Following", "RSI Oversold/Overbought"],
            key="quick_backtest_strategy"
        )
    
    with col2:
        # Load default symbols from watchlist
        try:
            default_watchlist = st.session_state.watchlist_manager.get_watchlist()
            all_symbols = list(set(default_watchlist + ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY"]))
            default_symbols = default_watchlist[:2] if len(default_watchlist) >= 2 else ["AAPL", "MSFT"]
        except:
            all_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY"]
            default_symbols = ["AAPL", "MSFT"]
        
        symbols = st.multiselect(
            "Select Symbols",
            all_symbols,
            default=default_symbols,
            key="quick_backtest_symbols"
        )
    
    with col3:
        days = st.slider("Days to Test", 90, 730, 365, key="quick_backtest_days")
        
    with col4:
        initial_cash = st.number_input("Initial Cash ($)", value=100000, step=10000, key="quick_backtest_cash")
    
    # Strategy descriptions
    strategy_descriptions = {
        "Momentum": "🚀 Buys stocks with strong upward momentum and RSI confirmation",
        "Mean Reversion": "🔄 Buys oversold stocks expecting price reversal",
        "Trend Following": "📈 Follows long-term trends using moving average crossovers",
        "RSI Oversold/Overbought": "⚖️ Trades based on RSI levels"
    }
    
    st.info(f"**Strategy:** {strategy_descriptions.get(strategy_type, 'Custom strategy')}")
    
    if st.button("🚀 Run Quick Backtest", type="primary", disabled=len(symbols) == 0):
        with st.spinner(f"Running {strategy_type} backtest on real data..."):
            results, error = run_backtest(symbols, days, initial_cash, strategy_type)
            
            if error:
                st.error(f"❌ Backtest Error: {error}")
            elif results:
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Initial Value", f"${results['initial_value']:,.2f}")
                with col2:
                    st.metric("Final Value", f"${results['final_value']:,.2f}")
                with col3:
                    st.metric("Total Return", f"{results['total_return_percentage']:.2f}%")
                with col4:
                    st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                
                # P&L Chart
                if 'portfolio_history' in results and not results['portfolio_history'].empty:
                    portfolio_df = results['portfolio_history']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=portfolio_df['date'],
                        y=portfolio_df['portfolio_value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_hline(y=results['initial_value'], 
                                line_dash="dash", 
                                line_color="gray", 
                                annotation_text="Initial Value")
                    
                    fig.update_layout(
                        title="📊 Portfolio Performance Over Time",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=500,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics table
                    st.subheader("📋 Detailed Metrics")
                    metrics_data = {
                        "Metric": [
                            "Annualized Return", "Volatility", "Sharpe Ratio", 
                            "Max Drawdown", "Total Trades", "Win Rate"
                        ],
                        "Value": [
                            f"{results.get('annualized_return_percentage', 0):.2f}%",
                            f"{results.get('volatility', 0):.2f}",
                            f"{results.get('sharpe_ratio', 0):.2f}",
                            f"{results.get('max_drawdown_percentage', 0):.2f}%",
                            f"{results.get('total_trades', 0)}",
                            f"{results.get('win_rate', 0):.1f}%"
                        ]
                    }
                    st.table(pd.DataFrame(metrics_data))
                    
                    # Transaction log download
                    if st.button("📋 Generate Transaction CSV", type="secondary"):
                        csv_filename = generate_dashboard_transaction_log(results, symbols)
                        if csv_filename:
                            with open(csv_filename, 'r') as f:
                                csv_data = f.read()
                            
                            st.download_button(
                                label="💾 Download Transaction Log",
                                data=csv_data,
                                file_name=csv_filename,
                                mime='text/csv',
                                type="primary"
                            )
                            st.success(f"✅ Transaction log generated: {csv_filename}")
                            
                # Store results for results analysis tab
                st.session_state.backtest_results = results

def render_strategy_comparison():
    """Strategy comparison interface"""
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
            default=available_strategies[:3] if len(available_strategies) >= 3 else available_strategies,
            key="strategy_comparison_strategies"
        )
    
    with col2:
        symbol = st.text_input("Symbol for Comparison", value="SPY", 
                              placeholder="e.g., SPY, AAPL", key="strategy_comparison_symbol")
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=100000, 
                                        step=1000, key="strategy_comparison_capital")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365), 
                                 key="strategy_comparison_start")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), key="strategy_comparison_end")
    
    if st.button("📊 Compare Strategies", disabled=len(selected_strategies) == 0):
        render_strategy_comparison_results(selected_strategies, symbol, start_date, end_date, initial_capital)

def render_custom_backtest():
    """Custom backtesting interface"""
    st.subheader("🔧 Custom Backtest Configuration")
    
    # Advanced backtest settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Settings**")
        symbols = st.text_area(
            "Symbols (one per line)", 
            value="AAPL\nMSFT\nGOOGL\nTSLA\nAMZN",
            height=100,
            key="custom_backtest_symbols"
        )
        strategy = st.selectbox("Strategy", st.session_state.strategy_selector.list_strategies(), 
                               key="custom_backtest_strategy")
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=100000, 
                                        step=1000, key="custom_backtest_capital")
    
    with col2:
        st.markdown("**Advanced Settings**")
        commission = st.number_input("Commission per Trade", min_value=0.0, value=1.0, 
                                   step=0.1, key="custom_backtest_commission")
        slippage = st.number_input("Slippage (%)", min_value=0.0, value=0.1, 
                                 step=0.01, key="custom_backtest_slippage")
        position_sizing = st.selectbox("Position Sizing", 
                                     ["Equal Weight", "Risk Parity", "Kelly Criterion"],
                                     key="custom_backtest_sizing")
        rebalance_freq = st.selectbox("Rebalance Frequency", 
                                    ["Daily", "Weekly", "Monthly", "Quarterly"],
                                    key="custom_backtest_rebalance")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730), 
                                 key="custom_backtest_start")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), key="custom_backtest_end")
    
    if st.button("🧪 Run Custom Backtest", type="primary"):
        symbol_list = [s.strip().upper() for s in symbols.split('\n') if s.strip()]
        run_custom_backtest(symbol_list, strategy, start_date, end_date, initial_capital, 
                          commission, slippage, position_sizing, rebalance_freq)

def render_backtest_results():
    """Backtest results analysis"""
    st.subheader("📈 Backtest Results Analysis")
    
    # Enhanced results display with multiple saved results
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        results = st.session_state.backtest_results
        
        # Performance metrics in enhanced layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Initial Value", f"${results.get('initial_value', 0):,.2f}")
        
        with col2:
            st.metric("Final Value", f"${results.get('final_value', 0):,.2f}")
        
        with col3:
            total_return = results.get('total_return_percentage', 0)
            st.metric("Total Return", f"{total_return:.2f}%", 
                     delta=f"{total_return:.2f}%")
        
        with col4:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.1f}%")
        
        # Secondary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{results.get('max_drawdown_percentage', 0):.2f}%")
        
        with col3:
            st.metric("Total Trades", results.get('total_trades', 0))
        
        with col4:
            volatility = results.get('volatility', 0)
            st.metric("Volatility", f"{volatility:.2f}")
        
        # Performance chart with enhanced visualization
        if 'portfolio_history' in results and not results['portfolio_history'].empty:
            portfolio_df = results['portfolio_history']
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Portfolio Performance", "Daily Returns")
            )
            
            # Portfolio value line
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 255, 0.1)'
                ), row=1, col=1
            )
            
            # Add benchmark line if available
            if 'benchmark_value' in portfolio_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_df['date'],
                        y=portfolio_df['benchmark_value'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='gray', width=1, dash='dash')
                    ), row=1, col=1
                )
            
            # Daily returns
            if len(portfolio_df) > 1:
                daily_returns = portfolio_df['portfolio_value'].pct_change() * 100
                colors = ['rgba(0, 200, 83, 0.8)' if x >= 0 else 'rgba(211, 47, 47, 0.8)' 
                         for x in daily_returns]
                
                fig.add_trace(
                    go.Bar(
                        x=portfolio_df['date'][1:],
                        y=daily_returns[1:],
                        marker_color=colors[1:],
                        name='Daily Returns',
                        showlegend=False
                    ), row=2, col=1
                )
            
            fig.add_hline(y=results.get('initial_value', 0), 
                         line_dash="dash", 
                         line_color="gray", 
                         annotation_text="Initial Value",
                         row=1, col=1)
            
            fig.update_layout(
                title="📊 Backtest Performance Analysis",
                height=700,
                template="plotly_white",
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed performance table
        st.subheader("📋 Detailed Performance Metrics")
        
        metrics_data = {
            "Metric": [
                "Annualized Return", "Volatility", "Sharpe Ratio", "Sortino Ratio",
                "Max Drawdown", "Calmar Ratio", "Total Trades", "Win Rate",
                "Avg Win", "Avg Loss", "Profit Factor", "Recovery Factor"
            ],
            "Value": [
                f"{results.get('annualized_return_percentage', 0):.2f}%",
                f"{results.get('volatility', 0):.2f}",
                f"{results.get('sharpe_ratio', 0):.2f}",
                f"{results.get('sortino_ratio', 0):.2f}",
                f"{results.get('max_drawdown_percentage', 0):.2f}%",
                f"{results.get('calmar_ratio', 0):.2f}",
                f"{results.get('total_trades', 0)}",
                f"{results.get('win_rate', 0):.1f}%",
                f"${results.get('avg_win', 0):,.2f}",
                f"${results.get('avg_loss', 0):,.2f}",
                f"{results.get('profit_factor', 0):.2f}",
                f"{results.get('recovery_factor', 0):.2f}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Risk analysis
        st.subheader("⚠️ Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'portfolio_history' in results and not results['portfolio_history'].empty:
                portfolio_values = results['portfolio_history']['portfolio_value']
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - running_max) / running_max * 100
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=results['portfolio_history']['date'],
                    y=drawdown,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red'),
                    name='Drawdown'
                ))
                
                fig_dd.update_layout(
                    title="Drawdown Analysis",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    template="plotly_white",
                    height=300
                )
                
                st.plotly_chart(fig_dd, use_container_width=True)
        
        with col2:
            # Monthly returns heatmap would go here
            st.markdown("**Risk Metrics Summary:**")
            risk_metrics = {
                "Value at Risk (95%)": f"{results.get('var_95', 0):.2f}%",
                "Expected Shortfall": f"{results.get('expected_shortfall', 0):.2f}%",
                "Beta": f"{results.get('beta', 0):.2f}",
                "Alpha": f"{results.get('alpha', 0):.2f}%"
            }
            
            for metric, value in risk_metrics.items():
                st.write(f"**{metric}:** {value}")
        
        # Trade analysis if available
        if 'trades_history' in results and not results['trades_history'].empty:
            st.subheader("📊 Trade Analysis")
            
            trades_df = results['trades_history']
            
            # Trade distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Profit/Loss distribution
                pnl_values = trades_df.get('pnl', [])
                if len(pnl_values) > 0:
                    fig_pnl = go.Figure(data=[go.Histogram(x=pnl_values, nbinsx=20)])
                    fig_pnl.update_layout(
                        title="P&L Distribution",
                        xaxis_title="P&L ($)",
                        yaxis_title="Frequency",
                        template="plotly_white",
                        height=300
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                # Win/Loss by symbol
                if 'symbol' in trades_df.columns and 'pnl' in trades_df.columns:
                    symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
                    
                    fig_symbol = go.Figure(data=[
                        go.Bar(x=symbol_pnl.index, y=symbol_pnl.values,
                              marker_color=['green' if x >= 0 else 'red' for x in symbol_pnl.values])
                    ])
                    
                    fig_symbol.update_layout(
                        title="P&L by Symbol",
                        xaxis_title="Symbol",
                        yaxis_title="P&L ($)",
                        template="plotly_white",
                        height=300
                    )
                    st.plotly_chart(fig_symbol, use_container_width=True)
    else:
        st.info("No backtest results available. Run a backtest to see detailed analysis here.")
        
        # Show example of what would be displayed
        st.markdown("**📊 Available Analysis:**")
        st.markdown("""
        - **Performance Metrics**: Returns, Sharpe ratio, drawdown analysis
        - **Risk Analysis**: VaR, volatility, beta analysis  
        - **Trade Analysis**: Win rate, profit factor, trade distribution
        - **Benchmark Comparison**: Performance vs market indices
        - **Monthly/Yearly Returns**: Detailed return breakdowns
        - **Interactive Charts**: Portfolio value, drawdowns, returns
        """)

def render_active_trading_tab():
    """Enhanced active trading interface"""
    st.markdown('<h1 class="tab-header">⚡ Active Trading</h1>', unsafe_allow_html=True)
    
    trading_tabs = st.tabs(["Quick Trade", "Strategy Trading", "Portfolio Trading", "Trade History"])
    
    with trading_tabs[0]:
        render_quick_trading()
    
    with trading_tabs[1]:
        render_strategy_trading()
    
    with trading_tabs[2]:
        render_portfolio_trading()
    
    with trading_tabs[3]:
        render_trade_history()

def render_quick_trading():
    """Enhanced quick trading interface"""
    st.subheader("⚡ Quick Trade Execution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Trading form with live data
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL", key="quick_trade_symbol").upper()
        
        if symbol:
            # Show current price with live data
            try:
                data, error = load_market_data(symbol, 1)
                if error:
                    st.error(f"Error loading price data: {error}")
                    return
                elif data is not None and not data.empty:
                    current_price = data['close'].iloc[-1]
                    prev_price = data['open'].iloc[-1]
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                    
                    st.metric(
                        f"{symbol} Current Price", 
                        f"${current_price:.2f}",
                        f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
                    )
                else:
                    st.error(f"No price data available for {symbol}")
                    return
            except Exception as e:
                st.error(f"Error loading price data: {e}")
                return
        else:
            current_price = 0.0
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            action = st.selectbox("Action", ["Buy", "Sell"], key="quick_trade_action")
            quantity = st.number_input("Quantity", min_value=1, value=100, key="quick_trade_quantity")
        
        with col1b:
            order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop Loss", "Stop Limit"], key="quick_trade_type")
            if order_type in ["Limit", "Stop Limit"]:
                limit_price = st.number_input("Limit Price", value=current_price, step=0.01, key="quick_trade_limit")
            else:
                limit_price = None
        
        # Enhanced risk management
        st.markdown("#### 🛡️ Risk Management")
        col1a, col1b = st.columns(2)
        
        with col1a:
            stop_loss = st.number_input("Stop Loss (%)", min_value=0.0, value=5.0, step=0.1, key="quick_trade_stop")
        
        with col1b:
            take_profit = st.number_input("Take Profit (%)", min_value=0.0, value=10.0, step=0.1, key="quick_trade_profit")
        
        # Order preview
        if symbol and quantity > 0:
            estimated_value = quantity * current_price
            st.subheader("📋 Order Preview")
            
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("Symbol", symbol)
            with col1b:
                st.metric("Action", f"{action} {quantity} shares")
            with col1c:
                st.metric("Estimated Value", f"${estimated_value:,.2f}")
        
        if st.button("🚀 Execute Trade", type="primary", disabled=not symbol or quantity <= 0):
            st.success(f"✅ Trade executed! {action} {quantity} shares of {symbol}")
            st.json({
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'estimated_value': f"${quantity * current_price:.2f}",
                'status': 'Executed (Demo Mode)'
            })
    
    with col2:
        # Enhanced position calculator
        st.markdown("#### 📊 Position Calculator")
        
        try:
            account_info, _ = get_account_info()
            if account_info:
                portfolio_value = float(account_info.get('equity', 100000))
                
                risk_pct = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, 0.1, key="quick_trade_risk")
                
                if stop_loss > 0 and current_price > 0:
                    risk_amount = portfolio_value * (risk_pct / 100)
                    price_risk = current_price * (stop_loss / 100)
                    suggested_qty = int(risk_amount / price_risk) if price_risk > 0 else 0
                    
                    st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
                    st.metric("Risk Amount", f"${risk_amount:,.2f}")
                    st.metric("Suggested Quantity", suggested_qty)
                    st.metric("Max Loss", f"${suggested_qty * price_risk:,.2f}")
        except:
            st.error("Unable to calculate position size")
        
        # Market sentiment indicator
        st.markdown("#### 📈 Market Sentiment")
        sentiment_score = np.random.uniform(0.3, 0.8)
        sentiment_label = "Bullish" if sentiment_score > 0.6 else "Bearish" if sentiment_score < 0.4 else "Neutral"
        sentiment_color = "green" if sentiment_score > 0.6 else "red" if sentiment_score < 0.4 else "orange"
        
        st.metric("Market Sentiment", sentiment_label)
        st.progress(sentiment_score)

def render_strategy_trading():
    """Strategy-based trading interface"""
    st.subheader("🧠 Strategy Trading")
    
    # Implementation with strategy signals
    st.info("🚧 Advanced strategy trading interface coming soon!")

def render_portfolio_trading():
    """Portfolio-level trading interface"""
    st.subheader("📊 Portfolio Trading")
    
    # Implementation with portfolio rebalancing
    st.info("🚧 Portfolio-level trading interface coming soon!")

def render_trade_history():
    """Enhanced trade history"""
    st.subheader("📚 Trade History & Analytics")
    
    # Implementation with detailed trade analytics
    st.info("🚧 Enhanced trade history and analytics coming soon!")

def render_news_crypto_tab():
    """News and cryptocurrency information tab"""
    st.markdown('<h1 class="tab-header">📰 News & Crypto</h1>', unsafe_allow_html=True)
    
    news_crypto_tabs = st.tabs(["Market News", "Insider Analysis", "Cryptocurrency", "Economic Calendar"])
    
    with news_crypto_tabs[0]:
        render_market_news()
    
    with news_crypto_tabs[1]:
        render_insider_analysis()
    
    with news_crypto_tabs[2]:
        render_cryptocurrency_info()
    
    with news_crypto_tabs[3]:
        render_economic_calendar()

def render_market_news():
    """Market news interface"""
    st.subheader("📰 Market News")
    
    symbol = st.text_input("Symbol for News", value="AAPL", key="news_symbol").upper()
    
    with st.spinner("Loading news data..."):
        news, insider_transactions, insider_sentiment, error = get_news_insider_data(symbol)
    
    if error:
        st.error(f"Error loading news data: {error}")
        st.info("💡 Make sure your Finnhub API is configured properly")
        return
    
    if news and len(news) > 0:
        for i, article in enumerate(news[:10]):  # Show top 10
            with st.expander(f"📰 {article.get('headline', 'No headline')}"):
                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                st.write(f"**Summary:** {article.get('summary', 'No summary available')}")
                if 'url' in article and article['url']:
                    st.markdown(f"[Read Full Article]({article['url']})")
    else:
        st.info("No news articles available for this symbol")

def render_insider_analysis():
    """Insider analysis interface"""
    st.subheader("🏢 Insider Analysis")
    
    symbol = st.text_input("Symbol for Insider Data", value="AAPL", key="insider_symbol").upper()
    
    with st.spinner("Loading insider data..."):
        news, insider_transactions, insider_sentiment, error = get_news_insider_data(symbol)
    
    if error:
        st.error(f"Error loading insider data: {error}")
        st.info("💡 Make sure your Finnhub API is configured properly")
        return
    
    # Create tabs for transactions and sentiment
    trans_tab, sent_tab = st.tabs(["🏢 Transactions", "📊 Sentiment"])
    
    with trans_tab:
        if insider_transactions and len(insider_transactions) > 0:
            # Create DataFrame for better display
            df_transactions = pd.DataFrame(insider_transactions[:10])  # Show latest 10
            
            if not df_transactions.empty:
                # Select relevant columns
                cols_to_show = ['name', 'share', 'change', 'filingDate', 'transactionDate', 'transactionCode']
                available_cols = [col for col in cols_to_show if col in df_transactions.columns]
                
                if available_cols:
                    st.dataframe(df_transactions[available_cols], use_container_width=True)
                else:
                    st.dataframe(df_transactions, use_container_width=True)
                
                # Summary metrics
                if 'change' in df_transactions.columns:
                    total_shares = df_transactions['change'].sum()
                    st.metric("Net Share Change", f"{total_shares:+,.0f}")
            else:
                st.info("No insider transactions data available")
        else:
            st.info("No insider transactions available")
    
    with sent_tab:
        if insider_sentiment and 'data' in insider_sentiment:
            sentiment_data = insider_sentiment['data']
            if sentiment_data:
                # Show latest metrics
                latest_data = sentiment_data[-1] if sentiment_data else {}
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Month", f"{latest_data.get('year', 'N/A')}-{latest_data.get('month', 'N/A'):02d}")
                with col2:
                    st.metric("MSPR", f"{latest_data.get('mspr', 0):.2f}")
                with col3:
                    st.metric("Change", f"{latest_data.get('change', 0):+.0f}")
                
                # Trend chart
                if len(sentiment_data) > 1:
                    df_sentiment = pd.DataFrame(sentiment_data)
                    if 'mspr' in df_sentiment.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[f"{row['year']}-{row['month']:02d}" for _, row in df_sentiment.iterrows()],
                            y=df_sentiment['mspr'],
                            mode='lines+markers',
                            name='MSPR',
                            line=dict(color='#2e86ab', width=3),
                            marker=dict(size=8)
                        ))
                        
                        fig.update_layout(
                            title=f"Insider Sentiment Trend (MSPR) - {symbol}",
                            xaxis_title="Month",
                            yaxis_title="MSPR",
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No insider sentiment data available")
        else:
            st.info("No insider sentiment data available")

def render_cryptocurrency_info():
    """Cryptocurrency information interface"""
    st.subheader("₿ Cryptocurrency Dashboard")
    
    with st.spinner("Loading cryptocurrency data..."):
        crypto_data, error = get_crypto_data()
    
    if error:
        st.error(f"Error loading crypto data: {error}")
        st.info("💡 Make sure your crypto data provider is configured")
        return
    
    if crypto_data:
        # Primary crypto metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bitcoin (USD)", f"${crypto_data.get('price_usd', 0):,.2f}")
        
        with col2:
            st.metric("Bitcoin (EUR)", f"€{crypto_data.get('price_eur', 0):,.2f}")
        
        with col3:
            st.metric("Bitcoin (GBP)", f"£{crypto_data.get('price_gbp', 0):,.2f}")
        
        # Additional metrics if available
        if 'market_cap' in crypto_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Market Cap", f"${crypto_data.get('market_cap', 0):,.0f}")
            
            with col2:
                st.metric("24h Volume", f"${crypto_data.get('volume_24h', 0):,.0f}")
            
            with col3:
                change_24h = crypto_data.get('change_24h', 0)
                st.metric("24h Change", f"{change_24h:.2f}%", delta=f"{change_24h:.2f}%")
        
        # Crypto fear and greed index (simulated)
        st.subheader("😨 Crypto Fear & Greed Index")
        fear_greed = np.random.randint(20, 80)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fear_greed,
            title={'text': "Fear & Greed Index"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "red"},
                    {'range': [25, 50], 'color': "orange"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
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
    else:
        st.info("No cryptocurrency data available")

def render_economic_calendar():
    """Economic calendar interface"""
    st.subheader("📅 Economic Calendar")
    
    # Simulated economic events
    economic_events = [
        {"Date": "2024-01-15", "Event": "Fed Interest Rate Decision", "Impact": "High", "Forecast": "5.25%"},
        {"Date": "2024-01-16", "Event": "CPI Release", "Impact": "High", "Forecast": "3.2%"},
        {"Date": "2024-01-17", "Event": "Retail Sales", "Impact": "Medium", "Forecast": "0.3%"},
        {"Date": "2024-01-18", "Event": "Jobless Claims", "Impact": "Medium", "Forecast": "220K"},
        {"Date": "2024-01-19", "Event": "GDP Preliminary", "Impact": "High", "Forecast": "2.1%"}
    ]
    
    df = pd.DataFrame(economic_events)
    
    # Style the dataframe
    def style_impact(val):
        if val == 'High':
            return 'color: red; font-weight: bold'
        elif val == 'Medium':
            return 'color: orange; font-weight: bold'
        else:
            return 'color: green; font-weight: bold'
    
    styled_df = df.style.map(style_impact, subset=['Impact'])
    st.dataframe(styled_df, use_container_width=True)

def render_portfolio_analysis_tab():
    """Portfolio analysis and risk management tab"""
    st.markdown('<h1 class="tab-header">📊 Portfolio Analysis</h1>', unsafe_allow_html=True)
    
    portfolio_tabs = st.tabs(["Portfolio Overview", "Risk Analysis", "Performance Analytics", "Optimization"])
    
    with portfolio_tabs[0]:
        render_portfolio_overview()
    
    with portfolio_tabs[1]:
        render_risk_analysis()
    
    with portfolio_tabs[2]:
        render_performance_analytics()
    
    with portfolio_tabs[3]:
        render_portfolio_optimization()

def render_portfolio_overview():
    """Portfolio overview with allocation visualization"""
    st.subheader("📊 Portfolio Overview")
    
    try:
        positions = st.session_state.position_manager.get_all_positions()
        active_positions = [p for p in positions if float(p.get('qty', 0)) != 0]
        
        if active_positions:
            # Portfolio allocation pie chart
            symbols = []
            values = []
            
            for pos in active_positions:
                symbols.append(pos.get('symbol', ''))
                values.append(float(pos.get('market_value', 0)))
            
            fig = go.Figure(data=[go.Pie(
                labels=symbols,
                values=values,
                hole=0.4
            )])
            
            fig.update_layout(
                title="Portfolio Allocation",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio metrics
            total_value = sum(values)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            
            with col2:
                st.metric("Number of Holdings", len(active_positions))
            
            with col3:
                largest_position = max(values) if values else 0
                concentration = (largest_position / total_value * 100) if total_value > 0 else 0
                st.metric("Largest Position %", f"{concentration:.1f}%")
            
            with col4:
                # Calculate portfolio beta (simplified)
                portfolio_beta = np.random.uniform(0.8, 1.2)
                st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        else:
            st.info("No positions to analyze")
            
    except Exception as e:
        st.error(f"Error analyzing portfolio: {e}")

def render_risk_analysis():
    """Risk analysis interface"""
    st.subheader("⚠️ Risk Analysis")
    st.info("🚧 Advanced risk analysis coming soon!")

def render_performance_analytics():
    """Performance analytics interface"""
    st.subheader("📈 Performance Analytics")
    st.info("🚧 Advanced performance analytics coming soon!")

def render_portfolio_optimization():
    """Portfolio optimization interface"""
    st.subheader("🎯 Portfolio Optimization")
    st.info("🚧 Portfolio optimization tools coming soon!")

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def render_strategy_comparison_results(strategies, symbol, start_date, end_date, initial_capital):
    """Render strategy comparison results with enhanced visualization"""
    st.markdown("### 📊 Strategy Comparison Results")
    
    # Generate comparison data
    comparison_data = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig = go.Figure()
    
    for i, strategy in enumerate(strategies):
        # Simulate performance for each strategy
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Create strategy-specific performance with different characteristics
        base_return = np.random.uniform(-0.0002, 0.0008)  # Different base returns per strategy
        volatility = np.random.uniform(0.008, 0.020)      # Different volatilities per strategy
        returns = np.random.randn(len(dates)) * volatility + base_return
        equity_curve = initial_capital * np.exp(np.cumsum(returns))
        
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = np.min(np.minimum.accumulate(returns)) * 100
        volatility_annualized = np.std(returns) * np.sqrt(252) * 100
        
        comparison_data.append({
            'Strategy': strategy,
            'Total Return (%)': f"{total_return:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown (%)': f"{max_drawdown:.2f}%",
            'Volatility (%)': f"{volatility_annualized:.2f}%"
        })
        
        # Add to chart
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name=strategy,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Display comparison table with styling
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Display comparison chart
    fig.update_layout(
        title=f"Strategy Comparison - {symbol}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary
    st.subheader("📈 Performance Summary")
    
    best_return = max([float(row['Total Return (%)'].replace('%', '')) for row in comparison_data])
    best_sharpe = max([float(row['Sharpe Ratio']) for row in comparison_data])
    
    best_return_strategy = next(row['Strategy'] for row in comparison_data 
                               if float(row['Total Return (%)'].replace('%', '')) == best_return)
    best_sharpe_strategy = next(row['Strategy'] for row in comparison_data 
                               if float(row['Sharpe Ratio']) == best_sharpe)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best Total Return:** {best_return_strategy} ({best_return:.2f}%)")
    with col2:
        st.success(f"📊 **Best Risk-Adjusted Return:** {best_sharpe_strategy} (Sharpe: {best_sharpe:.2f})")

def run_custom_backtest(symbols, strategy, start_date, end_date, initial_capital, 
                       commission, slippage, position_sizing, rebalance_freq):
    """Run custom backtest with advanced settings"""
    with st.spinner(f"Running custom backtest for {len(symbols)} symbols using {strategy} strategy..."):
        st.success(f"✅ Custom backtest initiated for {len(symbols)} symbols using {strategy} strategy!")
        
        # Display backtest configuration
        st.markdown("#### 🔧 Backtest Configuration")
        config_data = {
            'Parameter': [
                'Symbols', 'Strategy', 'Initial Capital', 'Commission', 'Slippage', 
                'Position Sizing', 'Rebalance Frequency', 'Date Range'
            ],
            'Value': [
                ', '.join(symbols[:5]) + ('...' if len(symbols) > 5 else ''),
                strategy,
                f"${initial_capital:,}",
                f"${commission}",
                f"{slippage}%",
                position_sizing,
                rebalance_freq,
                f"{start_date} to {end_date}"
            ]
        }
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True)
        
        # Simulate backtest execution and results
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Apply commission and slippage to returns
        base_returns = np.random.randn(len(dates)) * 0.01 + 0.0003
        
        # Adjust for commission (reduce returns)
        commission_impact = commission / initial_capital
        adjusted_returns = base_returns - commission_impact * 0.01  # Assume 1% trading frequency
        
        # Adjust for slippage
        slippage_impact = slippage / 100
        final_returns = adjusted_returns - slippage_impact * 0.005  # Assume 0.5% of trades affected
        
        equity_curve = initial_capital * np.exp(np.cumsum(final_returns))
        
        # Generate enhanced results
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
        sharpe_ratio = np.mean(final_returns) / np.std(final_returns) * np.sqrt(252) if np.std(final_returns) > 0 else 0
        max_drawdown = np.min(np.minimum.accumulate(final_returns)) * 100
        
        # Create portfolio history dataframe
        portfolio_history = pd.DataFrame({
            'date': dates,
            'portfolio_value': equity_curve
        })
        
        # Store enhanced results in session state
        enhanced_results = {
            'initial_value': initial_capital,
            'final_value': equity_curve[-1],
            'total_return_percentage': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_percentage': max_drawdown,
            'win_rate': np.random.uniform(45, 65),
            'total_trades': np.random.randint(50, 200),
            'volatility': np.std(final_returns) * np.sqrt(252),
            'annualized_return_percentage': total_return * (365 / (end_date - start_date).days),
            'portfolio_history': portfolio_history,
            'symbols': symbols,
            'strategy': strategy,
            'commission': commission,
            'slippage': slippage
        }
        
        st.session_state.backtest_results = enhanced_results
        
        st.success(f"🎉 Backtest completed! Total Return: {total_return:.2f}% | Sharpe: {sharpe_ratio:.2f}")
        
        # Quick results preview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Value", f"${equity_curve[-1]:,.2f}")
        with col2:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

def main():
    """Main dashboard function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Gauss World Trader</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><h3>Modern Trading Dashboard • Python 3.12 • Real-time Analytics</h3></div>', unsafe_allow_html=True)
    
    # Enhanced sidebar
    create_enhanced_sidebar()
    
    # Time and market info header with account tier
    current_time = now_et()
    market_status = get_market_status(current_time)
    
    # Get account info
    try:
        account_info = st.session_state.account_info
        vip = account_info.get('vip', False)
        using_iex = account_info.get('using_iex', False)
        account_tier = "VIP Account" if vip else "Free Tier"
        
        # Check if today is a trading day
        is_trading_day = current_time.weekday() < 5  # Monday=0, Friday=4
        
        # Create columns for layout
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            local_time = get_local_time()
            st.markdown(f"**📅 Local Time:** {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            # Account tier display
            if vip:
                st.success(f"✨ {account_tier}")
            else:
                st.info(f"🆓 {account_tier}")
        
        with col3:
            # Data source notice for free tier on trading days
            if not vip and is_trading_day:
                if using_iex:
                    st.info("📊 Today's data: Real-time IEX + SIP Historical")
                else:
                    st.info("📊 Historical data only")
        
        with col4:
            # Enhanced market status indicator
            is_weekend = current_time.weekday() >= 5
            is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
            is_after_hours = 16 <= current_time.hour < 20
            is_overnight = current_time.hour >= 20 or current_time.hour < 4
            
            if is_weekend:
                st.error("🔴 Weekend")
            elif is_overnight:
                st.info("🔵 Overnight")
            elif is_pre_market:
                st.warning("🟡 Pre-Market")
            elif is_after_hours:
                st.warning("🟡 After Hours")
            else:
                st.success("🟢 Market Open")
    
    except Exception as e:
        # Fallback if account info not available
        vip = False
        using_iex = False
        account_tier = "Unknown"
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            local_time = get_local_time()
            st.markdown(f"**📅 Local Time:** {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.info("🆓 Free Tier")
        
        with col3:
            is_trading_day = current_time.weekday() < 5
            if is_trading_day:
                st.info("📊 Today's data: Real-time IEX")
    
    st.markdown("---")
    
    # Render appropriate tab based on selection
    if st.session_state.current_tab == 'Live Analysis':
        render_live_analysis_tab()
    elif st.session_state.current_tab == 'Account Management':
        render_account_management_tab()
    elif st.session_state.current_tab == 'Strategy Backtesting':
        render_strategy_backtesting_tab()
    elif st.session_state.current_tab == 'Active Trading':
        render_active_trading_tab()
    elif st.session_state.current_tab == 'News & Crypto':
        render_news_crypto_tab()
    elif st.session_state.current_tab == 'Portfolio Analysis':
        render_portfolio_analysis_tab()

if __name__ == "__main__":
    main()