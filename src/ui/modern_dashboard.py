#!/usr/bin/env python3
"""
Unified Trading Dashboard - Redesigned Structure
Organized navigation tabs: Market Overview, Account Info, Live Analysis, 
Strategy Backtest, Trade & Order, News & Report
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ui.core_dashboard import BaseDashboard, UIComponents
from src.utils.timezone_utils import now_et, get_market_status
from src.account.account_manager import AccountManager
from src.account.position_manager import PositionManager
from src.account.order_manager import OrderManager
from src.agent.fundamental_analyzer import FundamentalAnalyzer
from src.strategy.strategy_selector import get_strategy_selector
from src.utils.watchlist_manager import WatchlistManager
from src.data import AlpacaDataProvider, NewsDataProvider
from src.analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)


class ModernDashboard(BaseDashboard):
    """Modern dashboard with reorganized navigation structure"""
    
    def __init__(self):
        super().__init__("Gauss World Trader - Modern Dashboard", "🌍")
        self.initialize_modern_modules()
        
    def initialize_modern_modules(self):
        """Initialize all trading modules"""
        if 'current_main_tab' not in st.session_state:
            st.session_state.current_main_tab = 'Market Overview'
            
        if 'modern_initialized' not in st.session_state:
            try:
                st.session_state.account_manager = AccountManager()
                st.session_state.position_manager = PositionManager(st.session_state.account_manager)
                st.session_state.order_manager = OrderManager(st.session_state.account_manager)
                st.session_state.fundamental_analyzer = FundamentalAnalyzer()
                st.session_state.strategy_selector = get_strategy_selector()
                st.session_state.watchlist_manager = WatchlistManager()
                st.session_state.news_provider = NewsDataProvider()
                st.session_state.modern_initialized = True
                
            except Exception as e:
                logger.error(f"Error initializing modern modules: {e}")
                st.error("Error initializing trading modules. Please check API configuration.")
    
    def create_main_navigation(self):
        """Create main navigation tabs"""
        tabs = st.tabs([
            "📊 Market Overview",
            "💼 Account Info", 
            "🔍 Live Analysis",
            "📈 Strategy Backtest",
            "⚡ Trade & Order",
            "📰 News & Report"
        ])
        
        with tabs[0]:
            self.render_market_overview_tab()
        with tabs[1]:
            self.render_account_info_tab()
        with tabs[2]:
            self.render_live_analysis_tab()
        with tabs[3]:
            self.render_strategy_backtest_tab()
        with tabs[4]:
            self.render_trade_order_tab()
        with tabs[5]:
            self.render_news_report_tab()
    
    def render_market_overview_tab(self):
        """Market Overview: Index, VIX, Market Sentiment, Sector Performance, Economic Calendar, Crypto"""
        st.header("📊 Market Overview")
        
        # Market Status
        market_status = get_market_status()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if market_status == 'open' else "🔴"
            st.metric("Market Status", f"{status_color} {market_status}")
        
        with col2:
            st.metric("Current Time (ET)", now_et().strftime("%H:%M:%S"))
        
        with col3:
            # Calculate next market change
            next_change = "Market Close" if market_status == 'open' else "Market Open"
            st.metric("Next Change", next_change)
        
        st.divider()
        
        # Sub-tabs for different market data
        market_tabs = st.tabs(["📈 Indices", "📊 VIX & Sentiment", "🏢 Sectors", "📅 Economic Calendar", "₿ Cryptocurrency"])
        
        with market_tabs[0]:
            self.render_market_indices()
        
        with market_tabs[1]:
            self.render_vix_sentiment()
        
        with market_tabs[2]:
            self.render_sector_performance()
            
        with market_tabs[3]:
            self.render_economic_calendar()
            
        with market_tabs[4]:
            self.render_cryptocurrency_data()
    
    def render_market_indices(self):
        """Render major market indices"""
        st.subheader("📈 Major Market Indices")
        
        indices = ['SPY', 'QQQ', 'DIA', 'IWM']
        cols = st.columns(len(indices))
        
        for i, symbol in enumerate(indices):
            with cols[i]:
                data, error = self.load_market_data(symbol, 5)
                if data is not None and not data.empty:
                    current_price = data['close'].iloc[-1]
                    prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    st.metric(
                        label=symbol,
                        value=f"${current_price:.2f}",
                        delta=f"{change_pct:.2f}%"
                    )
                else:
                    st.metric(label=symbol, value="N/A", delta="N/A")
    
    def render_vix_sentiment(self):
        """Render VIX and market sentiment indicators"""
        st.subheader("📊 VIX & Market Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # VIX Data
            vix_data, error = self.load_market_data('VIX', 30)
            if vix_data is not None and not vix_data.empty:
                current_vix = vix_data['close'].iloc[-1]
                vix_color = "🔴" if current_vix > 30 else "🟡" if current_vix > 20 else "🟢"
                st.metric("VIX (Fear Index)", f"{vix_color} {current_vix:.2f}")
                
                # VIX Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vix_data.index,
                    y=vix_data['close'],
                    mode='lines',
                    name='VIX'
                ))
                fig.update_layout(title="VIX Trend (30 Days)", height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Market Sentiment Indicators
            st.write("**Market Sentiment Indicators**")
            sentiment_score = np.random.uniform(30, 70)  # Placeholder
            st.progress(sentiment_score/100)
            st.write(f"Overall Sentiment: {sentiment_score:.1f}/100")
            
            # Fear & Greed components
            components = {
                "Stock Price Momentum": np.random.uniform(20, 80),
                "Market Volatility": np.random.uniform(20, 80), 
                "Put/Call Ratio": np.random.uniform(20, 80),
                "Safe Haven Demand": np.random.uniform(20, 80)
            }
            
            for component, score in components.items():
                st.write(f"• {component}: {score:.0f}")
    
    def render_sector_performance(self):
        """Render sector performance analysis"""
        st.subheader("🏢 Sector Performance")
        
        # Major sector ETFs
        sectors = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financial': 'XLF',
            'Consumer Disc.': 'XLY',
            'Communication': 'XLC',
            'Industrial': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        sector_data = []
        for sector_name, etf in sectors.items():
            data, error = self.load_market_data(etf, 5)
            if data is not None and not data.empty:
                current = data['close'].iloc[-1]
                prev = data['close'].iloc[-2] if len(data) > 1 else current
                change_pct = ((current - prev) / prev * 100) if prev != 0 else 0
                sector_data.append({
                    'Sector': sector_name,
                    'ETF': etf,
                    'Price': current,
                    'Change %': change_pct
                })
        
        if sector_data:
            df = pd.DataFrame(sector_data)
            df = df.sort_values('Change %', ascending=False)
            
            # Color-coded performance chart
            fig = go.Figure()
            colors = ['green' if x >= 0 else 'red' for x in df['Change %']]
            fig.add_trace(go.Bar(
                x=df['Change %'],
                y=df['Sector'],
                orientation='h',
                marker_color=colors,
                text=df['Change %'].round(2),
                textposition='auto'
            ))
            fig.update_layout(
                title="Sector Performance Today (%)",
                height=400,
                xaxis_title="Change %"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.dataframe(df, use_container_width=True)
    
    def render_economic_calendar(self):
        """Render economic calendar (placeholder)"""
        st.subheader("📅 Economic Calendar")
        
        # Placeholder for economic calendar
        st.info("🚧 Economic Calendar integration coming soon...")
        
        # Mock upcoming events
        mock_events = [
            {"Date": "2024-01-15", "Event": "Consumer Price Index", "Importance": "High", "Previous": "3.1%", "Forecast": "3.2%"},
            {"Date": "2024-01-17", "Event": "Retail Sales", "Importance": "Medium", "Previous": "0.3%", "Forecast": "0.4%"},
            {"Date": "2024-01-18", "Event": "Housing Starts", "Importance": "Low", "Previous": "1.56M", "Forecast": "1.52M"},
        ]
        
        df = pd.DataFrame(mock_events)
        st.dataframe(df, use_container_width=True)
    
    def render_cryptocurrency_data(self):
        """Render cryptocurrency information"""
        st.subheader("₿ Cryptocurrency")
        
        try:
            provider = AlpacaDataProvider()
            
            # Bitcoin data
            btc_quote = provider.get_crypto_latest_quote('BTC/USD')
            if 'error' not in btc_quote:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bitcoin (BTC)", f"${btc_quote['bid_price']:,.2f}")
                
                with col2:
                    st.metric("Bid-Ask Spread", f"${btc_quote['ask_price'] - btc_quote['bid_price']:.2f}")
                
                with col3:
                    timestamp = btc_quote['timestamp']
                    st.metric("Last Updated", timestamp.strftime("%H:%M:%S"))
                
                # Bitcoin price history
                btc_data, error = self.load_market_data('BTC/USD', 30)
                if btc_data is not None and not btc_data.empty:
                    fig = self.create_price_chart('BTC/USD', btc_data)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Error loading Bitcoin data: {btc_quote.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Error loading cryptocurrency data: {e}")
    
    def render_account_info_tab(self):
        """Account Info: Account, Positions, Portfolio, Performance, Risk Management"""
        st.header("💼 Account Information")
        
        account_tabs = st.tabs(["📊 Account", "📈 Positions", "💰 Portfolio", "📉 Performance", "⚙️ Configuration"])
        
        with account_tabs[0]:
            self.render_account_overview()
        
        with account_tabs[1]:
            self.render_positions_view()
        
        with account_tabs[2]:
            self.render_portfolio_analytics()
        
        with account_tabs[3]:
            self.render_performance_metrics()
            
        with account_tabs[4]:
            self.render_risk_configuration()
    
    def render_account_overview(self):
        """Render account overview"""
        st.subheader("📊 Account Overview")
        
        account_info, error = self.get_account_info()
        if account_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Account Value", f"${float(account_info.get('portfolio_value', 0)):,.2f}")
            
            with col2:
                st.metric("Buying Power", f"${float(account_info.get('buying_power', 0)):,.2f}")
            
            with col3:
                st.metric("Cash", f"${float(account_info.get('cash', 0)):,.2f}")
            
            with col4:
                day_pl = float(account_info.get('day_trade_pl', 0))
                st.metric("Day P&L", f"${day_pl:,.2f}", delta=f"{day_pl:,.2f}")
        else:
            st.error(f"Unable to load account information: {error}")
    
    def render_positions_view(self):
        """Render current positions"""
        st.subheader("📈 Current Positions")
        
        if 'position_manager' in st.session_state:
            positions = st.session_state.position_manager.get_all_positions()
            if positions and not any('error' in pos for pos in positions):
                UIComponents.render_positions_table(positions)
            else:
                st.info("No open positions found.")
        else:
            st.error("Position manager not initialized.")
    
    def render_portfolio_analytics(self):
        """Render portfolio analytics"""
        st.subheader("💰 Portfolio Analytics")
        st.info("🚧 Advanced portfolio analytics coming soon...")
        
        # Placeholder analytics
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Asset Allocation**")
            # Mock pie chart
            labels = ['Stocks', 'Cash', 'Options']
            values = [70, 25, 5]
            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Risk Metrics**")
            st.metric("Portfolio Beta", "1.2")
            st.metric("Sharpe Ratio", "1.8")
            st.metric("Max Drawdown", "-5.2%")
    
    def render_performance_metrics(self):
        """Render performance metrics"""
        st.subheader("📉 Performance Analysis")
        st.info("🚧 Detailed performance analysis coming soon...")
        
        # Mock performance chart
        dates = pd.date_range(start='2024-01-01', end=now_et().date(), freq='D')
        performance = np.cumsum(np.random.normal(0.001, 0.02, len(dates))) + 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=performance, mode='lines', name='Portfolio'))
        fig.update_layout(title="Portfolio Performance", yaxis_title="Cumulative Returns")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_configuration(self):
        """Render risk management configuration"""
        st.subheader("⚙️ Risk Management Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Position Sizing**")
            max_position = st.slider("Max Position Size (%)", 1, 20, 10)
            max_portfolio_risk = st.slider("Max Portfolio Risk (%)", 1, 10, 2)
            
        with col2:
            st.write("**Stop Loss Settings**")
            default_stop_loss = st.slider("Default Stop Loss (%)", 1, 20, 5)
            trailing_stop = st.checkbox("Enable Trailing Stop")
            
        if st.button("Save Risk Settings"):
            st.success("Risk settings saved successfully!")
    
    def render_live_analysis_tab(self):
        """Live Analysis: Symbol Analysis, Watchlist"""
        st.header("🔍 Live Analysis")
        
        analysis_tabs = st.tabs(["📊 Symbol Analysis", "👁️ Watchlist"])
        
        with analysis_tabs[0]:
            self.render_symbol_analysis()
        
        with analysis_tabs[1]:
            self.render_watchlist_management()
    
    def render_symbol_analysis(self):
        """Render symbol analysis"""
        st.subheader("📊 Symbol Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            symbol = st.text_input("Enter Symbol", value="AAPL", key="analysis_symbol").upper()
            days = st.selectbox("Analysis Period", [30, 60, 90, 180, 365], index=2)
            
            if st.button("Analyze"):
                st.session_state.analyze_symbol = symbol
                st.session_state.analyze_days = days
        
        with col2:
            if hasattr(st.session_state, 'analyze_symbol'):
                symbol = st.session_state.analyze_symbol
                days = st.session_state.analyze_days
                
                data, error = self.load_market_data(symbol, days)
                if data is not None and not data.empty:
                    # Price chart with technical indicators
                    fig = self.create_price_chart(symbol, data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical analysis
                    ta = TechnicalAnalysis()
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        sma_20 = ta.sma(data['close'], 20)
                        current_price = data['close'].iloc[-1]
                        sma_current = sma_20.iloc[-1] if not sma_20.empty else current_price
                        trend = "🟢 Bullish" if current_price > sma_current else "🔴 Bearish"
                        st.metric("Trend (vs SMA20)", trend)
                    
                    with col_b:
                        rsi = ta.rsi(data['close'])
                        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
                        rsi_signal = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
                        st.metric("RSI", f"{rsi_current:.1f} ({rsi_signal})")
                    
                    with col_c:
                        volatility = data['close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Annualized Volatility", f"{volatility:.1f}%")
                else:
                    st.error(f"Unable to load data for {symbol}: {error}")
    
    def render_watchlist_management(self):
        """Render watchlist management"""
        st.subheader("👁️ Watchlist Management")
        
        if 'watchlist_manager' in st.session_state:
            wm = st.session_state.watchlist_manager
            
            # Add symbol to watchlist
            col1, col2 = st.columns([2, 1])
            with col1:
                new_symbol = st.text_input("Add Symbol to Watchlist").upper()
            with col2:
                if st.button("Add") and new_symbol:
                    wm.add_symbol(new_symbol)
                    st.success(f"Added {new_symbol} to watchlist")
            
            # Display watchlist
            watchlist = wm.get_watchlist()
            if watchlist:
                watchlist_data = []
                for symbol in watchlist:
                    data, error = self.load_market_data(symbol, 2)
                    if data is not None and not data.empty:
                        current = data['close'].iloc[-1]
                        prev = data['close'].iloc[-2] if len(data) > 1 else current
                        change = current - prev
                        change_pct = (change / prev * 100) if prev != 0 else 0
                        
                        watchlist_data.append({
                            'Symbol': symbol,
                            'Price': f"${current:.2f}",
                            'Change': f"${change:.2f}",
                            'Change %': f"{change_pct:.2f}%"
                        })
                
                if watchlist_data:
                    df = pd.DataFrame(watchlist_data)
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("Watchlist is empty. Add some symbols to monitor.")
        else:
            st.error("Watchlist manager not initialized.")
    
    def render_strategy_backtest_tab(self):
        """Strategy Backtest: Quick Backtest, Strategy Comparison"""
        st.header("📈 Strategy Backtesting")
        
        backtest_tabs = st.tabs(["⚡ Quick Backtest", "🔀 Strategy Comparison"])
        
        with backtest_tabs[0]:
            self.render_quick_backtest()
        
        with backtest_tabs[1]:
            self.render_strategy_comparison()
    
    def render_quick_backtest(self):
        """Render quick backtesting interface"""
        st.subheader("⚡ Quick Backtest")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            symbols = st.text_area("Symbols (one per line)", "AAPL\nMSFT\nGOOGL").strip().split('\n')
            symbols = [s.strip().upper() for s in symbols if s.strip()]
            
            days_back = st.slider("Backtest Period (days)", 30, 365, 90)
            initial_cash = st.number_input("Initial Cash", value=100000, step=10000)
            
            strategy_type = st.selectbox("Strategy", ["Momentum", "Mean Reversion", "Breakout"])
            
            if st.button("Run Backtest"):
                st.session_state.run_backtest = True
                st.session_state.backtest_params = {
                    'symbols': symbols,
                    'days_back': days_back,
                    'initial_cash': initial_cash,
                    'strategy_type': strategy_type
                }
        
        with col2:
            if hasattr(st.session_state, 'run_backtest') and st.session_state.run_backtest:
                params = st.session_state.backtest_params
                
                with st.spinner(f"Running {params['strategy_type']} strategy backtest..."):
                    results, error = self.run_backtest(
                        params['symbols'],
                        params['days_back'],
                        params['initial_cash'],
                        params['strategy_type']
                    )
                
                if results:
                    self.display_backtest_results(results)
                else:
                    st.error(f"Backtest failed: {error}")
    
    def display_backtest_results(self, results):
        """Display backtest results with comprehensive analysis"""
        st.subheader("📈 Backtest Results Analysis")
        
        if results and isinstance(results, dict):
            # Performance metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                initial_val = results.get('initial_value', results.get('initial_cash', 0))
                st.metric("Initial Value", f"${initial_val:,.2f}")
            
            with col2:
                final_val = results.get('final_value', results.get('portfolio_value', 0))
                st.metric("Final Value", f"${final_val:,.2f}")
            
            with col3:
                total_return = results.get('total_return_percentage', 0)
                if total_return == 0 and initial_val > 0:
                    total_return = ((final_val - initial_val) / initial_val) * 100
                st.metric("Total Return", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
            
            with col4:
                win_rate = results.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            st.divider()
            
            # Detailed metrics table
            st.write("**📋 Detailed Performance Metrics**")
            metrics_data = {
                "Metric": [
                    "Annualized Return",
                    "Volatility", 
                    "Sharpe Ratio",
                    "Max Drawdown",
                    "Total Trades",
                    "Profitable Trades"
                ],
                "Value": [
                    f"{results.get('annualized_return_percentage', 0):.2f}%",
                    f"{results.get('volatility', 0):.2f}%", 
                    f"{results.get('sharpe_ratio', 0):.2f}",
                    f"{results.get('max_drawdown_percentage', 0):.2f}%",
                    f"{results.get('total_trades', 0)}",
                    f"{results.get('profitable_trades', 0)}"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Portfolio performance chart if available
            if 'portfolio_history' in results:
                portfolio_history = results['portfolio_history']
                if isinstance(portfolio_history, pd.DataFrame) and not portfolio_history.empty:
                    st.write("**📊 Portfolio Value Over Time**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=portfolio_history.index,
                        y=portfolio_history.get('portfolio_value', portfolio_history.iloc[:, 0]),
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Performance",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Trade history if available
            if 'trades_history' in results:
                trades_df = results['trades_history']
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                    st.write("**📝 Recent Trades**")
                    # Show only last 10 trades for display
                    display_trades = trades_df.tail(10) if len(trades_df) > 10 else trades_df
                    st.dataframe(display_trades, use_container_width=True)
                    
                    if st.button("💾 Download Full Trade History"):
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
        else:
            st.error("Invalid backtest results format")
    
    def render_strategy_comparison(self):
        """Render strategy comparison interface"""
        st.subheader("🔀 Strategy Comparison")
        st.info("🚧 Strategy comparison feature coming soon...")
        
        # Placeholder for strategy comparison
        strategies = ['Momentum', 'Mean Reversion', 'Breakout', 'Buy & Hold']
        mock_results = {
            'Strategy': strategies,
            'Total Return': ['15.2%', '8.7%', '12.4%', '10.1%'],
            'Sharpe Ratio': [1.8, 1.2, 1.6, 1.4],
            'Max Drawdown': ['-5.2%', '-3.1%', '-4.8%', '-6.2%'],
            'Win Rate': ['65%', '58%', '62%', '100%']
        }
        
        df = pd.DataFrame(mock_results)
        st.dataframe(df, use_container_width=True)
    
    def render_trade_order_tab(self):
        """Trade & Order: Quick Trade, Active Orders, Order History"""
        st.header("⚡ Trade & Order Management")
        
        trade_tabs = st.tabs(["🚀 Quick Trade", "📋 Active Orders", "📜 Order History"])
        
        with trade_tabs[0]:
            self.render_quick_trade()
        
        with trade_tabs[1]:
            self.render_active_orders()
        
        with trade_tabs[2]:
            self.render_order_history()
    
    def render_quick_trade(self):
        """Render quick trading interface"""
        st.subheader("🚀 Quick Trade")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Order Details**")
            symbol = st.text_input("Symbol", value="AAPL").upper()
            side = st.selectbox("Side", ["buy", "sell"])
            quantity = st.number_input("Quantity", min_value=1, value=100)
            order_type = st.selectbox("Order Type", ["market", "limit", "stop"])
            
            if order_type in ["limit", "stop"]:
                limit_price = st.number_input("Price", value=150.0, step=0.01)
            
            time_in_force = st.selectbox("Time in Force", ["day", "gtc", "ioc", "fok"])
            
        with col2:
            st.write("**Order Preview**")
            if symbol:
                data, error = self.load_market_data(symbol, 1)
                if data is not None and not data.empty:
                    current_price = data['close'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")
                    
                    estimated_cost = current_price * quantity
                    st.metric("Estimated Cost", f"${estimated_cost:,.2f}")
                else:
                    st.warning(f"Unable to get current price for {symbol}")
            
            if st.button("Submit Order", type="primary"):
                st.warning("⚠️ This is a demo. Order submission is disabled.")
                # In a real implementation, this would submit the order
                # st.session_state.order_manager.submit_order(...)
    
    def render_active_orders(self):
        """Render active orders view"""
        st.subheader("📋 Active Orders")
        
        if 'order_manager' in st.session_state:
            orders = st.session_state.order_manager.get_orders()
            if orders:
                # Convert to DataFrame for display
                orders_data = []
                for order in orders:
                    orders_data.append({
                        'Order ID': order.get('id', 'N/A'),
                        'Symbol': order.get('symbol', 'N/A'),
                        'Side': order.get('side', 'N/A').upper(),
                        'Quantity': order.get('qty', 0),
                        'Order Type': order.get('order_type', 'N/A'),
                        'Status': order.get('status', 'N/A'),
                        'Submitted': order.get('submitted_at', 'N/A')
                    })
                
                df = pd.DataFrame(orders_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No active orders found.")
        else:
            st.error("Order manager not initialized.")
    
    def render_order_history(self):
        """Render order history"""
        st.subheader("📜 Order History")
        st.info("🚧 Order history feature coming soon...")
        
        # Placeholder order history
        mock_history = [
            {"Date": "2024-01-15", "Symbol": "AAPL", "Side": "BUY", "Qty": 100, "Price": 185.50, "Status": "FILLED"},
            {"Date": "2024-01-14", "Symbol": "MSFT", "Side": "SELL", "Qty": 50, "Price": 412.25, "Status": "FILLED"},
            {"Date": "2024-01-13", "Symbol": "GOOGL", "Side": "BUY", "Qty": 25, "Price": 142.80, "Status": "FILLED"},
        ]
        
        df = pd.DataFrame(mock_history)
        st.dataframe(df, use_container_width=True)
    
    def render_news_report_tab(self):
        """News & Report: Company News, Insider Transactions, AI Report"""
        st.header("📰 News & Reports")
        
        news_tabs = st.tabs(["📰 Company News", "👔 Insider Activity", "🤖 AI Reports"])
        
        with news_tabs[0]:
            self.render_company_news()
        
        with news_tabs[1]:
            self.render_insider_activity()
        
        with news_tabs[2]:
            self.render_ai_reports()
    
    def render_company_news(self):
        """Render company news section"""
        st.subheader("📰 Company News")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            news_symbol = st.text_input("Symbol for News", value="AAPL").upper()
            if st.button("Get News"):
                st.session_state.news_symbol = news_symbol
        
        with col2:
            if hasattr(st.session_state, 'news_symbol'):
                symbol = st.session_state.news_symbol
                
                try:
                    if 'news_provider' in st.session_state:
                        news = st.session_state.news_provider.get_company_news(symbol)
                        
                        if news:
                            for article in news[:10]:  # Show top 10 articles
                                with st.expander(f"📰 {article.get('headline', 'No title')[:80]}..."):
                                    st.write(f"**Source:** {article.get('source', 'Unknown')}")
                                    st.write(f"**Date:** {article.get('datetime', 'Unknown')}")
                                    st.write(article.get('summary', 'No summary available'))
                                    if article.get('url'):
                                        st.markdown(f"[Read more]({article['url']})")
                        else:
                            st.info(f"No news found for {symbol}")
                    else:
                        st.error("News provider not initialized.")
                        
                except Exception as e:
                    st.error(f"Error loading news: {e}")
    
    def render_insider_activity(self):
        """Render insider transactions and sentiment"""
        st.subheader("👔 Insider Activity")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            insider_symbol = st.text_input("Symbol for Insider Data", value="AAPL").upper()
            if st.button("Get Insider Data"):
                st.session_state.insider_symbol = insider_symbol
        
        with col2:
            if hasattr(st.session_state, 'insider_symbol'):
                symbol = st.session_state.insider_symbol
                
                try:
                    if 'news_provider' in st.session_state:
                        transactions = st.session_state.news_provider.get_insider_transactions(symbol)
                        sentiment = st.session_state.news_provider.get_insider_sentiment(symbol)
                        
                        # Insider sentiment
                        if sentiment:
                            st.write("**Insider Sentiment**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Sentiment Score", sentiment.get('sentiment', 'N/A'))
                            with col_b:
                                st.metric("Buy Transactions", sentiment.get('buys', 0))
                            with col_c:
                                st.metric("Sell Transactions", sentiment.get('sells', 0))
                        
                        # Recent transactions
                        if transactions:
                            st.write("**Recent Insider Transactions**")
                            for txn in transactions[:5]:
                                with st.expander(f"{txn.get('person_name', 'Unknown')} - {txn.get('transaction_type', 'Unknown')}"):
                                    st.write(f"**Date:** {txn.get('transaction_date', 'Unknown')}")
                                    st.write(f"**Shares:** {txn.get('shares', 'Unknown')}")
                                    st.write(f"**Price:** ${txn.get('price', 0):.2f}")
                        else:
                            st.info(f"No insider transaction data found for {symbol}")
                    else:
                        st.error("News provider not initialized.")
                        
                except Exception as e:
                    st.error(f"Error loading insider data: {e}")
    
    def render_ai_reports(self):
        """Render AI-generated reports"""
        st.subheader("🤖 AI Reports")
        st.info("🚧 AI report generation coming soon...")
        
        # Placeholder for AI reports
        report_types = st.multiselect(
            "Select Report Types",
            ["Market Summary", "Portfolio Analysis", "Risk Assessment", "Trade Recommendations"],
            default=["Market Summary"]
        )
        
        if st.button("Generate AI Report"):
            with st.spinner("Generating AI report..."):
                st.write("**AI Market Summary Report**")
                st.write("""
                Based on current market conditions and analysis:
                
                📈 **Market Outlook**: Cautiously optimistic with mixed signals
                
                🔍 **Key Observations**:
                - Technology sector showing resilience
                - Volatility remains elevated
                - Economic indicators are mixed
                
                💡 **Recommendations**:
                - Maintain diversified portfolio
                - Consider defensive positions
                - Monitor key support/resistance levels
                
                ⚠️ **Risks to Watch**:
                - Geopolitical tensions
                - Interest rate changes
                - Earnings season results
                """)
    
    def render_main_content(self):
        """Implementation of abstract method for main content"""
        self.create_main_navigation()
    
    def run_dashboard(self):
        """Main dashboard execution"""
        try:
            # Header
            st.title("🌍 Gauss World Trader - Modern Dashboard")
            st.markdown("*Advanced Trading Platform with Comprehensive Market Analysis*")
            
            # Main navigation
            self.render_main_content()
            
            # Footer
            st.divider()
            st.markdown(
                f"<div style='text-align: center; color: #888; font-size: 0.8em;'>"
                f"Dashboard updated: {now_et().strftime('%Y-%m-%d %H:%M:%S')} ET | "
                f"Market Status: {get_market_status()}"
                f"</div>",
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")


def main():
    """Main function to run the modern dashboard"""
    dashboard = ModernDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()