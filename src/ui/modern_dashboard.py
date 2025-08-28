#!/usr/bin/env python3
"""
Unified Trading Dashboard - Redesigned Structure
Organized navigation tabs: Market Overview, Account Info, Live Analysis, 
Strategy Backtest, Trade & Order, News & Report
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.account.account_manager import AccountManager
from src.account.order_manager import OrderManager
from src.account.position_manager import PositionManager
from src.agent.fundamental_analyzer import FundamentalAnalyzer
from src.analysis import TechnicalAnalysis
from src.data import AlpacaDataProvider, NewsDataProvider
from src.data import FREDProvider
from src.strategy.strategy_selector import get_strategy_selector
from src.ui.core_dashboard import BaseDashboard, UIComponents
from src.utils.timezone_utils import get_market_status, now_et
from src.utils.watchlist_manager import WatchlistManager

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
                st.session_state.fred_provider = FREDProvider()
                st.session_state.modern_initialized = True

            except Exception as e:
                logger.error(f"Error initializing modern modules: {e}")
                st.error("Error initializing trading modules. Please check API configuration.")

    def create_main_navigation(self):
        """Create main navigation tabs on the left sidebar"""
        with st.sidebar:
            # Display logo above navigation
            logo_path = Path(__file__).parent.parent / "pic" / "logo.png"
            if logo_path.exists():
                st.image(str(logo_path), width=150)

            st.header("Navigation")

            # Create radio buttons for main navigation
            selected_tab = st.radio(
                "Choose a section:",
                [
                    "📊 Market Overview",
                    "💼 Account Info",
                    "🔍 Live Analysis",
                    "👁️ Watchlist",
                    "📈 Strategy Backtest",
                    "⚡ Trade & Order",
                    "📰 News & Report"
                ],
                key="main_navigation"
            )

            # Add market status and portfolio quick view below navigation
            st.divider()
            self.render_market_status_sidebar()
            self.render_portfolio_quick_view()

        # Render selected tab content
        if selected_tab == "📊 Market Overview":
            self.render_market_overview_tab()
        elif selected_tab == "💼 Account Info":
            self.render_account_info_tab()
        elif selected_tab == "🔍 Live Analysis":
            self.render_live_analysis_tab()
        elif selected_tab == "👁️ Watchlist":
            self.render_watchlist_tab()
        elif selected_tab == "📈 Strategy Backtest":
            self.render_strategy_backtest_tab()
        elif selected_tab == "⚡ Trade & Order":
            self.render_trade_order_tab()
        elif selected_tab == "📰 News & Report":
            self.render_news_report_tab()

    def render_market_status_sidebar(self):
        """Render market status in sidebar"""
        st.subheader("🏛️ Market Status")
        local_time, et_time = datetime.now(), now_et()
        market_status = get_market_status()
        status_color = "🟢" if market_status == 'open' else "🔴"
        
        st.write(f"📅 **Local Time:** {local_time.strftime('%H:%M:%S')}")
        st.write(f"📅 **ET Time:** {et_time.strftime('%H:%M:%S')}")
        st.write(f"**Status:** {status_color} {market_status.title()}")
        # st.write(f"**Next Change:** {'Market Close' if market_status == 'open' else 'Market Open'}")
        st.divider()
    
    def render_portfolio_quick_view(self):
        """Render portfolio quick view in sidebar"""
        st.subheader("📊 Quick View")

        try:
            # Get account info
            account_info, error = self.get_account_info()
            if account_info:
                portfolio_value = float(account_info.get('portfolio_value', 0))
                equity = float(account_info.get('equity', 0))
                last_equity = float(account_info.get('last_equity', equity))
                day_pl = equity - last_equity
                day_pl_pct = (day_pl / last_equity * 100) if last_equity > 0 else 0

                # Portfolio value and today's change
                st.metric(
                    "Portfolio Value",
                    f"${portfolio_value:,.2f}",
                    f"${day_pl:+,.2f} ({day_pl_pct:+.2f}%)"
                )

                # Get positions for winners/losers analysis
                if hasattr(st.session_state, 'position_manager'):
                    positions = st.session_state.position_manager.get_all_positions()
                    if positions and not any('error' in pos for pos in positions):
                        winners = []
                        losers = []
                        total_pl = 0

                        for pos in positions:
                            try:
                                symbol = pos.get('symbol', 'N/A')
                                unrealized_pl = float(pos.get('unrealized_pl', 0))
                                unrealized_plpc = float(pos.get('unrealized_plpc', 0))
                                total_pl += unrealized_pl

                                if unrealized_pl > 0:
                                    winners.append((symbol, unrealized_pl, unrealized_plpc))
                                elif unrealized_pl < 0:
                                    losers.append((symbol, unrealized_pl, unrealized_plpc))
                            except (ValueError, TypeError):
                                continue

                        # Sort winners and losers
                        winners.sort(key=lambda x: x[1], reverse=True)
                        losers.sort(key=lambda x: x[1])

                        # Overall position P&L
                        total_pl_pct = (total_pl / portfolio_value * 100) if portfolio_value > 0 else 0
                        st.metric(
                            "Overall P&L",
                            f"${total_pl:+,.2f}",
                            f"{total_pl_pct:+.2f}%"
                        )

                        # Top winner
                        if winners:
                            top_winner = winners[0]
                            st.write("🏆 **Top Winner**")
                            st.write(f"{top_winner[0]}: +${top_winner[1]:,.2f} ({top_winner[2]:+.2f}%)")

                        # Top loser
                        if losers:
                            top_loser = losers[0]
                            st.write("📉 **Top Loser**")
                            st.write(f"{top_loser[0]}: ${top_loser[1]:,.2f} ({top_loser[2]:+.2f}%)")

                        # Summary counts
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Winners", len(winners))
                        with col2:
                            st.metric("Losers", len(losers))
                    else:
                        st.info("No positions found")
                else:
                    st.info("Position data unavailable")
            else:
                st.error(f"Unable to load portfolio data: {error}")

        except Exception as e:
            st.error(f"Error loading quick view: {e}")
            logger.error(f"Portfolio quick view error: {e}")

    def render_market_overview_tab(self):
        """Market Overview: Index, VIX, Market Sentiment, Sector Performance, Economic Calendar, Crypto"""
        st.header("📊 Market Overview")

        st.divider()

        # Market indices with real data
        self.render_market_indices()

        st.divider()

        # Sub-tabs for different market data (Indices removed and integrated above)
        market_tabs = st.tabs(["📊 VXX & Sentiment", "🏢 Sectors", "📅 Economic Calendar", "₿ Cryptocurrency"])

        with market_tabs[0]:
            self.render_vix_sentiment()

        with market_tabs[1]:
            self.render_sector_performance()

        with market_tabs[2]:
            self.render_economic_calendar()

        with market_tabs[3]:
            self.render_cryptocurrency_data()

    def render_market_indices(self):
        """Render real market indices data"""
        col1, col2, col3, col4 = st.columns(4)
        indices = {'SPY': 'S&P 500', 'QQQ': 'NASDAQ', 'DIA': 'DOW', 'VXX': 'VXX'}

        try:
            provider = AlpacaDataProvider()
            for i, (symbol, name) in enumerate(indices.items()):
                with [col1, col2, col3, col4][i]:
                    try:
                        quote = provider.get_latest_quote(symbol)
                        if 'error' not in quote:
                            current_price = float(quote.get('bid_price', quote.get('ask_price', 0)))
                            historical_data = provider.get_bars(symbol, '1Day', start=now_et() - timedelta(days=5))
                            
                            if not historical_data.empty:
                                prev_close = float(historical_data['close'].iloc[-2 if len(historical_data) >= 2 else -1])
                                change = current_price - prev_close
                                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                            else:
                                change, change_pct = 0, 0

                            price_fmt = f"{current_price:.2f}" if symbol == 'VXX' else f"{current_price:,.2f}"
                            st.metric(name, price_fmt, f"{change:+.2f} ({change_pct:+.2f}%)")
                        else:
                            st.metric(name, "N/A", "Data unavailable")
                    except Exception as e:
                        st.metric(name, "N/A", f"Error: {str(e)[:20]}...")
        except Exception as e:
            st.error(f"Error loading market indices: {e}")
            for i, name in enumerate(['S&P 500', 'NASDAQ', 'DOW', 'VXX']):
                with [col1, col2, col3, col4][i]:
                    st.metric(name, "N/A", "Data unavailable")

    def render_vix_sentiment(self):
        """Render VXX and market sentiment indicators with real data"""
        st.subheader("📊 VXX: iPath S&P 500 VIX ST Futures ETN")
        col1, col2 = st.columns(2)

        try:
            provider = AlpacaDataProvider()
            vxx_data = provider.get_bars('VXX', '1Day', start=now_et() - timedelta(days=45))

            if not vxx_data.empty:
                current_vxx = float(vxx_data['close'].iloc[-1])
                vxx_30_avg = float(vxx_data['close'].mean())
                
                # Calculate Fear & Greed based on VXX levels
                fear_greed = (max(0, 30 - (current_vxx - 40) * 1.5) if current_vxx > 40 else
                             30 + (40 - current_vxx) * 2.67 if current_vxx > 25 else
                             70 + min(30, (25 - current_vxx) * 2))
                fear_greed = max(0, min(100, fear_greed))

                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=fear_greed,
                        title={'text': "Fear & Greed Index (VXX-based)"},
                        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 20], 'color': "red"}, {'range': [20, 40], 'color': "orange"},
                                        {'range': [40, 60], 'color': "yellow"}, {'range': [60, 80], 'color': "lightgreen"},
                                        {'range': [80, 100], 'color': "green"}],
                               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("**Market Sentiment Indicators**")
                    sentiment_color, sentiment_label = ("🔴", "Fearful") if current_vxx > 40 else ("🟡", "Neutral") if current_vxx > 25 else ("🟢", "Greedy")
                    vxx_trend = "Rising" if current_vxx > vxx_30_avg else "Falling"
                    
                    st.metric("Current VXX", f"${current_vxx:.2f}", f"30-day avg: ${vxx_30_avg:.2f}")
                    st.write(f"**Market Mood:** {sentiment_color} {sentiment_label}")
                    st.write(f"**VXX Trend:** {vxx_trend}")
                    st.write("**VXX Levels:** • Below $25: Low volatility (Complacency) • $25-40: Normal • Above $40: High volatility (Fear)")
                    
                    vxx_change = current_vxx - vxx_30_avg
                    vxx_change_pct = (vxx_change / vxx_30_avg * 100) if vxx_30_avg != 0 else 0
                    st.metric("VXX vs 30d Avg", f"{vxx_change_pct:+.1f}%", f"${vxx_change:+.2f}")
                    st.write(f"**VXX Volatility (30d):** ${float(vxx_data['close'].std()):.2f}")
            else:
                st.error("Unable to load VXX data")
        except Exception as e:
            st.error(f"Error loading VXX/sentiment data: {e}")
            st.info("Unable to load real-time VXX data. Please check API configuration.")

    def render_sector_performance(self):
        """Render sector performance analysis with real data"""
        st.subheader("🏢 Sector Performance")

        try:
            provider = AlpacaDataProvider()
            sector_etfs = {'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financial', 'XLE': 'Energy', 
                          'XLY': 'Consumer Discretionary', 'XLI': 'Industrial', 'XLB': 'Materials', 
                          'XLRE': 'Real Estate', 'XLU': 'Utilities'}
            sector_data = []

            for etf_symbol, sector_name in sector_etfs.items():
                try:
                    data = provider.get_bars(etf_symbol, '1Day', start=now_et() - timedelta(days=5))
                    if not data.empty and len(data) >= 2:
                        current_price = float(data['close'].iloc[-1])
                        start_price = float(data['close'].iloc[0])
                        performance = ((current_price - start_price) / start_price) * 100
                        sector_data.append({'sector': sector_name, 'performance': performance, 
                                          'symbol': etf_symbol, 'current_price': current_price})
                except:
                    continue

            if sector_data:
                sector_data.sort(key=lambda x: x['performance'], reverse=True)
                sectors = [item['sector'] for item in sector_data]
                performance = [item['performance'] for item in sector_data]

                import plotly.express as px
                fig = px.bar(x=sectors, y=performance, title="Sector Performance - Day (% Change)",
                           color=performance, color_continuous_scale="RdYlGn", 
                           text=[f"{p:+.2f}%" for p in performance])
                fig.update_layout(template="plotly_white", height=400, 
                                xaxis={'categoryorder': 'total descending'}, yaxis_title="Performance (%)")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                st.write("**📊 Sector Performance in Details**")
                df = pd.DataFrame([{
                    'Sector': item['sector'], 'ETF Symbol': item['symbol'],
                    'Current Price': f"${item['current_price']:.2f}" if item['current_price'] > 0 else "N/A",
                    'Day Performance': f"{item['performance']:+.2f}%"
                } for item in sector_data])
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Unable to load any sector performance data")
        except Exception as e:
            st.error(f"Error loading sector performance: {e}")
            st.info("Unable to load real-time sector data. Please check API configuration.")

    def render_economic_calendar(self):
        """Render economic calendar with real FRED data"""
        st.subheader("📅 Economic Calendar")
        
        try:
            if 'fred_provider' in st.session_state and st.session_state.fred_provider.client:
                fred = st.session_state.fred_provider
                indicators = {'UNRATE': 'Unemployment Rate', 'CPIAUCSL': 'CPI', 'FEDFUNDS': 'Fed Funds Rate'}
                
                col1, col2, col3 = st.columns(3)
                for i, (series_id, name) in enumerate(indicators.items()):
                    try:
                        data = fred.get_series_data(series_id, start_date='2024-01-01')
                        if not data.empty:
                            latest_value = data.iloc[-1, 0]
                            with [col1, col2, col3][i]:
                                st.metric(name, f"{latest_value:.2f}{'%' if 'Rate' in name else ''}")
                    except:
                        with [col1, col2, col3][i]:
                            st.metric(name, "N/A")
            else:
                st.warning("FRED API not configured")
                mock_events = [
                    {"Date": now_et().strftime('%Y-%m-%d'), "Event": "CPI", "Previous": "3.1%", "Forecast": "3.2%"},
                    {"Date": (now_et() + timedelta(days=2)).strftime('%Y-%m-%d'), "Event": "Retail Sales", "Previous": "0.3%", "Forecast": "0.4%"}
                ]
                st.dataframe(pd.DataFrame(mock_events), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading economic calendar: {e}")

    def render_cryptocurrency_data(self):
        """Render cryptocurrency information with comprehensive crypto data"""
        st.subheader("₿ Cryptocurrency")

        try:
            provider = AlpacaDataProvider()
            crypto_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
            crypto_names = {'BTC/USD': 'Bitcoin', 'ETH/USD': 'Ethereum', 'LTC/USD': 'Litecoin', 'BCH/USD': 'Bitcoin Cash'}
            
            cols = st.columns(len(crypto_symbols))
            for i, symbol in enumerate(crypto_symbols):
                with cols[i]:
                    try:
                        quote = provider.get_crypto_latest_quote(symbol)
                        if 'error' not in quote:
                            bid_price = float(quote.get('bid_price', 0))
                            ask_price = float(quote.get('ask_price', 0))
                            current_price = (bid_price + ask_price) / 2 if bid_price and ask_price else bid_price or ask_price

                            start_date = now_et() - timedelta(days=5)  # Get more days to ensure we have previous close
                            hist_data = provider.get_bars(symbol, '1Day', start=start_date)

                            if not hist_data.empty and len(hist_data) > 1:
                                prev_close = float(hist_data['close'].iloc[-2])
                                change = current_price - prev_close
                                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                                st.metric(crypto_names.get(symbol, symbol), f"${current_price:,.2f}", 
                                         f"${change:+,.2f} ({change_pct:+.2f}%)")
                            else:
                                st.metric(crypto_names.get(symbol, symbol), f"${current_price:,.2f}")
                        else:
                            st.metric(crypto_names.get(symbol, symbol), "N/A", "Error loading")
                    except:
                        st.metric(crypto_names.get(symbol, symbol), "N/A", "Error")

            st.divider()
            st.write("**📊 Bitcoin Detailed Analysis**")
            
            btc_quote = provider.get_crypto_latest_quote('BTC/USD')
            if 'error' not in btc_quote:
                col1, col2, col3, col4 = st.columns(4)
                bid_price = float(btc_quote.get('bid_price', 0))
                ask_price = float(btc_quote.get('ask_price', 0))
                spread = ask_price - bid_price if ask_price and bid_price else 0
                
                with col1: st.metric("Bid Price", f"${bid_price:,.2f}")
                with col2: st.metric("Ask Price", f"${ask_price:,.2f}")
                with col3: st.metric("Bid-Ask Spread", f"${spread:.2f}")
                with col4: st.metric("Last Updated", btc_quote.get('timestamp', 'N/A').strftime("%H:%M:%S") 
                                    if 'timestamp' in btc_quote else "N/A")

                st.write("**📈 Bitcoin Price Chart (30 Days)**")
                start_date = now_et() - timedelta(days=30)  # Get more days to ensure we have previous close
                btc_data = provider.get_bars('BTC/USD', '1Day', start=start_date)
                
                if not btc_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=btc_data.index, open=btc_data['open'], high=btc_data['high'],
                                               low=btc_data['low'], close=btc_data['close'], name='BTC/USD'))
                    sma_20 = btc_data['close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(x=btc_data.index, y=sma_20, mode='lines', name='20-day SMA',
                                           line=dict(color='orange', width=1)))
                    fig.update_layout(title="Bitcoin (BTC/USD) - 30 Day Chart", yaxis_title="Price (USD)",
                                    xaxis_title="Date", height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                    current_price = float(btc_data['close'].iloc[-1])
                    high_30d = float(btc_data['high'].max())
                    low_30d = float(btc_data['low'].min())
                    volatility = btc_data['close'].pct_change().std() * np.sqrt(365) * 100
                    range_pos = ((current_price - low_30d) / (high_30d - low_30d)) * 100 if (high_30d - low_30d) > 0 else 0

                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("30D High", f"${high_30d:,.2f}")
                    with col2: st.metric("30D Low", f"${low_30d:,.2f}")
                    with col3: st.metric("30D Range Position", f"{range_pos:.1f}%")
                    with col4: st.metric("Annualized Volatility", f"{volatility:.1f}%")
                else:
                    st.error("Unable to load Bitcoin historical data")
            else:
                st.error(f"Error loading Bitcoin data: {btc_quote.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error loading cryptocurrency data: {e}")

    def render_account_info_tab(self):
        """Account Info: Account, Positions, Portfolio, Performance, Risk Management"""
        st.header("💼 Account Information")

        account_tabs = st.tabs(["📊 Account", "📈 Positions", "💰 Portfolio", "⚙️ Configuration"])

        with account_tabs[0]:
            self.render_account_overview()

        with account_tabs[1]:
            self.render_positions_view()

        with account_tabs[2]:
            self.render_portfolio_analytics()

        with account_tabs[3]:
            self.render_risk_configuration()

    def render_account_overview(self):
        """Render account overview"""
        st.subheader("📊 Account Overview")
        account_info, error = self.get_account_info()
        
        if account_info:
            col1, col2, col3, col4 = st.columns(4)
            equity = float(account_info.get('equity', 0))
            last_equity = float(account_info.get('last_equity', equity))
            day_pl = equity - last_equity
            
            with col1: st.metric("Account Value", f"${float(account_info.get('portfolio_value', 0)):,.2f}")
            with col2: st.metric("Buying Power", f"${float(account_info.get('buying_power', 0)):,.2f}")
            with col3: st.metric("Cash", f"${float(account_info.get('cash', 0)):,.2f}")
            with col4: st.metric("Day P&L", f"${day_pl:,.2f}", delta=f"{day_pl:,.2f}")
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
        """Render portfolio analytics with real data"""
        st.subheader("💰 Portfolio Analytics")

        self.render_asset_allocation()
        self.render_portfolio_performance()

    def render_asset_allocation(self):
        """Render asset allocation analysis"""
        try:
            account_info, _ = self.get_account_info()
            positions = st.session_state.position_manager.get_all_positions() if 'position_manager' in st.session_state else []
            
            if not account_info or not positions or any('error' in pos for pos in positions):
                st.info("Portfolio data unavailable")
                return

            portfolio_value = float(account_info.get('portfolio_value', 0))
            cash = float(account_info.get('cash', 0))

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Asset Allocation**")
                allocation_data = {'Cash': cash}
                
                for pos in positions:
                    try:
                        symbol = pos.get('symbol', 'Unknown')
                        market_value = abs(float(pos.get('market_value', 0)))
                        if symbol in allocation_data:
                            allocation_data[symbol] += market_value
                        else:
                            allocation_data[symbol] = market_value
                    except (ValueError, TypeError):
                        continue

                if sum(allocation_data.values()) > 0:
                    fig = go.Figure(data=[go.Pie(labels=list(allocation_data.keys()), 
                                                values=list(allocation_data.values()))])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Portfolio Metrics**")
                
                total_pl = sum(float(pos.get('unrealized_pl', 0)) for pos in positions 
                              if pos.get('unrealized_pl'))
                total_pl_pct = (total_pl / portfolio_value * 100) if portfolio_value > 0 else 0
                
                winners = [pos for pos in positions if float(pos.get('unrealized_pl', 0)) > 0]
                losers = [pos for pos in positions if float(pos.get('unrealized_pl', 0)) < 0]
                win_rate = (len(winners) / len(positions) * 100) if positions else 0
                
                st.metric("Total P&L", f"${total_pl:+,.2f}", f"{total_pl_pct:+.2f}%")
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Active Positions", len(positions))

        except Exception as e:
            st.error(f"Error loading asset allocation: {e}")

    def render_portfolio_performance(self):
        """Render performance metrics using real portfolio history"""
        try:
            account_info, _ = self.get_account_info()
            if not account_info:
                st.info("Performance data unavailable")
                return

            # Use correct account fields
            portfolio_value = float(account_info.get('portfolio_value', 0))
            equity = float(account_info.get('equity', 0))
            last_equity = float(account_info.get('last_equity', equity))
            
            # Calculate daily P&L from equity change
            day_pl = equity - last_equity
            day_pl_pct = (day_pl / last_equity * 100) if last_equity > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Day P&L", f"${day_pl:+,.2f}")
            with col2:
                st.metric("Day Return", f"{day_pl_pct:+.2f}%")
            with col3:
                st.metric("Portfolio Value", f"${portfolio_value:,.2f}")

            # Get portfolio history using data provider
            try:
                
                # Get portfolio history from data provider
                provider = AlpacaDataProvider()
                portfolio_history = provider.get_portfolio_history()
                
                if portfolio_history and 'error' not in portfolio_history:
                    equity_values = portfolio_history.get('equity', [])
                    timestamps = portfolio_history.get('timestamp', [])
                    
                    if equity_values and timestamps:
                        # Filter out leading zeros
                        start_idx = 0
                        for i, val in enumerate(equity_values):
                            if val > 0:
                                start_idx = i
                                break
                        
                        # Use filtered data starting from first non-zero value
                        filtered_equity = equity_values[start_idx:]
                        filtered_timestamps = timestamps[start_idx:]
                        
                        if filtered_equity and filtered_timestamps:
                            # Convert timestamps to datetime if needed
                            if isinstance(filtered_timestamps[0], (int, float)):
                                dates = [datetime.fromtimestamp(ts) for ts in filtered_timestamps]
                            else:
                                dates = filtered_timestamps
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=dates, y=filtered_equity, mode='lines', 
                                name='Portfolio Value', line=dict(color='blue', width=2)
                            ))
                            fig.update_layout(
                                title="Portfolio Performance (30 Days)", 
                                yaxis_title="Value ($)", height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Performance metrics using filtered data
                            if len(filtered_equity) > 1:
                                total_return = ((filtered_equity[-1] - filtered_equity[0]) / filtered_equity[0] * 100)
                                max_value = max(filtered_equity)
                                min_value = min(filtered_equity)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("30 Day Return", f"{total_return:+.2f}%")
                                with col2:
                                    st.metric("30 Day High", f"${max_value:,.2f}")
                                with col3:
                                    st.metric("30 Day Low", f"${min_value:,.2f}")
                        else:
                            st.info("Insufficient data for performance metrics")
                    else:
                        st.info("No non-zero portfolio data available")
                else:
                    st.info("No portfolio history data available")
                    
            except Exception as hist_error:
                logger.error(f"Portfolio history error: {hist_error}")
                st.info("Portfolio history temporarily unavailable")

        except Exception as e:
            st.error(f"Error loading performance: {e}")
            logger.error(f"Portfolio performance error: {e}")


    def render_risk_configuration(self):
        """Render risk management configuration"""
        st.subheader("⚙️ Risk Management Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Position Sizing**")
            max_position = st.slider("Max Position Size (%)", 1, 20, 10, key="max_position_size")
            max_portfolio_risk = st.slider("Max Portfolio Risk (%)", 1, 10, 2, key="max_portfolio_risk")

        with col2:
            st.write("**Stop Loss Settings**")
            default_stop_loss = st.slider("Default Stop Loss (%)", 1, 20, 5, key="default_stop_loss")
            trailing_stop = st.checkbox("Enable Trailing Stop")

        if st.button("Save Risk Settings"):
            st.success("Risk settings saved successfully!")

    def render_live_analysis_tab(self):
        """Live Analysis: Symbol Analysis"""
        st.header("🔍 Live Analysis (Symbol)")

        # Only symbol analysis remains in this tab
        self.render_symbol_analysis()

    def render_symbol_analysis(self):
        """Render symbol analysis"""
        # st.subheader("📊 Symbol Analysis")

        col1, col2 = st.columns([1, 3])

        with col1:
            symbol = st.text_input("Enter Symbol", value="AAPL", key="analysis_symbol").upper()
            days = st.selectbox("Analysis Period", [30, 60, 90, 180, 365], index=2, key="analysis_period")

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
        if 'watchlist_manager' in st.session_state:
            wm = st.session_state.watchlist_manager

            # Add symbol section
            st.subheader("➕ Add Symbol")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                new_symbol = st.text_input("Enter Symbol to Add", key="add_symbol").upper()
            with col2:
                if st.button("Add Symbol", type="primary") and new_symbol:
                    if new_symbol not in wm.get_watchlist():
                        wm.add_symbol(new_symbol)
                        st.success(f"✅ Added {new_symbol} to watchlist")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ {new_symbol} is already in watchlist")
            with col3:
                if st.button("🔄 Refresh Data", help="Refresh all watchlist data"):
                    st.rerun()

            st.divider()

            # Display watchlist
            watchlist = wm.get_watchlist()
            if watchlist:
                st.subheader(f"👁️ Watchlist ({len(watchlist)} symbols)")
                
                # Delete symbol section
                col1, col2 = st.columns([2, 1])
                with col1:
                    symbol_to_delete = st.selectbox("Select Symbol to Remove", 
                                                   options=[""] + sorted(watchlist))
                with col2:
                    if st.button("🗑️ Remove", type="secondary") and symbol_to_delete:
                        wm.remove_symbol(symbol_to_delete)
                        st.success(f"🗑️ Removed {symbol_to_delete} from watchlist")
                        st.rerun()

                st.divider()

                # Watchlist data table
                watchlist_data = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, symbol in enumerate(watchlist):
                    status_text.text(f"Loading data for {symbol}...")
                    progress_bar.progress((i + 1) / len(watchlist))
                    
                    data, error = self.load_market_data(symbol, 2)
                    if data is not None and not data.empty:
                        current = data['close'].iloc[-1]
                        prev = data['close'].iloc[-2] if len(data) > 1 else current
                        change = current - prev
                        change_pct = (change / prev * 100) if prev != 0 else 0

                        # Add color coding for gains/losses
                        change_color = "🟢" if change >= 0 else "🔴"
                        watchlist_data.append({
                            'Symbol': symbol,
                            'Current Price': f"${current:.2f}",
                            'Change ($)': f"${change:+.2f}",
                            'Change (%)': f"{change_pct:+.2f}%",
                            'Status': f"{change_color}"
                        })
                    else:
                        watchlist_data.append({
                            'Symbol': symbol,
                            'Current Price': "N/A",
                            'Change ($)': "N/A", 
                            'Change (%)': "N/A",
                            'Status': "❌"
                        })

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                if watchlist_data:
                    df = pd.DataFrame(watchlist_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary stats
                    valid_data = [item for item in watchlist_data if item['Current Price'] != "N/A"]
                    if valid_data:
                        gains = sum(1 for item in valid_data if "+" in item['Change (%)'])
                        losses = sum(1 for item in valid_data if "-" in item['Change (%)'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Symbols", len(watchlist))
                        with col2:
                            st.metric("Gainers", gains, delta=f"+{gains}")
                        with col3:
                            st.metric("Losers", losses, delta=f"-{losses}")
                        with col4:
                            st.metric("Data Available", len(valid_data))
            else:
                st.info("📭 Watchlist is empty. Add some symbols to monitor.")
        else:
            st.error("❌ Watchlist manager not initialized.")

    def render_watchlist_tab(self):
        """Watchlist: Watchlist Management"""
        st.header("👁️ Watchlist")
        self.render_watchlist_management()

    def render_strategy_backtest_tab(self):
        """Strategy Backtest: Quick Backtest, Strategy Comparison"""
        st.header("📈 Strategy Backtesting")

        backtest_tabs = st.tabs(["⚡ Quick Backtest", "🔀 Strategy Comparison"])

        with backtest_tabs[0]:
            self.render_quick_backtest()

        with backtest_tabs[1]:
            self.render_strategy_comparison()

    def get_default_symbols(self):
        """Get default symbols from watchlist and current positions"""
        symbols = set()
        
        # Add symbols from watchlist
        try:
            if 'watchlist_manager' in st.session_state:
                watchlist = st.session_state.watchlist_manager.get_watchlist()
                symbols.update(watchlist)
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
        
        # Add symbols from current positions
        try:
            if 'position_manager' in st.session_state:
                positions = st.session_state.position_manager.get_all_positions()
                if positions and not any('error' in pos for pos in positions):
                    for pos in positions:
                        symbol = pos.get('symbol')
                        if symbol:
                            symbols.add(symbol)
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
        
        return list(symbols)

    def render_quick_backtest(self):
        """Render quick backtesting interface"""
        st.subheader("⚡ Quick Backtest")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Get default symbols from watchlist and current positions
            default_symbols = self.get_default_symbols()
            default_symbols_text = '\n'.join(default_symbols) if default_symbols else "AAPL\nMSFT\nGOOGL"
            
            symbols = st.text_area("Symbols (one per line)", default_symbols_text).strip().split('\n')
            symbols = [s.strip().upper() for s in symbols if s.strip()]

            days_back = st.slider("Backtest Period (days)", 30, 365, 90, key="backtest_period")
            initial_cash = st.number_input("Initial Cash", value=100000, step=10000, key="backtest_initial_cash")

            # Get all available strategies from strategy_selector
            display_strategies = []
            strategy_mapping = {}
            if 'strategy_selector' in st.session_state:
                strategy_names = st.session_state.strategy_selector.list_strategies()
                for name in strategy_names:
                    # Create display name
                    display_name = name.replace('_', ' ').title()
                    display_strategies.append(display_name)
                    strategy_mapping[display_name] = name
            
            # Add fallback strategies if none available
            if not display_strategies:
                display_strategies = ["Momentum", "Mean Reversion", "Trend Following"]
            
            strategy_display = st.selectbox("Strategy", display_strategies, key="backtest_strategy")
            # Pass the display name to core_dashboard (it will handle the mapping)
            strategy_type = strategy_display

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

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("**Configuration**")
            
            # Symbol selection
            default_symbols = self.get_default_symbols()
            if default_symbols:
                default_symbol = default_symbols[0]
            else:
                default_symbol = "AAPL"
            
            symbol = st.selectbox("Select Symbol", 
                                options=default_symbols + ["AAPL", "MSFT", "GOOGL", "TSLA"] if default_symbols 
                                else ["AAPL", "MSFT", "GOOGL", "TSLA"],
                                index=0 if default_symbols else 0)
            
            # Strategy pool management
            st.write("**Strategy Pool**")
            
            # Get all available strategies from strategy_selector
            available_strategies = []
            if 'strategy_selector' in st.session_state:
                strategy_names = st.session_state.strategy_selector.list_strategies()
                for name in strategy_names:
                    # Create display name
                    display_name = name.replace('_', ' ').title()
                    available_strategies.append(display_name)
            
            # Add fallback strategies if none available
            if not available_strategies:
                available_strategies = ["Momentum", "Mean Reversion", "Trend Following"]
            
            # Initialize selected strategies in session state
            if 'comparison_strategies' not in st.session_state:
                st.session_state.comparison_strategies = available_strategies[:3]  # First 3 by default
            
            # Display current strategy pool
            for strategy in st.session_state.comparison_strategies:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"• {strategy}")
                with col_b:
                    if st.button("❌", key=f"remove_{strategy}"):
                        st.session_state.comparison_strategies.remove(strategy)
                        st.rerun()
            
            # Add strategy dropdown
            remaining_strategies = [s for s in available_strategies 
                                 if s not in st.session_state.comparison_strategies]
            if remaining_strategies:
                add_strategy = st.selectbox("Add Strategy", [""] + remaining_strategies)
                if st.button("➕ Add Strategy") and add_strategy:
                    st.session_state.comparison_strategies.append(add_strategy)
                    st.rerun()
            
            st.divider()
            
            # Backtest parameters
            st.write("**Backtest Parameters**")
            days_back = st.slider("Period (days)", 30, 365, 90, key="comparison_period")
            initial_cash = st.number_input("Initial Cash", value=100000, step=10000, key="comparison_initial_cash")
            
            # Run comparison button
            if st.button("🚀 Run Strategy Comparison", type="primary"):
                if len(st.session_state.comparison_strategies) >= 2:
                    st.session_state.run_comparison = True
                    st.session_state.comparison_params = {
                        'symbol': symbol,
                        'strategies': st.session_state.comparison_strategies,
                        'days_back': days_back,
                        'initial_cash': initial_cash
                    }
                else:
                    st.error("Please select at least 2 strategies for comparison")

        with col2:
            if hasattr(st.session_state, 'run_comparison') and st.session_state.run_comparison:
                self.render_strategy_comparison_results()
            else:
                st.info("Configure strategies and click 'Run Strategy Comparison' to see results")

    def render_strategy_comparison_results(self):
        """Render strategy comparison results"""
        st.write("**📈 Strategy Comparison Results**")
        
        params = st.session_state.comparison_params
        symbol = params['symbol']
        strategies = params['strategies']
        days_back = params['days_back']
        initial_cash = params['initial_cash']
        
        with st.spinner(f"Running comparison for {symbol} with {len(strategies)} strategies..."):
            comparison_results = []
            
            for strategy_name in strategies:
                try:
                    # Run backtest for each strategy
                    results, error = self.run_backtest([symbol], days_back, initial_cash, strategy_name)
                    
                    if results and not error:
                        comparison_results.append({
                            'Strategy': strategy_name,
                            'Total Return': f"{results.get('total_return_percentage', 0):.2f}%",
                            'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.2f}",
                            'Max Drawdown': f"{results.get('max_drawdown_percentage', 0):.2f}%",
                            'Win Rate': f"{results.get('win_rate', 0):.1f}%",
                            'Total Trades': results.get('total_trades', 0),
                            'Final Value': f"${results.get('final_value', initial_cash):,.2f}",
                            'Volatility': f"{results.get('volatility', 0):.2f}%"
                        })
                    else:
                        comparison_results.append({
                            'Strategy': strategy_name,
                            'Total Return': 'Error',
                            'Sharpe Ratio': 'N/A',
                            'Max Drawdown': 'N/A',
                            'Win Rate': 'N/A',
                            'Total Trades': 'N/A',
                            'Final Value': 'N/A',
                            'Volatility': 'N/A'
                        })
                
                except Exception as e:
                    logger.error(f"Error running backtest for {strategy_name}: {e}")
                    comparison_results.append({
                        'Strategy': strategy_name,
                        'Total Return': f'Error: {str(e)[:20]}...',
                        'Sharpe Ratio': 'N/A',
                        'Max Drawdown': 'N/A',
                        'Win Rate': 'N/A',
                        'Total Trades': 'N/A',
                        'Final Value': 'N/A',
                        'Volatility': 'N/A'
                    })
        
        if comparison_results:
            df = pd.DataFrame(comparison_results)
            st.dataframe(df, use_container_width=True)
            
            # Performance metrics chart
            valid_results = [r for r in comparison_results if r['Total Return'] != 'Error' and 'Error:' not in str(r['Total Return'])]
            if valid_results:
                st.write("**📊 Performance Comparison Chart**")
                
                strategies_names = [r['Strategy'] for r in valid_results]
                returns = [float(r['Total Return'].rstrip('%')) for r in valid_results]
                sharpe_ratios = [float(r['Sharpe Ratio']) for r in valid_results if r['Sharpe Ratio'] != 'N/A']
                
                if returns:
                    import plotly.express as px
                    fig = px.bar(x=strategies_names, y=returns, title=f"Total Return Comparison - {symbol}",
                               color=returns, color_continuous_scale="RdYlGn",
                               text=[f"{r:.1f}%" for r in returns])
                    fig.update_layout(yaxis_title="Total Return (%)", height=400)
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                if len(valid_results) > 1:
                    best_return_idx = returns.index(max(returns))
                    best_return_strategy = strategies_names[best_return_idx]
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Best Strategy", best_return_strategy)
                    with col_b:
                        st.metric("Best Return", f"{max(returns):.2f}%")
                    with col_c:
                        avg_return = sum(returns) / len(returns)
                        st.metric("Average Return", f"{avg_return:.2f}%")
        else:
            st.error("No valid results to display")

    def render_trade_order_tab(self):
        """Trade & Order: Quick Trade, Active Orders, Order History"""
        st.header("⚡ Trade & Order Management")

        trade_tabs = st.tabs(["🚀 Quick Trade", "📋 Recent Orders"])

        with trade_tabs[0]:
            self.render_quick_trade()

        with trade_tabs[1]:
            self.render_recent_orders()

    def render_quick_trade(self):
        """Render quick trading interface"""
        st.subheader("🚀 Quick Trade")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Order Details**")
            symbol = st.text_input("Symbol", value="AAPL", key="trade_symbol").upper()
            side = st.selectbox("Side", ["buy", "sell"], key="trade_side")
            quantity = st.number_input("Quantity", min_value=1, value=100, key="trade_quantity")
            order_type = st.selectbox("Order Type", ["market", "limit", "stop"], key="trade_order_type")

            if order_type in ["limit", "stop"]:
                limit_price = st.number_input("Price", value=150.0, step=0.01, key="trade_limit_price")

            time_in_force = st.selectbox("Time in Force", ["day", "gtc", "ioc", "fok"], key="trade_time_in_force")

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

    def render_recent_orders(self):
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
            news_symbol = st.text_input("Symbol for News", value="AAPL", key="news_symbol").upper()
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
            insider_symbol = st.text_input("Symbol for Insider Data", value="AAPL", key="insider_symbol").upper()
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
            st.markdown("<h1 style='text-align: center;'>🌍 Gauss World Trader</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-style: italic;'>Advanced Trading Platform with Comprehensive Market Analysis</p>", unsafe_allow_html=True)

            # Add 4-column header info from BaseDashboard
            self.render_header_info()
            st.divider()

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
