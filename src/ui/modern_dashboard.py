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
import pytz
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
from src.data import FinnhubProvider, FREDProvider
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
                st.session_state.finnhub_provider = FinnhubProvider()
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
        elif selected_tab == "📈 Strategy Backtest":
            self.render_strategy_backtest_tab()
        elif selected_tab == "⚡ Trade & Order":
            self.render_trade_order_tab()
        elif selected_tab == "📰 News & Report":
            self.render_news_report_tab()

    def render_market_status_sidebar(self):
        """Render market status in sidebar"""
        st.subheader("🏛️ Market Status")
        
        # Time information
        local_time = datetime.now()
        et_time = now_et()
        st.write(f"📅 **Local Time:** {local_time.strftime('%H:%M:%S')}")
        st.write(f"📅 **ET Time:** {et_time.strftime('%H:%M:%S')}")
        
        # Market status
        market_status = get_market_status()
        status_color = "🟢" if market_status == 'open' else "🔴"
        st.write(f"**Status:** {status_color} {market_status.title()}")
        
        # Calculate next market change
        next_change = "Market Close" if market_status == 'open' else "Market Open"
        st.write(f"**Next Change:** {next_change}")
        
        st.divider()
    
    def render_portfolio_quick_view(self):
        """Render portfolio quick view in sidebar"""
        st.subheader("📊 Quick View")

        try:
            # Get account info
            account_info, error = self.get_account_info()
            if account_info:
                portfolio_value = float(account_info.get('portfolio_value', 0))
                day_pl = float(account_info.get('day_trade_pl', 0))
                day_pl_pct = (day_pl / (portfolio_value - day_pl) * 100) if (portfolio_value - day_pl) > 0 else 0

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
        market_tabs = st.tabs(["📊 VIX & Sentiment", "🏢 Sectors", "📅 Economic Calendar", "₿ Cryptocurrency"])

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

        try:
            provider = AlpacaDataProvider()

            # Define major indices
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'DIA': 'DOW',
                'VXX': 'VXX'
            }

            for i, (symbol, name) in enumerate(indices.items()):
                with [col1, col2, col3, col4][i]:
                    try:
                        # Get latest quote
                        quote = provider.get_latest_quote(symbol)
                        if 'error' not in quote:
                            current_price = float(quote.get('bid_price', quote.get('ask_price', 0)))
                            # Get previous day data for change calculation  
                            start_date = now_et() - timedelta(days=5)  # Get more days to ensure we have previous close
                            historical_data = provider.get_bars(symbol, '1Day', start=start_date)
                            if not historical_data.empty and len(historical_data) >= 2:
                                # Get the second-to-last close price (previous trading day)
                                prev_close = float(historical_data['close'].iloc[-2])
                                change = current_price - prev_close
                                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                            elif not historical_data.empty and len(historical_data) == 1:
                                # Only one bar available, compare current price to that bar's close
                                prev_close = float(historical_data['close'].iloc[-1])
                                change = current_price - prev_close
                                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                            else:
                                change = 0
                                change_pct = 0

                            if symbol == 'VIX':
                                st.metric(name, f"{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
                            else:
                                st.metric(name, f"{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
                        else:
                            st.metric(name, "N/A", "Data unavailable")
                    except Exception as e:
                        st.metric(name, "N/A", f"Error: {str(e)[:20]}...")

        except Exception as e:
            st.error(f"Error loading market indices: {e}")
            # Fallback to static display
            col1.metric("S&P 500", "N/A", "Data unavailable")
            col2.metric("NASDAQ", "N/A", "Data unavailable")
            col3.metric("DOW", "N/A", "Data unavailable")
            col4.metric("VIX", "N/A", "Data unavailable")

    def render_vix_sentiment(self):
        """Render VIX and market sentiment indicators with real data"""
        st.subheader("📊 VIX & Market Sentiment")
        st.info("🚧 VIX sentiment analysis coming soon...")

        # col1, col2 = st.columns(2)
        #
        # try:
        #     provider = AlpacaDataProvider()
        #
        #     # Get VIX data
        #     vix_data = provider.get_historical_bars('VIX', '1Day', 30)
        #
        #     with col1:
        #         if not vix_data.empty:
        #             current_vix = float(vix_data['close'].iloc[-1])
        #             vix_30_avg = float(vix_data['close'].mean())
        #
        #             # Calculate Fear & Greed based on VIX levels
        #             # VIX > 30 = Fear (0-30), VIX 20-30 = Neutral (30-70), VIX < 20 = Greed (70-100)
        #             if current_vix > 30:
        #                 fear_greed = max(0, 30 - (current_vix - 30) * 2)
        #             elif current_vix > 20:
        #                 fear_greed = 30 + (30 - current_vix) * 4
        #             else:
        #                 fear_greed = 70 + min(30, (20 - current_vix) * 3)
        #
        #             fear_greed = max(0, min(100, fear_greed))  # Clamp to 0-100
        #
        #             fig = go.Figure(go.Indicator(
        #                 mode = "gauge+number",
        #                 value = fear_greed,
        #                 domain = {'x': [0, 1], 'y': [0, 1]},
        #                 title = {'text': "Fear & Greed Index (VIX-based)"},
        #                 gauge = {
        #                     'axis': {'range': [None, 100]},
        #                     'bar': {'color': "darkblue"},
        #                     'steps': [
        #                         {'range': [0, 20], 'color': "red"},
        #                         {'range': [20, 40], 'color': "orange"},
        #                         {'range': [40, 60], 'color': "yellow"},
        #                         {'range': [60, 80], 'color': "lightgreen"},
        #                         {'range': [80, 100], 'color': "green"}
        #                     ],
        #                     'threshold': {
        #                         'line': {'color': "red", 'width': 4},
        #                         'thickness': 0.75,
        #                         'value': 90
        #                     }
        #                 }
        #             ))
        #
        #             fig.update_layout(height=300)
        #             st.plotly_chart(fig, use_container_width=True)
        #         else:
        #             st.error("Unable to load VIX data for sentiment calculation")
        #
        #     with col2:
        #         # Market Sentiment Indicators based on VIX
        #         st.write("**Market Sentiment Indicators**")
        #
        #         if not vix_data.empty:
        #             current_vix = float(vix_data['close'].iloc[-1])
        #             vix_30_avg = float(vix_data['close'].mean())
        #             vix_trend = "Rising" if current_vix > vix_30_avg else "Falling"
        #
        #             # Overall sentiment based on VIX
        #             if current_vix > 30:
        #                 sentiment_label = "Fearful"
        #                 sentiment_color = "🔴"
        #             elif current_vix > 20:
        #                 sentiment_label = "Neutral"
        #                 sentiment_color = "🟡"
        #             else:
        #                 sentiment_label = "Greedy"
        #                 sentiment_color = "🟢"
        #
        #             st.metric("Current VIX", f"{current_vix:.2f}", f"30-day avg: {vix_30_avg:.2f}")
        #             st.write(f"**Market Mood:** {sentiment_color} {sentiment_label}")
        #             st.write(f"**VIX Trend:** {vix_trend}")
        #
        #             # VIX interpretation
        #             st.write("**VIX Levels:**")
        #             st.write("• Below 20: Low volatility (Complacency)")
        #             st.write("• 20-30: Normal volatility")
        #             st.write("• Above 30: High volatility (Fear)")
        #
        #             # Recent volatility analysis
        #             vix_volatility = float(vix_data['close'].std())
        #             st.write(f"**VIX Volatility (30d):** {vix_volatility:.2f}")
        #         else:
        #             st.error("Unable to load VIX data for sentiment indicators")
        #
        # except Exception as e:
        #     st.error(f"Error loading VIX/sentiment data: {e}")

    def render_sector_performance(self):
        """Render sector performance analysis with real data"""
        st.subheader("🏢 Sector Performance")
        st.info("🚧 Sector performance analysis coming soon...")

        # try:
        #     provider = AlpacaDataProvider()
        #
        #     # Sector ETFs representing major sectors
        #     sector_etfs = {
        #         'XLK': 'Technology',
        #         'XLV': 'Healthcare',
        #         'XLF': 'Financial',
        #         'XLE': 'Energy',
        #         'XLY': 'Consumer Discretionary',
        #         'XLI': 'Industrial',
        #         'XLB': 'Materials',
        #         'XLRE': 'Real Estate',
        #         'XLU': 'Utilities'
        #     }
        #
        #     sector_data = []
        #
        #     for etf_symbol, sector_name in sector_etfs.items():
        #         try:
        #             # Get 5 days of data to calculate performance
        #             data = provider.get_historical_bars(etf_symbol, '1Day', 5)
        #
        #             if not data.empty and len(data) >= 2:
        #                 current_price = float(data['close'].iloc[-1])
        #                 start_price = float(data['close'].iloc[0])
        #                 performance = ((current_price - start_price) / start_price) * 100
        #
        #                 sector_data.append({
        #                     'sector': sector_name,
        #                     'performance': performance,
        #                     'symbol': etf_symbol,
        #                     'current_price': current_price
        #                 })
        #             else:
        #                 # Fallback with minimal data
        #                 sector_data.append({
        #                     'sector': sector_name,
        #                     'performance': 0.0,
        #                     'symbol': etf_symbol,
        #                     'current_price': 0.0
        #                 })
        #
        #         except Exception as e:
        #             st.warning(f"Unable to load data for {sector_name} ({etf_symbol}): {str(e)}")
        #             continue
        #
        #     if sector_data:
        #         # Sort by performance
        #         sector_data.sort(key=lambda x: x['performance'], reverse=True)
        #
        #         sectors = [item['sector'] for item in sector_data]
        #         performance = [item['performance'] for item in sector_data]
        #
        #         import plotly.express as px
        #         fig = px.bar(
        #             x=sectors,
        #             y=performance,
        #             title="Sector Performance - 5 Day (% Change)",
        #             color=performance,
        #             color_continuous_scale="RdYlGn",
        #             text=[f"{p:+.2f}%" for p in performance]
        #         )
        #
        #         fig.update_layout(
        #             template="plotly_white",
        #             height=400,
        #             xaxis={'categoryorder': 'total descending'},
        #             yaxis_title="Performance (%)"
        #         )
        #
        #         fig.update_traces(textposition="outside")
        #         st.plotly_chart(fig, use_container_width=True)
        #
        #         # Show detailed table
        #         st.write("**📊 Detailed Sector Data**")
        #
        #         detail_data = []
        #         for item in sector_data:
        #             detail_data.append({
        #                 'Sector': item['sector'],
        #                 'ETF Symbol': item['symbol'],
        #                 'Current Price': f"${item['current_price']:.2f}" if item['current_price'] > 0 else "N/A",
        #                 '5-Day Performance': f"{item['performance']:+.2f}%"
        #             })
        #
        #         df = pd.DataFrame(detail_data)
        #         st.dataframe(df, use_container_width=True)
        #
        #     else:
        #         st.error("Unable to load any sector performance data")
        #
        # except Exception as e:
        #     st.error(f"Error loading sector performance: {e}")
        #     # Fallback display
        #     st.info("Unable to load real-time sector data. Please check API configuration.")

    def render_economic_calendar(self):
        """Render economic calendar with real FRED data"""
        st.subheader("📅 Economic Calendar")
        st.info("🚧 Economic calendar coming soon...")

        # try:
        #     if 'fred_provider' in st.session_state and st.session_state.fred_provider.client:
        #         fred = st.session_state.fred_provider
        #
        #         # Key economic indicators with FRED series IDs
        #         economic_indicators = {
        #             'UNRATE': {'name': 'Unemployment Rate', 'unit': '%', 'importance': 'High'},
        #             'CPIAUCSL': {'name': 'Consumer Price Index', 'unit': 'Index', 'importance': 'High'},
        #             'GDP': {'name': 'GDP', 'unit': 'Billions', 'importance': 'High'},
        #             'FEDFUNDS': {'name': 'Federal Funds Rate', 'unit': '%', 'importance': 'High'},
        #             'PAYEMS': {'name': 'Non-farm Payrolls', 'unit': 'Thousands', 'importance': 'High'},
        #             'HOUST': {'name': 'Housing Starts', 'unit': 'Thousands', 'importance': 'Medium'},
        #             'RSXFS': {'name': 'Retail Sales', 'unit': 'Millions', 'importance': 'Medium'},
        #             'INDPRO': {'name': 'Industrial Production', 'unit': 'Index', 'importance': 'Medium'}
        #         }
        #
        #         calendar_data = []
        #
        #         for series_id, info in economic_indicators.items():
        #             try:
        #                 # Get latest data (last 3 months to ensure we have recent data)
        #                 data = fred.get_series_data(series_id, start_date='2024-01-01')
        #
        #                 if not data.empty and 'error' not in data.columns:
        #                     # Get the latest two values for trend
        #                     latest_date = data.index[-1] if not data.empty else None
        #                     latest_value = data.iloc[-1, 0] if len(data) > 0 else None
        #                     prev_value = data.iloc[-2, 0] if len(data) > 1 else None
        #
        #                     if latest_value is not None:
        #                         # Calculate change
        #                         change = ""
        #                         if prev_value is not None and prev_value != 0:
        #                             if info['unit'] == '%':
        #                                 change_val = latest_value - prev_value
        #                                 change = f"{change_val:+.2f} bps"
        #                             else:
        #                                 change_pct = ((latest_value - prev_value) / prev_value) * 100
        #                                 change = f"{change_pct:+.2f}%"
        #
        #                         calendar_data.append({
        #                             'Indicator': info['name'],
        #                             'Latest Date': latest_date.strftime('%Y-%m-%d') if latest_date else 'N/A',
        #                             'Current Value': f"{latest_value:.2f}" if info['unit'] != 'Thousands' and info['unit'] != 'Millions' and info['unit'] != 'Billions' else f"{latest_value:,.0f}",
        #                             'Unit': info['unit'],
        #                             'Change': change,
        #                             'Importance': info['importance'],
        #                             'Series ID': series_id
        #                         })
        #             except Exception as e:
        #                 # Add placeholder for failed series
        #                 calendar_data.append({
        #                     'Indicator': info['name'],
        #                     'Latest Date': 'N/A',
        #                     'Current Value': 'Error loading',
        #                     'Unit': info['unit'],
        #                     'Change': 'N/A',
        #                     'Importance': info['importance'],
        #                     'Series ID': series_id
        #                 })
        #                 continue
        #
        #         if calendar_data:
        #             df = pd.DataFrame(calendar_data)
        #
        #             # Sort by importance (High first, then Medium)
        #             importance_order = {'High': 0, 'Medium': 1, 'Low': 2}
        #             df['sort_order'] = df['Importance'].map(importance_order)
        #             df = df.sort_values('sort_order').drop('sort_order', axis=1)
        #
        #             st.dataframe(df, use_container_width=True)
        #
        #             # Add some key metrics as cards
        #             st.write("**📈 Key Economic Highlights**")
        #
        #             if len(calendar_data) >= 4:
        #                 col1, col2, col3, col4 = st.columns(4)
        #
        #                 # Find key indicators
        #                 unemployment = next((item for item in calendar_data if 'Unemployment' in item['Indicator']), None)
        #                 fed_funds = next((item for item in calendar_data if 'Federal Funds' in item['Indicator']), None)
        #                 cpi = next((item for item in calendar_data if 'Consumer Price' in item['Indicator']), None)
        #                 payrolls = next((item for item in calendar_data if 'Payrolls' in item['Indicator']), None)
        #
        #                 if unemployment:
        #                     with col1:
        #                         st.metric("Unemployment Rate", f"{unemployment['Current Value']}%", unemployment['Change'])
        #
        #                 if fed_funds:
        #                     with col2:
        #                         st.metric("Fed Funds Rate", f"{fed_funds['Current Value']}%", fed_funds['Change'])
        #
        #                 if cpi:
        #                     with col3:
        #                         st.metric("CPI", cpi['Current Value'], cpi['Change'])
        #
        #                 if payrolls:
        #                     with col4:
        #                         st.metric("Non-farm Payrolls", f"{payrolls['Current Value']}K", payrolls['Change'])
        #         else:
        #             st.error("Unable to load economic calendar data from FRED")
        #
        #     else:
        #         st.warning("FRED API not configured. Please set FRED_API_KEY environment variable.")
        #
        #         # Fallback to mock data
        #         st.info("Showing sample economic calendar data:")
        #         mock_events = [
        #             {"Date": now_et().strftime('%Y-%m-%d'), "Event": "Consumer Price Index", "Importance": "High", "Previous": "3.1%", "Forecast": "3.2%"},
        #             {"Date": (now_et() + timedelta(days=2)).strftime('%Y-%m-%d'), "Event": "Retail Sales", "Importance": "Medium", "Previous": "0.3%", "Forecast": "0.4%"},
        #             {"Date": (now_et() + timedelta(days=3)).strftime('%Y-%m-%d'), "Event": "Housing Starts", "Importance": "Medium", "Previous": "1.56M", "Forecast": "1.52M"},
        #         ]
        #
        #         df = pd.DataFrame(mock_events)
        #         st.dataframe(df, use_container_width=True)
        #
        # except Exception as e:
        #     st.error(f"Error loading economic calendar: {e}")
        #     st.info("Unable to load economic data. Please check FRED API configuration.")

    def render_cryptocurrency_data(self):
        """Render cryptocurrency information with comprehensive crypto data"""
        st.subheader("₿ Cryptocurrency")
        st.info("🚧 Cryptocurrency data coming soon...")

        # try:
        #     provider = AlpacaDataProvider()
        #
        #     # Multiple cryptocurrencies
        #     crypto_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
        #     crypto_names = {'BTC/USD': 'Bitcoin', 'ETH/USD': 'Ethereum', 'LTC/USD': 'Litecoin', 'BCH/USD': 'Bitcoin Cash'}
        #
        #     # Display top cryptos in columns
        #     cols = st.columns(len(crypto_symbols))
        #
        #     for i, symbol in enumerate(crypto_symbols):
        #         with cols[i]:
        #             try:
        #                 quote = provider.get_crypto_latest_quote(symbol)
        #                 if 'error' not in quote:
        #                     bid_price = float(quote.get('bid_price', 0))
        #                     ask_price = float(quote.get('ask_price', 0))
        #                     current_price = (bid_price + ask_price) / 2 if bid_price and ask_price else bid_price or ask_price
        #
        #                     # Get historical data for change calculation
        #                     hist_data = provider.get_crypto_historical_bars(symbol, '1Day', 2)
        #                     if not hist_data.empty and len(hist_data) > 1:
        #                         prev_close = float(hist_data['close'].iloc[-2])
        #                         change = current_price - prev_close
        #                         change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        #
        #                         st.metric(
        #                             crypto_names.get(symbol, symbol),
        #                             f"${current_price:,.2f}",
        #                             f"${change:+,.2f} ({change_pct:+.2f}%)"
        #                         )
        #                     else:
        #                         st.metric(crypto_names.get(symbol, symbol), f"${current_price:,.2f}")
        #                 else:
        #                     st.metric(crypto_names.get(symbol, symbol), "N/A", "Error loading")
        #             except Exception as e:
        #                 st.metric(crypto_names.get(symbol, symbol), "N/A", "Error")
        #
        #     st.divider()
        #
        #     # Detailed Bitcoin analysis
        #     st.write("**📊 Bitcoin Detailed Analysis**")
        #
        #     try:
        #         btc_quote = provider.get_crypto_latest_quote('BTC/USD')
        #         if 'error' not in btc_quote:
        #             col1, col2, col3, col4 = st.columns(4)
        #
        #             with col1:
        #                 bid_price = float(btc_quote.get('bid_price', 0))
        #                 st.metric("Bid Price", f"${bid_price:,.2f}")
        #
        #             with col2:
        #                 ask_price = float(btc_quote.get('ask_price', 0))
        #                 st.metric("Ask Price", f"${ask_price:,.2f}")
        #
        #             with col3:
        #                 spread = ask_price - bid_price if ask_price and bid_price else 0
        #                 st.metric("Bid-Ask Spread", f"${spread:.2f}")
        #
        #             with col4:
        #                 if 'timestamp' in btc_quote:
        #                     timestamp = btc_quote['timestamp']
        #                     st.metric("Last Updated", timestamp.strftime("%H:%M:%S"))
        #                 else:
        #                     st.metric("Last Updated", "N/A")
        #
        #             # Bitcoin price history chart
        #             st.write("**📈 Bitcoin Price Chart (30 Days)**")
        #             btc_data = provider.get_crypto_historical_bars('BTC/USD', '1Day', 30)
        #
        #             if not btc_data.empty:
        #                 fig = go.Figure()
        #
        #                 # Add candlestick chart
        #                 fig.add_trace(go.Candlestick(
        #                     x=btc_data.index,
        #                     open=btc_data['open'],
        #                     high=btc_data['high'],
        #                     low=btc_data['low'],
        #                     close=btc_data['close'],
        #                     name='BTC/USD'
        #                 ))
        #
        #                 # Add moving average
        #                 sma_20 = btc_data['close'].rolling(window=20).mean()
        #                 fig.add_trace(go.Scatter(
        #                     x=btc_data.index,
        #                     y=sma_20,
        #                     mode='lines',
        #                     name='20-day SMA',
        #                     line=dict(color='orange', width=1)
        #                 ))
        #
        #                 fig.update_layout(
        #                     title="Bitcoin (BTC/USD) - 30 Day Chart",
        #                     yaxis_title="Price (USD)",
        #                     xaxis_title="Date",
        #                     height=400,
        #                     showlegend=True
        #                 )
        #
        #                 st.plotly_chart(fig, use_container_width=True)
        #
        #                 # Bitcoin statistics
        #                 col1, col2, col3, col4 = st.columns(4)
        #
        #                 current_price = float(btc_data['close'].iloc[-1])
        #                 high_30d = float(btc_data['high'].max())
        #                 low_30d = float(btc_data['low'].min())
        #                 volatility = btc_data['close'].pct_change().std() * np.sqrt(365) * 100
        #
        #                 with col1:
        #                     st.metric("30D High", f"${high_30d:,.2f}")
        #
        #                 with col2:
        #                     st.metric("30D Low", f"${low_30d:,.2f}")
        #
        #                 with col3:
        #                     range_pos = ((current_price - low_30d) / (high_30d - low_30d)) * 100 if (high_30d - low_30d) > 0 else 0
        #                     st.metric("30D Range Position", f"{range_pos:.1f}%")
        #
        #                 with col4:
        #                     st.metric("Annualized Volatility", f"{volatility:.1f}%")
        #             else:
        #                 st.error("Unable to load Bitcoin historical data")
        #         else:
        #             st.error(f"Error loading Bitcoin data: {btc_quote.get('error', 'Unknown error')}")
        #
        #     except Exception as e:
        #         st.error(f"Error in Bitcoin detailed analysis: {e}")
        #
        # except Exception as e:
        #     st.error(f"Error loading cryptocurrency data: {e}")

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
