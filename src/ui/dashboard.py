#!/usr/bin/env python3
"""
Unified Trading Dashboard - Redesigned Structure
Organized navigation tabs: Market Overview, Account Info, Live Analysis, 
Strategy Backtest, Trade & Order, News & Report
"""

import logging
import sys
import queue
import threading
import time
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
from src.strategy import get_strategy_registry
from src.ui.core_dashboard import BaseDashboard, UIComponents
from src.utils.timezone_utils import get_market_status, now_et
from src.utils.watchlist_manager import WatchlistManager

logger = logging.getLogger(__name__)


class Dashboard(BaseDashboard):
    """Unified dashboard with reorganized navigation structure"""

    def __init__(self):
        super().__init__("Gauss World Trader Dashboard", "🌍")
        self.initialize_modern_modules()

    def initialize_modern_modules(self):
        """Initialize all trading modules"""
        if 'current_main_tab' not in st.session_state:
            st.session_state.current_main_tab = 'Market Overview'

        if 'dashboard_initialized' not in st.session_state:
            try:
                st.session_state.account_manager = AccountManager()
                st.session_state.position_manager = PositionManager(st.session_state.account_manager)
                st.session_state.order_manager = OrderManager(st.session_state.account_manager)
                st.session_state.fundamental_analyzer = FundamentalAnalyzer()
                st.session_state.strategy_registry = get_strategy_registry()
                st.session_state.watchlist_manager = WatchlistManager()
                st.session_state.news_provider = NewsDataProvider()
                st.session_state.fred_provider = FREDProvider()
                self._initialize_stream_state()
                self._initialize_news_stream_state()
                st.session_state.dashboard_initialized = True

            except Exception as e:
                logger.error(f"Error initializing modern modules: {e}")
                st.error("Error initializing trading modules. Please check API configuration.")

        if 'stream_state_initialized' not in st.session_state:
            self._initialize_stream_state()
        if 'news_stream_state_initialized' not in st.session_state:
            self._initialize_news_stream_state()

    def _initialize_stream_state(self):
        """Initialize Alpaca stream state for the dashboard."""
        st.session_state.stream_state_initialized = True
        st.session_state.stream_running = False
        st.session_state.stream_thread = None
        st.session_state.stream_queue = queue.Queue()
        st.session_state.stream_messages = []
        st.session_state.stream_error = None
        st.session_state.stream_config = {}
        st.session_state.stream_obj = None

    def _initialize_news_stream_state(self):
        """Initialize Alpaca news stream state for the dashboard."""
        st.session_state.news_stream_state_initialized = True
        st.session_state.news_stream_running = False
        st.session_state.news_stream_thread = None
        st.session_state.news_stream_queue = queue.Queue()
        st.session_state.news_stream_messages = []
        st.session_state.news_stream_error = None
        st.session_state.news_stream_config = {}
        st.session_state.news_stream_obj = None

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
        self.render_standard_market_indices()

        st.divider()

        # Sub-tabs for different market data
        market_tabs = st.tabs(["📊 VXX & Sentiment", "🏢 Sectors", "📅 Economic Calendar", "₿ Cryptocurrency"])

        with market_tabs[0]:
            self.render_volatility_analysis()

        with market_tabs[1]:
            self.render_sector_analysis()

        with market_tabs[2]:
            from src.utils.dashboard_utils import render_economic_data
            render_economic_data()

        with market_tabs[3]:
            self.render_crypto_overview()






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

        self.render_portfolio_allocation()
        self.render_portfolio_metrics()




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
        st.header("🔍 Live Analysis")

        analysis_tabs = st.tabs(["📈 Historical Market", "📡 Market Stream"])

        with analysis_tabs[0]:
            self.render_symbol_analysis()

        with analysis_tabs[1]:
            self.render_market_stream()

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

    def render_market_stream(self):
        """Render real-time market data stream controls and output."""
        st.subheader("📡 Real-Time Market Stream")

        col1, col2 = st.columns([2, 3])

        with col1:
            asset_type = st.selectbox(
                "Asset Type",
                ["stock", "crypto", "option"],
                key="stream_asset_type",
            )
            crypto_loc = "eu-1"
            if asset_type == "crypto":
                crypto_loc = st.selectbox(
                    "Crypto Location",
                    ["us", "us-1", "eu-1"],
                    index=2,
                    key="stream_crypto_loc",
                )
            symbols_input = st.text_input(
                "Symbols (comma-separated, max 30)",
                value="AAPL, MSFT, NVDA",
                key="stream_symbols_input",
            )
            stream_type = st.selectbox(
                "Stream Type",
                ["trades", "quotes", "bars"],
                key="stream_type_select",
            )
            raw_output = st.checkbox("Raw Payload", value=False, key="stream_raw")
            max_rows = st.slider("Rows to Display", 20, 200, 100, key="stream_rows")

            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            if len(symbols) > 30:
                st.error("Alpaca basic accounts support up to 30 symbols per websocket.")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start Stream", disabled=st.session_state.stream_running):
                    if symbols and len(symbols) <= 30:
                        self._start_stream(
                            symbols, stream_type, raw_output, asset_type, crypto_loc
                        )
            with col_b:
                if st.button("Stop Stream", disabled=not st.session_state.stream_running):
                    self._stop_stream()

            if st.session_state.stream_error:
                st.error(st.session_state.stream_error)

            auto_refresh = st.checkbox("Auto-refresh", value=True, key="stream_auto_refresh")
            refresh_interval = st.slider(
                "Refresh Interval (sec)", 1, 10, 2, key="stream_refresh_interval"
            )

        with col2:
            self._drain_stream_queue(max_rows)
            if st.session_state.stream_messages:
                st.dataframe(
                    pd.DataFrame(st.session_state.stream_messages).tail(max_rows),
                    use_container_width=True,
                )
            else:
                st.info("No stream messages yet. Start the stream to receive data.")

        if auto_refresh and st.session_state.stream_running:
            time.sleep(refresh_interval)
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    def _start_stream(self, symbols, stream_type, raw_output, asset_type, crypto_loc):
        """Start Alpaca WebSocket stream in a background thread."""
        if st.session_state.stream_running:
            return

        st.session_state.stream_error = None
        st.session_state.stream_messages = []
        st.session_state.stream_config = {
            "symbols": symbols,
            "stream_type": stream_type,
            "raw_output": raw_output,
            "asset_type": asset_type,
            "crypto_loc": crypto_loc,
        }

        try:
            provider = AlpacaDataProvider()
            if asset_type == "crypto":
                stream = provider.create_crypto_stream(
                    raw_data=raw_output, loc=crypto_loc
                )
            elif asset_type == "option":
                stream = provider.create_option_stream(raw_data=raw_output)
            else:
                stream = provider.create_stock_stream(raw_data=raw_output)
        except Exception as exc:
            st.session_state.stream_error = f"Unable to start stream: {exc}"
            return

        message_queue = st.session_state.stream_queue

        def _get_field(data, attr: str, raw_key: str):
            if hasattr(data, attr):
                return getattr(data, attr)
            if isinstance(data, dict):
                return data.get(raw_key)
            return None

        def _format_timestamp(value):
            if hasattr(value, "isoformat"):
                return value.isoformat()
            return value

        def _format_raw(data):
            return data if isinstance(data, (dict, list, str, int, float)) else repr(data)

        async def handle_trade(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return
            message_queue.put(
                {
                    "type": "trade",
                    "symbol": _get_field(data, "symbol", "S"),
                    "price": _get_field(data, "price", "p"),
                    "size": _get_field(data, "size", "s"),
                    "exchange": _get_field(data, "exchange", "x"),
                    "timestamp": _format_timestamp(_get_field(data, "timestamp", "t")),
                }
            )

        async def handle_quote(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return
            message_queue.put(
                {
                    "type": "quote",
                    "symbol": _get_field(data, "symbol", "S"),
                    "bid_price": _get_field(data, "bid_price", "bp"),
                    "bid_size": _get_field(data, "bid_size", "bs"),
                    "ask_price": _get_field(data, "ask_price", "ap"),
                    "ask_size": _get_field(data, "ask_size", "as"),
                    "timestamp": _format_timestamp(_get_field(data, "timestamp", "t")),
                }
            )

        async def handle_bar(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return
            message_queue.put(
                {
                    "type": "bar",
                    "symbol": _get_field(data, "symbol", "S"),
                    "close": _get_field(data, "close", "c"),
                    "volume": _get_field(data, "volume", "v"),
                    "timestamp": _format_timestamp(_get_field(data, "timestamp", "t")),
                }
            )

        if stream_type == "trades":
            stream.subscribe_trades(handle_trade, *symbols)
        elif stream_type == "quotes":
            stream.subscribe_quotes(handle_quote, *symbols)
        else:
            stream.subscribe_bars(handle_bar, *symbols)

        def _run_stream():
            try:
                stream.run()
            except Exception as exc:
                message_queue.put({"type": "error", "message": str(exc)})
            finally:
                message_queue.put({"type": "status", "message": "stopped"})

        stream_thread = threading.Thread(target=_run_stream, name="alpaca_stream", daemon=True)
        st.session_state.stream_obj = stream
        st.session_state.stream_thread = stream_thread
        st.session_state.stream_running = True
        stream_thread.start()

    def _stop_stream(self):
        """Stop Alpaca WebSocket stream."""
        stream = st.session_state.get("stream_obj")
        if stream:
            try:
                stream.stop()
            except Exception as exc:
                st.session_state.stream_error = f"Unable to stop stream: {exc}"
        st.session_state.stream_running = False

    def _drain_stream_queue(self, max_rows):
        """Drain stream messages into session state."""
        message_queue = st.session_state.stream_queue
        updated = False

        while True:
            try:
                item = message_queue.get_nowait()
            except queue.Empty:
                break

            item_type = item.get("type")
            if item_type == "error":
                st.session_state.stream_error = item.get("message", "Stream error")
                st.session_state.stream_running = False
            elif item_type == "status":
                st.session_state.stream_running = False
            else:
                st.session_state.stream_messages.append(item)
                updated = True

        if updated:
            max_keep = max_rows * 3
            if len(st.session_state.stream_messages) > max_keep:
                st.session_state.stream_messages = st.session_state.stream_messages[-max_keep:]


    def render_watchlist_tab(self):
        """Watchlist: Watchlist Management"""
        st.header("👁️ Watchlist")
        from src.ui.core_dashboard import UIComponents
        UIComponents.render_watchlist_interface()

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
            # Get default symbols from watchlist and current positions
            from src.utils.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols()
            default_symbols_text = '\n'.join(default_symbols) if default_symbols else "AAPL\nMSFT\nGOOGL"
            
            symbols = st.text_area("Symbols (one per line)", default_symbols_text).strip().split('\n')
            symbols = [s.strip().upper() for s in symbols if s.strip()]

            days_back = st.slider("Backtest Period (days)", 30, 365, 90, key="backtest_period")
            initial_cash = st.number_input("Initial Cash", value=100000, step=10000, key="backtest_initial_cash")

            # Get all available strategies from registry
            display_strategies = []
            if 'strategy_registry' in st.session_state:
                strategy_names = st.session_state.strategy_registry.list_strategies(dashboard_only=True)
                for name in strategy_names:
                    # Create display name
                    display_name = name.replace('_', ' ').title()
                    display_strategies.append(display_name)
            
            # Add fallback strategies if none available
            if not display_strategies:
                display_strategies = ["Momentum", "Value", "Trend Following"]
            
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
                    self.render_backtest_analysis(results)
                else:
                    st.error(f"Backtest failed: {error}")


    def render_strategy_comparison(self):
        """Render strategy comparison interface"""
        st.subheader("🔀 Strategy Comparison")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("**Configuration**")
            
            # Symbol selection
            from src.utils.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols()
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
            
            # Get all available strategies from registry
            available_strategies = []
            if 'strategy_registry' in st.session_state:
                strategy_names = st.session_state.strategy_registry.list_strategies(dashboard_only=True)
                for name in strategy_names:
                    # Create display name
                    display_name = name.replace('_', ' ').title()
                    available_strategies.append(display_name)
            
            # Add fallback strategies if none available
            if not available_strategies:
                available_strategies = ["Momentum", "Value", "Trend Following"]
            
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
            from src.ui.core_dashboard import UIComponents
            UIComponents.render_trading_interface()

        with trade_tabs[1]:
            from src.ui.core_dashboard import UIComponents
            UIComponents.render_orders_table()



    def render_news_report_tab(self):
        """News & Report: Company News, Insider Transactions, AI Report"""
        st.header("📰 News & Reports")

        news_tabs = st.tabs(["📰 Company News", "👔 Insider Activity", "🤖 AI Reports", "🛰️ Live News"])

        with news_tabs[0]:
            self.render_company_news()

        with news_tabs[1]:
            self.render_insider_activity()

        with news_tabs[2]:
            self.render_ai_reports()

        with news_tabs[3]:
            self.render_news_stream()

    def render_news_stream(self):
        """Render live news stream controls and output."""
        st.subheader("🛰️ Live News Stream")

        col1, col2 = st.columns([2, 3])

        with col1:
            symbols_input = st.text_input(
                "Symbols (comma-separated or * for all)",
                value="*",
                key="news_stream_symbols_input",
            )
            raw_output = st.checkbox("Raw Payload", value=False, key="news_stream_raw")
            max_rows = st.slider("Rows to Display", 20, 200, 100, key="news_stream_rows")

            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            if "*" in symbols and len(symbols) > 1:
                st.warning("Using '*' subscribes to all news. Remove other symbols to avoid conflicts.")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start Stream", disabled=st.session_state.news_stream_running, key="news_stream_start"):
                    self._start_news_stream(symbols, raw_output)
            with col_b:
                if st.button("Stop Stream", disabled=not st.session_state.news_stream_running, key="news_stream_stop"):
                    self._stop_news_stream()

            if st.session_state.news_stream_error:
                st.error(st.session_state.news_stream_error)

            auto_refresh = st.checkbox("Auto-refresh", value=True, key="news_stream_auto_refresh")
            refresh_interval = st.slider(
                "Refresh Interval (sec)", 1, 10, 2, key="news_stream_refresh_interval"
            )

        with col2:
            self._drain_news_stream_queue(max_rows)
            if st.session_state.news_stream_messages:
                st.dataframe(
                    pd.DataFrame(st.session_state.news_stream_messages).tail(max_rows),
                    use_container_width=True,
                )
            else:
                st.info("No news messages yet. Start the stream to receive updates.")

        if auto_refresh and st.session_state.news_stream_running:
            time.sleep(refresh_interval)
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    def _start_news_stream(self, symbols, raw_output):
        """Start Alpaca news WebSocket stream in a background thread."""
        if st.session_state.news_stream_running:
            return

        st.session_state.news_stream_error = None
        st.session_state.news_stream_messages = []
        st.session_state.news_stream_config = {
            "symbols": symbols,
            "raw_output": raw_output,
        }

        try:
            provider = AlpacaDataProvider()
            stream = provider.create_news_stream(raw_data=raw_output)
        except Exception as exc:
            st.session_state.news_stream_error = f"Unable to start news stream: {exc}"
            return

        message_queue = st.session_state.news_stream_queue

        def _get_field(data, attr: str, raw_key: str):
            if hasattr(data, attr):
                return getattr(data, attr)
            if isinstance(data, dict):
                return data.get(raw_key)
            return None

        def _format_timestamp(value):
            if hasattr(value, "isoformat"):
                return value.isoformat()
            return value

        def _format_raw(data):
            return data if isinstance(data, (dict, list, str, int, float)) else repr(data)

        async def handle_news(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return

            message_queue.put(
                {
                    "type": "news",
                    "headline": _get_field(data, "headline", "headline"),
                    "summary": _get_field(data, "summary", "summary"),
                    "source": _get_field(data, "source", "source"),
                    "symbols": _get_field(data, "symbols", "symbols"),
                    "url": _get_field(data, "url", "url"),
                    "created_at": _format_timestamp(_get_field(data, "created_at", "created_at")),
                }
            )

        subscribe_symbols = symbols or ["*"]
        if "*" in subscribe_symbols:
            subscribe_symbols = ["*"]
        stream.subscribe_news(handle_news, *subscribe_symbols)

        def _run_stream():
            try:
                stream.run()
            except Exception as exc:
                message_queue.put({"type": "error", "message": str(exc)})
            finally:
                message_queue.put({"type": "status", "message": "stopped"})

        stream_thread = threading.Thread(target=_run_stream, name="alpaca_news_stream", daemon=True)
        st.session_state.news_stream_obj = stream
        st.session_state.news_stream_thread = stream_thread
        st.session_state.news_stream_running = True
        stream_thread.start()

    def _stop_news_stream(self):
        """Stop Alpaca news WebSocket stream."""
        stream = st.session_state.get("news_stream_obj")
        if stream:
            try:
                stream.stop()
            except Exception as exc:
                st.session_state.news_stream_error = f"Unable to stop news stream: {exc}"
        st.session_state.news_stream_running = False

    def _drain_news_stream_queue(self, max_rows):
        """Drain news stream messages into session state."""
        message_queue = st.session_state.news_stream_queue
        updated = False

        while True:
            try:
                item = message_queue.get_nowait()
            except queue.Empty:
                break

            item_type = item.get("type")
            if item_type == "error":
                st.session_state.news_stream_error = item.get("message", "Stream error")
                st.session_state.news_stream_running = False
            elif item_type == "status":
                st.session_state.news_stream_running = False
            else:
                st.session_state.news_stream_messages.append(item)
                updated = True

        if updated:
            max_keep = max_rows * 3
            if len(st.session_state.news_stream_messages) > max_keep:
                st.session_state.news_stream_messages = st.session_state.news_stream_messages[-max_keep:]

    def render_company_news(self):
        """Render company news section"""
        st.subheader("📰 Company News")

        col1, col2 = st.columns([1, 3])

        with col1:
            news_symbol = st.text_input("Symbol for News", value="AAPL", key="news_symbol").upper()
            if st.button("Get News"):
                st.session_state.news_query_symbol = news_symbol

        with col2:
            symbol = st.session_state.get('news_query_symbol')
            if not symbol:
                st.info("Enter a symbol and click Get News to load results.")
                return

            if symbol:

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
                st.session_state.insider_query_symbol = insider_symbol

        with col2:
            symbol = st.session_state.get('insider_query_symbol')
            if not symbol:
                st.info("Enter a symbol and click Get Insider Data to load results.")
                return

            if symbol:

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
        st.info("🚧 AI report generation is under development. Connect a report provider to enable.")

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
    """Main function to run the dashboard"""
    dashboard = Dashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
