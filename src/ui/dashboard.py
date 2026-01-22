#!/usr/bin/env python3
"""
Dashboard - Main Streamlit dashboard for Gauss World Trader.

This module provides a comprehensive trading dashboard with market overview,
account management, trading interface, analysis tools, and news features.
Uses a mixin-based architecture for maintainability.
"""

import logging
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.account.account_manager import AccountManager
from src.account.order_manager import OrderManager
from src.account.position_manager import PositionManager
from src.agent.fundamental_analyzer import FundamentalAnalyzer
from src.agent.watchlist_manager import WatchlistManager
from src.analysis import TechnicalAnalysis
from src.data import AlpacaDataProvider, NewsDataProvider
from src.data import FREDProvider
from src.strategy import get_strategy_registry
from src.trade.backtester import Backtester
from src.utils.timezone_utils import get_market_status, now_et

from src.ui.market_views import MarketViewsMixin
from src.ui.account_views import AccountViewsMixin
from src.ui.trading_views import TradingViewsMixin
from src.ui.analysis_views import AnalysisViewsMixin
from src.ui.ui_components import UIComponents

logger = logging.getLogger(__name__)


def _get_field(data, attr: str, raw_key: str):
    """Extract field from stream data (object attribute or dict key)."""
    if hasattr(data, attr):
        return getattr(data, attr)
    if isinstance(data, dict):
        return data.get(raw_key)
    return None


def _format_timestamp(value):
    """Format timestamp for display."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _format_raw(data):
    """Format raw stream data for display."""
    return data if isinstance(data, (dict, list, str, int, float)) else repr(data)


class Dashboard(MarketViewsMixin, AccountViewsMixin, TradingViewsMixin, AnalysisViewsMixin):
    """
    Main dashboard class combining all view mixins.

    The Dashboard orchestrates the entire UI by combining specialized mixin classes
    that handle different functional areas (market views, account views, etc.).
    """

    def __init__(self, title: str = "Gauss World Trader Dashboard", icon: str = "chart_with_upwards_trend"):
        """Initialize the dashboard."""
        self.title = title
        self.icon = icon
        self.configure_page()
        self.apply_styles()
        self.initialize_modules()

    def configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.title,
            page_icon=f":{self.icon}:",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def apply_styles(self):
        """Apply custom CSS styles."""
        st.markdown("""
        <style>
        .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
        div[data-testid="stMetricValue"] { font-size: 1.5rem; }
        </style>
        """, unsafe_allow_html=True)

    def initialize_modules(self):
        """Initialize all trading modules."""
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
                logger.error(f"Error initializing modules: {e}")
                st.error("Error initializing trading modules. Please check API configuration.")

        if 'stream_state_initialized' not in st.session_state:
            self._initialize_stream_state()
        if 'news_stream_state_initialized' not in st.session_state:
            self._initialize_news_stream_state()

    def _initialize_stream_state(self):
        """Initialize Alpaca stream state."""
        st.session_state.stream_state_initialized = True
        st.session_state.stream_running = False
        st.session_state.stream_thread = None
        st.session_state.stream_queue = queue.Queue()
        st.session_state.stream_messages = []
        st.session_state.stream_error = None
        st.session_state.stream_config = {}
        st.session_state.stream_obj = None

    def _initialize_news_stream_state(self):
        """Initialize Alpaca news stream state."""
        st.session_state.news_stream_state_initialized = True
        st.session_state.news_stream_running = False
        st.session_state.news_stream_thread = None
        st.session_state.news_stream_queue = queue.Queue()
        st.session_state.news_stream_messages = []
        st.session_state.news_stream_error = None
        st.session_state.news_stream_config = {}
        st.session_state.news_stream_obj = None

    def get_account_info(self):
        """Get account information from Alpaca."""
        try:
            provider = AlpacaDataProvider()
            info = provider.get_account()
            if info and 'error' not in info:
                return info, None
            return None, info.get('error', 'Unknown error')
        except Exception as e:
            return None, str(e)

    def load_market_data(self, symbol: str, days: int):
        """Load market data for a symbol."""
        try:
            provider = AlpacaDataProvider()
            start_date = now_et() - timedelta(days=days)
            data = provider.get_bars(symbol, "1Day", start=start_date)
            if data is not None and not data.empty:
                return data, None
            return None, "No data available"
        except Exception as e:
            return None, str(e)

    def create_price_chart(self, symbol: str, data: pd.DataFrame):
        """Create a price chart with indicators."""
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['open'], high=data['high'],
            low=data['low'], close=data['close'], name=symbol
        ))
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=min(50, len(data))).mean()
        fig.add_trace(go.Scatter(x=data.index, y=sma_20, mode='lines', name='SMA 20',
                                 line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=data.index, y=sma_50, mode='lines', name='SMA 50',
                                 line=dict(color='purple', width=1)))
        fig.update_layout(title=f"{symbol} Price Chart", height=500, showlegend=True)
        return fig

    def run_backtest(self, symbols, days_back, initial_cash, strategy_type):
        """Run a backtest for the given parameters."""
        try:
            provider = AlpacaDataProvider()
            end_date = now_et()
            start_date = end_date - timedelta(days=days_back)
            strategy_name = strategy_type.lower().replace(' ', '_')
            registry = get_strategy_registry()
            try:
                strategy = registry.create(strategy_name)
            except KeyError:
                available = registry.list_strategies(dashboard_only=True)
                for name in available:
                    if name.replace('_', ' ').title() == strategy_type:
                        strategy = registry.create(name)
                        break
                else:
                    return None, f"Strategy '{strategy_type}' not found"

            backtester = Backtester(initial_cash=initial_cash, commission=0.01)
            for symbol in symbols:
                data = provider.get_bars(symbol, "1Day", start=start_date)
                if data is not None and not data.empty:
                    backtester.add_data(symbol, data)

            def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
                return strategy.generate_signals(
                    current_date, current_prices, current_data, historical_data, portfolio
                )

            results = backtester.run_backtest(strategy_func, symbols=symbols)
            if results:
                return {
                    'total_return_percentage': results.get('total_return_percentage', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown_percentage': results.get('max_drawdown_percentage', 0),
                    'win_rate': results.get('win_rate', 0),
                    'total_trades': results.get('total_trades', 0),
                    'final_value': results.get('final_value', initial_cash),
                    'volatility': results.get('volatility', 0),
                    'portfolio_history': results.get('portfolio_history'),
                    'trades_history': results.get('trades_history'),
                }, None
            return None, "Backtest produced no results"
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return None, str(e)

    def render_backtest_analysis(self, results):
        """Render backtest analysis results."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{results.get('total_return_percentage', 0):.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Max Drawdown", f"{results.get('max_drawdown_percentage', 0):.2f}%")
        with col4:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.1f}%")

        portfolio_history = results.get('portfolio_history')
        if portfolio_history is not None and not portfolio_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_history.index, y=portfolio_history['portfolio_value'],
                mode='lines', name='Portfolio Value', line=dict(color='blue', width=2)
            ))
            fig.update_layout(title="Portfolio Performance", yaxis_title="Value ($)", height=400)
            st.plotly_chart(fig, use_container_width=True)

    def create_main_navigation(self):
        """Create main navigation tabs in the sidebar."""
        with st.sidebar:
            logo_path = Path(__file__).parent.parent / "pic" / "logo2.png"
            if logo_path.exists():
                st.image(str(logo_path), width=150)

            st.header("Navigation")
            selected_tab = st.radio(
                "Choose a section:",
                ["📊 Market Overview", "💼 Account Info", "🔍 Live Analysis", "👁️ Watchlist",
                 "📈 Strategy Backtest", "⚡ Trade & Order", "📰 News & Report"],
                key="main_navigation"
            )

            st.divider()
            self.render_market_status_sidebar()
            self.render_portfolio_quick_view()

        if selected_tab == "📊 Market Overview":
            self.render_market_overview_tab()
        elif selected_tab == "💼 Account Info":
            self.render_account_info_tab()
        elif selected_tab == "🔍 Live Analysis":
            self.render_live_analysis_tab_extended()
        elif selected_tab == "👁️ Watchlist":
            self.render_watchlist_tab()
        elif selected_tab == "📈 Strategy Backtest":
            self.render_strategy_backtest_tab_extended()
        elif selected_tab == "⚡ Trade & Order":
            self.render_trade_order_tab_extended()
        elif selected_tab == "📰 News & Report":
            self.render_news_report_tab_extended()

    def render_market_status_sidebar(self):
        """Render market status in sidebar."""
        st.subheader("🏛️ Market Status")
        local_time, et_time = datetime.now(), now_et()
        market_status = get_market_status()
        status_color = "green" if market_status == 'open' else "red"
        st.write(f"**Local Time:** {local_time.strftime('%H:%M:%S')}")
        st.write(f"**ET Time:** {et_time.strftime('%H:%M:%S')}")
        st.markdown(f"**Status:** :{status_color}[{market_status.title()}]")
        st.divider()

    def render_portfolio_quick_view(self):
        """Render portfolio quick view in sidebar."""
        st.subheader("📊 Quick View")
        try:
            account_info, error = self.get_account_info()
            if account_info:
                portfolio_value = float(account_info.get('portfolio_value', 0))
                equity = float(account_info.get('equity', 0))
                last_equity = float(account_info.get('last_equity', equity))
                day_pl = equity - last_equity
                day_pl_pct = (day_pl / last_equity * 100) if last_equity > 0 else 0
                st.metric("Portfolio Value", f"${portfolio_value:,.2f}",
                          f"${day_pl:+,.2f} ({day_pl_pct:+.2f}%)")

                if 'position_manager' in st.session_state:
                    positions = st.session_state.position_manager.get_all_positions()
                    if positions and not any('error' in pos for pos in positions):
                        total_pl = sum(float(p.get('unrealized_pl', 0)) for p in positions)
                        total_pl_pct = (total_pl / portfolio_value * 100) if portfolio_value > 0 else 0
                        st.metric("Position P&L", f"${total_pl:+,.2f}", f"{total_pl_pct:+.2f}%")
            else:
                st.info("Account data unavailable")
        except Exception as e:
            st.error(f"Error: {e}")

    def render_live_analysis_tab_extended(self):
        """Extended live analysis with market stream."""
        st.header("🔍 Live Analysis")
        analysis_tabs = st.tabs(["📊 Historical Market", "📡 Market Stream"])
        with analysis_tabs[0]:
            self.render_symbol_analysis_extended()
        with analysis_tabs[1]:
            self.render_market_stream_extended()

    def render_symbol_analysis_extended(self):
        """Extended symbol analysis."""
        col1, col2 = st.columns([1, 3])
        with col1:
            symbol = st.text_input("Enter Symbol", value="AAPL", key="analysis_symbol_ext").upper()
            days = st.selectbox("Analysis Period", [30, 60, 90, 180, 365], index=2, key="analysis_period_ext")
            if st.button("Analyze", key="analyze_btn"):
                st.session_state.analyze_symbol = symbol
                st.session_state.analyze_days = days

        with col2:
            if hasattr(st.session_state, 'analyze_symbol'):
                symbol = st.session_state.analyze_symbol
                days = st.session_state.analyze_days
                data, error = self.load_market_data(symbol, days)
                if data is not None and not data.empty:
                    fig = self.create_price_chart(symbol, data)
                    st.plotly_chart(fig, use_container_width=True)
                    ta = TechnicalAnalysis()
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        sma_20 = ta.sma(data['close'], 20)
                        current_price = data['close'].iloc[-1]
                        sma_current = sma_20.iloc[-1] if not sma_20.empty else current_price
                        trend = "Bullish" if current_price > sma_current else "Bearish"
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

    def render_market_stream_extended(self):
        """Extended market stream with WebSocket support."""
        st.subheader("Real-Time Market Stream")
        col1, col2 = st.columns([2, 3])
        with col1:
            asset_type = st.selectbox("Asset Type", ["stock", "crypto", "option"], key="stream_asset_type")
            crypto_loc = "eu-1"
            if asset_type == "crypto":
                crypto_loc = st.selectbox("Crypto Location", ["us", "us-1", "eu-1"], index=2, key="stream_crypto_loc")
            symbols_input = st.text_input("Symbols (comma-separated, max 30)", value="AAPL, MSFT, NVDA",
                                          key="stream_symbols_input")
            stream_type = st.selectbox("Stream Type", ["trades", "quotes", "bars"], key="stream_type_select")
            raw_output = st.checkbox("Raw Payload", value=False, key="stream_raw")
            max_rows = st.slider("Rows to Display", 20, 200, 100, key="stream_rows")
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            if len(symbols) > 30:
                st.error("Max 30 symbols supported per websocket.")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start Stream", disabled=st.session_state.stream_running):
                    if symbols and len(symbols) <= 30:
                        self._start_stream(symbols, stream_type, raw_output, asset_type, crypto_loc)
            with col_b:
                if st.button("Stop Stream", disabled=not st.session_state.stream_running):
                    self._stop_stream()

            if st.session_state.stream_error:
                st.error(st.session_state.stream_error)
            auto_refresh = st.checkbox("Auto-refresh", value=True, key="stream_auto_refresh")
            refresh_interval = st.slider("Refresh Interval (sec)", 1, 10, 2, key="stream_refresh_interval")

        with col2:
            self._drain_stream_queue(max_rows)
            if st.session_state.stream_messages:
                st.dataframe(pd.DataFrame(st.session_state.stream_messages).tail(max_rows), use_container_width=True)
            else:
                st.info("No stream messages yet. Start the stream to receive data.")

        if auto_refresh and st.session_state.stream_running:
            time.sleep(refresh_interval)
            st.rerun()

    def _start_stream(self, symbols, stream_type, raw_output, asset_type, crypto_loc):
        """Start Alpaca WebSocket stream."""
        if st.session_state.stream_running:
            return
        st.session_state.stream_error = None
        st.session_state.stream_messages = []
        st.session_state.stream_config = {
            "symbols": symbols, "stream_type": stream_type, "raw_output": raw_output,
            "asset_type": asset_type, "crypto_loc": crypto_loc,
        }
        try:
            provider = AlpacaDataProvider()
            if asset_type == "crypto":
                stream = provider.create_crypto_stream(raw_data=raw_output, loc=crypto_loc)
            elif asset_type == "option":
                stream = provider.create_option_stream(raw_data=raw_output)
            else:
                stream = provider.create_stock_stream(raw_data=raw_output)
        except Exception as exc:
            st.session_state.stream_error = f"Unable to start stream: {exc}"
            return
        message_queue = st.session_state.stream_queue

        async def handle_trade(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return
            message_queue.put({
                "type": "trade", "symbol": _get_field(data, "symbol", "S"),
                "price": _get_field(data, "price", "p"), "size": _get_field(data, "size", "s"),
                "exchange": _get_field(data, "exchange", "x"),
                "timestamp": _format_timestamp(_get_field(data, "timestamp", "t")),
            })

        async def handle_quote(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return
            message_queue.put({
                "type": "quote", "symbol": _get_field(data, "symbol", "S"),
                "bid_price": _get_field(data, "bid_price", "bp"), "bid_size": _get_field(data, "bid_size", "bs"),
                "ask_price": _get_field(data, "ask_price", "ap"), "ask_size": _get_field(data, "ask_size", "as"),
                "timestamp": _format_timestamp(_get_field(data, "timestamp", "t")),
            })

        async def handle_bar(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return
            message_queue.put({
                "type": "bar", "symbol": _get_field(data, "symbol", "S"),
                "close": _get_field(data, "close", "c"), "volume": _get_field(data, "volume", "v"),
                "timestamp": _format_timestamp(_get_field(data, "timestamp", "t")),
            })

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
        """Render watchlist management."""
        st.header("👁️ Watchlist")
        UIComponents.render_watchlist_interface()

    def render_strategy_backtest_tab_extended(self):
        """Extended strategy backtest with comparison."""
        st.header("📈 Strategy Backtesting")
        backtest_tabs = st.tabs(["⚡ Quick Backtest", "📊 Strategy Comparison"])
        with backtest_tabs[0]:
            self.render_quick_backtest_extended()
        with backtest_tabs[1]:
            self.render_strategy_comparison_extended()

    def render_quick_backtest_extended(self):
        """Extended quick backtest."""
        st.subheader("Quick Backtest")
        col1, col2 = st.columns([1, 2])
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols()
            default_symbols_text = '\n'.join(default_symbols) if default_symbols else "AAPL\nMSFT\nGOOGL"
            symbols = st.text_area("Symbols (one per line)", default_symbols_text).strip().split('\n')
            symbols = [s.strip().upper() for s in symbols if s.strip()]
            days_back = st.slider("Backtest Period (days)", 30, 365, 90, key="backtest_period")
            initial_cash = st.number_input("Initial Cash", value=100000, step=10000, key="backtest_initial_cash")
            display_strategies = []
            if 'strategy_registry' in st.session_state:
                strategy_names = st.session_state.strategy_registry.list_strategies(dashboard_only=True)
                for name in strategy_names:
                    display_strategies.append(name.replace('_', ' ').title())
            if not display_strategies:
                display_strategies = ["Momentum", "Value", "Trend Following"]
            strategy_display = st.selectbox("Strategy", display_strategies, key="backtest_strategy")
            if st.button("Run Backtest"):
                st.session_state.run_backtest = True
                st.session_state.backtest_params = {
                    'symbols': symbols, 'days_back': days_back,
                    'initial_cash': initial_cash, 'strategy_type': strategy_display
                }
        with col2:
            if hasattr(st.session_state, 'run_backtest') and st.session_state.run_backtest:
                params = st.session_state.backtest_params
                with st.spinner(f"Running {params['strategy_type']} strategy backtest..."):
                    results, error = self.run_backtest(
                        params['symbols'], params['days_back'], params['initial_cash'], params['strategy_type']
                    )
                if results:
                    self.render_backtest_analysis(results)
                else:
                    st.error(f"Backtest failed: {error}")

    def render_strategy_comparison_extended(self):
        """Extended strategy comparison."""
        st.subheader("Strategy Comparison")
        col1, col2 = st.columns([1, 2])
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols()
            symbol = st.selectbox("Select Symbol", options=default_symbols + ["AAPL", "MSFT", "GOOGL", "TSLA"]
                                  if default_symbols else ["AAPL", "MSFT", "GOOGL", "TSLA"], key="comp_symbol")
            available_strategies = []
            if 'strategy_registry' in st.session_state:
                strategy_names = st.session_state.strategy_registry.list_strategies(dashboard_only=True)
                for name in strategy_names:
                    available_strategies.append(name.replace('_', ' ').title())
            if not available_strategies:
                available_strategies = ["Momentum", "Value", "Trend Following"]
            if 'comparison_strategies' not in st.session_state:
                st.session_state.comparison_strategies = available_strategies[:3]
            st.write("**Selected Strategies**")
            for strategy in st.session_state.comparison_strategies:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"- {strategy}")
                with col_b:
                    if st.button("X", key=f"remove_{strategy}"):
                        st.session_state.comparison_strategies.remove(strategy)
                        st.rerun()
            remaining = [s for s in available_strategies if s not in st.session_state.comparison_strategies]
            if remaining:
                add_strategy = st.selectbox("Add Strategy", [""] + remaining, key="add_strat")
                if st.button("Add Strategy") and add_strategy:
                    st.session_state.comparison_strategies.append(add_strategy)
                    st.rerun()
            st.divider()
            days_back = st.slider("Period (days)", 30, 365, 90, key="comparison_period")
            initial_cash = st.number_input("Initial Cash", value=100000, step=10000, key="comparison_cash")
            if st.button("Run Comparison", type="primary"):
                if len(st.session_state.comparison_strategies) >= 2:
                    st.session_state.run_comparison = True
                    st.session_state.comparison_params = {
                        'symbol': symbol, 'strategies': st.session_state.comparison_strategies,
                        'days_back': days_back, 'initial_cash': initial_cash
                    }
                else:
                    st.error("Please select at least 2 strategies")
        with col2:
            if hasattr(st.session_state, 'run_comparison') and st.session_state.run_comparison:
                params = st.session_state.comparison_params
                with st.spinner(f"Comparing strategies for {params['symbol']}..."):
                    comparison_results = []
                    for strategy_name in params['strategies']:
                        results, error = self.run_backtest(
                            [params['symbol']], params['days_back'], params['initial_cash'], strategy_name
                        )
                        if results:
                            comparison_results.append({
                                'Strategy': strategy_name,
                                'Total Return': f"{results.get('total_return_percentage', 0):.2f}%",
                                'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.2f}",
                                'Max Drawdown': f"{results.get('max_drawdown_percentage', 0):.2f}%",
                                'Win Rate': f"{results.get('win_rate', 0):.1f}%",
                                'Total Trades': results.get('total_trades', 0),
                            })
                        else:
                            comparison_results.append({
                                'Strategy': strategy_name, 'Total Return': 'Error',
                                'Sharpe Ratio': 'N/A', 'Max Drawdown': 'N/A',
                                'Win Rate': 'N/A', 'Total Trades': 'N/A',
                            })
                if comparison_results:
                    df = pd.DataFrame(comparison_results)
                    st.dataframe(df, use_container_width=True)
                    valid = [r for r in comparison_results if r['Total Return'] != 'Error']
                    if valid:
                        import plotly.express as px
                        strategies = [r['Strategy'] for r in valid]
                        returns = [float(r['Total Return'].rstrip('%')) for r in valid]
                        fig = px.bar(x=strategies, y=returns, title=f"Return Comparison - {params['symbol']}",
                                     color=returns, color_continuous_scale="RdYlGn",
                                     text=[f"{r:.1f}%" for r in returns])
                        fig.update_layout(yaxis_title="Total Return (%)", height=400)
                        fig.update_traces(textposition="outside")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure strategies and click 'Run Comparison' to see results")

    def render_trade_order_tab_extended(self):
        """Extended trade and order tab."""
        st.header("⚡ Trade & Order Management")
        trade_tabs = st.tabs(["🚀 Quick Trade", "📋 Recent Orders"])
        with trade_tabs[0]:
            UIComponents.render_trading_interface()
        with trade_tabs[1]:
            self.render_orders_table()

    def render_news_report_tab_extended(self):
        """Extended news and report tab."""
        st.header("📰 News & Reports")
        news_tabs = st.tabs(["📰 Company News", "👔 Insider Activity", "🤖 AI Reports", "🛰️ Live News"])
        with news_tabs[0]:
            self.render_company_news_extended()
        with news_tabs[1]:
            self.render_insider_activity_extended()
        with news_tabs[2]:
            st.info("AI report generation is under development.")
        with news_tabs[3]:
            self.render_news_stream_extended()

    def render_company_news_extended(self):
        """Extended company news."""
        st.subheader("📰 Company News")
        col1, col2 = st.columns([1, 3])
        with col1:
            news_symbol = st.text_input("Symbol", value="AAPL", key="news_symbol").upper()
            if st.button("Get News"):
                st.session_state.news_query_symbol = news_symbol
        with col2:
            symbol = st.session_state.get('news_query_symbol')
            if not symbol:
                st.info("Enter a symbol and click Get News.")
                return
            try:
                if 'news_provider' in st.session_state:
                    news = st.session_state.news_provider.get_company_news(symbol)
                    if news:
                        for article in news[:10]:
                            with st.expander(f"{article.get('headline', 'No title')[:80]}..."):
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

    def render_insider_activity_extended(self):
        """Extended insider activity."""
        st.subheader("👔 Insider Activity")
        col1, col2 = st.columns([1, 3])
        with col1:
            insider_symbol = st.text_input("Symbol", value="AAPL", key="insider_symbol").upper()
            if st.button("Get Insider Data"):
                st.session_state.insider_query_symbol = insider_symbol
        with col2:
            symbol = st.session_state.get('insider_query_symbol')
            if not symbol:
                st.info("Enter a symbol and click Get Insider Data.")
                return
            try:
                if 'news_provider' in st.session_state:
                    transactions = st.session_state.news_provider.get_insider_transactions(symbol)
                    sentiment = st.session_state.news_provider.get_insider_sentiment(symbol)
                    if sentiment:
                        st.write("**Insider Sentiment**")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Sentiment Score", sentiment.get('sentiment', 'N/A'))
                        with col_b:
                            st.metric("Buy Transactions", sentiment.get('buys', 0))
                        with col_c:
                            st.metric("Sell Transactions", sentiment.get('sells', 0))
                    if transactions:
                        st.write("**Recent Insider Transactions**")
                        for txn in transactions[:5]:
                            with st.expander(f"{txn.get('person_name', 'Unknown')} - {txn.get('transaction_type', 'Unknown')}"):
                                st.write(f"**Date:** {txn.get('transaction_date', 'Unknown')}")
                                st.write(f"**Shares:** {txn.get('shares', 'Unknown')}")
                                st.write(f"**Price:** ${txn.get('price', 0):.2f}")
                    else:
                        st.info(f"No insider data for {symbol}")
                else:
                    st.error("News provider not initialized.")
            except Exception as e:
                st.error(f"Error loading insider data: {e}")

    def render_news_stream_extended(self):
        """Extended live news stream."""
        st.subheader("🛰️ Live News Stream")
        col1, col2 = st.columns([2, 3])
        with col1:
            symbols_input = st.text_input("Symbols (* for all)", value="*", key="news_stream_symbols")
            raw_output = st.checkbox("Raw Payload", value=False, key="news_stream_raw")
            max_rows = st.slider("Rows to Display", 20, 200, 100, key="news_stream_rows")
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start", disabled=st.session_state.news_stream_running, key="news_start"):
                    self._start_news_stream(symbols, raw_output)
            with col_b:
                if st.button("Stop", disabled=not st.session_state.news_stream_running, key="news_stop"):
                    self._stop_news_stream()
            if st.session_state.news_stream_error:
                st.error(st.session_state.news_stream_error)
            auto_refresh = st.checkbox("Auto-refresh", value=True, key="news_auto_refresh")
            refresh_interval = st.slider("Refresh (sec)", 1, 10, 2, key="news_refresh")
        with col2:
            self._drain_news_stream_queue(max_rows)
            if st.session_state.news_stream_messages:
                st.dataframe(pd.DataFrame(st.session_state.news_stream_messages).tail(max_rows), use_container_width=True)
            else:
                st.info("No news messages yet. Start the stream.")
        if auto_refresh and st.session_state.news_stream_running:
            time.sleep(refresh_interval)
            st.rerun()

    def _start_news_stream(self, symbols, raw_output):
        """Start news stream."""
        if st.session_state.news_stream_running:
            return
        st.session_state.news_stream_error = None
        st.session_state.news_stream_messages = []
        try:
            provider = AlpacaDataProvider()
            stream = provider.create_news_stream(raw_data=raw_output)
        except Exception as exc:
            st.session_state.news_stream_error = f"Unable to start news stream: {exc}"
            return
        message_queue = st.session_state.news_stream_queue

        async def handle_news(data):
            if raw_output:
                message_queue.put({"type": "raw", "payload": _format_raw(data)})
                return
            message_queue.put({
                "type": "news", "headline": _get_field(data, "headline", "headline"),
                "summary": _get_field(data, "summary", "summary"),
                "source": _get_field(data, "source", "source"),
                "symbols": _get_field(data, "symbols", "symbols"),
                "url": _get_field(data, "url", "url"),
                "created_at": _format_timestamp(_get_field(data, "created_at", "created_at")),
            })

        subscribe_symbols = symbols if symbols and "*" not in symbols else ["*"]
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
        """Stop news stream."""
        stream = st.session_state.get("news_stream_obj")
        if stream:
            try:
                stream.stop()
            except Exception as exc:
                st.session_state.news_stream_error = f"Unable to stop: {exc}"
        st.session_state.news_stream_running = False

    def _drain_news_stream_queue(self, max_rows):
        """Drain news stream queue."""
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

    def render_header_info(self):
        """Render header with account tier and data source info."""
        current_time = now_et()
        try:
            provider = AlpacaDataProvider()
            account_info = provider.get_account_info()
            vip = account_info.get('vip', False)
            using_iex = account_info.get('using_iex', False)
            account_tier = "VIP Account" if vip else "Free Tier"
            is_trading_day = current_time.weekday() < 5

            col1, col2 = st.columns([1, 1])
            with col1:
                if vip:
                    st.success(f"✨ {account_tier}")
                else:
                    st.info(f"🆓 {account_tier}")
            with col2:
                if not vip and is_trading_day:
                    if using_iex:
                        st.info("📊 15 Mins Delay")
                    else:
                        st.info("📊 Live Data")
        except Exception:
            st.info("🆓 Free Tier")

    def run_dashboard(self):
        """Main dashboard execution."""
        try:
            st.markdown("<h1 style='text-align: center;'>🌍 Gauss World Trader</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-style: italic;'>Advanced Trading Platform with Comprehensive Market Analysis</p>",
                        unsafe_allow_html=True)
            self.render_header_info()
            st.divider()
            self.create_main_navigation()
            st.divider()
            st.markdown(
                f"<div style='text-align: center; color: #888; font-size: 0.8em;'>"
                f"Dashboard updated: {now_et().strftime('%Y-%m-%d %H:%M:%S')} ET | "
                f"Market Status: {get_market_status()}</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")


def main():
    """Main function to run the dashboard."""
    dashboard = Dashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
