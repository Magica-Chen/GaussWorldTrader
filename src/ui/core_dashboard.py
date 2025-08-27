#!/usr/bin/env python3
"""
Core Dashboard - Abstract base classes and shared functionality
Provides template and shared functionalities for all dashboards
Following principles: high cohesion, low coupling, less code is better
"""

import sys
from abc import ABC, abstractmethod
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

from src.utils.timezone_utils import EASTERN, now_et, get_market_status
from src.data import AlpacaDataProvider
from src.trade import TradingEngine, Backtester, Portfolio
from src.strategy import MomentumStrategy
from src.analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)


class BaseDashboard(ABC):
    """Abstract base dashboard providing common functionality"""
    
    def __init__(self, title: str, icon: str = "🌍"):
        self.title = title
        self.icon = icon
        self.configure_page()
        self.apply_styles()
        self.initialize_session_state()
    
    def configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{self.title}",
            page_icon=self.icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def apply_styles(self):
        """Apply common CSS styles"""
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
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize common session state variables"""
        if 'data_provider' not in st.session_state:
            st.session_state.data_provider = AlpacaDataProvider()
        
        if 'trading_engine' not in st.session_state:
            st.session_state.trading_engine = TradingEngine()
            
        if 'technical_analysis' not in st.session_state:
            st.session_state.technical_analysis = TechnicalAnalysis()
    
    def get_local_time(self) -> datetime:
        """Get current local time"""
        return datetime.now()
    
    def get_account_info(self) -> Tuple[Optional[Dict], Optional[str]]:
        """Get account information with error handling"""
        try:
            return st.session_state.trading_engine.get_account_info(), None
        except Exception as e:
            return None, str(e)
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def load_market_data(_self, symbol: str, days: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load market data with caching"""
        try:
            provider = AlpacaDataProvider()
            current_time = now_et()
            start_date = current_time - timedelta(days=days)
            
            data = provider.get_bars(symbol, '1Day', start_date)
            
            if data is not None and not data.empty:
                return data, None
            else:
                return None, f"No data available for {symbol}"
                
        except Exception as e:
            return None, str(e)
    
    @st.cache_data(ttl=60)
    def generate_trading_signals(_self, symbol: str, data: pd.DataFrame) -> Tuple[List[Dict], Optional[str]]:
        """Generate trading signals with caching"""
        try:
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
    
    @st.cache_data(ttl=600)
    def get_technical_indicators(_self, data: pd.DataFrame) -> Optional[Dict]:
        """Calculate technical indicators with caching"""
        try:
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
            
            # Support/Resistance
            support_resistance = ta.calculate_support_resistance(data)
            indicators['support_resistance'] = support_resistance
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return None
    
    def create_price_chart(self, symbol: str, data: pd.DataFrame) -> go.Figure:
        """Create standardized price chart with volume"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price Chart", "Volume")
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                opacity=0.3
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        return fig
    
    def render_account_metrics(self, account_data: Dict):
        """Render standardized account metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_value = float(account_data.get('portfolio_value', 0))
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
        
        with col2:
            buying_power = float(account_data.get('buying_power', 0))
            st.metric("Buying Power", f"${buying_power:,.2f}")
        
        with col3:
            cash = float(account_data.get('cash', 0))
            st.metric("Cash", f"${cash:,.2f}")
        
        with col4:
            day_trades = account_data.get('day_trade_count', 0)
            st.metric("Day Trades", day_trades)
    
    def render_current_metrics(self, data: pd.DataFrame):
        """Render current price metrics"""
        if data is None or data.empty:
            st.warning("No data available for metrics")
            return
            
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Price", 
                value=f"${current_price:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
            )
        
        with col2:
            st.metric(
                label="Volume",
                value=f"{data['volume'].iloc[-1]:,.0f}"
            )
        
        with col3:
            st.metric(
                label="Range (30d)",
                value=f"${data['low'].min():.2f} - ${data['high'].max():.2f}"
            )
    
    def render_technical_indicators(self, indicators: Dict):
        """Render technical indicators in standardized format"""
        if not indicators:
            st.warning("No technical indicators available")
            return
            
        st.subheader("🔬 Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("RSI (14)", f"{indicators.get('rsi', 0):.2f}")
            st.metric("SMA 20", f"${indicators.get('sma_20', 0):.2f}")
        
        with col2:
            st.metric("SMA 50", f"${indicators.get('sma_50', 0):.2f}")
            st.metric("MACD", f"{indicators.get('macd', 0):.4f}")
        
        # Trend Analysis
        if 'trends' in indicators and indicators['trends']:
            st.subheader("📈 Trend Analysis")
            trends = indicators['trends']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Short-term:**", trends.get('short_term_trend', 'N/A'))
            with col2:
                st.write("**Medium-term:**", trends.get('medium_term_trend', 'N/A'))
            with col3:
                st.write("**Long-term:**", trends.get('long_term_trend', 'N/A'))
    
    def render_trading_signals(self, signals: List[Dict]):
        """Render trading signals in standardized format"""
        st.subheader("🧠 Trading Signals")
        
        if not signals:
            st.info("📭 No trading signals generated")
            return
            
        for signal in signals:
            signal_color = "🟢" if signal['action'].upper() == 'BUY' else "🔴"
            st.success(f"""
            {signal_color} **{signal['action'].upper()} SIGNAL**
            
            - **Symbol**: {signal['symbol']}
            - **Quantity**: {signal['quantity']} shares  
            - **Confidence**: {signal.get('confidence', 0):.1%}
            - **Reason**: {signal.get('reason', 'N/A')}
            """)
    
    def render_header_info(self):
        """Render header with time and market status"""
        current_time = now_et()
        local_time = self.get_local_time()
        
        try:
            data_provider = AlpacaDataProvider()
            account_info = data_provider.get_account_info()
            
            vip = account_info.get('vip', False)
            using_iex = account_info.get('using_iex', False)
            account_tier = "VIP Account" if vip else "Free Tier"
            
            # Check if today is a trading day
            is_trading_day = current_time.weekday() < 5  # Monday=0, Friday=4
            
            # Create header info layout
            col1, col2= st.columns([1, 1])
            
            with col1:
                # Account tier display
                if vip:
                    st.success(f"✨ {account_tier}")
                else:
                    st.info(f"🆓 {account_tier}")

            with col2:
                # Data source notice for free tier on trading days
                if not vip and is_trading_day:
                    if using_iex:
                        st.info("📊 15 Mins Delay")
                    else:
                        st.info("📊 Live Data")
                    
        except Exception as e:
            # Fallback header

            st.info("🆓 Free Tier")
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def run_backtest(_self, symbols: List[str], days_back: int = 365, 
                    initial_cash: float = 100000, strategy_type: str = "Momentum") -> Tuple[Optional[Dict], Optional[str]]:
        """Run backtest with caching and strategy selection"""
        try:
            provider = AlpacaDataProvider()
            backtester = Backtester(initial_cash=initial_cash, commission=0.01)
            
            current_time = now_et()
            start_date = current_time - timedelta(days=days_back)
            
            # Load data for all symbols
            for symbol in symbols:
                data = provider.get_bars(symbol, '1Day', start_date)
                if not data.empty:
                    backtester.add_data(symbol, data)
            
            # Create strategy based on selection
            strategy_configs = {
                "Momentum": {},
                "Mean Reversion": {
                    'lookback_period': 10,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'position_size_pct': 0.1,
                    'stop_loss_pct': 0.05,
                    'take_profit_pct': 0.15
                },
                "Trend Following": {
                    'lookback_period': 50,
                    'rsi_period': 21,
                    'rsi_oversold': 40,
                    'rsi_overbought': 60,
                    'position_size_pct': 0.15,
                    'stop_loss_pct': 0.08,
                    'take_profit_pct': 0.25
                }
            }
            
            if strategy_type not in strategy_configs:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
                
            strategy = MomentumStrategy(strategy_configs[strategy_type])
            
            def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
                return strategy.generate_signals(
                    current_date, current_prices, current_data, historical_data, portfolio
                )
            
            # Run backtest (skip first 50 days for indicator warmup)
            results = backtester.run_backtest(
                strategy_func,
                start_date=start_date + timedelta(days=50),
                end_date=current_time,
                symbols=symbols
            )
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @abstractmethod
    def render_main_content(self):
        """Abstract method for rendering main dashboard content"""
        pass
    
    def run(self):
        """Main dashboard run method"""
        # Header
        st.title(f"{self.icon} {self.title}")
        self.render_header_info()
        st.markdown("---")
        
        # Main content (implemented by subclasses)
        self.render_main_content()


class UIComponents:
    """Static utility methods for common UI components"""
    
    @staticmethod
    def render_data_table(data: pd.DataFrame, title: str = "Data Table"):
        """Render a formatted data table"""
        st.subheader(title)
        st.dataframe(data, use_container_width=True)
    
    @staticmethod
    def render_positions_table(positions: List[Dict]):
        """Render positions table with P&L styling"""
        if not positions:
            st.info("No active positions found.")
            return
        
        positions_data = []
        for pos in positions:
            qty = float(pos.get('qty', 0))
            if qty != 0:
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
                })
        
        if positions_data:
            df = pd.DataFrame(positions_data)
            
            # Style P&L columns
            def style_pnl(val):
                if isinstance(val, str) and ('$' in val or '%' in val):
                    try:
                        num_val = float(val.replace('$', '').replace('%', '').replace(',', '').replace('+', ''))
                        return 'color: green; font-weight: bold' if num_val >= 0 else 'color: red; font-weight: bold'
                    except:
                        return ''
                return ''
            
            styled_df = df.style.map(style_pnl, subset=['Unrealized P&L', 'Unrealized %'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No active positions found.")
    
    @staticmethod
    def render_order_form():
        """Render standardized order placement form"""
        st.subheader("📤 Place Order")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
            action = st.selectbox("Action", ["BUY", "SELL"])
            quantity = st.number_input("Quantity", min_value=1, value=100)
        
        with col2:
            order_type = st.selectbox("Order Type", ["Market", "Limit"])
            
            if order_type == "Limit":
                limit_price = st.number_input("Limit Price ($)", min_value=0.01, value=100.0, step=0.01)
            else:
                limit_price = None
            
            time_in_force = st.selectbox("Time in Force", ["GTC", "DAY", "IOC", "FOK"])
        
        if st.button("🚀 Execute Trade", type="primary"):
            if symbol and quantity > 0:
                st.success(f"Order placed: {action} {quantity} shares of {symbol}")
                st.json({
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_type': order_type,
                    'limit_price': limit_price,
                    'time_in_force': time_in_force
                })
            else:
                st.error("Please fill in all required fields")