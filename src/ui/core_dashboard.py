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
from src.strategy import get_strategy_registry
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
    
    @st.cache_data(ttl=900)
    def load_market_data(
        _self, symbol: str, days: int = 30
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
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
    def generate_trading_signals(
        _self, symbol: str, data: pd.DataFrame
    ) -> Tuple[List[Dict], Optional[str]]:
        """Generate trading signals with caching"""
        try:
            if data is None or data.empty:
                return [], "No data available"
            
            registry = get_strategy_registry()
            strategy = registry.create("momentum")
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
    
    @st.cache_data(ttl=1800)
    def run_backtest(
        _self,
        symbols: List[str],
        days_back: int = 365,
        initial_cash: float = 100000,
        strategy_type: str = "Momentum"
    ) -> Tuple[Optional[Dict], Optional[str]]:
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
            
            # Create strategy using registry
            registry = get_strategy_registry()
            
            # Map display names to registry names
            strategy_name_mapping = {
                "Momentum": "momentum",
                "Trend Following": "trend_following",
                "Scalping": "scalping",
                "Statistical Arbitrage": "statistical_arbitrage",
                "Value": "value",
                "XGBoost": "xgboost",
                "Deep Learning": "deep_learning",
                "Gaussian Process": "gaussian_process"
            }
            
            # Special configs for specific strategies
            strategy_configs = {}
            
            # Get the actual strategy registry name
            default_name = strategy_type.lower().replace(' ', '_')
            registry_name = strategy_name_mapping.get(strategy_type, default_name)
            
            # Check if strategy exists in registry
            available_strategies = registry.list_strategies()
            if registry_name not in available_strategies:
                msg = f"Unknown strategy: {strategy_type}. Available: {available_strategies}"
                raise ValueError(msg)
            
            # Get strategy-specific config or use empty dict
            config = strategy_configs.get(strategy_type, {})
            
            # Create strategy instance
            strategy = registry.create(registry_name, config)
            
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

    def render_standard_market_indices(self):
        """Render real market indices data"""
        col1, col2, col3, col4 = st.columns(4)
        indices = {'SPY': 'S&P 500', 'QQQ': 'NASDAQ', 'DIA': 'DOW', 'VXX': 'VXX'}
        columns = [col1, col2, col3, col4]

        try:
            provider = AlpacaDataProvider()
            for i, (symbol, name) in enumerate(indices.items()):
                with columns[i]:
                    try:
                        quote = provider.get_latest_quote(symbol)
                        if 'error' not in quote:
                            bid = quote.get('bid_price', quote.get('ask_price', 0))
                            current_price = float(bid)
                            start = now_et() - timedelta(days=5)
                            historical_data = provider.get_bars(symbol, '1Day', start=start)

                            if not historical_data.empty:
                                idx = -2 if len(historical_data) >= 2 else -1
                                prev_close = float(historical_data['close'].iloc[idx])
                                change = current_price - prev_close
                                change_pct = (change / prev_close * 100) if prev_close else 0
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

    def render_volatility_analysis(self):
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
                if current_vxx > 40:
                    fear_greed = max(0, 30 - (current_vxx - 40) * 1.5)
                elif current_vxx > 25:
                    fear_greed = 30 + (40 - current_vxx) * 2.67
                else:
                    fear_greed = 70 + min(30, (25 - current_vxx) * 2)
                fear_greed = max(0, min(100, fear_greed))

                with col1:
                    gauge_steps = [
                        {'range': [0, 20], 'color': "red"},
                        {'range': [20, 40], 'color': "orange"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "lightgreen"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                    gauge_config = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': gauge_steps,
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fear_greed,
                        title={'text': "Fear & Greed Index (VXX-based)"},
                        gauge=gauge_config
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("**Market Sentiment Indicators**")
                    if current_vxx > 40:
                        sentiment_color, sentiment_label = "🔴", "Fearful"
                    elif current_vxx > 25:
                        sentiment_color, sentiment_label = "🟡", "Neutral"
                    else:
                        sentiment_color, sentiment_label = "🟢", "Greedy"

                    vxx_trend = "Rising" if current_vxx > vxx_30_avg else "Falling"
                    st.metric("Current VXX", f"${current_vxx:.2f}", f"30d avg: ${vxx_30_avg:.2f}")
                    st.write(f"**Market Mood:** {sentiment_color} {sentiment_label}")
                    st.write(f"**VXX Trend:** {vxx_trend}")
                    st.write("**VXX Levels:** Below $25: Low | $25-40: Normal | Above $40: High")

                    vxx_change = current_vxx - vxx_30_avg
                    vxx_change_pct = (vxx_change / vxx_30_avg * 100) if vxx_30_avg else 0
                    st.metric("VXX vs 30d Avg", f"{vxx_change_pct:+.1f}%", f"${vxx_change:+.2f}")
                    vxx_vol = float(vxx_data['close'].std())
                    st.write(f"**VXX Volatility (30d):** ${vxx_vol:.2f}")
            else:
                st.error("Unable to load VXX data")
        except Exception as e:
            st.error(f"Error loading VXX/sentiment data: {e}")
            st.info("Unable to load real-time VXX data. Please check API configuration.")

    def render_sector_analysis(self):
        """Render sector performance analysis with real data"""
        st.subheader("🏢 Sector Performance")

        try:
            provider = AlpacaDataProvider()
            sector_etfs = {
                'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financial',
                'XLE': 'Energy', 'XLY': 'Consumer Discretionary', 'XLI': 'Industrial',
                'XLB': 'Materials', 'XLRE': 'Real Estate', 'XLU': 'Utilities'
            }
            sector_data = []

            for etf_symbol, sector_name in sector_etfs.items():
                try:
                    start = now_et() - timedelta(days=5)
                    data = provider.get_bars(etf_symbol, '1Day', start=start)
                    if not data.empty and len(data) >= 2:
                        current_price = float(data['close'].iloc[-1])
                        start_price = float(data['close'].iloc[0])
                        perf = ((current_price - start_price) / start_price) * 100
                        sector_data.append({
                            'sector': sector_name, 'performance': perf,
                            'symbol': etf_symbol, 'current_price': current_price
                        })
                except:
                    continue

            if sector_data:
                sector_data.sort(key=lambda x: x['performance'], reverse=True)
                sectors = [item['sector'] for item in sector_data]
                performance = [item['performance'] for item in sector_data]

                import plotly.express as px
                fig = px.bar(
                    x=sectors, y=performance,
                    title="Sector Performance - Day (% Change)",
                    color=performance, color_continuous_scale="RdYlGn",
                    text=[f"{p:+.2f}%" for p in performance]
                )
                fig.update_layout(
                    template="plotly_white", height=400,
                    xaxis={'categoryorder': 'total descending'},
                    yaxis_title="Performance (%)"
                )
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

    def render_crypto_overview(self):
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

                            start_date = now_et() - timedelta(days=5)
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
                start_date = now_et() - timedelta(days=30)
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

    def render_portfolio_allocation(self):
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

    def render_portfolio_metrics(self):
        """Render performance metrics using real portfolio history"""
        try:
            account_info, _ = self.get_account_info()
            if not account_info:
                st.info("Performance data unavailable")
                return

            portfolio_value = float(account_info.get('portfolio_value', 0))
            equity = float(account_info.get('equity', 0))
            last_equity = float(account_info.get('last_equity', equity))
            
            day_pl = equity - last_equity
            day_pl_pct = (day_pl / last_equity * 100) if last_equity > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Day P&L", f"${day_pl:+,.2f}")
            with col2:
                st.metric("Day Return", f"{day_pl_pct:+.2f}%")
            with col3:
                st.metric("Portfolio Value", f"${portfolio_value:,.2f}")

            try:
                provider = AlpacaDataProvider()
                portfolio_history = provider.get_portfolio_history()
                
                if portfolio_history and 'error' not in portfolio_history:
                    equity_values = portfolio_history.get('equity', [])
                    timestamps = portfolio_history.get('timestamp', [])
                    
                    if equity_values and timestamps:
                        start_idx = 0
                        for i, val in enumerate(equity_values):
                            if val > 0:
                                start_idx = i
                                break
                        
                        filtered_equity = equity_values[start_idx:]
                        filtered_timestamps = timestamps[start_idx:]
                        
                        if filtered_equity and filtered_timestamps:
                            if isinstance(filtered_timestamps[0], (int, float)):
                                from datetime import datetime
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

    def render_backtest_analysis(self, results):
        """Display backtest results with comprehensive analysis"""
        st.subheader("📈 Backtest Results Analysis")

        if results and isinstance(results, dict):
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

            if 'trades_history' in results:
                trades_df = results['trades_history']
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                    st.write("**📝 Recent Trades**")
                    display_trades = trades_df.tail(10) if len(trades_df) > 10 else trades_df
                    st.dataframe(display_trades, use_container_width=True)

                    if st.button("💾 Download Full Trade History"):
                        csv = trades_df.to_csv(index=False)
                        from datetime import datetime
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
        else:
            st.error("Invalid backtest results format")
    
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

    @staticmethod
    def render_watchlist_interface():
        """Render watchlist management interface"""
        if 'watchlist_manager' in st.session_state:
            wm = st.session_state.watchlist_manager

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

            watchlist = wm.get_watchlist()
            if watchlist:
                st.subheader(f"👁️ Watchlist ({len(watchlist)} symbols)")
                
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

                watchlist_data = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Use the data provider directly instead of BaseDashboard
                from src.data import AlpacaDataProvider
                from src.utils.timezone_utils import now_et
                from datetime import timedelta
                
                provider = AlpacaDataProvider()
                
                for i, symbol in enumerate(watchlist):
                    status_text.text(f"Loading data for {symbol}...")
                    progress_bar.progress((i + 1) / len(watchlist))
                    
                    try:
                        data = provider.get_bars(symbol, '1Day', start=now_et() - timedelta(days=2))
                        if data is not None and not data.empty:
                            current = data['close'].iloc[-1]
                            prev = data['close'].iloc[-2] if len(data) > 1 else current
                            change = current - prev
                            change_pct = (change / prev * 100) if prev != 0 else 0

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
                    except Exception:
                        watchlist_data.append({
                            'Symbol': symbol,
                            'Current Price': "N/A",
                            'Change ($)': "N/A", 
                            'Change (%)': "N/A",
                            'Status': "❌"
                        })

                progress_bar.empty()
                status_text.empty()

                if watchlist_data:
                    df = pd.DataFrame(watchlist_data)
                    st.dataframe(df, use_container_width=True)
                    
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

    @staticmethod
    def render_trading_interface():
        """Render quick trading interface"""
        st.subheader("🚀 Quick Trade")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Order Details**")
            symbol = st.text_input("Symbol", value="AAPL", key="trade_symbol").upper()
            side = st.selectbox("Side", ["buy", "sell"], key="trade_side")
            quantity = st.number_input("Quantity", min_value=1, value=100, key="trade_quantity")
            order_type = st.selectbox("Order Type", ["market", "limit", "stop"], key="trade_order_type")

            limit_price = None
            stop_price = None
            if order_type == "limit":
                limit_price = st.number_input("Limit Price", value=150.0, step=0.01, key="trade_limit_price")
            elif order_type == "stop":
                stop_price = st.number_input("Stop Price", value=150.0, step=0.01, key="trade_stop_price")

            time_in_force = st.selectbox("Time in Force", ["day", "gtc", "ioc", "fok"], key="trade_time_in_force")

        with col2:
            st.write("**Order Preview**")
            if symbol:
                # Use data provider directly instead of BaseDashboard
                from src.data import AlpacaDataProvider
                from src.utils.timezone_utils import now_et
                from datetime import timedelta
                
                try:
                    provider = AlpacaDataProvider()
                    data = provider.get_bars(symbol, '1Day', start=now_et() - timedelta(days=1))
                    if data is not None and not data.empty:
                        current_price = data['close'].iloc[-1]
                        st.metric("Current Price", f"${current_price:.2f}")

                        estimated_cost = current_price * quantity
                        st.metric("Estimated Cost", f"${estimated_cost:,.2f}")
                    else:
                        st.warning(f"Unable to get current price for {symbol}")
                except Exception:
                    st.warning(f"Unable to get current price for {symbol}")

            if st.button("Submit Order", type="primary"):
                if not symbol:
                    st.error("Symbol is required.")
                elif order_type == "limit" and limit_price is None:
                    st.error("Limit price is required for limit orders.")
                elif order_type == "stop" and stop_price is None:
                    st.error("Stop price is required for stop orders.")
                elif 'order_manager' not in st.session_state:
                    st.error("Order manager not initialized.")
                else:
                    order_manager = st.session_state.order_manager
                    result = order_manager.place_order(
                        symbol=symbol,
                        qty=int(quantity),
                        side=side,
                        order_type=order_type,
                        time_in_force=time_in_force,
                        limit_price=limit_price,
                        stop_price=stop_price
                    )
                    if result and 'error' in result:
                        st.error(f"Order failed: {result['error']}")
                    else:
                        st.success(f"Order submitted: {result.get('id', 'unknown')}")

    @staticmethod
    def render_orders_table():
        """Render active orders view"""
        st.subheader("📋 Active Orders")

        if 'order_manager' in st.session_state:
            orders = st.session_state.order_manager.get_orders()
            if orders:
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
