"""
Market Views Mixin - Market overview, indices, sectors, and cryptocurrency views.
"""

from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data import AlpacaDataProvider
from src.utils.timezone_utils import now_et


class MarketViewsMixin:
    """Mixin providing market overview rendering methods."""

    def render_market_overview_tab(self):
        """Market Overview: Index, VIX, Market Sentiment, Sector Performance, Crypto"""
        st.header("📊 Market Overview")
        st.divider()
        self.render_standard_market_indices()
        st.divider()
        market_tabs = st.tabs(
            ["📊 VXX & Sentiment", "🏢 Sectors", "📅 Economic Calendar", "₿ Cryptocurrency"]
        )
        with market_tabs[0]:
            self.render_volatility_analysis()
        with market_tabs[1]:
            self.render_sector_analysis()
        with market_tabs[2]:
            from src.ui.dashboard_utils import render_economic_data
            render_economic_data()
        with market_tabs[3]:
            self.render_crypto_overview()

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
                            price_fmt = (
                                f"{current_price:.2f}"
                                if symbol == 'VXX'
                                else f"{current_price:,.2f}"
                            )
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
        st.subheader("VXX: iPath S&P 500 VIX ST Futures ETN")
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
                        sentiment_color, sentiment_label = "red", "Fearful"
                    elif current_vxx > 25:
                        sentiment_color, sentiment_label = "yellow", "Neutral"
                    else:
                        sentiment_color, sentiment_label = "green", "Greedy"
                    vxx_trend = "Rising" if current_vxx > vxx_30_avg else "Falling"
                    st.metric("Current VXX", f"${current_vxx:.2f}", f"30d avg: ${vxx_30_avg:.2f}")
                    st.write(f"**Market Mood:** {sentiment_label}")
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
        st.subheader("Sector Performance")
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
                except Exception:
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
                st.write("**Sector Performance in Details**")
                df = pd.DataFrame([{
                    'Sector': item['sector'], 'ETF Symbol': item['symbol'],
                    'Current Price': f"${item['current_price']:.2f}",
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
        st.subheader("Cryptocurrency")
        try:
            provider = AlpacaDataProvider()
            crypto_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
            crypto_names = {
                'BTC/USD': 'Bitcoin', 'ETH/USD': 'Ethereum',
                'LTC/USD': 'Litecoin', 'BCH/USD': 'Bitcoin Cash'
            }
            cols = st.columns(len(crypto_symbols))
            for i, symbol in enumerate(crypto_symbols):
                with cols[i]:
                    try:
                        quote = provider.get_crypto_latest_quote(symbol)
                        if 'error' not in quote:
                            bid_price = float(quote.get('bid_price', 0))
                            ask_price = float(quote.get('ask_price', 0))
                            current_price = (
                                (bid_price + ask_price) / 2
                                if bid_price and ask_price
                                else bid_price or ask_price
                            )
                            start_date = now_et() - timedelta(days=5)
                            hist_data = provider.get_bars(symbol, '1Day', start=start_date)
                            if not hist_data.empty and len(hist_data) > 1:
                                prev_close = float(hist_data['close'].iloc[-2])
                                change = current_price - prev_close
                                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                                st.metric(
                                    crypto_names.get(symbol, symbol),
                                    f"${current_price:,.2f}",
                                    f"${change:+,.2f} ({change_pct:+.2f}%)"
                                )
                            else:
                                st.metric(crypto_names.get(symbol, symbol), f"${current_price:,.2f}")
                        else:
                            st.metric(crypto_names.get(symbol, symbol), "N/A", "Error loading")
                    except Exception:
                        st.metric(crypto_names.get(symbol, symbol), "N/A", "Error")
            st.divider()
            st.write("**Bitcoin Detailed Analysis**")
            btc_quote = provider.get_crypto_latest_quote('BTC/USD')
            if 'error' not in btc_quote:
                col1, col2, col3, col4 = st.columns(4)
                bid_price = float(btc_quote.get('bid_price', 0))
                ask_price = float(btc_quote.get('ask_price', 0))
                spread = ask_price - bid_price if ask_price and bid_price else 0
                with col1:
                    st.metric("Bid Price", f"${bid_price:,.2f}")
                with col2:
                    st.metric("Ask Price", f"${ask_price:,.2f}")
                with col3:
                    st.metric("Bid-Ask Spread", f"${spread:.2f}")
                with col4:
                    ts = btc_quote.get('timestamp')
                    st.metric("Last Updated", ts.strftime("%H:%M:%S") if ts else "N/A")
                st.write("**Bitcoin Price Chart (30 Days)**")
                start_date = now_et() - timedelta(days=30)
                btc_data = provider.get_bars('BTC/USD', '1Day', start=start_date)
                if not btc_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=btc_data.index, open=btc_data['open'], high=btc_data['high'],
                        low=btc_data['low'], close=btc_data['close'], name='BTC/USD'
                    ))
                    sma_20 = btc_data['close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(
                        x=btc_data.index, y=sma_20, mode='lines', name='20-day SMA',
                        line=dict(color='orange', width=1)
                    ))
                    fig.update_layout(
                        title="Bitcoin (BTC/USD) - 30 Day Chart", yaxis_title="Price (USD)",
                        xaxis_title="Date", height=400, showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    current_price = float(btc_data['close'].iloc[-1])
                    high_30d = float(btc_data['high'].max())
                    low_30d = float(btc_data['low'].min())
                    volatility = btc_data['close'].pct_change().std() * np.sqrt(365) * 100
                    range_pos = (
                        ((current_price - low_30d) / (high_30d - low_30d)) * 100
                        if (high_30d - low_30d) > 0 else 0
                    )
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("30D High", f"${high_30d:,.2f}")
                    with col2:
                        st.metric("30D Low", f"${low_30d:,.2f}")
                    with col3:
                        st.metric("30D Range Position", f"{range_pos:.1f}%")
                    with col4:
                        st.metric("Annualized Volatility", f"{volatility:.1f}%")
                else:
                    st.error("Unable to load Bitcoin historical data")
            else:
                st.error(f"Error loading Bitcoin data: {btc_quote.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error loading cryptocurrency data: {e}")
