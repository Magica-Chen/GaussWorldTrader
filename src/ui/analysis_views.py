"""
Analysis Views Mixin - Live analysis, streams, and news views.
"""

from datetime import timedelta
import streamlit as st
import plotly.graph_objects as go

from src.data import AlpacaDataProvider
from src.utils.timezone_utils import now_et


class AnalysisViewsMixin:
    """Mixin providing live analysis and news rendering methods."""

    def render_live_analysis_tab(self):
        """Live Analysis: Symbol Analysis & Market Stream"""
        st.header("🔍 Live Analysis")
        analysis_tabs = st.tabs(["📊 Symbol Analysis", "📡 Market Stream"])
        with analysis_tabs[0]:
            self.render_symbol_analysis()
        with analysis_tabs[1]:
            self.render_market_stream()

    def render_symbol_analysis(self):
        """Render symbol analysis with AI insights"""
        st.subheader("📊 Symbol Analysis")
        col1, col2 = st.columns([3, 1])
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols("stock")
            symbol = st.selectbox(
                "Select Symbol", options=default_symbols if default_symbols else ["AAPL"],
                key="analysis_symbol"
            )
        with col2:
            analyze_btn = st.button("Analyze", type="primary")

        if analyze_btn:
            self._perform_symbol_analysis(symbol)

    def _perform_symbol_analysis(self, symbol: str):
        """Perform and display symbol analysis"""
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                provider = AlpacaDataProvider()
                end_date = now_et()
                start_date = end_date - timedelta(days=60)
                data = provider.get_bars(symbol, "1Day", start=start_date)

                if data.empty:
                    st.error(f"No data available for {symbol}")
                    return

                current_price = float(data['close'].iloc[-1])
                prev_price = float(data['close'].iloc[-2])
                change = current_price - prev_price
                change_pct = (change / prev_price * 100) if prev_price else 0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
                with col2:
                    st.metric("Day Change", f"{change_pct:+.2f}%")
                with col3:
                    st.metric("52W High", f"${data['high'].max():.2f}")
                with col4:
                    st.metric("52W Low", f"${data['low'].min():.2f}")

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index, open=data['open'], high=data['high'],
                    low=data['low'], close=data['close'], name=symbol
                ))
                sma_20 = data['close'].rolling(window=20).mean()
                sma_50 = data['close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=sma_20, mode='lines', name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=data.index, y=sma_50, mode='lines', name='SMA 50',
                    line=dict(color='purple', width=1)
                ))
                fig.update_layout(title=f"{symbol} Price Chart", height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                self._display_ai_analysis(symbol, data)
            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")

    def _display_ai_analysis(self, symbol: str, data):
        """Display AI-powered analysis"""
        st.write("**AI Analysis**")
        current_price = float(data['close'].iloc[-1])
        sma_20 = float(data['close'].rolling(window=20).mean().iloc[-1])
        sma_50 = float(data['close'].rolling(window=50).mean().iloc[-1])
        rsi = self._calculate_rsi(data['close'])

        if current_price > sma_20 > sma_50:
            trend = "Bullish"
            trend_color = "green"
        elif current_price < sma_20 < sma_50:
            trend = "Bearish"
            trend_color = "red"
        else:
            trend = "Neutral"
            trend_color = "yellow"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend", trend)
        with col2:
            st.metric("RSI (14)", f"{rsi:.1f}")
        with col3:
            if rsi > 70:
                signal = "Overbought"
            elif rsi < 30:
                signal = "Oversold"
            else:
                signal = "Neutral"
            st.metric("RSI Signal", signal)

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50

    def render_market_stream(self):
        """Render real-time market data stream"""
        st.subheader("📡 Market Stream")
        col1, col2 = st.columns(2)
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols("stock")
            stream_symbol = st.selectbox(
                "Symbol", options=default_symbols if default_symbols else ["AAPL"],
                key="stream_symbol"
            )
        with col2:
            stream_type = st.selectbox("Data Type", ["Trades", "Quotes", "Bars"], key="stream_type")

        if st.button("Refresh Data"):
            self._display_stream_data(stream_symbol, stream_type.lower())
        else:
            self._display_stream_data(stream_symbol, stream_type.lower())

    def _display_stream_data(self, symbol: str, stream_type: str):
        """Display stream data based on type"""
        try:
            provider = AlpacaDataProvider()
            if stream_type == "trades":
                trades = provider.get_latest_trade(symbol)
                if trades and 'error' not in trades:
                    st.write("**Latest Trade**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Price", f"${trades.get('price', 0):.2f}")
                    with col2:
                        st.metric("Size", trades.get('size', 0))
                    with col3:
                        st.metric("Exchange", trades.get('exchange', 'N/A'))
            elif stream_type == "quotes":
                quote = provider.get_latest_quote(symbol)
                if quote and 'error' not in quote:
                    st.write("**Latest Quote**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Bid", f"${quote.get('bid_price', 0):.2f}")
                    with col2:
                        st.metric("Bid Size", quote.get('bid_size', 0))
                    with col3:
                        st.metric("Ask", f"${quote.get('ask_price', 0):.2f}")
                    with col4:
                        st.metric("Ask Size", quote.get('ask_size', 0))
            else:
                start_date = now_et() - timedelta(days=5)
                bars = provider.get_bars(symbol, "1Hour", start=start_date)
                if not bars.empty:
                    st.write("**Recent Bars**")
                    st.dataframe(bars.tail(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading stream data: {e}")

    def render_news_report_tab(self):
        """News Report: Market News & Analysis"""
        st.header("📰 News & Analysis")
        news_tabs = st.tabs(["📰 Market News", "🤖 AI Analysis Report"])
        with news_tabs[0]:
            self.render_market_news()
        with news_tabs[1]:
            self.render_ai_report()

    def render_market_news(self):
        """Render market news"""
        st.subheader("Market News")
        try:
            if 'finnhub_provider' in st.session_state:
                finnhub = st.session_state.finnhub_provider
                from src.ui.dashboard_utils import get_default_symbols
                default_symbols = get_default_symbols("stock")
                news_symbol = st.selectbox(
                    "Select Symbol for News",
                    options=default_symbols if default_symbols else ["AAPL"],
                    key="news_symbol"
                )
                news = finnhub.get_company_news(news_symbol)
                if news:
                    for article in news[:10]:
                        with st.expander(article.get('headline', 'No title')):
                            st.write(f"**Source:** {article.get('source', 'Unknown')}")
                            st.write(f"**Summary:** {article.get('summary', 'No summary')}")
                            if article.get('url'):
                                st.markdown(f"[Read more]({article['url']})")
                else:
                    st.info("No recent news available for this symbol.")
            else:
                st.info("Finnhub provider not configured. News unavailable.")
        except Exception as e:
            st.error(f"Error loading news: {e}")

    def render_ai_report(self):
        """Render AI-generated market report"""
        st.subheader("AI Analysis Report")
        col1, col2 = st.columns(2)
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols("stock")
            report_symbol = st.selectbox(
                "Symbol", options=default_symbols if default_symbols else ["AAPL"],
                key="report_symbol"
            )
        with col2:
            report_type = st.selectbox(
                "Report Type", ["Technical Analysis", "Fundamental Analysis", "Full Report"],
                key="report_type"
            )

        if st.button("Generate Report", type="primary"):
            self._generate_ai_report(report_symbol, report_type)

    def _generate_ai_report(self, symbol: str, report_type: str):
        """Generate AI analysis report"""
        with st.spinner("Generating report..."):
            try:
                if 'agent_manager' in st.session_state:
                    agent = st.session_state.agent_manager
                    if report_type == "Technical Analysis":
                        report = agent.get_technical_analysis(symbol)
                    elif report_type == "Fundamental Analysis":
                        report = agent.get_fundamental_analysis(symbol)
                    else:
                        report = agent.get_full_analysis(symbol)

                    if report:
                        st.write("**Analysis Report**")
                        st.markdown(report)
                    else:
                        st.info("Unable to generate report. Please check API configuration.")
                else:
                    provider = AlpacaDataProvider()
                    end_date = now_et()
                    start_date = end_date - timedelta(days=30)
                    data = provider.get_bars(symbol, "1Day", start=start_date)

                    if not data.empty:
                        current_price = float(data['close'].iloc[-1])
                        sma_20 = float(data['close'].rolling(window=20).mean().iloc[-1])
                        sma_50 = float(data['close'].rolling(window=min(50, len(data))).mean().iloc[-1])
                        volatility = data['close'].pct_change().std() * 100

                        st.write(f"**{report_type} for {symbol}**")
                        st.write(f"- Current Price: ${current_price:.2f}")
                        st.write(f"- 20-day SMA: ${sma_20:.2f}")
                        st.write(f"- Daily Volatility: {volatility:.2f}%")
                        if current_price > sma_20:
                            st.write("- Trend: Price above 20-day moving average (Bullish)")
                        else:
                            st.write("- Trend: Price below 20-day moving average (Bearish)")
                    else:
                        st.info("No data available for analysis.")
            except Exception as e:
                st.error(f"Error generating report: {e}")
