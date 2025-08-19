import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Optional

from config import Config
from src.data import AlpacaDataProvider, CryptoDataProvider, NewsDataProvider
from src.trade import TradingEngine, Portfolio
from src.strategy import MomentumStrategy
from src.analysis import TechnicalAnalysis, FinancialMetrics

class Dashboard:
    def __init__(self):
        self.setup_page_config()
        self.initialize_components()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="🌍 Gauss World Trader Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_components(self):
        try:
            self.data_provider = AlpacaDataProvider()
            self.trading_engine = TradingEngine(paper_trading=True)
            self.crypto_provider = CryptoDataProvider()
            self.news_provider = NewsDataProvider()
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            self.data_provider = None
            self.trading_engine = None
    
    def sidebar_controls(self):
        st.sidebar.title("Trading System Controls")
        
        # Symbol selection
        symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()
        
        # Timeframe selection
        timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day", "1Week"]
        )
        
        # Date range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            days_back = st.number_input("Days Back", min_value=1, max_value=365, value=30)
        
        # Trading controls
        st.sidebar.subheader("Trading Controls")
        action = st.sidebar.selectbox("Action", ["BUY", "SELL"])
        quantity = st.sidebar.number_input("Quantity", min_value=1, value=100)
        order_type = st.sidebar.selectbox("Order Type", ["Market", "Limit"])
        
        if order_type == "Limit":
            limit_price = st.sidebar.number_input("Limit Price", min_value=0.01, step=0.01)
        
        # Execute trade button
        if st.sidebar.button("Execute Trade"):
            if self.trading_engine:
                try:
                    if order_type == "Market":
                        result = self.trading_engine.place_market_order(symbol, quantity, action.lower())
                    else:
                        result = self.trading_engine.place_limit_order(symbol, quantity, limit_price, action.lower())
                    
                    st.sidebar.success(f"Order placed: {result['id']}")
                except Exception as e:
                    st.sidebar.error(f"Error placing order: {e}")
        
        return symbol, timeframe, days_back
    
    def display_account_info(self):
        if not self.trading_engine:
            st.error("Trading engine not available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            account_info = self.trading_engine.get_account_info()
            
            with col1:
                st.metric(
                    label="Portfolio Value",
                    value=f"${account_info.get('portfolio_value', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    label="Buying Power",
                    value=f"${account_info.get('buying_power', 0):,.2f}"
                )
            
            with col3:
                st.metric(
                    label="Cash",
                    value=f"${account_info.get('cash', 0):,.2f}"
                )
            
            with col4:
                st.metric(
                    label="Day Trades",
                    value=account_info.get('day_trade_count', 0)
                )
        
        except Exception as e:
            st.error(f"Error fetching account info: {e}")
    
    def display_positions(self):
        if not self.trading_engine:
            return
        
        try:
            positions = self.trading_engine.get_current_positions()
            
            if positions:
                df = pd.DataFrame(positions)
                st.subheader("Current Positions")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No current positions")
        
        except Exception as e:
            st.error(f"Error fetching positions: {e}")
    
    def display_price_chart(self, symbol: str, timeframe: str, days_back: int):
        if not self.data_provider:
            st.error("Data provider not available")
            return
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            data = self.data_provider.get_bars(symbol, timeframe, start_date, end_date)
            
            if data.empty:
                st.warning(f"No data available for {symbol}")
                return
            
            # Create candlestick chart
            fig = go.Figure(data=go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol
            ))
            
            # Add volume subplot
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['volume'],
                yaxis='y2',
                name='Volume',
                fill='tozeroy',
                opacity=0.3
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Price Chart ({timeframe})',
                yaxis_title='Price ($)',
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right'
                ),
                xaxis_title='Date',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            return data
        
        except Exception as e:
            st.error(f"Error creating price chart: {e}")
            return None
    
    def display_technical_indicators(self, data: pd.DataFrame, symbol: str):
        if data is None or data.empty:
            return
        
        ta = TechnicalAnalysis()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Technical Indicators")
            
            # RSI
            rsi = ta.rsi(data['close'])
            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
            st.metric("RSI (14)", f"{current_rsi:.2f}")
            
            # Moving Averages
            sma_20 = ta.sma(data['close'], 20)
            sma_50 = ta.sma(data['close'], 50)
            
            if not sma_20.empty:
                st.metric("SMA 20", f"${sma_20.iloc[-1]:.2f}")
            if not sma_50.empty:
                st.metric("SMA 50", f"${sma_50.iloc[-1]:.2f}")
            
            # MACD
            macd, signal, histogram = ta.macd(data['close'])
            if not macd.empty:
                st.metric("MACD", f"{macd.iloc[-1]:.4f}")
        
        with col2:
            st.subheader("Trend Analysis")
            
            trend_info = ta.trend_analysis(data)
            st.write("**Short-term trend:**", trend_info['short_term_trend'])
            st.write("**Medium-term trend:**", trend_info['medium_term_trend'])
            st.write("**Long-term trend:**", trend_info['long_term_trend'])
            
            # Support/Resistance
            support_resistance = ta.calculate_support_resistance(data)
            st.write("**Resistance levels:**", support_resistance['resistance'][:3])
            st.write("**Support levels:**", support_resistance['support'][:3])
    
    def display_strategy_signals(self, symbol: str, data: pd.DataFrame):
        if data is None or data.empty:
            return
        
        st.subheader("Strategy Signals")
        
        try:
            strategy = MomentumStrategy()
            portfolio = Portfolio()
            
            # Get current data
            current_data = {
                symbol: {
                    'open': data['open'].iloc[-1],
                    'high': data['high'].iloc[-1],
                    'low': data['low'].iloc[-1],
                    'close': data['close'].iloc[-1],
                    'volume': data['volume'].iloc[-1]
                }
            }
            
            current_prices = {symbol: data['close'].iloc[-1]}
            historical_data = {symbol: data}
            
            signals = strategy.generate_signals(
                current_date=datetime.now(),
                current_prices=current_prices,
                current_data=current_data,
                historical_data=historical_data,
                portfolio=portfolio
            )
            
            if signals:
                for signal in signals:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Action", signal['action'])
                    with col2:
                        st.metric("Quantity", signal['quantity'])
                    with col3:
                        st.metric("Price", f"${signal['price']:.2f}")
                    with col4:
                        st.metric("Confidence", f"{signal['confidence']:.1%}")
                    
                    st.write("**Reason:**", signal['reason'])
            else:
                st.info("No signals generated")
        
        except Exception as e:
            st.error(f"Error generating signals: {e}")
    
    def display_news_sentiment(self, symbol: str):
        if not self.news_provider:
            return
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Company News")
                news = self.news_provider.get_company_news(symbol)
                
                for article in news[:3]:  # Show top 3
                    st.write(f"**{article['headline']}**")
                    st.write(f"Source: {article['source']}")
                    st.write(f"[Read more]({article['url']})")
                    st.write("---")
            
            with col2:
                st.subheader("News Sentiment")
                sentiment = self.news_provider.get_news_sentiment(symbol)
                
                if sentiment:
                    st.metric("Bullish %", f"{sentiment.get('sentiment_bullish_percent', 0):.1f}%")
                    st.metric("Bearish %", f"{sentiment.get('sentiment_bearish_percent', 0):.1f}%")
                    st.metric("News Score", f"{sentiment.get('company_news_score', 0):.2f}")
        
        except Exception as e:
            st.error(f"Error fetching news: {e}")
    
    def display_crypto_info(self):
        if not self.crypto_provider:
            return
        
        try:
            st.subheader("Cryptocurrency Information")
            
            btc_data = self.crypto_provider.get_bitcoin_price()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Bitcoin (USD)", f"${btc_data['price_usd']:,.2f}")
            with col2:
                st.metric("Bitcoin (EUR)", f"€{btc_data['price_eur']:,.2f}")
            with col3:
                st.metric("Bitcoin (GBP)", f"£{btc_data['price_gbp']:,.2f}")
        
        except Exception as e:
            st.error(f"Error fetching crypto data: {e}")
    
    def run(self):
        st.title("🌍 Gauss World Trader Dashboard")
        st.markdown("**Real-time Trading System • Python 3.12 Optimized • Named after Carl Friedrich Gauss**")
        
        # Time information
        current_time = datetime.now()
        st.markdown(f"**📅 Dashboard Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.warning("⏰ **Note:** Alpaca free tier has 15-minute delayed data")
        
        # Sidebar controls
        symbol, timeframe, days_back = self.sidebar_controls()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Charts & Analysis", "Trading", "News & Sentiment", "Crypto"
        ])
        
        with tab1:
            st.header("Account Overview")
            self.display_account_info()
            st.write("---")
            self.display_positions()
        
        with tab2:
            st.header("Technical Analysis")
            data = self.display_price_chart(symbol, timeframe, days_back)
            if data is not None:
                self.display_technical_indicators(data, symbol)
        
        with tab3:
            st.header("Strategy & Trading")
            if 'data' in locals() and data is not None:
                self.display_strategy_signals(symbol, data)
        
        with tab4:
            st.header("News & Sentiment")
            self.display_news_sentiment(symbol)
        
        with tab5:
            st.header("Cryptocurrency")
            self.display_crypto_info()

def main():
    dashboard = Dashboard()
    dashboard.run()

if __name__ == '__main__':
    main()