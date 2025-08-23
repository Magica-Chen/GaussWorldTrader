#!/usr/bin/env python3
"""
Simple Streamlit Dashboard for the Trading System
Enhanced with crypto, news, and technical analysis features
Designed to work with existing components
"""

import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional
from src.utils.timezone_utils import EASTERN, now_et, get_market_status

# Add project to path
sys.path.insert(0, '../..')  # Go up two levels to project root

def get_local_time():
    """Get current time in local timezone for display purposes"""
    return datetime.now()

# Page configuration
st.set_page_config(
    page_title="🌍 Gauss World Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=900)  # Cache for 15 minutes
def load_data(symbol, days=30):
    """Load market data with VIP/free tier detection"""
    try:
        from src.data import AlpacaDataProvider
        provider = AlpacaDataProvider()
        vip = provider.vip
        using_iex = provider.using_iex
        
        current_time = now_et()
        
        # Use centralized market status detection
        market_status = get_market_status(current_time)
        is_weekend = current_time.weekday() >= 5
        is_market_open = market_status == 'open'
        
        # Determine data context based on market status and account tier
        if market_status == 'closed':
            if is_weekend:
                days_to_friday = current_time.weekday() - 4
                market_close_date = current_time - timedelta(days=days_to_friday)
                market_close_date = market_close_date.replace(hour=16, minute=0, second=0, microsecond=0)
                end_date = market_close_date
                data_context = "Weekend - showing data through Friday's close"
            else:
                # Weekday closed hours
                market_close_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
                if current_time.hour < 16:
                    market_close_date = (current_time - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
                end_date = market_close_date
                data_context = f"Market closed - showing data through market close"
        elif market_status in ['pre-market', 'post-market']:
            end_date = current_time
            data_context = f"{market_status.title()} - showing today's data"
        else:
            # Market open
            end_date = current_time
            data_context = f"Market {market_status} - showing live data"
        
        # Add data source context
        if vip:
            data_context += " (SIP feed)"
        elif using_iex:
            data_context += " (IEX + SIP feeds)"
        else:
            data_context += " (Historical data)"
        
        start_date = end_date - timedelta(days=days)
        
        # Fetch data with fallback strategy
        try:
            data = provider.get_bars(symbol, '1Day', start_date, end_date)
            if data is not None and not data.empty:
                return data, data_context
        except Exception:
            pass  # Fall through to fallback
        
        # Fallback: Use current data if primary strategy fails
        end_date = current_time
        start_date = end_date - timedelta(days=days)
        data = provider.get_bars(symbol, '1Day', start_date, end_date)
        
        if data is not None and not data.empty:
            fallback_context = "Using historical data"
            if using_iex:
                fallback_context += " (IEX + SIP feeds)"
            return data, fallback_context
        else:
            return None, "No data available"
            
    except Exception as e:
        error_msg = str(e)
        if "subscription does not permit" in error_msg.lower():
            error_msg = "Free tier limitation: Cannot access recent SIP data. Showing historical data."
        return None, error_msg

@st.cache_data(ttl=300)
def get_account_info():
    """Get account information with caching"""
    try:
        from src.trade import TradingEngine
        engine = TradingEngine()
        return engine.get_account_info(), None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=60)
def generate_signals(symbol, data):
    """Generate trading signals with caching"""
    try:
        from src.strategy import MomentumStrategy
        from src.trade import Portfolio
        
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
def get_technical_indicators(data):
    """Get technical indicators with caching"""
    try:
        from src.analysis import TechnicalAnalysis
        
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
        st.error(f"Error calculating technical indicators: {e}")
        return None

@st.cache_data(ttl=600)
def get_crypto_data():
    """Get cryptocurrency data with caching"""
    try:
        from src.data import CryptoDataProvider
        provider = CryptoDataProvider()
        
        btc_data = provider.get_bitcoin_price()
        return btc_data, None
        
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=1200)
def get_news_sentiment(symbol):
    """Get news and sentiment data with caching"""
    try:
        from src.data import NewsDataProvider
        provider = NewsDataProvider()
        
        news = provider.get_company_news(symbol)
        sentiment = provider.get_news_sentiment(symbol)
        
        return news, sentiment, None
        
    except Exception as e:
        return [], None, str(e)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_backtest(symbols, days_back=365, initial_cash=100000, strategy_type="Momentum"):
    """Run backtest with caching and strategy selection"""
    try:
        from src.trade import Backtester
        from src.strategy import MomentumStrategy
        from src.data import AlpacaDataProvider
        
        provider = AlpacaDataProvider()
        backtester = Backtester(initial_cash=initial_cash, commission=0.01)
        
        # Load data for all symbols
        end_date = now_et()  # Use current date
        start_date = end_date - timedelta(days=days_back)
        
        for symbol in symbols:
            data = provider.get_bars(symbol, '1Day', start_date, end_date)
            if not data.empty:
                backtester.add_data(symbol, data)
        
        # Create strategy based on selection
        if strategy_type == "Momentum":
            strategy = MomentumStrategy()
        elif strategy_type == "Mean Reversion":
            # Use MomentumStrategy with modified parameters for mean reversion
            strategy = MomentumStrategy({
                'lookback_period': 10,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'position_size_pct': 0.1,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.15
            })
        elif strategy_type == "Trend Following":
            # Trend following with longer lookback
            strategy = MomentumStrategy({
                'lookback_period': 50,
                'rsi_period': 21,
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'position_size_pct': 0.15,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.25
            })
        elif strategy_type == "RSI Oversold/Overbought":
            # Pure RSI-based strategy
            strategy = MomentumStrategy({
                'lookback_period': 5,
                'rsi_period': 14,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'position_size_pct': 0.12,
                'stop_loss_pct': 0.06,
                'take_profit_pct': 0.18
            })
        else:
            strategy = MomentumStrategy()  # Default fallback
        
        def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
            return strategy.generate_signals(
                current_date, current_prices, current_data, historical_data, portfolio
            )
        
        # Run backtest (skip first 50 days for indicator warmup)
        results = backtester.run_backtest(
            strategy_func,
            start_date=start_date + timedelta(days=50),
            end_date=end_date,
            symbols=symbols
        )
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def generate_dashboard_transaction_log(results, symbols):
    """Generate transaction log for dashboard download"""
    try:
        if not results or 'trades_history' not in results:
            return None
        
        trades_df = results['trades_history']
        if trades_df.empty:
            return None
        
        # Enhanced transaction log
        enhanced_trades = []
        positions = {}
        trade_counter = 1
        
        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            action = trade['action'].upper()
            quantity = abs(trade['quantity'])
            price = trade.get('price', 0)
            trade_date = trade['date']
            trade_value = quantity * price
            
            # Position tracking
            position_before = positions.get(symbol, {'qty': 0, 'avg_cost': 0, 'total_cost': 0})
            
            if action == 'BUY':
                new_qty = position_before['qty'] + quantity
                new_total_cost = position_before['total_cost'] + trade_value
                new_avg_cost = new_total_cost / new_qty if new_qty > 0 else 0
                
                positions[symbol] = {
                    'qty': new_qty,
                    'avg_cost': new_avg_cost,
                    'total_cost': new_total_cost
                }
                realized_pnl = 0
                
            elif action == 'SELL':
                if position_before['qty'] >= quantity:
                    cost_basis = position_before['avg_cost'] * quantity
                    proceeds = trade_value
                    realized_pnl = proceeds - cost_basis
                    
                    new_qty = position_before['qty'] - quantity
                    new_total_cost = position_before['total_cost'] - cost_basis
                    
                    positions[symbol] = {
                        'qty': new_qty,
                        'avg_cost': position_before['avg_cost'] if new_qty > 0 else 0,
                        'total_cost': new_total_cost
                    }
                else:
                    realized_pnl = 0
            
            position_after = positions.get(symbol, {'qty': 0, 'avg_cost': 0, 'total_cost': 0})
            
            # Create enhanced trade record
            enhanced_trade = {
                'Trade_ID': trade_counter,
                'Date': trade_date.strftime('%Y-%m-%d') if hasattr(trade_date, 'strftime') else str(trade_date),
                'Symbol': symbol,
                'Action': action,
                'Quantity': quantity,
                'Price': f"{price:.4f}",
                'Trade_Value': f"{trade_value:.2f}",
                'Commission': f"{trade_value * 0.01:.2f}",
                'Net_Amount': f"{trade_value * (0.99 if action == 'BUY' else 1.01):.2f}",
                'Position_Before': position_before['qty'],
                'Position_After': position_after['qty'],
                'Avg_Cost_Basis': f"{position_after['avg_cost']:.4f}",
                'Realized_PnL': f"{realized_pnl:.2f}",
                'Strategy': 'Momentum',
                'Notes': 'Backtest transaction'
            }
            
            enhanced_trades.append(enhanced_trade)
            trade_counter += 1
        
        # Create DataFrame and save
        transactions_df = pd.DataFrame(enhanced_trades)
        timestamp = now_et().strftime('%Y%m%d_%H%M%S')
        filename = f"dashboard_transactions_{timestamp}.csv"
        
        transactions_df.to_csv(filename, index=False)
        return filename
        
    except Exception as e:
        st.error(f"Error generating transaction log: {e}")
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🌍 Gauss World Trader Dashboard")
    st.markdown("**Python 3.12 • Real-time Data • Advanced Analytics • Named after Carl Friedrich Gauss**")
    
    # Account tier and delay notice header
    current_time = now_et()
    
    try:
        # Initialize data provider to get account info
        from src.data import AlpacaDataProvider
        data_provider = AlpacaDataProvider()
        account_info = data_provider.get_account_info()
        
        vip = account_info.get('vip', False)
        using_iex = account_info.get('using_iex', False)
        account_tier = "VIP Account" if vip else "Free Tier"
        
        # Check if today is a trading day
        is_trading_day = current_time.weekday() < 5  # Monday=0, Friday=4
        
        # Create header info layout
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            local_time = get_local_time()
            st.markdown(f"**📅 Local Time:** {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            # Account tier display
            if vip:
                st.success(f"✨ {account_tier}")
            else:
                st.info(f"🆓 {account_tier}")
        
        with col3:
            # Data source notice for free tier on trading days
            if not vip and is_trading_day:
                if using_iex:
                    st.info("📊 Today's data: Real-time IEX + SIP Historical")
                else:
                    st.info("📊 Historical data only")
        
        with col4:
            # Enhanced Market Status Indicator
            is_weekend = current_time.weekday() >= 5  # Saturday=5, Sunday=6
            is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
            is_after_hours = 16 <= current_time.hour < 20  # 4:00 PM - 8:00 PM
            is_overnight = current_time.hour >= 20 or current_time.hour < 4  # 8:00 PM - 4:00 AM
            
            if is_weekend:
                st.error("🔴 Weekend")
            elif is_overnight:
                st.info("🔵 Overnight")
            elif is_pre_market:
                st.warning("🟡 Pre-Market")
            elif is_after_hours:
                st.warning("🟡 After Hours")
            else:
                st.success("🟢 Market Open")
                
    except Exception as e:
        # Fallback if subscription info not available
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            local_time = get_local_time()
            st.markdown(f"**📅 Local Time:** {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.info("🆓 Free Tier")
        
        with col3:
            is_trading_day = current_time.weekday() < 5
            if is_trading_day:
                st.info("📊 Today's data: Real-time IEX")
    
    # Sidebar
    st.sidebar.title("📊 Controls")
    
    # Symbol selection
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    
    # Time range
    days_back = st.sidebar.slider("Days of History", 7, 365, 30)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Create tabs with enhanced features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Live Analysis", "🔄 Backtesting", "💼 Account", 
        "🔄 Trading", "📰 News & Sentiment", "₿ Crypto"
    ])
    
    with tab1:
        # Live Analysis Tab (enhanced with technical indicators)
        main_analysis_tab(symbol, days_back, vip, using_iex)
    
    with tab2:
        # Backtesting Tab
        backtesting_tab()
    
    with tab3:
        # Account Tab
        account_tab()
    
    with tab4:
        # Trading Tab
        trading_tab(symbol)
    
    with tab5:
        # News & Sentiment Tab (new feature from class dashboard)
        news_sentiment_tab(symbol)
    
    with tab6:
        # Cryptocurrency Tab (new feature from class dashboard)
        crypto_tab()

def main_analysis_tab(symbol, days_back, vip, using_iex):
    """Main analysis tab content with technical indicators"""
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {symbol} Price Chart")
        
        # Load data
        with st.spinner(f"Loading {symbol} data..."):
            data, error = load_data(symbol, days_back)
        
        if error:
            if "Using" in error or "showing data through" in error or "delayed" in error:
                st.info(f"ℹ️ {error}")
            elif "previous day" in error or "older historical" in error:
                st.warning(f"⚠️ {error}")
            else:
                st.error(f"❌ Data Error: {error}")
                st.info("💡 Make sure your Alpaca API keys are configured in .env file")
                return
        
        if data is None or data.empty:
            st.warning(f"⚠️ No data found for {symbol}")
            return
        
        # Create enhanced candlestick chart with volume
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
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent data table
        st.subheader("📋 Recent Data")
        
        # Show data period information - handle potential NaT values
        try:
            data_start = data.index[0].strftime('%Y-%m-%d') if pd.notna(data.index[0]) else "N/A"
            data_end = data.index[-1].strftime('%Y-%m-%d') if pd.notna(data.index[-1]) else "N/A"
        except (AttributeError, ValueError):
            data_start = "N/A"
            data_end = "N/A"
        
        current_time = now_et()
        market_status = get_market_status(current_time)
        is_weekend = current_time.weekday() >= 5
        
        # Generate contextual message
        if is_weekend:
            context_msg = "Weekend - showing data through Friday's close"
        elif market_status == 'closed':
            if vip:
                context_msg = "Market closed - showing today's data (SIP real-time)"
            elif using_iex:
                context_msg = "Market closed - showing today's data (IEX + SIP)"
            else:
                context_msg = "Market closed - historical data"
        elif market_status == 'post-market':
            if vip:
                context_msg = "After hours - showing today's data (SIP real-time)"
            elif using_iex:
                context_msg = "After hours - showing today's data (IEX + SIP)"
            else:
                context_msg = "After hours - historical data"
        elif market_status == 'pre-market':
            context_msg = "Pre-market - showing data through previous trading day's close"
        elif market_status == 'open':
            if vip:
                context_msg = "Market open - showing today's data (SIP real-time)"
            elif using_iex:
                context_msg = "Market open - showing today's data (IEX + SIP)"
            else:
                context_msg = "Market open - historical data"
        
        st.info(f"📅 Data Period: {data_start} to {data_end} ({len(data)} trading days) | {context_msg}")
        
        recent_data = data.tail(10).copy()
        recent_data.index = recent_data.index.strftime('%Y-%m-%d')
        st.dataframe(recent_data, use_container_width=True)
    
    with col2:
        # Current metrics
        st.subheader("📊 Current Metrics")
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        st.metric(
            label="Current Price", 
            value=f"${current_price:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
        )
        
        st.metric(
            label="Volume",
            value=f"{data['volume'].iloc[-1]:,.0f}"
        )
        
        st.metric(
            label="Range (30d)",
            value=f"${data['low'].min():.2f} - ${data['high'].max():.2f}"
        )
        
        # Technical Indicators
        st.subheader("🔬 Technical Indicators")
        
        with st.spinner("Calculating indicators..."):
            indicators = get_technical_indicators(data)
        
        if indicators:
            st.metric("RSI (14)", f"{indicators['rsi']:.2f}")
            st.metric("SMA 20", f"${indicators['sma_20']:.2f}")
            st.metric("SMA 50", f"${indicators['sma_50']:.2f}")
            st.metric("MACD", f"{indicators['macd']:.4f}")
            
            # Trend Analysis
            st.subheader("📈 Trend Analysis")
            if 'trends' in indicators and indicators['trends']:
                trends = indicators['trends']
                st.write("**Short-term:**", trends.get('short_term_trend', 'N/A'))
                st.write("**Medium-term:**", trends.get('medium_term_trend', 'N/A'))
                st.write("**Long-term:**", trends.get('long_term_trend', 'N/A'))
        
        # Trading signals
        st.subheader("🧠 Trading Signals")
        
        with st.spinner("Generating signals..."):
            signals, signal_error = generate_signals(symbol, data)
        
        if signal_error:
            st.error(f"Signal Error: {signal_error}")
        elif signals:
            for signal in signals:
                signal_color = "🟢" if signal['action'].upper() == 'BUY' else "🔴"
                st.success(f"""
                {signal_color} **{signal['action'].upper()} SIGNAL**
                
                - **Symbol**: {signal['symbol']}
                - **Quantity**: {signal['quantity']} shares  
                - **Confidence**: {signal.get('confidence', 0):.1%}
                - **Reason**: {signal.get('reason', 'N/A')}
                """)
        else:
            st.info("📭 No trading signals generated")
    
    # Account information
    st.subheader("💼 Account Information")
    
    with st.spinner("Loading account data..."):
        account_data, account_error = get_account_info()
    
    if account_error:
        st.error(f"Account Error: {account_error}")
    elif account_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${account_data.get('portfolio_value', 0):,.2f}")
        
        with col2:
            st.metric("Buying Power", f"${account_data.get('buying_power', 0):,.2f}")
        
        with col3:
            st.metric("Cash", f"${account_data.get('cash', 0):,.2f}")
        
        with col4:
            st.metric("Day Trades", account_data.get('day_trade_count', 0))

def backtesting_tab():
    """Backtesting functionality tab"""
    st.subheader("🔄 Strategy Backtesting")
    st.markdown("**Test your quantitative trading strategies on historical data**")
    
    # Strategy selection and backtest parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strategy_type = st.selectbox(
            "Select Strategy",
            ["Momentum", "Mean Reversion", "Trend Following", "RSI Oversold/Overbought"],
            index=0,
            help="Choose the trading strategy to backtest"
        )
    
    with col2:
        backtest_symbols = st.multiselect(
            "Select Symbols",
            ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY"],
            default=["AAPL", "GOOGL", "MSFT"]
        )
    
    with col3:
        backtest_days = st.slider("Days to Test", 90, 730, 365)
        
    with col4:
        initial_cash = st.number_input("Initial Cash ($)", value=100000, step=10000)
    
    # Strategy description
    strategy_descriptions = {
        "Momentum": "🚀 Buys stocks with strong upward price momentum and high RSI confirmation",
        "Mean Reversion": "🔄 Buys oversold stocks (RSI < 30) expecting price reversal",
        "Trend Following": "📈 Follows long-term trends using moving average crossovers",
        "RSI Oversold/Overbought": "⚖️ Trades based on RSI overbought (>70) and oversold (<30) signals"
    }
    st.info(f"**Strategy:** {strategy_descriptions.get(strategy_type, 'Custom strategy')}")
    
    if st.button("🚀 Run Backtest", type="primary", disabled=len(backtest_symbols) == 0):
        with st.spinner(f"Running {strategy_type} strategy backtest... This may take a moment"):
            results, error = run_backtest(backtest_symbols, backtest_days, initial_cash, strategy_type)
        
        if error:
            st.error(f"❌ Backtest Error: {error}")
        elif results:
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Initial Value", f"${results['initial_value']:,.2f}")
            with col2:
                st.metric("Final Value", f"${results['final_value']:,.2f}")
            with col3:
                st.metric("Total Return", f"{results['total_return_percentage']:.2f}%")
            with col4:
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            
            # P&L Chart
            if 'portfolio_history' in results and not results['portfolio_history'].empty:
                portfolio_df = results['portfolio_history']
                
                # Create P&L chart
                fig = go.Figure()
                
                # Portfolio value line
                fig.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                # Initial value reference line
                fig.add_hline(y=results['initial_value'], 
                            line_dash="dash", 
                            line_color="gray", 
                            annotation_text="Initial Value")
                
                fig.update_layout(
                    title="📊 Portfolio Performance Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics table
                st.subheader("📋 Detailed Metrics")
                metrics_data = {
                    "Metric": [
                        "Annualized Return", "Volatility", "Sharpe Ratio", 
                        "Max Drawdown", "Total Trades", "Win Rate"
                    ],
                    "Value": [
                        f"{results.get('annualized_return_percentage', 0):.2f}%",
                        f"{results.get('volatility', 0):.2f}",
                        f"{results.get('sharpe_ratio', 0):.2f}",
                        f"{results.get('max_drawdown_percentage', 0):.2f}%",
                        f"{results.get('total_trades', 0)}",
                        f"{results.get('win_rate', 0):.1f}%"
                    ]
                }
                st.table(pd.DataFrame(metrics_data))
                
                # Transaction log download
                if st.button("📋 Generate Transaction CSV", type="secondary"):
                    csv_filename = generate_dashboard_transaction_log(results, backtest_symbols)
                    if csv_filename:
                        # Provide download link
                        with open(csv_filename, 'r') as f:
                            csv_data = f.read()
                        
                        st.download_button(
                            label="💾 Download Transaction Log",
                            data=csv_data,
                            file_name=csv_filename,
                            mime='text/csv',
                            type="primary"
                        )
                        st.success(f"✅ Transaction log generated: {csv_filename}")
    
    st.markdown("---")
    st.info("💡 **Tip**: Backtesting shows how the selected strategy would have performed on historical data. Past performance doesn't guarantee future results.")

def account_tab():
    """Account information tab"""
    st.subheader("💼 Account Overview")
    
    # Account information
    with st.spinner("Loading account data..."):
        account_data, account_error = get_account_info()
    
    if account_error:
        st.error(f"Account Error: {account_error}")
    elif account_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Portfolio Value", f"${account_data.get('portfolio_value', 0):,.2f}")
            st.metric("Cash", f"${account_data.get('cash', 0):,.2f}")
            st.metric("Day Trades", account_data.get('day_trade_count', 0))
        
        with col2:
            st.metric("Buying Power", f"${account_data.get('buying_power', 0):,.2f}")
            st.metric("Equity", f"${account_data.get('equity', 0):,.2f}")
            
            # Account status
            status = account_data.get('status', 'UNKNOWN')
            if status == 'ACTIVE':
                st.success(f"Status: ✅ {status}")
            else:
                st.warning(f"Status: ⚠️ {status}")
        
        # Show current positions
        try:
            from src.trade import TradingEngine
            engine = TradingEngine()
            positions = engine.get_current_positions()
            
            if positions:
                st.subheader("📊 Current Positions")
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("📭 No current positions")
                
        except Exception as e:
            st.error(f"Error fetching positions: {e}")
    
    # Data limitations notice
    st.markdown("---")
    current_time = now_et()
    is_weekend = current_time.weekday() >= 5
    is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
    is_after_hours = 16 <= current_time.hour < 20
    is_overnight = current_time.hour >= 20 or current_time.hour < 4
    
    if is_weekend:
        st.info("📊 **Data Notice**: Weekend - Historical data available through Friday's close. Free tier includes 1-day buffer for SIP compliance.")
    elif is_overnight:
        st.info("📊 **Data Notice**: Overnight period - Historical data available through previous day's close. Markets are closed for extended trading.")
    elif is_after_hours:
        st.info("📊 **Data Notice**: After hours - Historical data available through previous day's close. Consider upgrading for real-time after-hours data.")
    elif is_pre_market:
        st.info("📊 **Data Notice**: Pre-market - Historical data available through previous trading day's close. Consider upgrading for real-time pre-market data.")
    else:
        st.info("📊 **Data Notice**: Market open - Data delayed by 1 day for free tier SIP compliance. Consider upgrading for real-time data access.")

def trading_tab(symbol):
    """Trading panel functionality"""
    st.subheader("🔄 Trading Panel")
    st.markdown("**Execute trades directly from the dashboard**")
    
    # Trading form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trade_symbol = st.text_input("Symbol", value=symbol, key="trade_symbol").upper()
        action = st.selectbox("Action", ["BUY", "SELL"], key="trade_action")
        
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=100, step=1, key="trade_quantity")
        order_type = st.selectbox("Order Type", ["Market", "Limit"], key="trade_order_type")
        
    with col3:
        if order_type == "Limit":
            limit_price = st.number_input("Limit Price ($)", min_value=0.01, value=100.0, step=0.01, key="trade_limit_price")
        else:
            limit_price = None
        
        time_in_force = st.selectbox("Time in Force", ["GTC", "DAY", "IOC", "FOK"], key="trade_tif")
    
    # Order preview
    if trade_symbol and quantity:
        st.subheader("📋 Order Preview")
        estimated_value = quantity * (limit_price if limit_price else 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Symbol", trade_symbol)
        with col2:
            st.metric("Action", f"{action} {quantity} shares")
        with col3:
            if order_type == "Market":
                st.metric("Order Type", "Market Order")
            else:
                st.metric("Limit Price", f"${limit_price:.2f}")
        
        if order_type == "Limit":
            st.info(f"💰 Estimated Value: ${estimated_value:,.2f}")
    
    # Execute trade button
    col1, col2 = st.columns([1, 3])
    with col1:
        execute_trade = st.button("🚀 Execute Trade", type="primary", disabled=not trade_symbol or quantity <= 0)
    
    if execute_trade:
        with st.spinner("Executing trade..."):
            try:
                from src.trade import TradingEngine
                engine = TradingEngine()
                
                if order_type == "Market":
                    result = engine.place_market_order(trade_symbol, quantity, action.lower())
                else:
                    result = engine.place_limit_order(trade_symbol, quantity, limit_price, action.lower())
                
                if result and 'id' in result:
                    st.success(f"✅ Order placed successfully! Order ID: {result['id']}")
                    st.json(result)
                else:
                    st.error("❌ Failed to place order")
                    
            except Exception as e:
                st.error(f"❌ Trading Error: {str(e)}")
                st.info("💡 Make sure your Alpaca API keys are configured and account is active")
    
    # Recent orders section
    st.markdown("---")
    st.subheader("📊 Recent Orders")
    
    try:
        from src.trade import TradingEngine
        engine = TradingEngine()
        orders = engine.get_open_orders()
        
        if orders:
            orders_df = pd.DataFrame(orders)
            orders_df = orders_df[['symbol', 'side', 'qty', 'order_type', 'status', 'submitted_at']]
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.info("📭 No recent orders")
            
    except Exception as e:
        st.error(f"Error fetching orders: {str(e)}")
    
    # Trading warnings
    st.markdown("---")
    st.warning("⚠️ **Risk Warning**: Trading involves risk. Only trade with money you can afford to lose.")
    st.info("📈 **Paper Trading**: Make sure you're in paper trading mode for testing")

def news_sentiment_tab(symbol):
    """News and sentiment analysis tab"""
    st.subheader("📰 News & Sentiment Analysis")
    
    with st.spinner("Loading news and sentiment data..."):
        news, sentiment, error = get_news_sentiment(symbol)
    
    if error:
        st.error(f"Error loading news data: {error}")
        st.info("💡 Make sure your news API is configured properly")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📰 Company News - {symbol}")
        
        if news and len(news) > 0:
            for article in news[:5]:  # Show top 5
                st.write(f"**{article.get('headline', 'No headline')}**")
                st.write(f"Source: {article.get('source', 'Unknown')}")
                if 'url' in article and article['url']:
                    st.write(f"[Read more]({article['url']})")
                st.write("---")
        else:
            st.info("No news articles available")
    
    with col2:
        st.subheader("📊 Sentiment Analysis")
        
        if sentiment:
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                st.metric(
                    "Bullish %", 
                    f"{sentiment.get('sentiment_bullish_percent', 0):.1f}%"
                )
            
            with col2b:
                st.metric(
                    "Bearish %", 
                    f"{sentiment.get('sentiment_bearish_percent', 0):.1f}%"
                )
            
            with col2c:
                st.metric(
                    "News Score", 
                    f"{sentiment.get('company_news_score', 0):.2f}"
                )
            
            # Sentiment visualization
            if 'sentiment_bullish_percent' in sentiment and 'sentiment_bearish_percent' in sentiment:
                bullish = sentiment['sentiment_bullish_percent']
                bearish = sentiment['sentiment_bearish_percent']
                neutral = 100 - bullish - bearish
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Bullish', 'Bearish', 'Neutral'],
                    values=[bullish, bearish, neutral],
                    marker_colors=['green', 'red', 'gray']
                )])
                
                fig.update_layout(
                    title="Sentiment Distribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data available")

def crypto_tab():
    """Cryptocurrency information tab"""
    st.subheader("₿ Cryptocurrency Information")
    
    with st.spinner("Loading cryptocurrency data..."):
        crypto_data, error = get_crypto_data()
    
    if error:
        st.error(f"Error loading crypto data: {error}")
        st.info("💡 Make sure your crypto data provider is configured")
        return
    
    if crypto_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Bitcoin (USD)", 
                f"${crypto_data.get('price_usd', 0):,.2f}"
            )
        
        with col2:
            st.metric(
                "Bitcoin (EUR)", 
                f"€{crypto_data.get('price_eur', 0):,.2f}"
            )
        
        with col3:
            st.metric(
                "Bitcoin (GBP)", 
                f"£{crypto_data.get('price_gbp', 0):,.2f}"
            )
        
        # Additional crypto metrics if available
        if 'market_cap' in crypto_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Market Cap", 
                    f"${crypto_data.get('market_cap', 0):,.0f}"
                )
            
            with col2:
                st.metric(
                    "24h Volume", 
                    f"${crypto_data.get('volume_24h', 0):,.0f}"
                )
            
            with col3:
                st.metric(
                    "24h Change", 
                    f"{crypto_data.get('change_24h', 0):.2f}%"
                )
    else:
        st.info("No cryptocurrency data available")
    
    st.markdown("---")
    st.info("💡 **Crypto Data**: Real-time cryptocurrency prices and market information")

if __name__ == "__main__":
    main()