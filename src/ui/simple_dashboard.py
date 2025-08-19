#!/usr/bin/env python3
"""
Simple Streamlit Dashboard for the Trading System
Designed to work with existing components
"""

import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz

# Add project to path
sys.path.insert(0, '../..')  # Go up two levels to project root

def get_eastern_time():
    """Get current time in Eastern timezone (handles EST/EDT automatically)"""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

# Page configuration
st.set_page_config(
    page_title="🌍 Gauss World Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=900)  # Cache for 15 minutes (free tier limitation)
def load_data(symbol, days=30):
    """Load market data with smart market close detection and free tier compliance"""
    try:
        from src.data import AlpacaDataProvider
        provider = AlpacaDataProvider()
        
        current_time = get_eastern_time()
        
        # Smart Market Close Detection (all times in Eastern Time)
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        # Pre-market: 4:00 AM - 9:30 AM ET
        # After hours: 4:00 PM - 8:00 PM ET  
        # Overnight: 8:00 PM - 4:00 AM ET (next day)
        is_weekend = current_time.weekday() >= 5  # Saturday=5, Sunday=6
        is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
        is_after_hours = 16 <= current_time.hour < 20  # 4:00 PM - 8:00 PM
        is_overnight = current_time.hour >= 20 or current_time.hour < 4  # 8:00 PM - 4:00 AM
        is_market_open = not is_weekend and not is_pre_market and not is_after_hours and not is_overnight and (9 <= current_time.hour < 16)
        
        # Determine the most recent market close with 1-day buffer for SIP compliance
        if is_weekend:
            # Weekend: Show data through Friday's close
            days_to_friday = current_time.weekday() - 4  # Go back to Friday
            market_close_date = current_time - timedelta(days=days_to_friday)
            market_close_date = market_close_date.replace(hour=16, minute=0, second=0, microsecond=0)
            # Apply 1-day buffer for free tier compliance
            end_date = market_close_date - timedelta(days=1)
            data_context = "Weekend - showing data through Friday's close (1-day delayed)"
            
        elif is_after_hours:
            # After hours: Show data through today's close
            market_close_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            # Apply 1-day buffer for free tier compliance
            end_date = market_close_date- timedelta(days=1)
            data_context = "After hours - showing data through previous day's close"
            
        elif is_overnight:
            # Overnight: Show data through today's close (if late night) or previous day's close (if early morning)
            if current_time.hour >= 20:  # Late night (8 PM - midnight)
                market_close_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            else:  # Early morning (midnight - 4 AM)
                market_close_date = (current_time - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
            # Apply 1-day buffer for free tier compliance
            end_date = market_close_date - timedelta(days=1)
            data_context = "Overnight - showing data through previous day's close"
            
        elif is_pre_market:
            # Pre-market: Show data through previous trading day's close
            if current_time.weekday() == 0:  # Monday
                market_close_date = current_time - timedelta(days=3)  # Go back to Friday
            else:
                market_close_date = current_time - timedelta(days=1)  # Previous day
            market_close_date = market_close_date.replace(hour=16, minute=0, second=0, microsecond=0)
            # Apply 1-day buffer for free tier compliance
            end_date = market_close_date - timedelta(days=1)
            data_context = "Pre-market - showing data through previous trading day's close"
            
        else:
            # During market hours: Show delayed data with SIP compliance
            # Use data from previous day to avoid SIP restrictions
            end_date = current_time - timedelta(days=1)
            data_context = "Market open - showing delayed data (previous day)"
        
        start_date = end_date - timedelta(days=days)
        
        # Fetch data with fallback strategy
        try:
            data = provider.get_bars(symbol, '1Day', start_date, end_date)
            if data is not None and not data.empty:
                return data, data_context
        except Exception:
            pass  # Fall through to fallback
        
        # Fallback: Use older data if primary strategy fails
        end_date = current_time - timedelta(days=2)
        start_date = end_date - timedelta(days=days)
        data = provider.get_bars(symbol, '1Day', start_date, end_date)
        
        if data is not None and not data.empty:
            return data, "Using older historical data due to free tier limitations"
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
            current_date=get_eastern_time(),
            current_prices=current_prices,
            current_data=current_data,
            historical_data=historical_data,
            portfolio=portfolio
        )
        
        return signals, None
    except Exception as e:
        return [], str(e)

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
        end_date = get_eastern_time() - timedelta(days=2)  # Avoid weekend issues
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
        timestamp = get_eastern_time().strftime('%Y%m%d_%H%M%S')
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
    
    # Time information and enhanced market status
    current_time = get_eastern_time()
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**📅 Dashboard Time (ET):** {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    with col2:
        st.warning("⏰ Alpaca free tier: 15-min delay")
    
    with col3:
        # Enhanced Market Status Indicator
        is_weekend = current_time.weekday() >= 5  # Saturday=5, Sunday=6
        is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
        is_after_hours = 16 <= current_time.hour < 20  # 4:00 PM - 8:00 PM
        is_overnight = current_time.hour >= 20 or current_time.hour < 4  # 8:00 PM - 4:00 AM
        is_market_open = not is_weekend and not is_pre_market and not is_after_hours and not is_overnight and (9 <= current_time.hour < 16)
        
        if is_weekend:
            st.error("🔴 Market: Closed (Weekend)")
        elif is_overnight:
            st.info("🔵 Market: Overnight")
        elif is_pre_market:
            st.warning("🟡 Market: Pre-Market")
        elif is_after_hours:
            st.warning("🟡 Market: After Hours")
        else:
            st.success("🟢 Market: Open")
    
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
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Analysis", "🔄 Backtesting", "💼 Account", "🔄 Trading"])
    
    with tab1:
        # Live Analysis Tab
        main_analysis_tab(symbol, days_back)
    
    with tab2:
        # Backtesting Tab
        backtesting_tab()
    
    with tab3:
        # Account Tab
        account_tab()
    
    with tab4:
        # Trading Tab
        trading_tab(symbol)

def main_analysis_tab(symbol, days_back):
    """Main analysis tab content"""
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {symbol} Price Chart")
        
        # Load data
        with st.spinner(f"Loading {symbol} data..."):
            data, error = load_data(symbol, days_back)
        
        if error:
            if "Using" in error or "showing data through" in error or "delayed" in error:
                # This is contextual information or a warning about using fallback data
                st.info(f"ℹ️ {error}")
            elif "previous day" in error or "older historical" in error:
                # This is a warning about using fallback data
                st.warning(f"⚠️ {error}")
            else:
                # This is a real error
                st.error(f"❌ Data Error: {error}")
                st.info("💡 Make sure your Alpaca API keys are configured in .env file")
                return
        
        if data is None or data.empty:
            st.warning(f"⚠️ No data found for {symbol}")
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
        
        fig.update_layout(
            title=f"{symbol} Price Chart",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent data table
        st.subheader("📋 Recent Data")
        
        # Show data period information with contextual messaging
        data_start = data.index[0].strftime('%Y-%m-%d')
        data_end = data.index[-1].strftime('%Y-%m-%d')
        
        # Add contextual information based on current market state
        current_time = get_eastern_time()
        is_weekend = current_time.weekday() >= 5
        is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
        is_after_hours = 16 <= current_time.hour < 20  # 4:00 PM - 8:00 PM
        is_overnight = current_time.hour >= 20 or current_time.hour < 4  # 8:00 PM - 4:00 AM
        is_market_open = not is_weekend and not is_pre_market and not is_after_hours and not is_overnight and (9 <= current_time.hour < 16)
        
        # Generate contextual message
        if is_weekend:
            context_msg = "Weekend - showing data through Friday's close (1-day delayed for free tier)"
        elif is_overnight:
            context_msg = "Overnight - showing data through previous day's close (free tier compliance)"
        elif is_after_hours:
            context_msg = "After hours - showing data through previous day's close (free tier compliance)"
        elif is_pre_market:
            context_msg = "Pre-market - showing data through previous trading day's close"
        else:
            context_msg = "Market open - showing delayed data (previous day for SIP compliance)"
        
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
    
    # Data limitations notice with market context
    st.markdown("---")
    current_time = get_eastern_time()
    is_weekend = current_time.weekday() >= 5
    is_pre_market = (4 <= current_time.hour < 9) or (current_time.hour == 9 and current_time.minute < 30)
    is_after_hours = 16 <= current_time.hour < 20  # 4:00 PM - 8:00 PM
    is_overnight = current_time.hour >= 20 or current_time.hour < 4  # 8:00 PM - 4:00 AM
    is_market_open = not is_weekend and not is_pre_market and not is_after_hours and not is_overnight and (9 <= current_time.hour < 16)
    
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

if __name__ == "__main__":
    main()