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

# Add project to path
sys.path.insert(0, '.')

# Page configuration
st.set_page_config(
    page_title="🚀 Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=900)  # Cache for 15 minutes (free tier limitation)
def load_data(symbol, days=30):
    """Load market data with caching - adjusted for Alpaca free tier"""
    try:
        from src.data import AlpacaDataProvider
        provider = AlpacaDataProvider()
        
        # Alpaca free tier: data is delayed by 15 minutes
        # Use previous trading day to avoid real-time data issues
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        
        data = provider.get_bars(symbol, '1Day', start_date, end_date)
        return data, None
    except Exception as e:
        return None, str(e)

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
            current_date=datetime.now(),
            current_prices=current_prices,
            current_data=current_data,
            historical_data=historical_data,
            portfolio=portfolio
        )
        
        return signals, None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_backtest(symbols, days_back=365, initial_cash=100000):
    """Run backtest with caching"""
    try:
        from src.trade import Backtester
        from src.strategy import MomentumStrategy
        from src.data import AlpacaDataProvider
        
        provider = AlpacaDataProvider()
        backtester = Backtester(initial_cash=initial_cash, commission=0.01)
        
        # Load data for all symbols
        end_date = datetime.now() - timedelta(days=2)  # Avoid weekend issues
        start_date = end_date - timedelta(days=days_back)
        
        for symbol in symbols:
            data = provider.get_bars(symbol, '1Day', start_date, end_date)
            if not data.empty:
                backtester.add_data(symbol, data)
        
        # Create strategy
        strategy = MomentumStrategy()
        
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"dashboard_transactions_{timestamp}.csv"
        
        transactions_df.to_csv(filename, index=False)
        return filename
        
    except Exception as e:
        st.error(f"Error generating transaction log: {e}")
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🚀 Quantitative Trading System Dashboard")
    st.markdown("**Python 3.12 • Real-time Data • Advanced Analytics**")
    
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
    tab1, tab2, tab3 = st.tabs(["📈 Live Analysis", "🔄 Backtesting", "💼 Account"])
    
    with tab1:
        # Live Analysis Tab
        main_analysis_tab(symbol, days_back)
    
    with tab2:
        # Backtesting Tab
        backtesting_tab()
    
    with tab3:
        # Account Tab
        account_tab()

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
    st.markdown("**Test your momentum strategy on historical data**")
    
    # Backtest parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backtest_symbols = st.multiselect(
            "Select Symbols",
            ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY"],
            default=["AAPL", "GOOGL", "MSFT"]
        )
    
    with col2:
        backtest_days = st.slider("Days to Test", 90, 730, 365)
        
    with col3:
        initial_cash = st.number_input("Initial Cash ($)", value=100000, step=10000)
    
    if st.button("🚀 Run Backtest", type="primary", disabled=len(backtest_symbols) == 0):
        with st.spinner("Running backtest... This may take a moment"):
            results, error = run_backtest(backtest_symbols, backtest_days, initial_cash)
        
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
    st.info("💡 **Tip**: Backtesting shows how the momentum strategy would have performed on historical data. Past performance doesn't guarantee future results.")

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
    
    # Data limitations notice
    st.markdown("---")
    st.info("📊 **Data Notice**: Free tier accounts have 15-minute delayed data. Consider upgrading for real-time data access.")

if __name__ == "__main__":
    main()