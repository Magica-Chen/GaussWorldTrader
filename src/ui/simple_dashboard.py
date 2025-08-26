#!/usr/bin/env python3
"""
Simple Streamlit Dashboard for the Trading System
Refactored to use core_dashboard base class
Enhanced with crypto, news, and technical analysis features
"""

import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ui.core_dashboard import BaseDashboard, UIComponents
from src.data import NewsDataProvider


class SimpleDashboard(BaseDashboard):
    """Simple dashboard implementation extending BaseDashboard"""
    
    def __init__(self):
        super().__init__(
            title="Gauss World Trader Dashboard",
            icon="📈"
        )
    
    @st.cache_data(ttl=600)
    def get_crypto_data(_self):
        """Get cryptocurrency data with caching"""
        try:
            from src.data import AlpacaDataProvider
            provider = AlpacaDataProvider()
            
            btc_quote = provider.get_crypto_latest_quote('BTC/USD')
            btc_data = {
                'symbol': 'BTC',
                'price_usd': btc_quote.get('bid_price', 0) if 'error' not in btc_quote else 0,
                'last_updated': btc_quote.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if 'error' not in btc_quote else 'Unknown'
            }
            return btc_data, None
            
        except Exception as e:
            return None, str(e)
    
    @st.cache_data(ttl=1200)
    def get_news_insider_data(_self, symbol):
        """Get news and insider data with caching"""
        try:
            provider = NewsDataProvider()
            
            news = provider.get_company_news(symbol)
            insider_transactions = provider.get_insider_transactions(symbol)
            insider_sentiment = provider.get_insider_sentiment(symbol)
            
            return news, insider_transactions, insider_sentiment, None
            
        except Exception as e:
            return [], [], {}, str(e)
    
    def generate_dashboard_transaction_log(self, results, symbols):
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
            from src.utils.timezone_utils import now_et
            timestamp = now_et().strftime('%Y%m%d_%H%M%S')
            filename = f"dashboard_transactions_{timestamp}.csv"
            
            transactions_df.to_csv(filename, index=False)
            return filename
            
        except Exception as e:
            st.error(f"Error generating transaction log: {e}")
            return None
    
    def main_analysis_tab(self, symbol, days_back):
        """Main analysis tab content with technical indicators"""
        # Main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 {symbol} Price Chart")
            
            # Load data
            with st.spinner(f"Loading {symbol} data..."):
                data, error = self.load_market_data(symbol, days_back)
            
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
            
            # Create enhanced chart using base class
            fig = self.create_price_chart(symbol, data)
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
            
            st.info(f"📅 Data Period: {data_start} to {data_end} ({len(data)} trading days)")
            
            recent_data = data.tail(10).copy()
            recent_data.index = recent_data.index.strftime('%Y-%m-%d')
            st.dataframe(recent_data, use_container_width=True)
        
        with col2:
            # Current metrics using base class
            st.subheader("📊 Current Metrics")
            self.render_current_metrics(data)
            
            # Technical Indicators using base class
            with st.spinner("Calculating indicators..."):
                indicators = self.get_technical_indicators(data)
            
            if indicators:
                self.render_technical_indicators(indicators)
            
            # Trading signals using base class
            with st.spinner("Generating signals..."):
                signals, signal_error = self.generate_trading_signals(symbol, data)
            
            if signal_error:
                st.error(f"Signal Error: {signal_error}")
            else:
                self.render_trading_signals(signals)
        
        # Account information using base class
        st.subheader("💼 Account Information")
        
        with st.spinner("Loading account data..."):
            account_data, account_error = self.get_account_info()
        
        if account_error:
            st.error(f"Account Error: {account_error}")
        elif account_data:
            self.render_account_metrics(account_data)
    
    def backtesting_tab(self):
        """Backtesting functionality tab"""
        st.subheader("🔄 Strategy Backtesting")
        st.markdown("**Test your quantitative trading strategies on historical data**")
        
        # Strategy selection and backtest parameters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            strategy_type = st.selectbox(
                "Select Strategy",
                ["Momentum", "Mean Reversion", "Trend Following"],
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
            "Trend Following": "📈 Follows long-term trends using moving average crossovers"
        }
        st.info(f"**Strategy:** {strategy_descriptions.get(strategy_type, 'Custom strategy')}")
        
        if st.button("🚀 Run Backtest", type="primary", disabled=len(backtest_symbols) == 0):
            with st.spinner(f"Running {strategy_type} strategy backtest... This may take a moment"):
                results, error = self.run_backtest(backtest_symbols, backtest_days, initial_cash, strategy_type)
            
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
                        csv_filename = self.generate_dashboard_transaction_log(results, backtest_symbols)
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
    
    def account_tab(self):
        """Account information tab"""
        st.subheader("💼 Account Overview")
        
        # Account information using base class
        with st.spinner("Loading account data..."):
            account_data, account_error = self.get_account_info()
        
        if account_error:
            st.error(f"Account Error: {account_error}")
        elif account_data:
            self.render_account_metrics(account_data)
            
            # Show current positions
            try:
                positions = st.session_state.trading_engine.get_current_positions()
                
                if positions:
                    st.subheader("📊 Current Positions")
                    UIComponents.render_positions_table(positions)
                else:
                    st.info("📭 No current positions")
                    
            except Exception as e:
                st.error(f"Error fetching positions: {e}")
    
    def trading_tab(self, symbol):
        """Trading panel functionality"""
        st.subheader("🔄 Trading Panel")
        st.markdown("**Execute trades directly from the dashboard**")
        
        # Use the standardized order form from UIComponents
        UIComponents.render_order_form()
        
        # Recent orders section
        st.markdown("---")
        st.subheader("📊 Recent Orders")
        
        try:
            orders = st.session_state.trading_engine.get_open_orders()
            
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
    
    def news_insider_tab(self, symbol):
        """News and insider analysis tab"""
        st.subheader("📰 News & Insider Analysis")
        
        with st.spinner("Loading news and insider data..."):
            news, insider_transactions, insider_sentiment, error = self.get_news_insider_data(symbol)
        
        if error:
            st.error(f"Error loading data: {error}")
            st.info("💡 Make sure your Finnhub API is configured properly")
            return
        
        # Create tabs for different data types
        news_tab, transactions_tab, sentiment_tab = st.tabs(["📰 Company News", "🏢 Insider Transactions", "📊 Insider Sentiment"])
        
        with news_tab:
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
        
        with transactions_tab:
            st.subheader(f"🏢 Insider Transactions - {symbol}")
            
            if insider_transactions and len(insider_transactions) > 0:
                # Create a table for insider transactions
                transactions_df = pd.DataFrame(insider_transactions[:10])  # Show latest 10
                if not transactions_df.empty:
                    # Select relevant columns
                    cols_to_show = ['name', 'share', 'change', 'filingDate', 'transactionDate', 'transactionCode']
                    available_cols = [col for col in cols_to_show if col in transactions_df.columns]
                    if available_cols:
                        st.dataframe(transactions_df[available_cols], use_container_width=True)
                    else:
                        st.dataframe(transactions_df, use_container_width=True)
                else:
                    st.info("No insider transactions data available")
            else:
                st.info("No insider transactions available")
        
        with sentiment_tab:
            st.subheader(f"📊 Insider Sentiment - {symbol}")
            
            if insider_sentiment and 'data' in insider_sentiment:
                sentiment_data = insider_sentiment['data']
                if sentiment_data:
                    # Create metrics
                    col1, col2, col3 = st.columns(3)
                    
                    latest_data = sentiment_data[-1] if sentiment_data else {}
                    
                    with col1:
                        st.metric(
                            "Month", 
                            f"{latest_data.get('year', 'N/A')}-{latest_data.get('month', 'N/A'):02d}"
                        )
                    
                    with col2:
                        st.metric(
                            "MSPR", 
                            f"{latest_data.get('mspr', 0):.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Change", 
                            f"{latest_data.get('change', 0):+.0f}"
                        )
                    
                    # Show trend chart if multiple data points
                    if len(sentiment_data) > 1:
                        df_sentiment = pd.DataFrame(sentiment_data)
                        if 'mspr' in df_sentiment.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=[f"{row['year']}-{row['month']:02d}" for _, row in df_sentiment.iterrows()],
                                y=df_sentiment['mspr'],
                                mode='lines+markers',
                                name='MSPR',
                                line=dict(color='blue')
                            ))
                            
                            fig.update_layout(
                                title="Insider Sentiment Trend (MSPR)",
                                xaxis_title="Date",
                                yaxis_title="MSPR",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No insider sentiment data available")
            else:
                st.info("No insider sentiment data available")
    
    def crypto_tab(self):
        """Cryptocurrency information tab"""
        st.subheader("₿ Cryptocurrency Information")
        
        with st.spinner("Loading cryptocurrency data..."):
            crypto_data, error = self.get_crypto_data()
        
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
                    "Last Updated", 
                    crypto_data.get('last_updated', 'Unknown')
                )
            
            with col3:
                st.metric(
                    "Symbol", 
                    crypto_data.get('symbol', 'BTC')
                )
        else:
            st.info("No cryptocurrency data available")
        
        st.markdown("---")
        st.info("💡 **Crypto Data**: Real-time cryptocurrency prices and market information")
    
    def render_main_content(self):
        """Main dashboard content implementation"""
        st.markdown("**Python 3.12 • Real-time Data • Advanced Analytics • Named after Carl Friedrich Gauss**")
        
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
            "🔄 Trading", "📰 News & Insider", "₿ Crypto"
        ])
        
        with tab1:
            # Live Analysis Tab
            self.main_analysis_tab(symbol, days_back)
        
        with tab2:
            # Backtesting Tab
            self.backtesting_tab()
        
        with tab3:
            # Account Tab
            self.account_tab()
        
        with tab4:
            # Trading Tab
            self.trading_tab(symbol)
        
        with tab5:
            # News & Insider Tab
            self.news_insider_tab(symbol)
        
        with tab6:
            # Cryptocurrency Tab
            self.crypto_tab()


def main():
    """Main function to run the dashboard"""
    dashboard = SimpleDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()