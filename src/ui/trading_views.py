"""
Trading Views Mixin - Trading interface and backtesting views.
"""

from datetime import timedelta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data import AlpacaDataProvider
from src.strategy.registry import get_strategy_registry
from src.trade.backtester import Backtester
from src.utils.timezone_utils import now_et
from src.ui.ui_components import UIComponents


class TradingViewsMixin:
    """Mixin providing trading and backtesting rendering methods."""

    def render_strategy_backtest_tab(self):
        """Strategy Backtest: Quick Backtest & Strategy Comparison"""
        st.header("📈 Strategy Backtest")
        backtest_tabs = st.tabs(["⚡ Quick Backtest", "📊 Strategy Comparison"])
        with backtest_tabs[0]:
            self.render_quick_backtest()
        with backtest_tabs[1]:
            self.render_strategy_comparison()

    def render_quick_backtest(self):
        """Render simplified single strategy backtest"""
        st.subheader("⚡ Quick Backtest")
        col1, col2, col3 = st.columns(3)
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols("stock")
            backtest_symbol = st.selectbox(
                "Symbol", options=default_symbols if default_symbols else ["AAPL", "MSFT", "GOOGL"],
                key="quick_backtest_symbol"
            )
        with col2:
            registry = get_strategy_registry()
            available_strategies = registry.list_strategies(dashboard_only=True)
            backtest_strategy = st.selectbox("Strategy", available_strategies, key="quick_strategy")
        with col3:
            lookback_days = st.slider("Lookback (days)", 30, 365, 90, key="quick_lookback")

        if st.button("Run Quick Backtest", type="primary"):
            self._run_quick_backtest(backtest_symbol, backtest_strategy, lookback_days)

    def _run_quick_backtest(self, symbol: str, strategy_name: str, lookback_days: int):
        """Execute quick backtest and display results"""
        with st.spinner("Running backtest..."):
            try:
                provider = AlpacaDataProvider()
                end_date = now_et()
                start_date = end_date - timedelta(days=lookback_days)
                historical_data = provider.get_bars(symbol, "1Day", start=start_date)

                if historical_data.empty:
                    st.error(f"No data available for {symbol}")
                    return

                registry = get_strategy_registry()
                strategy = registry.create(strategy_name)
                backtester = Backtester(initial_cash=100000, commission=0.01)
                backtester.add_data(symbol, historical_data)

                def strategy_func(date, prices, current, hist, portfolio):
                    return strategy.generate_signals(date, prices, current, hist, portfolio)

                results = backtester.run_backtest(strategy_func, symbols=[symbol])

                if results:
                    st.success("Backtest completed!")
                    self._display_backtest_results(results, symbol)
                else:
                    st.warning("Backtest produced no results")
            except Exception as e:
                st.error(f"Error running backtest: {e}")

    def _display_backtest_results(self, results: dict, symbol: str):
        """Display backtest results"""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_return = results.get('total_return_percentage', 0)
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            total_trades = results.get('total_trades', 0)
            st.metric("Total Trades", total_trades)
        with col3:
            win_rate = results.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            max_dd = results.get('max_drawdown_percentage', 0)
            st.metric("Max Drawdown", f"{max_dd:.2f}%")

        portfolio_history = results.get('portfolio_history')
        if portfolio_history is not None and not portfolio_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_history.index, y=portfolio_history['portfolio_value'],
                mode='lines', name='Portfolio Value', line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f"Backtest Results - {symbol}",
                yaxis_title="Portfolio Value ($)", height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        trades_history = results.get('trades_history')
        if trades_history is not None and not trades_history.empty:
            st.write("**Trade History**")
            st.dataframe(trades_history, use_container_width=True)

    def render_strategy_comparison(self):
        """Render strategy comparison"""
        st.subheader("📊 Strategy Comparison")
        col1, col2 = st.columns(2)
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols("stock")
            comparison_symbol = st.selectbox(
                "Symbol", options=default_symbols if default_symbols else ["AAPL", "MSFT", "GOOGL"],
                key="comparison_symbol"
            )
        with col2:
            comparison_days = st.slider("Lookback (days)", 30, 365, 90, key="comparison_lookback")

        registry = get_strategy_registry()
        all_strategies = registry.list_strategies(dashboard_only=True)
        selected_strategies = st.multiselect(
            "Select strategies to compare", all_strategies,
            default=all_strategies[:3] if len(all_strategies) >= 3 else all_strategies,
            key="comparison_strategies"
        )

        if st.button("Compare Strategies", type="primary"):
            if len(selected_strategies) < 2:
                st.warning("Please select at least 2 strategies to compare")
            else:
                self._run_strategy_comparison(
                    comparison_symbol, selected_strategies, comparison_days
                )

    def _run_strategy_comparison(self, symbol: str, strategies: list, lookback_days: int):
        """Execute strategy comparison and display results"""
        with st.spinner("Comparing strategies..."):
            try:
                provider = AlpacaDataProvider()
                end_date = now_et()
                start_date = end_date - timedelta(days=lookback_days)
                historical_data = provider.get_bars(symbol, "1Day", start=start_date)

                if historical_data.empty:
                    st.error(f"No data available for {symbol}")
                    return

                registry = get_strategy_registry()
                comparison_results = []

                for strategy_name in strategies:
                    try:
                        strategy = registry.create(strategy_name)
                        backtester = Backtester(initial_cash=100000, commission=0.01)
                        backtester.add_data(symbol, historical_data)

                        def strategy_func(date, prices, current, hist, portfolio):
                            return strategy.generate_signals(date, prices, current, hist, portfolio)

                        results = backtester.run_backtest(strategy_func, symbols=[symbol])

                        if results:
                            comparison_results.append({
                                'Strategy': strategy_name,
                                'Total Return': f"{results.get('total_return_percentage', 0):.2f}%",
                                'Total Trades': results.get('total_trades', 0),
                                'Win Rate': f"{results.get('win_rate', 0):.1f}%",
                                'Max Drawdown': f"{results.get('max_drawdown_percentage', 0):.2f}%"
                            })
                    except Exception as e:
                        comparison_results.append({
                            'Strategy': strategy_name, 'Total Return': f'Error: {str(e)[:20]}',
                            'Total Trades': '-', 'Win Rate': '-', 'Max Drawdown': '-'
                        })

                if comparison_results:
                    self.render_strategy_comparison_results(comparison_results, symbol)
            except Exception as e:
                st.error(f"Error comparing strategies: {e}")

    def render_strategy_comparison_results(self, comparison_results: list, symbol: str):
        """Display strategy comparison results"""
        st.success("Comparison completed!")
        df = pd.DataFrame(comparison_results)
        st.dataframe(df, use_container_width=True)

        valid = [r for r in comparison_results if 'Error' not in str(r['Total Return'])]
        if valid:
            strategies = [r['Strategy'] for r in valid]
            returns = [float(r['Total Return'].rstrip('%')) for r in valid]
            import plotly.express as px
            fig = px.bar(
                x=strategies, y=returns, title=f"Strategy Returns Comparison - {symbol}",
                color=returns, color_continuous_scale="RdYlGn",
                text=[f"{r:+.2f}%" for r in returns]
            )
            fig.update_layout(yaxis_title="Total Return (%)", height=400)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    def render_trade_order_tab(self):
        """Trade: Order Entry, Watchlist, Orders History"""
        st.header("⚡ Trade")
        trade_tabs = st.tabs(["🚀 Order Entry", "👁️ Watchlist", "📋 Order History"])
        with trade_tabs[0]:
            UIComponents.render_trading_interface()
        with trade_tabs[1]:
            UIComponents.render_watchlist_interface()
        with trade_tabs[2]:
            self.render_orders_table()

    def render_orders_table(self):
        """Render orders history table"""
        st.subheader("📋 Recent Orders")
        try:
            if 'order_manager' in st.session_state:
                orders = st.session_state.order_manager.get_orders(status='all', limit=50)
                if orders and isinstance(orders, list) and not any('error' in o for o in orders):
                    UIComponents.render_orders_table(orders)
                else:
                    st.info("No recent orders found.")
            else:
                st.info("Order manager not initialized.")
        except Exception as e:
            st.error(f"Error loading orders: {e}")
