"""
Trading Views Mixin - Trading interface and backtesting views.
"""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import Backtester
from src.data import AlpacaDataProvider
from src.strategy.registry import get_strategy_registry
from src.ui.ui_components import UIComponents
from src.utils.timezone_utils import now_et


class TradingViewsMixin:
    """Mixin providing trading and backtesting rendering methods."""

    @staticmethod
    def _format_strategy_name(name: str) -> str:
        """Format a strategy slug for display."""
        return name.replace("_", " ").title()

    def _get_backtest_strategies(self, asset_type: str) -> list[str]:
        """Return dashboard-visible strategies for an asset type."""
        registry = get_strategy_registry()
        return [
            name
            for name in registry.list_strategies(dashboard_only=True)
            if registry.get_meta(name).asset_type == asset_type
        ]

    @staticmethod
    def _default_symbols_for_asset(asset_type: str) -> list[str]:
        """Return sensible fallback symbols for dashboard backtests."""
        if asset_type == "crypto":
            return ["BTC/USD", "ETH/USD"]
        return ["AAPL", "MSFT", "GOOGL"]

    def _create_backtest_strategy(self, strategy_name: str):
        """Create a strategy with dashboard-safe defaults."""
        params = {"mode": "fast"} if strategy_name == "multi_agent" else None
        return get_strategy_registry().create(strategy_name, params)

    @staticmethod
    def _load_backtest_data(
        provider: AlpacaDataProvider,
        symbol: str,
        asset_type: str,
        start_date,
    ) -> pd.DataFrame:
        """Load historical bars for the selected asset type."""
        if asset_type == "crypto":
            return provider.get_crypto_bars(symbol, "1Day", start=start_date)
        return provider.get_bars(symbol, "1Day", start=start_date)

    def render_strategy_backtest_tab(self):
        """Strategy Backtest: Quick Backtest & Strategy Comparison"""
        st.header("Strategy Backtest")
        backtest_tabs = st.tabs(["Quick Backtest", "Strategy Comparison"])
        with backtest_tabs[0]:
            self.render_quick_backtest()
        with backtest_tabs[1]:
            self.render_strategy_comparison()

    def render_quick_backtest(self):
        """Render simplified single strategy backtest"""
        st.subheader("Quick Backtest")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            asset_type = st.selectbox(
                "Asset Type",
                ["stock", "crypto"],
                key="quick_backtest_asset_type",
            )
        with col2:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols(asset_type)
            backtest_symbol = st.selectbox(
                "Symbol",
                options=default_symbols or self._default_symbols_for_asset(asset_type),
                key="quick_backtest_symbol",
            )
        with col3:
            available_strategies = self._get_backtest_strategies(asset_type)
            backtest_strategy = st.selectbox(
                "Strategy",
                available_strategies,
                format_func=self._format_strategy_name,
                key="quick_strategy",
            )
        with col4:
            lookback_days = st.slider("Lookback (days)", 30, 365, 90, key="quick_lookback")

        if st.button("Run Quick Backtest", type="primary"):
            self._run_quick_backtest(
                backtest_symbol,
                backtest_strategy,
                lookback_days,
                asset_type,
            )

    def _run_quick_backtest(
        self,
        symbol: str,
        strategy_name: str,
        lookback_days: int,
        asset_type: str,
    ):
        """Execute quick backtest and display results"""
        with st.spinner("Running backtest..."):
            try:
                provider = AlpacaDataProvider()
                end_date = now_et()
                start_date = end_date - timedelta(days=lookback_days)
                historical_data = self._load_backtest_data(
                    provider,
                    symbol,
                    asset_type,
                    start_date,
                )

                if historical_data.empty:
                    st.error(f"No data available for {symbol}")
                    return

                strategy = self._create_backtest_strategy(strategy_name)
                backtester = Backtester(initial_cash=100000, commission=0.01)
                backtester.add_data(symbol, historical_data)

                def strategy_func(date, prices, current, hist, portfolio):
                    return strategy.generate_signals(date, prices, current, hist, portfolio)

                results = backtester.run_backtest(
                    strategy_func,
                    symbols=[symbol],
                    strategy=strategy,
                )

                if results:
                    st.success("Backtest completed!")
                    self._display_backtest_results(results, symbol)
                else:
                    st.warning("Backtest produced no results")
            except Exception as e:
                st.error(f"Error running backtest: {e}")

    def _display_backtest_results(self, results: dict, symbol: str):
        """Display backtest results"""
        summary = results.get("summary", results)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_return = summary.get('total_return_percentage', 0)
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            total_trades = summary.get('total_trades', 0)
            st.metric("Total Trades", total_trades)
        with col3:
            win_rate = summary.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            max_dd = summary.get('max_drawdown_percentage', 0)
            st.metric("Max Drawdown", f"{max_dd:.2f}%")

        portfolio_history = results.get('portfolio_history')
        if portfolio_history is not None and not portfolio_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_history.index, y=portfolio_history['portfolio_value'],
                mode='lines', name='Portfolio Value', line={"color": "blue", "width": 2}
            ))
            fig.update_layout(
                title=f"Backtest Results - {symbol}",
                yaxis_title="Portfolio Value ($)", height=400
            )
            st.plotly_chart(fig, width="stretch")

        trades_history = results.get('trades_history')
        if trades_history is not None and not trades_history.empty:
            st.write("**Trade History**")
            st.dataframe(trades_history, width="stretch")

    def render_strategy_comparison(self):
        """Render strategy comparison"""
        st.subheader("Strategy Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            asset_type = st.selectbox(
                "Asset Type",
                ["stock", "crypto"],
                key="comparison_asset_type",
            )
        with col2:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols(asset_type)
            comparison_symbol = st.selectbox(
                "Symbol",
                options=default_symbols or self._default_symbols_for_asset(asset_type),
                key="comparison_symbol",
            )
        with col3:
            comparison_days = st.slider("Lookback (days)", 30, 365, 90, key="comparison_lookback")

        all_strategies = self._get_backtest_strategies(asset_type)
        selected_strategies = st.multiselect(
            "Select strategies to compare",
            all_strategies,
            default=all_strategies[:3] if len(all_strategies) >= 3 else all_strategies,
            format_func=self._format_strategy_name,
            key=f"comparison_strategies_{asset_type}",
        )

        if st.button("Compare Strategies", type="primary"):
            if len(selected_strategies) < 2:
                st.warning("Please select at least 2 strategies to compare")
            else:
                self._run_strategy_comparison(
                    comparison_symbol,
                    selected_strategies,
                    comparison_days,
                    asset_type,
                )

    def _run_strategy_comparison(
        self,
        symbol: str,
        strategies: list[str],
        lookback_days: int,
        asset_type: str,
    ):
        """Execute strategy comparison and display results"""
        with st.spinner("Comparing strategies..."):
            try:
                provider = AlpacaDataProvider()
                end_date = now_et()
                start_date = end_date - timedelta(days=lookback_days)
                historical_data = self._load_backtest_data(
                    provider,
                    symbol,
                    asset_type,
                    start_date,
                )

                if historical_data.empty:
                    st.error(f"No data available for {symbol}")
                    return

                comparison_results = []

                for strategy_name in strategies:
                    try:
                        strategy = self._create_backtest_strategy(strategy_name)
                        backtester = Backtester(initial_cash=100000, commission=0.01)
                        backtester.add_data(symbol, historical_data)

                        def strategy_func(
                            date,
                            prices,
                            current,
                            hist,
                            portfolio,
                            _strategy=strategy,
                        ):
                            return _strategy.generate_signals(
                                date,
                                prices,
                                current,
                                hist,
                                portfolio,
                            )

                        results = backtester.run_backtest(
                            strategy_func,
                            symbols=[symbol],
                            strategy=strategy,
                        )

                        if results:
                            summary = results.get("summary", results)
                            comparison_results.append({
                                'Strategy': self._format_strategy_name(strategy_name),
                                'Total Return': (
                                    f"{summary.get('total_return_percentage', 0):.2f}%"
                                ),
                                'Total Trades': summary.get('total_trades', 0),
                                'Win Rate': f"{summary.get('win_rate', 0):.1f}%",
                                'Max Drawdown': (
                                    f"{summary.get('max_drawdown_percentage', 0):.2f}%"
                                ),
                            })
                    except Exception as e:
                        comparison_results.append({
                            'Strategy': self._format_strategy_name(strategy_name),
                            'Total Return': f'Error: {str(e)[:20]}',
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
        st.dataframe(df, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

    def render_trade_order_tab(self):
        """Trade: Order Entry, Watchlist, Orders History"""
        st.header("Trade")
        trade_tabs = st.tabs(["Order Entry", "Watchlist", "Order History"])
        with trade_tabs[0]:
            UIComponents.render_trading_interface()
        with trade_tabs[1]:
            UIComponents.render_watchlist_interface()
        with trade_tabs[2]:
            self.render_orders_table()

    def render_orders_table(self):
        """Render orders history table"""
        st.subheader("Recent Orders")
        try:
            if 'order_manager' in st.session_state:
                orders = st.session_state.order_manager.get_orders(status='all', limit=50)
                if orders and isinstance(orders, list):
                    UIComponents.render_orders_table(orders)
                else:
                    st.info("No recent orders found.")
            else:
                st.info("Order manager not initialized.")
        except Exception as e:
            st.error(f"Error loading orders: {e}")
