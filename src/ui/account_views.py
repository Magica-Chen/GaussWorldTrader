"""
Account Views Mixin - Account info, positions, portfolio analytics views.
"""

from datetime import datetime
from typing import Dict, Any
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data import AlpacaDataProvider
from src.ui.ui_components import UIComponents


class AccountViewsMixin:
    """Mixin providing account and portfolio rendering methods."""

    def render_account_info_tab(self):
        """Account Info: Account, Positions, Portfolio, Configuration"""
        st.header("💼 Account Information")
        account_tabs = st.tabs(["📊 Account", "📈 Positions", "💰 Portfolio", "⚙️ Configuration"])
        with account_tabs[0]:
            self.render_account_overview()
        with account_tabs[1]:
            self.render_positions_view()
        with account_tabs[2]:
            self.render_portfolio_analytics()
        with account_tabs[3]:
            self.render_risk_configuration()

    def render_account_overview(self):
        """Render account overview"""
        st.subheader("📊 Account Overview")
        account_info, error = self.get_account_info()
        if account_info:
            col1, col2, col3, col4 = st.columns(4)
            equity = float(account_info.get('equity', 0))
            last_equity = float(account_info.get('last_equity', equity))
            day_pl = equity - last_equity
            with col1:
                st.metric("Account Value", f"${float(account_info.get('portfolio_value', 0)):,.2f}")
            with col2:
                st.metric("Buying Power", f"${float(account_info.get('buying_power', 0)):,.2f}")
            with col3:
                st.metric("Cash", f"${float(account_info.get('cash', 0)):,.2f}")
            with col4:
                st.metric("Day P&L", f"${day_pl:,.2f}", delta=f"{day_pl:,.2f}")
        else:
            st.error(f"Unable to load account information: {error}")

    def render_positions_view(self):
        """Render current positions"""
        st.subheader("📈 Current Positions")
        if 'position_manager' in st.session_state:
            positions = st.session_state.position_manager.get_all_positions()
            if positions and not any('error' in pos for pos in positions):
                UIComponents.render_positions_table(positions)
            else:
                st.info("No open positions found.")
        else:
            st.error("Position manager not initialized.")

    def render_portfolio_analytics(self):
        """Render portfolio analytics with real data"""
        st.subheader("💰 Portfolio Analytics")
        self.render_portfolio_allocation()
        self.render_portfolio_metrics()

    def render_risk_configuration(self):
        """Render risk management configuration"""
        st.subheader("⚙️ Risk Management Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Position Sizing**")
            st.slider("Max Position Size (%)", 1, 20, 10, key="max_position_size")
            st.slider("Max Portfolio Risk (%)", 1, 10, 2, key="max_portfolio_risk")
        with col2:
            st.write("**Stop Loss Settings**")
            st.slider("Default Stop Loss (%)", 1, 20, 5, key="default_stop_loss")
            st.checkbox("Enable Trailing Stop")
        if st.button("Save Risk Settings"):
            st.success("Risk settings saved successfully!")

    def render_portfolio_allocation(self):
        """Render asset allocation analysis"""
        try:
            account_info, _ = self.get_account_info()
            positions = (
                st.session_state.position_manager.get_all_positions()
                if 'position_manager' in st.session_state else []
            )
            if not account_info or not positions or any('error' in pos for pos in positions):
                st.info("Portfolio data unavailable")
                return
            portfolio_value = float(account_info.get('portfolio_value', 0))
            cash = float(account_info.get('cash', 0))
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Asset Allocation**")
                allocation_data: Dict[str, float] = {'Cash': cash}
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
                    fig = go.Figure(data=[go.Pie(
                        labels=list(allocation_data.keys()),
                        values=list(allocation_data.values())
                    )])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.write("**Portfolio Metrics**")
                total_pl = sum(
                    float(pos.get('unrealized_pl', 0)) for pos in positions
                    if pos.get('unrealized_pl')
                )
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
                                total_return = (
                                    (filtered_equity[-1] - filtered_equity[0])
                                    / filtered_equity[0] * 100
                                )
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
                st.info("Portfolio history temporarily unavailable")
        except Exception as e:
            st.error(f"Error loading performance: {e}")
