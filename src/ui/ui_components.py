"""
UI Components - Reusable UI elements for the dashboard.
"""

from typing import List, Dict, Any
import pandas as pd
import streamlit as st

from src.data import AlpacaDataProvider
from src.trade.stock_engine import TradingStockEngine


class UIComponents:
    """Collection of reusable UI components."""

    @staticmethod
    def render_positions_table(positions: List[Dict[str, Any]]):
        """Render positions as a formatted table"""
        if not positions:
            st.info("No positions to display")
            return

        df_data = []
        for pos in positions:
            try:
                symbol = pos.get('symbol', 'Unknown')
                qty = float(pos.get('qty', 0))
                avg_entry = float(pos.get('avg_entry_price', 0))
                current = float(pos.get('current_price', 0))
                market_value = float(pos.get('market_value', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                unrealized_plpc = float(pos.get('unrealized_plpc', 0)) * 100

                df_data.append({
                    'Symbol': symbol,
                    'Qty': qty,
                    'Avg Entry': f"${avg_entry:.2f}",
                    'Current': f"${current:.2f}",
                    'Market Value': f"${market_value:,.2f}",
                    'P&L': f"${unrealized_pl:,.2f}",
                    'P&L %': f"{unrealized_plpc:+.2f}%"
                })
            except (ValueError, TypeError):
                continue

        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No valid positions to display")

    @staticmethod
    def render_orders_table(orders: List[Dict[str, Any]]):
        """Render orders as a formatted table"""
        if not orders:
            st.info("No orders to display")
            return

        df_data = []
        for order in orders:
            try:
                df_data.append({
                    'Symbol': order.get('symbol', 'Unknown'),
                    'Side': order.get('side', '').upper(),
                    'Type': order.get('type', '').upper(),
                    'Qty': order.get('qty', 0),
                    'Status': order.get('status', '').upper(),
                    'Filled Qty': order.get('filled_qty', 0),
                    'Created': str(order.get('created_at', ''))[:19]
                })
            except (ValueError, TypeError):
                continue

        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No valid orders to display")

    @staticmethod
    def render_trading_interface():
        """Render trading order entry interface"""
        st.subheader("🚀 Place Order")

        col1, col2 = st.columns(2)
        with col1:
            from src.ui.dashboard_utils import get_default_symbols
            default_symbols = get_default_symbols("stock")
            symbol = st.text_input(
                "Symbol", value=default_symbols[0] if default_symbols else "AAPL",
                key="order_symbol"
            ).upper()
        with col2:
            side = st.selectbox("Side", ["buy", "sell"], key="order_side")

        col3, col4 = st.columns(2)
        with col3:
            order_type = st.selectbox("Order Type", ["market", "limit"], key="order_type")
        with col4:
            qty = st.number_input("Quantity", min_value=1, value=1, key="order_qty")

        limit_price = None
        if order_type == "limit":
            limit_price = st.number_input("Limit Price", min_value=0.01, step=0.01, key="limit_price")

        UIComponents._render_order_preview(symbol, side, order_type, qty, limit_price)

        if st.button("Place Order", type="primary", key="place_order_btn"):
            UIComponents._execute_order(symbol, side, order_type, qty, limit_price)

    @staticmethod
    def _render_order_preview(symbol, side, order_type, qty, limit_price=None):
        """Display order preview"""
        st.write("**Order Preview**")
        preview_data = {
            'Symbol': symbol, 'Side': side.upper(), 'Type': order_type.upper(), 'Quantity': qty
        }
        if limit_price:
            preview_data['Limit Price'] = f"${limit_price:.2f}"

        preview_df = pd.DataFrame([preview_data])
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    @staticmethod
    def _execute_order(symbol, side, order_type, qty, limit_price=None):
        """Execute the order"""
        try:
            engine = TradingStockEngine()
            if order_type == "market":
                result = engine.place_market_order(symbol, qty, side=side)
            else:
                result = engine.place_limit_order(symbol, qty, limit_price, side=side)

            if result and 'error' not in result:
                st.success(f"Order placed successfully! Order ID: {result.get('id', 'N/A')}")
            else:
                st.error(f"Order failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error placing order: {e}")

    @staticmethod
    def render_watchlist_interface():
        """Render watchlist management interface"""
        st.subheader("👁️ Watchlist Management")

        if 'watchlist_manager' not in st.session_state:
            st.error("Watchlist manager not initialized")
            return

        manager = st.session_state.watchlist_manager
        UIComponents._render_watchlist_table(manager)
        UIComponents._render_add_symbol(manager)
        UIComponents._render_remove_symbol(manager)

    @staticmethod
    def _render_watchlist_table(manager):
        """Display current watchlist"""
        st.write("**Current Watchlist**")
        entries = manager.get_watchlist_entries()
        if entries:
            df = pd.DataFrame(entries)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Watchlist is empty")

    @staticmethod
    def _render_add_symbol(manager):
        """Render add symbol form"""
        st.write("**Add Symbol**")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            new_symbol = st.text_input("Symbol", key="add_symbol_input").upper()
        with col2:
            asset_type = st.selectbox("Asset Type", ["stock", "crypto", "option"], key="add_asset_type")
        with col3:
            if st.button("Add", key="add_symbol_btn"):
                if new_symbol:
                    try:
                        result = manager.add_symbol(new_symbol, asset_type)
                        if result:
                            st.success(f"Added {new_symbol} to watchlist")
                            st.rerun()
                        else:
                            st.warning(f"{new_symbol} already in watchlist")
                    except Exception as e:
                        st.error(f"Error adding symbol: {e}")
                else:
                    st.warning("Please enter a symbol")

    @staticmethod
    def _render_remove_symbol(manager):
        """Render remove symbol form"""
        st.write("**Remove Symbol**")
        entries = manager.get_watchlist_entries()
        if entries:
            symbols = [f"{e['symbol']} ({e['asset_type']})" for e in entries]
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox("Select symbol to remove", symbols, key="remove_symbol_select")
            with col2:
                if st.button("Remove", key="remove_symbol_btn"):
                    symbol = selected.split(" (")[0]
                    asset_type = selected.split("(")[1].rstrip(")")
                    try:
                        result = manager.remove_symbol(symbol, asset_type)
                        if result:
                            st.success(f"Removed {symbol} from watchlist")
                            st.rerun()
                        else:
                            st.warning(f"{symbol} not found in watchlist")
                    except Exception as e:
                        st.error(f"Error removing symbol: {e}")

    @staticmethod
    def render_data_table(data: pd.DataFrame, title: str = None):
        """Render a data table with optional title"""
        if title:
            st.write(f"**{title}**")
        if data is not None and not data.empty:
            st.dataframe(data, use_container_width=True)
        else:
            st.info("No data available")
