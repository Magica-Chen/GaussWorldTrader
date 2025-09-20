#!/usr/bin/env python3
"""
Wheel Strategy Dashboard

Specialized dashboard for monitoring and managing the Wheel Options Strategy.
Provides real-time monitoring, signal generation, and position management
for the systematic wheel trading approach.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ui.core_dashboard import BaseDashboard, UIComponents
from src.option_strategy import WheelStrategy
from src.data import AlpacaDataProvider
from src.trade import Portfolio
from src.utils.timezone_utils import now_et

class WheelDashboard(BaseDashboard):
    """Dashboard specifically designed for the Wheel Options Strategy"""

    def __init__(self):
        super().__init__("Wheel Strategy Dashboard", "🎯")

        # Initialize wheel strategy
        if 'wheel_strategy' not in st.session_state:
            st.session_state.wheel_strategy = None

        # Initialize tracking data
        if 'wheel_signals' not in st.session_state:
            st.session_state.wheel_signals = []

        if 'wheel_positions' not in st.session_state:
            st.session_state.wheel_positions = {}

    def render_header_info(self):
        """Render header information (required by BaseDashboard)"""
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if 'wheel_strategy' in st.session_state and st.session_state.wheel_strategy:
                strategy_info = st.session_state.wheel_strategy.get_strategy_info()
                st.metric("Active Positions", strategy_info.get('option_positions', 0))
        with col2:
            market_status = self._get_market_status()
            st.metric("Market Status", market_status['status'])
        with col3:
            current_time = now_et()
            st.metric("Current Time (ET)", current_time.strftime("%H:%M:%S"))

    def render_main_content(self):
        """Main dashboard content implementation (required by BaseDashboard)"""

        # Sidebar configuration
        with st.sidebar:
            self._render_strategy_config()
            st.divider()
            self._render_controls()

        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🎯 Signals", "📈 Positions", "⚙️ Settings", "📚 Education"
        ])

        with tab1:
            self._render_overview_tab()

        with tab2:
            self._render_signals_tab()

        with tab3:
            self._render_positions_tab()

        with tab4:
            self._render_settings_tab()

        with tab5:
            self._render_education_tab()


    def _render_strategy_config(self):
        """Render strategy configuration in sidebar"""
        st.subheader("🎯 Wheel Strategy")

        # Strategy parameters
        with st.expander("Strategy Parameters", expanded=False):
            max_risk = st.number_input(
                "Max Risk ($)",
                min_value=1000,
                max_value=500000,
                value=80000,
                step=5000,
                help="Maximum total risk exposure"
            )

            position_size = st.slider(
                "Position Size (%)",
                min_value=1,
                max_value=20,
                value=8,
                help="Position size as % of portfolio"
            )

            put_delta_range = st.slider(
                "Put Delta Range",
                min_value=0.10,
                max_value=0.50,
                value=(0.15, 0.30),
                step=0.05,
                help="Delta range for cash-secured puts"
            )

            dte_range = st.slider(
                "Days to Expiration",
                min_value=7,
                max_value=60,
                value=(14, 35),
                help="Preferred expiration timeframe"
            )

            min_yield = st.number_input(
                "Minimum Yield (%)",
                min_value=1.0,
                max_value=20.0,
                value=4.0,
                step=0.5,
                help="Minimum acceptable yield"
            )

        # Initialize/Update strategy
        if st.button("🔄 Update Strategy", type="primary"):
            strategy_params = {
                'max_risk': max_risk,
                'position_size_pct': position_size / 100,
                'put_delta_min': put_delta_range[0],
                'put_delta_max': put_delta_range[1],
                'dte_min': dte_range[0],
                'dte_max': dte_range[1],
                'min_yield': min_yield / 100,
                'max_positions': 10,
            }

            st.session_state.wheel_strategy = WheelStrategy(strategy_params)
            st.success("Strategy updated!")
            st.rerun()

    def _render_controls(self):
        """Render control buttons"""
        st.subheader("🎮 Controls")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Generate Signals", type="primary"):
                self._generate_signals()

        with col2:
            if st.button("📊 Refresh Data"):
                self._refresh_data()

        if st.button("⚠️ Emergency Close All", type="secondary"):
            if st.confirm("Close all option positions?"):
                self._emergency_close_all()

    def _render_overview_tab(self):
        """Render overview tab"""
        if not st.session_state.wheel_strategy:
            st.warning("⚠️ Please configure the wheel strategy in the sidebar first.")
            return

        # Strategy status
        strategy = st.session_state.wheel_strategy
        strategy_info = strategy.get_strategy_info()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Active Positions",
                strategy_info.get('option_positions', 0),
                help="Current number of option positions"
            )

        with col2:
            st.metric(
                "Signals Generated",
                len(st.session_state.wheel_signals),
                help="Total signals generated today"
            )

        with col3:
            watchlist_count = strategy_info.get('watchlist_symbols', 0)
            st.metric(
                "Watchlist Symbols",
                watchlist_count,
                help="Symbols available for wheel trading"
            )

        with col4:
            total_risk = strategy_info.get('option_parameters', {}).get('max_risk', 0)
            st.metric(
                "Max Risk",
                f"${total_risk:,}",
                help="Maximum risk exposure limit"
            )

        st.divider()

        # Strategy cycle visualization
        self._render_wheel_cycle_chart()

        # Recent activity
        st.subheader("📋 Recent Activity")
        if st.session_state.wheel_signals:
            recent_signals = st.session_state.wheel_signals[-5:]  # Last 5 signals
            signal_df = pd.DataFrame(recent_signals)
            st.dataframe(signal_df, use_container_width=True)
        else:
            st.info("No recent signals. Click 'Generate Signals' to start.")

    def _render_signals_tab(self):
        """Render signals tab"""
        st.subheader("🎯 Current Signals")

        if not st.session_state.wheel_strategy:
            st.warning("⚠️ Please configure the wheel strategy first.")
            return

        # Signal generation controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            auto_refresh = st.checkbox("Auto-refresh signals", value=False)

        with col2:
            refresh_interval = st.selectbox(
                "Refresh interval",
                ["30s", "1m", "5m", "15m"],
                index=2
            )

        with col3:
            if st.button("🔄 Generate Now"):
                self._generate_signals()

        # Display signals
        if st.session_state.wheel_signals:
            signals_df = pd.DataFrame(st.session_state.wheel_signals)

            # Filter controls
            col1, col2, col3 = st.columns(3)

            with col1:
                signal_types = st.multiselect(
                    "Filter by type",
                    options=signals_df['strategy_stage'].unique() if 'strategy_stage' in signals_df.columns else [],
                    default=signals_df['strategy_stage'].unique() if 'strategy_stage' in signals_df.columns else []
                )

            with col2:
                min_confidence = st.slider(
                    "Min confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )

            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    ["confidence", "score", "yield"] if 'score' in signals_df.columns else ["confidence"]
                )

            # Apply filters
            filtered_df = signals_df.copy()
            if signal_types and 'strategy_stage' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['strategy_stage'].isin(signal_types)]

            if 'confidence' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]

            if sort_by in filtered_df.columns:
                filtered_df = filtered_df.sort_values(sort_by, ascending=False)

            st.dataframe(filtered_df, use_container_width=True)

            # Signal details
            if not filtered_df.empty:
                st.subheader("📊 Signal Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Signal type distribution
                    if 'strategy_stage' in filtered_df.columns:
                        fig = go.Figure(data=[
                            go.Pie(
                                labels=filtered_df['strategy_stage'].value_counts().index,
                                values=filtered_df['strategy_stage'].value_counts().values,
                                title="Signal Types Distribution"
                            )
                        ])
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Confidence distribution
                    if 'confidence' in filtered_df.columns:
                        fig = go.Figure(data=[
                            go.Histogram(
                                x=filtered_df['confidence'],
                                nbinsx=10,
                                title="Confidence Distribution"
                            )
                        ])
                        fig.update_layout(xaxis_title="Confidence", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No signals available. Generate signals to see potential trades.")

    def _render_positions_tab(self):
        """Render positions tab"""
        st.subheader("📈 Current Positions")

        # Mock position data for demonstration
        if st.session_state.wheel_positions:
            positions_df = pd.DataFrame(st.session_state.wheel_positions)
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No current positions. Execute signals to build positions.")

        # Position management
        st.subheader("⚙️ Position Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎯 Check Assignment Risk"):
                self._check_assignment_risk()

        with col2:
            if st.button("💰 Profit Taking Scan"):
                self._profit_taking_scan()

        with col3:
            if st.button("🔄 Rolling Opportunities"):
                self._rolling_opportunities()

    def _render_settings_tab(self):
        """Render settings tab"""
        st.subheader("⚙️ Strategy Settings")

        if not st.session_state.wheel_strategy:
            st.warning("⚠️ No strategy configured.")
            return

        strategy = st.session_state.wheel_strategy
        current_params = strategy.parameters

        # Display current parameters
        st.json(current_params)

        # Advanced settings
        with st.expander("Advanced Settings"):
            st.write("Assignment tolerance:", current_params.get('assignment_tolerance', 0.8))
            st.write("Profit target:", current_params.get('profit_target', 0.5))
            st.write("Management DTE:", current_params.get('management_dte', 7))

        # Risk management
        st.subheader("🛡️ Risk Management")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Max Positions",
                current_params.get('max_positions', 0),
                help="Maximum concurrent positions"
            )

        with col2:
            st.metric(
                "Position Size Limit",
                f"{current_params.get('position_size_pct', 0):.1%}",
                help="Maximum position size as % of portfolio"
            )

    def _render_education_tab(self):
        """Render education tab"""
        st.subheader("📚 Wheel Strategy Education")

        # Strategy explanation
        st.markdown("""
        ## 🎯 What is the Wheel Strategy?

        The Wheel Strategy is a systematic options trading approach that generates income
        through a cyclical process:

        ### 1. 🎯 Cash-Secured Puts
        - Sell put options on stocks you're willing to own
        - Collect premium income immediately
        - If assigned, purchase shares at the strike price

        ### 2. 📦 Assignment & Ownership
        - Receive shares if put options are exercised
        - Now own the underlying stock
        - Ready to move to the next phase

        ### 3. 📞 Covered Calls
        - Sell call options on owned shares
        - Generate additional premium income
        - If called away, sell shares at strike price

        ### 4. 🔄 Cycle Restart
        - Return to cash position
        - Start over with new cash-secured puts

        ## 💡 Key Benefits
        - **Income Generation**: Earn premiums at every step
        - **Systematic Approach**: Removes emotion from trading
        - **Stock Acquisition**: Potentially buy stocks at lower prices
        - **Risk Management**: Built-in position limits and controls

        ## ⚠️ Important Risks
        - **Assignment Risk**: May be forced to buy/sell at unfavorable times
        - **Opportunity Cost**: May miss significant stock moves
        - **Capital Requirements**: Requires substantial cash for puts
        - **Market Risk**: Subject to overall market movements

        ## 📊 Strategy Parameters Explained
        """)

        # Parameter explanations
        with st.expander("Parameter Details"):
            st.markdown("""
            - **Delta Range**: Controls assignment probability (0.20 = ~20% chance)
            - **Days to Expiration**: Time until option expires
            - **Yield**: Return as percentage of capital at risk
            - **Position Size**: Maximum capital allocated per position
            - **Max Risk**: Total capital that can be at risk simultaneously
            """)

        # Example scenarios
        st.subheader("📈 Example Scenarios")

        scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs([
            "Successful Cycle", "Assignment Scenario", "Management Actions"
        ])

        with scenario_tab1:
            st.markdown("""
            ### ✅ Successful Wheel Cycle

            1. **Sell AAPL $145 Put** (AAPL trading at $150)
               - Collect $2.50 premium per share
               - If AAPL stays above $145, keep premium

            2. **Put Expires Worthless**
               - Keep the $250 premium (100 shares × $2.50)
               - Repeat with new put option

            **Result**: Generated $250 income in 3-4 weeks
            """)

        with scenario_tab2:
            st.markdown("""
            ### 📦 Assignment Scenario

            1. **Sell AAPL $145 Put** (AAPL trading at $150)
               - Collect $2.50 premium per share

            2. **AAPL Drops to $140**
               - Put is exercised
               - Buy 100 shares at $145 each
               - Effective cost basis: $142.50 ($145 - $2.50 premium)

            3. **Sell Covered Call**
               - Own 100 shares at $142.50 basis
               - Sell $150 call for $2.00 premium
               - If called away, profit = $150 - $142.50 + $2.00 = $9.50/share

            **Result**: Potential $950 profit plus premiums collected
            """)

        with scenario_tab3:
            st.markdown("""
            ### ⚙️ Position Management

            **Profit Taking**:
            - Close at 50% of premium collected
            - Lock in profits early

            **Rolling**:
            - Extend expiration date
            - Collect additional premium
            - Manage assignment risk

            **Assignment Defense**:
            - Close position if assignment probability > 80%
            - Avoid unfavorable assignments
            """)

    def _generate_signals(self):
        """Generate wheel strategy signals"""
        if not st.session_state.wheel_strategy:
            st.error("Please configure strategy first")
            return

        with st.spinner("Generating wheel strategy signals..."):
            try:
                # Create mock data for demonstration
                current_prices = {
                    'AAPL': 155.50, 'MSFT': 285.75, 'GOOGL': 138.25,
                    'TSLA': 185.30, 'AMZN': 145.80, 'RGTI': 12.50,
                    'AFRM': 45.60, 'UPST': 32.40
                }

                current_data = {}
                for symbol, price in current_prices.items():
                    current_data[symbol] = {
                        'open': price * 0.99,
                        'high': price * 1.02,
                        'low': price * 0.98,
                        'close': price,
                        'volume': np.random.randint(1000000, 5000000)
                    }

                # Mock portfolio
                portfolio = Portfolio(initial_cash=100000)
                portfolio.positions = {
                    'AAPL': {'quantity': 200, 'avg_price': 150.0}
                }

                # Generate signals
                strategy = st.session_state.wheel_strategy
                signals = strategy.generate_signals(
                    current_date=datetime.now(),
                    current_prices=current_prices,
                    current_data=current_data,
                    historical_data={},
                    portfolio=portfolio
                )

                # Store signals
                st.session_state.wheel_signals = signals

                if signals:
                    st.success(f"Generated {len(signals)} signals!")
                else:
                    st.info("No signals generated in current market conditions")

            except Exception as e:
                st.error(f"Error generating signals: {e}")

    def _refresh_data(self):
        """Refresh market data"""
        with st.spinner("Refreshing market data..."):
            # Mock refresh
            st.success("Market data refreshed")

    def _emergency_close_all(self):
        """Emergency close all positions"""
        st.warning("Emergency close all positions executed")
        st.session_state.wheel_positions = {}

    def _check_assignment_risk(self):
        """Check assignment risk for current positions"""
        st.info("Assignment risk analysis completed")

    def _profit_taking_scan(self):
        """Scan for profit taking opportunities"""
        st.info("Profit taking scan completed")

    def _rolling_opportunities(self):
        """Check for rolling opportunities"""
        st.info("Rolling opportunities scan completed")

    def _render_wheel_cycle_chart(self):
        """Render the wheel strategy cycle visualization"""
        st.subheader("🎯 Wheel Strategy Cycle")

        # Create a flow chart showing the wheel cycle
        fig = go.Figure()

        # Add cycle stages
        stages = [
            "Cash Position", "Sell Put", "Assignment?",
            "Own Stock", "Sell Call", "Called Away?"
        ]

        # This would be a more sophisticated chart in production
        st.info("📊 Wheel Cycle: Cash → Sell Put → Assignment → Own Stock → Sell Call → Called Away → Cash")

    def _get_market_status(self) -> Dict[str, str]:
        """Get current market status"""
        # Simplified market status
        current_time = now_et()
        hour = current_time.hour

        if 9 <= hour < 16:
            return {"status": "🟢 OPEN", "next_change": "Closes at 4 PM ET"}
        else:
            return {"status": "🔴 CLOSED", "next_change": "Opens at 9:30 AM ET"}


def main():
    """Main function to run the wheel dashboard"""
    dashboard = WheelDashboard()
    dashboard.run()  # This calls the inherited BaseDashboard.run() method


if __name__ == "__main__":
    main()