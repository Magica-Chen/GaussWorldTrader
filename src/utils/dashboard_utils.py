"""
Shared dashboard utilities to reduce code duplication
Following the principle: rewrite existing components over adding new ones
"""

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from src.utils.timezone_utils import now_et
from src.data import AlpacaDataProvider
from src.strategy import MomentumStrategy
from src.trade import Backtester
from typing import Dict, List, Any, Optional, Tuple


def get_shared_market_data(symbol: str, days: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Centralized market data loading with consistent error handling"""
    try:
        provider = AlpacaDataProvider()
        current_time = now_et()
        start_date = current_time - timedelta(days=days)
        
        data = provider.get_bars(symbol, '1Day', start_date)
        
        if data is not None and not data.empty:
            return data, None
        else:
            return None, f"No data available for {symbol}"
            
    except Exception as e:
        return None, str(e)


def run_shared_backtest(symbols: List[str], days_back: int = 365, 
                       initial_cash: float = 100000, 
                       strategy_type: str = "Momentum") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Centralized backtesting function to eliminate duplication"""
    try:
        provider = AlpacaDataProvider()
        backtester = Backtester(initial_cash=initial_cash, commission=0.01)
        
        current_time = now_et()
        start_date = current_time - timedelta(days=days_back)
        
        # Load data for all symbols
        for symbol in symbols:
            try:
                data = provider.get_bars(symbol, '1Day', start_date)
                if not data.empty:
                    backtester.add_data(symbol, data)
            except Exception:
                continue
        
        # Create strategy based on selection
        strategy_configs = {
            "Momentum": {},
            "Mean Reversion": {
                'lookback_period': 10,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'position_size_pct': 0.1,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.15
            },
            "Trend Following": {
                'lookback_period': 50,
                'rsi_period': 21,
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'position_size_pct': 0.15,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.25
            }
        }
        
        if strategy_type not in strategy_configs:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
        strategy = MomentumStrategy(strategy_configs[strategy_type])
        
        def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
            return strategy.generate_signals(
                current_date, current_prices, current_data, historical_data, portfolio
            )
        
        # Run backtest
        results = backtester.run_backtest(
            strategy_func,
            start_date=start_date + timedelta(days=50),  # Warmup period
            end_date=current_time,
            symbols=symbols
        )
        
        return results, None
        
    except Exception as e:
        return None, str(e)


def render_shared_positions_table(positions: List[Dict]) -> None:
    """Shared position table rendering with consistent formatting"""
    if not positions:
        st.info("No active positions found.")
        return
    
    # Filter and process positions
    positions_data = []
    for pos in positions:
        qty = float(pos.get('qty', 0))
        if qty != 0:
            market_value = float(pos.get('market_value', 0))
            cost_basis = float(pos.get('cost_basis', 0))
            unrealized_pl = float(pos.get('unrealized_pl', 0))
            unrealized_plpc = float(pos.get('unrealized_plpc', 0)) * 100
            
            positions_data.append({
                'Symbol': pos.get('symbol', ''),
                'Quantity': int(qty),
                'Market Value': f"${market_value:,.2f}",
                'Cost Basis': f"${cost_basis:,.2f}",
                'Unrealized P&L': f"${unrealized_pl:,.2f}",
                'Unrealized %': f"{unrealized_plpc:+.2f}%",
                'Side': pos.get('side', ''),
                'Exchange': pos.get('exchange', '')
            })
    
    if positions_data:
        df = pd.DataFrame(positions_data)
        
        # Enhanced styling
        def style_pnl(val):
            if isinstance(val, str) and '$' in val:
                num_val = float(val.replace('$', '').replace(',', ''))
                return 'color: green; font-weight: bold' if num_val >= 0 else 'color: red; font-weight: bold'
            elif isinstance(val, str) and '%' in val:
                num_val = float(val.replace('%', '').replace('+', ''))
                return 'color: green; font-weight: bold' if num_val >= 0 else 'color: red; font-weight: bold'
            return ''
        
        styled_df = df.style.map(style_pnl, subset=['Unrealized P&L', 'Unrealized %'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary metrics
        total_market_value = sum([float(row['Market Value'].replace('$', '').replace(',', '')) for row in positions_data])
        total_unrealized_pl = sum([float(row['Unrealized P&L'].replace('$', '').replace(',', '')) for row in positions_data])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Positions", len(positions_data))
        with col2:
            st.metric("Total Market Value", f"${total_market_value:,.2f}")
        with col3:
            st.metric("Total Unrealized P&L", f"${total_unrealized_pl:,.2f}")
    else:
        st.info("No active positions found.")


def create_shared_price_chart(symbol: str, data: pd.DataFrame) -> object:
    """Shared chart creation function"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        subplot_titles=(f"{symbol} Price", "Volume")
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
        go.Bar(x=data.index, y=data['volume'], name='Volume', opacity=0.3),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


def get_shared_account_info() -> Tuple[Optional[Dict], Optional[str]]:
    """Shared account info retrieval with caching"""
    try:
        account_info = st.session_state.account_manager.get_trading_account_status()
        return account_info, None
    except Exception as e:
        return None, str(e)