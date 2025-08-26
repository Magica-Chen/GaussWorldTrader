"""
Dashboard utilities - Deprecated in favor of core_dashboard.py

NOTICE: This file has been largely superseded by src/ui/core_dashboard.py
The functionality has been moved to the BaseDashboard and UIComponents classes 
which provide better abstraction and inheritance patterns.

Legacy functions kept for backward compatibility with any external code.
New development should use the core_dashboard module instead.

Migration Guide:
- get_shared_market_data() -> BaseDashboard.load_market_data()  
- run_shared_backtest() -> BaseDashboard.run_backtest()
- render_shared_positions_table() -> UIComponents.render_positions_table()
- create_shared_price_chart() -> BaseDashboard.create_price_chart()
- get_shared_account_info() -> BaseDashboard.get_account_info()
"""

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from src.utils.timezone_utils import now_et
from src.data import AlpacaDataProvider
from src.strategy import MomentumStrategy
from src.trade import Backtester
from typing import Dict, List, Any, Optional, Tuple

# Deprecated: Use BaseDashboard.load_market_data() instead
def get_shared_market_data(symbol: str, days: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    DEPRECATED: Use BaseDashboard.load_market_data() instead
    
    Centralized market data loading with consistent error handling
    """
    from src.ui.core_dashboard import BaseDashboard
    dashboard = BaseDashboard("Legacy", "⚠️")
    return dashboard.load_market_data(symbol, days)

# Deprecated: Use BaseDashboard.run_backtest() instead  
def run_shared_backtest(symbols: List[str], days_back: int = 365, 
                       initial_cash: float = 100000, 
                       strategy_type: str = "Momentum") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    DEPRECATED: Use BaseDashboard.run_backtest() instead
    
    Centralized backtesting function to eliminate duplication
    """
    from src.ui.core_dashboard import BaseDashboard
    dashboard = BaseDashboard("Legacy", "⚠️")
    return dashboard.run_backtest(symbols, days_back, initial_cash, strategy_type)

# Deprecated: Use UIComponents.render_positions_table() instead
def render_shared_positions_table(positions: List[Dict]) -> None:
    """
    DEPRECATED: Use UIComponents.render_positions_table() instead
    
    Shared position table rendering with consistent formatting
    """
    from src.ui.core_dashboard import UIComponents
    UIComponents.render_positions_table(positions)

# Deprecated: Use BaseDashboard.create_price_chart() instead
def create_shared_price_chart(symbol: str, data: pd.DataFrame) -> object:
    """
    DEPRECATED: Use BaseDashboard.create_price_chart() instead
    
    Shared chart creation function
    """
    from src.ui.core_dashboard import BaseDashboard
    dashboard = BaseDashboard("Legacy", "⚠️")
    return dashboard.create_price_chart(symbol, data)

# Deprecated: Use BaseDashboard.get_account_info() instead
def get_shared_account_info() -> Tuple[Optional[Dict], Optional[str]]:
    """
    DEPRECATED: Use BaseDashboard.get_account_info() instead
    
    Shared account info retrieval with caching
    """
    from src.ui.core_dashboard import BaseDashboard
    dashboard = BaseDashboard("Legacy", "⚠️")
    return dashboard.get_account_info()


# Modern Dashboard Utilities
def generate_transaction_log(results: Dict[str, Any], symbols: List[str]) -> Optional[str]:
    """
    Generate enhanced transaction log for dashboard download
    """
    try:
        if not results or 'trades_history' not in results:
            return None
        
        trades_df = results['trades_history']
        if trades_df.empty:
            return None
        
        # Enhanced transaction processing
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
            
            # Position tracking logic
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
                'Strategy': 'Modern Dashboard',
                'Notes': 'Backtest transaction'
            }
            
            enhanced_trades.append(enhanced_trade)
            trade_counter += 1
        
        # Create DataFrame and save
        transactions_df = pd.DataFrame(enhanced_trades)
        from src.utils.timezone_utils import now_et
        timestamp = now_et().strftime('%Y%m%d_%H%M%S')
        filename = f"modern_dashboard_transactions_{timestamp}.csv"
        
        transactions_df.to_csv(filename, index=False)
        return filename
        
    except Exception as e:
        return None


def calculate_portfolio_metrics(positions: List[Dict]) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio metrics
    """
    if not positions:
        return {}
    
    active_positions = [p for p in positions if float(p.get('qty', 0)) != 0]
    
    if not active_positions:
        return {}
    
    total_value = sum(float(pos.get('market_value', 0)) for pos in active_positions)
    total_cost = sum(float(pos.get('cost_basis', 0)) for pos in active_positions)
    total_pnl = sum(float(pos.get('unrealized_pl', 0)) for pos in active_positions)
    
    # Calculate concentration metrics
    values = [float(pos.get('market_value', 0)) for pos in active_positions]
    largest_position = max(values) if values else 0
    concentration_ratio = (largest_position / total_value * 100) if total_value > 0 else 0
    
    # Calculate sector diversification (simplified)
    num_positions = len(active_positions)
    diversification_score = min(100, (num_positions / 10) * 100)  # Max 10 positions for full diversification
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'total_return_pct': (total_pnl / total_cost * 100) if total_cost > 0 else 0,
        'num_positions': num_positions,
        'largest_position_pct': concentration_ratio,
        'diversification_score': diversification_score,
        'average_position_size': total_value / num_positions if num_positions > 0 else 0
    }


def calculate_risk_metrics(portfolio_history: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate advanced risk metrics for portfolio
    """
    if portfolio_history.empty or 'portfolio_value' not in portfolio_history.columns:
        return {}
    
    values = portfolio_history['portfolio_value']
    
    # Calculate returns
    returns = values.pct_change().dropna()
    
    if returns.empty:
        return {}
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
    
    # Drawdown analysis
    running_max = values.expanding().max()
    drawdown = (values - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # VaR calculation (5% VaR)
    var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
    
    # Sharpe and Sortino ratios
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
    sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
    
    return {
        'volatility': volatility * 100,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'downside_volatility': downside_volatility * 100,
        'calmar_ratio': (excess_returns / abs(max_drawdown)) if max_drawdown != 0 else 0
    }


def format_crypto_data(crypto_response: Dict) -> Dict[str, Any]:
    """
    Format cryptocurrency data for dashboard display
    """
    if not crypto_response or 'error' in crypto_response:
        return {}
    
    # Extract price information
    price_usd = crypto_response.get('bid_price', 0)
    timestamp = crypto_response.get('timestamp', datetime.now())
    
    # Format for display
    formatted_data = {
        'symbol': 'BTC',
        'price_usd': price_usd,
        'price_eur': price_usd * 0.85,  # Approximate EUR conversion
        'price_gbp': price_usd * 0.75,  # Approximate GBP conversion
        'last_updated': timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp),
        'market_cap': price_usd * 19_500_000,  # Approximate BTC supply
        'volume_24h': price_usd * np.random.uniform(50000, 100000),  # Simulated volume
        'change_24h': np.random.uniform(-5.0, 5.0)  # Simulated change
    }
    
    return formatted_data