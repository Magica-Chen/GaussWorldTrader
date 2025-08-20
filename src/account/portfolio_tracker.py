"""
Portfolio Tracking and Analysis

Comprehensive portfolio monitoring and performance tracking
"""

import requests
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class PortfolioTracker:
    """Advanced portfolio tracking and analysis"""
    
    def __init__(self, account_manager):
        self.account_manager = account_manager
        self.logger = logging.getLogger(__name__)
    
    def get_portfolio_performance(self, period: str = '1D', 
                                timeframe: str = '1Min') -> Dict[str, Any]:
        """Get portfolio performance data"""
        portfolio_history = self.account_manager.get_portfolio_history(period, timeframe)
        
        if 'error' in portfolio_history:
            return portfolio_history
        
        # Process portfolio history data
        timestamps = portfolio_history.get('timestamp', [])
        equity = portfolio_history.get('equity', [])
        profit_loss = portfolio_history.get('profit_loss', [])
        profit_loss_pct = portfolio_history.get('profit_loss_pct', [])
        
        if not timestamps or not equity:
            return {"error": "No portfolio history data available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([datetime.fromtimestamp(ts) for ts in timestamps]),
            'equity': equity,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        })
        
        # Calculate performance metrics
        performance = {
            'period': period,
            'timeframe': timeframe,
            'start_equity': equity[0] if equity else 0,
            'end_equity': equity[-1] if equity else 0,
            'total_return': equity[-1] - equity[0] if len(equity) > 1 else 0,
            'total_return_pct': ((equity[-1] - equity[0]) / equity[0] * 100) if len(equity) > 1 and equity[0] > 0 else 0,
            'max_equity': max(equity) if equity else 0,
            'min_equity': min(equity) if equity else 0,
            'current_drawdown': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'data_points': len(equity),
            'raw_data': portfolio_history
        }
        
        # Calculate drawdown
        if len(equity) > 1:
            peak = equity[0]
            max_drawdown = 0
            current_drawdown = 0
            
            for value in equity:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                current_drawdown = (peak - equity[-1]) / peak * 100
            
            performance['max_drawdown'] = max_drawdown
            performance['current_drawdown'] = current_drawdown
        
        # Calculate volatility (if we have enough data points)
        if len(profit_loss_pct) > 1:
            returns = [pct for pct in profit_loss_pct if pct is not None]
            if returns:
                performance['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        
        return performance
    
    def get_asset_allocation(self) -> Dict[str, Any]:
        """Analyze current asset allocation"""
        # Get account info for cash
        account = self.account_manager.get_account()
        if 'error' in account:
            return account
        
        # Get positions for holdings
        from .position_manager import PositionManager
        position_manager = PositionManager(self.account_manager)
        positions = position_manager.get_all_positions()
        
        if not positions or (len(positions) == 1 and 'error' in positions[0]):
            cash = float(account.get('cash', 0))
            portfolio_value = float(account.get('portfolio_value', cash))
            
            return {
                'total_portfolio_value': portfolio_value,
                'cash': cash,
                'cash_percentage': 100.0,
                'equity_positions': 0,
                'equity_value': 0,
                'equity_percentage': 0,
                'asset_breakdown': {'CASH': 100.0},
                'position_count': 0
            }
        
        # Calculate allocation
        cash = float(account.get('cash', 0))
        portfolio_value = float(account.get('portfolio_value', 0))
        equity_value = 0
        position_values = {}
        
        for pos in positions:
            try:
                symbol = pos.get('symbol', 'UNKNOWN')
                market_value = abs(float(pos.get('market_value', 0)))
                equity_value += market_value
                position_values[symbol] = market_value
            except (ValueError, TypeError):
                continue
        
        allocation = {
            'total_portfolio_value': portfolio_value,
            'cash': cash,
            'cash_percentage': (cash / portfolio_value * 100) if portfolio_value > 0 else 0,
            'equity_positions': len(positions),
            'equity_value': equity_value,
            'equity_percentage': (equity_value / portfolio_value * 100) if portfolio_value > 0 else 0,
            'position_count': len(positions)
        }
        
        # Asset breakdown by position
        asset_breakdown = {'CASH': allocation['cash_percentage']}
        for symbol, value in position_values.items():
            percentage = (value / portfolio_value * 100) if portfolio_value > 0 else 0
            asset_breakdown[symbol] = percentage
        
        allocation['asset_breakdown'] = asset_breakdown
        allocation['top_holdings'] = sorted(
            [(symbol, pct) for symbol, pct in asset_breakdown.items() if symbol != 'CASH'],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        return allocation
    
    def calculate_risk_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Calculate portfolio risk metrics"""
        # Get portfolio performance for analysis
        performance = self.get_portfolio_performance('1M', '1D')
        
        if 'error' in performance:
            return performance
        
        # Get positions for concentration risk
        from .position_manager import PositionManager
        position_manager = PositionManager(self.account_manager)
        positions_analysis = position_manager.analyze_positions()
        
        if 'error' in positions_analysis:
            return {"error": "Could not analyze positions for risk calculation"}
        
        # Extract risk data
        raw_data = performance.get('raw_data', {})
        profit_loss_pct = raw_data.get('profit_loss_pct', [])
        
        risk_metrics = {
            'max_drawdown': performance.get('max_drawdown', 0),
            'current_drawdown': performance.get('current_drawdown', 0),
            'volatility': performance.get('volatility', 0),
            'total_positions': positions_analysis.get('total_positions', 0),
            'concentration_risk': 'Low',
            'position_risk_score': 0,
            'portfolio_beta': 'N/A',  # Would need market data for calculation
            'var_95': 0,  # Value at Risk
            'sharpe_ratio': 'N/A'  # Would need risk-free rate
        }
        
        # Calculate VaR if we have daily returns
        if profit_loss_pct and len(profit_loss_pct) > 5:
            daily_returns = [pct for pct in profit_loss_pct if pct is not None]
            if daily_returns:
                risk_metrics['var_95'] = np.percentile(daily_returns, 5)
        
        # Assess concentration risk
        allocation = self.get_asset_allocation()
        if 'top_holdings' in allocation:
            top_holdings = allocation['top_holdings']
            if top_holdings:
                largest_position_pct = top_holdings[0][1] if top_holdings else 0
                top_5_concentration = sum([holding[1] for holding in top_holdings[:5]])
                
                if largest_position_pct > 20:
                    risk_metrics['concentration_risk'] = 'High'
                    risk_metrics['position_risk_score'] = 3
                elif largest_position_pct > 10 or top_5_concentration > 60:
                    risk_metrics['concentration_risk'] = 'Medium'
                    risk_metrics['position_risk_score'] = 2
                else:
                    risk_metrics['concentration_risk'] = 'Low'
                    risk_metrics['position_risk_score'] = 1
        
        return risk_metrics
    
    def generate_portfolio_report(self) -> str:
        """Generate comprehensive portfolio report"""
        # Get all necessary data
        account_status = self.account_manager.get_trading_account_status()
        performance = self.get_portfolio_performance('1D', '5Min')
        allocation = self.get_asset_allocation()
        risk_metrics = self.calculate_risk_metrics()
        
        # Generate report
        report = f"""
🌍 GAUSS WORLD TRADER - PORTFOLIO REPORT
=======================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO OVERVIEW:
------------------
"""
        
        if 'error' not in account_status:
            report += f"""• Portfolio Value: ${account_status.get('portfolio_value', 0):,.2f}
• Cash Available: ${account_status.get('cash', 0):,.2f}
• Buying Power: ${account_status.get('buying_power', 0):,.2f}
• Daily P&L: ${account_status.get('equity_change', 0):,.2f} ({account_status.get('equity_change_percentage', 0):+.2f}%)
"""
        
        # Performance metrics
        if 'error' not in performance:
            report += f"""
PERFORMANCE METRICS:
-------------------
• Period Return: {performance.get('total_return_pct', 0):+.2f}%
• Current Drawdown: {performance.get('current_drawdown', 0):.2f}%
• Max Drawdown: {performance.get('max_drawdown', 0):.2f}%
• Volatility (Ann.): {performance.get('volatility', 0):.2f}%
"""
        
        # Asset allocation
        if 'error' not in allocation:
            report += f"""
ASSET ALLOCATION:
----------------
• Cash: {allocation.get('cash_percentage', 0):.1f}%
• Equities: {allocation.get('equity_percentage', 0):.1f}%
• Total Positions: {allocation.get('position_count', 0)}
"""
            
            # Top holdings
            top_holdings = allocation.get('top_holdings', [])
            if top_holdings:
                report += """
TOP HOLDINGS:
------------
"""
                for i, (symbol, percentage) in enumerate(top_holdings[:5], 1):
                    report += f"{i}. {symbol}: {percentage:.1f}%\n"
        
        # Risk assessment
        if 'error' not in risk_metrics:
            report += f"""
RISK ASSESSMENT:
---------------
• Concentration Risk: {risk_metrics.get('concentration_risk', 'Unknown')}
• Risk Score: {risk_metrics.get('position_risk_score', 0)}/3
• Value at Risk (95%): {risk_metrics.get('var_95', 0):.2f}%
• Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2f}%
"""
        
        report += f"""
ACCOUNT STATUS:
--------------
• Account Active: {account_status.get('status', 'Unknown') == 'ACTIVE'}
• Trading Enabled: {not account_status.get('trading_blocked', True)}
• Paper Trading: {'Yes' if 'paper' in self.account_manager.base_url else 'No'}

Generated by Gauss World Trader - Named after Carl Friedrich Gauss
Report Timestamp: {datetime.now().isoformat()}
"""
        
        return report
    
    def plot_portfolio_performance(self, period: str = '1D', save_path: str = None):
        """Plot portfolio performance chart"""
        performance = self.get_portfolio_performance(period, '5Min')
        
        if 'error' in performance:
            print(f"Error plotting performance: {performance['error']}")
            return
        
        raw_data = performance.get('raw_data', {})
        timestamps = raw_data.get('timestamp', [])
        equity = raw_data.get('equity', [])
        
        if not timestamps or not equity:
            print("No data available for plotting")
            return
        
        # Convert timestamps
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, linewidth=2, color='blue')
        plt.title(f'Portfolio Performance - {period}', fontsize=16)
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add performance stats
        total_return_pct = performance.get('total_return_pct', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        
        plt.figtext(0.02, 0.02, 
                   f'Return: {total_return_pct:+.2f}% | Max DD: {max_drawdown:.2f}%',
                   fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio chart saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()