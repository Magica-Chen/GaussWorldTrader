"""
Performance Analysis Module for Backtesting

Enhanced performance metrics and visualization capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime
import seaborn as sns

class PerformanceAnalyzer:
    """Advanced performance analysis for backtest results"""
    
    def __init__(self, backtest_results: Dict[str, Any]):
        self.results = backtest_results
        self.portfolio_history = backtest_results.get('portfolio_history', pd.DataFrame())
        self.trades_history = backtest_results.get('trades_history', pd.DataFrame())
        self.daily_returns = backtest_results.get('daily_returns', [])
        
    def calculate_advanced_metrics(self) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        if self.daily_returns:
            returns = np.array(self.daily_returns)
            
            # Risk metrics
            var_95 = np.percentile(returns, 5)  # Value at Risk (95%)
            cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
            
            # Ratios
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio()
            
            # Rolling metrics
            rolling_sharpe = self._calculate_rolling_sharpe(returns)
            rolling_volatility = self._calculate_rolling_volatility(returns)
            
            return {
                'value_at_risk_95': var_95,
                'conditional_var_95': cvar_95,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'avg_rolling_sharpe': np.mean(rolling_sharpe) if rolling_sharpe else 0,
                'avg_rolling_volatility': np.mean(rolling_volatility) if rolling_volatility else 0,
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns),
                'tail_ratio': self._calculate_tail_ratio(returns)
            }
        
        return {}
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation focus)"""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        
        if downside_deviation == 0:
            return float('inf')
        
        return (np.mean(excess_returns) * 252) / downside_deviation
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = self.results.get('annualized_return', 0)
        max_drawdown = self.results.get('max_drawdown', 0)
        
        if max_drawdown == 0:
            return float('inf')
        
        return annual_return / max_drawdown
    
    def _calculate_rolling_sharpe(self, returns: np.ndarray, window: int = 30) -> List[float]:
        """Calculate rolling Sharpe ratio"""
        if len(returns) < window:
            return []
        
        rolling_sharpe = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)
            
            if std_return > 0:
                sharpe = (mean_return * np.sqrt(252)) / (std_return * np.sqrt(252))
                rolling_sharpe.append(sharpe)
        
        return rolling_sharpe
    
    def _calculate_rolling_volatility(self, returns: np.ndarray, window: int = 30) -> List[float]:
        """Calculate rolling volatility"""
        if len(returns) < window:
            return []
        
        rolling_vol = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            volatility = np.std(window_returns) * np.sqrt(252)
            rolling_vol.append(volatility)
        
        return rolling_vol
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        return np.mean(((returns - mean_return) / std_return) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        if len(returns) < 20:
            return 0
        
        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)
        
        if percentile_5 == 0:
            return float('inf')
        
        return abs(percentile_95 / percentile_5)
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        base_metrics = self.results
        advanced_metrics = self.calculate_advanced_metrics()
        
        report = f"""
🌍 Gauss World Trader - Performance Analysis Report
================================================

BASIC METRICS:
--------------
• Period: {base_metrics.get('start_date', 'N/A')} to {base_metrics.get('end_date', 'N/A')}
• Initial Value: ${base_metrics.get('initial_value', 0):,.2f}
• Final Value: ${base_metrics.get('final_value', 0):,.2f}
• Total Return: {base_metrics.get('total_return_percentage', 0):.2f}%
• Annualized Return: {base_metrics.get('annualized_return_percentage', 0):.2f}%
• Volatility: {base_metrics.get('volatility', 0):.2f}
• Sharpe Ratio: {base_metrics.get('sharpe_ratio', 0):.2f}
• Max Drawdown: {base_metrics.get('max_drawdown_percentage', 0):.2f}%

ADVANCED RISK METRICS:
---------------------
• Value at Risk (95%): {advanced_metrics.get('value_at_risk_95', 0):.4f}
• Conditional VaR (95%): {advanced_metrics.get('conditional_var_95', 0):.4f}
• Sortino Ratio: {advanced_metrics.get('sortino_ratio', 0):.2f}
• Calmar Ratio: {advanced_metrics.get('calmar_ratio', 0):.2f}
• Skewness: {advanced_metrics.get('skewness', 0):.2f}
• Kurtosis: {advanced_metrics.get('kurtosis', 0):.2f}
• Tail Ratio: {advanced_metrics.get('tail_ratio', 0):.2f}

TRADING STATISTICS:
------------------
• Total Trades: {base_metrics.get('total_trades', 0)}
• Winning Trades: {base_metrics.get('winning_trades', 0)}
• Losing Trades: {base_metrics.get('losing_trades', 0)}
• Win Rate: {base_metrics.get('win_rate', 0):.2f}%
• Profit Factor: {base_metrics.get('profit_factor', 0):.2f}
• Total Profit: ${base_metrics.get('total_profit', 0):,.2f}
• Total Loss: ${base_metrics.get('total_loss', 0):,.2f}

ROLLING METRICS:
---------------
• Avg Rolling Sharpe (30d): {advanced_metrics.get('avg_rolling_sharpe', 0):.2f}
• Avg Rolling Volatility (30d): {advanced_metrics.get('avg_rolling_volatility', 0):.2f}
"""
        
        return report
    
    def plot_performance_charts(self, save_path: Optional[str] = None) -> None:
        """Generate performance visualization charts"""
        if self.portfolio_history.empty:
            print("No portfolio history data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gauss World Trader - Performance Analysis', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(self.portfolio_history['date'], self.portfolio_history['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Drawdown chart
        if 'drawdowns' in self.results:
            drawdowns = self.results['drawdowns']
            axes[0, 1].fill_between(range(len(drawdowns)), drawdowns, alpha=0.7, color='red')
            axes[0, 1].set_title('Drawdown Over Time')
            axes[0, 1].set_xlabel('Trading Days')
            axes[0, 1].set_ylabel('Drawdown (%)')
        
        # Daily returns distribution
        if self.daily_returns:
            axes[1, 0].hist(self.daily_returns, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].set_ylabel('Frequency')
        
        # Rolling Sharpe ratio
        if self.daily_returns:
            rolling_sharpe = self._calculate_rolling_sharpe(np.array(self.daily_returns))
            if rolling_sharpe:
                axes[1, 1].plot(rolling_sharpe)
                axes[1, 1].set_title('Rolling Sharpe Ratio (30-day)')
                axes[1, 1].set_xlabel('Trading Days')
                axes[1, 1].set_ylabel('Sharpe Ratio')
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance charts saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()