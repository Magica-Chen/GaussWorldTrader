import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
    def add_position(self, symbol: str, quantity: float, price: float, 
                    timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol in self.positions:
            current_qty = self.positions[symbol]['quantity']
            current_cost = self.positions[symbol]['cost_basis'] * current_qty
            new_cost = price * quantity
            total_qty = current_qty + quantity
            
            if total_qty != 0:
                new_avg_cost = (current_cost + new_cost) / total_qty
                self.positions[symbol] = {
                    'quantity': total_qty,
                    'cost_basis': new_avg_cost,
                    'last_price': price,
                    'last_updated': timestamp
                }
            else:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {
                'quantity': quantity,
                'cost_basis': price,
                'last_price': price,
                'last_updated': timestamp
            }
        
        cost = quantity * price
        self.cash -= cost
        
        self.transactions.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'timestamp': timestamp,
            'type': 'BUY' if quantity > 0 else 'SELL'
        })
    
    def remove_position(self, symbol: str, quantity: float, price: float,
                       timestamp: Optional[datetime] = None):
        if symbol not in self.positions:
            raise ValueError(f"No position found for symbol {symbol}")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        current_qty = self.positions[symbol]['quantity']
        if abs(quantity) > abs(current_qty):
            raise ValueError(f"Cannot sell {quantity} shares, only {current_qty} available")
        
        new_qty = current_qty - quantity
        proceeds = quantity * price
        self.cash += proceeds
        
        if new_qty == 0:
            del self.positions[symbol]
        else:
            self.positions[symbol]['quantity'] = new_qty
            self.positions[symbol]['last_price'] = price
            self.positions[symbol]['last_updated'] = timestamp
        
        self.transactions.append({
            'symbol': symbol,
            'quantity': -quantity,
            'price': price,
            'cost': -proceeds,
            'timestamp': timestamp,
            'type': 'SELL'
        })
    
    def update_prices(self, price_data: Dict[str, float], timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol]['last_price'] = price
                self.positions[symbol]['last_updated'] = timestamp
    
    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        if current_prices:
            self.update_prices(current_prices)
        
        portfolio_value = self.cash
        for symbol, position in self.positions.items():
            portfolio_value += position['quantity'] * position['last_price']
        
        return portfolio_value
    
    def get_position_value(self, symbol: str) -> float:
        if symbol not in self.positions:
            return 0.0
        position = self.positions[symbol]
        return position['quantity'] * position['last_price']
    
    def get_unrealized_pnl(self, symbol: str) -> float:
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        current_value = position['quantity'] * position['last_price']
        cost_basis = position['quantity'] * position['cost_basis']
        return current_value - cost_basis
    
    def get_total_unrealized_pnl(self) -> float:
        total_pnl = 0.0
        for symbol in self.positions:
            total_pnl += self.get_unrealized_pnl(symbol)
        return total_pnl
    
    def get_realized_pnl(self) -> float:
        realized_pnl = 0.0
        position_lots: Dict[str, List[Dict[str, float]]] = {}

        for transaction in self.transactions:
            symbol = transaction['symbol']
            quantity = float(transaction['quantity'])
            price = float(transaction['price'])

            lots = position_lots.setdefault(symbol, [])

            if quantity > 0:
                lots.append({"qty": quantity, "price": price})
                continue

            sell_qty = abs(quantity)
            while sell_qty > 0 and lots:
                lot = lots[0]
                lot_qty = lot["qty"]
                take_qty = min(lot_qty, sell_qty)
                realized_pnl += take_qty * (price - lot["price"])
                lot["qty"] = lot_qty - take_qty
                sell_qty -= take_qty

                if lot["qty"] <= 0:
                    lots.pop(0)

        return realized_pnl
    
    def get_performance_metrics(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        current_value = self.get_portfolio_value(current_prices)
        total_return = current_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'current_portfolio_value': current_value,
            'total_return': total_return,
            'total_return_percentage': total_return_pct,
            'unrealized_pnl': self.get_total_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'number_of_positions': len(self.positions),
            'number_of_transactions': len(self.transactions)
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'cost_basis': position['cost_basis'],
                'last_price': position['last_price'],
                'market_value': position['quantity'] * position['last_price'],
                'unrealized_pnl': self.get_unrealized_pnl(symbol),
                'last_updated': position['last_updated']
            })
        return pd.DataFrame(data)
    
    def get_transactions_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.transactions)
    
    def record_performance(self, timestamp: Optional[datetime] = None, 
                         current_prices: Optional[Dict[str, float]] = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics = self.get_performance_metrics(current_prices)
        metrics['timestamp'] = timestamp
        self.performance_history.append(metrics)
    
    def get_performance_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.performance_history)


class FinancialMetrics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()

    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        volatility = returns.std()
        if annualize:
            volatility *= np.sqrt(252)  # Assuming 252 trading days per year
        return volatility

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, int, int]:
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = drawdown.min()

        # Find the period of maximum drawdown
        max_dd_end = drawdown.idxmin()
        max_dd_start = cumulative[:max_dd_end].idxmax()

        return max_drawdown, max_dd_start, max_dd_end

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_drawdown, _, _ = FinancialMetrics.calculate_max_drawdown(returns)
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        return returns.quantile(confidence_level)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        var = FinancialMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 0

        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data.iloc[:, 1].var()

        return covariance / market_variance if market_variance > 0 else 0

    @staticmethod
    def calculate_alpha(
        asset_returns: pd.Series, market_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        beta = FinancialMetrics.calculate_beta(asset_returns, market_returns)
        asset_return = asset_returns.mean() * 252
        market_return = market_returns.mean() * 252

        return asset_return - (risk_free_rate + beta * (market_return - risk_free_rate))

    @staticmethod
    def calculate_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        excess_returns = asset_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)

        return (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

    @staticmethod
    def calculate_treynor_ratio(
        returns: pd.Series, market_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        beta = FinancialMetrics.calculate_beta(returns, market_returns)
        annual_return = returns.mean() * 252

        return (annual_return - risk_free_rate) / beta if beta != 0 else 0

    def portfolio_performance_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:

        metrics = {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'annualized_return': (1 + portfolio_returns.mean()) ** 252 - 1,
            'volatility': self.calculate_volatility(portfolio_returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_returns, risk_free_rate),
            'sortino_ratio': self.calculate_sortino_ratio(portfolio_returns, risk_free_rate),
            'calmar_ratio': self.calculate_calmar_ratio(portfolio_returns),
            'max_drawdown': self.calculate_max_drawdown(portfolio_returns)[0],
            'var_5%': self.calculate_var(portfolio_returns, 0.05),
            'cvar_5%': self.calculate_cvar(portfolio_returns, 0.05),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'downside_deviation': portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        }

        if benchmark_returns is not None:
            metrics.update({
                'beta': self.calculate_beta(portfolio_returns, benchmark_returns),
                'alpha': self.calculate_alpha(portfolio_returns, benchmark_returns, risk_free_rate),
                'information_ratio': self.calculate_information_ratio(portfolio_returns, benchmark_returns),
                'treynor_ratio': self.calculate_treynor_ratio(portfolio_returns, benchmark_returns, risk_free_rate)
            })

        return metrics

    def calculate_portfolio_var(
        self,
        weights: np.array,
        returns: pd.DataFrame,
        confidence_level: float = 0.05,
        holding_period: int = 1
    ) -> float:

        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_std = portfolio_returns.std()
        portfolio_mean = portfolio_returns.mean()

        # Assuming normal distribution
        from scipy.stats import norm
        var = norm.ppf(confidence_level, portfolio_mean, portfolio_std)

        # Adjust for holding period
        return var * np.sqrt(holding_period)

    def monte_carlo_var(
        self,
        returns: pd.Series,
        initial_value: float = 1000000,
        confidence_level: float = 0.05,
        time_horizon: int = 252,
        num_simulations: int = 10000
    ) -> Dict[str, float]:

        mean_return = returns.mean()
        std_return = returns.std()

        # Generate random returns
        random_returns = np.random.normal(mean_return, std_return,
                                          (num_simulations, time_horizon))

        # Calculate portfolio values
        portfolio_values = initial_value * (1 + random_returns).cumprod(axis=1)
        final_values = portfolio_values[:, -1]

        # Calculate VaR and CVaR
        var = np.percentile(final_values, confidence_level * 100) - initial_value
        cvar = final_values[final_values <= (initial_value + var)].mean() - initial_value

        return {
            'var': var,
            'cvar': cvar,
            'var_percentage': var / initial_value,
            'cvar_percentage': cvar / initial_value,
            'expected_value': final_values.mean(),
            'std_final_value': final_values.std()
        }

    def rolling_performance_metrics(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        rolling_metrics = pd.DataFrame(index=returns.index)

        rolling_metrics['rolling_return'] = (1 + returns).rolling(window).apply(
            lambda x: x.prod() - 1, raw=True)

        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)

        rolling_metrics['rolling_sharpe'] = returns.rolling(window).apply(
            lambda x: self.calculate_sharpe_ratio(pd.Series(x)), raw=False)

        rolling_metrics['rolling_max_drawdown'] = returns.rolling(window).apply(
            lambda x: self.calculate_max_drawdown(pd.Series(x))[0], raw=False)

        return rolling_metrics.dropna()

    def correlation_analysis(self, returns_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        correlation_matrix = returns_data.corr()

        # Calculate rolling correlations (30-day window)
        rolling_corr = {}
        symbols = returns_data.columns.tolist()

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i + 1:]:
                pair_name = f"{symbol1}_{symbol2}"
                rolling_corr[pair_name] = returns_data[symbol1].rolling(30).corr(
                    returns_data[symbol2]
                )

        return {
            'correlation_matrix': correlation_matrix,
            'rolling_correlation': rolling_corr
        }


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
            window_returns = returns[i - window:i]
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
            window_returns = returns[i - window:i]
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

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Performance Analysis', fontsize=16)

        # Portfolio value over time
        axes[0, 0].plot(self.portfolio_history['timestamp'], self.portfolio_history['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Drawdown chart
        if 'drawdown' in self.portfolio_history.columns:
            axes[0, 1].fill_between(
                self.portfolio_history['timestamp'],
                self.portfolio_history['drawdown'],
                0,
                alpha=0.3,
                color='red'
            )
            axes[0, 1].set_title('Drawdown Over Time')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Drawdown')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Daily returns histogram
        if self.daily_returns:
            axes[1, 0].hist(self.daily_returns, bins=50, alpha=0.7, color='blue')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Returns')
            axes[1, 0].set_ylabel('Frequency')

        # Trade outcome analysis
        if not self.trades_history.empty and 'pnl' in self.trades_history.columns:
            profitable_trades = self.trades_history[self.trades_history['pnl'] > 0]
            losing_trades = self.trades_history[self.trades_history['pnl'] <= 0]
            axes[1, 1].bar(['Profitable', 'Losing'],
                           [len(profitable_trades), len(losing_trades)],
                           color=['green', 'red'])
            axes[1, 1].set_title('Trade Outcomes')
            axes[1, 1].set_ylabel('Number of Trades')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Charts saved to {save_path}")
        else:
            plt.show()


class PortfolioTracker:
    """Advanced portfolio tracking and analysis"""

    def __init__(self, account_manager):
        self.account_manager = account_manager
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _calc_return_pct(equity):
        """Calculate return percentage from equity list"""
        if len(equity) > 1 and equity[0] > 0:
            return ((equity[-1] - equity[0]) / equity[0] * 100)
        return 0

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
            'total_return_pct': self._calc_return_pct(equity),
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
        from src.account.position_manager import PositionManager
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

        pct = lambda x: (x / portfolio_value * 100) if portfolio_value > 0 else 0
        allocation = {
            'total_portfolio_value': portfolio_value,
            'cash': cash,
            'cash_percentage': pct(cash),
            'equity_positions': len(positions),
            'equity_value': equity_value,
            'equity_percentage': pct(equity_value),
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
        from src.account.position_manager import PositionManager
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
            pv = account_status.get('portfolio_value', 0)
            cash = account_status.get('cash', 0)
            bp = account_status.get('buying_power', 0)
            eq_change = account_status.get('equity_change', 0)
            eq_pct = account_status.get('equity_change_percentage', 0)
            report += f"""* Portfolio Value: ${pv:,.2f}
* Cash Available: ${cash:,.2f}
* Buying Power: ${bp:,.2f}
* Daily P&L: ${eq_change:,.2f} ({eq_pct:+.2f}%)
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

        import matplotlib.pyplot as plt

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
