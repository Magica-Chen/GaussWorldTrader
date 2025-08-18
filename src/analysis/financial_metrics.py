import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

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
    def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: float = 0.02) -> float:
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
    def calculate_treynor_ratio(returns: pd.Series, market_returns: pd.Series, 
                               risk_free_rate: float = 0.02) -> float:
        beta = FinancialMetrics.calculate_beta(returns, market_returns)
        annual_return = returns.mean() * 252
        
        return (annual_return - risk_free_rate) / beta if beta != 0 else 0
    
    def portfolio_performance_metrics(self, portfolio_returns: pd.Series, 
                                    benchmark_returns: Optional[pd.Series] = None,
                                    risk_free_rate: float = 0.02) -> Dict[str, float]:
        
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
    
    def calculate_portfolio_var(self, weights: np.array, returns: pd.DataFrame, 
                               confidence_level: float = 0.05, 
                               holding_period: int = 1) -> float:
        
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_std = portfolio_returns.std()
        portfolio_mean = portfolio_returns.mean()
        
        # Assuming normal distribution
        from scipy.stats import norm
        var = norm.ppf(confidence_level, portfolio_mean, portfolio_std)
        
        # Adjust for holding period
        return var * np.sqrt(holding_period)
    
    def monte_carlo_var(self, returns: pd.Series, initial_value: float = 1000000,
                       confidence_level: float = 0.05, time_horizon: int = 252,
                       num_simulations: int = 10000) -> Dict[str, float]:
        
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
            for symbol2 in symbols[i+1:]:
                pair_name = f"{symbol1}_{symbol2}"
                rolling_corr[pair_name] = returns_data[symbol1].rolling(30).corr(
                    returns_data[symbol2])
        
        rolling_corr_df = pd.DataFrame(rolling_corr)
        
        return {
            'static_correlation': correlation_matrix,
            'rolling_correlation': rolling_corr_df
        }
    
    def risk_adjusted_metrics(self, returns: pd.Series, 
                             benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        
        metrics = {}
        
        # Basic risk metrics
        metrics['volatility'] = self.calculate_volatility(returns)
        metrics['var_95'] = self.calculate_var(returns, 0.05)
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.05)
        
        # Drawdown metrics
        max_dd, start_date, end_date = self.calculate_max_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_duration'] = (end_date - start_date).days if hasattr(start_date, 'days') else 0
        
        # Risk-adjusted return metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
        
        if benchmark_returns is not None:
            metrics['beta'] = self.calculate_beta(returns, benchmark_returns)
            metrics['alpha'] = self.calculate_alpha(returns, benchmark_returns)
            metrics['treynor_ratio'] = self.calculate_treynor_ratio(returns, benchmark_returns)
            metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
        
        return metrics
    
    def performance_attribution(self, portfolio_returns: pd.DataFrame, 
                               weights: pd.DataFrame,
                               benchmark_returns: pd.Series) -> Dict[str, pd.Series]:
        
        # Calculate asset contributions to portfolio return
        weighted_returns = portfolio_returns.multiply(weights, axis=0)
        asset_contributions = weighted_returns.div(weighted_returns.sum(axis=1), axis=0)
        
        # Calculate excess returns over benchmark
        excess_returns = portfolio_returns.subtract(benchmark_returns, axis=0)
        
        return {
            'asset_contributions': asset_contributions,
            'excess_returns': excess_returns,
            'total_attribution': weighted_returns.sum(axis=1) - benchmark_returns
        }