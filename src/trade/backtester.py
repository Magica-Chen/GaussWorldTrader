import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from .portfolio import Portfolio

class Backtester:
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.01):
        self.initial_cash = initial_cash
        self.commission = commission
        self.portfolio = Portfolio(initial_cash)
        self.data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    def add_data(self, symbol: str, data: pd.DataFrame):
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        self.data[symbol] = data.copy()
        self.logger.info(f"Added data for {symbol}: {len(data)} rows")
    
    def run_backtest(self, strategy_func: Callable, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        
        if not self.data:
            raise ValueError("No data loaded for backtesting")
        
        if symbols is None:
            symbols = list(self.data.keys())
        
        all_dates = set()
        for symbol in symbols:
            if symbol in self.data:
                all_dates.update(self.data[symbol].index)
        
        if not all_dates:
            raise ValueError("No valid data found for specified symbols")
        
        # Normalize all dates to be timezone-naive for comparison
        def normalize_datetime(dt):
            if hasattr(dt, 'tz_localize'):
                return dt.tz_localize(None) if dt.tz is None else dt.tz_convert(None).tz_localize(None)
            elif hasattr(dt, 'replace') and hasattr(dt, 'tzinfo') and dt.tzinfo:
                return dt.replace(tzinfo=None)
            return dt
        
        # Normalize all dates
        all_dates_normalized = [normalize_datetime(d) for d in all_dates]
        date_range = sorted(all_dates_normalized)
        
        if start_date:
            start_date_normalized = normalize_datetime(start_date)
            date_range = [d for d in date_range if d >= start_date_normalized]
        if end_date:
            end_date_normalized = normalize_datetime(end_date)
            date_range = [d for d in date_range if d <= end_date_normalized]
        
        if not date_range:
            raise ValueError("No data in specified date range")
        
        self.logger.info(f"Starting backtest from {date_range[0]} to {date_range[-1]}")
        
        portfolio_values = []
        daily_returns = []
        trades_log = []
        
        for i, current_date in enumerate(date_range):
            current_prices = {}
            current_data = {}
            
            for symbol in symbols:
                if symbol in self.data:
                    # Normalize the dataframe index for comparison
                    df = self.data[symbol].copy()
                    if hasattr(df.index, 'tz_localize'):
                        if df.index.tz is None:
                            df.index = df.index.tz_localize(None)
                        else:
                            df.index = df.index.tz_convert(None).tz_localize(None)
                    
                    # Find the closest date match
                    try:
                        if current_date in df.index:
                            row = df.loc[current_date]
                        else:
                            # Find nearest date
                            nearest_idx = df.index.get_indexer([current_date], method='nearest')[0]
                            if nearest_idx >= 0:
                                row = df.iloc[nearest_idx]
                            else:
                                continue
                        
                        current_prices[symbol] = row['close']
                        current_data[symbol] = row.to_dict()
                    except (KeyError, IndexError):
                        continue
            
            if not current_prices:
                continue
            
            historical_data = {}
            for symbol in symbols:
                if symbol in self.data:
                    df = self.data[symbol].copy()
                    # Normalize timezone for historical data
                    if hasattr(df.index, 'tz_localize'):
                        if df.index.tz is None:
                            df.index = df.index.tz_localize(None)
                        else:
                            df.index = df.index.tz_convert(None).tz_localize(None)
                    
                    historical_data[symbol] = df[df.index <= current_date]
            
            signals = strategy_func(
                current_date=current_date,
                current_prices=current_prices,
                current_data=current_data,
                historical_data=historical_data,
                portfolio=self.portfolio
            )
            
            for signal in signals:
                if self._execute_signal(signal, current_prices, current_date):
                    trades_log.append({
                        'date': current_date,
                        'symbol': signal['symbol'],
                        'action': signal['action'],
                        'quantity': signal['quantity'],
                        'price': current_prices.get(signal['symbol'], 0),
                        'portfolio_value': self.portfolio.get_portfolio_value(current_prices)
                    })
            
            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            portfolio_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'positions_value': portfolio_value - self.portfolio.cash
            })
            
            if len(portfolio_values) > 1:
                prev_value = portfolio_values[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
                daily_returns.append(daily_return)
        
        self.results = self._calculate_performance_metrics(portfolio_values, daily_returns, trades_log)
        self.logger.info("Backtest completed successfully")
        
        return self.results
    
    def _execute_signal(self, signal: Dict[str, Any], current_prices: Dict[str, float], 
                       current_date: datetime) -> bool:
        try:
            symbol = signal['symbol']
            action = signal['action'].upper()
            quantity = signal['quantity']
            
            if symbol not in current_prices:
                return False
            
            price = current_prices[symbol]
            commission_cost = abs(quantity) * price * self.commission
            
            if action == 'BUY':
                total_cost = quantity * price + commission_cost
                if total_cost <= self.portfolio.cash:
                    self.portfolio.add_position(symbol, quantity, price, current_date)
                    self.portfolio.cash -= commission_cost
                    return True
            
            elif action == 'SELL':
                if symbol in self.portfolio.positions:
                    available_qty = self.portfolio.positions[symbol]['quantity']
                    if quantity <= available_qty:
                        self.portfolio.remove_position(symbol, quantity, price, current_date)
                        self.portfolio.cash -= commission_cost
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False
    
    def _calculate_performance_metrics(self, portfolio_values: List[Dict], 
                                     daily_returns: List[float], 
                                     trades_log: List[Dict]) -> Dict[str, Any]:
        
        if not portfolio_values:
            return {}
        
        df_portfolio = pd.DataFrame(portfolio_values)
        df_trades = pd.DataFrame(trades_log)
        
        initial_value = portfolio_values[0]['portfolio_value']
        final_value = portfolio_values[-1]['portfolio_value']
        
        total_return = (final_value - initial_value) / initial_value
        total_return_pct = total_return * 100
        
        trading_days = len(portfolio_values)
        years = trading_days / 252
        
        if years > 0:
            annualized_return = (final_value / initial_value) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        if daily_returns:
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
            
            peak = initial_value
            max_drawdown = 0
            drawdowns = []
            
            for pv in portfolio_values:
                value = pv['portfolio_value']
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            drawdowns = []
        
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for trade in trades_log:
            if trade['action'].upper() == 'SELL':
                symbol = trade['symbol']
                if symbol in self.portfolio.positions:
                    pnl = self.portfolio.get_unrealized_pnl(symbol)
                    if pnl > 0:
                        winning_trades += 1
                        total_profit += pnl
                    elif pnl < 0:
                        losing_trades += 1
                        total_loss += abs(pnl)
        
        total_trades = len([t for t in trades_log if t['action'].upper() in ['BUY', 'SELL']])
        win_rate = (winning_trades / (winning_trades + losing_trades)) * 100 if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_percentage': total_return_pct,
            'annualized_return': annualized_return,
            'annualized_return_percentage': annualized_return * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': (total_profit / total_loss) if total_loss > 0 else float('inf'),
            'trading_days': trading_days,
            'start_date': portfolio_values[0]['date'],
            'end_date': portfolio_values[-1]['date'],
            'portfolio_history': df_portfolio,
            'trades_history': df_trades,
            'daily_returns': daily_returns,
            'drawdowns': drawdowns
        }
    
    def get_results_summary(self) -> str:
        if not self.results:
            return "No backtest results available"
        
        summary = f"""
Backtest Results Summary
========================
Period: {self.results['start_date']} to {self.results['end_date']}
Initial Value: ${self.results['initial_value']:,.2f}
Final Value: ${self.results['final_value']:,.2f}
Total Return: {self.results['total_return_percentage']:.2f}%
Annualized Return: {self.results['annualized_return_percentage']:.2f}%
Volatility: {self.results['volatility']:.2f}
Sharpe Ratio: {self.results['sharpe_ratio']:.2f}
Max Drawdown: {self.results['max_drawdown_percentage']:.2f}%

Trading Statistics:
Total Trades: {self.results['total_trades']}
Winning Trades: {self.results['winning_trades']}
Losing Trades: {self.results['losing_trades']}
Win Rate: {self.results['win_rate']:.2f}%
Profit Factor: {self.results['profit_factor']:.2f}
"""
        return summary
    
    def reset(self):
        self.portfolio = Portfolio(self.initial_cash)
        self.results = {}
        self.logger.info("Backtester reset to initial state")