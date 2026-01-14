#!/usr/bin/env python3
"""
Gauss World Trader - Momentum Strategy Backtest Example
Repository: https://github.com/Magica-Chen/GaussWorldTrader
Author: Magica Chen
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from src.data.alpaca_provider import AlpacaDataProvider
from src.trade.backtester import Backtester
from src.strategy import MomentumStrategy

def generate_pnl_plot(results):
    """Generate Profit & Loss plot from backtest results"""
    try:
        if not results or 'portfolio_history' not in results:
            print("❌ No portfolio history data available for plotting")
            return
        
        portfolio_df = results['portfolio_history']
        if portfolio_df.empty:
            print("❌ Portfolio history is empty")
            return
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Gauss World Trader - Momentum Strategy Results', fontsize=16, fontweight='bold')
        
        # Portfolio value over time
        dates = portfolio_df['date']
        portfolio_values = portfolio_df['portfolio_value']
        initial_value = portfolio_values.iloc[0]
        
        ax1.plot(dates, portfolio_values, linewidth=2, color='blue', label='Portfolio Value')
        ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Profit/Loss percentage
        pnl_pct = ((portfolio_values - initial_value) / initial_value) * 100
        colors = ['red' if x < 0 else 'green' for x in pnl_pct]
        
        ax2.fill_between(dates, pnl_pct, 0, alpha=0.3, color='green', where=(pnl_pct >= 0))
        ax2.fill_between(dates, pnl_pct, 0, alpha=0.3, color='red', where=(pnl_pct < 0))
        ax2.plot(dates, pnl_pct, linewidth=2, color='black')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_title('Profit & Loss (%)')
        ax2.set_ylabel('P&L (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add statistics text
        final_return = pnl_pct.iloc[-1]
        max_gain = pnl_pct.max()
        max_loss = pnl_pct.min()
        
        stats_text = f"""
Final Return: {final_return:.2f}%
Max Gain: {max_gain:.2f}%
Max Loss: {max_loss:.2f}%
Total Trades: {results.get('total_trades', 0)}
Win Rate: {results.get('win_rate', 0):.1f}%
Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
        """
        
        ax2.text(0.02, 0.98, stats_text.strip(), transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        plot_filename = output_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ P&L plot saved as: {plot_filename}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error generating P&L plot: {e}")

def generate_transaction_log(results, symbols):
    """Generate detailed transaction log CSV file"""
    try:
        if not results or 'trades_history' not in results:
            print("❌ No trade history data available for transaction log")
            return
        
        trades_df = results['trades_history']
        if trades_df.empty:
            print("❌ Trade history is empty")
            return
        
        # Enhanced transaction log with additional calculated fields
        enhanced_trades = []
        portfolio_values = results.get('portfolio_history', pd.DataFrame())
        
        # Track positions for P&L calculation
        positions = {}  # symbol: {'qty': qty, 'avg_cost': cost, 'total_cost': total}
        trade_counter = 1
        
        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            action = trade['action'].upper()
            quantity = abs(trade['quantity'])  # Ensure positive
            price = trade.get('price', 0)
            trade_date = trade['date']
            
            # Calculate trade value
            trade_value = quantity * price
            
            # Calculate position before trade
            position_before = positions.get(symbol, {'qty': 0, 'avg_cost': 0, 'total_cost': 0})
            
            # Update position tracking
            if action == 'BUY':
                new_qty = position_before['qty'] + quantity
                new_total_cost = position_before['total_cost'] + trade_value
                new_avg_cost = new_total_cost / new_qty if new_qty > 0 else 0
                
                positions[symbol] = {
                    'qty': new_qty,
                    'avg_cost': new_avg_cost,
                    'total_cost': new_total_cost
                }
                
                realized_pnl = 0  # No P&L on buy
                
            elif action == 'SELL':
                if position_before['qty'] >= quantity:
                    # Calculate realized P&L for the sold shares
                    cost_basis = position_before['avg_cost'] * quantity
                    proceeds = trade_value
                    realized_pnl = proceeds - cost_basis
                    
                    # Update position
                    new_qty = position_before['qty'] - quantity
                    new_total_cost = position_before['total_cost'] - cost_basis
                    new_avg_cost = position_before['avg_cost']  # Avg cost stays same
                    
                    positions[symbol] = {
                        'qty': new_qty,
                        'avg_cost': new_avg_cost if new_qty > 0 else 0,
                        'total_cost': new_total_cost
                    }
                else:
                    print(f"⚠️ Warning: Trying to sell {quantity} {symbol} but only have {position_before['qty']}")
                    realized_pnl = 0
            
            # Get portfolio value at this date
            portfolio_value = 0
            if not portfolio_values.empty:
                try:
                    portfolio_row = portfolio_values[portfolio_values['date'] == trade_date]
                    if not portfolio_row.empty:
                        portfolio_value = portfolio_row['portfolio_value'].iloc[0]
                except:
                    portfolio_value = 0
            
            # Calculate position after trade
            position_after = positions.get(symbol, {'qty': 0, 'avg_cost': 0, 'total_cost': 0})
            
            # Enhanced trade record
            enhanced_trade = {
                'Trade_ID': trade_counter,
                'Date': trade_date.strftime('%Y-%m-%d') if hasattr(trade_date, 'strftime') else str(trade_date),
                'Time': trade_date.strftime('%H:%M:%S') if hasattr(trade_date, 'strftime') else '09:30:00',
                'Symbol': symbol,
                'Action': action,
                'Quantity': quantity,
                'Price': f"{price:.4f}",
                'Trade_Value': f"{trade_value:.2f}",
                'Commission': f"{trade_value * 0.01:.2f}",  # 1% commission assumption
                'Net_Amount': f"{trade_value * (0.99 if action == 'BUY' else 1.01):.2f}",
                'Position_Before': position_before['qty'],
                'Position_After': position_after['qty'],
                'Avg_Cost_Basis': f"{position_after['avg_cost']:.4f}",
                'Realized_PnL': f"{realized_pnl:.2f}",
                'Portfolio_Value': f"{portfolio_value:.2f}",
                'Notes': f"Momentum strategy signal"
            }
            
            enhanced_trades.append(enhanced_trade)
            trade_counter += 1
        
        # Create DataFrame and save to CSV
        transactions_df = pd.DataFrame(enhanced_trades)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"transactions_{timestamp}.csv"

        transactions_df.to_csv(filename, index=False)
        
        # Display summary
        print(f"✅ Transaction log saved as: {filename}")
        print(f"📊 Total transactions: {len(transactions_df)}")
        
        # Show sample of transactions
        print(f"\\n📋 Sample transactions (first 5):")
        print(transactions_df.head().to_string(index=False))
        
        # Trade summary statistics
        buy_trades = transactions_df[transactions_df['Action'] == 'BUY']
        sell_trades = transactions_df[transactions_df['Action'] == 'SELL']
        
        total_bought = buy_trades['Trade_Value'].astype(float).sum()
        total_sold = sell_trades['Trade_Value'].astype(float).sum()
        total_realized_pnl = transactions_df['Realized_PnL'].astype(float).sum()
        total_commissions = transactions_df['Commission'].astype(float).sum()
        
        print(f"\\n💰 Trading Summary:")
        print(f"  Total Buy Value: ${total_bought:,.2f}")
        print(f"  Total Sell Value: ${total_sold:,.2f}")
        print(f"  Total Realized P&L: ${total_realized_pnl:,.2f}")
        print(f"  Total Commissions: ${total_commissions:,.2f}")
        print(f"  Net P&L (after commissions): ${total_realized_pnl - total_commissions:,.2f}")
        
        # Position summary
        final_positions = {}
        for symbol in symbols:
            pos = positions.get(symbol, {'qty': 0, 'avg_cost': 0})
            if pos['qty'] != 0:
                final_positions[symbol] = pos
        
        if final_positions:
            print(f"\\n📊 Final Positions:")
            for symbol, pos in final_positions.items():
                print(f"  {symbol}: {pos['qty']} shares @ ${pos['avg_cost']:.2f} avg cost")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error generating transaction log: {e}")
        return None

def momentum_backtest_example():
    """Gauss World Trader - Example of running a momentum strategy backtest"""
    
    # Initialize components
    try:
        data_provider = AlpacaDataProvider()
        backtester = Backtester(initial_cash=100000, commission=0.001)
        
        # Define symbols to test
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # Load historical data
        # end_date = datetime.now()
        end_date = datetime.now() - timedelta(days=4)
        start_date = end_date - timedelta(days=365)
        
        print(f"Loading data for {len(symbols)} symbols...")
        for symbol in symbols:
            try:
                data = data_provider.get_bars(symbol, '1Day', start_date)
                backtester.add_data(symbol, data)
                print(f"Loaded {len(data)} bars for {symbol}")
            except Exception as e:
                print(f"Error loading data for {symbol}: {e}")
        
        # Create momentum strategy
        strategy_params = {
            'lookback_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'position_size_pct': 0.15,
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.20
        }
        
        strategy = MomentumStrategy(strategy_params)
        
        # Define strategy function for backtester
        def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
            return strategy.generate_signals(
                current_date, current_prices, current_data, historical_data, portfolio
            )
        
        # Run backtest
        print("\\nRunning backtest...")
        results = backtester.run_backtest(
            strategy_func,
            start_date=start_date + timedelta(days=50),  # Allow for indicator warmup
            end_date=end_date,
            symbols=symbols
        )
        
        # Display results
        print("\\n" + "="*70)
        print("GAUSS WORLD TRADER - MOMENTUM STRATEGY BACKTEST RESULTS")
        print("="*70)
        print(backtester.get_results_summary())
        
        # Generate P&L plot
        print("\\n📊 Generating Profit & Loss plot...")
        generate_pnl_plot(results)
        
        # Generate detailed transaction log
        print("\\n📋 Generating transaction log...")
        generate_transaction_log(results, symbols)
        
        # Additional analysis
        if results and 'trades_history' in results:
            trades_df = results['trades_history']
            if not trades_df.empty:
                print(f"\\nTrade Analysis:")
                print(f"Total number of trades: {len(trades_df)}")
                print(f"Average trades per month: {len(trades_df) / 12:.1f}")
                
                # Most traded symbols
                if 'symbol' in trades_df.columns:
                    symbol_counts = trades_df['symbol'].value_counts()
                    print(f"\\nMost active symbols:")
                    print(symbol_counts.head())
        
        # Portfolio evolution
        if results and 'portfolio_history' in results:
            portfolio_df = results['portfolio_history']
            if not portfolio_df.empty:
                print(f"\\nPortfolio Evolution:")
                print(f"Start Value: ${portfolio_df['portfolio_value'].iloc[0]:,.2f}")
                print(f"End Value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}")
                print(f"Peak Value: ${portfolio_df['portfolio_value'].max():,.2f}")
                print(f"Lowest Value: ${portfolio_df['portfolio_value'].min():,.2f}")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        print("Make sure you have:")
        print("1. Alpaca API credentials configured in .env file")
        print("2. All required packages installed (pip install -r requirements.txt)")

if __name__ == '__main__':
    momentum_backtest_example()
