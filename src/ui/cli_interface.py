import sys
import argparse
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json

from config import Config
from src.data import AlpacaDataProvider, CryptoDataProvider, NewsDataProvider, MacroDataProvider
from src.trade import TradingEngine, Backtester
from src.strategy import MomentumStrategy
from src.analysis import TechnicalAnalysis, FinancialMetrics
from src.utils import setup_logger

class CLIInterface:
    def __init__(self):
        self.logger = setup_logger('CLI', Config.LOG_LEVEL)
        self.trading_engine = None
        self.data_provider = None
        
        try:
            self.data_provider = AlpacaDataProvider()
            self.trading_engine = TradingEngine(paper_trading=True)
        except Exception as e:
            self.logger.warning(f"Could not initialize trading components: {e}")
    
    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Quantitative Trading System')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Account commands
        account_parser = subparsers.add_parser('account', help='Account operations')
        account_parser.add_argument('--info', action='store_true', help='Show account information')
        account_parser.add_argument('--positions', action='store_true', help='Show current positions')
        
        # Data commands
        data_parser = subparsers.add_parser('data', help='Data operations')
        data_parser.add_argument('--symbol', required=True, help='Stock symbol')
        data_parser.add_argument('--timeframe', default='1Day', help='Timeframe (1Day, 1Hour, etc.)')
        data_parser.add_argument('--days', type=int, default=30, help='Number of days to fetch')
        data_parser.add_argument('--save', help='Save data to file')
        
        # Trading commands
        trade_parser = subparsers.add_parser('trade', help='Trading operations')
        trade_parser.add_argument('--symbol', required=True, help='Stock symbol')
        trade_parser.add_argument('--action', choices=['buy', 'sell'], required=True, help='Buy or sell')
        trade_parser.add_argument('--quantity', type=int, required=True, help='Number of shares')
        trade_parser.add_argument('--order_type', choices=['market', 'limit'], default='market', help='Order type')
        trade_parser.add_argument('--price', type=float, help='Limit price (for limit orders)')
        
        # Strategy commands
        strategy_parser = subparsers.add_parser('strategy', help='Strategy operations')
        strategy_parser.add_argument('--run', help='Run strategy (momentum)')
        strategy_parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to trade')
        strategy_parser.add_argument('--dry-run', action='store_true', help='Simulate without actual trades')
        
        # Backtest commands
        backtest_parser = subparsers.add_parser('backtest', help='Backtesting operations')
        backtest_parser.add_argument('--strategy', default='momentum', help='Strategy to backtest')
        backtest_parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to backtest')
        backtest_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
        backtest_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
        backtest_parser.add_argument('--initial-cash', type=float, default=100000, help='Initial cash')
        
        # Analysis commands
        analysis_parser = subparsers.add_parser('analysis', help='Technical analysis')
        analysis_parser.add_argument('--symbol', required=True, help='Stock symbol')
        analysis_parser.add_argument('--indicators', nargs='+', 
                                   choices=['rsi', 'macd', 'bollinger', 'sma', 'ema'],
                                   help='Technical indicators to calculate')
        
        # News commands
        news_parser = subparsers.add_parser('news', help='News and sentiment analysis')
        news_parser.add_argument('--symbol', help='Stock symbol for company news')
        news_parser.add_argument('--market', action='store_true', help='Get market news')
        news_parser.add_argument('--sentiment', help='Get news sentiment for symbol')
        
        return parser
    
    def handle_account_command(self, args):
        if not self.trading_engine:
            print("Trading engine not available")
            return
        
        if args.info:
            account_info = self.trading_engine.get_account_info()
            print("\\nAccount Information:")
            print("-" * 30)
            for key, value in account_info.items():
                print(f"{key}: {value}")
        
        if args.positions:
            positions = self.trading_engine.get_current_positions()
            if positions:
                print("\\nCurrent Positions:")
                print("-" * 50)
                for position in positions:
                    print(f"Symbol: {position['symbol']}, "
                          f"Qty: {position['qty']}, "
                          f"Market Value: ${position['market_value']:.2f}, "
                          f"P&L: ${position['unrealized_pl']:.2f}")
            else:
                print("No current positions")
    
    def handle_data_command(self, args):
        if not self.data_provider:
            print("Data provider not available")
            return
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            
            data = self.data_provider.get_bars(
                args.symbol, args.timeframe, start_date, end_date
            )
            
            print(f"\\nData for {args.symbol} ({args.timeframe}):")
            print("-" * 50)
            print(data.tail())
            
            if args.save:
                data.to_csv(args.save)
                print(f"Data saved to {args.save}")
        
        except Exception as e:
            print(f"Error fetching data: {e}")
    
    def handle_trade_command(self, args):
        if not self.trading_engine:
            print("Trading engine not available")
            return
        
        try:
            if args.order_type == 'market':
                result = self.trading_engine.place_market_order(
                    args.symbol, args.quantity, args.action
                )
            else:
                if not args.price:
                    print("Price is required for limit orders")
                    return
                result = self.trading_engine.place_limit_order(
                    args.symbol, args.quantity, args.price, args.action
                )
            
            print(f"Order placed successfully:")
            print(f"Order ID: {result['id']}")
            print(f"Status: {result['status']}")
        
        except Exception as e:
            print(f"Error placing order: {e}")
    
    def handle_strategy_command(self, args):
        if args.run == 'momentum':
            try:
                strategy = MomentumStrategy()
                
                # Get current market data
                current_prices = {}
                historical_data = {}
                
                for symbol in args.symbols:
                    if self.data_provider:
                        data = self.data_provider.get_bars(symbol, '1Day', limit=100)
                        historical_data[symbol] = data
                        current_prices[symbol] = data['close'].iloc[-1]
                
                # Generate signals
                from src.trade import Portfolio
                mock_portfolio = Portfolio()
                
                signals = strategy.generate_signals(
                    current_date=datetime.now(),
                    current_prices=current_prices,
                    current_data={},
                    historical_data=historical_data,
                    portfolio=mock_portfolio
                )
                
                print(f"\\nGenerated {len(signals)} signals:")
                print("-" * 50)
                for signal in signals:
                    print(f"Symbol: {signal['symbol']}, "
                          f"Action: {signal['action']}, "
                          f"Quantity: {signal['quantity']}, "
                          f"Reason: {signal['reason']}")
                
                if not args.dry_run and self.trading_engine:
                    print("\\nExecuting signals...")
                    # Execute signals (implement this based on your requirements)
            
            except Exception as e:
                print(f"Error running strategy: {e}")
    
    def handle_backtest_command(self, args):
        try:
            backtester = Backtester(initial_cash=args.initial_cash)
            
            # Load data
            for symbol in args.symbols:
                if self.data_provider:
                    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
                    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
                    
                    data = self.data_provider.get_bars(symbol, '1Day')
                    backtester.add_data(symbol, data)
            
            # Create strategy
            if args.strategy == 'momentum':
                strategy = MomentumStrategy()
                
                # Define strategy function for backtester
                def strategy_func(current_date, current_prices, current_data, historical_data, portfolio):
                    return strategy.generate_signals(
                        current_date, current_prices, current_data, historical_data, portfolio
                    )
                
                # Run backtest
                results = backtester.run_backtest(
                    strategy_func,
                    start_date=datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None,
                    end_date=datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None,
                    symbols=args.symbols
                )
                
                print(backtester.get_results_summary())
        
        except Exception as e:
            print(f"Error running backtest: {e}")
    
    def handle_analysis_command(self, args):
        if not self.data_provider:
            print("Data provider not available")
            return
        
        try:
            data = self.data_provider.get_bars(args.symbol, '1Day', limit=200)
            ta = TechnicalAnalysis()
            
            print(f"\\nTechnical Analysis for {args.symbol}:")
            print("-" * 50)
            
            if not args.indicators or 'rsi' in args.indicators:
                rsi = ta.rsi(data['close'])
                print(f"RSI (14): {rsi.iloc[-1]:.2f}")
            
            if not args.indicators or 'macd' in args.indicators:
                macd, signal, histogram = ta.macd(data['close'])
                print(f"MACD: {macd.iloc[-1]:.4f}, Signal: {signal.iloc[-1]:.4f}")
            
            if not args.indicators or 'sma' in args.indicators:
                sma_20 = ta.sma(data['close'], 20)
                sma_50 = ta.sma(data['close'], 50)
                print(f"SMA 20: {sma_20.iloc[-1]:.2f}, SMA 50: {sma_50.iloc[-1]:.2f}")
            
            # Trend analysis
            trend_info = ta.trend_analysis(data)
            print(f"\\nTrend Analysis:")
            print(f"Short-term: {trend_info['short_term_trend']}")
            print(f"Medium-term: {trend_info['medium_term_trend']}")
            print(f"Long-term: {trend_info['long_term_trend']}")
        
        except Exception as e:
            print(f"Error performing analysis: {e}")
    
    def handle_news_command(self, args):
        try:
            news_provider = NewsDataProvider()
            
            if args.symbol:
                news = news_provider.get_company_news(args.symbol)
                print(f"\\nCompany News for {args.symbol}:")
                print("-" * 50)
                for article in news[:5]:  # Show top 5
                    print(f"Headline: {article['headline']}")
                    print(f"Source: {article['source']}")
                    print(f"URL: {article['url']}")
                    print("-" * 30)
            
            elif args.market:
                news = news_provider.get_market_news()
                print("\\nMarket News:")
                print("-" * 50)
                for article in news[:5]:  # Show top 5
                    print(f"Headline: {article['headline']}")
                    print(f"Source: {article['source']}")
                    print("-" * 30)
            
            elif args.sentiment:
                sentiment = news_provider.get_news_sentiment(args.sentiment)
                print(f"\\nNews Sentiment for {args.sentiment}:")
                print("-" * 50)
                for key, value in sentiment.items():
                    print(f"{key}: {value}")
        
        except Exception as e:
            print(f"Error fetching news: {e}")
    
    def run(self):
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        # Route commands
        if args.command == 'account':
            self.handle_account_command(args)
        elif args.command == 'data':
            self.handle_data_command(args)
        elif args.command == 'trade':
            self.handle_trade_command(args)
        elif args.command == 'strategy':
            self.handle_strategy_command(args)
        elif args.command == 'backtest':
            self.handle_backtest_command(args)
        elif args.command == 'analysis':
            self.handle_analysis_command(args)
        elif args.command == 'news':
            self.handle_news_command(args)

if __name__ == '__main__':
    cli = CLIInterface()
    cli.run()