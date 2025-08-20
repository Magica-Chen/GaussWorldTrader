"""
Portfolio and Watchlist Command Implementations

Separated from main.py to avoid circular imports
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

def check_positions_and_orders() -> None:
    """Check current positions and recent orders"""
    try:
        from src.account import AccountManager, PositionManager, OrderManager
        HAS_ACCOUNT_MODULE = True
    except ImportError as e:
        print(f"❌ Account module not available: {e}")
        return
    
    try:
        from rich.console import Console
        HAS_RICH = True
        console = Console()
    except ImportError:
        HAS_RICH = False
        console = None
    
    try:
        # Initialize account manager
        account_manager = AccountManager()
        position_manager = PositionManager(account_manager)
        order_manager = OrderManager(account_manager)
        
        if HAS_RICH and console:
            console.print("\n[bold blue]🌍 Gauss World Trader - Positions & Orders Check[/bold blue]")
            
            # Account summary
            console.print("\n[bold cyan]📊 Account Summary:[/bold cyan]")
            account_summary = account_manager.get_account_summary()
            console.print(account_summary)
            
            # Positions summary
            console.print("\n[bold cyan]📈 Positions Summary:[/bold cyan]")
            positions_summary = position_manager.get_positions_summary()
            console.print(positions_summary)
            
            # Recent orders
            console.print("\n[bold cyan]📋 Recent Orders:[/bold cyan]")
            orders_summary = order_manager.get_recent_orders_summary()
            console.print(orders_summary)
            
        else:
            print("\n🌍 Gauss World Trader - Positions & Orders Check")
            print("=" * 50)
            
            # Account summary
            print("\n📊 Account Summary:")
            print(account_manager.get_account_summary())
            
            # Positions summary
            print("\n📈 Positions Summary:")
            print(position_manager.get_positions_summary())
            
            # Recent orders
            print("\n📋 Recent Orders:")
            print(order_manager.get_recent_orders_summary())
            
    except Exception as e:
        if HAS_RICH and console:
            console.print(f"[red]❌ Error checking positions and orders: {e}[/red]")
        else:
            print(f"❌ Error checking positions and orders: {e}")

def get_watchlists_and_trade() -> None:
    """Get watchlist info and run strategy with direct trading"""
    try:
        from src.account import AccountManager, OrderManager
        HAS_ACCOUNT_MODULE = True
    except ImportError as e:
        print(f"❌ Account module not available: {e}")
        return
    
    try:
        from rich.console import Console
        from rich.prompt import Confirm
        HAS_RICH = True
        console = Console()
    except ImportError:
        HAS_RICH = False
        console = None
    
    try:
        # Initialize managers
        account_manager = AccountManager()
        order_manager = OrderManager(account_manager)
        
        # Default watchlist symbols (can be expanded to read from external source)
        watchlist_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        
        if HAS_RICH and console:
            console.print("\n[bold blue]🌍 Gauss World Trader - Watchlist Trading[/bold blue]")
            console.print(f"[cyan]Watchlist Symbols: {', '.join(watchlist_symbols)}[/cyan]")
            
            # Check market status
            market_clock = account_manager.get_market_clock()
            if 'error' not in market_clock:
                is_open = market_clock.get('is_open', False)
                next_open = market_clock.get('next_open', 'N/A')
                next_close = market_clock.get('next_close', 'N/A')
                
                console.print(f"[yellow]Market Status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}[/yellow]")
                if not is_open:
                    console.print(f"[yellow]Next Open: {next_open}[/yellow]")
            
            # Get account status
            account_status = account_manager.get_trading_account_status()
            if 'error' not in account_status:
                buying_power = account_status.get('buying_power', 0)
                console.print(f"[green]Available Buying Power: ${buying_power:,.2f}[/green]")
            
            # Ask for confirmation before proceeding
            if not Confirm.ask("\nProceed with strategy analysis and potential trading?"):
                console.print("[yellow]Trading cancelled by user[/yellow]")
                return
            
            console.print("\n[cyan]🔍 Analyzing watchlist symbols...[/cyan]")
            
        else:
            print("\n🌍 Gauss World Trader - Watchlist Trading")
            print("=" * 50)
            print(f"Watchlist Symbols: {', '.join(watchlist_symbols)}")
            
            # Check market status
            market_clock = account_manager.get_market_clock()
            if 'error' not in market_clock:
                is_open = market_clock.get('is_open', False)
                print(f"Market Status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
            
            # Simple confirmation
            response = input("\nProceed with strategy analysis and potential trading? (y/N): ")
            if response.lower() != 'y':
                print("Trading cancelled by user")
                return
            
            print("\n🔍 Analyzing watchlist symbols...")
        
        # Import and initialize strategy components
        from src.data import AlpacaDataProvider
        from src.strategy import MomentumStrategy
        
        data_provider = AlpacaDataProvider()
        strategy = MomentumStrategy()
        
        # Analyze each symbol in watchlist
        signals = []
        for symbol in watchlist_symbols:
            try:
                # Get recent data for analysis
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                data = data_provider.get_bars(symbol, '1Day', start_date, end_date)
                if data.empty:
                    continue
                
                # Generate signals
                current_prices = {symbol: data['close'].iloc[-1]}
                current_data = {symbol: {
                    'open': data['open'].iloc[-1],
                    'high': data['high'].iloc[-1],
                    'low': data['low'].iloc[-1],
                    'close': data['close'].iloc[-1],
                    'volume': data['volume'].iloc[-1]
                }}
                
                symbol_signals = strategy.generate_signals(
                    current_date=datetime.now(),
                    current_prices=current_prices,
                    current_data=current_data,
                    historical_data={symbol: data},
                    portfolio=None  # Would need actual portfolio for position sizing
                )
                
                signals.extend(symbol_signals)
                
                if HAS_RICH and console:
                    console.print(f"[dim]Analyzed {symbol}: {len(symbol_signals)} signals[/dim]")
                else:
                    print(f"Analyzed {symbol}: {len(symbol_signals)} signals")
                    
            except Exception as e:
                if HAS_RICH and console:
                    console.print(f"[red]Error analyzing {symbol}: {e}[/red]")
                else:
                    print(f"Error analyzing {symbol}: {e}")
        
        # Display signals and execute trades
        if signals:
            if HAS_RICH and console:
                console.print(f"\n[green]📈 Found {len(signals)} trading signals:[/green]")
                
                for i, signal in enumerate(signals, 1):
                    console.print(f"{i}. {signal.get('symbol', 'N/A')} - {signal.get('action', 'N/A')} {signal.get('quantity', 0)} shares")
                
                if Confirm.ask("\nExecute these trades?"):
                    console.print("[cyan]🚀 Executing trades...[/cyan]")
                    
                    for signal in signals:
                        try:
                            order_result = order_manager.place_order(
                                symbol=signal['symbol'],
                                qty=signal['quantity'],
                                side=signal['action'],
                                order_type='market',
                                time_in_force='day'
                            )
                            
                            if 'error' not in order_result:
                                order_id = order_result.get('id', 'N/A')
                                console.print(f"[green]✅ Order placed: {signal['symbol']} {signal['action']} {signal['quantity']} (ID: {order_id})[/green]")
                            else:
                                console.print(f"[red]❌ Order failed: {signal['symbol']} - {order_result['error']}[/red]")
                                
                        except Exception as e:
                            console.print(f"[red]❌ Error placing order for {signal['symbol']}: {e}[/red]")
                else:
                    console.print("[yellow]Trades cancelled by user[/yellow]")
            else:
                print(f"\n📈 Found {len(signals)} trading signals:")
                for i, signal in enumerate(signals, 1):
                    print(f"{i}. {signal.get('symbol', 'N/A')} - {signal.get('action', 'N/A')} {signal.get('quantity', 0)} shares")
                
                response = input("\nExecute these trades? (y/N): ")
                if response.lower() == 'y':
                    print("🚀 Executing trades...")
                    
                    for signal in signals:
                        try:
                            order_result = order_manager.place_order(
                                symbol=signal['symbol'],
                                qty=signal['quantity'],
                                side=signal['action'],
                                order_type='market',
                                time_in_force='day'
                            )
                            
                            if 'error' not in order_result:
                                order_id = order_result.get('id', 'N/A')
                                print(f"✅ Order placed: {signal['symbol']} {signal['action']} {signal['quantity']} (ID: {order_id})")
                            else:
                                print(f"❌ Order failed: {signal['symbol']} - {order_result['error']}")
                                
                        except Exception as e:
                            print(f"❌ Error placing order for {signal['symbol']}: {e}")
                else:
                    print("Trades cancelled by user")
        else:
            if HAS_RICH and console:
                console.print("[yellow]No trading signals generated[/yellow]")
            else:
                print("No trading signals generated")
                
    except Exception as e:
        if HAS_RICH and console:
            console.print(f"[red]❌ Error in watchlist trading: {e}[/red]")
        else:
            print(f"❌ Error in watchlist trading: {e}")