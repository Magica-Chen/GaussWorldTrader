"""
Portfolio and Watchlist Command Implementations

Separated from main.py to avoid circular imports
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytz
EASTERN = pytz.timezone('US/Eastern')

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

def get_watchlists_and_trade(strategy_name: str = "momentum") -> None:
    """Get watchlist info and run strategy with direct trading
    
    Args:
        strategy_name: Strategy to use ('momentum', 'value', etc.). Default is 'momentum'
    """
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
        
        # Step 1: Load watchlist symbols from JSON file
        try:
            from src.utils.watchlist_manager import get_default_watchlist
            watchlist_symbols = get_default_watchlist()
            if not watchlist_symbols:
                # Fallback to default symbols if JSON is empty
                watchlist_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        except Exception as e:
            # Fallback to default symbols if watchlist manager not available
            print(f"Warning: Could not load JSON watchlist ({e}), using fallback symbols")
            watchlist_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        
        # Step 2: Get current positions
        try:
            from src.account import PositionManager
            position_manager = PositionManager(account_manager)
            current_positions = position_manager.get_positions()
            position_symbols = [pos.get('symbol', '') for pos in current_positions if pos.get('symbol')]
        except Exception as e:
            if HAS_RICH and console:
                console.print(f"[yellow]Warning: Could not get current positions: {e}[/yellow]")
            else:
                print(f"Warning: Could not get current positions: {e}")
            position_symbols = []
        
        # Step 3: Combine and deduplicate symbols (watchlist + current positions)
        all_symbols = list(set(watchlist_symbols + position_symbols))
        
        # Step 4: Initialize strategy based on parameter
        strategy = None
        try:
            if strategy_name.lower() == "momentum":
                from src.strategy import MomentumStrategy
                strategy = MomentumStrategy()
            elif strategy_name.lower() == "value":
                from src.strategy import ValueStrategy
                strategy = ValueStrategy()
            else:
                # Default to momentum if unknown strategy
                from src.strategy import MomentumStrategy
                strategy = MomentumStrategy()
                if HAS_RICH and console:
                    console.print(f"[yellow]Unknown strategy '{strategy_name}', using Momentum strategy[/yellow]")
                else:
                    print(f"Unknown strategy '{strategy_name}', using Momentum strategy")
        except ImportError as e:
            # Fallback to momentum strategy
            from src.strategy import MomentumStrategy
            strategy = MomentumStrategy()
            if HAS_RICH and console:
                console.print(f"[yellow]Could not load {strategy_name} strategy ({e}), using Momentum strategy[/yellow]")
            else:
                print(f"Could not load {strategy_name} strategy ({e}), using Momentum strategy")
        
        if HAS_RICH and console:
            console.print("\n[bold blue]🌍 Gauss World Trader - Strategic Trading Analysis[/bold blue]")
            console.print(f"[cyan]Strategy: {strategy_name.title()}[/cyan]")
            console.print(f"[cyan]Watchlist Symbols ({len(watchlist_symbols)}): {', '.join(watchlist_symbols)}[/cyan]")
            if position_symbols:
                console.print(f"[cyan]Current Positions ({len(position_symbols)}): {', '.join(position_symbols)}[/cyan]")
            console.print(f"[cyan]Total Symbols to Analyze ({len(all_symbols)}): {', '.join(sorted(all_symbols))}[/cyan]")
            
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
            
            console.print(f"\n[cyan]🔍 Analyzing {len(all_symbols)} symbols with {strategy_name.title()} strategy...[/cyan]")
            
        else:
            print("\n🌍 Gauss World Trader - Strategic Trading Analysis")
            print("=" * 50)
            print(f"Strategy: {strategy_name.title()}")
            print(f"Watchlist Symbols ({len(watchlist_symbols)}): {', '.join(watchlist_symbols)}")
            if position_symbols:
                print(f"Current Positions ({len(position_symbols)}): {', '.join(position_symbols)}")
            print(f"Total Symbols to Analyze ({len(all_symbols)}): {', '.join(sorted(all_symbols))}")
            
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
            
            print(f"\n🔍 Analyzing {len(all_symbols)} symbols with {strategy_name.title()} strategy...")
        
        # Step 5: Initialize data provider 
        from src.data import AlpacaDataProvider
        data_provider = AlpacaDataProvider()
        
        # Step 6: Analyze each symbol (watchlist + current positions)
        signals = []
        analysis_results = {}
        
        for symbol in all_symbols:
            try:
                # Get recent data for analysis - use ET time for trading logic
                end_date = datetime.now(EASTERN)
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
                    current_date=datetime.now(EASTERN),  # Use ET time for trading logic
                    current_prices=current_prices,
                    current_data=current_data,
                    historical_data={symbol: data},
                    portfolio=None  # Would need actual portfolio for position sizing
                )

                # Store analysis results
                analysis_results[symbol] = {
                    'in_watchlist': symbol in watchlist_symbols,
                    'in_positions': symbol in position_symbols,
                    'signals': symbol_signals,
                    'current_price': current_prices[symbol]
                }
                
                signals.extend(symbol_signals)
                
                if HAS_RICH and console:
                    status = "📋" if symbol in watchlist_symbols else "💼" if symbol in position_symbols else "🔍"
                    console.print(f"[dim]{status} Analyzed {symbol}: {len(symbol_signals)} signals[/dim]")
                else:
                    status = "[W]" if symbol in watchlist_symbols else "[P]" if symbol in position_symbols else "[N]"
                    print(f"{status} Analyzed {symbol}: {len(symbol_signals)} signals")
                    
            except Exception as e:
                if HAS_RICH and console:
                    console.print(f"[red]Error analyzing {symbol}: {e}[/red]")
                else:
                    print(f"Error analyzing {symbol}: {e}")
        
        # Step 7: Categorize and display signals
        if signals:
            # Categorize signals by action and symbol type
            buy_signals = [s for s in signals if s.get('action', '').lower() == 'buy']
            sell_signals = [s for s in signals if s.get('action', '').lower() == 'sell']
            
            if HAS_RICH and console:
                console.print(f"\n[green]📈 Strategy Analysis Complete - Found {len(signals)} trading signals:[/green]")
                console.print(f"[green]  • Buy Signals: {len(buy_signals)}[/green]")
                console.print(f"[red]  • Sell Signals: {len(sell_signals)}[/red]")
                
                # Display detailed signals
                for i, signal in enumerate(signals, 1):
                    symbol = signal.get('symbol', 'N/A')
                    action = signal.get('action', 'N/A').upper()
                    quantity = signal.get('quantity', 0)
                    reason = signal.get('reason', 'N/A')
                    
                    # Get symbol context
                    result = analysis_results.get(symbol, {})
                    context = ""
                    if result.get('in_watchlist') and result.get('in_positions'):
                        context = "📋💼"
                    elif result.get('in_watchlist'):
                        context = "📋"
                    elif result.get('in_positions'):
                        context = "💼"
                    
                    action_color = "green" if action == "BUY" else "red"
                    console.print(f"[{action_color}]{i:2d}. {context} {symbol} - {action} {quantity} shares[/{action_color}]")
                    console.print(f"     [dim]Reason: {reason}[/dim]")
                
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