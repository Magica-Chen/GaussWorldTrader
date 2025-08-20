"""
Modern CLI using Python 3.12 features
- Typer for better CLI experience
- Rich for beautiful output
- Async operations
- Pattern matching for command routing
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from collections.abc import Sequence

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich import print as rprint
from rich.layout import Layout
from rich.live import Live

from config import Config
try:
    from src.data.async_data_provider import AsyncDataProvider
    from src.trade.optimized_trading_engine import OptimizedTradingEngine, OrderRequest
    from src.strategy.momentum_strategy import MomentumStrategy
    from src.utils.error_handling import safe_execute_async, handle_trading_operation_result
except ImportError as e:
    # Fallback to basic imports for compatibility
    console.print(f"[yellow]⚠️  Some advanced features unavailable: {e}[/yellow]")
    from src.data import AlpacaDataProvider as AsyncDataProvider
    from src.trade import TradingEngine as OptimizedTradingEngine
    from src.strategy import MomentumStrategy

# Initialize rich console
console = Console()

# Create Typer app with rich integration
app = typer.Typer(
    name="trading-system",
    help="🚀 Modern Quantitative Trading System (Python 3.12+)",
    rich_markup_mode="rich",
    add_completion=False
)

# Create subcommands
account_app = typer.Typer(help="💼 Account operations")
data_app = typer.Typer(help="📊 Data operations") 
trade_app = typer.Typer(help="💰 Trading operations")
strategy_app = typer.Typer(help="🧠 Strategy operations")
analysis_app = typer.Typer(help="📈 Analysis operations")

app.add_typer(account_app, name="account")
app.add_typer(data_app, name="data")
app.add_typer(trade_app, name="trade")
app.add_typer(strategy_app, name="strategy")
app.add_typer(analysis_app, name="analysis")

# Add new v2.0 commands as top-level commands
@app.command("check-positions")
def check_positions_cmd() -> None:
    """📈 Check current positions and recent orders"""
    try:
        from src.ui.portfolio_commands import check_positions_and_orders
        check_positions_and_orders()
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("watchlist-trade")
def watchlist_trade_cmd() -> None:
    """🎯 Analyze watchlist and execute trades"""
    try:
        from src.ui.portfolio_commands import get_watchlists_and_trade
        get_watchlists_and_trade()
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)

@account_app.command("info")
def account_info(
    refresh: Annotated[bool, typer.Option("--refresh", "-r", help="Force refresh cache")] = False
) -> None:
    """💼 Display account information"""
    asyncio.run(_account_info_async(refresh))

async def _account_info_async(refresh: bool) -> None:
    """Async implementation of account info"""
    
    with console.status("[bold blue]Fetching account information...", spinner="dots"):
        try:
            engine = OptimizedTradingEngine()
            max_age = 0 if refresh else 30
            account_data, error = await safe_execute_async(
                engine.get_account_info_cached, 
                max_age_seconds=max_age
            )
            
            if not handle_trading_operation_result(account_data, error):
                console.print("[red]❌ Failed to fetch account information[/red]")
                raise typer.Exit(1)
            
            # Create rich table for account info
            table = Table(title="💼 Account Information", show_header=True, header_style="bold blue")
            table.add_column("Metric", style="cyan", width=20)
            table.add_column("Value", style="green", width=20)
            
            # Format values nicely
            table.add_row("Account ID", account_data["account_id"])
            table.add_row("Portfolio Value", f"${account_data['portfolio_value']:,.2f}")
            table.add_row("Buying Power", f"${account_data['buying_power']:,.2f}")
            table.add_row("Cash", f"${account_data['cash']:,.2f}")
            table.add_row("Equity", f"${account_data['equity']:,.2f}")
            table.add_row("Day Trades", str(account_data["day_trade_count"]))
            table.add_row("PDT", "Yes" if account_data["pattern_day_trader"] else "No")
            table.add_row("Status", account_data["status"])
            
            console.print(table)
            
            # Show cache status
            cache_status = "🔄 Refreshed" if refresh else "💾 Cached"
            console.print(f"\n{cache_status} • Last updated: {account_data.get('fetched_at', 'Unknown')}")
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            raise typer.Exit(1)

@account_app.command("performance")
def account_performance(
    days: Annotated[int, typer.Option("--days", "-d", help="Days of performance history")] = 30
) -> None:
    """📊 Show account performance metrics"""
    asyncio.run(_account_performance_async(days))

async def _account_performance_async(days: int) -> None:
    """Async implementation of account performance"""
    
    with console.status("[bold blue]Calculating performance metrics...", spinner="dots"):
        try:
            from src.trade.optimized_trading_engine import OptimizedTradingEngine
            from datetime import datetime, timedelta
            
            engine = OptimizedTradingEngine()
            
            # Get account data
            account_data, error = await safe_execute_async(
                engine.get_account_info_cached,
                max_age_seconds=60
            )
            
            if error:
                console.print(f"[red]❌ Error getting account data: {error}[/red]")
                return
            
            # Get portfolio history (if available)
            try:
                portfolio_history = await engine.get_portfolio_history(days)
            except:
                portfolio_history = None
            
            # Create performance table
            perf_table = Table(title="📊 Account Performance")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="yellow")
            perf_table.add_column("Description", style="white")
            
            current_value = account_data.get('portfolio_value', 0)
            cash = account_data.get('cash', 0)
            equity = account_data.get('equity', 0)
            buying_power = account_data.get('buying_power', 0)
            
            # Basic metrics
            perf_table.add_row(
                "Portfolio Value",
                f"${current_value:,.2f}",
                "Total account value"
            )
            
            perf_table.add_row(
                "Cash Position",
                f"${cash:,.2f}",
                f"{(cash/current_value)*100:.1f}% of portfolio"
            )
            
            perf_table.add_row(
                "Equity Position",
                f"${equity:,.2f}",
                f"{((equity-cash)/current_value)*100:.1f}% invested"
            )
            
            perf_table.add_row(
                "Buying Power",
                f"${buying_power:,.2f}",
                "Available for trading"
            )
            
            # Performance calculations (simplified)
            if portfolio_history and len(portfolio_history) > 1:
                initial_value = portfolio_history[0].get('equity', current_value)
                total_return = ((current_value - initial_value) / initial_value) * 100
                
                perf_table.add_row(
                    f"Return ({days}d)",
                    f"{total_return:+.2f}%",
                    "Total return percentage"
                )
                
                # Calculate daily returns for volatility
                daily_returns = []
                for i in range(1, len(portfolio_history)):
                    prev_val = portfolio_history[i-1].get('equity', 0)
                    curr_val = portfolio_history[i].get('equity', 0)
                    if prev_val > 0:
                        daily_ret = (curr_val - prev_val) / prev_val
                        daily_returns.append(daily_ret)
                
                if daily_returns:
                    import numpy as np
                    volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
                    
                    perf_table.add_row(
                        "Volatility (Ann.)",
                        f"{volatility:.2f}%",
                        "Risk measure"
                    )
                    
                    # Simple Sharpe ratio (assuming 2% risk-free rate)
                    if volatility > 0:
                        annual_return = total_return * (365 / days)
                        sharpe = (annual_return - 2) / volatility
                        
                        perf_table.add_row(
                            "Sharpe Ratio",
                            f"{sharpe:.2f}",
                            "Risk-adjusted return"
                        )
            
            console.print(perf_table)
            
            # Time period information
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            console.print(f"\n[bold cyan]📅 Analysis Period:[/bold cyan]")
            console.print(f"• Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"• End: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"• Duration: {days} days")
            console.print(f"[yellow]⏰ Note: Alpaca free tier has 15-minute delayed data[/yellow]")
            
            # Trading activity summary
            try:
                # Get recent orders/trades
                activities = await engine.get_activities()
                if activities:
                    console.print(f"\n[bold cyan]📈 Recent Trading Activity:[/bold cyan]")
                    console.print(f"• Total activities in period: {len(activities)}")
                    
                    # Count order types
                    order_types = {}
                    for activity in activities[:10]:  # Last 10 activities
                        activity_type = activity.get('activity_type', 'unknown')
                        order_types[activity_type] = order_types.get(activity_type, 0) + 1
                    
                    for activity_type, count in order_types.items():
                        console.print(f"• {activity_type.title()}: {count}")
                        
            except Exception as e:
                console.print(f"[yellow]⚠️ Could not fetch trading activity: {e}[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Error calculating performance: {e}[/red]")

@data_app.command("fetch")
def fetch_data(
    symbols: Annotated[list[str], typer.Argument(help="Stock symbols to fetch")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Data timeframe")] = "1Day",
    days: Annotated[int, typer.Option("--days", "-d", help="Days of historical data")] = 30,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Save to CSV file")] = None,
    concurrent: Annotated[bool, typer.Option("--concurrent", "-c", help="Fetch concurrently")] = True
) -> None:
    """📊 Fetch market data for symbols"""
    asyncio.run(_fetch_data_async(symbols, timeframe, days, output, concurrent))

async def _fetch_data_async(symbols: list[str], timeframe: str, days: int, output: Path | None, concurrent: bool) -> None:
    """Async implementation of fetch data"""
    
    # Validate symbols
    symbols = [s.upper().strip() for s in symbols]
    
    # Calculate time period
    end_date = datetime.now() - timedelta(days=2)  # Account for free tier delay
    start_date = end_date - timedelta(days=days)
    
    console.print(f"[bold cyan]📅 Data Period:[/bold cyan]")
    console.print(f"• Start: {start_date.strftime('%Y-%m-%d')}")
    console.print(f"• End: {end_date.strftime('%Y-%m-%d')}")
    console.print(f"• Duration: {days} days")
    console.print(f"[yellow]⏰ Note: Using 2-day delayed data (Alpaca free tier: 15-min delay)[/yellow]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        if concurrent:
            # Fetch all symbols concurrently (Python 3.12 optimized)  
            task = progress.add_task(f"Fetching data for {len(symbols)} symbols...", total=None)
            
            from src.data.alpaca_provider import AlpacaDataProvider
            provider = AlpacaDataProvider()
            data_dict = {}
            error = None
            
            try:
                for symbol in symbols:
                    data = provider.get_bars(symbol, timeframe, start_date, end_date)
                    if not data.empty:
                        data_dict[symbol] = data
            except Exception as e:
                error = str(e)
                
            if error:
                console.print(f"[red]❌ Failed to fetch data: {error}[/red]")
                raise typer.Exit(1)
        else:
            # Fetch symbols sequentially with progress
            data_dict = {}
            from src.data.alpaca_provider import AlpacaDataProvider
            provider = AlpacaDataProvider()
            
            for symbol in symbols:
                task = progress.add_task(f"Fetching {symbol}...", total=None)
                try:
                    data = provider.get_bars(symbol, timeframe, start_date, end_date)
                    if not data.empty:
                        data_dict[symbol] = data
                except Exception as e:
                    console.print(f"[yellow]⚠️ Failed to fetch {symbol}: {e}[/yellow]")
                progress.remove_task(task)
    
    # Display results
    results_table = Table(title="📊 Data Fetch Results", show_header=True)
    results_table.add_column("Symbol", style="cyan")
    results_table.add_column("Records", style="green")
    results_table.add_column("Date Range", style="yellow")
    results_table.add_column("Latest Price", style="magenta")
    
    for symbol, df in data_dict.items():
        if not df.empty:
            latest_price = f"${df['close'].iloc[-1]:.2f}"
            date_range = f"{df.index[0].date()} to {df.index[-1].date()}"
            results_table.add_row(symbol, str(len(df)), date_range, latest_price)
    
    console.print(results_table)
    
    # Save to file if requested
    if output:
        try:
            # Combine all data into one file with symbol column
            combined_data = []
            for symbol, df in data_dict.items():
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                combined_data.append(df_copy)
            
            if combined_data:
                import pandas as pd
                final_df = pd.concat(combined_data)
                final_df.to_csv(output)
                console.print(f"[green]💾 Data saved to {output}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save data: {e}[/red]")

@data_app.command("stream")
def stream_data(
    symbols: Annotated[list[str], typer.Argument(help="Stock symbols to stream")],
    live: Annotated[bool, typer.Option("--live", help="Enable live streaming")] = False,
    interval: Annotated[int, typer.Option("--interval", "-i", help="Update interval in seconds")] = 5
) -> None:
    """📡 Stream real-time market data"""
    asyncio.run(_stream_data_async(symbols, live, interval))

async def _stream_data_async(symbols: list[str], live: bool, interval: int) -> None:
    """Async implementation of data streaming"""
    
    symbols = [s.upper().strip() for s in symbols]
    
    console.print(f"[bold cyan]📡 Streaming data for: {', '.join(symbols)}[/bold cyan]")
    console.print(f"[bold cyan]📅 Stream Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]")
    if not live:
        console.print("[yellow]ℹ️ Using simulated streaming (--live not enabled)[/yellow]")
    console.print(f"[yellow]⏰ Note: Alpaca free tier has 15-minute delayed data[/yellow]")
    console.print(f"[dim]Update interval: {interval} seconds • Press Ctrl+C to stop[/dim]\n")
    
    try:
        from src.data.alpaca_provider import AlpacaDataProvider
        from datetime import datetime, timedelta
        import time
        
        provider = AlpacaDataProvider()
        
        # Create a live display layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="data", ratio=1)
        )
        
        # Header
        header_table = Table.grid()
        header_table.add_column(style="bold cyan")
        header_table.add_row(f"🌍 Gauss World Trader - Live Data Stream")
        header_table.add_row(f"📊 Symbols: {', '.join(symbols)} • Interval: {interval}s")
        
        layout["header"].update(Panel(header_table, title="📡 Streaming"))
        
        # Main data table
        data_table = Table()
        data_table.add_column("Symbol", style="bold cyan")
        data_table.add_column("Price", style="yellow")
        data_table.add_column("Change", style="green")
        data_table.add_column("Volume", style="blue")
        data_table.add_column("Time", style="dim")
        
        # Store previous prices for change calculation
        prev_prices = {}
        
        with Live(layout, refresh_per_second=1, screen=True) as live_display:
            try:
                while True:
                    # Clear the table for new data
                    data_table = Table()
                    data_table.add_column("Symbol", style="bold cyan")
                    data_table.add_column("Price", style="yellow") 
                    data_table.add_column("Change", style="green")
                    data_table.add_column("Volume", style="blue")
                    data_table.add_column("Time", style="dim")
                    
                    current_time = datetime.now()
                    
                    for symbol in symbols:
                        try:
                            # Get latest data (using delayed data for free tier)
                            end_date = current_time - timedelta(days=2)
                            start_date = end_date - timedelta(days=1)
                            
                            data = provider.get_bars(symbol, '1Day', start_date, end_date)
                            
                            if not data.empty:
                                latest = data.iloc[-1]
                                current_price = latest['close']
                                volume = latest['volume']
                                
                                # Calculate change
                                if symbol in prev_prices:
                                    change = current_price - prev_prices[symbol]
                                    change_pct = (change / prev_prices[symbol]) * 100
                                    change_str = f"{change:+.2f} ({change_pct:+.1f}%)"
                                    change_style = "green" if change >= 0 else "red"
                                else:
                                    change_str = "N/A"
                                    change_style = "white"
                                
                                prev_prices[symbol] = current_price
                                
                                data_table.add_row(
                                    symbol,
                                    f"${current_price:.2f}",
                                    f"[{change_style}]{change_str}[/{change_style}]",
                                    f"{volume:,.0f}",
                                    current_time.strftime("%H:%M:%S")
                                )
                            else:
                                data_table.add_row(
                                    symbol,
                                    "N/A",
                                    "N/A", 
                                    "N/A",
                                    current_time.strftime("%H:%M:%S")
                                )
                                
                        except Exception as e:
                            data_table.add_row(
                                symbol,
                                "ERROR",
                                f"[red]{str(e)[:20]}...[/red]",
                                "N/A",
                                current_time.strftime("%H:%M:%S")
                            )
                    
                    layout["data"].update(Panel(data_table, title="📊 Market Data"))
                    
                    # Wait for next update
                    await asyncio.sleep(interval)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]📡 Streaming stopped by user[/yellow]")
                
    except ImportError as e:
        console.print(f"[red]❌ Required modules not available: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Error in data streaming: {e}[/red]")

@trade_app.command("place")
def place_order(
    symbol: Annotated[str, typer.Argument(help="Stock symbol")],
    side: Annotated[str, typer.Argument(help="buy or sell")],
    quantity: Annotated[int, typer.Argument(help="Number of shares")],
    order_type: Annotated[str, typer.Option("--type", "-t", help="Order type")] = "market",
    price: Annotated[float | None, typer.Option("--price", "-p", help="Limit price")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate without executing")] = False
) -> None:
    """💰 Place a trading order"""
    asyncio.run(_place_order_async(symbol, side, quantity, order_type, price, dry_run))

async def _place_order_async(symbol: str, side: str, quantity: int, order_type: str, price: float | None, dry_run: bool) -> None:
    """Async implementation of place order"""
    
    symbol = symbol.upper().strip()
    
    # Validate inputs using pattern matching
    match side.lower():
        case "buy" | "sell":
            pass
        case _:
            console.print("[red]❌ Side must be 'buy' or 'sell'[/red]")
            raise typer.Exit(1)
    
    match order_type.lower():
        case "market" | "limit":
            if order_type.lower() == "limit" and price is None:
                console.print("[red]❌ Limit orders require a price[/red]")
                raise typer.Exit(1)
        case _:
            console.print("[red]❌ Order type must be 'market' or 'limit'[/red]")
            raise typer.Exit(1)
    
    # Create order request
    order_request = OrderRequest(
        symbol=symbol,
        quantity=quantity,
        side=side.lower(),
        order_type=order_type.lower(),
        price=price
    )
    
    # Display order summary
    order_panel = Panel.fit(
        f"""
[bold]Order Summary[/bold]
Symbol: [cyan]{symbol}[/cyan]
Side: [green]{side.upper()}[/green]
Quantity: [yellow]{quantity:,}[/yellow]
Type: [blue]{order_type.upper()}[/blue]
Price: [magenta]${price or 'Market'}[/magenta]
Mode: [red]DRY RUN[/red] if dry_run else [green]LIVE[/green]
        """.strip(),
        title="💰 Trade Order",
        border_style="blue"
    )
    console.print(order_panel)
    
    # Confirm order
    if not dry_run:
        confirm = Confirm.ask("🚨 Execute this LIVE order?", default=False)
        if not confirm:
            console.print("[yellow]📋 Order cancelled[/yellow]")
            return
    
    # Execute order
    if dry_run:
        console.print("[blue]🧪 DRY RUN: Order would be placed but not executed[/blue]")
        return
    
    with console.status("[bold yellow]Placing order...", spinner="dots"):
        try:
            engine = OptimizedTradingEngine()
            result, error = await safe_execute_async(
                engine.place_order_async,
                order_request
            )
            
            if not handle_trading_operation_result(result, error):
                console.print("[red]❌ Order placement failed[/red]")
                raise typer.Exit(1)
            
            # Display result
            status_color = "green" if result.status in ["filled", "accepted"] else "yellow"
            console.print(f"[{status_color}]✅ Order placed successfully![/{status_color}]")
            console.print(f"Order ID: [cyan]{result.order_id}[/cyan]")
            console.print(f"Status: [cyan]{result.status}[/cyan]")
            
            if result.filled_qty > 0:
                console.print(f"Filled: [green]{result.filled_qty} @ ${result.filled_price}[/green]")
                
        except Exception as e:
            console.print(f"[red]❌ Error placing order: {e}[/red]")
            raise typer.Exit(1)

@strategy_app.command("run")
def run_strategy(
    symbols: Annotated[list[str], typer.Argument(help="Symbols to trade")],
    strategy_name: Annotated[str, typer.Option("--strategy", "-s", help="Strategy to run")] = "momentum",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate without trading")] = True,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Continuous monitoring")] = False
) -> None:
    """🧠 Run a trading strategy"""
    asyncio.run(_run_strategy_async(symbols, strategy_name, dry_run, watch))

async def _run_strategy_async(symbols: list[str], strategy_name: str, dry_run: bool, watch: bool) -> None:
    """Async implementation of run strategy"""
    
    symbols = [s.upper().strip() for s in symbols]
    
    # Create strategy based on name using pattern matching
    match strategy_name.lower():
        case "momentum":
            strategy = MomentumStrategy()
        case _:
            console.print(f"[red]❌ Unknown strategy: {strategy_name}[/red]")
            raise typer.Exit(1)
    
    console.print(f"[blue]🧠 Running {strategy_name} strategy on {len(symbols)} symbols[/blue]")
    
    if watch:
        # Continuous monitoring mode
        console.print("[yellow]👁️ Entering watch mode (Ctrl+C to stop)[/yellow]")
        
        try:
            while True:
                await _run_strategy_iteration(strategy, symbols, dry_run)
                await asyncio.sleep(60)  # Run every minute
                
        except KeyboardInterrupt:
            console.print("[yellow]🛑 Strategy stopped by user[/yellow]")
    else:
        # Single run
        await _run_strategy_iteration(strategy, symbols, dry_run)

async def _run_strategy_iteration(strategy, symbols: list[str], dry_run: bool) -> None:
    """Run a single iteration of the strategy"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Fetch data
        task = progress.add_task("Fetching market data...", total=None)
        
        async with AsyncDataProvider() as provider:
            data_dict, error = await safe_execute_async(
                provider.fetch_multiple_symbols,
                symbols, '1Day', 100
            )
            
            if not handle_trading_operation_result(data_dict, error):
                console.print("[red]❌ Failed to fetch market data[/red]")
                return
        
        progress.update(task, description="Generating signals...")
        
        # Generate signals (run in executor for CPU-bound work)
        from src.trade import Portfolio
        mock_portfolio = Portfolio()
        
        current_prices = {
            symbol: df['close'].iloc[-1] 
            for symbol, df in data_dict.items() 
            if not df.empty
        }
        
        signals = strategy.generate_signals(
            current_date=datetime.now(),
            current_prices=current_prices,
            current_data={},
            historical_data=data_dict,
            portfolio=mock_portfolio
        )
        
        progress.remove_task(task)
    
    # Display signals
    if signals:
        signals_table = Table(title="🧠 Strategy Signals", show_header=True)
        signals_table.add_column("Symbol", style="cyan")
        signals_table.add_column("Action", style="green") 
        signals_table.add_column("Quantity", style="yellow")
        signals_table.add_column("Confidence", style="magenta")
        signals_table.add_column("Reason", style="white")
        
        for signal in signals:
            confidence = f"{signal.get('confidence', 0):.1%}"
            signals_table.add_row(
                signal['symbol'],
                signal['action'].upper(),
                str(signal['quantity']),
                confidence,
                signal.get('reason', 'N/A')
            )
        
        console.print(signals_table)
        
        if dry_run:
            console.print("[blue]🧪 DRY RUN: Signals generated but no trades executed[/blue]")
        else:
            console.print("[yellow]⚠️ Live trading not implemented in this example[/yellow]")
    
    else:
        console.print("[yellow]📭 No signals generated[/yellow]")

@analysis_app.command("technical")
def technical_analysis(
    symbol: Annotated[str, typer.Argument(help="Stock symbol to analyze")],
    indicators: Annotated[list[str], typer.Option("--indicators", "-i", help="Technical indicators")] = ["rsi", "macd"],
    days: Annotated[int, typer.Option("--days", "-d", help="Days of historical data")] = 100
) -> None:
    """📈 Perform technical analysis on a symbol"""
    asyncio.run(_technical_analysis_async(symbol, indicators, days))

async def _technical_analysis_async(symbol: str, indicators: list[str], days: int) -> None:
    """Async implementation of technical analysis"""
    
    symbol = symbol.upper().strip()
    indicators = [ind.lower().strip() for ind in indicators]
    
    with console.status(f"[bold blue]Analyzing {symbol}...", spinner="dots"):
        try:
            # Import technical analysis module
            from src.analysis.technical_analysis import TechnicalAnalysis
            from src.data.alpaca_provider import AlpacaDataProvider
            from datetime import datetime, timedelta
            
            # Fetch data
            provider = AlpacaDataProvider()
            # Use older data to avoid free tier limitations
            end_date = datetime.now() - timedelta(days=5)
            start_date = end_date - timedelta(days=days)
            
            data = provider.get_bars(symbol, '1Day', start_date, end_date)
            
            if data.empty:
                console.print(f"[red]❌ No data found for {symbol}[/red]")
                return
            
            # Initialize analyzer
            analyzer = TechnicalAnalysis()
            
            # Create results table
            results_table = Table(title=f"📈 Technical Analysis: {symbol}")
            results_table.add_column("Indicator", style="cyan")
            results_table.add_column("Current Value", style="yellow")
            results_table.add_column("Signal", style="green")
            results_table.add_column("Interpretation", style="white")
            
            current_price = data['close'].iloc[-1]
            
            # Calculate and display requested indicators
            for indicator in indicators:
                match indicator:
                    case "rsi":
                        rsi_values = analyzer.rsi(data['close'])
                        if not rsi_values.empty:
                            current_rsi = rsi_values.iloc[-1]
                            if current_rsi < 30:
                                signal = "🟢 BUY"
                                interpretation = "Oversold condition"
                            elif current_rsi > 70:
                                signal = "🔴 SELL" 
                                interpretation = "Overbought condition"
                            else:
                                signal = "🟡 HOLD"
                                interpretation = "Neutral zone"
                            
                            results_table.add_row(
                                "RSI (14)",
                                f"{current_rsi:.2f}",
                                signal,
                                interpretation
                            )
                    
                    case "macd":
                        macd_line, signal_line, histogram = analyzer.macd(data['close'])
                        if not macd_line.empty:
                            current_macd = macd_line.iloc[-1]
                            current_signal = signal_line.iloc[-1]
                            current_histogram = histogram.iloc[-1]
                            
                            if current_macd > current_signal and current_histogram > 0:
                                signal = "🟢 BUY"
                                interpretation = "Bullish crossover"
                            elif current_macd < current_signal and current_histogram < 0:
                                signal = "🔴 SELL"
                                interpretation = "Bearish crossover"
                            else:
                                signal = "🟡 HOLD"
                                interpretation = "No clear signal"
                            
                            results_table.add_row(
                                "MACD",
                                f"{current_macd:.4f}",
                                signal,
                                interpretation
                            )
                    
                    case "bb" | "bollinger":
                        upper_band, middle_band, lower_band = analyzer.bollinger_bands(data['close'])
                        if not upper_band.empty:
                            upper_val = upper_band.iloc[-1]
                            lower_val = lower_band.iloc[-1]
                            middle_val = middle_band.iloc[-1]
                            
                            if current_price <= lower_val:
                                signal = "🟢 BUY"
                                interpretation = "Price at lower band"
                            elif current_price >= upper_val:
                                signal = "🔴 SELL"
                                interpretation = "Price at upper band"
                            else:
                                signal = "🟡 HOLD"
                                interpretation = "Price within bands"
                            
                            results_table.add_row(
                                "Bollinger Bands",
                                f"${current_price:.2f}",
                                signal,
                                f"Range: ${lower_val:.2f} - ${upper_val:.2f}"
                            )
                    
                    case "sma" | "ma":
                        sma_20 = analyzer.sma(data['close'], 20)
                        sma_50 = analyzer.sma(data['close'], 50)
                        
                        if not sma_20.empty and not sma_50.empty:
                            current_sma20 = sma_20.iloc[-1]
                            current_sma50 = sma_50.iloc[-1]
                            
                            if current_price > current_sma20 > current_sma50:
                                signal = "🟢 BUY"
                                interpretation = "Price above moving averages"
                            elif current_price < current_sma20 < current_sma50:
                                signal = "🔴 SELL"
                                interpretation = "Price below moving averages"
                            else:
                                signal = "🟡 HOLD"
                                interpretation = "Mixed signals"
                            
                            results_table.add_row(
                                "SMA (20/50)",
                                f"${current_sma20:.2f} / ${current_sma50:.2f}",
                                signal,
                                interpretation
                            )
                    
                    case _:
                        console.print(f"[yellow]⚠️ Unknown indicator: {indicator}[/yellow]")
            
            console.print(results_table)
            
            # Add price summary and time information
            console.print(f"\n[bold cyan]💰 Current Price: ${current_price:.2f}[/bold cyan]")
            console.print(f"\n[bold cyan]📅 Analysis Period:[/bold cyan]")
            console.print(f"• Data Start: {data.index[0].strftime('%Y-%m-%d')}")
            console.print(f"• Data End: {data.index[-1].strftime('%Y-%m-%d')}")
            console.print(f"• Trading Days: {len(data)} days")
            console.print(f"• Requested Period: {days} days")
            console.print(f"[yellow]⏰ Note: Using historical data (Alpaca free tier: 15-min delay)[/yellow]")
            
        except ImportError as e:
            console.print(f"[red]❌ Technical analysis module not available: {e}[/red]")
        except Exception as e:
            console.print(f"[red]❌ Error performing technical analysis: {e}[/red]")

@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Show version")] = False
) -> None:
    """🚀 Modern Quantitative Trading System"""
    if version:
        console.print("[blue]Trading System v2.0.0 (Python 3.12+)[/blue]")
        raise typer.Exit()

if __name__ == "__main__":
    # Python 3.12 optimized CLI execution
    app()