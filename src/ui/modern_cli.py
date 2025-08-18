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

@account_app.command("info")
async def account_info(
    refresh: Annotated[bool, typer.Option("--refresh", "-r", help="Force refresh cache")] = False
) -> None:
    """💼 Display account information"""
    
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

@data_app.command("fetch")
async def fetch_data(
    symbols: Annotated[list[str], typer.Argument(help="Stock symbols to fetch")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Data timeframe")] = "1Day",
    days: Annotated[int, typer.Option("--days", "-d", help="Days of historical data")] = 30,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Save to CSV file")] = None,
    concurrent: Annotated[bool, typer.Option("--concurrent", "-c", help="Fetch concurrently")] = True
) -> None:
    """📊 Fetch market data for symbols"""
    
    # Validate symbols
    symbols = [s.upper().strip() for s in symbols]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        if concurrent:
            # Fetch all symbols concurrently (Python 3.12 optimized)
            task = progress.add_task(f"Fetching data for {len(symbols)} symbols...", total=None)
            
            async with AsyncDataProvider() as provider:
                data_dict, error = await safe_execute_async(
                    provider.fetch_multiple_symbols,
                    symbols, timeframe, days
                )
                
                if not handle_trading_operation_result(data_dict, error):
                    console.print("[red]❌ Failed to fetch data[/red]")
                    raise typer.Exit(1)
        else:
            # Fetch symbols sequentially with progress
            data_dict = {}
            async with AsyncDataProvider() as provider:
                for symbol in symbols:
                    task = progress.add_task(f"Fetching {symbol}...", total=None)
                    data, error = await safe_execute_async(
                        provider._fetch_symbol_data,
                        symbol, timeframe, days
                    )
                    
                    if handle_trading_operation_result(data, error):
                        data_dict[symbol] = data
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

@trade_app.command("place")
async def place_order(
    symbol: Annotated[str, typer.Argument(help="Stock symbol")],
    side: Annotated[str, typer.Argument(help="buy or sell")],
    quantity: Annotated[int, typer.Argument(help="Number of shares")],
    order_type: Annotated[str, typer.Option("--type", "-t", help="Order type")] = "market",
    price: Annotated[float | None, typer.Option("--price", "-p", help="Limit price")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate without executing")] = False
) -> None:
    """💰 Place a trading order"""
    
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
async def run_strategy(
    symbols: Annotated[list[str], typer.Argument(help="Symbols to trade")],
    strategy_name: Annotated[str, typer.Option("--strategy", "-s", help="Strategy to run")] = "momentum",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate without trading")] = True,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Continuous monitoring")] = False
) -> None:
    """🧠 Run a trading strategy"""
    
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