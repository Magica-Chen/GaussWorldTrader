"""
Simple CLI interface with Python 3.12 compatibility
Uses existing components with modern CLI features where possible
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Annotated

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    # Fallback without rich/typer
    HAS_RICH = False
    print("Installing typer and rich for better CLI experience:")
    print("pip install typer rich")

from config import Config
from src.data import AlpacaDataProvider
from src.trade import TradingEngine
from src.strategy import MomentumStrategy

if HAS_RICH:
    console = Console()
    
    # Create Typer app
    app = typer.Typer(
        name="trading-system",
        help="🚀 Quantitative Trading System (Python 3.12 Compatible)",
        add_completion=False
    )
    
    @app.command("account-info")
    def account_info():
        """💼 Display account information"""
        account_info_impl()
    
    @app.command("validate-config") 
    def validate_config_cmd():
        """🔧 Validate configuration"""
        validate_config()
        
    def account_info_impl():
        try:
            engine = TradingEngine()
            account_data = engine.get_account_info()
            
            if not account_data:
                console.print("[red]❌ Failed to fetch account information[/red]")
                raise typer.Exit(1)
            
            # Create table
            table = Table(title="💼 Account Information", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Portfolio Value", f"${account_data.get('portfolio_value', 0):,.2f}")
            table.add_row("Buying Power", f"${account_data.get('buying_power', 0):,.2f}")
            table.add_row("Cash", f"${account_data.get('cash', 0):,.2f}")
            table.add_row("Day Trades", str(account_data.get('day_trade_count', 0)))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            raise typer.Exit(1)
    
    @app.command("get-data")
    def get_data(
        symbol: Annotated[str, typer.Argument(help="Stock symbol")],
        days: Annotated[int, typer.Option("--days", "-d", help="Days back")] = 30
    ):
        """📊 Fetch market data"""
        try:
            provider = AlpacaDataProvider()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            console.print(f"[blue]📊 Fetching data for {symbol.upper()}...[/blue]")
            
            data = provider.get_bars(symbol.upper(), '1Day', start_date, end_date)
            
            if data.empty:
                console.print(f"[yellow]⚠️  No data found for {symbol}[/yellow]")
                return
            
            # Show summary
            latest_price = data['close'].iloc[-1]
            console.print(f"""
[green]✅ Data Retrieved Successfully[/green]

Symbol: [cyan]{symbol.upper()}[/cyan]
Records: [yellow]{len(data)}[/yellow]
Date Range: {data.index[0].date()} to {data.index[-1].date()}
Latest Price: [magenta]${latest_price:.2f}[/magenta]
            """)
            
            # Show recent data
            table = Table(title=f"Recent Data for {symbol.upper()}")
            table.add_column("Date")
            table.add_column("Open", style="cyan")
            table.add_column("High", style="green")
            table.add_column("Low", style="red")
            table.add_column("Close", style="yellow")
            table.add_column("Volume", style="blue")
            
            for idx, row in data.tail(5).iterrows():
                table.add_row(
                    str(idx.date()),
                    f"${row['open']:.2f}",
                    f"${row['high']:.2f}",
                    f"${row['low']:.2f}",
                    f"${row['close']:.2f}",
                    f"{int(row['volume']):,}"
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            raise typer.Exit(1)
    
    @app.command("run-strategy")
    def run_strategy(
        symbols: Annotated[list[str], typer.Argument(help="Stock symbols")],
        dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate only")] = True
    ):
        """🧠 Run momentum strategy"""
        try:
            console.print(f"[blue]🧠 Running momentum strategy on: {', '.join(symbols)}[/blue]")
            
            provider = AlpacaDataProvider()
            strategy = MomentumStrategy()
            
            # Fetch data for all symbols
            historical_data = {}
            current_prices = {}
            
            for symbol in symbols:
                symbol = symbol.upper()
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                
                data = provider.get_bars(symbol, '1Day', start_date, end_date)
                if not data.empty:
                    historical_data[symbol] = data
                    current_prices[symbol] = data['close'].iloc[-1]
            
            if not historical_data:
                console.print("[red]❌ No data retrieved for any symbols[/red]")
                return
            
            # Generate signals
            from src.trade import Portfolio
            portfolio = Portfolio()
            
            signals = strategy.generate_signals(
                current_date=datetime.now(),
                current_prices=current_prices,
                current_data={},
                historical_data=historical_data,
                portfolio=portfolio
            )
            
            if signals:
                table = Table(title="🧠 Strategy Signals")
                table.add_column("Symbol", style="cyan")
                table.add_column("Action", style="green")
                table.add_column("Quantity", style="yellow")
                table.add_column("Confidence", style="magenta")
                table.add_column("Reason", style="white")
                
                for signal in signals:
                    confidence = f"{signal.get('confidence', 0):.1%}"
                    table.add_row(
                        signal['symbol'],
                        signal['action'].upper(),
                        str(signal['quantity']),
                        confidence,
                        signal.get('reason', 'N/A')[:50] + "..." if len(signal.get('reason', '')) > 50 else signal.get('reason', 'N/A')
                    )
                
                console.print(table)
                
                if dry_run:
                    console.print("[blue]🧪 DRY RUN: No actual trades executed[/blue]")
                else:
                    console.print("[yellow]⚠️  Live trading requires manual implementation[/yellow]")
            else:
                console.print("[yellow]📭 No signals generated[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            raise typer.Exit(1)
    
    @app.command("validate-config")
    def validate_config():
        """🔧 Validate configuration"""
        try:
            console.print("[blue]🔧 Validating configuration...[/blue]")
            
            # Check Alpaca
            alpaca_valid = Config.validate_alpaca_config()
            alpaca_status = "✅ Valid" if alpaca_valid else "❌ Invalid/Missing"
            
            # Check Finhub
            finhub_valid = Config.validate_finhub_config()
            finhub_status = "✅ Valid" if finhub_valid else "❌ Invalid/Missing"
            
            # Check FRED
            fred_valid = Config.validate_fred_config()
            fred_status = "✅ Valid" if fred_valid else "❌ Invalid/Missing"
            
            table = Table(title="🔧 Configuration Status")
            table.add_column("Service", style="cyan")
            table.add_column("Status", style="white")
            
            table.add_row("Alpaca Trading", alpaca_status)
            table.add_row("Finhub News", finhub_status)
            table.add_row("FRED Economic", fred_status)
            
            console.print(table)
            
            if all([alpaca_valid, finhub_valid, fred_valid]):
                console.print("[green]🎉 All configurations valid![/green]")
            else:
                console.print("[yellow]⚠️  Some configurations missing. Check your .env file[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    @app.callback()
    def main_callback(
        version: Annotated[bool, typer.Option("--version", help="Show version")] = False
    ):
        """🚀 Quantitative Trading System"""
        if version:
            console.print("[blue]Trading System v2.0.0 (Python 3.12 Compatible)[/blue]")
            raise typer.Exit()

else:
    # Fallback for systems without rich/typer
    def main():
        print("🚀 Quantitative Trading System")
        print("Available commands:")
        print("1. Account info")
        print("2. Get data")  
        print("3. Validate config")
        
        choice = input("Enter choice (1-3): ").strip()
        
        try:
            if choice == "1":
                engine = TradingEngine()
                account_data = engine.get_account_info()
                print(f"Portfolio Value: ${account_data.get('portfolio_value', 0):,.2f}")
                print(f"Cash: ${account_data.get('cash', 0):,.2f}")
                
            elif choice == "2":
                symbol = input("Enter symbol: ").strip().upper()
                provider = AlpacaDataProvider()
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                data = provider.get_bars(symbol, '1Day', start_date, end_date)
                print(f"Retrieved {len(data)} records for {symbol}")
                if not data.empty:
                    print(f"Latest price: ${data['close'].iloc[-1]:.2f}")
                    
            elif choice == "3":
                print("Alpaca:", "✅" if Config.validate_alpaca_config() else "❌")
                print("Finhub:", "✅" if Config.validate_finhub_config() else "❌") 
                print("FRED:", "✅" if Config.validate_fred_config() else "❌")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if HAS_RICH:
        app()
    else:
        main()