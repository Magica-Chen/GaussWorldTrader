"""
Core CLI abstraction providing shared functionality for all CLI interfaces.
Eliminates code duplication and provides consistent command patterns.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from src.utils.timezone_utils import now_et

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from config import Config
from src.data import AlpacaDataProvider
from src.trade import TradingEngine
from src.strategy import MomentumStrategy


class BaseCLI(ABC):
    """Base CLI class providing shared functionality for all CLI implementations."""
    
    def __init__(self, app_name: str = "trading-system", app_help: str = "Trading System"):
        """Initialize base CLI with common setup."""
        self.console = Console() if HAS_RICH else None
        self.has_rich = HAS_RICH
        
        if HAS_RICH:
            self.app = typer.Typer(
                name=app_name,
                help=app_help,
                add_completion=False
            )
        else:
            self.app = None
    
    def print_message(self, message: str, style: str = "") -> None:
        """Print message with optional rich styling."""
        if self.has_rich and self.console:
            if style:
                self.console.print(f"[{style}]{message}[/{style}]")
            else:
                self.console.print(message)
        else:
            print(message)
    
    def print_error(self, message: str) -> None:
        """Print error message with consistent formatting."""
        self.print_message(f"❌ Error: {message}", "red")
    
    def print_warning(self, message: str) -> None:
        """Print warning message with consistent formatting."""
        self.print_message(f"⚠️  {message}", "yellow")
    
    def print_success(self, message: str) -> None:
        """Print success message with consistent formatting."""
        self.print_message(f"✅ {message}", "green")
    
    def print_info(self, message: str) -> None:
        """Print info message with consistent formatting."""
        self.print_message(f"📊 {message}", "blue")
    
    def create_account_table(self, account_data: dict) -> Table | None:
        """Create standardized account information table."""
        if not self.has_rich:
            return None
        
        table = Table(title="💼 Account Information", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Portfolio Value", f"${account_data.get('portfolio_value', 0):,.2f}")
        table.add_row("Buying Power", f"${account_data.get('buying_power', 0):,.2f}")
        table.add_row("Cash", f"${account_data.get('cash', 0):,.2f}")
        table.add_row("Day Trades", str(account_data.get('day_trade_count', 0)))
        
        return table
    
    def create_config_table(self) -> Table | None:
        """Create standardized configuration validation table."""
        if not self.has_rich:
            return None
        
        alpaca_valid = Config.validate_alpaca_config()
        finnhub_valid = Config.validate_finnhub_config()
        fred_valid = Config.validate_fred_config()
        
        alpaca_status = "✅ Valid" if alpaca_valid else "❌ Invalid/Missing"
        finnhub_status = "✅ Valid" if finnhub_valid else "❌ Invalid/Missing"
        fred_status = "✅ Valid" if fred_valid else "❌ Invalid/Missing"
        
        table = Table(title="🔧 Configuration Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="white")
        
        table.add_row("Alpaca Trading", alpaca_status)
        table.add_row("Finnhub News", finnhub_status)
        table.add_row("FRED Economic", fred_status)
        
        return table
    
    def create_data_table(self, symbol: str, data, recent_rows: int = 5) -> Table | None:
        """Create standardized market data table."""
        if not self.has_rich or data.empty:
            return None
        
        table = Table(title=f"Recent Data for {symbol.upper()}")
        table.add_column("Date")
        table.add_column("Open", style="cyan")
        table.add_column("High", style="green")
        table.add_column("Low", style="red")
        table.add_column("Close", style="yellow")
        table.add_column("Volume", style="blue")
        
        for idx, row in data.tail(recent_rows).iterrows():
            table.add_row(
                str(idx.date()),
                f"${row['open']:.2f}",
                f"${row['high']:.2f}",
                f"${row['low']:.2f}",
                f"${row['close']:.2f}",
                f"{int(row['volume']):,}"
            )
        
        return table
    
    def get_account_info_impl(self) -> dict | None:
        """Common account info implementation."""
        try:
            engine = TradingEngine()
            account_data = engine.get_account_info()
            
            if not account_data:
                self.print_error("Failed to fetch account information")
                return None
            
            return account_data
            
        except Exception as e:
            self.print_error(str(e))
            return None
    
    def display_account_info(self) -> bool:
        """Display account information using common logic."""
        account_data = self.get_account_info_impl()
        if not account_data:
            return False
        
        if self.has_rich and self.console:
            table = self.create_account_table(account_data)
            if table:
                self.console.print(table)
        else:
            print(f"Portfolio Value: ${account_data.get('portfolio_value', 0):,.2f}")
            print(f"Buying Power: ${account_data.get('buying_power', 0):,.2f}")
            print(f"Cash: ${account_data.get('cash', 0):,.2f}")
            print(f"Day Trades: {account_data.get('day_trade_count', 0)}")
        
        return True
    
    def validate_config_impl(self) -> bool:
        """Common configuration validation implementation."""
        try:
            self.print_info("Validating configuration...")
            
            alpaca_valid = Config.validate_alpaca_config()
            finnhub_valid = Config.validate_finnhub_config()
            fred_valid = Config.validate_fred_config()
            
            if self.has_rich and self.console:
                table = self.create_config_table()
                if table:
                    self.console.print(table)
            else:
                print("Alpaca:", "✅" if alpaca_valid else "❌")
                print("Finnhub:", "✅" if finnhub_valid else "❌")
                print("FRED:", "✅" if fred_valid else "❌")
            
            if all([alpaca_valid, finnhub_valid, fred_valid]):
                self.print_success("All configurations valid!")
                return True
            else:
                self.print_warning("Some configurations missing. Check your .env file")
                return False
                
        except Exception as e:
            self.print_error(str(e))
            return False
    
    def get_market_data(self, symbol: str, days: int = 30) -> tuple[Any, float | None]:
        """Common market data fetching implementation."""
        try:
            provider = AlpacaDataProvider()
            current_time = now_et()
            start_date = current_time - timedelta(days=days)
            
            self.print_info(f"Fetching data for {symbol.upper()}...")
            
            data = provider.get_bars(symbol.upper(), '1Day', start_date)
            
            if data.empty:
                self.print_warning(f"No data found for {symbol}")
                return None, None
            
            latest_price = data['close'].iloc[-1]
            return data, latest_price
            
        except Exception as e:
            self.print_error(str(e))
            return None, None
    
    def display_market_data(self, symbol: str, days: int = 30) -> bool:
        """Display market data using common logic."""
        data, latest_price = self.get_market_data(symbol, days)
        
        if data is None:
            return False
        
        self.print_success("Data Retrieved Successfully")
        self.print_message(f"Symbol: {symbol.upper()}", "cyan")
        self.print_message(f"Records: {len(data)}", "yellow")
        self.print_message(f"Date Range: {data.index[0].date()} to {data.index[-1].date()}")
        self.print_message(f"Latest Price: ${latest_price:.2f}", "magenta")
        
        if self.has_rich and self.console:
            table = self.create_data_table(symbol, data)
            if table:
                self.console.print(table)
        else:
            print(f"\nRecent data for {symbol.upper()}:")
            for idx, row in data.tail(5).iterrows():
                print(f"{idx.date()}: Open=${row['open']:.2f}, Close=${row['close']:.2f}, "
                      f"Volume={int(row['volume']):,}")
        
        return True
    
    def handle_portfolio_command(self, command: str) -> None:
        """Handle common portfolio-related commands."""
        try:
            if command == "check-positions":
                from src.ui.portfolio_commands import check_positions_and_orders
                check_positions_and_orders()
            elif command == "watchlist-trade":
                from src.ui.portfolio_commands import get_watchlists_and_trade
                get_watchlists_and_trade()
            else:
                self.print_error(f"Unknown portfolio command: {command}")
        except Exception as e:
            self.print_error(str(e))
    
    def exit_with_error(self, exit_code: int = 1) -> None:
        """Common error exit handling."""
        if self.has_rich and hasattr(typer, 'Exit'):
            raise typer.Exit(exit_code)
        else:
            exit(exit_code)
    
    @abstractmethod
    def setup_commands(self) -> None:
        """Setup CLI commands - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run(self) -> None:
        """Run the CLI - must be implemented by subclasses."""
        pass


# Utility functions that can be used without instantiation
def handle_portfolio_command(command: str) -> None:
    """Handle common portfolio-related commands."""
    try:
        if command == "check-positions":
            from src.ui.portfolio_commands import check_positions_and_orders
            check_positions_and_orders()
        elif command == "watchlist-trade":
            from src.ui.portfolio_commands import get_watchlists_and_trade
            get_watchlists_and_trade()
        else:
            print(f"❌ Error: Unknown portfolio command: {command}")
    except Exception as e:
        print(f"❌ Error: {e}")

def print_error(message: str) -> None:
    """Print error message with consistent formatting."""
    if HAS_RICH:
        console = Console()
        console.print(f"[red]❌ Error: {message}[/red]")
    else:
        print(f"❌ Error: {message}")

def print_warning(message: str) -> None:
    """Print warning message with consistent formatting."""
    if HAS_RICH:
        console = Console()
        console.print(f"[yellow]⚠️  {message}[/yellow]")
    else:
        print(f"⚠️  {message}")

def print_success(message: str) -> None:
    """Print success message with consistent formatting."""
    if HAS_RICH:
        console = Console()
        console.print(f"[green]✅ {message}[/green]")
    else:
        print(f"✅ {message}")

def print_info(message: str) -> None:
    """Print info message with consistent formatting."""
    if HAS_RICH:
        console = Console()
        console.print(f"[blue]📊 {message}[/blue]")
    else:
        print(f"📊 {message}")


class SimpleFallbackConcrete(BaseCLI):
    """Concrete implementation for fallback use."""
    
    def setup_commands(self) -> None:
        pass
    
    def run(self) -> None:
        self.run_interactive()
    
    def run_interactive(self) -> None:
        """Run interactive CLI without rich/typer."""
        print("🚀 Quantitative Trading System")
        print("Available commands:")
        print("1. Account info")
        print("2. Get data")
        print("3. Validate config")
        print("4. Check positions")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            self.display_account_info()
        elif choice == "2":
            symbol = input("Enter symbol: ").strip().upper()
            self.display_market_data(symbol)
        elif choice == "3":
            self.validate_config_impl()
        elif choice == "4":
            self.handle_portfolio_command("check-positions")
        else:
            print("Invalid choice")