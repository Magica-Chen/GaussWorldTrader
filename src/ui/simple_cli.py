"""
Simple CLI interface with Python 3.12 compatibility
Uses core CLI abstraction to eliminate code duplication
"""
from __future__ import annotations

from typing import Annotated

from src.ui.core_cli import BaseCLI, SimpleFallbackConcrete, HAS_RICH

if HAS_RICH:
    import typer
    from src.trade import Portfolio
    from src.stock_strategy import MomentumStrategy
    from src.utils.timezone_utils import now_et
    from datetime import timedelta
    from src.data import AlpacaDataProvider


class SimpleCLI(BaseCLI):
    """Simple CLI implementation using core abstraction."""
    
    def __init__(self):
        super().__init__(
            app_name="simple-trading",
            app_help="🚀 Simple Quantitative Trading System"
        )
        self.setup_commands()
    
    def setup_commands(self) -> None:
        """Setup CLI commands specific to simple interface."""
        if not self.has_rich or not self.app:
            return
        
        @self.app.command("account-info")
        def account_info():
            """💼 Display account information"""
            if not self.display_account_info():
                self.exit_with_error()
        
        @self.app.command("validate-config")
        def validate_config_cmd():
            """🔧 Validate configuration"""
            if not self.validate_config_impl():
                self.exit_with_error()
        
        @self.app.command("check-positions")
        def check_positions_cmd():
            """📈 Check current positions and recent orders"""
            self.handle_portfolio_command("check-positions")
        
        @self.app.command("watchlist-trade")
        def watchlist_trade_cmd():
            """🎯 Analyze watchlist and execute trades"""
            self.handle_portfolio_command("watchlist-trade")
        
        @self.app.command("get-data")
        def get_data(
            symbol: Annotated[str, typer.Argument(help="Stock symbol")],
            days: Annotated[int, typer.Option("--days", "-d", help="Days back")] = 30
        ):
            """📊 Fetch market data"""
            if not self.display_market_data(symbol, days):
                self.exit_with_error()
        
        @self.app.command("run-strategy")
        def run_strategy(
            symbols: Annotated[list[str], typer.Argument(help="Stock symbols")],
            dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate only")] = True
        ):
            """🧠 Run momentum strategy"""
            self._run_momentum_strategy(symbols, dry_run)
        
        @self.app.callback()
        def main_callback(
            version: Annotated[bool, typer.Option("--version", help="Show version")] = False
        ):
            """🚀 Simple Quantitative Trading System"""
            if version:
                self.print_message("Simple Trading System v2.0.0 (Python 3.12 Compatible)", "blue")
                self.exit_with_error(0)
    
    def _run_momentum_strategy(self, symbols: list[str], dry_run: bool) -> None:
        """Run momentum strategy on given symbols."""
        try:
            self.print_info(f"Running momentum strategy on: {', '.join(symbols)}")
            
            provider = AlpacaDataProvider()
            strategy = MomentumStrategy()
            
            historical_data = {}
            current_prices = {}
            
            for symbol in symbols:
                symbol = symbol.upper()
                current_time = now_et()
                start_date = current_time - timedelta(days=100)
                
                data = provider.get_bars(symbol, '1Day', start_date)
                if not data.empty:
                    historical_data[symbol] = data
                    current_prices[symbol] = data['close'].iloc[-1]
            
            if not historical_data:
                self.print_error("No data retrieved for any symbols")
                return
            
            portfolio = Portfolio()
            signals = strategy.generate_signals(
                current_date=now_et(),
                current_prices=current_prices,
                current_data={},
                historical_data=historical_data,
                portfolio=portfolio
            )
            
            if signals:
                if self.has_rich and self.console:
                    from rich.table import Table
                    table = Table(title="🧠 Strategy Signals")
                    table.add_column("Symbol", style="cyan")
                    table.add_column("Action", style="green")
                    table.add_column("Quantity", style="yellow")
                    table.add_column("Confidence", style="magenta")
                    table.add_column("Reason", style="white")
                    
                    for signal in signals:
                        confidence = f"{signal.get('confidence', 0):.1%}"
                        reason = signal.get('reason', 'N/A')
                        if len(reason) > 50:
                            reason = reason[:50] + "..."
                        
                        table.add_row(
                            signal['symbol'],
                            signal['action'].upper(),
                            str(signal['quantity']),
                            confidence,
                            reason
                        )
                    
                    self.console.print(table)
                else:
                    for signal in signals:
                        print(f"{signal['symbol']}: {signal['action']} {signal['quantity']} "
                              f"(confidence: {signal.get('confidence', 0):.1%})")
                
                if dry_run:
                    self.print_info("DRY RUN: No actual trades executed")
                else:
                    self.print_warning("Live trading requires manual implementation")
            else:
                self.print_warning("No signals generated")
                
        except Exception as e:
            self.print_error(str(e))
            self.exit_with_error()
    
    def run(self) -> None:
        """Run the simple CLI."""
        if self.has_rich and self.app:
            self.app()
        else:
            fallback = SimpleFallbackConcrete("fallback", "Fallback CLI")
            fallback.run_interactive()


# Global app instance for backward compatibility
if HAS_RICH:
    _cli_instance = SimpleCLI()
    app = _cli_instance.app
    
    def main():
        _cli_instance.run()
else:
    app = None
    
    def main():
        fallback = SimpleFallbackConcrete("fallback", "Fallback CLI")
        fallback.run_interactive()


if __name__ == "__main__":
    main()