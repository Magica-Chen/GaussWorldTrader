#!/usr/bin/env python3
"""Unified live trading script with interactive CLI."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from src.trade.live_trading_crypto import create_crypto_engines, get_default_crypto_symbols
from src.trade.live_trading_option import create_option_engines, get_default_option_symbols
from src.trade.live_trading_stock import create_stock_engines, get_default_stock_symbols
from src.strategy.registry import get_strategy_registry
from src.trade.live_runner import run_live_engines
from src.agent.watchlist_manager import WatchlistManager

console = Console()


BANNER = """
 ██████╗  █████╗ ██╗   ██╗███████╗███████╗ ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗
██╔════╝ ██╔══██╗██║   ██║██╔════╝██╔════╝ ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
██║  ███╗███████║██║   ██║███████╗███████╗ ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
██║   ██║██╔══██║██║   ██║╚════██║╚════██║ ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
╚██████╔╝██║  ██║╚██████╔╝███████║███████║ ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
 ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝
                              T  R  A  D  E  R
"""


# Strategies available per asset type
STOCK_STRATEGIES = ["momentum", "value", "trend_following", "scalping", "statistical_arbitrage"]
CRYPTO_STRATEGIES = ["crypto_momentum"]
OPTION_STRATEGIES = ["wheel"]


@dataclass
class TradingConfig:
    """Trading configuration for a session."""
    asset_types: List[str] = field(default_factory=list)
    symbols: Dict[str, List[str]] = field(default_factory=dict)
    strategies: Dict[str, str] = field(default_factory=dict)
    timeframe: str = "1Hour"
    lookback_days: int = 30
    risk_pct: float = 0.05
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    execute: bool = True
    auto_exit: bool = True
    # Stock-specific
    fractional: bool = False
    extended_hours: bool = False
    # Crypto-specific
    crypto_loc: str = "us"
    # Option-specific
    roll_days: int = 5


def show_banner() -> None:
    """Display the application banner."""
    banner_text = Text(BANNER, style="bold cyan")
    console.print(banner_text)
    console.print()


def show_watchlist_summary() -> None:
    """Display current watchlist summary."""
    manager = WatchlistManager()
    table = Table(title="Current Watchlist", show_header=True, header_style="bold magenta")
    table.add_column("Asset Type", style="cyan", width=12)
    table.add_column("Symbols", style="green")

    for asset_type in ["stock", "crypto", "option"]:
        symbols = manager.get_watchlist(asset_type=asset_type)
        if symbols:
            table.add_row(asset_type.upper(), ", ".join(symbols))
        else:
            table.add_row(asset_type.upper(), "[dim]None[/dim]")

    console.print(table)
    console.print()


def select_asset_types() -> List[str]:
    """Interactive selection of asset types to trade."""
    console.print(Panel("[bold]Asset Type Selection[/bold]", style="blue"))
    console.print("Select which asset types to trade:\n")

    options = [
        ("1", "stock", "Stocks (equities)"),
        ("2", "crypto", "Cryptocurrency (24/7)"),
        ("3", "option", "Options (wheel strategy)"),
        ("4", "all", "All asset types"),
    ]

    for key, _, desc in options:
        console.print(f"  [cyan]{key}[/cyan] - {desc}")

    console.print()
    choice = Prompt.ask(
        "Enter your choice",
        choices=["1", "2", "3", "4"],
        default="4",
    )

    if choice == "4":
        return ["stock", "crypto", "option"]
    return [options[int(choice) - 1][1]]


def get_symbols_for_type(asset_type: str) -> List[str]:
    """Get symbols for a specific asset type."""
    if asset_type == "stock":
        return get_default_stock_symbols()
    elif asset_type == "crypto":
        return get_default_crypto_symbols()
    elif asset_type == "option":
        return get_default_option_symbols()
    return []


def configure_symbols(asset_types: List[str]) -> Dict[str, List[str]]:
    """Configure symbols for each asset type."""
    console.print()
    console.print(Panel("[bold]Symbol Configuration[/bold]", style="blue"))

    symbols: Dict[str, List[str]] = {}

    for asset_type in asset_types:
        defaults = get_symbols_for_type(asset_type)
        console.print(f"\n[cyan]{asset_type.upper()}[/cyan] defaults: {', '.join(defaults)}")

        use_defaults = Confirm.ask(
            f"Use default {asset_type} symbols?",
            default=True,
        )

        if use_defaults:
            symbols[asset_type] = defaults
        else:
            custom = Prompt.ask(
                f"Enter {asset_type} symbols (comma-separated)",
                default=",".join(defaults),
            )
            symbols[asset_type] = [s.strip() for s in custom.split(",") if s.strip()]

    return symbols


def get_strategies_for_type(asset_type: str) -> List[str]:
    """Get available strategies for a specific asset type."""
    if asset_type == "stock":
        return STOCK_STRATEGIES
    elif asset_type == "crypto":
        return CRYPTO_STRATEGIES
    elif asset_type == "option":
        return OPTION_STRATEGIES
    return []


def get_default_strategy(asset_type: str) -> str:
    """Get default strategy for an asset type."""
    defaults = {"stock": "momentum", "crypto": "crypto_momentum", "option": "wheel"}
    return defaults.get(asset_type, "momentum")


def configure_strategies(asset_types: List[str]) -> Dict[str, str]:
    """Configure strategies for each asset type."""
    console.print()
    console.print(Panel("[bold]Strategy Selection[/bold]", style="blue"))

    strategies: Dict[str, str] = {}

    for asset_type in asset_types:
        available = get_strategies_for_type(asset_type)
        default = get_default_strategy(asset_type)

        if len(available) == 1:
            # Only one strategy available, use it automatically
            strategies[asset_type] = available[0]
            console.print(f"[cyan]{asset_type.upper()}[/cyan]: {available[0]} (only option)")
        else:
            console.print(f"\n[cyan]{asset_type.upper()}[/cyan] strategies:")
            for i, strat in enumerate(available, 1):
                marker = " (default)" if strat == default else ""
                console.print(f"  [dim]{i}[/dim] - {strat}{marker}")

            choice = Prompt.ask(
                f"Select {asset_type} strategy",
                choices=[str(i) for i in range(1, len(available) + 1)],
                default="1",
            )
            strategies[asset_type] = available[int(choice) - 1]

    return strategies


def configure_parameters(config: TradingConfig) -> TradingConfig:
    """Configure trading parameters interactively."""
    console.print()
    console.print(Panel("[bold]Trading Parameters[/bold]", style="blue"))

    # Show defaults
    table = Table(show_header=True, header_style="bold")
    table.add_column("Parameter", style="cyan")
    table.add_column("Default", style="green")
    table.add_column("Description")

    table.add_row("Timeframe", config.timeframe, "Bar timeframe for signals")
    table.add_row("Lookback", f"{config.lookback_days} days", "Historical data period")
    table.add_row("Risk %", f"{config.risk_pct:.1%}", "Portfolio risk per trade")
    table.add_row("Stop Loss", f"{config.stop_loss_pct:.1%}", "Stop-loss percentage")
    table.add_row("Take Profit", f"{config.take_profit_pct:.1%}", "Take-profit percentage")
    table.add_row("Execute", str(config.execute), "Execute live trades")
    table.add_row("Auto Exit", str(config.auto_exit), "Auto-close on SL/TP")

    console.print(table)
    console.print()

    use_defaults = Confirm.ask("Use default parameters?", default=True)

    if not use_defaults:
        config.timeframe = Prompt.ask("Timeframe", default=config.timeframe)
        config.lookback_days = int(Prompt.ask(
            "Lookback days", default=str(config.lookback_days)
        ))
        config.risk_pct = float(Prompt.ask(
            "Risk % (decimal)", default=str(config.risk_pct)
        ))
        config.stop_loss_pct = float(Prompt.ask(
            "Stop loss % (decimal)", default=str(config.stop_loss_pct)
        ))
        config.take_profit_pct = float(Prompt.ask(
            "Take profit % (decimal)", default=str(config.take_profit_pct)
        ))
        config.execute = Confirm.ask("Execute live trades?", default=config.execute)
        config.auto_exit = Confirm.ask("Auto-exit on SL/TP?", default=config.auto_exit)

        if "stock" in config.asset_types:
            config.fractional = Confirm.ask("Allow fractional shares?", default=False)
            config.extended_hours = Confirm.ask("Trade extended hours?", default=False)

    return config


def show_final_config(config: TradingConfig) -> None:
    """Display final configuration before starting."""
    console.print()
    console.print(Panel("[bold]Trading Configuration Summary[/bold]", style="green"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    for asset_type, symbols in config.symbols.items():
        table.add_row(f"{asset_type.upper()} Symbols", ", ".join(symbols))
        strategy = config.strategies.get(asset_type, get_default_strategy(asset_type))
        table.add_row(f"{asset_type.upper()} Strategy", strategy)

    table.add_row("Timeframe", config.timeframe)
    table.add_row("Lookback", f"{config.lookback_days} days")
    table.add_row("Risk", f"{config.risk_pct:.1%}")
    table.add_row("Stop Loss", f"{config.stop_loss_pct:.1%}")
    table.add_row("Take Profit", f"{config.take_profit_pct:.1%}")
    table.add_row("Execute", "[green]Yes[/green]" if config.execute else "[red]No (Dry Run)[/red]")
    table.add_row("Auto Exit", "[green]Yes[/green]" if config.auto_exit else "[yellow]No[/yellow]")

    console.print(table)
    console.print()


def run_trading(config: TradingConfig) -> None:
    """Execute trading based on configuration."""
    console.print()
    mode = "[green]LIVE[/green]" if config.execute else "[yellow]DRY RUN[/yellow]"
    console.print(Panel(f"[bold]Starting Trading - {mode}[/bold]", style="cyan"))
    console.print()

    engine_groups: Dict[str, list] = {}

    for asset_type in config.asset_types:
        symbols = config.symbols.get(asset_type, [])
        if not symbols:
            continue

        console.print(f"[cyan]Creating {asset_type.upper()} engines...[/cyan]")

        strategy = config.strategies.get(asset_type, get_default_strategy(asset_type))

        if asset_type == "stock":
            engines = create_stock_engines(
                symbols=symbols,
                timeframe=config.timeframe,
                lookback_days=config.lookback_days,
                risk_pct=config.risk_pct,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
                execute=config.execute,
                auto_exit=config.auto_exit,
                fractional=config.fractional,
                extended_hours=config.extended_hours,
                strategy=strategy,
            )
            if engines:
                engine_groups["stock"] = engines
        elif asset_type == "crypto":
            engines = create_crypto_engines(
                symbols=symbols,
                timeframe=config.timeframe,
                lookback_days=config.lookback_days,
                crypto_loc=config.crypto_loc,
                risk_pct=config.risk_pct,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
                execute=config.execute,
                auto_exit=config.auto_exit,
                strategy=strategy,
            )
            if engines:
                engine_groups["crypto"] = engines
        elif asset_type == "option":
            engines = create_option_engines(
                symbols=symbols,
                timeframe=config.timeframe,
                lookback_days=config.lookback_days,
                risk_pct=config.risk_pct,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
                execute=config.execute,
                auto_exit=config.auto_exit,
                roll_days=config.roll_days,
                strategy=strategy,
            )
            if engines:
                engine_groups["option"] = engines

    if not engine_groups:
        console.print("[yellow]No engines to run.[/yellow]")
        return

    def run_group(asset_type: str, engines: list) -> None:
        """Run a group of same-type engines."""
        console.print(f"[cyan]Running {asset_type.upper()} ({len(engines)} symbols)...[/cyan]")
        if len(engines) == 1:
            engines[0].start()
        else:
            run_live_engines(engines)

    # Single asset type - run directly
    if len(engine_groups) == 1:
        asset_type, engines = next(iter(engine_groups.items()))
        run_group(asset_type, engines)
        return

    # Multiple asset types - Alpaca connection limit requires sequential execution
    console.print()
    console.print(Panel(
        "[yellow]Multiple asset types selected.[/yellow]\n"
        "Due to Alpaca connection limits, only one stream can run at a time.\n"
        "Press [bold]Ctrl+C[/bold] to stop current type and move to the next.",
        title="Connection Limit",
        style="yellow",
    ))
    console.print()

    asset_types = list(engine_groups.keys())
    for i, asset_type in enumerate(asset_types):
        engines = engine_groups[asset_type]
        remaining = asset_types[i + 1:] if i + 1 < len(asset_types) else []

        if remaining:
            console.print(f"[dim]Next up: {', '.join(t.upper() for t in remaining)}[/dim]")

        try:
            run_group(asset_type, engines)
        except KeyboardInterrupt:
            if remaining:
                console.print(f"\n[yellow]Stopped {asset_type.upper()}. Moving to next...[/yellow]\n")
                continue
            else:
                console.print("\n[yellow]Trading stopped.[/yellow]")
                break


def quick_start() -> Optional[TradingConfig]:
    """Quick start with all defaults from watchlist."""
    console.print()
    console.print(Panel("[bold]Quick Start[/bold]", style="green"))
    console.print("Starting with all defaults from watchlist.json\n")

    config = TradingConfig()
    config.asset_types = ["stock", "crypto", "option"]

    for asset_type in config.asset_types:
        symbols = get_symbols_for_type(asset_type)
        if symbols:
            config.symbols[asset_type] = symbols
            config.strategies[asset_type] = get_default_strategy(asset_type)
            console.print(
                f"  [cyan]{asset_type.upper()}[/cyan]: {', '.join(symbols)} "
                f"([dim]{config.strategies[asset_type]}[/dim])"
            )

    console.print()
    config.execute = Confirm.ask(
        "Execute live trades? (No = dry run)",
        default=False,
    )

    return config


def main() -> None:
    """Main entry point for interactive trading CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    try:
        console.clear()
        show_banner()
        show_watchlist_summary()

        # Main menu
        console.print(Panel("[bold]Trading Mode Selection[/bold]", style="blue"))
        console.print("  [cyan]1[/cyan] - Quick Start (use watchlist defaults)")
        console.print("  [cyan]2[/cyan] - Custom Configuration")
        console.print("  [cyan]q[/cyan] - Quit")
        console.print()

        choice = Prompt.ask("Select mode", choices=["1", "2", "q"], default="1")

        if choice == "q":
            console.print("\n[yellow]Exiting...[/yellow]")
            sys.exit(0)

        if choice == "1":
            config = quick_start()
        else:
            config = TradingConfig()
            config.asset_types = select_asset_types()
            config.symbols = configure_symbols(config.asset_types)
            config.strategies = configure_strategies(config.asset_types)
            config = configure_parameters(config)

        if config:
            show_final_config(config)
            if Confirm.ask("[bold]Start trading?[/bold]", default=True):
                run_trading(config)
            else:
                console.print("\n[yellow]Trading cancelled.[/yellow]")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
