#!/usr/bin/env python3
"""
Gauss World Trader - Python 3.12+ Optimized Entry Point
Features async operations, rich CLI, and performance monitoring
Named after Carl Friedrich Gauss, pioneer of mathematical finance
"""
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check Python version
if sys.version_info < (3, 12):
    print("🚨 Gauss World Trader is optimized for Python 3.12+")
    print(f"📍 Current version: Python {sys.version_info.major}.{sys.version_info.minor}")
    print("💡 Please upgrade to Python 3.12 for optimal performance")
    sys.exit(1)

from config.optimized_config import get_config

# Import new modules
try:
    from src.account import AccountManager, PositionManager, OrderManager
    from src.agent import AgentManager
    ACCOUNT_MODULE_AVAILABLE = True
    AGENT_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    ACCOUNT_MODULE_AVAILABLE = False
    AGENT_MODULE_AVAILABLE = False

try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


def parse_cli_arguments():
    """Parse command line arguments for CLI selection."""
    parser = argparse.ArgumentParser(
        description="🌍 Gauss World Trader - Quantitative Trading System",
        add_help=False  # We'll handle help through the selected CLI
    )
    
    parser.add_argument(
        '--cli',
        choices=['modern', 'simple'],
        default='modern',
        help='Choose CLI interface: modern (default) or simple'
    )
    
    # Parse only known args to allow CLI-specific arguments to pass through
    args, remaining = parser.parse_known_args()
    return args.cli, remaining


def show_startup_banner(cli_type: str):
    """Show startup banner with CLI type information."""
    cli_description = "Modern Rich CLI" if cli_type == "modern" else "Simple CLI"
    
    if HAS_RICH and console:
        console.print(f"""
[bold blue]🌍 Gauss World Trader[/bold blue]
[cyan]Python 3.12 Compatible • High Performance Trading • Named after Carl Friedrich Gauss[/cyan]
[dim]Interface: {cli_description}[/dim]
        """)
        
        # Basic config validation
        try:
            config = get_config()
            console.print(config.get_validation_summary())
                
        except Exception as e:
            console.print(f"[red]❌ Configuration Error: {e}[/red]")
            console.print("[yellow]💡 Please check your .env file and API credentials[/yellow]")
            sys.exit(1)
    else:
        print("🌍 Gauss World Trader")
        print("Python 3.12 Compatible • High Performance Trading")
        print("Named after Carl Friedrich Gauss")
        print(f"Interface: {cli_description}")
        print("=" * 50)


def main() -> None:
    """
    Main entry point with CLI selection
    - Parses CLI selection arguments
    - Validates configuration
    - Shows system status
    - Launches selected CLI interface
    """
    
    # Parse CLI selection arguments
    cli_type, remaining_args = parse_cli_arguments()
    
    # Show startup banner with selected CLI info
    show_startup_banner(cli_type)
    
    # Temporarily modify sys.argv to pass remaining arguments to the selected CLI
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_args
    
    try:
        if cli_type == 'simple':
            # Import and launch simple CLI
            from src.ui.simple_cli import main as simple_main
            simple_main()
        else:
            # Import and launch modern CLI (default)
            from src.ui.modern_cli import app
            app()
    except KeyboardInterrupt:
        if HAS_RICH and console:
            console.print("\n[yellow]👋 Trading system shutdown complete[/yellow]")
        else:
            print("\n👋 Trading system shutdown complete")
    except Exception as e:
        if HAS_RICH and console:
            console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        else:
            print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

if __name__ == '__main__':
    # Use Python 3.12's improved startup performance
    main()