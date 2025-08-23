#!/usr/bin/env python3
"""
Gauss World Trader - Python 3.12+ Optimized Entry Point
Features async operations, rich CLI, and performance monitoring
Named after Carl Friedrich Gauss, pioneer of mathematical finance
"""
import sys
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

# Import modern CLI components
from src.ui.modern_cli import app
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


def main() -> None:
    """
    Main entry point with Python 3.12 optimizations
    - Validates configuration
    - Shows system status
    - Launches modern CLI
    """
    
    # Show startup banner
    if HAS_RICH and console:
        console.print("""
[bold blue]🌍 Gauss World Trader[/bold blue]
[cyan]Python 3.12 Compatible • High Performance Trading • Named after Carl Friedrich Gauss[/cyan]
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
        print("=" * 50)
    
    # Launch CLI
    try:
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

if __name__ == '__main__':
    # Use Python 3.12's improved startup performance
    main()