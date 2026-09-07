"""Restrained Rich presentation shared by the command-line entry points."""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme({"brand": "cyan", "muted": "dim", "success": "green", "warning": "yellow"})


def make_console(**kwargs):
    return Console(theme=THEME, highlight=False, **kwargs)


def banner(console, workspace, detail="Research. Validate. Execute."):
    title = Text("GAUSS", style="bold cyan")
    title.append("  WORLD TRADER", style="bold")
    title.append(f"\n{workspace}", style="bold")
    title.append(f"\n{detail}", style="dim")
    console.print(Panel(title, border_style="cyan", box=box.ROUNDED, padding=(1, 2)))


def table(title, *columns):
    result = Table(
        title=title,
        title_style="bold",
        title_justify="left",
        box=box.SIMPLE,
        header_style="bold cyan",
        padding=(0, 1),
        expand=False,
    )
    for column in columns:
        result.add_column(column, overflow="fold")
    return result
