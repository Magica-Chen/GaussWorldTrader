#!/usr/bin/env python3
"""
Smoke tests for the unified dashboard and strategy registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports() -> None:
    import streamlit  # noqa: F401
    import plotly  # noqa: F401
    import pandas  # noqa: F401
    import numpy  # noqa: F401


def test_dashboard_module() -> None:
    dashboard_path = project_root / "src" / "ui" / "dashboard.py"
    assert dashboard_path.exists()

    from src.ui import dashboard as dashboard_module

    assert hasattr(dashboard_module, "Dashboard")
    assert hasattr(dashboard_module, "main")


def test_strategy_registry() -> None:
    from src.strategy import get_strategy_registry

    registry = get_strategy_registry()
    strategies = registry.list_strategies()
    assert "momentum" in strategies

    instance = registry.create("momentum")
    assert instance.meta.name == "momentum"
