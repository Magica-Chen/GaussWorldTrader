#!/usr/bin/env python3
"""
Gauss World Trader - Dashboard Launcher
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).parent
    dashboard_file = project_root / "src" / "ui" / "dashboard.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_file),
        "--server.port=3721",
        "--server.address=localhost",
        "--theme.base=light",
        "--theme.primaryColor=#1f77b4",
        "--theme.backgroundColor=#ffffff",
        "--theme.secondaryBackgroundColor=#f0f2f6",
        "--theme.textColor=#262730",
    ]

    print("🚀 Launching dashboard on http://localhost:3721")
    print("🔄 Press Ctrl+C to stop")

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
