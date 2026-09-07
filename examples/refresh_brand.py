"""Sync shared brand assets and the GitHub Pages registry snapshot.

Run: python examples/refresh_brand.py
"""

import json
import shutil
import sys
from pathlib import Path

# Repository bootstrap makes these scripts runnable before an editable install.
# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.strategy.registry import get_strategy_registry


def main():
    for name in ["gauss-mark.png", "gauss-banner.png"]:
        shutil.copyfile(ROOT / "assets/brand" / name, ROOT / "site/assets" / name)
    shutil.copyfile(ROOT / "assets/brand/gauss-mark.png", ROOT / "src/ui/assets/gauss-mark.png")
    for name in ["dashboard-preview.png", "terminal-preview.png"]:
        shutil.copyfile(ROOT / "docs/images" / name, ROOT / "site/assets" / name)
    registry = get_strategy_registry()
    items = []
    for name in registry.list_strategies():
        meta = registry.get_meta(name)
        source = f"src/strategy/{meta.asset_type}/{name}.py"
        if name == "crypto_momentum":
            source = "src/strategy/stock/momentum.py"
        if name == "multi_agent":
            source = "src/strategy/multi_agent_strategy.py"
        if not (ROOT / source).is_file():
            raise FileNotFoundError(source)
        items.append(
            {
                "name": name,
                "label": meta.label,
                "description": meta.description,
                "asset_type": meta.asset_type,
                "source": source,
            }
        )
    (ROOT / "site/strategies.js").write_text(
        "// Registry snapshot. Refresh with: python examples/refresh_brand.py\n"
        + "window.GAUSS_STRATEGIES = "
        + json.dumps(items, indent=2)
        + ";\n"
    )
    print("Shared brand assets and strategy snapshot synchronized.")


if __name__ == "__main__":
    main()
