"""Helpers for live trading scripts."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from src.utils.asset_utils import infer_asset_type, merge_unique_symbols, normalize_asset_type


def parse_symbol_args(symbols: Iterable[str] | None, single_symbol: Optional[str]) -> List[str]:
    """Parse comma-separated CLI symbol arguments into a list."""
    parsed: List[str] = []
    if symbols:
        for item in symbols:
            for part in item.split(","):
                symbol = part.strip()
                if symbol:
                    parsed.append(symbol)
    if single_symbol:
        parsed.append(single_symbol)
    return parsed


def positions_for_asset_type(
    positions: Iterable[dict[str, Any]], asset_type: str
) -> List[str]:
    """Return position symbols filtered by asset type."""
    normalized_type = normalize_asset_type(asset_type)
    symbols = []
    for pos in positions:
        symbol = pos.get("symbol") if isinstance(pos, dict) else None
        if not symbol:
            continue
        if infer_asset_type(symbol) != normalized_type:
            continue
        symbols.append(symbol)
    return merge_unique_symbols(symbols, normalized_type)


def merge_symbol_sources(asset_type: str, *symbol_lists: Iterable[str]) -> List[str]:
    """Merge symbol lists into a unique, normalized list."""
    combined: List[str] = []
    for symbols in symbol_lists:
        if symbols:
            combined.extend(list(symbols))
    return merge_unique_symbols(combined, asset_type)
