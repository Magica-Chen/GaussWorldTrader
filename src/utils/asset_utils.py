"""Helpers for asset-type inference and symbol normalization."""
from __future__ import annotations

from typing import Iterable, List, Optional


def normalize_asset_type(asset_type: Optional[str]) -> str:
    """Normalize asset type names to stock/crypto/option."""
    if not asset_type:
        return "stock"
    value = asset_type.strip().lower()
    if value in {"stock", "stocks", "equity", "equities"}:
        return "stock"
    if value in {"crypto", "cryptocurrency", "cryptocurrencies"}:
        return "crypto"
    if value in {"option", "options"}:
        return "option"
    return "stock"


def is_option_symbol(symbol: str) -> bool:
    """Check if a symbol matches OCC option format."""
    if not isinstance(symbol, str):
        return False
    symbol = symbol.strip().upper()
    return (
        len(symbol) > 10
        and ("C" in symbol[-9:] or "P" in symbol[-9:])
        and any(char.isdigit() for char in symbol[-8:])
    ) or "C00" in symbol or "P00" in symbol


def is_crypto_symbol(symbol: str) -> bool:
    """Check if a symbol looks like a crypto pair."""
    if not isinstance(symbol, str):
        return False
    symbol = symbol.strip().upper()
    return "/" in symbol or (symbol.endswith("USD") and len(symbol) > 3)


def infer_asset_type(symbol: str) -> str:
    """Infer asset type from a symbol string."""
    if not isinstance(symbol, str) or not symbol.strip():
        return "stock"
    if is_option_symbol(symbol):
        return "option"
    if is_crypto_symbol(symbol):
        return "crypto"
    return "stock"


def normalize_symbol(symbol: str, asset_type: Optional[str] = None) -> str:
    """Normalize symbol formatting based on asset type."""
    if not isinstance(symbol, str):
        return ""
    symbol = symbol.strip().upper()
    if not symbol:
        return ""
    normalized_type = normalize_asset_type(asset_type or infer_asset_type(symbol))
    if normalized_type == "crypto":
        if "/" in symbol:
            return symbol
        if symbol.endswith("USD") and len(symbol) > 3:
            return f"{symbol[:-3]}/USD"
    return symbol


def merge_unique_symbols(symbols: Iterable[str], asset_type: Optional[str] = None) -> List[str]:
    """Merge symbols into a unique, normalized list preserving order."""
    result: List[str] = []
    seen = set()
    for symbol in symbols:
        normalized = normalize_symbol(symbol, asset_type)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
