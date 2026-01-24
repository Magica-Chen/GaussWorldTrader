"""Crypto strategies package.

CryptoMomentumStrategy is now consolidated into the unified MomentumStrategy.
Import here for backward compatibility.
"""

from src.strategy.stock.momentum import MomentumStrategy

# Re-export with crypto defaults for backward compatibility
# Usage: CryptoMomentumStrategy() creates a momentum strategy with asset_type="crypto"


def CryptoMomentumStrategy(params=None):
    """Create a MomentumStrategy configured for crypto trading."""
    merged = {"asset_type": "crypto", **(params or {})}
    return MomentumStrategy(merged)


__all__ = ["CryptoMomentumStrategy"]
