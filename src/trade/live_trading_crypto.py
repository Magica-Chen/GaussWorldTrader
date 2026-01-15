"""Live crypto trading with 24/7 streaming."""
from __future__ import annotations

from typing import Any

from src.strategy.crypto.momentum import CryptoMomentumStrategy

from .crypto_engine import TradingCryptoEngine
from .live_trading_base import LiveTradingEngine


class LiveTradingCrypto(LiveTradingEngine):
    """Live trading engine for cryptocurrency.

    Features:
    - 24/7 trading availability
    - Signal cycles based on timeframe
    - Real-time trade streaming
    - No short selling support
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        lookback_days: int = 14,
        crypto_loc: str = "us",
        risk_pct: float = 0.10,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        execute: bool = True,
        auto_exit: bool = True,
    ) -> None:
        self.crypto_loc = crypto_loc
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execute=execute,
            auto_exit=auto_exit,
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol to Alpaca format (BTC/USD)."""
        symbol = symbol.strip().upper()
        if "/" in symbol:
            return symbol
        if symbol.endswith("USD") and len(symbol) > 3:
            return f"{symbol[:-3]}/USD"
        return symbol

    def _get_trading_engine(self) -> TradingCryptoEngine:
        """Return crypto trading engine."""
        return TradingCryptoEngine()

    def _get_strategy(self) -> CryptoMomentumStrategy:
        """Return crypto momentum strategy."""
        return CryptoMomentumStrategy()

    def _create_stream(self) -> Any:
        """Create crypto data stream."""
        return self.provider.create_crypto_stream(raw_data=False, loc=self.crypto_loc)

    def _subscribe_to_stream(self, handler: Any) -> None:
        """Subscribe to crypto trade stream."""
        self._stream.subscribe_trades(handler, self.symbol)

    def _get_signal_interval_seconds(self) -> float:
        """Crypto trades 24/7, check at timeframe intervals."""
        return self._seconds_until_next_interval()
