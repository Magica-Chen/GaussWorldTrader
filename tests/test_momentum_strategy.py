"""Tests for momentum strategy and utilities.

Tests the dual momentum crossover strategy implementation including:
- Rate of Change (ROC) calculation
- Momentum crossover detection
- Signal generation with stop-loss and take-profit
"""
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from src.strategy.utils import rate_of_change, detect_momentum_crossover
from src.strategy.base import RiskConfig, StrategySignal
from src.strategy import get_strategy_registry
from src.strategy.stock.momentum import MomentumStrategy


# Sample price data (similar to BTC hourly closes)
SAMPLE_PRICES = [
    95407.06, 95200.40, 95204.74, 95328.24, 95678.55,
    95184.37, 94979.97, 94980.40, 95160.80, 94870.19,
    94993.31, 95063.22, 94741.40, 95021.00, 94961.67,
    96481.30, 96793.47, 97311.88, 96968.81, 96819.18,
    97248.89, 97667.79, 97572.82, 96886.75, 96944.55,
    97100.00, 97500.00, 98000.00,
]


class TestRateOfChange:
    """Tests for rate_of_change function."""

    def test_basic_roc_calculation(self):
        prices = [100, 102, 104, 106, 108]
        roc = rate_of_change(prices, 2)

        assert roc[0] is None
        assert roc[1] is None
        assert roc[2] == pytest.approx(0.04, rel=1e-3)  # (104-100)/100
        assert roc[3] == pytest.approx(0.0392, rel=1e-3)  # (106-102)/102
        assert roc[4] == pytest.approx(0.0385, rel=1e-3)  # (108-104)/104

    def test_roc_with_zero_price(self):
        prices = [0, 100, 200]
        roc = rate_of_change(prices, 1)

        assert roc[0] is None
        assert roc[1] is None  # Division by zero avoided
        assert roc[2] == pytest.approx(1.0, rel=1e-3)

    def test_roc_insufficient_data(self):
        prices = [100, 105]
        roc = rate_of_change(prices, 5)

        assert all(v is None for v in roc)

    def test_roc_with_sample_data(self):
        roc = rate_of_change(SAMPLE_PRICES, 12)

        # First 12 values should be None
        assert all(v is None for v in roc[:12])
        # Remaining should have values
        assert all(v is not None for v in roc[12:])


class TestMomentumCrossover:
    """Tests for detect_momentum_crossover function."""

    def test_bullish_crossover(self):
        # Short momentum crosses above long momentum
        short_mom = [None, None, -0.01, 0.005, 0.02]
        long_mom = [None, None, None, 0.01, 0.01]

        signal = detect_momentum_crossover(short_mom, long_mom, threshold=0.005)
        assert signal == "BUY"

    def test_bearish_crossover(self):
        # Short momentum crosses below long momentum
        short_mom = [None, None, 0.02, 0.01, -0.01]
        long_mom = [None, None, None, 0.005, 0.01]

        signal = detect_momentum_crossover(short_mom, long_mom, threshold=0.005)
        assert signal == "SELL"

    def test_no_crossover_hold(self):
        # No crossover detected
        short_mom = [None, None, 0.01, 0.015, 0.02]
        long_mom = [None, None, None, 0.01, 0.012]

        signal = detect_momentum_crossover(short_mom, long_mom, threshold=0.005)
        assert signal == "HOLD"

    def test_insufficient_data(self):
        short_mom = [0.01]
        long_mom = [0.02]

        signal = detect_momentum_crossover(short_mom, long_mom, threshold=0.005)
        assert signal == "HOLD"

    def test_none_values_hold(self):
        short_mom = [None, None, None]
        long_mom = [None, None, None]

        signal = detect_momentum_crossover(short_mom, long_mom, threshold=0.005)
        assert signal == "HOLD"


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_values(self):
        config = RiskConfig()

        assert config.stop_loss_pct == 0.03
        assert config.take_profit_pct == 0.06
        assert config.position_size_pct == 0.10

    def test_custom_values(self):
        config = RiskConfig(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            position_size_pct=0.15,
        )

        assert config.stop_loss_pct == 0.05
        assert config.take_profit_pct == 0.10
        assert config.position_size_pct == 0.15


class TestStrategySignal:
    """Tests for StrategySignal with stop_loss and take_profit."""

    def test_signal_with_risk_levels(self):
        signal = StrategySignal(
            symbol="AAPL",
            action="BUY",
            quantity=10,
            price=150.0,
            reason="momentum crossover",
            stop_loss=145.5,
            take_profit=159.0,
        )

        assert signal.stop_loss == 145.5
        assert signal.take_profit == 159.0

    def test_signal_to_dict(self):
        signal = StrategySignal(
            symbol="BTC/USD",
            action="BUY",
            quantity=1,
            price=97000.0,
            stop_loss=94090.0,
            take_profit=102820.0,
        )

        d = signal.to_dict()
        assert d["symbol"] == "BTC/USD"
        assert d["stop_loss"] == 94090.0
        assert d["take_profit"] == 102820.0


class TestMomentumStrategy:
    """Tests for MomentumStrategy class."""

    def test_strategy_initialization(self):
        strategy = MomentumStrategy()

        assert strategy.meta.name == "momentum"
        assert strategy.params["short_period"] == 12
        assert strategy.params["long_period"] == 26
        assert strategy.params["threshold"] == 0.005
        assert strategy.params["stop_loss_pct"] == 0.03
        assert strategy.params["take_profit_pct"] == 0.06

    def test_custom_params(self):
        strategy = MomentumStrategy({"short_period": 10, "long_period": 20})

        assert strategy.params["short_period"] == 10
        assert strategy.params["long_period"] == 20

    def test_risk_calculations(self):
        strategy = MomentumStrategy()
        price = 100.0

        stop_loss = strategy.calculate_stop_loss(price, "long")
        take_profit = strategy.calculate_take_profit(price, "long")

        assert stop_loss == pytest.approx(97.0, rel=1e-3)  # 3% below
        assert take_profit == pytest.approx(106.0, rel=1e-3)  # 6% above

    def test_short_side_risk(self):
        strategy = MomentumStrategy()
        price = 100.0

        stop_loss = strategy.calculate_stop_loss(price, "short")
        take_profit = strategy.calculate_take_profit(price, "short")

        assert stop_loss == pytest.approx(103.0, rel=1e-3)  # 3% above
        assert take_profit == pytest.approx(94.0, rel=1e-3)  # 6% below

    def test_generate_signals_insufficient_data(self):
        strategy = MomentumStrategy()
        current_date = datetime.now()

        # Only 10 bars - not enough for long_period=26
        short_data = pd.DataFrame({"close": SAMPLE_PRICES[:10]})
        historical_data = {"AAPL": short_data}
        current_prices = {"AAPL": 95000.0}

        signals = strategy.generate_signals(
            current_date, current_prices, {}, historical_data
        )

        assert signals == []

    def test_generate_signals_with_data(self):
        strategy = MomentumStrategy()
        current_date = datetime.now()

        # Create dataframe with enough data
        df = pd.DataFrame({"close": SAMPLE_PRICES})
        historical_data = {"AAPL": df}
        current_prices = {"AAPL": SAMPLE_PRICES[-1]}

        signals = strategy.generate_signals(
            current_date, current_prices, {}, historical_data
        )

        # Should return list (may be empty if no crossover)
        assert isinstance(signals, list)

        # If signals generated, verify structure
        for signal in signals:
            assert "symbol" in signal
            assert "action" in signal
            assert "stop_loss" in signal
            assert "take_profit" in signal


class TestCryptoMomentumStrategy:
    """Tests for the registry-backed crypto momentum strategy."""

    def test_crypto_strategy_defaults(self):
        strategy = get_strategy_registry().create("crypto_momentum")

        # Unified strategy with crypto asset type
        assert strategy._asset_type == "crypto"
        assert strategy.params["risk_pct"] == 0.10  # Higher for crypto
        assert strategy.params["qty_precision"] == 6  # Crypto-specific
        assert strategy.params["min_qty"] == 0.000001  # Crypto-specific

    def test_crypto_visible_in_dashboard(self):
        strategy = get_strategy_registry().create("crypto_momentum")

        assert strategy.meta.visible_in_dashboard is True

    def test_crypto_position_sizing(self):
        """Crypto should use decimal precision for position sizing."""
        strategy = get_strategy_registry().create("crypto_momentum")
        qty = strategy._position_size(50000.0, 100000.0, 0.10)

        assert isinstance(qty, float)
        assert qty == round(qty, 6)  # 6 decimal precision


class TestIntegration:
    """Integration tests for the full momentum strategy flow."""

    def test_full_signal_generation_flow(self):
        """Test complete flow from prices to signals with risk levels."""
        # Create price data that should trigger a signal
        # Prices trending up strongly to trigger bullish crossover
        prices = list(range(100, 130))  # 30 prices from 100 to 129

        strategy = MomentumStrategy({"short_period": 5, "long_period": 10})
        current_date = datetime.now()

        df = pd.DataFrame({"close": [float(p) for p in prices]})
        historical_data = {"TEST": df}
        current_prices = {"TEST": float(prices[-1])}

        signals = strategy.generate_signals(
            current_date, current_prices, {}, historical_data
        )

        # With steadily rising prices, we might get a signal
        # The actual signal depends on crossover timing
        assert isinstance(signals, list)

    def test_portfolio_integration(self):
        """Test signal generation with mock portfolio."""
        strategy = MomentumStrategy()
        current_date = datetime.now()

        # Mock portfolio
        portfolio = MagicMock()
        portfolio.get_portfolio_value.return_value = 100000.0

        df = pd.DataFrame({"close": SAMPLE_PRICES})
        historical_data = {"AAPL": df}
        current_prices = {"AAPL": SAMPLE_PRICES[-1]}

        signals = strategy.generate_signals(
            current_date, current_prices, {}, historical_data, portfolio
        )

        # Verify portfolio was called
        if signals:
            portfolio.get_portfolio_value.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
