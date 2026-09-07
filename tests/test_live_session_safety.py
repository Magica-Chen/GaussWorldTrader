"""Calendar and option exposure regressions; no SDK network calls."""
from datetime import datetime, timezone
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

from src.trade.live.live_trading_option import LiveTradingOption
from src.trade.live.live_trading_base import PositionState
from src.trade.engine.option_engine import TradingOptionEngine


def test_zero_net_spread_remains_exposure_and_filters_underlying():
    live = LiveTradingOption.__new__(LiveTradingOption)
    live.engine = Mock(spec=TradingOptionEngine)
    live.underlying_symbol = 'AAPL'
    live._lock = Lock()
    live.position = PositionState()
    live.engine.get_option_positions.return_value = [
        {'symbol': 'AAPL261218C00100000', 'underlying': 'AAPL', 'qty': 1, 'cost_basis': 200},
        {'symbol': 'AAPL261218C00110000', 'underlying': 'AAPL', 'qty': -1, 'cost_basis': -100},
        {'symbol': 'MSFT261218C00100000', 'underlying': 'MSFT', 'qty': 1, 'cost_basis': 400},
    ]
    live._refresh_position_state()
    assert live.position.side != 'flat'
    assert len(live.option_legs) == 2
    assert sum(p['qty'] for p in live.option_legs.values()) == 0


def test_underlying_price_never_triggers_option_premium_exit():
    live = LiveTradingOption.__new__(LiveTradingOption)
    live._lock = Lock()
    live.position = PositionState(qty=1, side='long', entry_price=2, stop_loss=1, take_profit=3)
    live.auto_exit = True
    live._last_monitor_log = 0
    live.logger = Mock()
    live._close_position = Mock()
    live._monitor_position(100)
    live._close_position.assert_not_called()
