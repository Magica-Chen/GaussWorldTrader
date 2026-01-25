# Implementation Plan: Strategy/Execution Separation Refactor

## Design Decisions (Confirmed)
- **Backward compatibility**: `generate_signals()` can change (will become wrapper)
- **Option strategies**: Use the new interface, but keep existing logic (wrap old flow)
- **Order type (user-selectable)**: User can choose order type. Default behavior:
  - Strategy provides `target_price` → use **limit** order at `target_price + min_increment`
  - Strategy provides no price → use **market** order
- **Sell-to-open gating**: Default is **disabled**. Only allow sell-to-open if user enables it
  **and** account supports margin/shorting; otherwise skip with a clear warning.
- **CLI framework**: Use **Typer**, not Click.
- **Account capability check**: Determine margin/shorting/fractional using account info
  (e.g., `account_type`, buying power, daytrading buying power) plus account config flags.

---

## Implementation Steps

### Step 1: Strategy Template (`src/strategy/base.py`)

**Add new dataclasses:**
```python
@dataclass
class SignalSnapshot:
    """Pure signal without quantity - what the strategy sees"""
    symbol: str
    indicators: Dict[str, float]  # e.g., {"short_mom": 0.05, "long_mom": 0.02}
    signal_strength: float  # -1.0 to 1.0
    timestamp: datetime

@dataclass
class ActionPlan:
    """Abstract action recommendation"""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    target_price: Optional[float]  # If set, suggests limit order
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str
    strength: float  # 0.0 to 1.0 (confidence)
    timestamp: datetime
```

**Add new methods to StrategyBase:**
```python
def get_signal(self, symbol, current_data, historical_data) -> SignalSnapshot:
    """Compute indicators and signal state - no quantity"""

def get_action_plan(self, signal: SignalSnapshot, current_price: float) -> ActionPlan:
    """Translate signal into abstract action with optional target price"""

def generate_signals(...) -> List[Dict]:
    """Legacy wrapper - calls get_signal + get_action_plan + sizing"""
```

**Files:** `src/strategy/base.py`

---

### Step 2: Execution Layer (`src/trade/execution.py`)

**Create new module with:**

```python
@dataclass
class ExecutionContext:
    """Read-only account state for execution decisions"""
    buying_power: float
    cash: float
    portfolio_value: float
    margin_enabled: bool
    fractional_enabled: bool
    shorting_enabled: bool
    current_positions: Dict[str, PositionState]

@dataclass
class ExecutionDecision:
    """Final order to place"""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str  # "market" or "limit"
    limit_price: Optional[float]
    reason: str

class ExecutionEngine:
    def __init__(self, account_manager, account_config):
        ...

    def load_context(self) -> ExecutionContext:
        """Load current account state"""

    def check_permissions(self, action_plan: ActionPlan, context: ExecutionContext) -> Tuple[bool, str]:
        """Check if action is allowed (e.g., sell-to-open needs margin)"""

    def size_order(self, action_plan: ActionPlan, context: ExecutionContext,
                   risk_config: dict, override_qty: Optional[float] = None) -> float:
        """Calculate quantity based on risk config and account state"""

    def build_decision(self, action_plan: ActionPlan, context: ExecutionContext,
                       risk_config: dict, override_qty: Optional[float] = None,
                       order_pref: Optional[str] = None) -> ExecutionDecision:
        """Build complete execution decision"""
```

**Order type logic:**
- If user specifies an order type, respect it.
- Otherwise:
  - If `action_plan.target_price` is set → limit order at `target_price + min_increment`
  - If `action_plan.target_price` is None → market order

**Files:** `src/trade/execution.py` (new)

---

### Step 3: Update Live Trading Base (`src/trade/live_trading_base.py`)

**Modify `_run_signal_cycle()` flow:**

```python
# Old flow:
signal = strategy.generate_signals(...)
if signal.action == "BUY":
    engine.place_market_order(symbol, signal.quantity, "buy")

# New flow:
signal_snapshot = strategy.get_signal(symbol, current_data, historical_data)
action_plan = strategy.get_action_plan(signal_snapshot, current_price)
if action_plan.action == "HOLD":
    return

allowed, reason = execution_engine.check_permissions(action_plan, context)
if not allowed:
    log.warning(f"Action blocked: {reason}")
    return

decision = execution_engine.build_decision(
    action_plan, context, risk_config, override_qty, order_pref
)
if decision.order_type == "limit":
    engine.place_limit_order(symbol, decision.quantity, decision.limit_price, decision.side)
else:
    engine.place_market_order(symbol, decision.quantity, decision.side)
```

**Files:** `src/trade/live_trading_base.py`, `src/trade/live_trading_stock.py`, `src/trade/live_trading_crypto.py`

---

### Step 4: Account Pre-checks in `live_script.py`

**Add startup validation:**

```python
def display_account_capabilities():
    """Show resolved account state before trading starts"""
    account_info = account_manager.get_account()
    account_config = config_manager.get_account_configurations()

    print("Account Capabilities:")
    print(f"  Buying Power: ${account_info['buying_power']:,.2f}")
    print(f"  Margin Enabled: {account_info.get('account_type') == 'MARGIN'}")
    print(f"  Fractional Trading: {account_config.get('fractional_trading', False)}")
    print(f"  Extended Hours: {account_config.trading_hours == 'EXTENDED'}")

def reconcile_user_config(user_wants, account_has):
    """Resolve conflicts between user config and account capabilities"""
    # If user wants fractional but account doesn't support → disable + warn
    # If user wants sell-to-open but no margin → disable + warn
```

**Files:** `live_script.py`

---

### Step 5: Migrate Stock Strategies

**Target strategies:**
- `src/strategy/stock/momentum.py`
- `src/strategy/stock/trend_following.py`
- `src/strategy/stock/value.py`
- `src/strategy/stock/scalping.py`
- `src/strategy/stock/statistical_arbitrage.py`

**Changes per strategy:**

1. Extract indicator calculation into `get_signal()`:
```python
def get_signal(self, symbol, current_data, historical_data) -> SignalSnapshot:
    prices = historical_data[symbol]["close"]
    short_mom = rate_of_change(prices, self.params["short_period"])
    long_mom = rate_of_change(prices, self.params["long_period"])
    signal_strength = (short_mom - long_mom) / abs(long_mom) if long_mom else 0

    return SignalSnapshot(
        symbol=symbol,
        indicators={"short_mom": short_mom, "long_mom": long_mom},
        signal_strength=signal_strength,
        timestamp=current_date
    )
```

2. Implement `get_action_plan()`:
```python
def get_action_plan(self, signal: SignalSnapshot, current_price: float) -> ActionPlan:
    if signal.signal_strength > self.threshold:
        action = "BUY"
        target_price = current_price  # or SMA level for limit order
    elif signal.signal_strength < -self.threshold:
        action = "SELL"
        target_price = current_price
    else:
        action = "HOLD"
        target_price = None

    return ActionPlan(
        symbol=signal.symbol,
        action=action,
        target_price=target_price,
        stop_loss=self.calculate_stop_loss(current_price, action),
        take_profit=self.calculate_take_profit(current_price, action),
        reason=f"Momentum signal: {signal.signal_strength:.2f}",
        strength=abs(signal.signal_strength),
        timestamp=signal.timestamp
    )
```

3. Update `generate_signals()` to be wrapper:
```python
def generate_signals(self, ...) -> List[Dict]:
    """Legacy interface - wraps new methods"""
    signals = []
    for symbol in historical_data.keys():
        snapshot = self.get_signal(symbol, current_data, historical_data)
        action_plan = self.get_action_plan(snapshot, current_prices[symbol])
        if action_plan.action != "HOLD":
            # Apply sizing for backward compat
            qty = self._position_size(current_prices[symbol], portfolio_value, risk_pct)
            signals.append(self._to_legacy_signal(action_plan, qty))
    return signals
```

---

### Step 6: CLI Updates (`main_cli.py`)

**Add per-symbol quantity override (Typer):**
```python
def run_strategy(
    ...,
    quantity: List[str] = typer.Option(
        None, "--quantity", "-q", help="Per-symbol quantity: AAPL=10"
    ),
):
    qty_overrides = parse_quantity_overrides(quantity or [])
    # {"AAPL": 10, "MSFT": 5}
```

**Update execution path:**
- Replace direct `strategy.generate_signals()` + `place_market_order()`
- Use action plan + execution engine flow

---

### Step 7: Backtest Compatibility

**Update `Backtester` to use new flow:**
- Create `PaperExecutionEngine` that mirrors sizing logic
- Accept action plans instead of quantity-included signals
- Apply same order type logic (limit vs market simulation)

**Files:** `src/trade/backtester.py`, `src/ui/trading_views.py`

---

## Files Summary

| File | Action |
|------|--------|
| `src/strategy/base.py` | Add SignalSnapshot, ActionPlan, new methods |
| `src/trade/execution.py` | **New** - ExecutionContext, ExecutionEngine |
| `src/trade/live_trading_base.py` | Use new execution flow |
| `src/trade/live_trading_stock.py` | Asset-specific execution |
| `src/trade/live_trading_crypto.py` | Asset-specific execution |
| `src/strategy/stock/momentum.py` | Implement new interface |
| `src/strategy/stock/trend_following.py` | Implement new interface |
| `src/strategy/stock/value.py` | Implement new interface |
| `src/strategy/stock/scalping.py` | Implement new interface |
| `src/strategy/stock/statistical_arbitrage.py` | Implement new interface |
| `live_script.py` | Account pre-checks, display |
| `main_cli.py` | Per-symbol quantity overrides |
| `src/trade/backtester.py` | Accept action plans |

---

## Verification Plan

1. **Unit tests:**
   - `get_signal()` returns correct indicators
   - `get_action_plan()` returns correct action for signal strength
   - `ExecutionEngine.check_permissions()` blocks sell-to-open without margin
   - `ExecutionEngine.build_decision()` chooses correct order type

2. **Integration tests:**
   - `python main_cli.py list-strategies` - Registry works
   - `python main_cli.py run-strategy momentum AAPL --execute false` - Dry run
   - Full flow: signal → action plan → execution decision

3. **Manual tests:**
   - `python live_script.py` - Account capabilities displayed
   - Limit order generated when strategy provides target_price
   - Sell-to-open blocked when margin not enabled

---

## Out of Scope
- Crypto-specific execution changes (beyond base flow update)
- UI dashboard signal display changes
