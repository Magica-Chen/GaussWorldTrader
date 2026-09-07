// Registry snapshot. Refresh with: python examples/refresh_brand.py
window.GAUSS_STRATEGIES = [
  {
    "name": "btc_volatility_breakout",
    "label": "BTC Volatility Breakout",
    "description": "Long-only BTC breakout strategy using rolling highs/lows, EMA trend confirmation, and ATR-based volatility expansion filters.",
    "asset_type": "crypto",
    "source": "src/strategy/crypto/btc_volatility_breakout.py"
  },
  {
    "name": "crypto_momentum",
    "label": "Crypto Momentum",
    "description": "Dual momentum crossover strategy for crypto with risk management.",
    "asset_type": "crypto",
    "source": "src/strategy/stock/momentum.py"
  },
  {
    "name": "macro_factor",
    "label": "Macro Factor",
    "description": "FRED macro regime model with moving-average confirmation.",
    "asset_type": "stock",
    "source": "src/strategy/stock/macro_factor.py"
  },
  {
    "name": "mean_reversion",
    "label": "Mean Reversion",
    "description": "Bollinger Band reversions confirmed by RSI extremes.",
    "asset_type": "stock",
    "source": "src/strategy/stock/mean_reversion.py"
  },
  {
    "name": "momentum",
    "label": "Momentum",
    "description": "Dual momentum crossover strategy with stop-loss and take-profit.",
    "asset_type": "stock",
    "source": "src/strategy/stock/momentum.py"
  },
  {
    "name": "multi_agent",
    "label": "Multi-Agent",
    "description": "Committee-style strategy using technical, fundamental, and sentiment analysts.",
    "asset_type": "stock",
    "source": "src/strategy/multi_agent_strategy.py"
  },
  {
    "name": "scalping",
    "label": "Scalping",
    "description": "Short-term mean reversion around a short EMA.",
    "asset_type": "stock",
    "source": "src/strategy/stock/scalping.py"
  },
  {
    "name": "statistical_arbitrage",
    "label": "Statistical Arbitrage",
    "description": "Mean reversion using z-score of recent returns.",
    "asset_type": "stock",
    "source": "src/strategy/stock/statistical_arbitrage.py"
  },
  {
    "name": "trend_following",
    "label": "Trend Following",
    "description": "Trades on moving average crossovers.",
    "asset_type": "stock",
    "source": "src/strategy/stock/trend_following.py"
  },
  {
    "name": "value",
    "label": "Value",
    "description": "Looks for price discounts to a long-term average.",
    "asset_type": "stock",
    "source": "src/strategy/stock/value.py"
  },
  {
    "name": "vertical_spread",
    "label": "Vertical Spread",
    "description": "Vertical spread strategy using trend + RSI with IV/greeks filters.",
    "asset_type": "option",
    "source": "src/strategy/option/vertical_spread.py"
  },
  {
    "name": "wheel",
    "label": "Wheel",
    "description": "Options wheel strategy for income-focused trading.",
    "asset_type": "option",
    "source": "src/strategy/option/wheel.py"
  }
];
