# 🎯 Wheel Strategy Dashboard Guide

## Quick Start

### Launch the Wheel Dashboard
```bash
# Launch wheel strategy dashboard
python dashboard.py --wheel
```

The dashboard will open at: **http://localhost:3721**

## Dashboard Overview

The Wheel Strategy Dashboard provides comprehensive tools for managing the wheel options strategy:

### 🎯 Overview Tab
- Strategy performance metrics
- Wheel cycle visualization
- Recent activity summary
- Key performance indicators

### 📊 Signals Tab
- Real-time signal generation
- Signal filtering and analysis
- Confidence scoring
- Auto-refresh capabilities

### 📈 Positions Tab
- Current option positions
- Assignment risk monitoring
- Profit/loss tracking
- Position management tools

### ⚙️ Settings Tab
- Strategy parameter configuration
- Risk management controls
- Advanced settings
- Configuration export/import

### 📚 Education Tab
- Wheel strategy explanation
- Parameter guides
- Example scenarios
- Risk considerations

## Configuration

### Default Configuration
The dashboard uses `config/wheel_config.json` for default parameters:

```json
{
  "max_risk": 80000,
  "position_size_pct": 0.08,
  "put_delta_min": 0.15,
  "put_delta_max": 0.30,
  "min_yield": 0.04,
  "dte_min": 7,
  "dte_max": 45
}
```

### Key Parameters Explained

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `max_risk` | Maximum total risk exposure ($) | $10,000 - $500,000 |
| `position_size_pct` | Position size as % of portfolio | 5% - 15% |
| `put_delta_min/max` | Assignment probability range | 0.15 - 0.35 |
| `min_yield` | Minimum acceptable yield | 3% - 8% |
| `dte_min/max` | Days to expiration range | 7 - 45 days |

## Workflow

### 1. 🎯 Configure Strategy
1. Open Settings tab
2. Adjust parameters for your risk tolerance
3. Update strategy configuration

### 2. 📊 Generate Signals
1. Go to Signals tab
2. Click "Generate Signals"
3. Review signal quality and confidence

### 3. 📈 Monitor Positions
1. Check Positions tab
2. Monitor assignment risk
3. Manage profit-taking opportunities

### 4. 🔄 Manage Cycle
1. Close profitable positions
2. Handle assignments
3. Roll threatened positions
4. Restart with new signals

## Command Line Alternatives

### CLI Interface
```bash
# Generate config template
python wheel_cli.py --generate-config

# Run strategy with CLI
python wheel_cli.py --run --verbose

# Show strategy information
python wheel_cli.py --info
```

### Example Scripts
```bash
# Test the strategy
python examples/wheel_strategy_example.py
```

## Integration with Main System

The wheel dashboard integrates with your existing GaussWorldTrader system:

- **Uses your watchlist**: Reads symbols from `watchlist.json`
- **Portfolio integration**: Works with existing portfolio system
- **Risk management**: Follows your configured risk limits
- **Data providers**: Uses existing Alpaca data integration

## Safety Features

### Built-in Risk Management
- ✅ Position size limits
- ✅ Maximum risk exposure
- ✅ Assignment probability monitoring
- ✅ Profit-taking automation
- ✅ Emergency position closure

### Educational Content
- ✅ Strategy explanation
- ✅ Parameter guidance
- ✅ Risk warnings
- ✅ Example scenarios

## Troubleshooting

### Common Issues

**Dashboard won't start:**
```bash
# Check if Streamlit is installed
pip install streamlit

# Verify file exists
ls src/ui/wheel_dashboard.py
```

**No signals generated:**
- Check your watchlist has symbols
- Verify strategy parameters aren't too restrictive
- Ensure market is open (for live data)

**Configuration issues:**
```bash
# Regenerate config
python wheel_cli.py --generate-config
```

## Next Steps

1. **Paper Trading**: Start with small position sizes
2. **Live Integration**: Connect to Alpaca for real option data
3. **Backtesting**: Test historical performance
4. **Customization**: Adjust parameters based on results

## Support

For issues or questions:
- Check the Education tab in the dashboard
- Review the wheel strategy documentation
- Test with the CLI tools first
- Start with paper trading

---

**⚠️ Important**: The wheel strategy involves significant risk. Always understand the strategy fully and start with small positions before scaling up.