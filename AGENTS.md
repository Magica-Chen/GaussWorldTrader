# Repository Guidelines

## Project Structure & Module Organization
- Entry points: `main_cli.py` (CLI), `dashboard.py` (Streamlit UI), and live runners `crypto_script.py`, `stock_script.py`, `option_script.py`.
- Core code lives in `src/` with `strategy/` (base + registry + per-asset submodules), `trade/` (engines, live trading), `data/` (providers), `account/`, `analysis/`, `ui/`, and `utils/`.
- Strategy layout: `src/strategy/base.py` defines `StrategyBase`, `BaseCryptoStrategy`, `BaseOptionStrategy` and trading-plan helpers; one strategy per file under `src/strategy/stock/`, `src/strategy/crypto/`, `src/strategy/option/`. Register new strategies in `src/strategy/registry.py`.
- Trade engines: `src/trade/trading_engine.py` (shared), asset-specific engines in `src/trade/stock_engine.py`, `src/trade/crypto_engine.py`, `src/trade/option_engine.py`, plus live trading in `src/trade/live_trading_*.py`.
- Configuration is in `config/`, `.env`, and `watchlist.json` (entries include `asset_type`, e.g., `{\"symbol\":\"AAPL\",\"asset_type\":\"stock\"}`). Examples live in `examples/`. Docs live in `docs/`.
- Asset helpers live in `src/utils/asset_utils.py` and live-script helpers in `src/utils/live_utils.py`.
- `tests/` exists locally but is gitignored; keep tests local unless asked to reintroduce them to git.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime dependencies.
- `python main_cli.py list-strategies` verifies CLI wiring.
- `python main_cli.py run-strategy --strategy momentum AAPL` runs a strategy on recent data.
- `python main_cli.py stream-market --asset-type crypto --crypto-loc eu-1 --symbols BTC/USD,ETH/USD` streams live data.
- `python dashboard.py` launches the Streamlit dashboard (local web UI).
- `python crypto_script.py --symbols BTC/USD,ETH/USD --timeframe 5Min --no-execute` runs live crypto in dry-run mode.
- `python stock_script.py --symbols AAPL,MSFT --timeframe 15Min --no-execute` runs live stock trading in dry-run mode.
- `python option_script.py --symbols AAPL,MSFT --timeframe 1Day --no-execute` runs live options trading in dry-run mode.
- If no symbols are provided, live scripts use `watchlist.json` + current positions filtered by asset type.
- Multi-symbol live runs use a shared websocket per asset type; for crypto, all symbols must share the same `--crypto-loc`.
- Stock and option scripts exit immediately when the market is closed and log time-to-open.
- Optional tooling: `black .`, `ruff check .`, `mypy src` (configured in `pyproject.toml`).

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, line length 100 (Black/Ruff).
- Prefer concise, high-cohesion modules; avoid duplicating functionality across strategies.
- Rewrite existing components instead of adding parallel versions; avoid fallback paths.
- Strategy classes must define `meta` and `summary`, and implement `generate_signals` (use `generate_trading_plan` helpers from `StrategyBase`).

## Testing Guidelines
- Framework: pytest (see `pyproject.toml`).
- Naming: `test_*.py` or `*_test.py`, with functions `test_*`.
- Keep tests lightweight; add coverage for new behaviors where feasible. Note: `tests/` is gitignored.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and lowercase (e.g., "re-structure the repo").
- Keep commit subjects under 100 characters.
- PRs should describe changes, link issues if applicable, and include screenshots for UI changes.

## Security & Configuration Tips
- Do not commit real API keys. Use `.env` based on `.env.example`.
- Ensure Alpaca/Finnhub/FRED keys are present before running live data features.
