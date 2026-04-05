"""Backtesting utilities for CLI and shared consumers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.trade.portfolio import Portfolio

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import vectorbt as vbt

    from src.strategy.base import StrategyBase


class Backtester:
    """Backtest strategies using vectorbt for stock/crypto and legacy logic for options."""

    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.01):
        self.initial_cash = initial_cash
        self.commission = commission
        self.portfolio = Portfolio(initial_cash)
        self.data: dict[str, pd.DataFrame] = {}
        self.results: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        self.data[symbol] = self._normalized_frame(data)
        self.logger.info("Added data for %s: %s rows", symbol, len(data))

    def run_backtest(
        self,
        strategy_func: Callable[..., list[dict[str, Any]]],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
        benchmark_symbol: str | None = None,
        strategy: StrategyBase | None = None,
    ) -> dict[str, Any]:
        self.portfolio = Portfolio(self.initial_cash)
        symbols_list = self._resolve_symbols(symbols)
        self._reset_strategy(strategy)

        if self._should_use_vectorbt(strategy):
            results = self._run_vectorbt_backtest(
                strategy=strategy,
                symbols=symbols_list,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            date_range = self._build_date_range(symbols_list, start_date, end_date)
            history = self._run_event_loop(strategy_func, symbols_list, date_range)
            results = self._calculate_performance_metrics(**history)

        if benchmark_symbol is not None:
            date_range = self._build_date_range(symbols_list, start_date, end_date)
            results["benchmark"] = self._benchmark_metrics(
                benchmark_symbol,
                date_range[0],
                date_range[-1],
            )

        results["summary"] = self._summary_dict(results)
        self.results = results
        self.logger.info("Backtest completed successfully")
        return results

    def run_walk_forward(
        self,
        strategy_func: Callable[..., list[dict[str, Any]]],
        splits: int = 5,
        symbols: list[str] | None = None,
        benchmark_symbol: str | None = None,
        strategy: StrategyBase | None = None,
    ) -> dict[str, Any]:
        if splits < 2:
            raise ValueError("splits must be at least 2")

        symbols_list = self._resolve_symbols(symbols)
        date_range = self._build_date_range(symbols_list, None, None)
        windows = self._walk_forward_windows(date_range, splits)
        split_summaries: list[dict[str, Any]] = []

        for index, (start_date, end_date) in enumerate(windows, start=1):
            runner = Backtester(
                initial_cash=self.initial_cash,
                commission=self.commission,
            )
            for symbol, data in self.data.items():
                runner.add_data(symbol, data)

            result = runner.run_backtest(
                strategy_func,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols_list,
                benchmark_symbol=benchmark_symbol,
                strategy=strategy,
            )
            split_summary = {
                "split": index,
                **result["summary"],
            }
            if "benchmark" in result:
                split_summary["benchmark_return_percentage"] = result["benchmark"][
                    "total_return_percentage"
                ]
            split_summaries.append(split_summary)

        results = {
            "split_summaries": split_summaries,
            "summary": self._walk_forward_summary(split_summaries),
        }
        self.results = results
        self.logger.info("Walk-forward backtest completed successfully")
        return results

    def _should_use_vectorbt(self, strategy: StrategyBase | None) -> bool:
        if strategy is None:
            return False
        return strategy.meta.asset_type in {"stock", "crypto"}

    def _run_vectorbt_backtest(
        self,
        strategy: StrategyBase,
        symbols: list[str],
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> dict[str, Any]:
        date_range = self._build_date_range(symbols, start_date, end_date)
        close = self._price_matrix(symbols, date_range, "close")
        signal_data = self._build_vectorbt_signal_data(strategy, symbols, date_range)
        vbt = self._import_vectorbt()
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=signal_data["entries"],
            exits=signal_data["exits"],
            size=signal_data["size"],
            size_type="percent",
            open=self._price_matrix(symbols, date_range, "open"),
            high=self._price_matrix(symbols, date_range, "high"),
            low=self._price_matrix(symbols, date_range, "low"),
            sl_stop=signal_data["sl_stop"],
            tp_stop=signal_data["tp_stop"],
            fees=self.commission,
            init_cash=self.initial_cash,
            cash_sharing=True,
            size_granularity=self._size_granularity(strategy),
        )
        return self._vectorbt_results(portfolio)

    def _build_vectorbt_signal_data(
        self,
        strategy: StrategyBase,
        symbols: list[str],
        date_range: list[datetime],
    ) -> dict[str, pd.DataFrame]:
        index = pd.DatetimeIndex(date_range)
        entries = pd.DataFrame(False, index=index, columns=symbols)
        exits = pd.DataFrame(False, index=index, columns=symbols)
        size = pd.DataFrame(0.0, index=index, columns=symbols)
        sl_stop = pd.DataFrame(np.nan, index=index, columns=symbols)
        tp_stop = pd.DataFrame(np.nan, index=index, columns=symbols)
        risk_pct = float(strategy.params.get("risk_pct", 0.0))

        for symbol in symbols:
            symbol_data = self.data[symbol].reindex(index).dropna(subset=["close"])
            self._populate_symbol_signals(
                strategy=strategy,
                symbol=symbol,
                symbol_data=symbol_data,
                entries=entries,
                exits=exits,
                size=size,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                risk_pct=risk_pct,
            )

        return {
            "entries": entries,
            "exits": exits,
            "size": size,
            "sl_stop": sl_stop,
            "tp_stop": tp_stop,
        }

    def _populate_symbol_signals(
        self,
        strategy: StrategyBase,
        symbol: str,
        symbol_data: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        size: pd.DataFrame,
        sl_stop: pd.DataFrame,
        tp_stop: pd.DataFrame,
        risk_pct: float,
    ) -> None:
        for current_date, row in symbol_data.iterrows():
            historical_data = symbol_data.loc[:current_date]
            current_price = float(row["close"])
            snapshot = strategy.get_signal(
                symbol=symbol,
                current_date=current_date.to_pydatetime(),
                current_price=current_price,
                current_data=row.to_dict(),
                historical_data=historical_data,
                portfolio=None,
            )
            if snapshot is None:
                continue

            plan = strategy.get_action_plan(
                snapshot,
                current_price,
                current_date.to_pydatetime(),
            )
            if plan is None or plan.action == "HOLD":
                continue

            action = plan.action.upper()
            if action == "BUY":
                entries.at[current_date, symbol] = True
                size.at[current_date, symbol] = risk_pct
                sl_stop.at[current_date, symbol] = self._stop_pct(
                    entry_price=current_price,
                    exit_price=plan.stop_loss,
                    side="loss",
                )
                tp_stop.at[current_date, symbol] = self._stop_pct(
                    entry_price=current_price,
                    exit_price=plan.take_profit,
                    side="profit",
                )
                continue

            if action == "SELL":
                exits.at[current_date, symbol] = True
                continue

            raise ValueError(f"vectorbt backtests do not support action {plan.action}")

    def _vectorbt_results(self, portfolio: vbt.Portfolio) -> dict[str, Any]:
        portfolio_value = portfolio.value().rename("portfolio_value")
        cash = portfolio.cash().rename("cash")
        positions_value = portfolio.asset_value().rename("positions_value")
        portfolio_history = pd.concat(
            [portfolio_value, cash, positions_value],
            axis=1,
        )
        portfolio_history.index.name = "date"

        daily_returns = portfolio_value.pct_change().dropna()
        drawdowns = self._drawdown_series(portfolio_value)
        closed_trades = portfolio.trades.records_readable.copy()
        trades_history = self._vectorbt_order_history(
            portfolio,
            portfolio_value,
        )

        final_value = float(portfolio_value.iloc[-1])
        total_return = 0.0 if self.initial_cash <= 0 else (
            (final_value - self.initial_cash) / self.initial_cash
        )
        trading_days = len(portfolio_history)
        years = trading_days / 252 if trading_days > 0 else 0.0
        annualized_return = 0.0
        if years > 0 and self.initial_cash > 0:
            annualized_return = (final_value / self.initial_cash) ** (1 / years) - 1

        volatility = float(daily_returns.std() * np.sqrt(252)) if not daily_returns.empty else 0.0
        sharpe_ratio = 0.0
        if volatility > 0:
            sharpe_ratio = float((annualized_return - 0.02) / volatility)

        realized_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        if not closed_trades.empty:
            pnl = closed_trades["PnL"].astype(float)
            realized_pnl = float(pnl.sum())
            winning_trades = int((pnl > 0).sum())
            losing_trades = int((pnl < 0).sum())

        total_trades = winning_trades + losing_trades
        win_rate = 0.0 if total_trades == 0 else (winning_trades / total_trades) * 100
        profit_factor = self._profit_factor_from_trades(closed_trades)

        return {
            "initial_value": float(self.initial_cash),
            "final_value": final_value,
            "total_return": float(total_return),
            "total_return_percentage": float(total_return * 100),
            "annualized_return": float(annualized_return),
            "annualized_return_percentage": float(annualized_return * 100),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(drawdowns.max()) if not drawdowns.empty else 0.0,
            "max_drawdown_percentage": (
                float(drawdowns.max()) * 100 if not drawdowns.empty else 0.0
            ),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate),
            "total_profit": max(realized_pnl, 0.0),
            "total_loss": abs(min(realized_pnl, 0.0)),
            "profit_factor": float(profit_factor),
            "trading_days": trading_days,
            "start_date": portfolio_history.index[0],
            "end_date": portfolio_history.index[-1],
            "portfolio_history": portfolio_history,
            "trades_history": trades_history,
            "daily_returns": daily_returns.tolist(),
            "drawdowns": drawdowns.tolist(),
        }

    def _vectorbt_order_history(
        self,
        portfolio: vbt.Portfolio,
        portfolio_value: pd.Series,
    ) -> pd.DataFrame:
        orders = portfolio.orders.records_readable.copy()
        if orders.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "action",
                    "quantity",
                    "price",
                    "fees",
                    "portfolio_value",
                ]
            )

        orders["date"] = pd.to_datetime(orders["Timestamp"])
        orders["symbol"] = orders["Column"].astype(str)
        orders["action"] = orders["Side"].str.upper()
        orders["quantity"] = orders["Size"].astype(float)
        orders["price"] = orders["Price"].astype(float)
        orders["fees"] = orders["Fees"].astype(float)
        orders["portfolio_value"] = orders["date"].map(portfolio_value)
        history = orders[
            [
                "date",
                "symbol",
                "action",
                "quantity",
                "price",
                "fees",
                "portfolio_value",
            ]
        ]
        return history.set_index("date")

    def _size_granularity(self, strategy: StrategyBase) -> float | None:
        if strategy.meta.asset_type == "stock":
            return 1.0
        min_qty = strategy.params.get("min_qty")
        if min_qty is None:
            return None
        return float(min_qty)

    def _price_matrix(
        self,
        symbols: list[str],
        date_range: list[datetime],
        column: str,
    ) -> pd.DataFrame:
        index = pd.DatetimeIndex(date_range)
        frames = []
        for symbol in symbols:
            series = self.data[symbol][column].reindex(index)
            frames.append(series.rename(symbol))
        return pd.concat(frames, axis=1)

    def _stop_pct(
        self,
        entry_price: float,
        exit_price: float | None,
        side: str,
    ) -> float:
        if exit_price is None or entry_price <= 0:
            return np.nan
        if side == "loss":
            stop_pct = (entry_price - exit_price) / entry_price
        else:
            stop_pct = (exit_price - entry_price) / entry_price
        if stop_pct <= 0:
            return np.nan
        return float(stop_pct)

    def _import_vectorbt(self) -> Any:
        cache_dir = os.getenv("NUMBA_CACHE_DIR")
        if not cache_dir:
            cache_dir = "/tmp/gaussworldtrader-numba-cache"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ["NUMBA_CACHE_DIR"] = cache_dir

        import vectorbt as vbt

        return vbt

    def _reset_strategy(self, strategy: StrategyBase | None) -> None:
        if strategy is None:
            return
        reset_state = getattr(strategy, "reset_strategy_state", None)
        if callable(reset_state):
            reset_state()

    def _resolve_symbols(self, symbols: list[str] | None) -> list[str]:
        if not self.data:
            raise ValueError("No data loaded for backtesting")
        if symbols is None:
            return sorted(self.data.keys())
        result = [symbol for symbol in symbols if symbol in self.data]
        if not result:
            raise ValueError("No valid data found for specified symbols")
        return result

    def _build_date_range(
        self,
        symbols: list[str],
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[datetime]:
        all_dates: set[datetime] = set()
        for symbol in symbols:
            all_dates.update(self.data[symbol].index)

        if not all_dates:
            raise ValueError("No valid data found for specified symbols")

        date_range = sorted(all_dates)
        if start_date is not None:
            start_date = self._normalize_datetime(start_date)
            date_range = [date for date in date_range if date >= start_date]
        if end_date is not None:
            end_date = self._normalize_datetime(end_date)
            date_range = [date for date in date_range if date <= end_date]
        if not date_range:
            raise ValueError("No data in specified date range")
        return date_range

    def _run_event_loop(
        self,
        strategy_func: Callable[..., list[dict[str, Any]]],
        symbols: list[str],
        date_range: list[datetime],
    ) -> dict[str, Any]:
        portfolio_values: list[dict[str, Any]] = []
        daily_returns: list[float] = []
        trades_log: list[dict[str, Any]] = []
        self.logger.info(
            "Starting backtest from %s to %s",
            date_range[0],
            date_range[-1],
        )

        for current_date in date_range:
            current_prices, current_data = self._current_market_snapshot(
                symbols,
                current_date,
            )
            if not current_prices:
                continue

            historical_data = self._historical_snapshot(symbols, current_date)
            signals = strategy_func(
                current_date=current_date,
                current_prices=current_prices,
                current_data=current_data,
                historical_data=historical_data,
                portfolio=self.portfolio,
            )
            self._apply_signals(signals, current_prices, current_date, trades_log)
            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            portfolio_values.append(
                {
                    "date": current_date,
                    "portfolio_value": portfolio_value,
                    "cash": self.portfolio.cash,
                    "positions_value": portfolio_value - self.portfolio.cash,
                }
            )
            if len(portfolio_values) > 1:
                previous_value = portfolio_values[-2]["portfolio_value"]
                daily_return = 0.0
                if previous_value > 0:
                    daily_return = (portfolio_value - previous_value) / previous_value
                daily_returns.append(daily_return)

        return {
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "trades_log": trades_log,
        }

    def _current_market_snapshot(
        self,
        symbols: list[str],
        current_date: datetime,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        current_prices: dict[str, float] = {}
        current_data: dict[str, Any] = {}
        for symbol in symbols:
            df = self.data[symbol]
            try:
                row = df.loc[current_date]
            except KeyError:
                continue
            current_prices[symbol] = float(row["close"])
            current_data[symbol] = row.to_dict()
        return current_prices, current_data

    def _historical_snapshot(
        self,
        symbols: list[str],
        current_date: datetime,
    ) -> dict[str, pd.DataFrame]:
        return {
            symbol: self.data[symbol][self.data[symbol].index <= current_date]
            for symbol in symbols
        }

    def _apply_signals(
        self,
        signals: list[dict[str, Any]],
        current_prices: dict[str, float],
        current_date: datetime,
        trades_log: list[dict[str, Any]],
    ) -> None:
        for signal in signals:
            if not self._execute_signal(signal, current_prices, current_date):
                continue
            trades_log.append(
                {
                    "date": current_date,
                    "symbol": signal["symbol"],
                    "action": signal["action"],
                    "quantity": signal["quantity"],
                    "price": current_prices.get(signal["symbol"], 0.0),
                    "portfolio_value": self.portfolio.get_portfolio_value(
                        current_prices
                    ),
                }
            )

    def _execute_signal(
        self,
        signal: dict[str, Any],
        current_prices: dict[str, float],
        current_date: datetime,
    ) -> bool:
        symbol = signal["symbol"]
        action = signal["action"].upper()
        quantity = float(signal["quantity"])

        if symbol not in current_prices:
            return False

        price = float(current_prices[symbol])
        commission_cost = abs(quantity) * price * self.commission

        if action == "BUY":
            total_cost = quantity * price + commission_cost
            if total_cost > self.portfolio.cash:
                return False
            self.portfolio.add_position(symbol, quantity, price, current_date)
            self.portfolio.cash -= commission_cost
            return True

        if action == "SELL" and symbol in self.portfolio.positions:
            available_qty = float(self.portfolio.positions[symbol]["quantity"])
            if quantity > available_qty:
                return False
            self.portfolio.remove_position(symbol, quantity, price, current_date)
            self.portfolio.cash -= commission_cost
            return True

        return False

    def _calculate_performance_metrics(
        self,
        portfolio_values: list[dict[str, Any]],
        daily_returns: list[float],
        trades_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not portfolio_values:
            return {}

        df_portfolio = pd.DataFrame(portfolio_values).set_index("date")
        df_trades = pd.DataFrame(trades_log)
        final_value = float(df_portfolio["portfolio_value"].iloc[-1])
        total_return = 0.0 if self.initial_cash <= 0 else (
            (final_value - self.initial_cash) / self.initial_cash
        )
        trading_days = len(df_portfolio)
        years = trading_days / 252 if trading_days > 0 else 0.0
        annualized_return = 0.0
        if years > 0 and self.initial_cash > 0:
            annualized_return = (final_value / self.initial_cash) ** (1 / years) - 1

        drawdowns = self._drawdown_series(df_portfolio["portfolio_value"])
        volatility = float(np.std(daily_returns) * np.sqrt(252)) if daily_returns else 0.0
        sharpe_ratio = 0.0
        if volatility > 0:
            sharpe_ratio = float((annualized_return - 0.02) / volatility)

        realized_pnl = float(self.portfolio.get_realized_pnl())
        winning_trades, losing_trades = self._trade_outcomes(df_trades)
        total_trades = winning_trades + losing_trades
        win_rate = 0.0 if total_trades == 0 else (winning_trades / total_trades) * 100

        return {
            "initial_value": float(self.initial_cash),
            "final_value": final_value,
            "total_return": float(total_return),
            "total_return_percentage": float(total_return * 100),
            "annualized_return": float(annualized_return),
            "annualized_return_percentage": float(annualized_return * 100),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(drawdowns.max()) if not drawdowns.empty else 0.0,
            "max_drawdown_percentage": (
                float(drawdowns.max()) * 100 if not drawdowns.empty else 0.0
            ),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate),
            "total_profit": max(realized_pnl, 0.0),
            "total_loss": abs(min(realized_pnl, 0.0)),
            "profit_factor": self._profit_factor(realized_pnl),
            "trading_days": trading_days,
            "start_date": df_portfolio.index[0],
            "end_date": df_portfolio.index[-1],
            "portfolio_history": df_portfolio,
            "trades_history": df_trades,
            "daily_returns": daily_returns,
            "drawdowns": drawdowns.tolist(),
        }

    def _trade_outcomes(self, trades_history: pd.DataFrame) -> tuple[int, int]:
        if trades_history.empty or "action" not in trades_history.columns:
            return 0, 0
        sell_count = int((trades_history["action"].str.upper() == "SELL").sum())
        realized_pnl = float(self.portfolio.get_realized_pnl())
        if sell_count == 0:
            return 0, 0
        if realized_pnl > 0:
            return sell_count, 0
        if realized_pnl < 0:
            return 0, sell_count
        return 0, 0

    def _profit_factor(self, realized_pnl: float) -> float:
        if realized_pnl > 0:
            return float("inf")
        if realized_pnl < 0:
            return 0.0
        return 0.0

    def _profit_factor_from_trades(self, trades_history: pd.DataFrame) -> float:
        if trades_history.empty:
            return 0.0
        gross_profit = float(trades_history.loc[trades_history["PnL"] > 0, "PnL"].sum())
        gross_loss = abs(
            float(trades_history.loc[trades_history["PnL"] < 0, "PnL"].sum())
        )
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _drawdown_series(self, portfolio_value: pd.Series) -> pd.Series:
        running_max = portfolio_value.cummax()
        return ((running_max - portfolio_value) / running_max).fillna(0.0)

    def _benchmark_metrics(
        self,
        benchmark_symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        if benchmark_symbol not in self.data:
            raise ValueError(f"Benchmark symbol {benchmark_symbol} has no loaded data")

        series = self.data[benchmark_symbol]["close"]
        series = series[(series.index >= start_date) & (series.index <= end_date)]
        if len(series) < 2:
            raise ValueError(f"Insufficient benchmark data for {benchmark_symbol}")

        start_price = float(series.iloc[0])
        end_price = float(series.iloc[-1])
        total_return = (end_price - start_price) / start_price
        returns = series.pct_change().dropna()
        annualized_return = 0.0
        trading_days = len(series)
        years = trading_days / 252 if trading_days > 0 else 0.0
        if years > 0:
            annualized_return = (end_price / start_price) ** (1 / years) - 1
        volatility = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
        sharpe_ratio = 0.0
        if volatility > 0:
            sharpe_ratio = float((annualized_return - 0.02) / volatility)
        running_max = series.cummax()
        drawdown = ((running_max - series) / running_max).fillna(0.0)
        max_drawdown = float(drawdown.max())

        return {
            "symbol": benchmark_symbol,
            "start_price": start_price,
            "end_price": end_price,
            "total_return": total_return,
            "total_return_percentage": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_percentage": annualized_return * 100,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_percentage": max_drawdown * 100,
        }

    def _walk_forward_windows(
        self,
        date_range: list[datetime],
        splits: int,
    ) -> list[tuple[datetime, datetime]]:
        split_size = len(date_range) // splits
        if split_size == 0:
            raise ValueError("Not enough data points for the requested number of splits")

        windows = []
        for index in range(splits):
            start_idx = index * split_size
            end_idx = len(date_range) if index == splits - 1 else (index + 1) * split_size
            windows.append((date_range[start_idx], date_range[end_idx - 1]))
        return windows

    def _walk_forward_summary(
        self,
        split_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        total_returns = [split["total_return_percentage"] for split in split_summaries]
        sharpe_ratios = [split["sharpe_ratio"] for split in split_summaries]
        max_drawdowns = [split["max_drawdown_percentage"] for split in split_summaries]
        summary = {
            "splits": len(split_summaries),
            "average_total_return_percentage": float(np.mean(total_returns)),
            "average_sharpe_ratio": float(np.mean(sharpe_ratios)),
            "average_max_drawdown_percentage": float(np.mean(max_drawdowns)),
            "best_split_return_percentage": float(np.max(total_returns)),
            "worst_split_return_percentage": float(np.min(total_returns)),
            "total_trades": int(sum(split["total_trades"] for split in split_summaries)),
        }
        benchmark_returns = [
            split["benchmark_return_percentage"]
            for split in split_summaries
            if "benchmark_return_percentage" in split
        ]
        if benchmark_returns:
            summary["average_benchmark_return_percentage"] = float(
                np.mean(benchmark_returns)
            )
        return summary

    def _summary_dict(self, results: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "start_date",
            "end_date",
            "initial_value",
            "final_value",
            "total_return_percentage",
            "annualized_return_percentage",
            "volatility",
            "sharpe_ratio",
            "max_drawdown_percentage",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "profit_factor",
        ]
        return {key: results.get(key) for key in keys}

    def get_results_summary(self) -> str:
        if not self.results:
            return "No backtest results available"

        summary = self.results.get("summary", self.results)
        if "splits" in summary:
            return (
                "Walk-Forward Summary\n"
                "====================\n"
                f"Splits: {summary['splits']}\n"
                f"Average Return: {summary['average_total_return_percentage']:.2f}%\n"
                f"Average Sharpe: {summary['average_sharpe_ratio']:.2f}\n"
                f"Average Max Drawdown: "
                f"{summary['average_max_drawdown_percentage']:.2f}%\n"
            )

        return (
            "Backtest Results Summary\n"
            "========================\n"
            f"Period: {summary['start_date']} to {summary['end_date']}\n"
            f"Initial Value: ${summary['initial_value']:,.2f}\n"
            f"Final Value: ${summary['final_value']:,.2f}\n"
            f"Total Return: {summary['total_return_percentage']:.2f}%\n"
            f"Annualized Return: {summary['annualized_return_percentage']:.2f}%\n"
            f"Volatility: {summary['volatility']:.2f}\n"
            f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {summary['max_drawdown_percentage']:.2f}%\n"
            f"Total Trades: {summary['total_trades']}\n"
            f"Winning Trades: {summary['winning_trades']}\n"
            f"Losing Trades: {summary['losing_trades']}\n"
            f"Win Rate: {summary['win_rate']:.2f}%\n"
            f"Profit Factor: {summary['profit_factor']:.2f}\n"
        )

    def reset(self) -> None:
        self.portfolio = Portfolio(self.initial_cash)
        self.results = {}
        self.logger.info("Backtester reset to initial state")

    def _normalized_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        normalized = data.copy()
        normalized.index = normalized.index.map(self._normalize_datetime)
        normalized = normalized[~normalized.index.duplicated(keep="last")]
        normalized = normalized.sort_index()
        return normalized

    def _normalize_datetime(self, value: Any) -> datetime:
        if hasattr(value, "tz_localize"):
            if value.tz is None:
                return value.tz_localize(None).to_pydatetime()
            return value.tz_convert(None).tz_localize(None).to_pydatetime()
        timestamp = pd.Timestamp(value)
        if timestamp.tz is not None:
            return timestamp.tz_convert(None).to_pydatetime()
        return timestamp.to_pydatetime()
