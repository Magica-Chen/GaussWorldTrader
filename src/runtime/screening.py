"""One-shot, read-only market research using registered signal strategies."""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_STRATEGIES = ("momentum", "trend_following", "mean_reversion")
NY = ZoneInfo("America/New_York")


def session_time(value):
    value = pd.Timestamp(value).to_pydatetime()
    return value.replace(tzinfo=NY) if value.tzinfo is None else value


def completed_daily(frame, last_date):
    if frame.empty:
        return frame
    frame = frame.reset_index()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame[frame.timestamp.dt.tz_convert(NY).dt.date <= last_date]
    return frame.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def analyze(frame, strategies, as_of):
    """Evaluate supplied complete daily bars; rank signals without claiming expected returns."""
    from src.strategy.registry import get_strategy_registry

    registry = get_strategy_registry()
    unsupported = set(strategies) - set(DEFAULT_STRATEGIES)
    if unsupported:
        raise ValueError(f"One-shot daily research does not support: {sorted(unsupported)}")
    implementations = {name: registry.create(name) for name in strategies}
    rows = []
    for symbol, data in frame.groupby("symbol"):
        data = data.sort_values("timestamp").set_index("timestamp")
        if len(data) < 60 or data.index[-1].tz_convert(NY).date() != as_of.date():
            continue
        if (
            not data[["open", "high", "low", "close", "volume"]]
            .apply(lambda col: col.map(math.isfinite))
            .all()
            .all()
            or (data.close <= 0).any()
        ):
            continue
        close = float(data.close.iloc[-1])
        prev = data.close.shift(1)
        tr = pd.concat(
            [data.high - data.low, (data.high - prev).abs(), (data.low - prev).abs()], axis=1
        ).max(axis=1)
        atr = float(tr.tail(14).mean())
        if atr <= 0:
            continue
        signals = {}
        for name, strategy in implementations.items():
            signal = strategy.get_signal(
                symbol=symbol,
                current_date=as_of,
                current_price=close,
                current_data={},
                historical_data=data,
                portfolio=None,
            )
            signals[name] = {
                "signal": signal.signal if signal else "HOLD",
                "reason": signal.reason if signal else "No signal",
            }
        buy = [name for name, value in signals.items() if value["signal"] == "BUY"]
        # Conditional opening-session scenario, separate from historical signal price.
        entry = math.ceil((float(data.high.iloc[-1]) + 0.05 * atr) * 100) / 100
        risk = 1.5 * atr
        rows.append(
            {
                "symbol": symbol,
                "close": close,
                "atr14": atr,
                "return20_pct": 100 * (close / float(data.close.iloc[-21]) - 1),
                "dollar_volume20": float((data.close * data.volume).tail(20).mean()),
                "signals": signals,
                "buy_strategies": buy,
                "entry": entry,
                "max_entry": round(entry + 0.25 * atr, 2),
                "stop": round(entry - risk, 2),
                "target": round(entry + 2 * risk, 2),
                "reward_risk": 2.0,
            }
        )
    return sorted(
        rows, key=lambda r: (len(r["buy_strategies"]), r["dollar_volume20"]), reverse=True
    )


def research_calendar(config, now):
    from .events import configured_event_calendar

    try:
        provider = configured_event_calendar(config)
        try:
            snapshot = provider.snapshot(now)
        finally:
            if hasattr(provider, "close"):
                provider.close()
        return {
            "status": ("available" if snapshot.verified else "partial")
            + ": "
            + snapshot.source_reference,
            **snapshot.model_dump(mode="json"),
        }
    except Exception as exc:
        return {"status": f"unavailable ({type(exc).__name__})", "events": [], "coverage": {}}


def render_report(report):
    coverage = report["coverage"]
    lines = [
        f"# Market research for {report['target_session']}",
        "",
        f"Completed session: {report['completed_session']}. Generated: {report['generated_at']}.",
        f"Universe: {coverage['universe']} active tradable US equities; "
        f"{coverage['with_recent_bars']} with recent bars; {coverage['liquid']} pass liquidity filters; "
        f"{coverage['analyzed']} analyzed with at least 60 daily bars.",
        "",
        "Screen: price ≥ $5 and average recent dollar volume ≥ $20m. Analyze the most liquid "
        "candidates first. Exclude OTC listings and names identifying leveraged/inverse products, "
        "warrants, preferred shares or rights. ETFs can remain in the universe.",
        "",
        "Signals use the existing momentum (12/26-day), trend-following (20/50-day SMA), "
        "and mean-reversion (Bollinger bands/RSI) implementations selected in the allowlist. "
        "Rank by BUY-signal count, then 20-session dollar volume; ranking is not a return forecast.",
        "",
        "## Conditional watchlist",
        "",
        "| Symbol | BUY strategies | Last close | Trigger | Max entry | Stop | Target | 20-day change |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["candidates"]:
        lines.append(
            f"| {r['symbol']} | {', '.join(r['buy_strategies'])} | ${r['close']:.2f} | "
            f"${r['entry']:.2f} | ${r['max_entry']:.2f} | ${r['stop']:.2f} | ${r['target']:.2f} | {r['return20_pct']:+.1f}% |"
        )
    if not report["candidates"]:
        lines += [
            "",
            "No qualifying BUY candidates: hold cash and reassess after the next completed session.",
        ]
    lines += [
        "",
        "## Next-session plan",
        "",
        f"Market opens {report['target_open']}; closes {report['target_close']}.",
        "After the first 15 minutes, consider a long entry only after a completed 5-minute bar "
        "closes above the trigger and the current executable price stays below Max entry. "
        "Skip a gap above Max entry or a break below the stop before entry. "
        "The trigger is the prior high plus 0.05 ATR; stop distance is 1.5 ATR; target is 2 times that risk.",
        "Review current news, earnings and scheduled economic releases before selecting any trade. "
        "An unavailable event calendar leaves these scenarios pending review.",
        "For an intraday paper trial, risk at most 0.25% of equity per position, allocate at most "
        "10% of equity to one name, and take at most two positions with combined planned risk ≤ 0.5%. "
        "Shares = floor(min(0.0025 × equity / (entry − stop), 0.10 × equity / entry, available cash / entry)). "
        "Apply stricter account-policy limits and account for existing exposure. Stops can slip. "
        "Close intraday positions before the session ends.",
        "FREE_DELAYED data supports this end-of-day screen; the opening trigger requires fresh quotes "
        "and bars in the broker interface. This run ends after writing the report and does not monitor triggers.",
        "",
        "## Evidence and remaining checks",
        "",
        f"Event calendar: {report['event_calendar']}. News: {report['news_status']}.",
        "These are unvalidated research scenarios, not execution approvals or backtested expectancy estimates. "
        "No orders, strategy approvals or changes to the running session database are made.",
        "",
    ]
    details = report.get("event_calendar_details", {})
    if details:
        lines += [
            "Calendar coverage: "
            + "; ".join(f"{k}: {v}" for k, v in details.get("coverage", {}).items()),
            f"Dates covered: {details.get('coverage_start', '?')} through {details.get('coverage_end', '?')}.",
            f"Calendar checked: {details.get('published_at', 'unknown')}.",
            "FRED supplies release dates, not intraday times. Major CPI, PPI, employment, GDP, "
            "income/outlays and retail-sales releases use a full-day entry veto in the "
            "continuous service until a timed operator calendar is supplied. Earnings veto "
            "entries in the affected symbol for the date. Routine FRED releases are informational.",
            "FRED release categories can include routine dataset updates. A “FOMC Press Release” "
            "entry is not proof of a policy meeting that day; use the "
            "[Federal Reserve meeting calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) "
            "for scheduled policy decisions. Speeches and unscheduled announcements are outside these feeds.",
            "",
            "### Events for the target session",
            "",
        ]
        target_events = [
            e for e in details.get("events", []) if e["occurs_at"][:10] == report["target_session"]
        ]
        names = {c["symbol"] for c in report["candidates"]}
        relevant = [
            e for e in target_events if not e.get("symbols") or names.intersection(e["symbols"])
        ]
        for event in relevant:
            label = "date only" if event.get("time_precision") == "date" else event["occurs_at"]
            lines.append(
                f"- {label}: [{event.get('title') or event['category']}]({event['source_reference']})"
                + (" — entry veto" if event.get("blocks_entries", True) else "")
            )
        if not relevant:
            lines.append(
                "No matching events in the retrieved coverage; see source availability above."
            )
        lines += ["", "### Recent news", ""]
    shown = set()
    for candidate in report["candidates"]:
        symbol = candidate["symbol"]
        relevant = [item for item in report["news"] if symbol in item.get("symbols", [])][:2]
        if not relevant:
            lines.append(f"- {symbol}: no headline in the retrieved news sample.")
        for item in relevant:
            if item["url"] in shown:
                continue
            shown.add(item["url"])
            words = item["headline"].split()
            title = " ".join(words[:18]) + ("…" if len(words) > 18 else "")
            lines.append(f"- {symbol}, {item['created_at']}: [{title}]({item['url']})")
    lines += [
        "",
        "Sources: [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars), "
        "[NYSE calendar](https://www.nyse.com/trade/hours-calendars).",
        "",
    ]
    return "\n".join(lines)


def run_research(config, *, progress=None, now=None):
    """Read Alpaca data, save reproducible evidence and return; never acquire a trading lease."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import NewsClient, StockHistoricalDataClient
    from alpaca.data.requests import NewsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest, GetCalendarRequest

    from src.settings import get_config

    from .broker import _with_timeout
    from .evidence import EvidenceClock

    progress = progress or (lambda message: print(message, file=sys.stderr, flush=True))
    now = now or datetime.now(UTC)
    strategies = config.strategy_allowlist or DEFAULT_STRATEGIES
    if set(strategies) - set(DEFAULT_STRATEGIES):
        raise ValueError("One-shot research supports momentum, trend_following and mean_reversion")
    credentials = get_config().alpaca
    trading = TradingClient(
        credentials.api_key, credentials.secret_key, paper=config.environment == "paper"
    )
    stock = StockHistoricalDataClient(credentials.api_key, credentials.secret_key)
    news = NewsClient(credentials.api_key, credentials.secret_key)
    for client in (trading, stock, news):
        client._session.request = _with_timeout(client._session.request)
    # Pace actual HTTP requests, including SDK pagination, below 60 requests/minute.
    request = stock._session.request
    last_request = [0.0]

    def paced(*args, **kwargs):
        time.sleep(max(0, 1.05 - (time.monotonic() - last_request[0])))
        last_request[0] = time.monotonic()
        return request(*args, **kwargs)

    stock._session.request = paced
    try:
        cutoff = EvidenceClock(
            config.data_profile, config.entitlement_boundary_buffer
        ).safe_request_end(now)
        sessions = trading.get_calendar(
            GetCalendarRequest(
                start=(now - timedelta(days=400)).date(), end=(now + timedelta(days=14)).date()
            )
        )
        complete = [s for s in sessions if session_time(s.close) <= cutoff]
        future = [s for s in sessions if session_time(s.open) > now]
        if not complete or not future:
            raise RuntimeError("A completed session and next opening are required")
        last, target = complete[-1], future[0]
        last_date = session_time(last.close).date()
        assets = trading.get_all_assets(
            GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        )
        excluded = re.compile(r"\b(inverse|leveraged|ultra|2x|3x|warrant|preferred|rights)\b", re.I)
        symbols = sorted(
            {
                a.symbol
                for a in assets
                if a.tradable
                and str(a.exchange).split(".")[-1] != "OTC"
                and not excluded.search(a.name or "")
            }
        )
        if config.symbols:
            symbols = sorted(set(symbols) & set(config.symbols))
        if not symbols:
            raise RuntimeError("No active tradable symbols in the selected universe")
        progress(f"Screening {len(symbols)} US equities using completed data through {last_date}.")

        def fetch(selected, days):
            parts = []
            for offset in range(0, len(selected), 500):
                result = stock.get_stock_bars(
                    StockBarsRequest(
                        symbol_or_symbols=selected[offset : offset + 500],
                        timeframe=TimeFrame.Day,
                        start=session_time(last.close) - timedelta(days=days),
                        end=min(
                            cutoff,
                            datetime.combine(
                                last_date + timedelta(days=1), datetime.min.time(), NY
                            ),
                        ),
                        feed=DataFeed.SIP,
                        adjustment=Adjustment.SPLIT,
                    )
                )
                if result.data:
                    parts.append(result.df)
                progress(
                    f"Downloaded {min(offset + 500, len(selected))}/{len(selected)} symbols ({days}-day window)."
                )
            if not parts:
                raise RuntimeError("No historical bars returned; no report generated")
            return completed_daily(pd.concat(parts), last_date)

        recent = fetch(symbols, 12)
        liquid = []
        for symbol, group in recent.groupby("symbol"):
            if group.timestamp.iloc[-1].tz_convert(NY).date() != last_date or len(group) < 3:
                continue
            turnover = float((group.close * group.volume).mean())
            if float(group.close.iloc[-1]) >= 5 and turnover >= 20_000_000:
                liquid.append((symbol, turnover))
        selected = [
            s
            for s, _ in sorted(liquid, key=lambda x: x[1], reverse=True)[
                : config.scan_universe_limit
            ]
        ]
        history = fetch(selected, 180) if selected else recent.iloc[:0]
        rows = analyze(history, strategies, session_time(last.close)) if selected else []
        candidates = [r for r in rows if r["buy_strategies"] and r["stop"] > 0][
            : config.active_candidate_limit
        ]
        headlines, news_status = [], "not requested (no candidates)"
        if candidates:
            try:
                response = news.get_news(
                    NewsRequest(
                        symbols=",".join(r["symbol"] for r in candidates),
                        start=now - timedelta(days=5),
                        end=now,
                        limit=30,
                        include_content=False,
                    )
                )
                headlines = [
                    {
                        "headline": n.headline,
                        "url": n.url,
                        "created_at": str(n.created_at),
                        "symbols": n.symbols,
                    }
                    for n in response.data.get("news", [])
                ]
                news_status = (
                    f"{len(headlines)} recent headlines retrieved; coverage is not exhaustive"
                )
            except Exception as exc:
                news_status = f"unavailable ({type(exc).__name__})"
        calendar_details = research_calendar(config, now)
        event_status = calendar_details["status"]
        report = {
            "generated_at": now.isoformat(),
            "completed_session": str(last_date),
            "target_session": str(session_time(target.open).date()),
            "target_open": str(session_time(target.open)),
            "target_close": str(session_time(target.close)),
            "execution_eligible": False,
            "strategies": list(strategies),
            "coverage": {
                "universe": len(symbols),
                "with_recent_bars": recent.symbol.nunique(),
                "liquid": len(liquid),
                "analyzed": len(rows),
            },
            "candidates": candidates,
            "rankings": rows,
            "news": headlines,
            "news_status": news_status,
            "event_calendar": event_status,
            "event_calendar_details": calendar_details,
        }
        directory = (
            Path(config.payload_directory).parent / "research" / now.strftime("%Y%m%dT%H%M%S%fZ")
        )
        directory.mkdir(parents=True, exist_ok=False)
        recent.to_csv(directory / "screen.csv", index=False)
        history.to_csv(directory / "history.csv", index=False)
        (directory / "universe.json").write_text(json.dumps(symbols, indent=2))
        (directory / "report.json").write_text(json.dumps(report, indent=2, default=str))
        path = directory / "report.md"
        path.write_text(render_report(report))
        progress(f"Research complete: {path}")
        return report, path
    finally:
        for client in (trading, stock, news):
            client._session.close()
