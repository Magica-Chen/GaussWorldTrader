"""Offline history bootstrap contracts using the installed SDK pagination loop."""

import json
from datetime import UTC, datetime, timedelta

import pytest
import requests

from src.runtime.broker import AlpacaMarket
from src.runtime.models import RuntimeConfig

NOW = datetime(2026, 9, 26, 11, tzinfo=UTC)
SYMBOLS = ("SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "TLT")


def bars(count, offset=0):
    start = NOW - timedelta(days=4)
    return [
        {
            "t": (start + timedelta(minutes=index + offset)).isoformat(),
            "o": 100,
            "h": 101,
            "l": 99,
            "c": 100,
            "v": 10,
            "n": 1,
            "vw": 100,
        }
        for index in range(count)
    ]


def paginated_market(monkeypatch, pages, **config):
    calls = []

    def request(_session, method, url, **kwargs):
        assert method == "GET" and url.endswith("/stocks/bars")
        assert kwargs["timeout"] == 15
        params = dict(kwargs["params"])
        calls.append(params)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(pages[params.get("page_token")]).encode()
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr("src.runtime.broker.utc_now", lambda: NOW)
    market = AlpacaMarket("offline-key", "offline-secret", RuntimeConfig(**config))
    monkeypatch.setattr(market, "collect_quotes", lambda symbols, now: [])
    monkeypatch.setattr(market, "start_stream", lambda symbols: None)
    return market, calls


def test_bootstrap_follows_sdk_pages_after_first_ten_thousand_bars(monkeypatch):
    pages = {
        None: {
            "bars": {"AAPL": bars(4000), "DIA": bars(4000), "IWM": bars(2000)},
            "next_page_token": "page-2",
        },
        "page-2": {
            "bars": {
                "IWM": bars(1, offset=2000),
                **{symbol: bars(1) for symbol in ("MSFT", "NVDA", "QQQ", "SPY", "TLT")},
            },
            "next_page_token": None,
        },
    }
    market, calls = paginated_market(monkeypatch, pages)
    persisted = []
    market.event_sink = persisted.append
    try:
        events, news = market.collect(SYMBOLS, NOW)
        assert len(events) == len(persisted) == 10006
        assert {event.symbol for event in events} == set(SYMBOLS)
        assert set(market.cursors) == set(SYMBOLS)
        assert all(event.feed == "sip" for event in events)
        assert news == []
        assert [call["page_token"] for call in calls] == [None, "page-2"]
        assert [call["limit"] for call in calls] == [10000, 10000]
        assert len(market.request_budget.calls) == 2
        assert market.health["price_data"] == "HEALTHY"
    finally:
        market.close()


def test_pagination_budget_exhaustion_keeps_history_cursors_unadvanced(monkeypatch):
    pages = {None: {"bars": {"AAPL": bars(1)}, "next_page_token": "page-2"}}
    market, calls = paginated_market(
        monkeypatch, pages, market_requests_per_minute=2, news_reserved_requests_per_minute=1
    )
    persisted = []
    market.event_sink = persisted.append
    try:
        with pytest.raises(RuntimeError, match="MARKET_REQUEST_BUDGET_EXHAUSTED"):
            market.collect(SYMBOLS, NOW)
        assert len(calls) == 1
        assert market.cursors == {}
        assert persisted == []
        assert market.health["price_data"] != "HEALTHY"
    finally:
        market.close()
