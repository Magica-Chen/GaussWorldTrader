"""Explicit Alpaca adapters. Credentials stay out of session research workers."""

from __future__ import annotations

import hashlib
import json
import threading
import time as monotonic_time
from collections import deque
from datetime import timedelta, timezone
from decimal import Decimal
from queue import Full, Queue
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, StrictBool

from .models import (
    AccountProfile,
    AccountOperationalState,
    DataCapabilitySnapshot,
    DataProfile,
    Instrument,
    MarketEvent,
    NewsEvent,
    PositionLeg,
    TradingSession,
    utc_now,
)


def raw(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


class _AccountTradingPermissions(BaseModel):
    """Current REST permissions, independent of retired SDK configuration fields."""

    model_config = ConfigDict(extra="ignore")
    fractional_trading: StrictBool
    no_shorting: StrictBool
    suspend_trade: StrictBool
    closing_transactions_only: StrictBool


class AlpacaBroker:
    def __init__(self, key, secret, environment):
        from alpaca.trading.client import TradingClient

        self.environment = environment
        self.client = TradingClient(key, secret, paper=environment == "paper")
        # Requests have bounded timeouts, including account/reconciliation calls.
        self.client._session.request = _with_timeout(self.client._session.request)
        self.key, self.secret = key, secret
        self._stream = None
        self.metadata_budget = RequestBudget(limit=60, news_reserve=0)
        self.events = Queue(maxsize=10000)
        self.stream_health = "NOT_STARTED"

    def identity(self):
        value = self.client.get_account()
        if not getattr(value, "id", None):
            raise ValueError("BROKER_ACCOUNT_ID_UNAVAILABLE")
        return {"account_id": str(value.id), "environment": self.environment}

    def operational_account(self, now=None):
        return AccountOperationalState(
            **self.identity(), positions=(), observed_at=now or utc_now()
        )

    def account(self, policy, now=None):
        value = self.client.get_account()
        # alpaca-py 0.42's AccountConfiguration requires dtbp_check/pdt_check,
        # which the API removed. Validate the actual permissions we consume.
        configuration = _AccountTradingPermissions.model_validate(
            self.client.get("/account/configurations")
        )
        holdings = self.client.get_all_positions()
        now = now or utc_now()
        unsupported = [
            p
            for p in holdings
            if getattr(p.asset_class, "value", p.asset_class) not in {"us_equity", "us_option"}
        ]
        out_of_scope_exposure = sum(
            (max(abs(Decimal(p.cost_basis)), abs(Decimal(p.market_value))) for p in unsupported),
            Decimal("0"),
        )
        exposure = sum(
            (max(abs(Decimal(p.cost_basis)), abs(Decimal(p.market_value))) for p in holdings),
            Decimal("0"),
        )
        if not getattr(value, "id", None):
            raise ValueError("BROKER_ACCOUNT_ID_UNAVAILABLE")
        # Unknown or non-finite fields are rejected by Decimal/Pydantic, never replaced.
        return AccountProfile(
            account_id=str(value.id),
            environment=self.environment,
            equity=Decimal(value.equity),
            cash=Decimal(value.cash),
            buying_power=Decimal(value.buying_power),
            options_buying_power=Decimal(value.options_buying_power)
            if value.options_buying_power is not None
            else None,
            currency=value.currency,
            observed_at=now,
            trading_blocked=(
                value.trading_blocked
                or value.account_blocked
                or value.trade_suspended_by_user
                or configuration.suspend_trade
                or configuration.closing_transactions_only
                or getattr(value.status, "value", value.status) != "ACTIVE"
            ),
            fractional_allowed=configuration.fractional_trading,
            shorting_allowed=bool(value.shorting_enabled) and not configuration.no_shorting,
            options_level=int(getattr(value, "options_trading_level", 0) or 0),
            margin_allowed=Decimal(value.multiplier) > 1,
            mandate_id=policy.id,
            exposure=exposure,
            out_of_scope_exposure=out_of_scope_exposure,
            out_of_scope_symbols=tuple(sorted(p.symbol for p in unsupported)),
        )

    def positions(self, account_id, now=None):
        now = now or utc_now()
        result = []
        for value in self.client.get_all_positions():
            asset_class = getattr(value.asset_class, "value", value.asset_class)
            if asset_class not in {"us_equity", "us_option"}:
                continue
            asset_type = "option" if asset_class == "us_option" else "stock"
            multiplier = Decimal("1")
            if asset_type == "option":
                contract = self.client.get_option_contract(value.symbol)
                multiplier = Decimal(str(contract.size))
            result.append(
                PositionLeg(
                    account_id=account_id,
                    environment=self.environment,
                    group_id="unmanaged:" + value.symbol,
                    symbol=value.symbol,
                    asset_type=asset_type,
                    quantity=Decimal(value.qty),
                    multiplier=multiplier,
                    cost_basis=Decimal(value.cost_basis),
                    market_value=Decimal(value.market_value),
                    reconciled_at=now,
                    contract_id=str(value.asset_id),
                )
            )
        return tuple(result)

    def orders(self):
        from alpaca.common.enums import Sort
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        # Open and recent terminal orders; unresolved older IDs queried independently by reconciler.
        return [
            raw(o)
            for o in self.client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.ALL, limit=500, direction=Sort.DESC, nested=True
                )
            )
        ]

    def activities(self):
        # Assignment/exercise events require REST. Poll recent days and deduplicate by broker ID.
        activities = []
        token = None
        for _ in range(100):
            params = {
                "after": (utc_now() - timedelta(days=7)).date().isoformat(),
                "direction": "desc",
                "page_size": 100,
            }
            if token:
                params["page_token"] = token
            page = self.client.get("/account/activities", data=params)
            activities.extend(page)
            if len(page) < 100:
                return activities
            token = page[-1]["id"]
        raise RuntimeError("ACTIVITY_PAGINATION_BUDGET_EXHAUSTED")

    def order_by_client_id(self, client_id):
        try:
            return raw(self.client.get_order_by_client_id(client_id))
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise

    def submit(self, intent, quantity, client_id):
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce, PositionIntent

        first = intent.legs[0]
        options = first.instrument.asset_type == "option"
        fields = {
            "qty": float(quantity),
            "time_in_force": TimeInForce.DAY,
            "client_order_id": client_id,
            "limit_price": float(intent.limit_price),
        }
        if len(intent.legs) > 1:
            fields["limit_price"] *= -1 if intent.limit_effect == "credit" else 1
            fields.update(
                order_class=OrderClass.MLEG,
                legs=[
                    OptionLegRequest(
                        symbol=leg.instrument.symbol,
                        ratio_qty=leg.ratio,
                        side=OrderSide(leg.side),
                        position_intent=PositionIntent(leg.position_intent),
                    )
                    for leg in intent.legs
                ],
            )
        else:
            fields.update(symbol=first.instrument.symbol, side=OrderSide(first.side))
            if options:
                fields["position_intent"] = PositionIntent(first.position_intent)
        return raw(self.client.submit_order(LimitOrderRequest(**fields)))

    def cancel(self, broker_id):
        return self.client.cancel_order_by_id(broker_id)

    def replace(self, broker_id, *, limit_price, client_order_id):
        """Quantity stays fixed; the gateway authorizes single-leg replacements."""
        from alpaca.trading.requests import ReplaceOrderRequest

        return raw(
            self.client.replace_order_by_id(
                broker_id,
                ReplaceOrderRequest(
                    limit_price=float(limit_price), client_order_id=client_order_id
                ),
            )
        )

    def instrument(self, symbol, capacity, *, operational=False):
        if not operational and hasattr(self, "metadata_budget"):
            self.metadata_budget.acquire()
        asset = self.client.get_asset(symbol)
        if getattr(asset.asset_class, "value", asset.asset_class) not in {"us_equity", "us_option"}:
            raise ValueError("UNSUPPORTED_SESSION_ASSET_CLASS")
        asset_type = (
            "option"
            if getattr(asset.asset_class, "value", asset.asset_class) == "us_option"
            else "stock"
        )
        if asset_type == "stock":
            return Instrument(
                symbol=symbol,
                tradable=asset.tradable,
                liquidity_capacity=capacity,
                quantity_increment=Decimal("1"),
            )
        contract = self.client.get_option_contract(symbol)
        return Instrument(
            symbol=symbol,
            asset_type="option",
            # Initial option policy uses a dime grid valid across penny/non-penny classes.
            tick_size=Decimal(".10"),
            multiplier=Decimal(str(contract.size)),
            liquidity_capacity=capacity,
            contract_id=str(contract.id),
            expiry=str(contract.expiration_date),
            underlying=contract.underlying_symbol,
            option_type=getattr(contract.type, "value", contract.type),
            strike=Decimal(str(contract.strike_price)),
            standard_contract=_standard_contract(contract),
            tradable=contract.tradable,
        )

    def operational_instrument(self, symbol, capacity):
        """Confirmed holding mechanics never compete with discovery metadata quotas."""
        return self.instrument(symbol, capacity, operational=True)

    def option_contracts(self, underlying, selector, now):
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import AssetStatus

        request = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=now.date() + timedelta(days=selector.minimum_dte),
            expiration_date_lte=now.date() + timedelta(days=selector.maximum_dte),
            limit=100,
        )
        instruments = []
        for _ in range(20):
            if hasattr(self, "metadata_budget"):
                self.metadata_budget.acquire()
            result = self.client.get_option_contracts(request)
            for contract in result.option_contracts:
                instruments.append(
                    Instrument(
                        symbol=contract.symbol,
                        asset_type="option",
                        tick_size=Decimal(".10"),
                        multiplier=Decimal(str(contract.size)),
                        quantity_increment=Decimal("1"),
                        liquidity_capacity=Decimal("0"),
                        contract_id=str(contract.id),
                        expiry=str(contract.expiration_date),
                        underlying=contract.underlying_symbol,
                        option_type=getattr(contract.type, "value", contract.type),
                        strike=Decimal(str(contract.strike_price)),
                        standard_contract=_standard_contract(contract),
                        tradable=contract.tradable,
                    )
                )
            if not result.next_page_token:
                return tuple(instruments)
            request = request.model_copy(update={"page_token": result.next_page_token})
        raise RuntimeError("OPTION_CONTRACT_PAGINATION_BUDGET_EXHAUSTED")

    def calendar(self, start, end):
        from alpaca.trading.requests import GetCalendarRequest

        values = self.client.get_calendar(GetCalendarRequest(start=start, end=end))
        result = []
        ny = ZoneInfo("America/New_York")
        for value in values:
            opening = value.open
            closing = value.close
            if opening.tzinfo is None:
                opening = opening.replace(tzinfo=ny)
            if closing.tzinfo is None:
                closing = closing.replace(tzinfo=ny)
            result.append(
                TradingSession(
                    id=f"US_EQUITIES:{value.date}",
                    session_date=str(value.date),
                    calendar_version="alpaca-calendar-v1",
                    open=opening.astimezone(timezone.utc),
                    close=closing.astimezone(timezone.utc),
                )
            )
        return tuple(result)

    def start_stream(self):
        if self._stream:
            return
        from alpaca.trading.stream import TradingStream

        self._stream = TradingStream(self.key, self.secret, paper=self.environment == "paper")

        async def handler(event):
            try:
                self.events.put_nowait(raw(event))
                self.stream_health = "HEALTHY"
            except Full:
                self.stream_health = "OVERFLOW_RECONCILIATION_REQUIRED"

        self._stream.subscribe_trade_updates(handler)

        def run():
            try:
                self.stream_health = "CONNECTING"
                self._stream.run()
            except Exception:
                self.stream_health = "DISCONNECTED_REST_RECONCILIATION"

        threading.Thread(target=run, name="gauss-broker-stream", daemon=True).start()

    def close(self):
        if self._stream:
            self._stream.stop()


def _standard_contract(contract):
    # Adjusted roots/deliverables are outside the initial lifecycle policy.
    return (
        Decimal(str(contract.size)) == 100
        and getattr(contract, "root_symbol", None) == contract.underlying_symbol
    )


def _with_timeout(method):
    def bounded(*args, **kwargs):
        kwargs.setdefault("timeout", 15)
        return method(*args, **kwargs)

    return bounded


class RequestBudget:
    """Shared market endpoint quota with a protected allowance for news polling.

    Exhaustion returns immediately to the scheduler. Broker operational requests
    use a different client and are never held behind historical work.
    """

    def __init__(self, limit=120, news_reserve=20, clock=monotonic_time.monotonic):
        if limit <= 0 or not 0 <= news_reserve < limit:
            raise ValueError("invalid endpoint quota")
        self.limit, self.news_reserve, self.clock = limit, news_reserve, clock
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self, news=False):
        now = self.clock()
        with self.lock:
            while self.calls and self.calls[0][0] <= now - 60:
                self.calls.popleft()
            market_calls = sum(not is_news for _, is_news in self.calls)
            if len(self.calls) >= self.limit or (
                not news and market_calls >= self.limit - self.news_reserve
            ):
                raise RuntimeError("MARKET_REQUEST_BUDGET_EXHAUSTED")
            self.calls.append((now, news))

    def wrap(self, request, *, news=False):
        def bounded(*args, **kwargs):
            self.acquire(news)
            kwargs.setdefault("timeout", 15)
            return request(*args, **kwargs)

        return bounded


class AlpacaMarket:
    def __init__(self, key, secret, config):
        from alpaca.data.historical import (
            StockHistoricalDataClient,
            OptionHistoricalDataClient,
            NewsClient,
        )

        self.stock = StockHistoricalDataClient(key, secret)
        self.option = OptionHistoricalDataClient(key, secret)
        self.news_client = NewsClient(key, secret)
        self.request_budget = RequestBudget(
            config.market_requests_per_minute, config.news_reserved_requests_per_minute
        )
        for client in (self.stock, self.option, self.news_client):
            client._session.request = self.request_budget.wrap(
                client._session.request, news=client is self.news_client
            )
        self.key, self.secret, self.config = key, secret, config
        self.events = Queue(maxsize=10000)
        self.event_sink = None
        self.gaps = []
        self.stream = None
        self.option_stream = None
        self.subscribed = set()
        self.option_subscribed = set()
        self.cursors = {}
        self.news_cursor = None
        self.health = {
            "price_data": "NOT_STARTED",
            "news": "NOT_STARTED",
            "options": "NOT_REQUESTED",
        }

    def probe(self, account_id, now, symbols):
        from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
        from alpaca.data.timeframe import TimeFrame

        results = []
        from .evidence import EvidenceClock

        cutoff = EvidenceClock(
            self.config.data_profile, self.config.entitlement_boundary_buffer
        ).safe_request_end(now)
        probes = (
            [
                (
                    "stock_historical",
                    "sip",
                    lambda: self.stock.get_stock_bars(
                        StockBarsRequest(
                            symbol_or_symbols=list(symbols[:1]),
                            timeframe=TimeFrame.Minute,
                            start=cutoff - timedelta(minutes=10),
                            end=cutoff,
                            feed=DataFeed.SIP,
                        )
                    ),
                )
            ]
            if symbols
            else []
        )
        if self.config.data_profile == DataProfile.SUBSCRIBED_REALTIME and symbols:
            probes.append(
                (
                    "stock_latest",
                    "sip",
                    lambda: self.stock.get_stock_latest_quote(
                        StockLatestQuoteRequest(
                            symbol_or_symbols=list(symbols[:1]), feed=DataFeed.SIP
                        )
                    ),
                )
            )
        for endpoint, feed, fn in probes:
            try:
                fn()
                outcome, details = (
                    "AVAILABLE",
                    "Endpoint access verified; symbol freshness checked separately.",
                )
            except Exception as exc:
                outcome, details = "UNAVAILABLE", type(exc).__name__
            results.append(
                DataCapabilitySnapshot(
                    account_id=account_id,
                    environment=self.config.environment,
                    endpoint=endpoint,
                    feed=feed,
                    outcome=outcome,
                    quality_class="genuine",
                    expires_at=now + timedelta(minutes=5),
                    details=details,
                )
            )
        if self.config.policy.options_enabled:
            # Option capability is probed against selected real contracts, never inferred from SIP.
            results.append(
                DataCapabilitySnapshot(
                    account_id=account_id,
                    environment=self.config.environment,
                    endpoint="option_latest",
                    feed="opra",
                    outcome="UNKNOWN",
                    quality_class="genuine",
                    expires_at=now + timedelta(minutes=5),
                    details="Requires a selected contract probe.",
                )
            )
        return results

    @staticmethod
    def bar(value, received):
        stamp = value.timestamp
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return MarketEvent(
            symbol=value.symbol,
            source="alpaca",
            source_version=received.isoformat(),
            event_type="bar",
            effective_event_time=stamp + timedelta(minutes=1),
            interval_start=stamp,
            received_at=received,
            available_at=received,
            feed="sip",
            quality_class="genuine",
            price=Decimal(str(value.close)),
            open_price=Decimal(str(value.open)),
            high_price=Decimal(str(value.high)),
            low_price=Decimal(str(value.low)),
            volume=Decimal(str(value.volume)),
        )

    @staticmethod
    def quote(value, received, feed):
        return MarketEvent(
            symbol=value.symbol,
            source="alpaca",
            source_version=received.isoformat(),
            event_type="quote",
            effective_event_time=value.timestamp,
            received_at=received,
            available_at=received,
            feed=feed,
            quality_class="genuine" if feed != "indicative" else "indicative",
            bid=Decimal(str(value.bid_price)),
            ask=Decimal(str(value.ask_price)),
            bid_size=Decimal(str(value.bid_size)),
            ask_size=Decimal(str(value.ask_size)),
        )

    def collect_quotes(self, symbols, now):
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed

        if self.config.data_profile != DataProfile.SUBSCRIBED_REALTIME or not symbols:
            return []
        values = self.stock.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=DataFeed.SIP)
        )
        quotes = [
            self.quote(value, utc_now(), "sip") for value in values.values() if value.ask_price > 0
        ]
        if self.event_sink:
            for event in quotes:
                self.event_sink(event)
        return quotes

    def collect(self, symbols, now):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import DataFeed
        from alpaca.data.timeframe import TimeFrame
        from .evidence import EvidenceClock

        cutoff = EvidenceClock(
            self.config.data_profile, self.config.entitlement_boundary_buffer
        ).safe_request_end(now)
        events = []
        while not self.events.empty():
            event = self.events.get_nowait()
            events.append(event)
            if self.event_sink:
                self.event_sink(event)
        # Current quotes are persisted before a long/paginated historical request.
        events.extend(self.collect_quotes(symbols, now))
        if self.config.data_profile == DataProfile.SUBSCRIBED_REALTIME and symbols:
            self.start_stream(symbols)
        universe = tuple(dict.fromkeys(symbols))
        offset = getattr(self, "_history_rotation", 0) % max(1, len(universe))
        ordered = universe[offset:] + universe[:offset]
        for beginning in range(0, len(ordered), 16):
            batch = ordered[beginning : beginning + 16]
            start = min(
                self.cursors.get(symbol, cutoff - timedelta(days=5)) for symbol in batch
            ) - timedelta(minutes=2)
            # The SDK follows pagination. A completed batch is durable before the
            # next request, and the rotation resumes at an unfinished batch.
            response = self.stock.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=list(batch),
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=cutoff,
                    feed=DataFeed.SIP,
                    limit=10000,
                )
            )
            received = utc_now()
            for symbol in batch:
                parsed = [self.bar(value, received) for value in response.data.get(symbol, [])]
                eligible = [event for event in parsed if event.effective_event_time <= cutoff]
                events.extend(eligible)
                if self.event_sink:
                    for event in eligible:
                        self.event_sink(event)
                if eligible:
                    self.cursors[symbol] = max(event.effective_event_time for event in eligible)
            self._history_rotation = (offset + beginning + len(batch)) % max(1, len(universe))
        self.health["price_data"] = "HEALTHY" if not self.gaps else "GAP_REQUIRES_RECOVERY"
        return events, []

    def collect_news(self, symbols, now):
        from alpaca.data.requests import NewsRequest

        news = []
        request = NewsRequest(
            start=self.news_cursor or now - timedelta(days=2),
            end=now,
            symbols=",".join(symbols) if symbols else None,
            include_content=False,
            limit=50,
        )
        for page_number in range(20):
            result = self.news_client.get_news(request)
            received = utc_now()
            for value in result.data.get("news", []):
                body = raw(value)
                headline = value.headline
                # Conservative uncertainty suspends affected entries; it never grants permission.
                material = any(
                    term in headline.lower()
                    for term in (
                        "trading halt",
                        "bankruptcy",
                        "fraud",
                        "restatement",
                        "recall",
                        "sec investigation",
                        "earnings",
                        "acquisition",
                        "merger",
                        "offering",
                        "guidance cut",
                        "default",
                        "retraction",
                    )
                )
                news.append(
                    NewsEvent(
                        provider="alpaca",
                        article_id=str(value.id),
                        source_version=value.updated_at.isoformat(),
                        published_at=value.created_at,
                        updated_at=value.updated_at,
                        received_at=received,
                        available_at=received,
                        symbols=tuple(value.symbols),
                        headline=headline,
                        content_reference=value.url,
                        content_hash=hashlib.sha256(
                            json.dumps(body, sort_keys=True, default=str).encode()
                        ).hexdigest(),
                        category="material_unverified" if material else "unclassified",
                        verified=False,
                        blocking=material,
                    )
                )
            token = result.next_page_token
            if not token:
                self.health["news"] = "HEALTHY"
                self.news_cursor = now - timedelta(minutes=2)
                break
            request = request.model_copy(update={"page_token": token})
        else:
            self.health["news"] = "INCOMPLETE_PAGINATION"
        return news

    def start_stream(self, symbols):
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        if self.stream is None:
            self.stream = StockDataStream(self.key, self.secret, feed=DataFeed.SIP)
        added = set(symbols) - self.subscribed
        if not added:
            return

        async def handler(value):
            try:
                event = (
                    self.quote(value, utc_now(), "sip")
                    if hasattr(value, "bid_price")
                    else self.bar(value, utc_now())
                )
                self.events.put_nowait(event)
            except Full:
                self.gaps.append(
                    {
                        "symbol": value.symbol,
                        "reason": "QUEUE_OVERFLOW",
                        "at": utc_now().isoformat(),
                    }
                )
            except ValueError:
                self.gaps.append(
                    {"symbol": value.symbol, "reason": "INVALID_QUOTE", "at": utc_now().isoformat()}
                )

        self.stream.subscribe_quotes(handler, *sorted(added))
        self.stream.subscribe_bars(handler, *sorted(added))
        first = not self.subscribed
        self.subscribed.update(added)
        if first:

            def run():
                try:
                    self.stream.run()
                except Exception:
                    self.health["price_data"] = "STREAM_DISCONNECTED_REST_RECOVERY"
                    self.gaps.append({"reason": "STREAM_DISCONNECTED", "at": utc_now().isoformat()})
                finally:
                    self.stream = None
                    self.subscribed.clear()

            threading.Thread(target=run, name="gauss-stock-stream", daemon=True).start()

    def collect_options(self, symbols, now):
        from alpaca.data.requests import OptionLatestQuoteRequest
        from alpaca.data.enums import OptionsFeed

        if not symbols:
            return []
        feed = (
            OptionsFeed.OPRA
            if self.config.data_profile == DataProfile.SUBSCRIBED_REALTIME
            else OptionsFeed.INDICATIVE
        )
        quotes = self.option.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=feed)
        )
        if feed == OptionsFeed.OPRA:
            self.start_option_stream(symbols)
        self.health["options"] = (
            "HEALTHY_OPRA" if feed == OptionsFeed.OPRA else "INDICATIVE_RESEARCH_ONLY"
        )
        return [
            self.quote(q, utc_now(), getattr(feed, "value", feed))
            for q in quotes.values()
            if q.ask_price > 0
        ]

    def start_option_stream(self, symbols):
        from alpaca.data.live import OptionDataStream
        from alpaca.data.enums import OptionsFeed

        if self.option_stream is None:
            self.option_stream = OptionDataStream(self.key, self.secret, feed=OptionsFeed.OPRA)
        added = set(symbols) - self.option_subscribed
        if not added:
            return

        async def handler(value):
            try:
                self.events.put_nowait(self.quote(value, utc_now(), "opra"))
            except (Full, ValueError):
                self.gaps.append(
                    {
                        "symbol": value.symbol,
                        "reason": "OPTION_STREAM_GAP",
                        "at": utc_now().isoformat(),
                    }
                )

        self.option_stream.subscribe_quotes(handler, *sorted(added))
        first = not self.option_subscribed
        self.option_subscribed.update(added)
        if first:

            def run():
                try:
                    self.option_stream.run()
                except Exception:
                    self.health["options"] = "DISCONNECTED_REST_RECOVERY"
                    self.gaps.append(
                        {"reason": "OPTION_STREAM_DISCONNECTED", "at": utc_now().isoformat()}
                    )
                finally:
                    self.option_stream = None
                    self.option_subscribed.clear()

            threading.Thread(target=run, name="gauss-option-stream", daemon=True).start()

    def close(self):
        if self.stream:
            self.stream.stop()
        if self.option_stream:
            self.option_stream.stop()
