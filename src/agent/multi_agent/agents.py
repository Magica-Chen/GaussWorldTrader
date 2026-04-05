"""Agent implementations for the multi-agent decision system."""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.agent.fundamental_analyzer import FundamentalAnalyzer
from src.agent.multi_agent.types import (
    AgentReport,
    ConsensusDecision,
    DebatePosition,
    RiskAssessment,
)
from src.analysis.technical_analysis import TechnicalAnalysis
from src.data.news_provider import NewsDataProvider

if TYPE_CHECKING:
    from src.llm import BaseLLMProvider
    from src.strategy.base import MarketDataContext


def _clean_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


class BaseAnalystAgent(ABC):
    """Base async wrapper around synchronous agent implementations."""

    role = "base"

    def __init__(self, llm: BaseLLMProvider | None = None) -> None:
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.{self.role}")

    async def analyze(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> Any:
        return await asyncio.to_thread(
            self._analyze_sync,
            symbol,
            market_context,
            **kwargs,
        )

    @abstractmethod
    def _analyze_sync(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> Any:
        """Run the agent synchronously."""

    def _require_llm(self) -> BaseLLMProvider:
        if self.llm is None:
            raise ValueError(f"{self.__class__.__name__} requires an LLM provider")
        return self.llm

    def _llm_context(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        context = {
            "symbol": symbol,
            "agent_role": self.role,
            "current_date": market_context.current_date.isoformat(),
        }
        run_id = kwargs.get("run_id")
        if run_id is not None:
            context["run_id"] = run_id
        return context


class TechnicalAnalystAgent(BaseAnalystAgent):
    """Technical analyst using indicators plus LLM synthesis."""

    role = "technical"

    def __init__(self, llm: BaseLLMProvider) -> None:
        super().__init__(llm)
        self.ta = TechnicalAnalysis()

    def _analyze_sync(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> AgentReport:
        bars = market_context.historical_bars[symbol]
        price = float(market_context.current_prices[symbol])
        indicators = self._build_indicator_snapshot(bars, price)
        prompt = (
            "You are the technical analyst in a trading committee.\n"
            f"Symbol: {symbol}\n"
            f"Current price: {price:.4f}\n"
            f"Technical snapshot:\n{json.dumps(indicators, indent=2)}\n\n"
            "Return a trading stance using only the technical evidence."
        )
        report = self._require_llm().generate_structured(
            prompt,
            AgentReport,
            context=self._llm_context(symbol, market_context, **kwargs),
        )
        report.agent_name = self.role
        report.symbol = symbol
        return report

    def _build_indicator_snapshot(
        self,
        bars: pd.DataFrame,
        current_price: float,
    ) -> dict[str, float | None]:
        close = bars["close"]
        macd_line, signal_line, histogram = self.ta.macd(close)
        atr = self.ta.atr(bars["high"], bars["low"], close)
        sma_20 = self.ta.sma(close, 20)
        sma_50 = self.ta.sma(close, 50)
        return {
            "current_price": current_price,
            "sma_20": _clean_number(sma_20.iloc[-1]),
            "sma_50": _clean_number(sma_50.iloc[-1]),
            "rsi_14": _clean_number(self.ta.rsi(close, 14).iloc[-1]),
            "macd": _clean_number(macd_line.iloc[-1]),
            "macd_signal": _clean_number(signal_line.iloc[-1]),
            "macd_histogram": _clean_number(histogram.iloc[-1]),
            "atr_14": _clean_number(atr.iloc[-1]),
            "price_vs_sma_20": _clean_number(current_price - sma_20.iloc[-1]),
            "price_vs_sma_50": _clean_number(current_price - sma_50.iloc[-1]),
        }


class FundamentalAnalystAgent(BaseAnalystAgent):
    """Fundamental analyst reusing the current analyzer stack."""

    role = "fundamental"

    def __init__(
        self,
        llm: BaseLLMProvider,
        llm_provider: str,
        llm_model: str | None = None,
        finnhub_key: str | None = None,
        fred_key: str | None = None,
    ) -> None:
        super().__init__(llm)
        self.analyzer = FundamentalAnalyzer(
            finnhub_key=finnhub_key,
            fred_key=fred_key,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    def _analyze_sync(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> AgentReport:
        analysis = self._build_analysis_snapshot(symbol, market_context.current_date)
        prompt = (
            "You are the fundamental analyst in a trading committee.\n"
            f"Symbol: {symbol}\n"
            f"Fundamental snapshot:\n{json.dumps(analysis, indent=2)}\n\n"
            "Return a trading stance using only the fundamental evidence."
        )
        report = self._require_llm().generate_structured(
            prompt,
            AgentReport,
            context=self._llm_context(symbol, market_context, **kwargs),
        )
        report.agent_name = self.role
        report.symbol = symbol
        return report

    def _build_analysis_snapshot(
        self,
        symbol: str,
        current_date: datetime,
    ) -> dict[str, Any]:
        market_data = self.analyzer._get_comprehensive_market_data(
            symbol,
            current_date=current_date,
        )
        return {
            "company_profile": market_data.get("company_profile", {}),
            "financial_analysis": self.analyzer._analyze_financial_ratios(
                market_data.get("basic_financials", {})
            ),
            "insider_analysis": self.analyzer._analyze_insider_data(
                market_data.get("insider_transactions", []),
                market_data.get("insider_sentiment", {}),
            ),
            "economic_analysis": self.analyzer._analyze_economic_context(
                market_data.get("economic_indicators", {})
            ),
            "analyst_analysis": self.analyzer._analyze_analyst_recommendations(
                market_data.get("recommendations", {}),
                market_data.get("price_target", {}),
            ),
        }


class SentimentAnalystAgent(BaseAnalystAgent):
    """Sentiment analyst using recent company headlines."""

    role = "sentiment"

    def __init__(self, llm: BaseLLMProvider, finnhub_key: str | None = None) -> None:
        super().__init__(llm)
        self.news_provider = NewsDataProvider(finnhub_key)

    def _analyze_sync(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> AgentReport:
        headlines = self._load_headlines(symbol, market_context.current_date)
        prompt = (
            "You are the sentiment analyst in a trading committee.\n"
            f"Symbol: {symbol}\n"
            f"Recent headlines:\n{json.dumps(headlines, indent=2)}\n\n"
            "Return a trading stance using only the headline and summary flow."
        )
        report = self._require_llm().generate_structured(
            prompt,
            AgentReport,
            context=self._llm_context(symbol, market_context, **kwargs),
        )
        report.agent_name = self.role
        report.symbol = symbol
        return report

    def _load_headlines(
        self,
        symbol: str,
        current_date: datetime,
    ) -> list[dict[str, str]]:
        start = current_date - timedelta(days=14)
        news = self.news_provider.get_company_news(
            symbol,
            from_date=start,
            to_date=current_date,
        )
        items = []
        for article in news[:10]:
            items.append(
                {
                    "headline": article.get("headline", ""),
                    "summary": article.get("summary", ""),
                    "source": article.get("source", ""),
                    "datetime": str(article.get("datetime", "")),
                }
            )
        return items


class RiskManagerAgent(BaseAnalystAgent):
    """Math-only risk manager."""

    role = "risk"

    def __init__(self, base_position_pct: float = 0.05) -> None:
        super().__init__(None)
        self.ta = TechnicalAnalysis()
        self.base_position_pct = base_position_pct

    def _analyze_sync(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> RiskAssessment:
        bars = market_context.historical_bars[symbol]
        reports: list[AgentReport] = kwargs["reports"]
        price = float(market_context.current_prices[symbol])
        atr_value = self._atr_value(bars)
        volatility_pct = float("nan")
        if price > 0 and pd.notna(atr_value):
            volatility_pct = atr_value / price
        disagreement = len({report.action.upper() for report in reports}) > 1
        level, factor = self._risk_profile(volatility_pct, disagreement)
        max_position_pct = self.base_position_pct * factor
        stop_loss_pct = 0.015
        if pd.notna(volatility_pct):
            stop_loss_pct = max(0.015, volatility_pct * 1.25)
        take_profit_pct = stop_loss_pct * 2.0
        risk_flags = self._risk_flags(reports, volatility_pct, disagreement)
        volatility_text = "n/a" if pd.isna(volatility_pct) else f"{volatility_pct:.2%}"
        return RiskAssessment(
            symbol=symbol,
            risk_level=level,
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            atr=atr_value,
            volatility_pct=volatility_pct,
            risk_flags=risk_flags,
            rationale=(
                f"ATR volatility={volatility_text}, disagreement={disagreement}, "
                f"position cap={max_position_pct:.2%}"
            ),
        )

    def _atr_value(self, bars: pd.DataFrame) -> float:
        if len(bars) < 14:
            return float("nan")
        atr = self.ta.atr(bars["high"], bars["low"], bars["close"])
        if atr.empty:
            return float("nan")
        value = _clean_number(atr.iloc[-1])
        return float("nan") if value is None else value

    def _risk_profile(self, volatility_pct: float, disagreement: bool) -> tuple[str, float]:
        if pd.isna(volatility_pct):
            return ("medium", 0.65) if disagreement else ("low", 1.0)
        if volatility_pct >= 0.05:
            return "high", 0.4
        if volatility_pct >= 0.03 or disagreement:
            return "medium", 0.65
        return "low", 1.0

    def _risk_flags(
        self,
        reports: list[AgentReport],
        volatility_pct: float,
        disagreement: bool,
    ) -> list[str]:
        flags = [flag for report in reports for flag in report.risk_flags]
        if pd.isna(volatility_pct):
            flags.append("atr_warmup_incomplete")
        if volatility_pct >= 0.05:
            flags.append("elevated_atr_volatility")
        if disagreement:
            flags.append("agent_disagreement")
        return sorted(set(flags))


class DecisionMakerAgent(BaseAnalystAgent):
    """Decision maker combining reports, debate, and risk controls."""

    role = "decision"

    def __init__(self, llm: BaseLLMProvider) -> None:
        super().__init__(llm)

    def _analyze_sync(
        self,
        symbol: str,
        market_context: MarketDataContext,
        **kwargs: Any,
    ) -> ConsensusDecision:
        price = float(market_context.current_prices[symbol])
        reports: list[AgentReport] = kwargs["reports"]
        risk_assessment: RiskAssessment = kwargs["risk_assessment"]
        debate_positions: list[DebatePosition] = kwargs.get("debate_positions", [])
        payload = {
            "symbol": symbol,
            "current_price": price,
            "reports": [asdict(report) for report in reports],
            "risk_assessment": asdict(risk_assessment),
            "debate_positions": [asdict(position) for position in debate_positions],
        }
        prompt = (
            "You are the final decision maker in a trading committee.\n"
            "Choose BUY, SELL, or HOLD.\n"
            "When action is BUY or SELL, set target_price to the current price and "
            "keep stop_loss and take_profit aligned with the risk assessment.\n"
            f"Decision inputs:\n{json.dumps(payload, indent=2)}"
        )
        decision = self._require_llm().generate_structured(
            prompt,
            ConsensusDecision,
            context=self._llm_context(symbol, market_context, **kwargs),
        )
        decision.symbol = symbol
        return decision
