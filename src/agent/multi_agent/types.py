"""Shared dataclasses for the multi-agent decision system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentReport:
    """Structured report returned by an analyst agent."""

    action: str
    confidence: float
    thesis: str
    agent_name: str = ""
    symbol: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DebatePosition:
    """Bull or bear position generated during the debate step."""

    side: str
    confidence: float
    thesis: str
    symbol: str = ""
    key_points: list[str] = field(default_factory=list)
    rebuttal: str = ""


@dataclass(slots=True)
class RiskAssessment:
    """Risk limits computed from market structure and agent disagreement."""

    symbol: str
    risk_level: str
    max_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    atr: float
    volatility_pct: float
    risk_flags: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class ConsensusDecision:
    """Final portfolio action after synthesis of all agent inputs."""

    action: str
    confidence: float
    reason: str
    symbol: str = ""
    target_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    participating_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)
    debate_summary: str = ""
