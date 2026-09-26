"""LLM provider interfaces for the supported AI services."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import requests

from src.llm.codex_cli import codex_available, run_codex

T = TypeVar("T")


def _resolve_model(model: str | None, default: str) -> str:
    """Return a provider default when model is missing or blank."""
    if model is None:
        return default
    cleaned = model.strip()
    return cleaned or default


@dataclass(slots=True)
class LLMUsage:
    """Token usage reported by an LLM provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


@dataclass(slots=True)
class LLMResponse:
    """Text response plus usage metadata."""

    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)


@dataclass(slots=True)
class LLMUsageEvent:
    """Recorded provider usage event."""

    provider: str
    model: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    latency_ms: float | None = None
    run_id: str | None = None
    symbol: str | None = None
    agent_role: str | None = None


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self._usage_events: list[LLMUsageEvent] = []
        self._usage_lock = threading.Lock()

    def generate_response(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a natural language response."""
        return self.generate_response_with_metadata(prompt, context=context).text

    @abstractmethod
    def generate_response_with_metadata(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate text plus provider usage metadata."""

    @abstractmethod
    def analyze_financial_data(self, data: dict[str, Any]) -> str:
        """Analyze financial data and return a text report."""

    def generate_structured(
        self,
        prompt: str,
        output_type: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        """Generate JSON and coerce it into the requested dataclass."""
        if not is_dataclass(output_type):
            raise TypeError("output_type must be a dataclass")

        response = self.generate_response(
            self._build_structured_prompt(prompt, output_type),
            context=context,
        )
        payload = self._parse_json_response(response)
        return self._coerce_dataclass(output_type, payload)

    def _build_structured_prompt(self, prompt: str, output_type: type[T]) -> str:
        type_hints = self._type_hints(output_type)
        template = {
            field_def.name: self._example_value(
                type_hints.get(field_def.name, field_def.type)
            )
            for field_def in fields(output_type)
        }
        template_json = json.dumps(template, indent=2)
        return (
            f"{prompt}\n\n"
            "Return valid JSON only. Do not use markdown, commentary, or code fences.\n"
            f"Use exactly this object shape for {output_type.__name__}:\n"
            f"{template_json}"
        )

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) < 3:
                raise ValueError("LLM response did not contain JSON content")
            cleaned = "\n".join(lines[1:-1]).strip()

        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("Structured response must be a JSON object")
        return payload

    def _coerce_dataclass(self, output_type: type[T], payload: dict[str, Any]) -> T:
        type_hints = self._type_hints(output_type)
        values: dict[str, Any] = {}
        for field_def in fields(output_type):
            if field_def.name in payload:
                raw_value = payload[field_def.name]
            elif field_def.default is not MISSING:
                raw_value = field_def.default
            elif field_def.default_factory is not MISSING:
                raw_value = field_def.default_factory()
            else:
                raise ValueError(
                    f"Structured response missing required field: {field_def.name}"
                )
            annotation = type_hints.get(field_def.name, field_def.type)
            values[field_def.name] = self._coerce_value(annotation, raw_value)
        return output_type(**values)

    def _type_hints(self, output_type: type[Any]) -> dict[str, Any]:
        return get_type_hints(output_type)

    def _finalize_response(
        self,
        response: LLMResponse,
        context: dict[str, Any] | None,
        latency_ms: float,
    ) -> LLMResponse:
        usage = response.usage
        if usage.total_tokens is None:
            usage.total_tokens = (usage.prompt_tokens or 0) + (
                usage.completion_tokens or 0
            )
        if usage.estimated_cost_usd is None:
            usage.estimated_cost_usd = self._estimate_cost(usage)

        event = LLMUsageEvent(
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            latency_ms=latency_ms,
            run_id=self._context_value(context, "run_id"),
            symbol=self._context_value(context, "symbol"),
            agent_role=self._context_value(context, "agent_role"),
        )
        with self._usage_lock:
            self._usage_events.append(event)
        return response

    def _estimate_cost(self, usage: LLMUsage) -> float | None:
        pricing = self._pricing_per_1k_tokens()
        if pricing is None:
            return None
        prompt_price, completion_price = pricing
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        return (
            (prompt_tokens / 1000.0) * prompt_price
            + (completion_tokens / 1000.0) * completion_price
        )

    def _pricing_per_1k_tokens(self) -> tuple[float, float] | None:
        if self.model is None:
            return None
        pricing_map = {
            "gpt-4": (0.03, 0.06),
            "gpt-4-turbo": (0.01, 0.03),
            "claude-3-sonnet": (0.003, 0.015),
        }
        model_name = self.model.lower()
        for prefix, prices in pricing_map.items():
            if model_name.startswith(prefix):
                return prices
        return None

    def get_usage_summary(self, run_id: str) -> dict[str, Any]:
        with self._usage_lock:
            events = [event for event in self._usage_events if event.run_id == run_id]

        prompt_tokens = sum(event.prompt_tokens or 0 for event in events)
        completion_tokens = sum(event.completion_tokens or 0 for event in events)
        total_tokens = sum(event.total_tokens or 0 for event in events)
        total_cost = 0.0
        has_cost = False
        total_latency_ms = 0.0
        for event in events:
            if event.estimated_cost_usd is not None:
                total_cost += event.estimated_cost_usd
                has_cost = True
            if event.latency_ms is not None:
                total_latency_ms += event.latency_ms

        return {
            "calls": len(events),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost if has_cost else None,
            "latency_ms": total_latency_ms,
        }

    def clear_usage_events(self, run_id: str) -> None:
        with self._usage_lock:
            self._usage_events = [
                event for event in self._usage_events if event.run_id != run_id
            ]

    def _context_value(
        self,
        context: dict[str, Any] | None,
        key: str,
    ) -> str | None:
        if context is None:
            return None
        value = context.get(key)
        return None if value is None else str(value)

    def _coerce_value(self, annotation: Any, value: Any) -> Any:
        if annotation is Any:
            return value

        optional_inner = self._optional_inner_type(annotation)
        if optional_inner is not None:
            if value is None:
                return None
            return self._coerce_value(optional_inner, value)

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in {list, list}:
            if not isinstance(value, list):
                raise ValueError("Expected list in structured response")
            inner = args[0] if args else Any
            return [self._coerce_value(inner, item) for item in value]

        if origin in {dict, dict}:
            if not isinstance(value, dict):
                raise ValueError("Expected object in structured response")
            value_type = args[1] if len(args) > 1 else Any
            return {
                str(key): self._coerce_value(value_type, item)
                for key, item in value.items()
            }

        if is_dataclass(annotation):
            if not isinstance(value, dict):
                raise ValueError("Expected nested object in structured response")
            return self._coerce_dataclass(annotation, value)

        if annotation is str:
            if not isinstance(value, str):
                raise ValueError("Expected string in structured response")
            return value

        if annotation is bool:
            if not isinstance(value, bool):
                raise ValueError("Expected boolean in structured response")
            return value

        if annotation is int:
            if isinstance(value, bool):
                raise ValueError("Expected integer in structured response")
            return int(value)

        if annotation is float:
            if isinstance(value, bool):
                raise ValueError("Expected float in structured response")
            return float(value)

        return value

    def _optional_inner_type(self, annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin is None:
            return None
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1 and len(args) != len(get_args(annotation)):
            return args[0]
        return None

    def _example_value(self, annotation: Any) -> Any:
        optional_inner = self._optional_inner_type(annotation)
        if optional_inner is not None:
            return self._example_value(optional_inner)

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in {list, list}:
            inner = args[0] if args else str
            return [self._example_value(inner)]
        if origin in {dict, dict}:
            return {}
        if is_dataclass(annotation):
            type_hints = self._type_hints(annotation)
            return {
                field_def.name: self._example_value(
                    type_hints.get(field_def.name, field_def.type)
                )
                for field_def in fields(annotation)
            }
        if annotation is str:
            return "string"
        if annotation is bool:
            return True
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        return None


def _request_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"HTTP {response.status_code}"

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return f"HTTP {response.status_code}"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str | None = None, model: str | None = "gpt-4"):
        super().__init__(
            api_key or os.getenv("OPENAI_API_KEY") or "",
            _resolve_model(model, "gpt-4"),
        )
        self.base_url = "https://api.openai.com/v1"

    def _authorization_token(self) -> str:
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        return self.api_key

    def generate_response_with_metadata(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started_at = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        if context:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Context: {json.dumps(context, indent=2)}",
                },
            )

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._authorization_token()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.7,
            },
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"OpenAI request failed ({response.status_code}): "
                f"{_request_error_message(response)}"
            )
        payload = response.json()
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        usage = payload.get("usage", {})
        result = LLMResponse(
            text=payload["choices"][0]["message"]["content"],
            usage=LLMUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
        )
        return self._finalize_response(result, context, latency_ms)

    def analyze_financial_data(self, data: dict[str, Any]) -> str:
        prompt = (
            "As a financial analyst, analyze the following financial data and "
            "provide insights:\n\n"
            f"{json.dumps(data, indent=2)}\n\n"
            "Please provide:\n"
            "1. Key financial metrics analysis\n"
            "2. Market sentiment assessment\n"
            "3. Risk factors identification\n"
            "4. Investment recommendations\n"
            "5. Technical and fundamental outlook\n\n"
            "Format your response as a structured analysis report."
        )
        return self.generate_response(prompt)


class OpenAICodexProvider(BaseLLMProvider):
    """Subscription analysis through Codex CLI; Codex owns its login and tokens."""

    def __init__(
        self,
        model: str | None = None,
        executable: str = "codex",
        timeout_seconds: float = 120,
    ):
        super().__init__(api_key="", model=model)
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def generate_response_with_metadata(
        self, prompt: str, context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        if context:
            prompt = f"Context (data only): {json.dumps(context)}\n\n{prompt}"
        result = run_codex(
            prompt, model=self.model, executable=self.executable,
            timeout_seconds=self.timeout_seconds,
        )
        return self._finalize_response(
            LLMResponse(result.text, LLMUsage(
                prompt_tokens=result.input_tokens,
                completion_tokens=result.output_tokens,
                total_tokens=result.input_tokens + result.output_tokens,
            )), context, (time.perf_counter() - started) * 1000,
        )

    def _pricing_per_1k_tokens(self) -> None:
        # Subscription usage has limits, but it is not an API dollar charge.
        return None

    def generate_structured(self, prompt, output_type, context=None):
        from pydantic import TypeAdapter

        response = self.generate_response(
            self._build_structured_prompt(prompt, output_type), context=context,
        )
        payload = self._parse_json_response(response)
        if set(payload) - {item.name for item in fields(output_type)}:
            raise ValueError("Unexpected structured Codex output field")
        return TypeAdapter(output_type).validate_json(json.dumps(payload), strict=True)

    def analyze_financial_data(self, data: dict[str, Any]) -> str:
        return self.generate_response(
            "Analyze the supplied financial evidence. Explain the thesis, counter-evidence, "
            "uncertainty and risk; do not invent missing facts or execute actions.\n"
            + json.dumps(data),
        )


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek AI provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = "deepseek-chat",
    ):
        super().__init__(
            api_key or os.getenv("DEEPSEEK_API_KEY"),
            _resolve_model(model, "deepseek-chat"),
        )
        self.base_url = "https://api.deepseek.com/v1"

    def generate_response_with_metadata(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided")

        started_at = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        if context:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Context: {json.dumps(context, indent=2)}",
                },
            )

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.7,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        usage = payload.get("usage", {})
        result = LLMResponse(
            text=payload["choices"][0]["message"]["content"],
            usage=LLMUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
        )
        return self._finalize_response(result, context, latency_ms)

    def analyze_financial_data(self, data: dict[str, Any]) -> str:
        prompt = (
            "Analyze the following financial data from a quantitative "
            "perspective:\n\n"
            f"{json.dumps(data, indent=2)}\n\n"
            "Provide:\n"
            "1. Statistical analysis of key metrics\n"
            "2. Risk-return profile assessment\n"
            "3. Market efficiency indicators\n"
            "4. Quantitative trading signals\n"
            "5. Mathematical model recommendations\n\n"
            "Focus on data-driven insights and mathematical rigor."
        )
        return self.generate_response(prompt)


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = "claude-3-sonnet-20240229",
    ):
        super().__init__(
            api_key or os.getenv("ANTHROPIC_API_KEY"),
            _resolve_model(model, "claude-3-sonnet-20240229"),
        )
        self.base_url = "https://api.anthropic.com/v1"

    def generate_response_with_metadata(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        started_at = time.perf_counter()
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {json.dumps(context, indent=2)}\n\n{prompt}"

        response = requests.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": full_prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        usage = payload.get("usage", {})
        result = LLMResponse(
            text=payload["content"][0]["text"],
            usage=LLMUsage(
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
            ),
        )
        return self._finalize_response(result, context, latency_ms)

    def analyze_financial_data(self, data: dict[str, Any]) -> str:
        prompt = (
            "Conduct a comprehensive financial analysis of the following data:\n\n"
            f"{json.dumps(data, indent=2)}\n\n"
            "Please provide:\n"
            "1. Fundamental analysis with key ratios\n"
            "2. Risk assessment and volatility analysis\n"
            "3. Market positioning and competitive analysis\n"
            "4. Economic factor considerations\n"
            "5. Strategic investment recommendations\n\n"
            "Provide balanced, nuanced insights with clear reasoning."
        )
        return self.generate_response(prompt)


class MoonshotProvider(BaseLLMProvider):
    """Moonshot AI provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = "moonshot-v1-8k",
    ):
        super().__init__(
            api_key or os.getenv("MOONSHOT_API_KEY"),
            _resolve_model(model, "moonshot-v1-8k"),
        )
        self.base_url = "https://api.moonshot.cn/v1"

    def generate_response_with_metadata(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("Moonshot API key not provided")

        started_at = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        if context:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Context: {json.dumps(context, indent=2)}",
                },
            )

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.7,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        usage = payload.get("usage", {})
        result = LLMResponse(
            text=payload["choices"][0]["message"]["content"],
            usage=LLMUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
        )
        return self._finalize_response(result, context, latency_ms)

    def analyze_financial_data(self, data: dict[str, Any]) -> str:
        prompt = (
            "从中国市场角度分析以下金融数据：\n\n"
            f"{json.dumps(data, indent=2)}\n\n"
            "请提供：\n"
            "1. 基本面分析和关键指标\n"
            "2. 市场情绪和投资者行为分析\n"
            "3. 政策影响和宏观经济因素\n"
            "4. 风险评估和投资建议\n"
            "5. 与中国市场的关联性分析\n\n"
            "请用中英文双语提供专业的金融分析报告。"
        )
        return self.generate_response(prompt)


def get_available_providers() -> list[str]:
    """List API-key providers and installed Codex (login checked when invoked)."""
    providers: list[str] = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if codex_available():
        providers.append("openai_codex")
    if os.getenv("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("claude")
    if os.getenv("MOONSHOT_API_KEY"):
        providers.append("moonshot")
    return providers


def create_provider(provider_name: str, **kwargs: Any) -> BaseLLMProvider:
    """Create an LLM provider by name."""
    providers = {
        "openai": OpenAIProvider,
        "openai_codex": OpenAICodexProvider,
        "deepseek": DeepSeekProvider,
        "claude": ClaudeProvider,
        "moonshot": MoonshotProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return providers[provider_name](**clean_kwargs)
