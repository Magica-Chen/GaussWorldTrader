"""Offline coverage of the subscription CLI boundary; never invokes real Codex."""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from src.llm import codex_cli
from src.llm.providers import OpenAICodexProvider


@pytest.fixture(autouse=True)
def quota_reader(monkeypatch):
    """The default test path must never inspect the actual account's quota."""
    original = codex_cli.read_codex_usage
    monkeypatch.setattr(
        codex_cli,
        "read_codex_usage",
        lambda **kwargs: codex_cli.CodexUsage(
            (codex_cli.CodexQuotaWindow("codex", "primary", 10, 2000000000, 300),),
            time.time(),
        ),
    )
    return original


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["SUPPORT", "HOLD"]
    source_ids: tuple[str, ...]


def completed_events(**usage):
    return [
        {"type": "thread.started", "thread_id": "fixture"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "fixture"}},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 42,
                "output_tokens": 10,
                "cached_input_tokens": 20,
                **usage,
            },
        },
    ]


def fake_cli(monkeypatch, *, output=None, events=None, error=b"", exit_code=0, missing=False):
    calls = []
    monkeypatch.setattr(codex_cli.shutil, "which", lambda name: "/fixture/bin/codex")

    class Process:
        pid = 123456
        returncode = exit_code

        def __init__(self, command, **kwargs):
            calls.append({"command": command, **kwargs})
            kwargs["stdout"].write(
                b"\n".join(
                    json.dumps(event).encode()
                    for event in (completed_events() if events is None else events)
                )
            )
            kwargs["stdout"].flush()
            kwargs["stderr"].write(error)
            kwargs["stderr"].flush()
            if not missing:
                result = (
                    output
                    if output is not None
                    else '{"decision":"SUPPORT","source_ids":["bar-1"]}'
                )
                Path(command[command.index("--output-last-message") + 1]).write_text(result)
            if "--output-schema" in command:
                calls[-1]["schema"] = json.loads(
                    Path(command[command.index("--output-schema") + 1]).read_text()
                )

        def communicate(self, input=None, timeout=None):
            if input is not None:
                calls[-1]["input"] = input
            return None, None

        def poll(self):
            return self.returncode

    monkeypatch.setattr(codex_cli.subprocess, "Popen", Process)
    monkeypatch.setattr(codex_cli, "_stop", lambda process: calls[-1].update(stopped=True))
    return calls


def test_cli_uses_subscription_auth_and_neutral_environment(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fixture-broker-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-paid-key")
    monkeypatch.setenv("CODEX_API_KEY", "fixture-paid-codex-key")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "fixture-webhook")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "fixture-email-secret")
    monkeypatch.setenv("NODE_OPTIONS", "--require=/untrusted.js")
    monkeypatch.setenv("BASH_ENV", "/untrusted-shell")
    monkeypatch.setenv("CODEX_HOME", "/fixture/codex-login")
    calls = fake_cli(monkeypatch)
    result = codex_cli.run_codex(
        "Only bar-1 is evidence.", output_type=Review, model="fixture-model"
    )
    invocation = calls[0]
    command = invocation["command"]
    assert result.value == Review(decision="SUPPORT", source_ids=("bar-1",))
    assert (result.input_tokens, result.output_tokens, result.cached_input_tokens) == (42, 10, 20)
    assert command[1] == "exec" and command[-1] == "-"
    assert "--ignore-user-config" in command and "--ignore-rules" in command
    assert "--ephemeral" in command and "read-only" in command
    assert 'forced_login_method="chatgpt"' in command
    assert 'approval_policy="never"' in command and 'web_search="disabled"' in command
    assert "mcp_servers={}" in command and "agents.enabled=false" in command
    assert set(codex_cli.DISABLED_FEATURES).issubset(command)
    assert invocation["start_new_session"] and invocation["close_fds"]
    assert not Path(invocation["cwd"]).exists()  # temporary evidence/output removed
    environment = invocation["env"]
    assert environment["CODEX_HOME"] == "/fixture/codex-login"
    assert set(environment) == {
        "PATH",
        "HOME",
        "LANG",
        "TZ",
        "PYTHON_DOTENV_DISABLED",
        "CODEX_HOME",
    }
    assert invocation["input"] == b"Only bar-1 is evidence."
    assert invocation["schema"]["additionalProperties"] is False
    assert invocation["schema"]["required"] == ["decision", "source_ids"]


@pytest.mark.parametrize(
    "output",
    [
        '{"decision":"BUY","source_ids":["bar-1"]}',
        '{"decision":"SUPPORT","source_ids":[3]}',
        '{"decision":"SUPPORT","source_ids":[],"submit_order":true}',
        '{"decision":"SUPPORT","decision":"HOLD","source_ids":[]}',
        '{"decision":"SUPPORT"}',
        '```json\n{"decision":"HOLD","source_ids":[]}\n```',
        '{"decision":"SUPPORT","source_ids":[NaN]}',
        "",
    ],
)
def test_malformed_or_unsafe_final_output_fails_closed(monkeypatch, output):
    fake_cli(monkeypatch, output=output)
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_OUTPUT_INVALID"):
        codex_cli.run_codex("fixture", output_type=Review)


@pytest.mark.parametrize(
    "events,reason",
    [
        ([], "CODEX_RESPONSE_INCOMPLETE"),
        (completed_events(output_tokens=None), "CODEX_USAGE_MISSING"),
        (completed_events(input_tokens=True), "CODEX_USAGE_MISSING"),
        (
            [{"type": "item.started", "item": {"type": "command_execution"}}],
            "CODEX_TOOL_USE_REJECTED",
        ),
        (
            [{"type": "item.completed", "item": {"type": "mcp_tool_call"}}],
            "CODEX_TOOL_USE_REJECTED",
        ),
        (
            [{"type": "turn.failed", "error": {"message": "usage limit reached"}}],
            "CODEX_QUOTA_EXHAUSTED",
        ),
    ],
)
def test_incomplete_turn_usage_and_tool_events_are_rejected(monkeypatch, events, reason):
    fake_cli(monkeypatch, events=events)
    with pytest.raises(codex_cli.CodexCLIError, match=reason):
        codex_cli.run_codex("fixture", output_type=Review)


def test_startup_diagnostics_require_a_successfully_completed_turn(monkeypatch):
    diagnostic = {"type": "item.completed", "item": {"type": "error", "message": "fixture notice"}}
    fake_cli(monkeypatch, events=[diagnostic, *completed_events()])
    assert codex_cli.run_codex("fixture", output_type=Review).value.decision == "SUPPORT"
    fake_cli(monkeypatch, events=[diagnostic])
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_RESPONSE_INCOMPLETE"):
        codex_cli.run_codex("fixture", output_type=Review)


@pytest.mark.parametrize(
    "error,reason",
    [
        (b"401 Unauthorized: token=do-not-log", "CODEX_AUTH_REQUIRED"),
        (b"429 Too many requests", "CODEX_QUOTA_EXHAUSTED"),
        (b"unexpected argument --ignore-rules", "CODEX_VERSION_UNSUPPORTED"),
        (b"model is not supported", "CODEX_MODEL_UNAVAILABLE"),
        (b"unexpected upstream failure secret=hidden", "CODEX_PROCESS_FAILED"),
    ],
)
def test_errors_are_classified_without_returning_raw_diagnostics(monkeypatch, error, reason):
    fake_cli(monkeypatch, error=error, exit_code=1)
    with pytest.raises(codex_cli.CodexCLIError) as raised:
        codex_cli.run_codex("fixture")
    assert str(raised.value) == raised.value.reason == reason


def test_output_and_input_bounds(monkeypatch):
    calls = fake_cli(monkeypatch, output="x" * 101)
    monkeypatch.setattr(codex_cli, "MAX_OUTPUT_BYTES", 100)
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_OUTPUT_TOO_LARGE"):
        codex_cli.run_codex("fixture")
    assert calls[0]["stopped"]
    calls.clear()
    monkeypatch.setattr(codex_cli, "MAX_INPUT_BYTES", 100)
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_INPUT_TOO_LARGE"):
        codex_cli.run_codex("x" * 101)
    assert not calls


def test_no_fallback_when_cli_is_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-use")
    monkeypatch.setattr(codex_cli.shutil, "which", lambda name: None)
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_NOT_INSTALLED"):
        codex_cli.run_codex("fixture")


@pytest.mark.skipif(os.name != "posix", reason="process group cleanup is POSIX")
def test_timeout_kills_cli_and_its_child(tmp_path):
    # A local fixture executable, with no Codex/network/auth access.
    child_file = tmp_path / "child.pid"
    program = tmp_path / "fixture-codex"
    program.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(child_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    program.chmod(0o700)
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_TIMEOUT"):
        codex_cli.run_codex("fixture", executable=str(program), timeout_seconds=0.5)
    assert child_file.exists()
    child = Path("/proc") / child_file.read_text() / "stat"
    # A killed child can briefly remain a zombie until adopted/reaped by init.
    assert not child.exists() or child.read_text().split()[2] == "Z"


def test_provider_records_subscription_usage_without_api_cost(monkeypatch):
    fake_cli(monkeypatch, output="Evidence is incomplete.")
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    provider = OpenAICodexProvider(model="gpt-4")
    response = provider.generate_response_with_metadata("fixture", {"run_id": "run-1"})
    assert provider.api_key == "" and response.text == "Evidence is incomplete."
    assert response.usage.estimated_cost_usd is None
    assert provider.get_usage_summary("run-1")["total_tokens"] == 52


def test_legacy_dataclass_output_is_strictly_validated(monkeypatch):
    @dataclass
    class Decision:
        action: str
        confidence: float

    fake_cli(monkeypatch, output='{"action":"HOLD","confidence":"0.9"}')
    with pytest.raises(ValueError):
        OpenAICodexProvider().generate_structured("fixture", Decision)


@pytest.mark.parametrize(
    "used,remaining,blocked", [(0, 100, False), (94, 6, False), (95, 5, True), (100, 0, True)]
)
def test_authoritative_quota_threshold(used, remaining, blocked):
    usage = codex_cli._quota(
        {
            "rateLimits": {
                "limitId": "codex",
                "primary": {
                    "usedPercent": used,
                    "windowDurationMins": 300,
                    "resetsAt": 2000000000,
                },
            }
        }
    )
    assert usage.remaining_percent == remaining
    assert usage.threshold_reached is blocked
    assert usage.windows[0].resets_at == 2000000000


def test_quota_uses_most_consumed_reported_window():
    usage = codex_cli._quota(
        {
            "rateLimitsByLimitId": {
                "codex": {
                    "primary": {"usedPercent": 3},
                    "secondary": {"usedPercent": 96},
                }
            }
        }
    )
    assert usage.remaining_percent == 4 and usage.threshold_reached


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"rateLimits": {"primary": None}},
        {"rateLimits": {"primary": {"usedPercent": None}}},
        {"rateLimits": {"primary": {"usedPercent": True}}},
        {"rateLimits": {"primary": {"usedPercent": float("nan")}}},
    ],
)
def test_unknown_or_invalid_quotas_never_infer_capacity(result):
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_QUOTA_UNAVAILABLE"):
        codex_cli._quota(result)


def test_quota_threshold_blocks_before_starting_model(monkeypatch):
    calls = fake_cli(monkeypatch)
    limits = codex_cli._quota({"rateLimits": {"primary": {"usedPercent": 95}}})
    monkeypatch.setattr(codex_cli, "read_codex_usage", lambda **kwargs: limits)
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_QUOTA_THRESHOLD_REACHED") as raised:
        codex_cli.run_codex("fixture")
    assert raised.value.usage_limits is limits
    assert calls == []


def test_quota_threshold_after_analysis_discards_decision(monkeypatch):
    calls = fake_cli(monkeypatch)
    readings = iter(
        [
            codex_cli._quota({"rateLimits": {"primary": {"usedPercent": 94}}}),
            codex_cli._quota({"rateLimits": {"primary": {"usedPercent": 95}}}),
        ]
    )
    monkeypatch.setattr(codex_cli, "read_codex_usage", lambda **kwargs: next(readings))
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_QUOTA_THRESHOLD_REACHED") as raised:
        codex_cli.run_codex("fixture", output_type=Review)
    assert raised.value.usage_limits.remaining_percent == 5
    assert len(calls) == 1


def test_failed_analysis_still_checks_if_quota_threshold_was_crossed(monkeypatch):
    fake_cli(monkeypatch, output="invalid")
    readings = iter(
        [
            codex_cli._quota({"rateLimits": {"primary": {"usedPercent": 94}}}),
            codex_cli._quota({"rateLimits": {"primary": {"usedPercent": 96}}}),
        ]
    )
    monkeypatch.setattr(codex_cli, "read_codex_usage", lambda **kwargs: next(readings))
    with pytest.raises(codex_cli.CodexCLIError, match="CODEX_QUOTA_THRESHOLD_REACHED") as raised:
        codex_cli.run_codex("fixture", output_type=Review)
    assert raised.value.usage_limits.remaining_percent == 4


def test_quota_app_server_protocol_is_read_only(tmp_path, quota_reader):
    # Exercise the real JSONL RPC transport against a local, network-free fixture.
    capture = tmp_path / "requests.jsonl"
    program = tmp_path / "fixture-app-server"
    program.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "for line in sys.stdin:\n"
        " request = json.loads(line)\n"
        f" with Path({str(capture)!r}).open('a') as target: target.write(line)\n"
        " method = request['method']\n"
        " if method == 'initialized': continue\n"
        " if method == 'initialize': result = {'userAgent':'fixture'}\n"
        " elif method == 'account/read': result = {'account':{'type':'chatgpt','email':'private'}}\n"
        " elif method == 'account/rateLimits/read': result = {'rateLimits':{'limitId':'codex','primary':{'usedPercent':12,'resetsAt':2000000000,'windowDurationMins':300}}}\n"
        " else: raise RuntimeError('unexpected write RPC')\n"
        " print(json.dumps({'id':request['id'],'result':result}), flush=True)\n"
    )
    program.chmod(0o700)
    usage = quota_reader(executable=str(program), timeout_seconds=3)
    assert usage.remaining_percent == 88
    assert [json.loads(line)["method"] for line in capture.read_text().splitlines()] == [
        "initialize",
        "initialized",
        "account/read",
        "account/rateLimits/read",
    ]
