"""Session CLI: read projections or send authenticated commands to the service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib

import typer

app = typer.Typer(
    help="Persistent four-agent sessions. Shadow execution is the default.", add_completion=False
)


def emit(value, format="json"):
    rendered = json.dumps(value, indent=2, default=str)
    typer.echo(f"# Gauss Session\n\n```json\n{rendered}\n```" if format == "markdown" else rendered)


def configuration(path=None, mode=None, data_profile=None):
    from src.settings import get_gauss_config
    from src.runtime.models import RuntimeConfig

    config = get_gauss_config(path)
    values = config.model_dump()
    if mode is not None:
        values["execution_mode"] = mode
    if data_profile is not None:
        values["data_profile"] = data_profile
    return RuntimeConfig.model_validate(values)


def client_for(path=None, data_profile=None):
    from src.runtime.service import SessionClient

    config = configuration(path, data_profile=data_profile)
    if not Path(config.database_path).is_file():
        raise typer.BadParameter("Session database does not exist; start the service separately.")
    return SessionClient(
        config.database_path, account_id=config.account_id or None, environment=config.environment
    )


@app.command("validate-config")
def validate_config(config: Path | None = typer.Option(None)):
    emit(configuration(config).model_dump(mode="json"))


@app.command()
def doctor(config: Path | None = typer.Option(None)):
    from src.runtime.service import doctor as probe

    emit(probe(configuration(config)))


@app.command()
def run(
    config: Path | None = typer.Option(None),
    mode: str | None = typer.Option(None),
    data_profile: str | None = typer.Option(None),
    once: bool = typer.Option(False),
    task: str | None = typer.Option(None),
    reason: str = typer.Option("requested role run"),
    operator: str = typer.Option("local"),
):
    """Start the separate session service, or queue a gated role task."""
    if task:
        send_command("run_task", config, operator, reason, payload={"role": normalize_role(task)})
        return
    from src.runtime.service import build_service

    service = build_service(configuration(config, mode, data_profile))
    from src.runtime.runner import run_service

    code = run_service(service, once=once)
    if code:
        raise typer.Exit(code)


def normalize_role(role):
    roles = {"post": "PostGauss", "close": "CloseGauss", "pre": "PreGauss", "live": "LiveGauss"}
    if role in roles.values():
        return role
    if role not in roles:
        raise typer.BadParameter("Task must be post, close, pre, live or its full Gauss role name")
    return roles[role]


@app.command("run-task")
def run_task(
    task: str = typer.Option(...),
    config: Path | None = typer.Option(None),
    operator: str = typer.Option("local"),
    reason: str = typer.Option(...),
):
    send_command("run_task", config, operator, reason, payload={"role": normalize_role(task)})


@app.command()
def status(config: Path | None = typer.Option(None)):
    emit(client_for(config).status())


@app.command()
def plans(config: Path | None = typer.Option(None), session: str = typer.Option("latest")):
    client = client_for(config)
    emit(client.plans() if session == "latest" else client.report(session).get("plans", []))


@app.command("assess-account")
def assess_account(config: Path | None = typer.Option(None)):
    client = client_for(config)
    emit(
        {
            "account": client.account(),
            "suitability_reports": client.report().get("suitability_reports", []),
            "order_submission": False,
        }
    )


def read_document(path):
    with Path(path).open("rb") as source:
        return tomllib.load(source) if Path(path).suffix == ".toml" else json.load(source)


@app.command("compare-capital")
def compare_capital(
    scenarios: Path = typer.Option(..., exists=True),
    config: Path | None = typer.Option(None),
    data_profile: str | None = typer.Option(None),
):
    values = read_document(scenarios)
    values = values.get("scenarios") if isinstance(values, dict) else values
    if not isinstance(values, list) or not values:
        raise typer.BadParameter("Scenarios must be an explicit nonempty list")
    emit(client_for(config, data_profile).scenarios(values, data_profile=data_profile))


def send_command(action, config, operator, reason, confirmed=False, payload=None):
    if not reason.strip() or not operator.strip():
        raise typer.BadParameter("Operator and reason must be nonempty")
    cfg = configuration(config)
    token = os.environ.get(cfg.control_token_env)
    if not token:
        raise typer.BadParameter(f"Set {cfg.control_token_env} for authenticated operator commands")
    emit(
        client_for(config).command(
            action,
            operator=operator,
            token=token,
            confirmed=confirmed,
            payload={**(payload or {}), "reason": reason},
        )
    )


def register_command(name, action):
    def command(
        config: Path | None = typer.Option(None),
        operator: str = typer.Option("local"),
        reason: str = typer.Option(...),
    ):
        send_command(action, config, operator, reason)

    app.command(name)(command)


for name in [
    "pause-entries",
    "resume-entries",
    "cancel-pending-entries",
    "manage-only",
    "reconcile",
]:
    register_command(name, name.replace("-", "_"))


@app.command("request-flatten")
def request_flatten(
    config: Path | None = typer.Option(None),
    operator: str = typer.Option("local"),
    reason: str = typer.Option(...),
    confirm_account: str = typer.Option(...),
):
    actual = client_for(config).status().get("account_id")
    if not actual or confirm_account != actual:
        raise typer.BadParameter("--confirm-account must match the actual session account ID")
    send_command("flatten", config, operator, reason, confirmed=True)


@app.command()
def report(
    config: Path | None = typer.Option(None),
    session: str = typer.Option("latest"),
    format: str = typer.Option("json"),
):
    if format not in {"json", "markdown"}:
        raise typer.BadParameter("Format must be json or markdown")
    emit(client_for(config).report(None if session == "latest" else session), format)


@app.command()
def replay(fixture: Path = typer.Option(..., exists=True)):
    from src.runtime.service import replay_fixture

    emit(replay_fixture(fixture))


@app.command()
def backup(
    destination: Path = typer.Option(...),
    config: Path | None = typer.Option(None),
    evidence_directory: Path | None = typer.Option(None),
):
    from src.runtime.operations import backup_state

    emit(
        backup_state(
            configuration(config).database_path, destination, evidence_directory=evidence_directory
        )
    )


@app.command("verify-backup")
def verify_backup(backup_directory: Path = typer.Option(..., exists=True)):
    from src.runtime.operations import verify_backup as verify

    emit(verify(backup_directory))


@app.command()
def restore(
    backup_directory: Path = typer.Option(..., exists=True), destination: Path = typer.Option(...)
):
    """Restore verified state to a new destination; never overwrite the running database."""
    from src.runtime.operations import restore_state

    emit(restore_state(backup_directory, destination))


@app.command()
def evaluate(matrix: Path = typer.Option(..., exists=True)):
    from src.runtime.evaluation import EvaluationSample, compare_evaluations

    document = read_document(matrix)
    rows = document.get("samples") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise typer.BadParameter("Provide a nonempty samples list")
    emit(compare_evaluations([EvaluationSample.model_validate(row) for row in rows]))


@app.command("change-profile")
def change_profile(
    data_profile: str = typer.Option(...),
    config: Path | None = typer.Option(None),
    operator: str = typer.Option("local"),
    reason: str = typer.Option(...),
):
    from src.runtime.models import DataProfile

    profile = DataProfile(data_profile)
    send_command(
        "change_profile", config, operator, reason, payload={"data_profile": profile.value}
    )


@app.command()
def stop(
    config: Path | None = typer.Option(None),
    operator: str = typer.Option("local"),
    reason: str = typer.Option(...),
    confirm_account: str = typer.Option(...),
):
    actual = client_for(config).status().get("account_id")
    if not actual or confirm_account != actual:
        raise typer.BadParameter("--confirm-account must acknowledge the supervised account")
    send_command("stop", config, operator, reason, confirmed=True)


@app.command("arm-live")
def arm_live(
    config: Path | None = typer.Option(None),
    operator: str = typer.Option("local"),
    reason: str = typer.Option(...),
    confirm_account: str = typer.Option(...),
    risk_policy_id: str = typer.Option(...),
    data_policy_version: str = typer.Option(...),
    deployment_version: str = typer.Option(...),
):
    """Explicitly bind live arming to the account and reviewed policy/deployment versions."""
    status = client_for(config).status()
    if not status.get("account_id") or confirm_account != status["account_id"]:
        raise typer.BadParameter("--confirm-account must match the session account ID")
    emit(
        {
            "arming_scope": status.get("approved_strategy_versions", []),
            "data_profile": status.get("data_profile"),
        }
    )
    send_command(
        "arm_live",
        config,
        operator,
        reason,
        confirmed=True,
        payload={
            "account_id": confirm_account,
            "environment": status.get("environment"),
            "risk_policy_id": risk_policy_id,
            "data_policy_version": data_policy_version,
            "deployment_version": deployment_version,
            "data_profile": status.get("data_profile"),
            "approved_strategy_versions": status.get("approved_strategy_versions", []),
        },
    )


@app.command("approve-model-pricing")
def approve_model_pricing(
    pricing: Path = typer.Option(..., exists=True),
    config: Path | None = typer.Option(None),
    operator: str = typer.Option("local"),
    reason: str = typer.Option(""),
    confirm_account: str = typer.Option(""),
    preview: bool = typer.Option(False),
):
    """Inspect or register an operator-verified, expiring pricing document."""
    from src.runtime.research import ModelPricing

    document = ModelPricing.model_validate(read_document(pricing))
    emit(document.model_dump(mode="json"))
    if preview:
        return
    if document.verified_by != operator:
        raise typer.BadParameter("Pricing verified_by must match --operator")
    actual = client_for(config).status().get("account_id")
    if not actual or confirm_account != actual:
        raise typer.BadParameter("--confirm-account must match the pricing registration account")
    send_command(
        "approve_model_pricing",
        config,
        operator,
        reason,
        confirmed=True,
        payload={"pricing": document.model_dump(mode="json")},
    )


@app.command("model-pricing")
def model_pricing(config: Path | None = typer.Option(None)):
    emit(client_for(config).report().get("model_pricing", []))
