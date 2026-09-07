#!/usr/bin/env python3
"""Continuous calendar-aware stock/options session service entry point."""

from __future__ import annotations

import argparse
import json
import os


def parser():
    result = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Research once: --once\nSession logs:  --output text | --output json",
        description=(
            "GAUSS WORLD TRADER | Session service\n\n"
            "Run PostGauss, CloseGauss, PreGauss and LiveGauss in one persistent service. "
            "Default: paper account, shadow execution, FREE_DELAYED information."
        )
    )
    result.add_argument("--config", help="Existing TOML configuration file")
    result.add_argument(
        "--output",
        choices=["auto", "text", "json"],
        default="auto",
        help="Readable terminal status or JSON logs (auto detects a terminal)",
    )
    result.add_argument("--mode", choices=["shadow", "paper", "live", "replay"])
    result.add_argument("--data-profile", choices=["FREE_DELAYED", "SUBSCRIBED_REALTIME"])
    result.add_argument(
        "--symbols", help="Comma-separated stock underlyings; holdings remain monitored"
    )
    result.add_argument(
        "--once",
        action="store_true",
        help="Scan the market, write a next-session research report, and exit without monitoring",
    )
    result.add_argument(
        "--task",
        choices=["post", "close", "pre", "live"],
        help="Queue an authenticated role request; time and data gates still apply",
    )
    result.add_argument(
        "--acknowledge-unmanaged-exposure",
        action="store_true",
        help="Explicitly allow shutdown to end supervision with reported residual exposure",
    )
    result.add_argument("--operator", default="local")
    result.add_argument("--reason", help="Audit reason for a role request")
    return result


def load_configuration(args):
    from src.runtime.models import RuntimeConfig
    from src.settings import get_gauss_config

    values = get_gauss_config(args.config).model_dump()
    if args.mode is not None:
        values["execution_mode"] = args.mode
    if args.data_profile is not None:
        values["data_profile"] = args.data_profile
    if args.symbols is not None:
        symbols = tuple(
            dict.fromkeys(
                value.strip().upper() for value in args.symbols.split(",") if value.strip()
            )
        )
        if not symbols:
            raise ValueError("--symbols requires at least one stock underlying")
        if any("/" in value for value in symbols):
            raise ValueError("The four-agent bot accepts stock underlyings, not crypto pairs")
        values["symbols"] = symbols
    return RuntimeConfig.model_validate(values)


def main(argv=None, *, service_factory=None):
    args = parser().parse_args(argv)
    config = load_configuration(args)
    if args.task:
        if args.once or args.mode or args.data_profile or args.symbols:
            raise ValueError(
                "Role requests use the running service configuration; omit run overrides"
            )
        from src.runtime.cli import normalize_role
        from src.runtime.service import SessionClient

        if not args.reason or not args.reason.strip() or not args.operator.strip():
            raise ValueError("Role requests require an operator and --reason")
        token = os.environ.get(config.control_token_env)
        if not token:
            raise ValueError(f"Set {config.control_token_env} for authenticated role requests")
        client = SessionClient(
            config.database_path,
            account_id=config.account_id or None,
            environment=config.environment,
        )
        result = client.command(
            "run_task",
            operator=args.operator,
            token=token,
            payload={"role": normalize_role(args.task), "reason": args.reason},
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.once:
        from src.runtime.screening import render_report, run_research

        report, path = run_research(config)
        if args.output == "json":
            print(json.dumps({"report_path": str(path), **report}, default=str))
        else:
            from rich.markdown import Markdown
            from src.utils.branding import banner, make_console

            console = make_console()
            banner(console, "Market research", "Completed-session evidence · Next-session watchlist")
            console.print(Markdown(render_report(report)))
        return 0
    if service_factory is None:
        from src.runtime.service import build_service

        service_factory = build_service
    from src.runtime.runner import run_service

    return run_service(
        service_factory(config),
        once=args.once,
        acknowledge_unmanaged_exposure=args.acknowledge_unmanaged_exposure,
        output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
