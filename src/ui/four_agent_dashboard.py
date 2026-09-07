"""Read persisted session projections; queue authenticated operator commands only."""

from datetime import datetime
import json
from pathlib import Path

import streamlit as st


def session_client():
    from src.runtime.service import SessionClient
    from src.settings import get_gauss_config

    config = get_gauss_config()
    path = config.database_path
    if not Path(path).is_file():
        raise FileNotFoundError(
            "Session database is unavailable. Start the session service separately."
        )
    return SessionClient(
        path,
        account_id=config.account_id or None,
        environment=config.environment,
    )


def show_records(label, records):
    st.subheader(label)
    if records:
        st.json(records)
    else:
        st.info(f"No {label.lower()} recorded.")


def render_controls(client, status):
    st.caption("Commands are authenticated, recorded, and handled by the separate session service.")
    scope_fields = (
        "account_id",
        "environment",
        "risk_policy_id",
        "data_policy_version",
        "deployment_version",
        "data_profile",
        "approved_strategy_versions",
    )
    current_scope = {key: status.get(key) for key in scope_fields}
    reviewed_scope = st.session_state.get("gauss_reviewed_arming_scope", current_scope)
    with st.form("gauss_operator_command", clear_on_submit=True):
        operator = st.text_input("Operator identity")
        token = st.text_input("Control token", type="password")
        action = st.selectbox(
            "Command",
            [
                "pause_entries",
                "manage_only",
                "cancel_pending_entries",
                "resume_entries",
                "change_profile",
                "flatten",
                "stop",
                "arm_live",
                "reconcile",
                "approve_model_pricing",
                "approve_strategy",
                "adopt_position",
            ],
        )
        reason = st.text_input("Reason")
        profile = st.selectbox(
            "Requested profile for change_profile",
            [
                "FREE_DELAYED",
                "SUBSCRIBED_REALTIME",
            ],
        )
        st.caption("Live arming applies only to this displayed account and these policy versions.")
        st.json(current_scope)
        approval_json = st.text_area(
            "Approval JSON for model pricing, strategy approval or position adoption",
            help="Model pricing requires id, model, input_usd_per_million, output_usd_per_million, verified_by, source_reference and timezone-aware expires_at. Review the document before confirming.",
        )
        confirmation = st.text_input(
            "For flatten, stop, live arming or approvals, type the account ID to confirm"
        )
        submitted = st.form_submit_button("Submit audited command")
    if not submitted:
        st.session_state["gauss_reviewed_arming_scope"] = current_scope
    if submitted:
        if not operator.strip() or not reason.strip() or not token:
            st.error("Operator identity, reason and control token are required.")
            return
        confirmed = bool(status.get("account_id")) and confirmation == status["account_id"]
        if (
            action
            in {
                "flatten",
                "stop",
                "arm_live",
                "approve_model_pricing",
                "approve_strategy",
                "adopt_position",
            }
            and not confirmed
        ):
            st.error("Confirm the account ID before submitting this command.")
            return
        payload = {"reason": reason.strip(), "data_profile": profile}
        if action == "arm_live":
            payload.update(reviewed_scope)
        if action in {"approve_model_pricing", "approve_strategy", "adopt_position"}:
            try:
                document = json.loads(approval_json)
                if action == "approve_model_pricing":
                    from src.runtime.research import ModelPricing

                    document = ModelPricing.model_validate(document).model_dump(mode="json")
                    if document["verified_by"] != operator.strip():
                        raise ValueError("Pricing verifier must match the operator identity")
                    payload["pricing"] = document
                elif action == "approve_strategy":
                    from src.runtime.models import StrategyApproval

                    payload["approval"] = StrategyApproval.model_validate(document).model_dump(
                        mode="json"
                    )
                else:
                    if not isinstance(document, dict):
                        raise ValueError("Position adoption requires a JSON object")
                    payload.update(document)
            except (ValueError, TypeError) as exc:
                st.error(f"Invalid approval document: {exc}")
                return
        try:
            result = client.command(
                action,
                operator=operator.strip(),
                token=token,
                confirmed=confirmed,
                payload=payload,
            )
        except Exception as exc:
            st.error(f"Command rejected: {exc}")
        else:
            st.success("Command recorded. Check its audit status for completion.")
            st.json(result)


def render_gauss_session(client=None):
    st.header("Gauss Session")
    st.caption("Persistent service status • refreshing this view does not start trading or feeds")
    try:
        client = client or session_client()
        status = client.status()
        health = client.health()
        report = client.report()
        account = client.account()
        plans = client.plans()
    except FileNotFoundError:
        st.info("Your session workspace is ready. Start the service to see its activity here.")
        st.markdown("Use the same configuration for the service and dashboard.")
        st.code("python gauss_bot.py --config examples/gauss.free-delayed.example.toml", language="bash")
        st.caption("Explore the interface without credentials with the offline preview:")
        st.code("python -m streamlit run examples/dashboard_preview.py", language="bash")
        return
    except Exception as exc:
        st.error(f"Session service state unavailable: {exc}")
        return

    mode = status.get("execution_mode", "UNKNOWN")
    profile = status.get("data_profile", "UNKNOWN")
    overview = st.columns(4)
    overview[0].metric("Execution mode", str(mode).title())
    overview[1].metric("Environment", str(status.get("environment", "Unknown")).title())
    overview[2].metric("Plans recorded", str(len(plans)))
    overview[3].metric("Entries", "Paused" if status.get("entries_paused") else "Review readiness")
    if str(mode).lower() == "shadow":
        st.info("SHADOW MODE — decisions are observations; broker submission is disabled.")
    else:
        st.warning(
            f"Execution mode: {mode} • account: {status.get('account_id', 'unknown')} "
            f"• environment: {status.get('environment', 'unknown')}"
        )
    st.subheader(f"Data profile: {profile}")
    delayed = profile == "FREE_DELAYED"
    st.caption("Delayed observation" if delayed else "Market observation")
    watermarks = status.get("watermarks") or {}
    try:
        market_as_of = (
            min(
                watermarks.values(),
                key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
            )
            if watermarks
            else status.get("market_as_of")
        )
    except (ValueError, TypeError):
        market_as_of = None
    observed_lag = status.get("observed_lag_seconds", health.get("observed_lag_seconds"))
    if market_as_of and status.get("updated_at"):
        try:
            observed_lag = (
                datetime.fromisoformat(status["updated_at"].replace("Z", "+00:00"))
                - datetime.fromisoformat(market_as_of.replace("Z", "+00:00"))
            ).total_seconds()
        except (ValueError, TypeError):
            observed_lag = None
    capabilities = report.get("data_capabilities") or []
    quality = sorted({str(item.get("quality_class", "unverified")) for item in capabilities})
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Oldest market as-of", "Nominal lag", "Observed lag", "Feed / quality"],
        [
            market_as_of or "Unavailable",
            "15 minutes" if delayed else "Real time requested",
            f"{observed_lag:g} seconds"
            if isinstance(observed_lag, (int, float))
            else "Unavailable",
            ", ".join(quality)
            or status.get("feed_quality", health.get("feed_quality", "Unverified")),
        ],
    ):
        col.caption(label)
        col.write(str(value))
    st.caption(
        f"Information cutoff: {status.get('signal_as_of', 'unavailable')} • "
        f"Service updated: {status.get('updated_at', 'unavailable')}"
    )
    if watermarks:
        with st.expander("Market as-of by symbol"):
            st.json(watermarks)
    st.caption(
        "A deliberate market-information delay does not delay broker orders and positions. "
        "Indicative quotes do not establish genuine OPRA coverage."
    )
    with st.expander("Execution readiness"):
        show_records("Entry checks", status.get("readiness"))
    st.write(
        "Runtime:",
        status.get("runtime_state", "Unknown"),
        "• Entries paused:",
        status.get("entries_paused", "Unknown"),
        "• Supervision:",
        status.get("supervision", "Unknown"),
    )
    if status.get("supervision") in {"ENDED", "NOT_RUNNING"}:
        st.warning("The session service is not supervising exposure. These are persisted records.")
    for col, role in zip(st.columns(4), ["PostGauss", "CloseGauss", "PreGauss", "LiveGauss"]):
        with col:
            st.subheader(role)
            role_status = status.get("roles", {}).get(role, "No run recorded")
            if isinstance(role_status, dict):
                st.write(role_status.get("state", "Unknown"))
                st.caption(str(role_status.get("completed_at", "No completed run")))
                if role_status.get("failure"):
                    st.error(str(role_status["failure"]))
                with st.expander("Run details"):
                    st.json(role_status)
            else:
                st.write(role_status)
    tabs = st.tabs(
        [
            "Timeline & health",
            "Research & plans",
            "Decisions & exposure",
            "Account suitability",
            "Hypothetical capital",
            "Operator controls",
        ]
    )
    with tabs[0]:
        show_records("Session timeline", status.get("schedule") or status.get("session"))
        for key in ["price_data", "news", "broker_stream", "reconciliation", "model_workers"]:
            st.write(key.replace("_", " ").title(), health.get(key, "Unverified / unavailable"))
        show_records("Service health details", health)
        show_records("Feed capabilities and coverage", capabilities)
        show_records("Agent runs", report.get("agent_runs"))
    with tabs[1]:
        st.caption(
            f"{profile} • market as-of {status.get('signal_as_of', 'unavailable')} • "
            "Execution eligibility must be established by the current readiness checks."
        )
        show_records("Candidates", report.get("candidates"))
        show_records("Plans", plans)
        if plans:
            chosen = st.selectbox(
                "Plan detail",
                range(len(plans)),
                format_func=lambda i: str(plans[i].get("plan_id", plans[i].get("id", i))),
            )
            plan = plans[chosen]
            st.json(plan)
            plan_id = plan.get("plan_id", plan.get("id"))
            for key in ["plan_events", "validations", "orders", "fills"]:
                rows = report.get(key) or []
                linked = (
                    [row for row in rows if isinstance(row, dict) and row.get("plan_id") == plan_id]
                    if plan_id is not None
                    else []
                )
                show_records(f"Linked {key.replace('_', ' ')}", linked)
        show_records("Evidence snapshots", report.get("snapshots"))
    with tabs[2]:
        st.caption("Current operational broker state • separate from delayed market evidence")
        st.write("Last reconciliation:", status.get("last_reconciliation", "Unavailable"))
        for key in ["decisions", "orders", "fills", "position_groups", "positions"]:
            show_records(key.replace("_", " ").title(), report.get(key))
        show_records("Trading P&L", report.get("pnl"))
        show_records("Allocated operating costs", report.get("costs"))
        st.caption(
            "Spreads already reflected in fills must not be deducted again as operating costs."
        )
    with tabs[3]:
        st.subheader("Actual account")
        st.json(account)
        show_records("Suitability reports", report.get("suitability_reports"))
        st.caption(
            "Review deployable capital, permissions, sizing limits, rejected alternatives and "
            "costs. Missing account facts block eligibility; no trade is a valid outcome."
        )
    with tabs[4]:
        st.warning("HYPOTHETICAL — execution disabled. These balances are not the actual account.")
        show_records("Capital scenarios", report.get("capital_scenarios"))
    with tabs[5]:
        render_controls(client, status)
        show_records("Operator command audit", report.get("operator_commands"))
        show_records("Command completion and rejection", report.get("command_results"))
        show_records("Registered model pricing", report.get("model_pricing"))
        show_records("Strategy approvals", report.get("strategy_approvals"))
