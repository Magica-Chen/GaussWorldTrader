"""Render the production session view using synthetic, read-only records.

Run from the repository root:
    python -m streamlit run examples/dashboard_preview.py
"""

import sys
from pathlib import Path

# Streamlit launches scripts with their directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.ui.brand import MARK, apply_brand, wordmark, workspace_header
from src.ui.four_agent_dashboard import render_gauss_session


class PreviewSession:
    """Explicit synthetic fixture; never creates providers or a broker client."""

    def status(self):
        return {
            "account_id": "DEMO-ACCOUNT",
            "environment": "paper",
            "execution_mode": "shadow",
            "data_profile": "FREE_DELAYED",
            "runtime_state": "OBSERVING",
            "supervision": "OFFLINE_FIXTURE",
            "entries_paused": True,
            "signal_as_of": "2026-09-04T19:45:00+00:00",
            "market_as_of": "2026-09-04T19:45:00+00:00",
            "updated_at": "2026-09-04T20:00:00+00:00",
            "last_reconciliation": "2026-09-04T20:00:00+00:00",
            "configured_delay_seconds": 900,
            "readiness": ["OFFLINE_PREVIEW", "ENTRIES_PAUSED"],
            "health": {"reconciliation": "FIXTURE", "market_data": "FIXTURE"},
            "roles": {
                "PostGauss": {"state": "COMPLETE", "completed_at": "2026-09-04 16:10 ET"},
                "CloseGauss": {"state": "COMPLETE", "completed_at": "2026-09-04 17:00 ET"},
                "PreGauss": {"state": "WAITING"},
                "LiveGauss": {"state": "PAUSED"},
            },
            "schedule": {
                "open": "2026-09-08T13:30:00+00:00",
                "close": "2026-09-08T20:00:00+00:00",
                "pre_start": "2026-09-08T12:30:00+00:00",
                "entry_start": "2026-09-08T13:45:00+00:00",
            },
        }

    def health(self):
        return {
            "price_data": "Synthetic delayed observation",
            "news": "Fixture evidence",
            "broker_stream": "Disconnected",
            "reconciliation": "Fixture only",
            "model_workers": "Idle",
            "observed_lag_seconds": 900,
            "feed_quality": "Synthetic demonstration",
        }

    def report(self):
        return {
            "candidates": [{"symbol": "DEMO", "strategy": "momentum", "status": "REVIEW"}],
            "agent_runs": [
                {"role": "PostGauss", "status": "COMPLETE", "source": "synthetic fixture"}
            ],
            "capital_scenarios": [
                {"equity": "25000", "hypothetical": True, "execution_eligible": False}
            ],
            "suitability_reports": [{"feasible": False, "binding_limits": ["Offline fixture"]}],
            "operator_commands": [],
        }

    def account(self):
        return {
            "account_id": "DEMO-ACCOUNT",
            "environment": "paper",
            "equity": "100000",
            "cash": "100000",
            "hypothetical": True,
            "execution_eligible": False,
            "source": "Synthetic demonstration, not an actual account",
        }

    def plans(self):
        return [
            {
                "plan_id": "DEMO-PLAN-01",
                "symbol": "DEMO",
                "state": "RESEARCH",
                "strategy": "momentum",
                "execution_eligible": False,
            }
        ]

    def command(self, *args, **kwargs):
        raise RuntimeError("Offline preview: operator commands are disabled.")


def main():
    st.set_page_config(
        page_title="Gauss World Trader | Offline preview", page_icon=str(MARK), layout="wide"
    )
    apply_brand()
    with st.sidebar:
        wordmark()
        st.caption("WORKSPACE PREVIEW")
        st.radio("Section", ["Gauss Session"], label_visibility="collapsed")
        st.divider()
        st.caption("EXPLORE THE SESSION")
        st.write("Research & plans")
        st.write("Account suitability")
        st.write("Timeline & health")
        st.caption("Use the tabs in the workspace to explore the fixture.")
        st.divider()
        st.caption("Connect your own workspace")
        st.code("python dashboard.py", language="bash")
        st.caption("Market data, analysis, backtests, watchlist, account, orders, and news.")
    workspace_header("Research. Validate. Execute.")
    st.caption("OFFLINE PREVIEW · SYNTHETIC RECORDS · NO BROKER CONNECTED")
    render_gauss_session(PreviewSession())


if __name__ == "__main__":
    main()
