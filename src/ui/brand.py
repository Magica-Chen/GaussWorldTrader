"""Shared visual identity for the Streamlit workspace."""

from base64 import b64encode
from html import escape
from pathlib import Path

import plotly.io as pio
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
MARK = ASSETS / "gauss-mark.png"


def apply_brand():
    st.html(f"<style>{(ASSETS / 'brand.css').read_text()}</style>")
    pio.templates["gauss"] = pio.templates["plotly_white"]
    pio.templates["gauss"].layout.update(
        font={"family": "Inter, Arial, sans-serif", "color": "#183448"},
        colorway=["#237f80", "#183448", "#b58950", "#748ca3", "#b95f61"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 24, "r": 24, "t": 55, "b": 30},
        xaxis={"gridcolor": "#e7ecee"},
        yaxis={"gridcolor": "#e7ecee"},
    )
    pio.templates.default = "gauss"


def wordmark():
    encoded = b64encode(MARK.read_bytes()).decode("ascii")
    st.html(
        '<div class="gauss-wordmark">'
        f'<img src="data:image/png;base64,{encoded}" alt="Gaussian curve brand mark">'
        "<div><strong>Gauss</strong><span>WORLD TRADER</span></div></div>"
    )


def workspace_header(subtitle="Research. Validate. Execute."):
    st.html(
        '<div class="gauss-masthead"><div><p class="gauss-eyebrow">YOUR MARKET WORKSPACE</p>'
        "<h1>Clarity in every decision.</h1>"
        f'<p>{escape(subtitle)}</p></div><span class="gauss-label">GAUSS WORLD TRADER</span></div>'
    )
