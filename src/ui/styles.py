"""Custom CSS styling and reusable HTML metric card components for Streamlit."""

from __future__ import annotations

import html
import streamlit as st


def inject_global_css() -> None:
    """Inject modern SaaS enterprise styling into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Clean responsive spacing */
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1460px;
        }
        /* Top Hero Banner */
        .ci-hero {
            background: linear-gradient(120deg, #0F766E 0%, #0D9488 52%, #2563EB 100%);
            padding: 1.25rem 1.6rem;
            border-radius: 14px;
            color: #FFFFFF;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 118, 110, 0.16);
        }
        .ci-hero h1 {
            margin: 0;
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF;
        }
        .ci-hero p {
            margin: 0.35rem 0 0 0;
            font-size: 0.94rem;
            color: rgba(255, 255, 255, 0.90);
        }
        /* KPI Metric Card */
        .ci-kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.15rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            border-top: 4px solid var(--accent, #0D9488);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            height: 100%;
        }
        .ci-kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }
        .ci-kpi-label {
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
            margin-bottom: 0.35rem;
        }
        .ci-kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.15;
        }
        .ci-kpi-sub {
            font-size: 0.78rem;
            color: #475569;
            margin-top: 0.4rem;
        }
        /* Section panel */
        .ci-panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.35rem;
        }
        .ci-badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .ci-badge-high { background: #FEE2E2; color: #B91C1C; }
        .ci-badge-med  { background: #FEF3C7; color: #B45309; }
        .ci-badge-low  { background: #D1FAE5; color: #047857; }
        .ci-badge-info { background: #DBEAFE; color: #1D4ED8; }
        /* Sidebar polish */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }
        .ci-sidebar-status {
            background: #F1F5F9;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            font-size: 0.80rem;
            color: #334155;
            margin-top: 0.5rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    """Render top page hero banner."""
    st.markdown(
        f"""
        <div class="ci-hero">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str = "", accent: str = "#0D9488") -> None:
    """Render a styled KPI metric card."""
    st.markdown(
        f"""
        <div class="ci-kpi-card" style="--accent: {accent};">
            <div class="ci-kpi-label">{html.escape(label)}</div>
            <div class="ci-kpi-value">{html.escape(value)}</div>
            <div class="ci-kpi-sub">{html.escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(risk_level: str) -> str:
    """Return colored HTML pill badge for a risk level."""
    cls = "ci-badge-low"
    if risk_level == "High Risk":
        cls = "ci-badge-high"
    elif risk_level == "Medium Risk":
        cls = "ci-badge-med"
    return f'<span class="ci-badge {cls}">{html.escape(risk_level)}</span>'
