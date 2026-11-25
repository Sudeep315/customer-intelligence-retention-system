
"""Main Streamlit entry point for the Customer Intelligence & Retention System."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHURN_MODEL_PATH, ENRICHED_CUSTOMERS_CSV, get_db_settings
from src.database import fetch_customers, get_cached_connection, reset_db_cache, test_mysql_connection
from src.ui.pages import (
    ai_recommendations,
    churn_prediction,
    customer_management,
    overview,
    segmentation,
    visual_analytics,
)
from src.ui.styles import inject_global_css

@st.cache_data(ttl=120, show_spinner=False)
def load_customer_data() -> pd.DataFrame:
    """Load customer intelligence records from the active database."""
    engine, _, _ = get_cached_connection()
    df = fetch_customers(engine)
    if df.empty and ENRICHED_CUSTOMERS_CSV.exists():
        df = pd.read_csv(ENRICHED_CUSTOMERS_CSV)
    return df


def _ensure_bootstrapped() -> None:
    """Automatically initialize ML artifacts and DB if running for the very first time."""
    if not CHURN_MODEL_PATH.exists() or not ENRICHED_CUSTOMERS_CSV.exists():
        with st.spinner("First-time setup: training ML models and seeding database..."):
            from scripts.init_project import run_initialization

            run_initialization(replace_db=True)


def main() -> None:
    st.set_page_config(
        page_title="Customer Intelligence & Retention System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()
    _ensure_bootstrapped()

    engine, backend, backend_msg = get_cached_connection()
    df = load_customer_data()

    # ------------------------------------------------------------------
    # Sidebar Navigation, Global Filters, and DB Connection Status
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Customer Intelligence")
        st.caption("Churn Prediction • Segmentation • Retention AI")

        page = st.radio(
            "Navigation",
            [
                "Executive Overview",
                "Churn Prediction Lab",
                "Segmentation & SQL Analytics",
                "Visual Analytics Studio",
                "Customer Management (CRUD)",
                "AI Retention Advisor",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("#### Global Cohort Filters")

        contract_opts = sorted(df["contract"].dropna().unique().tolist()) if not df.empty else []
        sel_contracts = st.multiselect("Contract Type", contract_opts, default=contract_opts)

        inet_opts = sorted(df["internet_service"].dropna().unique().tolist()) if not df.empty else []
        sel_internet = st.multiselect("Internet Service", inet_opts, default=inet_opts)

        risk_opts = ["High Risk", "Medium Risk", "Low Risk"]
        sel_risks = st.multiselect("Risk Tier", risk_opts, default=risk_opts)

        seg_opts = sorted(df["cluster_segment"].dropna().unique().tolist()) if not df.empty else []
        sel_segments = st.multiselect("ML Segment", seg_opts, default=seg_opts)

        max_tenure = int(df["tenure"].max()) if not df.empty else 72
        tenure_range = st.slider("Tenure Range (Months)", 0, max(max_tenure, 72), (0, max(max_tenure, 72)))

        st.divider()
        st.markdown(
            f"""
            <div class="ci-sidebar-status">
                <strong>Database Backend:</strong> {backend}<br/>
                <strong>Active Records:</strong> {len(df):,} customers<br/>
                <span style="color:#64748B;">{backend_msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("MySQL Connection Diagnostics"):
            settings = get_db_settings()
            m_host = st.text_input("MySQL Host", value=settings.mysql_host)
            m_port = st.number_input("Port", value=settings.mysql_port, step=1)
            m_user = st.text_input("User", value=settings.mysql_user)
            m_pass = st.text_input("Password", value=settings.mysql_password, type="password")
            m_db = st.text_input("Database", value=settings.mysql_database)
            if st.button("Test MySQL Connection", use_container_width=True):
                ok, msg = test_mysql_connection(m_host, int(m_port), m_user, m_pass, m_db)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)
            if st.button("Refresh Data Cache", use_container_width=True):
                reset_db_cache()
                st.cache_data.clear()
                st.rerun()

    # Apply global filters safely
    filtered_df = df.copy()
    if not filtered_df.empty:
        if sel_contracts:
            filtered_df = filtered_df[filtered_df["contract"].isin(sel_contracts)]
        if sel_internet:
            filtered_df = filtered_df[filtered_df["internet_service"].isin(sel_internet)]
        if sel_risks:
            filtered_df = filtered_df[filtered_df["risk_level"].isin(sel_risks)]
        if sel_segments:
            filtered_df = filtered_df[filtered_df["cluster_segment"].isin(sel_segments)]
        filtered_df = filtered_df[
            (filtered_df["tenure"] >= tenure_range[0]) & (filtered_df["tenure"] <= tenure_range[1])
        ]

    # Route to selected page
    if page == "Executive Overview":
        overview.render(filtered_df)
    elif page == "Churn Prediction Lab":
        churn_prediction.render(filtered_df)
    elif page == "Segmentation & SQL Analytics":
        segmentation.render(filtered_df)
    elif page == "Visual Analytics Studio":
        visual_analytics.render(filtered_df)
    elif page == "Customer Management (CRUD)":
        customer_management.render(filtered_df)
    elif page == "AI Retention Advisor":
        ai_recommendations.render(filtered_df)


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        sys.exit(stcli.main())
