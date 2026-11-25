"""Page 1: Executive Customer Intelligence Dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics import (
    compute_breakdown,
    compute_executive_kpis,
    compute_tenure_cohorts,
    generate_business_insights,
)
from src.ui.charts import (
    chart_dual_churn_volume,
    chart_risk_donut,
    chart_tenure_cohorts,
    chart_value_tier_breakdown,
)
from src.ui.styles import render_hero, render_kpi_card


def render(df: pd.DataFrame) -> None:
    """Render Executive Overview dashboard page."""
    render_hero(
        "Executive Customer Intelligence Dashboard",
        "Unified Churn Risk Telemetry, Revenue Exposure, Customer Value Tiers, and Live Business Insights",
    )

    kpis = compute_executive_kpis(df)

    # Row 1: Primary KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card(
            "Total Customers",
            f"{kpis['total_customers']:,}",
            f"Avg tenure: {kpis['avg_tenure']} mos",
            accent="#0D9488",
        )
    with c2:
        render_kpi_card(
            "Historical Churn Rate",
            f"{kpis['actual_churn_rate']:.2f}%",
            f"{kpis['churned_customers']:,} churned accounts",
            accent="#EF4444",
        )
    with c3:
        render_kpi_card(
            "High-Risk Predicted",
            f"{kpis['high_risk_count']:,}",
            f"Avg predicted risk: {kpis['predicted_risk_rate']:.1f}%",
            accent="#F59E0B",
        )
    with c4:
        render_kpi_card(
            "Lifetime Revenue",
            f"${kpis['lifetime_charges'] / 1e6:.2f}M",
            f"Monthly MRR: ${kpis['mrr']:,.0f}",
            accent="#2563EB",
        )
    with c5:
        render_kpi_card(
            "Annual Revenue at Risk",
            f"${kpis['revenue_at_risk']:,.0f}",
            f"Tickets: {kpis['tech_tickets']:,} Tech / {kpis['admin_tickets']:,} Admin",
            accent="#6366F1",
        )

    st.markdown("")

    # Automated Business Insights Banner
    with st.container(border=True):
        st.markdown("#### Automated Executive Insights")
        insights = generate_business_insights(df)
        cols = st.columns(2)
        for idx, bullet in enumerate(insights):
            cols[idx % 2].markdown(f"- {bullet}")

    # Primary Chart Grid
    left, right = st.columns([1.35, 1])
    with left:
        contract_bd = compute_breakdown(df, "contract")
        if not contract_bd.empty:
            st.plotly_chart(
                chart_dual_churn_volume(contract_bd, "contract", "Churn Rate & Volume by Contract Type"),
                use_container_width=True,
            )
    with right:
        if not df.empty:
            st.plotly_chart(chart_risk_donut(df), use_container_width=True)

    left2, right2 = st.columns([1.35, 1])
    with left2:
        cohorts = compute_tenure_cohorts(df)
        if not cohorts.empty:
            st.plotly_chart(chart_tenure_cohorts(cohorts), use_container_width=True)
    with right2:
        if not df.empty:
            st.plotly_chart(chart_value_tier_breakdown(df), use_container_width=True)

    # Priority Watchlist Table
    st.markdown("#### Top High-Value At-Risk Watchlist")
    if df.empty:
        st.info("No records available under current filters.")
        return

    watchlist = df.copy()
    watchlist["annual_risk_exposure"] = (watchlist["annual_spend"] * watchlist["churn_probability"]).round(2)
    watchlist = watchlist.sort_values(["annual_risk_exposure", "churn_probability"], ascending=False).head(10)
    display_cols = [
        "customer_id",
        "customer_name",
        "country",
        "contract",
        "tenure",
        "monthly_charges",
        "num_tech_tickets",
        "cluster_segment",
        "value_tier",
        "churn_probability",
        "risk_level",
        "annual_risk_exposure",
    ]
    st.dataframe(
        watchlist[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Risk", format="%.1f%%", min_value=0.0, max_value=1.0
            ),
            "monthly_charges": st.column_config.NumberColumn("Monthly ($)", format="$%.2f"),
            "annual_risk_exposure": st.column_config.NumberColumn("Annual Risk ($)", format="$%.0f"),
        },
    )
