"""Page 4: Interactive Data Visualization Studio (Power BI + SQL Analytics Parity)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import compute_breakdown, compute_service_adoption
from src.config import PALETTE
from src.ui.charts import (
    chart_dual_churn_volume,
    chart_service_adoption,
    chart_ticket_impact,
)
from src.ui.styles import render_hero


def render(df: pd.DataFrame) -> None:
    """Render Visual Analytics Studio page."""
    render_hero(
        "Interactive Data Visualization Studio",
        "Explore Service Adoption, Billing Friction, Support Tickets, Demographics, and Geographic Behaviour",
    )

    if df.empty:
        st.warning("No customer records match your active sidebar filters.")
        return

    tab1, tab2, tab3 = st.tabs(
        ["Services & Billing Risks", "Support Tickets & Spend Scatter", "Demographics & Geography"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            inet_bd = compute_breakdown(df, "internet_service")
            st.plotly_chart(
                chart_dual_churn_volume(inet_bd, "internet_service", "Churn Rate & Volume by Internet Service"),
                use_container_width=True,
            )
        with col2:
            pay_bd = compute_breakdown(df, "payment_method")
            st.plotly_chart(
                chart_dual_churn_volume(pay_bd, "payment_method", "Churn Rate & Volume by Payment Method"),
                use_container_width=True,
            )

        col3, col4 = st.columns([1.3, 1])
        with col3:
            svc_df = compute_service_adoption(df)
            st.plotly_chart(chart_service_adoption(svc_df), use_container_width=True)
        with col4:
            paper_bd = compute_breakdown(df, "paperless_billing")
            fig_pb = px.pie(
                paper_bd,
                names="paperless_billing",
                values="customers",
                hole=0.55,
                title="Paperless Billing Share & Churn",
                hover_data=["churn_rate_pct"],
                color_discrete_sequence=[PALETTE["primary"], PALETTE["warning"]],
                template="plotly_white",
            )
            fig_pb.update_traces(textinfo="label+percent")
            fig_pb.update_layout(height=380)
            st.plotly_chart(fig_pb, use_container_width=True)

    with tab2:
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(chart_ticket_impact(df), use_container_width=True)
        with c_right:
            sample_df = df.sample(n=min(2000, len(df)), random_state=42) if len(df) > 2000 else df
            fig_scatter = px.scatter(
                sample_df,
                x="tenure",
                y="monthly_charges",
                color="risk_level",
                size="total_tickets",
                hover_data=["customer_id", "customer_name", "contract", "churn_probability"],
                color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F59E0B", "Low Risk": "#10B981"},
                title="Tenure vs Monthly Charges (Bubble Size = Support Tickets)",
                template="plotly_white",
            )
            fig_scatter.update_layout(height=360)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Monthly charges histogram by actual churn
        fig_hist = px.histogram(
            df[df["actual_churn"].isin(["Yes", "No"])],
            x="monthly_charges",
            color="actual_churn",
            barmode="overlay",
            nbins=35,
            title="Monthly Charges Distribution: Retained vs Churned",
            color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
            template="plotly_white",
            opacity=0.72,
        )
        fig_hist.update_layout(height=340)
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        d1, d2 = st.columns(2)
        with d1:
            country_bd = compute_breakdown(df, "country")
            fig_country = px.bar(
                country_bd,
                x="country",
                y="customers",
                color="churn_rate_pct",
                text="customers",
                title="Customer Distribution & Churn Rate by Country",
                color_continuous_scale="Tealrose",
                template="plotly_white",
            )
            fig_country.update_traces(textposition="outside")
            fig_country.update_layout(height=360)
            st.plotly_chart(fig_country, use_container_width=True)
        with d2:
            age_bd = compute_breakdown(df, "age_group")
            fig_age = px.bar(
                age_bd,
                x="age_group",
                y="churn_rate_pct",
                color="avg_risk_pct",
                text="churn_rate_pct",
                title="Churn Rate by Demographic Age Group",
                color_continuous_scale="YlOrRd",
                template="plotly_white",
            )
            fig_age.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_age.update_layout(height=360)
            st.plotly_chart(fig_age, use_container_width=True)

        # Gender, Partner, Dependents summary table
        st.markdown("#### Demographic & Household Stability Indicators")
        demo_rows = []
        for col, label in [("gender", "Gender"), ("senior_citizen", "Senior Citizen"), ("partner", "Has Partner"), ("dependents", "Has Dependents")]:
            bd = compute_breakdown(df, col)
            bd.insert(0, "Dimension", label)
            bd = bd.rename(columns={col: "Category"})
            bd["Category"] = bd["Category"].astype(str)
            demo_rows.append(bd[["Dimension", "Category", "customers", "share_pct", "churn_rate_pct", "avg_risk_pct"]])
        st.dataframe(pd.concat(demo_rows, ignore_index=True), use_container_width=True, hide_index=True)
