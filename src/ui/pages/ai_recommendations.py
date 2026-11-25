"""Page 6: AI Retention Strategy Advisor & Intervention Campaign Tracker."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PALETTE
from src.database import (
    fetch_retention_actions,
    get_cached_connection,
    log_retention_action,
    update_retention_status,
)
from src.ml_churn import explain_customer_churn
from src.recommender import (
    generate_segment_playbooks,
    optional_llm_refine_strategy,
    recommend_for_customer,
)
from src.ui.styles import render_hero, render_kpi_card, risk_badge_html


def render(df: pd.DataFrame) -> None:
    """Render AI Retention Recommendations & Campaign Tracker page."""
    render_hero(
        "AI Retention Advisor & Campaign Tracker",
        "Actionable Retention Playbooks, Revenue-at-Risk Prioritization, Outreach Scripts, and Live DB Intervention Logs",
    )

    engine, _, _ = get_cached_connection()
    actions_df = fetch_retention_actions(engine)

    high_risk_df = df[df["risk_level"].isin(["High Risk", "Medium Risk"])].copy()
    high_risk_df["annual_risk"] = high_risk_df["annual_spend"] * high_risk_df["churn_probability"]
    total_rev_risk = float(high_risk_df["annual_risk"].sum()) if not high_risk_df.empty else 0.0
    est_recoverable = total_rev_risk * 0.45
    applied_cnt = int((actions_df["status"] == "Applied").sum()) if not actions_df.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("At-Risk Cohort", f"{len(high_risk_df):,}", "High + Medium risk accounts", accent="#EF4444")
    with k2:
        render_kpi_card("Annual Revenue at Risk", f"${total_rev_risk:,.0f}", "Weighted by churn probability", accent="#F59E0B")
    with k3:
        render_kpi_card("Est. Saveable Revenue", f"${est_recoverable:,.0f}", "At ~45% avg save rate", accent="#10B981")
    with k4:
        render_kpi_card("Logged Campaigns", f"{len(actions_df):,}", f"{applied_cnt} interventions applied", accent="#2563EB")

    st.markdown("")
    tab_indiv, tab_seg, tab_tracker = st.tabs(
        ["Individual Retention Studio", "Segment-Level Playbooks", "Retention Campaign Tracker"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Individual Retention Action & Email Generator
    # ------------------------------------------------------------------
    with tab_indiv:
        queue_df = df.copy()
        queue_df["annual_risk"] = queue_df["annual_spend"] * queue_df["churn_probability"]
        queue_df = queue_df.sort_values(["annual_risk", "churn_probability"], ascending=False)

        if queue_df.empty:
            st.info("No customers available.")
        else:
            options = [
                f"{r.customer_id} — {r.customer_name} | Risk: {r.churn_probability*100:.0f}% (${r.annual_risk:,.0f}/yr)"
                for r in queue_df.head(200).itertuples()
            ]
            selected = st.selectbox("Select Priority Customer for AI Retention Plan", options)
            cid = selected.split(" — ")[0]
            row_dict = queue_df[queue_df["customer_id"] == cid].iloc[0].to_dict()
            rec = recommend_for_customer(row_dict)
            drivers = explain_customer_churn(row_dict)

            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                with st.container(border=True):
                    st.markdown(f"#### {rec['customer_name']} (`{rec['customer_id']}`)")
                    st.markdown(
                        f"Risk: {risk_badge_html(rec['risk_level'])} &nbsp;|&nbsp; "
                        f"Priority: **{rec['priority']}** &nbsp;|&nbsp; Segment: **{rec['segment']}**",
                        unsafe_allow_html=True,
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Churn Probability", f"{rec['churn_probability']*100:.1f}%")
                    m2.metric("Revenue at Risk", f"${rec['revenue_at_risk']:,.0f}")
                    m3.metric("Est. Saved Revenue", f"${rec['expected_saved_revenue']:,.0f}", f"{rec['estimated_save_rate']*100:.0f}% save rate")

                    st.markdown(f"**Primary AI Strategy:** `{rec['recommended_strategy']}`")
                    st.markdown("**Tailored Action Plan:**")
                    for t in rec["tactics"]:
                        st.markdown(f"- {t}")

                    if drivers["risk_drivers"]:
                        st.markdown("**Detected Churn Triggers:**")
                        for d in drivers["risk_drivers"]:
                            st.markdown(f"- :red[{d}]")

            with c_right:
                with st.container(border=True):
                    st.markdown("#### Personalized Retention Outreach Email")
                    st.text_input("Email Subject", value=rec["email_subject"], key=f"subj_{cid}")
                    custom_body = st.text_area("Email Body (Editable)", value=rec["email_body"], height=210, key=f"body_{cid}")

                    btn_col1, btn_col2 = st.columns(2)
                    if btn_col1.button("Log Action to Database", type="primary", use_container_width=True):
                        log_retention_action(
                            engine,
                            {
                                "customer_id": rec["customer_id"],
                                "risk_level": rec["risk_level"],
                                "churn_probability": rec["churn_probability"],
                                "recommended_strategy": rec["recommended_strategy"],
                                "priority": rec["priority"],
                                "estimated_save_rate": rec["estimated_save_rate"],
                                "revenue_at_risk": rec["revenue_at_risk"],
                                "action_notes": rec["action_notes"],
                                "status": "In Progress",
                            },
                        )
                        st.cache_data.clear()
                        st.success(f"Logged retention campaign for `{cid}` into database!")

                    if btn_col2.button("Generate LLM Executive Brief", use_container_width=True):
                        llm_text = optional_llm_refine_strategy(rec)
                        if llm_text:
                            st.info(llm_text)
                        else:
                            st.info(
                                "Using built-in deterministic AI Engine (add `OPENAI_API_KEY` in `.env` to enable optional GPT call-script generation)."
                            )

    # ------------------------------------------------------------------
    # TAB 2: Segment-Level Retention Playbooks
    # ------------------------------------------------------------------
    with tab_seg:
        playbooks_df = generate_segment_playbooks(df)
        if playbooks_df.empty:
            st.info("No segment data available.")
        else:
            st.dataframe(playbooks_df, use_container_width=True, hide_index=True)
            fig_pb = px.bar(
                playbooks_df,
                x="Segment",
                y="Annual Revenue at Risk ($)",
                color="Avg Churn Risk (%)",
                text="Annual Revenue at Risk ($)",
                color_continuous_scale=["#0D9488", "#F59E0B", "#EF4444"],
                title="Annual Revenue at Risk Across ML Customer Segments",
                template="plotly_white",
            )
            fig_pb.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_pb.update_layout(height=360)
            st.plotly_chart(fig_pb, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3: DB-Backed Retention Campaign Tracker
    # ------------------------------------------------------------------
    with tab_tracker:
        actions_df = fetch_retention_actions(engine)
        if actions_df.empty:
            st.info("No retention interventions logged yet.")
        else:
            status_filter = st.multiselect(
                "Filter by Campaign Status",
                ["Recommended", "In Progress", "Applied", "Dismissed"],
                default=["Recommended", "In Progress", "Applied"],
            )
            filt_actions = actions_df[actions_df["status"].isin(status_filter)] if status_filter else actions_df
            st.dataframe(
                filt_actions[
                    [
                        "action_id", "customer_id", "customer_name", "cluster_segment",
                        "priority", "churn_probability", "revenue_at_risk",
                        "recommended_strategy", "status", "action_notes"
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "churn_probability": st.column_config.ProgressColumn("Risk", format="%.1f%%", min_value=0.0, max_value=1.0),
                    "revenue_at_risk": st.column_config.NumberColumn("Revenue at Risk", format="$%.0f"),
                },
            )

            with st.form("update_campaign_status_form"):
                u1, u2, u3 = st.columns([1, 1, 2])
                act_ids = filt_actions["action_id"].tolist() if not filt_actions.empty else actions_df["action_id"].tolist()
                sel_aid = u1.selectbox("Action ID", act_ids)
                new_status = u2.selectbox("New Status", ["Recommended", "In Progress", "Applied", "Dismissed"])
                new_notes = u3.text_input("Update Notes (Optional)", placeholder="Customer accepted 1-year discount offer")
                if st.form_submit_button("Update Campaign Status", type="primary"):
                    update_retention_status(engine, int(sel_aid), new_status, new_notes or None)
                    st.cache_data.clear()
                    st.success(f"Action #{sel_aid} updated to `{new_status}`.")
                    st.rerun()
