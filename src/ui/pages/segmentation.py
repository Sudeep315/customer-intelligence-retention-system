"""Page 3: Customer Segmentation (K-Means + PCA) & SQL Warehouse Analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PALETTE, RFM_CUSTOMERS_CSV
from src.database import fetch_rfm_summary, fetch_warehouse_sql_aggregates, get_cached_connection
from src.ml_segmentation import (
    assign_behavioral_segment,
    load_behavioral_segmentation,
    load_rfm_segmentation,
)
from src.ui.charts import (
    chart_category_share,
    chart_elbow_silhouette,
    chart_monthly_revenue_trend,
    chart_pca_clusters,
)
from src.ui.styles import render_hero, render_kpi_card


@st.cache_resource(show_spinner=False)
def _load_behav_bundle_cached():
    return load_behavioral_segmentation()


@st.cache_data(ttl=300, show_spinner=False)
def _load_warehouse_cached():
    engine, _, _ = get_cached_connection()
    rfm_df = fetch_rfm_summary(engine)
    sql_aggs = fetch_warehouse_sql_aggregates(engine)
    _, rfm_meta = load_rfm_segmentation()
    rfm_plot = pd.read_csv(RFM_CUSTOMERS_CSV) if RFM_CUSTOMERS_CSV.exists() else rfm_df
    return rfm_df, sql_aggs, rfm_meta, rfm_plot


def render(df: pd.DataFrame) -> None:
    """Render Customer Segmentation & SQL Warehouse Analytics page."""
    render_hero(
        "Customer Segmentation & Purchasing Analytics",
        "K-Means Behavioral Clustering, PCA Cluster Maps, and SQL Warehouse RFM & Revenue Intelligence",
    )

    tab_behav, tab_rfm = st.tabs(
        ["Behavioral & Spend Clustering (Subscription)", "Purchasing Behavior RFM & SQL Analytics"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Behavioral Clustering on Subscription Customers
    # ------------------------------------------------------------------
    with tab_behav:
        try:
            seg_bundle, seg_meta = _load_behav_bundle_cached()
            seg_df = assign_behavioral_segment(df, bundle=seg_bundle) if not df.empty else df
        except Exception as exc:
            st.error(f"Segmentation model missing: {exc}")
            return

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card("Behavioral Clusters", str(seg_meta["n_clusters"]), "K-Means standardized features", accent="#0D9488")
        with c2:
            render_kpi_card("Silhouette Score", f"{seg_meta['silhouette_score']:.3f}", "Optimal k=4 separation", accent="#2563EB")
        with c3:
            var_sum = sum(seg_meta["pca_explained_variance"]) * 100
            render_kpi_card("PCA Variance Explained", f"{var_sum:.1f}%", "2D principal projection", accent="#6366F1")
        with c4:
            if not seg_df.empty:
                top_seg = seg_df["cluster_segment"].value_counts().index[0]
                render_kpi_card("Largest Segment", top_seg, f"{int((seg_df['cluster_segment'] == top_seg).sum()):,} accounts", accent="#F59E0B")

        st.markdown("")
        if not seg_df.empty:
            prof = (
                seg_df.groupby("cluster_segment", as_index=False)
                .agg(
                    customers=("customer_id", "count"),
                    avg_tenure=("tenure", "mean"),
                    avg_monthly=("monthly_charges", "mean"),
                    avg_total_spend=("total_charges", "mean"),
                    avg_services=("service_count", "mean"),
                    avg_tickets=("total_tickets", "mean"),
                    churn_rate_pct=("actual_churn", lambda s: (s == "Yes").mean() * 100),
                    avg_pred_risk=("churn_probability", lambda s: s.mean() * 100),
                )
                .round(2)
                .sort_values("avg_pred_risk", ascending=False)
            )
            st.markdown("#### Segment Behavioral Profiles")
            st.dataframe(
                prof,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "avg_monthly": st.column_config.NumberColumn("Avg Monthly ($)", format="$%.2f"),
                    "avg_total_spend": st.column_config.NumberColumn("Avg Lifetime ($)", format="$%.0f"),
                    "churn_rate_pct": st.column_config.NumberColumn("Actual Churn %", format="%.1f%%"),
                    "avg_pred_risk": st.column_config.NumberColumn("Predicted Risk %", format="%.1f%%"),
                },
            )

            col_pca, col_k = st.columns([1.3, 1])
            with col_pca:
                st.plotly_chart(
                    chart_pca_clusters(
                        seg_df,
                        "pca_1",
                        "pca_2",
                        "cluster_segment",
                        "2D PCA Customer Behavioral Cluster Map",
                        ["customer_id", "tenure", "monthly_charges", "total_tickets", "risk_level"],
                    ),
                    use_container_width=True,
                )
            with col_k:
                st.plotly_chart(
                    chart_elbow_silhouette(seg_meta["k_evaluation"], "Elbow & Silhouette Validation (k=2..6)"),
                    use_container_width=True,
                )

            box_df = seg_df.sample(n=min(2500, len(seg_df)), random_state=42) if len(seg_df) > 2500 else seg_df
            col_box1, col_box2 = st.columns(2)
            with col_box1:
                fig_m = px.box(
                    box_df,
                    x="cluster_segment",
                    y="monthly_charges",
                    color="cluster_segment",
                    title="Monthly Spend Distribution by Segment",
                    template="plotly_white",
                    color_discrete_sequence=[PALETTE["primary"], PALETTE["danger"], PALETTE["secondary"], PALETTE["accent"]],
                )
                fig_m.update_layout(showlegend=False, height=340)
                st.plotly_chart(fig_m, use_container_width=True)
            with col_box2:
                fig_t = px.box(
                    box_df,
                    x="cluster_segment",
                    y="tenure",
                    color="cluster_segment",
                    title="Customer Tenure (Months) by Segment",
                    template="plotly_white",
                    color_discrete_sequence=[PALETTE["primary"], PALETTE["danger"], PALETTE["secondary"], PALETTE["accent"]],
                )
                fig_t.update_layout(showlegend=False, height=340)
                st.plotly_chart(fig_t, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: Retail Warehouse RFM Clustering & SQL Analytics
    # ------------------------------------------------------------------
    with tab_rfm:
        try:
            rfm_df, sql_aggs, rfm_meta, rfm_plot_src = _load_warehouse_cached()
        except Exception as exc:
            st.error(f"Could not load SQL Warehouse analytics: {exc}")
            return

        if rfm_df.empty:
            st.info("Warehouse RFM tables are empty. Run `python scripts/init_project.py`.")
            return

        cat_df = sql_aggs["category_sales"]
        monthly_df = sql_aggs["monthly_trend"]
        top_prod_df = sql_aggs["top_products"]
        total_rev = float(cat_df["total_sales"].sum()) if not cat_df.empty else float(rfm_df["total_sales"].sum())
        total_orders = int(sql_aggs["total_orders"]) or int(rfm_df["total_orders"].sum())

        w1, w2, w3, w4 = st.columns(4)
        with w1:
            render_kpi_card("Warehouse Customers", f"{len(rfm_df):,}", "SQL Gold Customer Report", accent="#0D9488")
        with w2:
            render_kpi_card("Total Retail Orders", f"{total_orders:,}", f"{sql_aggs['total_tx_lines']:,} transaction lines", accent="#2563EB")
        with w3:
            render_kpi_card("Total Retail Sales", f"${total_rev / 1e6:.2f}M", f"Avg Order Value: ${rfm_df['avg_order_value'].mean():,.0f}", accent="#6366F1")
        with w4:
            render_kpi_card("RFM Silhouette", f"{rfm_meta['silhouette_score']:.3f}", "4 purchasing behavior clusters", accent="#10B981")

        st.markdown("")
        st.markdown("#### RFM Purchasing Behavior Clusters (K-Means)")
        st.dataframe(pd.DataFrame(rfm_meta["cluster_profiles"]), use_container_width=True, hide_index=True)

        p1, p2 = st.columns([1.3, 1])
        with p1:
            if "rfm_pca_1" in rfm_plot_src.columns:
                st.plotly_chart(
                    chart_pca_clusters(
                        rfm_plot_src,
                        "rfm_pca_1",
                        "rfm_pca_2",
                        "rfm_segment",
                        "2D PCA Purchasing Behavior (RFM) Cluster Map",
                        ["customer_number", "total_orders", "total_sales", "recency", "customer_segment"],
                    ),
                    use_container_width=True,
                )
        with p2:
            if not cat_df.empty:
                st.plotly_chart(chart_category_share(cat_df), use_container_width=True)

        if not monthly_df.empty:
            st.plotly_chart(chart_monthly_revenue_trend(monthly_df), use_container_width=True)

        tier_summary = (
            rfm_df.groupby("customer_segment", as_index=False)
            .agg(
                customers=("customer_key", "count"),
                total_sales=("total_sales", "sum"),
                avg_order_value=("avg_order_value", "mean"),
                avg_orders=("total_orders", "mean"),
                avg_recency=("recency", "mean"),
            )
            .round(2)
            .sort_values("total_sales", ascending=False)
        )

        b_left, b_right = st.columns(2)
        with b_left:
            st.markdown("##### SQL Customer Tier Summary (VIP / Regular / New)")
            st.dataframe(tier_summary, use_container_width=True, hide_index=True)
        with b_right:
            st.markdown("##### Top 10 Revenue-Generating Products")
            if not top_prod_df.empty:
                st.dataframe(top_prod_df, use_container_width=True, hide_index=True)
