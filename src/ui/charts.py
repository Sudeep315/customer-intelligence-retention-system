"""Reusable Plotly visualization builders with unified enterprise styling."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import PALETTE, RISK_COLORS, VALUE_TIER_COLORS


def _apply_clean_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color="#0F172A"),
            x=0.01,
            xanchor="left",
            y=0.96,
            yanchor="top",
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=78 if title else 30, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#334155", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0", automargin=True)
    fig.update_yaxes(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", automargin=True)
    return fig


def chart_risk_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of customer risk distribution."""
    counts = (
        df["risk_level"]
        .value_counts()
        .reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["risk_level", "customers"]
    fig = px.pie(
        counts,
        names="risk_level",
        values="customers",
        hole=0.58,
        color="risk_level",
        color_discrete_map=RISK_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+value")
    return _apply_clean_layout(fig, "Customer Risk Distribution", height=340)


def chart_dual_churn_volume(bd_df: pd.DataFrame, x_col: str, title: str) -> go.Figure:
    """Combo bar + line chart showing Churn Rate (%) and Customer Volume."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=bd_df[x_col],
            y=bd_df["churn_rate_pct"],
            name="Churn Rate (%)",
            marker_color=PALETTE["primary"],
            text=[f"{v:.1f}%" for v in bd_df["churn_rate_pct"]],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#FFFFFF", size=12),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bd_df[x_col],
            y=bd_df["customers"],
            name="Total Customers",
            mode="lines+markers+text",
            line=dict(color=PALETTE["secondary"], width=3),
            marker=dict(size=8),
            text=[f"{int(v):,}" for v in bd_df["customers"]],
            textposition="top center",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Churn Rate (%)", range=[0, max(bd_df["churn_rate_pct"].max() * 1.28, 12)], secondary_y=False)
    max_cust = float(bd_df["customers"].max()) if not bd_df.empty else 10.0
    fig.update_yaxes(title_text="Customers", range=[0, max_cust * 1.22], showgrid=False, secondary_y=True)
    return _apply_clean_layout(fig, title, height=380)


def chart_tenure_cohorts(cohort_df: pd.DataFrame) -> go.Figure:
    """Tenure cohort bar + line chart matching Power BI Years of Contract insight."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=cohort_df["tenure_cohort"].astype(str),
            y=cohort_df["churn_rate"],
            name="Churn Rate (%)",
            marker_color=["#EF4444", "#F59E0B", "#0D9488", "#0EA5E9", "#6366F1", "#10B981"][: len(cohort_df)],
            text=[f"{v:.1f}%" for v in cohort_df["churn_rate"]],
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=cohort_df["tenure_cohort"].astype(str),
            y=cohort_df["monthly_revenue"],
            name="Monthly Revenue ($)",
            mode="lines+markers",
            line=dict(color="#1E293B", width=2.5),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Churn Rate (%)", range=[0, max(cohort_df["churn_rate"].max() * 1.25, 10)], secondary_y=False)
    fig.update_yaxes(title_text="Monthly Revenue ($)", showgrid=False, secondary_y=True)
    return _apply_clean_layout(fig, "Subscription Tenure Cohorts vs Churn & Revenue", height=360)


def chart_service_adoption(svc_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing service adoption and churn rate."""
    fig = px.bar(
        svc_df.sort_values("Adoption (%)", ascending=True),
        x="Adoption (%)",
        y="Service",
        orientation="h",
        color="Churn Rate (%)",
        color_continuous_scale="Tealrose",
        text="Adoption (%)",
        hover_data=["Subscribers", "Churn Rate (%)"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_xaxes(range=[0, 112])
    return _apply_clean_layout(fig, "Subscribed Services Adoption & Churn Exposure", height=380)


def chart_ticket_impact(df: pd.DataFrame) -> go.Figure:
    """Grouped bar showing actual churn rate by number of tech tickets."""
    work = df.copy()
    work["tech_ticket_bucket"] = np.where(work["num_tech_tickets"] >= 4, "4+ Tickets", work["num_tech_tickets"].astype(str) + " Ticket(s)")
    grp = (
        work.groupby("tech_ticket_bucket", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            churn_rate=("actual_churn", lambda s: round((s == "Yes").mean() * 100, 1)),
        )
        .sort_values("tech_ticket_bucket")
    )
    fig = px.bar(
        grp,
        x="tech_ticket_bucket",
        y="churn_rate",
        color="churn_rate",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        text="churn_rate",
        hover_data=["customers"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_yaxes(title_text="Historical Churn Rate (%)", range=[0, 108])
    fig.update_xaxes(title_text="Technical Support Tickets")
    return _apply_clean_layout(fig, "Technical Support Tickets vs Churn Escalation", height=360)


def chart_roc_curves(roc_curves: Dict[str, Dict[str, List[float]]], leaderboard: List[Dict[str, Any]]) -> go.Figure:
    """Plot multi-model ROC curve comparison."""
    auc_map = {m["model"]: m["roc_auc"] for m in leaderboard}
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["warning"]]
    fig = go.Figure()
    for idx, (model_name, coords) in enumerate(roc_curves.items()):
        auc_val = auc_map.get(model_name, 0.0)
        fig.add_trace(
            go.Scatter(
                x=coords["fpr"],
                y=coords["tpr"],
                mode="lines",
                name=f"{model_name} (AUC={auc_val:.3f})",
                line=dict(width=2.5, color=colors[idx % len(colors)]),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(dash="dash", color="#94A3B8"),
        )
    )
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate")
    return _apply_clean_layout(fig, "ROC Curve Comparison Across Models", height=370)


def chart_feature_importance(top_features: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal feature importance plot."""
    fdf = pd.DataFrame(top_features).sort_values("importance", ascending=True)
    fig = px.bar(
        fdf,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Teal",
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title_text="Normalized Importance")
    fig.update_yaxes(title_text="")
    return _apply_clean_layout(fig, "Top Predictive Churn Drivers", height=400)


def chart_confusion_matrix(cm: List[List[int]], model_name: str) -> go.Figure:
    """Heatmap confusion matrix."""
    fig = px.imshow(
        cm,
        x=["Predicted Stay", "Predicted Churn"],
        y=["Actual Stay", "Actual Churn"],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    return _apply_clean_layout(fig, f"Confusion Matrix ({model_name})", height=320)


def chart_pca_clusters(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str, hover_cols: List[str]) -> go.Figure:
    """2D PCA cluster visualization (sampled if very large for snappy browser rendering)."""
    plot_df = df.sample(n=min(2500, len(df)), random_state=42) if len(df) > 2500 else df
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_cols,
        opacity=0.75,
        color_discrete_sequence=[PALETTE["primary"], PALETTE["danger"], PALETTE["secondary"], PALETTE["accent"], PALETTE["warning"]],
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_xaxes(title_text="Principal Component 1")
    fig.update_yaxes(title_text="Principal Component 2")
    fig = _apply_clean_layout(fig, title, height=430)
    fig.update_layout(
        margin=dict(l=24, r=24, t=54, b=82),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5, title_text=""),
    )
    return fig


def chart_elbow_silhouette(k_eval: List[Dict[str, Any]], title: str) -> go.Figure:
    """Dual chart showing Elbow Inertia & Silhouette Score for k=2..6."""
    edf = pd.DataFrame(k_eval)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=edf["k"], y=edf["inertia"], mode="lines+markers", name="Inertia (Elbow)", line=dict(color=PALETTE["secondary"], width=2.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=edf["k"], y=edf["silhouette"], name="Silhouette Score", marker_color=PALETTE["primary"], opacity=0.65),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Number of Clusters (k)", dtick=1)
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", range=[0, 0.8], secondary_y=True)
    return _apply_clean_layout(fig, title, height=340)


def chart_monthly_revenue_trend(monthly_df: pd.DataFrame) -> go.Figure:
    """Monthly revenue and cumulative running total from SQL Warehouse analytics."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=monthly_df["order_month"], y=monthly_df["total_revenue"], name="Monthly Revenue ($)", marker_color=PALETTE["primary"]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly_df["order_month"], y=monthly_df["running_total"], name="Cumulative Revenue ($)", mode="lines", line=dict(color=PALETTE["accent"], width=2.5)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Monthly Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Running Total ($)", showgrid=False, secondary_y=True)
    return _apply_clean_layout(fig, "Warehouse Sales Revenue Over Time & Cumulative Growth", height=360)


def chart_category_share(cat_df: pd.DataFrame) -> go.Figure:
    """Part-to-whole category sales breakdown."""
    fig = px.pie(
        cat_df,
        names="category",
        values="total_sales",
        hole=0.52,
        color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["warning"]],
    )
    fig.update_traces(textinfo="percent", textposition="inside")
    fig = _apply_clean_layout(fig, "Part-to-Whole Revenue by Product Category", height=360)
    fig.update_layout(
        margin=dict(l=24, r=24, t=54, b=54),
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5),
    )
    return fig


def chart_value_tier_breakdown(df: pd.DataFrame) -> go.Figure:
    """Stacked/grouped value tier comparison (VIP / Regular / New)."""
    bd = (
        df.groupby(["value_tier", "risk_level"], as_index=False)
        .agg(customers=("customer_id", "count"))
    )
    fig = px.bar(
        bd,
        x="value_tier",
        y="customers",
        color="risk_level",
        barmode="stack",
        color_discrete_map=RISK_COLORS,
        category_orders={"value_tier": ["VIP", "Regular", "New"]},
    )
    fig.update_xaxes(title_text="Customer Value Tier")
    fig.update_yaxes(title_text="Customers")
    return _apply_clean_layout(fig, "Value Tier (VIP / Regular / New) vs Churn Risk", height=350)
