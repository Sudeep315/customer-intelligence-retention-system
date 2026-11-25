"""Analytical KPI engine and SQL-style aggregation helper functions."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def compute_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute core dashboard KPIs matching Power BI reference metrics + ML risk exposure."""
    if df.empty:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "actual_churn_rate": 0.0,
            "high_risk_count": 0,
            "predicted_risk_rate": 0.0,
            "mrr": 0.0,
            "arr": 0.0,
            "lifetime_charges": 0.0,
            "revenue_at_risk": 0.0,
            "admin_tickets": 0,
            "tech_tickets": 0,
            "avg_monthly_charges": 0.0,
            "avg_tenure": 0.0,
        }

    known_mask = df["actual_churn"].isin(["Yes", "No"])
    churned_count = int((df["actual_churn"] == "Yes").sum())
    denom = int(known_mask.sum()) or len(df)
    actual_churn_rate = (churned_count / denom) * 100.0

    high_risk_count = int((df["risk_level"] == "High Risk").sum())
    predicted_risk_rate = float(df["churn_probability"].mean() * 100.0)
    mrr = float(df["monthly_charges"].sum())
    arr = float(df["annual_spend"].sum())
    lifetime_charges = float(df["total_charges"].sum())
    revenue_at_risk = float((df["annual_spend"] * df["churn_probability"]).sum())

    return {
        "total_customers": int(len(df)),
        "churned_customers": churned_count,
        "actual_churn_rate": round(actual_churn_rate, 2),
        "high_risk_count": high_risk_count,
        "predicted_risk_rate": round(predicted_risk_rate, 2),
        "mrr": round(mrr, 2),
        "arr": round(arr, 2),
        "lifetime_charges": round(lifetime_charges, 2),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "admin_tickets": int(df["num_admin_tickets"].sum()),
        "tech_tickets": int(df["num_tech_tickets"].sum()),
        "avg_monthly_charges": round(float(df["monthly_charges"].mean()), 2),
        "avg_tenure": round(float(df["tenure"].mean()), 1),
    }


def compute_tenure_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Group customers into <1 Year through <6 Year+ subscription cohorts."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    bins = [-1, 12, 24, 36, 48, 60, 1200]
    labels = ["<1 Year", "<2 Year", "<3 Year", "<4 Year", "<5 Year", "5+ Years"]
    work["tenure_cohort"] = pd.cut(work["tenure"], bins=bins, labels=labels)

    grp = (
        work.groupby("tenure_cohort", observed=False, as_index=False)
        .agg(
            customers=("customer_id", "count"),
            churned=("actual_churn", lambda s: int((s == "Yes").sum())),
            avg_risk=("churn_probability", lambda s: round(float(s.mean() * 100), 2)),
            monthly_revenue=("monthly_charges", "sum"),
        )
    )
    grp["churn_rate"] = np.where(
        grp["customers"] > 0, np.round((grp["churned"] / grp["customers"]) * 100.0, 2), 0.0
    )
    return grp


def compute_breakdown(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute customer count, churn rate, predicted risk, and revenue across a categorical dimension."""
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    grp = (
        df.groupby(group_col, as_index=False)
        .agg(
            customers=("customer_id", "count"),
            churned=("actual_churn", lambda s: int((s == "Yes").sum())),
            avg_risk_pct=("churn_probability", lambda s: round(float(s.mean() * 100), 2)),
            monthly_revenue=("monthly_charges", "sum"),
            lifetime_revenue=("total_charges", "sum"),
            tech_tickets=("num_tech_tickets", "sum"),
        )
    )
    grp["churn_rate_pct"] = np.where(
        grp["customers"] > 0, np.round((grp["churned"] / grp["customers"]) * 100.0, 2), 0.0
    )
    grp["share_pct"] = np.round((grp["customers"] / len(df)) * 100.0, 2)
    return grp.sort_values("customers", ascending=False)


def compute_service_adoption(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate adoption percentage and churn rate per subscribed service."""
    if df.empty:
        return pd.DataFrame()
    services = [
        ("Phone Service", "phone_service", "Yes"),
        ("Multiple Lines", "multiple_lines", "Yes"),
        ("Internet Service", "internet_service", "NOT_NO"),
        ("Online Security", "online_security", "Yes"),
        ("Online Backup", "online_backup", "Yes"),
        ("Device Protection", "device_protection", "Yes"),
        ("Tech Support", "tech_support", "Yes"),
        ("Streaming TV", "streaming_tv", "Yes"),
        ("Streaming Movies", "streaming_movies", "Yes"),
    ]
    rows = []
    total = len(df)
    for label, col, target in services:
        if col not in df.columns:
            continue
        mask = df[col].astype(str) != "No" if target == "NOT_NO" else df[col].astype(str) == target
        sub = df[mask]
        adopted_cnt = int(mask.sum())
        churn_pct = float((sub["actual_churn"] == "Yes").mean() * 100) if adopted_cnt > 0 else 0.0
        rows.append(
            {
                "Service": label,
                "Subscribers": adopted_cnt,
                "Adoption (%)": round((adopted_cnt / total) * 100.0, 1),
                "Churn Rate (%)": round(churn_pct, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("Adoption (%)", ascending=False)


def generate_business_insights(df: pd.DataFrame) -> List[str]:
    """Produce live automated executive insights from the active filtered customer cohort."""
    if df.empty:
        return ["No customers match the current filter selection."]

    insights: List[str] = []
    kpis = compute_executive_kpis(df)

    # Contract insight
    contract_bd = compute_breakdown(df, "contract")
    if not contract_bd.empty:
        top_risk_contract = contract_bd.sort_values("churn_rate_pct", ascending=False).iloc[0]
        insights.append(
            f"**Contract Vulnerability:** `{top_risk_contract['contract']}` accounts represent "
            f"{top_risk_contract['share_pct']:.1f}% of selected customers with a **{top_risk_contract['churn_rate_pct']:.1f}%** churn rate."
        )

    # Tech ticket insight
    high_ticket = df[df["num_tech_tickets"] >= 2]
    no_ticket = df[df["num_tech_tickets"] == 0]
    if not high_ticket.empty and not no_ticket.empty:
        ht_churn = (high_ticket["actual_churn"] == "Yes").mean() * 100
        nt_churn = (no_ticket["actual_churn"] == "Yes").mean() * 100
        insights.append(
            f"**Support Ticket Multiplier:** Customers with 2+ technical tickets churn at **{ht_churn:.1f}%** "
            f"versus **{nt_churn:.1f}%** for zero-ticket accounts."
        )

    # Payment insight
    pay_bd = compute_breakdown(df, "payment_method")
    if not pay_bd.empty:
        worst_pay = pay_bd.sort_values("churn_rate_pct", ascending=False).iloc[0]
        insights.append(
            f"**Payment Friction:** `{worst_pay['payment_method']}` carries the highest churn rate "
            f"(**{worst_pay['churn_rate_pct']:.1f}%**); migrating these accounts to Auto-Pay is high ROI."
        )

    # Financial exposure insight
    insights.append(
        f"**Revenue at Risk:** **{kpis['high_risk_count']:,}** High-Risk accounts expose "
        f"**${kpis['revenue_at_risk']:,.0f}** in projected annual recurring revenue."
    )
    return insights


def compute_warehouse_analytics(sales_df: pd.DataFrame, rfm_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute SQL-analytics-style time series, category part-to-whole, and segment summaries."""
    out: Dict[str, pd.DataFrame] = {}
    if not sales_df.empty:
        s = sales_df.dropna(subset=["order_date"]).copy()
        s["order_dt"] = pd.to_datetime(s["order_date"], errors="coerce")
        s = s.dropna(subset=["order_dt"])
        s["order_month"] = s["order_dt"].dt.to_period("M").astype(str)

        monthly = (
            s.groupby("order_month", as_index=False)
            .agg(
                total_revenue=("sales_amount", "sum"),
                total_orders=("order_number", "nunique"),
                active_customers=("customer_key", "nunique"),
                total_units=("quantity", "sum"),
            )
            .sort_values("order_month")
        )
        monthly["running_total"] = monthly["total_revenue"].cumsum()
        out["monthly_trend"] = monthly

        cat = (
            s.groupby("category", dropna=False, as_index=False)
            .agg(
                total_sales=("sales_amount", "sum"),
                total_orders=("order_number", "nunique"),
                total_units=("quantity", "sum"),
            )
            .sort_values("total_sales", ascending=False)
        )
        cat["category"] = cat["category"].fillna("Other")
        total_rev = cat["total_sales"].sum() or 1.0
        cat["share_pct"] = np.round((cat["total_sales"] / total_rev) * 100.0, 2)
        out["category_sales"] = cat

        top_products = (
            s.groupby(["product_name", "category"], as_index=False)
            .agg(total_sales=("sales_amount", "sum"), units_sold=("quantity", "sum"))
            .sort_values("total_sales", ascending=False)
            .head(10)
        )
        out["top_products"] = top_products

    if not rfm_df.empty:
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
        out["tier_summary"] = tier_summary

        age_summary = (
            rfm_df.groupby("age_group", as_index=False)
            .agg(customers=("customer_key", "count"), total_sales=("total_sales", "sum"))
            .sort_values("customers", ascending=False)
        )
        out["age_summary"] = age_summary

    return out
