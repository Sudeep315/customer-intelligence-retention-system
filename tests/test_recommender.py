"""Tests for AI Retention Recommendation Engine and Analytics."""

from __future__ import annotations

import pandas as pd

from src.analytics import compute_executive_kpis, generate_business_insights
from src.recommender import generate_segment_playbooks, recommend_for_customer


def test_recommender_high_risk_customer():
    row = {
        "customer_id": "9999-HIGH",
        "customer_name": "Jordan Blake",
        "churn_probability": 0.84,
        "risk_level": "High Risk",
        "monthly_charges": 92.0,
        "annual_spend": 1104.0,
        "tenure": 3,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "num_tech_tickets": 3,
        "num_admin_tickets": 1,
        "tech_support": "No",
        "online_security": "No",
        "value_tier": "New",
        "cluster_segment": "At-Risk High-Friction",
    }
    rec = recommend_for_customer(row)
    assert rec["priority"] == "Critical"
    assert rec["revenue_at_risk"] > 900
    assert rec["expected_saved_revenue"] > 0
    assert len(rec["tactics"]) >= 3
    assert "Jordan" in rec["email_body"]


def test_analytics_kpis_and_playbooks():
    df = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "actual_churn": "Yes",
                "risk_level": "High Risk",
                "churn_probability": 0.85,
                "monthly_charges": 80.0,
                "annual_spend": 960.0,
                "total_charges": 400.0,
                "num_admin_tickets": 1,
                "num_tech_tickets": 2,
                "total_tickets": 3,
                "tenure": 5,
                "contract": "Month-to-month",
                "payment_method": "Electronic check",
                "cluster_segment": "At-Risk High-Friction",
            },
            {
                "customer_id": "C2",
                "actual_churn": "No",
                "risk_level": "Low Risk",
                "churn_probability": 0.12,
                "monthly_charges": 50.0,
                "annual_spend": 600.0,
                "total_charges": 2400.0,
                "num_admin_tickets": 0,
                "num_tech_tickets": 0,
                "total_tickets": 0,
                "tenure": 48,
                "contract": "Two year",
                "payment_method": "Credit card (automatic)",
                "cluster_segment": "Loyal High-Value Champions",
            },
        ]
    )
    kpis = compute_executive_kpis(df)
    assert kpis["total_customers"] == 2
    assert kpis["actual_churn_rate"] == 50.0
    assert kpis["high_risk_count"] == 1

    insights = generate_business_insights(df)
    assert len(insights) >= 3

    playbooks = generate_segment_playbooks(df)
    assert len(playbooks) == 2
