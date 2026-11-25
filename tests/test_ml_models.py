"""Tests for Churn Prediction and Segmentation ML models."""

from __future__ import annotations

import pandas as pd

from src.ml_churn import explain_customer_churn, load_churn_model, probability_to_risk_level, score_single_customer
from src.ml_segmentation import assign_behavioral_segment, load_behavioral_segmentation


def test_risk_thresholds():
    assert probability_to_risk_level(0.75) == "High Risk"
    assert probability_to_risk_level(0.45) == "Medium Risk"
    assert probability_to_risk_level(0.15) == "Low Risk"


def test_churn_model_scoring_and_explanation():
    pipe, meta = load_churn_model()
    assert meta["leaderboard"][0]["roc_auc"] >= 0.85

    high_risk_customer = {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 2,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "num_admin_tickets": 1,
        "num_tech_tickets": 3,
    }
    low_risk_customer = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "Yes",
        "tenure": 60,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 55.0,
        "total_charges": 3300.0,
        "num_admin_tickets": 0,
        "num_tech_tickets": 0,
    }

    res_high = score_single_customer(high_risk_customer, pipe=pipe)
    res_low = score_single_customer(low_risk_customer, pipe=pipe)
    assert res_high["churn_probability"] > res_low["churn_probability"]
    assert res_high["risk_level"] == "High Risk"
    assert res_low["risk_level"] == "Low Risk"

    exp = explain_customer_churn(high_risk_customer)
    assert len(exp["risk_drivers"]) >= 3


def test_behavioral_segmentation_assignment():
    bundle, meta = load_behavioral_segmentation()
    assert meta["n_clusters"] == 4
    sample = pd.DataFrame(
        [
            {
                "tenure": 48,
                "monthly_charges": 89.5,
                "total_charges": 4296.0,
                "service_count": 6,
                "total_tickets": 0,
            }
        ]
    )
    out = assign_behavioral_segment(sample, bundle=bundle)
    assert out.loc[0, "cluster_segment"] in set(meta["cluster_names"].values())
    assert "pca_1" in out.columns and "pca_2" in out.columns
