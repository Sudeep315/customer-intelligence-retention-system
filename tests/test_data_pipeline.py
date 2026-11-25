"""Tests for data ingestion, cleaning, and feature engineering."""

from __future__ import annotations

import pandas as pd

from src.data_pipeline import (
    compute_age_group,
    compute_value_tier,
    normalize_churn_dataframe,
    to_model_feature_frame,
)


def test_compute_age_group_boundaries():
    assert compute_age_group(18) == "Under 20"
    assert compute_age_group(25) == "20-29"
    assert compute_age_group(35) == "30-39"
    assert compute_age_group(49) == "40-49"
    assert compute_age_group(65) == "50 and above"
    assert compute_age_group(None) == "30-39"


def test_compute_value_tier():
    assert compute_value_tier(24, 3200.0) == "VIP"
    assert compute_value_tier(24, 1500.0) == "Regular"
    assert compute_value_tier(6, 4000.0) == "New"


def test_normalize_handles_blank_total_charges():
    raw = pd.DataFrame(
        [
            {
                "customerID": "0001-TEST",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 55.0,
                "TotalCharges": " ",
                "numAdminTickets": 1,
                "numTechTickets": 2,
                "Churn": "No",
            }
        ]
    )
    norm = normalize_churn_dataframe(raw)
    assert float(norm.loc[0, "total_charges"]) == 0.0
    assert int(norm.loc[0, "service_count"]) == 3
    assert int(norm.loc[0, "total_tickets"]) == 3
    assert float(norm.loc[0, "annual_spend"]) == 660.0

    feats = to_model_feature_frame(norm)
    assert "MonthlyCharges" in feats.columns
    assert "ticket_rate_per_year" in feats.columns
