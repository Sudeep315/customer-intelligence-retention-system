"""Tests for Database Schema and Customer + Retention CRUD Operations."""

from __future__ import annotations

from pathlib import Path

from src.database import (
    add_customer,
    create_db_engine,
    delete_customer,
    fetch_customers,
    fetch_retention_actions,
    get_customer_by_id,
    init_schema,
    log_retention_action,
    update_customer,
    update_retention_status,
)


def test_customer_and_retention_crud(tmp_path: Path):
    db_file = tmp_path / "crud_test.db"
    engine, backend, _ = create_db_engine(force_sqlite_path=str(db_file))
    assert backend == "SQLite"
    init_schema(engine)

    payload = {
        "customer_id": "CRUD-1001",
        "customer_name": "Test User",
        "email": "test.user@example.com",
        "country": "United States",
        "age": 32,
        "age_group": "30-39",
        "marital_status": "Single",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 5,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 82.0,
        "total_charges": 410.0,
        "num_admin_tickets": 0,
        "num_tech_tickets": 2,
        "service_count": 3,
        "total_tickets": 2,
        "annual_spend": 984.0,
        "value_tier": "New",
        "actual_churn": "No",
        "churn_probability": 0.81,
        "predicted_churn": "Yes",
        "risk_level": "High Risk",
        "cluster_id": 1,
        "cluster_segment": "At-Risk High-Friction",
    }

    created = add_customer(engine, payload)
    assert created["customer_id"] == "CRUD-1001"
    assert len(fetch_customers(engine)) == 1

    updated = update_customer(engine, "CRUD-1001", {"contract": "One year", "churn_probability": 0.25, "risk_level": "Low Risk"})
    assert updated is not None
    assert updated["contract"] == "One year"
    assert updated["risk_level"] == "Low Risk"

    aid = log_retention_action(
        engine,
        {
            "customer_id": "CRUD-1001",
            "risk_level": "High Risk",
            "churn_probability": 0.81,
            "recommended_strategy": "Priority Engineering Escalation",
            "priority": "Critical",
            "estimated_save_rate": 0.52,
            "revenue_at_risk": 797.04,
            "action_notes": "Assigned Tier-2 engineer",
            "status": "Recommended",
        },
    )
    assert aid > 0
    assert update_retention_status(engine, aid, "Applied", "Customer retained")
    actions = fetch_retention_actions(engine)
    assert actions.loc[0, "status"] == "Applied"

    assert delete_customer(engine, "CRUD-1001")
    assert get_customer_by_id(engine, "CRUD-1001") is None


def test_warehouse_sql_aggregates_on_seeded_db():
    from src.database import fetch_warehouse_sql_aggregates, get_cached_connection

    engine, _, _ = get_cached_connection()
    aggs = fetch_warehouse_sql_aggregates(engine)
    assert aggs["total_tx_lines"] == 60398
    assert not aggs["monthly_trend"].empty
    assert not aggs["category_sales"].empty
    assert len(aggs["top_products"]) == 10
