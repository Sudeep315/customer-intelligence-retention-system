"""Data ingestion, cleaning, feature engineering, and warehouse ETL pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    ENRICHED_CUSTOMERS_CSV,
    PROCESSED_DATA_DIR,
    RAW_CHURN_XLSX,
    RFM_CUSTOMERS_CSV,
    SERVICE_COLUMNS,
    SQL_WAREHOUSE_RAW_DIR,
)

# Column mapping from raw Excel headers to normalized snake_case DB schema
RAW_TO_SNAKE_COLS = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "numAdminTickets": "num_admin_tickets",
    "numTechTickets": "num_tech_tickets",
    "Churn": "actual_churn",
}

SNAKE_TO_RAW_COLS = {v: k for k, v in RAW_TO_SNAKE_COLS.items() if k not in ("Churn", "customerID")}


def compute_age_group(age: int | float | None) -> str:
    """Categorize numeric age into standardized demographic age buckets."""
    if age is None or pd.isna(age):
        return "30-39"
    age_val = int(age)
    if age_val < 20:
        return "Under 20"
    if age_val <= 29:
        return "20-29"
    if age_val <= 39:
        return "30-39"
    if age_val <= 49:
        return "40-49"
    return "50 and above"


def compute_value_tier(tenure: int, total_charges: float) -> str:
    """Assign SQL-inspired VIP / Regular / New tier based on tenure and lifetime spend."""
    if tenure >= 12 and total_charges > 2500:
        return "VIP"
    if tenure >= 12:
        return "Regular"
    return "New"


def normalize_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn dataframe (Excel or CSV) and engineer behavioral features."""
    work = df.copy()
    rename_map = {k: v for k, v in RAW_TO_SNAKE_COLS.items() if k in work.columns}
    work = work.rename(columns=rename_map)

    if "customer_id" not in work.columns:
        work["customer_id"] = [f"CUST-{1000 + i}" for i in range(len(work))]

    work["customer_id"] = work["customer_id"].astype(str).str.strip()
    work["senior_citizen"] = pd.to_numeric(work.get("senior_citizen", 0), errors="coerce").fillna(0).astype(int)
    work["tenure"] = pd.to_numeric(work.get("tenure", 0), errors="coerce").fillna(0).clip(lower=0).astype(int)
    work["monthly_charges"] = (
        pd.to_numeric(work.get("monthly_charges", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0).round(2)
    )
    work["total_charges"] = pd.to_numeric(work.get("total_charges", 0.0), errors="coerce")
    # Blank TotalCharges in Telco dataset occur when tenure == 0
    missing_tc = work["total_charges"].isna()
    if missing_tc.any():
        work.loc[missing_tc, "total_charges"] = (
            work.loc[missing_tc, "monthly_charges"] * work.loc[missing_tc, "tenure"]
        )
    work["total_charges"] = work["total_charges"].fillna(0.0).clip(lower=0.0).round(2)

    work["num_admin_tickets"] = (
        pd.to_numeric(work.get("num_admin_tickets", 0), errors="coerce").fillna(0).clip(lower=0).astype(int)
    )
    work["num_tech_tickets"] = (
        pd.to_numeric(work.get("num_tech_tickets", 0), errors="coerce").fillna(0).clip(lower=0).astype(int)
    )

    # Derived behavioral & spending metrics
    snake_service_cols = [RAW_TO_SNAKE_COLS[c] for c in SERVICE_COLUMNS]
    service_count = np.zeros(len(work), dtype=int)
    for col in snake_service_cols:
        if col in work.columns:
            service_count += (work[col].astype(str) == "Yes").astype(int)
    if "internet_service" in work.columns:
        service_count += (work["internet_service"].astype(str) != "No").astype(int)

    work["service_count"] = service_count
    work["total_tickets"] = work["num_admin_tickets"] + work["num_tech_tickets"]
    work["annual_spend"] = (work["monthly_charges"] * 12.0).round(2)
    work["value_tier"] = [
        compute_value_tier(int(t), float(tc))
        for t, tc in zip(work["tenure"], work["total_charges"])
    ]

    if "actual_churn" in work.columns:
        work["actual_churn"] = work["actual_churn"].astype(str).str.strip().replace({"1": "Yes", "0": "No"})
    else:
        work["actual_churn"] = "Unknown"

    return work


def to_model_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert snake_case customer dataframe back to exact ML pipeline input columns."""
    out = pd.DataFrame(index=df.index)
    for snake_col, raw_col in SNAKE_TO_RAW_COLS.items():
        if snake_col in df.columns:
            out[raw_col] = df[snake_col]
        elif raw_col in df.columns:
            out[raw_col] = df[raw_col]
    # Add derived behavioral features safe for ML (deterministic functions of raw columns)
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce").fillna(0).astype(int)
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0.0)
    out["numAdminTickets"] = pd.to_numeric(out["numAdminTickets"], errors="coerce").fillna(0).astype(int)
    out["numTechTickets"] = pd.to_numeric(out["numTechTickets"], errors="coerce").fillna(0).astype(int)

    svc_cnt = np.zeros(len(out), dtype=int)
    for c in SERVICE_COLUMNS:
        svc_cnt += (out[c].astype(str) == "Yes").astype(int)
    svc_cnt += (out["InternetService"].astype(str) != "No").astype(int)

    out["service_count"] = svc_cnt
    out["total_tickets"] = out["numAdminTickets"] + out["numTechTickets"]
    out["charge_per_service"] = np.where(svc_cnt > 0, out["MonthlyCharges"] / svc_cnt, out["MonthlyCharges"])
    out["ticket_rate_per_year"] = out["total_tickets"] / np.maximum(out["tenure"] / 12.0, 0.25)
    return out


def enrich_with_crm_profiles(df: pd.DataFrame, dim_customers_path: Path | None = None) -> pd.DataFrame:
    """Attach deterministic display CRM attributes (name, email, country, age) without leaking into ML."""
    work = df.copy()
    n = len(work)

    dim_path = dim_customers_path or (SQL_WAREHOUSE_RAW_DIR / "gold.dim_customers.csv")
    if dim_path.exists():
        dim_df = pd.read_csv(dim_path)
        dim_df = dim_df.dropna(subset=["first_name", "last_name"]).reset_index(drop=True)
        dim_df["country"] = dim_df["country"].fillna("United States")
        dim_df["marital_status"] = dim_df["marital_status"].fillna("Single")
    else:
        dim_df = pd.DataFrame(
            {
                "first_name": ["Alex", "Priya", "Jordan", "Maya", "Liam", "Sofia"],
                "last_name": ["Morgan", "Sharma", "Patel", "Chen", "Walker", "Garcia"],
                "country": ["United States", "United Kingdom", "Canada", "Australia", "Germany", "France"],
                "marital_status": ["Married", "Single", "Married", "Single", "Married", "Single"],
                "birthdate": ["1986-05-12"] * 6,
            }
        )

    # Match gender cleanly when picking deterministic profile names
    male_pool = dim_df[dim_df["gender"].astype(str).str.lower() == "male"].reset_index(drop=True) if "gender" in dim_df.columns else dim_df
    female_pool = dim_df[dim_df["gender"].astype(str).str.lower() == "female"].reset_index(drop=True) if "gender" in dim_df.columns else dim_df
    if male_pool.empty:
        male_pool = dim_df
    if female_pool.empty:
        female_pool = dim_df

    rng = np.random.default_rng(42)
    names, emails, countries, ages, age_groups, maritals = [], [], [], [], [], []
    ref_year = date(2025, 1, 1).year

    for idx, row in work.iterrows():
        pool = female_pool if str(row.get("gender", "")).lower() == "female" else male_pool
        pick = pool.iloc[int(idx) % len(pool)]
        fname = str(pick["first_name"]).strip()
        lname = str(pick["last_name"]).strip()
        full_name = f"{fname} {lname}"
        cid_clean = str(row["customer_id"]).lower().replace("-", "")
        email = f"{fname.lower()}.{lname.lower()}.{cid_clean[:4]}@custintel.io"
        country = str(pick["country"])

        # Align SeniorCitizen flag authentically with age
        if int(row.get("senior_citizen", 0)) == 1:
            age = int(rng.integers(60, 81))
        else:
            bdate = str(pick.get("birthdate", ""))
            if len(bdate) >= 4 and bdate[:4].isdigit():
                parsed_age = max(21, min(59, ref_year - int(bdate[:4])))
                age = parsed_age
            else:
                age = int(rng.integers(22, 58))

        marital = "Married" if str(row.get("partner", "No")) == "Yes" else str(pick.get("marital_status", "Single"))
        names.append(full_name)
        emails.append(email)
        countries.append(country)
        ages.append(age)
        age_groups.append(compute_age_group(age))
        maritals.append(marital)

    if "customer_name" not in work.columns:
        work["customer_name"] = names
    if "email" not in work.columns:
        work["email"] = emails
    if "country" not in work.columns:
        work["country"] = countries
    if "age" not in work.columns:
        work["age"] = ages
    work["age_group"] = [compute_age_group(a) for a in work["age"]]
    if "marital_status" not in work.columns:
        work["marital_status"] = maritals

    return work


def load_and_prepare_datasets(save_processed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Excel churn data + SQL warehouse customer RFM report and return cleaned frames."""
    if not RAW_CHURN_XLSX.exists():
        raise FileNotFoundError(f"Missing churn dataset at {RAW_CHURN_XLSX}")

    raw_df = pd.read_excel(RAW_CHURN_XLSX, sheet_name="01 Churn-Dataset")
    churn_df = normalize_churn_dataframe(raw_df)
    churn_df = enrich_with_crm_profiles(churn_df)

    rfm_path = SQL_WAREHOUSE_RAW_DIR / "gold.report_customers.csv"
    if not rfm_path.exists():
        raise FileNotFoundError(f"Missing SQL warehouse customer report at {rfm_path}")

    rfm_df = pd.read_csv(rfm_path)
    rfm_df["age"] = pd.to_numeric(rfm_df["age"], errors="coerce").fillna(rfm_df["age"].median()).astype(int)
    rfm_df["age_group"] = rfm_df["age"].apply(compute_age_group)
    rfm_df["customer_segment"] = rfm_df["customer_segment"].fillna("New")
    rfm_df["last_order_date"] = pd.to_datetime(rfm_df["last_order_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        churn_df.to_csv(ENRICHED_CUSTOMERS_CSV, index=False)
        rfm_df.to_csv(RFM_CUSTOMERS_CSV, index=False)

    return churn_df, rfm_df
