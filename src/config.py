"""Centralized configuration, paths, constants, and environment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SQL_WAREHOUSE_RAW_DIR = RAW_DATA_DIR / "sql_warehouse"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = PROJECT_ROOT / "database"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CHURN_XLSX = RAW_DATA_DIR / "customer_churn_dataset.xlsx"
ENRICHED_CUSTOMERS_CSV = PROCESSED_DATA_DIR / "customers_enriched.csv"
RFM_CUSTOMERS_CSV = PROCESSED_DATA_DIR / "rfm_customers_segmented.csv"

CHURN_MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
CHURN_METRICS_PATH = MODELS_DIR / "churn_metrics.json"
SEG_MODEL_PATH = MODELS_DIR / "segmentation_pipeline.joblib"
SEG_METRICS_PATH = MODELS_DIR / "segmentation_metrics.json"
RFM_MODEL_PATH = MODELS_DIR / "rfm_pipeline.joblib"
RFM_METRICS_PATH = MODELS_DIR / "rfm_metrics.json"

RISK_THRESHOLDS = {
    "high": 0.65,
    "medium": 0.35,
}

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CHURN_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "numAdminTickets",
    "numTechTickets",
]

PALETTE = {
    "primary": "#0D9488",
    "primary_dark": "#0F766E",
    "secondary": "#2563EB",
    "accent": "#6366F1",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "slate": "#475569",
    "muted": "#94A3B8",
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
}

RISK_COLORS = {
    "High Risk": "#EF4444",
    "Medium Risk": "#F59E0B",
    "Low Risk": "#10B981",
}

VALUE_TIER_COLORS = {
    "VIP": "#6366F1",
    "Regular": "#0D9488",
    "New": "#38BDF8",
}


@dataclass(frozen=True)
class DatabaseSettings:
    """Database connection configuration loaded from environment variables."""

    backend: str = os.getenv("DB_BACKEND", "auto").strip().lower()
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1").strip()
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root").strip()
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "customer_intelligence_db").strip()
    sqlite_path: Path = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/customer_intelligence.db")

    def mysql_url(self, include_db: bool = True) -> str:
        from urllib.parse import quote_plus

        pwd = quote_plus(self.mysql_password)
        auth = f"{self.mysql_user}:{pwd}" if self.mysql_password else self.mysql_user
        db_part = f"/{self.mysql_database}" if include_db else ""
        return f"mysql+pymysql://{auth}@{self.mysql_host}:{self.mysql_port}{db_part}?charset=utf8mb4"

    def sqlite_url(self) -> str:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.sqlite_path.as_posix()}"


def get_db_settings() -> DatabaseSettings:
    """Load environment variables from .env (preserving explicit process overrides) and return DatabaseSettings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return DatabaseSettings(
        backend=os.getenv("DB_BACKEND", "auto").strip().lower(),
        mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1").strip(),
        mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
        mysql_user=os.getenv("MYSQL_USER", "root").strip(),
        mysql_password=os.getenv("MYSQL_PASSWORD", ""),
        mysql_database=os.getenv("MYSQL_DATABASE", "customer_intelligence_db").strip(),
        sqlite_path=PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/customer_intelligence.db"),
    )
