"""Dedicated MySQL 8.0 database setup and seeding utility."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import ENRICHED_CUSTOMERS_CSV, RFM_CUSTOMERS_CSV, get_db_settings
from src.database import (
    create_db_engine,
    seed_customers,
    seed_initial_retention_actions,
    seed_warehouse_tables,
    test_mysql_connection,
)
from src.recommender import build_initial_retention_queue


def main() -> None:
    """Validate MySQL connection, create schema, and seed processed datasets into MySQL."""
    os.environ["DB_BACKEND"] = "mysql"
    settings = get_db_settings()
    ok, msg = test_mysql_connection(
        settings.mysql_host,
        settings.mysql_port,
        settings.mysql_user,
        settings.mysql_password,
        settings.mysql_database,
    )
    if not ok:
        print(f"[ERROR] {msg}")
        print("Ensure MySQL 8.0 Server is running and credentials in .env are accurate.")
        sys.exit(1)

    print(f"[OK] {msg}")
    if not ENRICHED_CUSTOMERS_CSV.exists() or not RFM_CUSTOMERS_CSV.exists():
        print("Processed CSVs not found. Running full initialization pipeline first...")
        from scripts.init_project import run_initialization

        run_initialization(replace_db=True)
        return

    engine, backend, detail = create_db_engine()
    print(f"Connected via {backend}: {detail}")
    customers_df = pd.read_csv(ENRICHED_CUSTOMERS_CSV)
    rfm_df = pd.read_csv(RFM_CUSTOMERS_CSV)

    cust_cnt = seed_customers(engine, customers_df, replace=True)
    wh_cnts = seed_warehouse_tables(engine, rfm_df, replace=True)
    actions = build_initial_retention_queue(customers_df, top_n=40)
    act_cnt = seed_initial_retention_actions(engine, customers_df, actions)
    print(f"Seeded MySQL successfully: customers={cust_cnt:,}, retention_actions={act_cnt}, warehouse={wh_cnts}")


if __name__ == "__main__":
    main()
