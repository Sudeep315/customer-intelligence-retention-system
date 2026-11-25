"""One-command initializer: runs ETL, trains ML models, scores datasets, and seeds DB."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ENRICHED_CUSTOMERS_CSV, RFM_CUSTOMERS_CSV
from src.data_pipeline import load_and_prepare_datasets
from src.database import (
    create_db_engine,
    seed_customers,
    seed_initial_retention_actions,
    seed_warehouse_tables,
)
from src.ml_churn import score_customers, train_churn_models
from src.ml_segmentation import train_behavioral_segmentation, train_rfm_segmentation
from src.recommender import build_initial_retention_queue


def run_initialization(replace_db: bool = True) -> None:
    """Execute full data, ML training, scoring, and DB seeding pipeline."""
    print("1/5 Loading and cleaning raw datasets...")
    churn_df, rfm_df = load_and_prepare_datasets(save_processed=False)
    print(f"    Loaded {len(churn_df):,} churn accounts and {len(rfm_df):,} RFM warehouse records.")

    print("2/5 Training & benchmarking Churn Prediction ML models...")
    churn_pipe, churn_meta = train_churn_models(churn_df, save_artifacts=True)
    best = churn_meta["leaderboard"][0]
    print(
        f"    Best Churn Model: {best['model']} | Accuracy={best['accuracy']:.4f} | "
        f"F1={best['f1_score']:.4f} | ROC-AUC={best['roc_auc']:.4f}"
    )

    print("3/5 Training Customer Segmentation (K-Means + PCA) models...")
    scored_df = score_customers(churn_df, pipe=churn_pipe)
    segmented_df, seg_meta = train_behavioral_segmentation(scored_df, n_clusters=4, save_artifacts=True)
    rfm_seg_df, rfm_meta = train_rfm_segmentation(rfm_df, n_clusters=4, save_artifacts=True)
    print(
        f"    Behavioral K-Means Silhouette={seg_meta['silhouette_score']:.4f} | "
        f"RFM K-Means Silhouette={rfm_meta['silhouette_score']:.4f}"
    )

    print("4/5 Saving enriched datasets to data/processed/...")
    ENRICHED_CUSTOMERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    segmented_df.to_csv(ENRICHED_CUSTOMERS_CSV, index=False)
    rfm_seg_df.to_csv(RFM_CUSTOMERS_CSV, index=False)

    print("5/5 Initializing Database and seeding records...")
    engine, backend, msg = create_db_engine()
    print(f"    Active Database Backend: {backend} — {msg}")
    cust_count = seed_customers(engine, segmented_df, replace=replace_db)
    wh_counts = seed_warehouse_tables(engine, rfm_seg_df, replace=replace_db)
    queue = build_initial_retention_queue(segmented_df, top_n=40)
    action_count = seed_initial_retention_actions(engine, segmented_df, queue)
    print(
        f"    Seeded {cust_count:,} customers, {action_count} retention actions, "
        f"and warehouse tables: {wh_counts}"
    )
    print("\nInitialization complete! Launch dashboard with: streamlit run app.py")


if __name__ == "__main__":
    run_initialization(replace_db=True)
