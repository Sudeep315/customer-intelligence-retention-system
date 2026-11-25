"""Customer Segmentation (K-Means + PCA) for Subscription Behavior and Retail RFM Purchasing."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODELS_DIR,
    RFM_METRICS_PATH,
    RFM_MODEL_PATH,
    SEG_METRICS_PATH,
    SEG_MODEL_PATH,
)

BEHAVIORAL_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "service_count",
    "total_tickets",
]

RFM_FEATURES = [
    "recency",
    "total_orders",
    "total_sales",
    "total_quantity",
    "lifespan",
    "avg_order_value",
]


def _assign_behavioral_cluster_names(centroids_df: pd.DataFrame) -> Dict[int, str]:
    """Deterministically assign meaningful business names based on standardized centroids."""
    remaining = set(centroids_df.index.tolist())
    mapping: Dict[int, str] = {}

    # 1. Highest ticket intensity / risk friction cluster
    at_risk_cid = max(remaining, key=lambda c: centroids_df.loc[c, "total_tickets"] + 0.4 * centroids_df.loc[c, "monthly_charges"])
    mapping[at_risk_cid] = "At-Risk High-Friction"
    remaining.remove(at_risk_cid)

    # 2. Highest lifetime spend & tenure & services -> VIP Champions
    champ_cid = max(remaining, key=lambda c: centroids_df.loc[c, "total_charges"] + centroids_df.loc[c, "tenure"] + 0.5 * centroids_df.loc[c, "service_count"])
    mapping[champ_cid] = "Loyal High-Value Champions"
    remaining.remove(champ_cid)

    # 3. Lowest monthly charges & moderate/high tenure -> Budget Core
    budget_cid = min(remaining, key=lambda c: centroids_df.loc[c, "monthly_charges"] - 0.3 * centroids_df.loc[c, "tenure"])
    mapping[budget_cid] = "Budget Steady Core"
    remaining.remove(budget_cid)

    # 4. Remaining -> Early-Stage Growth / Explorers
    for cid in remaining:
        mapping[cid] = "New Digital Explorers"

    return mapping


def _assign_rfm_cluster_names(centroids_df: pd.DataFrame) -> Dict[int, str]:
    """Deterministically name RFM retail clusters from standardized centroids."""
    remaining = set(centroids_df.index.tolist())
    mapping: Dict[int, str] = {}

    # 1. Highest cumulative spend & lifespan -> VIP Repeat Spenders
    vip_cid = max(remaining, key=lambda c: centroids_df.loc[c, "total_sales"] + 0.8 * centroids_df.loc[c, "lifespan"])
    mapping[vip_cid] = "VIP Repeat Spenders"
    remaining.remove(vip_cid)

    # 2. Highest order frequency -> Frequent Multi-Order Buyers
    freq_cid = max(remaining, key=lambda c: centroids_df.loc[c, "total_orders"] + 0.5 * centroids_df.loc[c, "total_quantity"])
    mapping[freq_cid] = "Frequent Multi-Order Buyers"
    remaining.remove(freq_cid)

    # 3. Highest recency (most dormant) + high average order value -> Dormant High-Ticket
    dormant_cid = max(remaining, key=lambda c: centroids_df.loc[c, "recency"] + 0.6 * centroids_df.loc[c, "avg_order_value"])
    mapping[dormant_cid] = "Dormant High-Ticket"
    remaining.remove(dormant_cid)

    # 4. Remaining -> Occasional / Entry Buyers
    for cid in remaining:
        mapping[cid] = "Occasional / Entry Buyers"

    return mapping


def _evaluate_k_range(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 6) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    sample_sz = min(2500, len(X_scaled))
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels, sample_size=sample_sz, random_state=42))
        results.append({"k": k, "inertia": round(float(km.inertia_), 2), "silhouette": round(sil, 4)})
    return results


def train_behavioral_segmentation(
    df: pd.DataFrame, n_clusters: int = 4, save_artifacts: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit K-Means + PCA on customer subscription spending, services, tenure, and support tickets."""
    work = df.copy()
    X_raw = work[BEHAVIORAL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    k_eval = _evaluate_k_range(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    centroids_df = pd.DataFrame(km.cluster_centers_, columns=BEHAVIORAL_FEATURES)
    name_map = _assign_behavioral_cluster_names(centroids_df)

    work["cluster_id"] = labels.astype(int)
    work["cluster_segment"] = [name_map[int(c)] for c in labels]
    work["pca_1"] = np.round(coords[:, 0], 4)
    work["pca_2"] = np.round(coords[:, 1], 4)

    sil = float(silhouette_score(X_scaled, labels, sample_size=min(2500, len(X_scaled)), random_state=42))
    profile_df = (
        work.groupby(["cluster_id", "cluster_segment"], as_index=False)
        .agg(
            customers=("customer_id", "count"),
            avg_tenure=("tenure", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_total_charges=("total_charges", "mean"),
            avg_services=("service_count", "mean"),
            avg_tickets=("total_tickets", "mean"),
            churn_rate=("actual_churn", lambda s: (s == "Yes").mean() * 100),
        )
        .round(2)
    )

    bundle = {"scaler": scaler, "kmeans": km, "pca": pca, "name_map": name_map, "features": BEHAVIORAL_FEATURES}
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "k_evaluation": k_eval,
        "cluster_names": {str(k): v for k, v in name_map.items()},
        "cluster_profiles": profile_df.to_dict(orient="records"),
    }

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, SEG_MODEL_PATH)
        SEG_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return work, metrics


def train_rfm_segmentation(
    rfm_df: pd.DataFrame, n_clusters: int = 4, save_artifacts: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit K-Means + PCA on retail warehouse RFM purchasing metrics."""
    work = rfm_df.copy()
    X_raw = work[RFM_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    k_eval = _evaluate_k_range(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    centroids_df = pd.DataFrame(km.cluster_centers_, columns=RFM_FEATURES)
    name_map = _assign_rfm_cluster_names(centroids_df)

    work["rfm_cluster_id"] = labels.astype(int)
    work["rfm_segment"] = [name_map[int(c)] for c in labels]
    work["rfm_pca_1"] = np.round(coords[:, 0], 4)
    work["rfm_pca_2"] = np.round(coords[:, 1], 4)

    sil = float(silhouette_score(X_scaled, labels, sample_size=min(2500, len(X_scaled)), random_state=42))
    profile_df = (
        work.groupby(["rfm_cluster_id", "rfm_segment"], as_index=False)
        .agg(
            customers=("customer_key", "count"),
            avg_recency=("recency", "mean"),
            avg_orders=("total_orders", "mean"),
            avg_total_sales=("total_sales", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_lifespan=("lifespan", "mean"),
        )
        .round(2)
    )

    bundle = {"scaler": scaler, "kmeans": km, "pca": pca, "name_map": name_map, "features": RFM_FEATURES}
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "k_evaluation": k_eval,
        "cluster_names": {str(k): v for k, v in name_map.items()},
        "cluster_profiles": profile_df.to_dict(orient="records"),
    }

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, RFM_MODEL_PATH)
        RFM_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return work, metrics


def load_behavioral_segmentation() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load persisted behavioral K-Means bundle and metrics."""
    bundle = joblib.load(SEG_MODEL_PATH)
    metrics = json.loads(SEG_METRICS_PATH.read_text(encoding="utf-8"))
    return bundle, metrics


def load_rfm_segmentation() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load persisted RFM K-Means bundle and metrics."""
    bundle = joblib.load(RFM_MODEL_PATH)
    metrics = json.loads(RFM_METRICS_PATH.read_text(encoding="utf-8"))
    return bundle, metrics


def assign_behavioral_segment(df: pd.DataFrame, bundle: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Assign behavioral cluster_id and cluster_segment to new or updated customers."""
    if bundle is None:
        bundle, _ = load_behavioral_segmentation()
    out = df.copy()
    X_raw = out[bundle["features"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_scaled = bundle["scaler"].transform(X_raw)
    labels = bundle["kmeans"].predict(X_scaled)
    coords = bundle["pca"].transform(X_scaled)
    out["cluster_id"] = labels.astype(int)
    out["cluster_segment"] = [bundle["name_map"][int(c)] for c in labels]
    out["pca_1"] = np.round(coords[:, 0], 4)
    out["pca_2"] = np.round(coords[:, 1], 4)
    return out
