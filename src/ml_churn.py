"""Customer Churn Prediction ML pipeline, model comparison, scoring, and explainability."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CHURN_METRICS_PATH, CHURN_MODEL_PATH, MODELS_DIR, RISK_THRESHOLDS
from src.data_pipeline import to_model_feature_frame


def probability_to_risk_level(prob: float) -> str:
    """Map predicted churn probability to business risk tier."""
    if prob >= RISK_THRESHOLDS["high"]:
        return "High Risk"
    if prob >= RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "Low Risk"


def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def _extract_feature_importance(pipe: Pipeline, top_n: int = 15) -> List[Dict[str, Any]]:
    pre: ColumnTransformer = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]
    feature_names = pre.get_feature_names_out()
    clean_names = [fn.replace("num__", "").replace("cat__", "") for fn in feature_names]

    if hasattr(clf, "feature_importances_"):
        raw_imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        raw_imp = np.abs(clf.coef_[0])
    else:
        raw_imp = np.zeros(len(clean_names))

    total = float(np.sum(raw_imp)) or 1.0
    norm_imp = raw_imp / total
    pairs = sorted(zip(clean_names, norm_imp), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": name, "importance": round(float(val), 4)} for name, val in pairs]


def train_churn_models(df: pd.DataFrame, save_artifacts: bool = True) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train & benchmark multiple classifiers, select best model by ROC-AUC, and persist artifacts."""
    train_df = df[df["actual_churn"].isin(["Yes", "No"])].copy()
    y = (train_df["actual_churn"] == "Yes").astype(int)
    X = to_model_feature_frame(train_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor, num_cols, cat_cols = _build_preprocessor(X_train)

    candidates = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=220, learning_rate=0.05, max_depth=4, subsample=0.9, random_state=42
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=260,
            max_depth=11,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        ),
    }

    leaderboard: List[Dict[str, Any]] = []
    fitted_pipes: Dict[str, Pipeline] = {}
    roc_curves: Dict[str, Dict[str, List[float]]] = {}
    confusion_matrices: Dict[str, List[List[int]]] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)

        metrics = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
        }
        leaderboard.append(metrics)
        fitted_pipes[name] = pipe

        fpr, tpr, _ = roc_curve(y_test, probs)
        # Downsample ROC points for compact JSON storage while keeping shape smooth
        step = max(1, len(fpr) // 60)
        roc_curves[name] = {
            "fpr": [round(float(v), 4) for v in fpr[::step]] + [1.0],
            "tpr": [round(float(v), 4) for v in tpr[::step]] + [1.0],
        }
        confusion_matrices[name] = confusion_matrix(y_test, preds).tolist()

    leaderboard.sort(key=lambda item: (item["roc_auc"], item["f1_score"]), reverse=True)
    best_name = leaderboard[0]["model"]
    best_pipe = fitted_pipes[best_name]

    # Prefer GradientBoosting or RandomForest feature importances if best model lacks built-in importances
    if hasattr(best_pipe.named_steps["classifier"], "feature_importances_") or hasattr(
        best_pipe.named_steps["classifier"], "coef_"
    ):
        top_features = _extract_feature_importance(best_pipe, top_n=15)
    else:
        fallback_name = "Gradient Boosting" if "Gradient Boosting" in fitted_pipes else best_name
        top_features = _extract_feature_importance(fitted_pipes[fallback_name], top_n=15)

    artifacts_meta: Dict[str, Any] = {
        "best_model": best_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "leaderboard": leaderboard,
        "roc_curves": roc_curves,
        "confusion_matrices": confusion_matrices,
        "top_features": top_features,
    }

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipe, CHURN_MODEL_PATH)
        CHURN_METRICS_PATH.write_text(json.dumps(artifacts_meta, indent=2), encoding="utf-8")

    return best_pipe, artifacts_meta


def load_churn_model() -> Tuple[Pipeline, Dict[str, Any]]:
    """Load persisted churn ML pipeline and benchmark metadata."""
    if not CHURN_MODEL_PATH.exists() or not CHURN_METRICS_PATH.exists():
        raise FileNotFoundError("Churn model artifacts not found. Run scripts/init_project.py first.")
    pipe = joblib.load(CHURN_MODEL_PATH)
    meta = json.loads(CHURN_METRICS_PATH.read_text(encoding="utf-8"))
    return pipe, meta


def score_customers(df: pd.DataFrame, pipe: Optional[Pipeline] = None) -> pd.DataFrame:
    """Score customer dataframe with churn_probability, predicted_churn, and risk_level."""
    if pipe is None:
        pipe, _ = load_churn_model()
    scored = df.copy()
    X = to_model_feature_frame(scored)
    probs = pipe.predict_proba(X)[:, 1]
    scored["churn_probability"] = np.round(probs, 4)
    scored["predicted_churn"] = np.where(probs >= 0.50, "Yes", "No")
    scored["risk_level"] = [probability_to_risk_level(float(p)) for p in probs]
    return scored


def score_single_customer(record: Dict[str, Any], pipe: Optional[Pipeline] = None) -> Dict[str, Any]:
    """Predict churn probability, risk level, and human-readable explanations for one customer."""
    df = pd.DataFrame([record])
    scored = score_customers(df, pipe=pipe).iloc[0]
    prob = float(scored["churn_probability"])
    risk = str(scored["risk_level"])
    pred = str(scored["predicted_churn"])
    drivers = explain_customer_churn(record)
    return {
        "churn_probability": prob,
        "predicted_churn": pred,
        "risk_level": risk,
        "risk_drivers": drivers["risk_drivers"],
        "protective_factors": drivers["protective_factors"],
    }


def explain_customer_churn(row: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate transparent domain + ML-aligned explanations for a customer's churn risk."""
    risk_drivers: List[str] = []
    protective_factors: List[str] = []

    tech_tickets = int(row.get("num_tech_tickets", 0) or 0)
    admin_tickets = int(row.get("num_admin_tickets", 0) or 0)
    tenure = int(row.get("tenure", 0) or 0)
    monthly = float(row.get("monthly_charges", 0.0) or 0.0)
    contract = str(row.get("contract", ""))
    internet = str(row.get("internet_service", ""))
    payment = str(row.get("payment_method", ""))
    tech_support = str(row.get("tech_support", "No"))
    security = str(row.get("online_security", "No"))
    backup = str(row.get("online_backup", "No"))

    if tech_tickets >= 1:
        risk_drivers.append(f"Elevated technical friction ({tech_tickets} technical support tickets)")
    if admin_tickets >= 2:
        risk_drivers.append(f"Billing/admin friction ({admin_tickets} administrative tickets)")
    if contract == "Month-to-month":
        risk_drivers.append("Flexible Month-to-month contract (42.7% historical churn cohort)")
    if tenure < 12:
        risk_drivers.append(f"Early lifecycle stage ({tenure} months tenure; highest churn window)")
    if internet == "Fiber optic":
        risk_drivers.append("Fiber optic tier exhibits higher price/expectation sensitivity")
    if payment == "Electronic check":
        risk_drivers.append("Manual Electronic check payment (45.3% historical churn rate)")
    if monthly >= 75.0:
        risk_drivers.append(f"High monthly billing exposure (${monthly:.2f}/mo)")
    if internet != "No" and tech_support == "No" and security == "No":
        risk_drivers.append("Lacks sticky protection add-ons (Online Security & Tech Support)")

    if contract in ("One year", "Two year"):
        protective_factors.append(f"{contract} contract lock-in strongly stabilizes retention")
    if tenure >= 24:
        protective_factors.append(f"Established loyalty ({tenure} months tenure)")
    if "automatic" in payment.lower():
        protective_factors.append(f"Auto-pay enabled ({payment})")
    if tech_support == "Yes" or security == "Yes" or backup == "Yes":
        protective_factors.append("Subscribed to sticky value-add protection/support services")
    if tech_tickets == 0 and admin_tickets == 0:
        protective_factors.append("Zero support or billing escalations logged")

    return {"risk_drivers": risk_drivers, "protective_factors": protective_factors}
