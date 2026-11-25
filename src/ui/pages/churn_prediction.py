"""Page 2: Customer Churn Prediction Simulator, Batch Scoring, and Model Diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_pipeline import compute_age_group, compute_value_tier, normalize_churn_dataframe
from src.ml_churn import load_churn_model, score_customers, score_single_customer
from src.ml_segmentation import assign_behavioral_segment, load_behavioral_segmentation
from src.recommender import recommend_for_customer
from src.ui.charts import chart_confusion_matrix, chart_feature_importance, chart_roc_curves
from src.ui.styles import render_hero, render_kpi_card, risk_badge_html


def render(df: pd.DataFrame) -> None:
    """Render Churn Prediction & ML Risk Lab page."""
    render_hero(
        "Customer Churn Prediction & ML Risk Lab",
        "Predict Individual & Batch Churn Probabilities, Inspect Model Benchmarks, and Explain Risk Drivers",
    )

    try:
        churn_pipe, churn_meta = load_churn_model()
        seg_bundle, _ = load_behavioral_segmentation()
    except Exception as exc:
        st.error(f"ML model artifacts are missing or corrupted: {exc}")
        st.info("Run `python scripts/init_project.py` to train and save the models.")
        return

    tab_sim, tab_batch, tab_eval = st.tabs(
        ["Single Customer Risk Simulator", "Batch CSV / Excel Scoring", "Model Leaderboard & Diagnostics"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Interactive Single-Customer Simulator
    # ------------------------------------------------------------------
    with tab_sim:
        st.markdown("Configure a customer profile below to simulate real-time churn probability and retention actions.")
        with st.form("churn_simulator_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                customer_name = st.text_input("Customer Name", value="Alex Morgan")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (60+)" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                tenure = st.slider("Tenure (Months)", 0, 72, 6)
            with c2:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                payment = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=200.0, value=84.5, step=1.0)
                total_charges = st.number_input(
                    "Total Charges ($)", min_value=0.0, max_value=15000.0, value=float(round(tenure * monthly_charges, 2)), step=10.0
                )
            with c3:
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                t_col1, t_col2 = st.columns(2)
                num_tech = t_col1.number_input("Tech Tickets", min_value=0, max_value=20, value=2)
                num_admin = t_col2.number_input("Admin Tickets", min_value=0, max_value=20, value=1)

            submitted = st.form_submit_button("Run ML Churn Prediction", type="primary", use_container_width=True)

        if submitted or "last_sim_result" not in st.session_state:
            svc_flags = [phone, multi, sec, backup, prot, tech_sup, tv, movies]
            svc_cnt = sum(1 for v in svc_flags if v == "Yes") + (1 if internet != "No" else 0)
            payload = {
                "customer_id": "SIM-PREVIEW",
                "customer_name": customer_name,
                "gender": gender,
                "senior_citizen": senior,
                "partner": partner,
                "dependents": dependents,
                "tenure": tenure,
                "phone_service": phone,
                "multiple_lines": multi if phone == "Yes" else "No phone service",
                "internet_service": internet,
                "online_security": sec if internet != "No" else "No internet service",
                "online_backup": backup if internet != "No" else "No internet service",
                "device_protection": prot if internet != "No" else "No internet service",
                "tech_support": tech_sup if internet != "No" else "No internet service",
                "streaming_tv": tv if internet != "No" else "No internet service",
                "streaming_movies": movies if internet != "No" else "No internet service",
                "contract": contract,
                "paperless_billing": paperless,
                "payment_method": payment,
                "monthly_charges": float(monthly_charges),
                "total_charges": float(total_charges),
                "num_admin_tickets": int(num_admin),
                "num_tech_tickets": int(num_tech),
                "service_count": svc_cnt,
                "total_tickets": int(num_admin + num_tech),
                "annual_spend": round(float(monthly_charges) * 12.0, 2),
                "value_tier": compute_value_tier(int(tenure), float(total_charges)),
                "age": 65 if senior == 1 else 36,
                "age_group": compute_age_group(65 if senior == 1 else 36),
            }
            pred_res = score_single_customer(payload, pipe=churn_pipe)
            seg_res = assign_behavioral_segment(pd.DataFrame([payload]), bundle=seg_bundle).iloc[0]
            payload.update(pred_res)
            payload["cluster_segment"] = str(seg_res["cluster_segment"])
            rec = recommend_for_customer(payload)
            st.session_state["last_sim_result"] = (payload, pred_res, rec)

        payload, pred_res, rec = st.session_state["last_sim_result"]
        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        prob_pct = pred_res["churn_probability"] * 100
        accent = "#EF4444" if pred_res["risk_level"] == "High Risk" else ("#F59E0B" if pred_res["risk_level"] == "Medium Risk" else "#10B981")
        with r1:
            render_kpi_card("Predicted Churn Probability", f"{prob_pct:.1f}%", f"Prediction: {pred_res['predicted_churn']}", accent=accent)
        with r2:
            render_kpi_card("Risk Classification", pred_res["risk_level"], f"Priority: {rec['priority']}", accent=accent)
        with r3:
            render_kpi_card("Assigned ML Segment", payload["cluster_segment"], f"Value Tier: {payload['value_tier']}", accent="#6366F1")
        with r4:
            render_kpi_card("Annual Revenue at Risk", f"${rec['revenue_at_risk']:,.0f}", f"Est. Saveable: ${rec['expected_saved_revenue']:,.0f}", accent="#0D9488")

        st.markdown("")
        d_left, d_right = st.columns(2)
        with d_left:
            with st.container(border=True):
                st.markdown("##### Key Risk Drivers & Protective Signals")
                st.markdown(f"Risk Status: {risk_badge_html(pred_res['risk_level'])}", unsafe_allow_html=True)
                if pred_res["risk_drivers"]:
                    st.markdown("**Risk Escalators:**")
                    for drv in pred_res["risk_drivers"]:
                        st.markdown(f"- :red[{drv}]")
                if pred_res["protective_factors"]:
                    st.markdown("**Protective Factors:**")
                    for prot_f in pred_res["protective_factors"]:
                        st.markdown(f"- :green[{prot_f}]")
        with d_right:
            with st.container(border=True):
                st.markdown("##### Recommended AI Retention Action")
                st.markdown(f"**Primary Offer:** `{rec['recommended_strategy']}`")
                st.markdown(f"**Expected Save Rate:** `{rec['estimated_save_rate']*100:.0f}%`")
                for tac in rec["tactics"]:
                    st.markdown(f"- {tac}")

    # ------------------------------------------------------------------
    # TAB 2: Batch CSV / Excel Scoring
    # ------------------------------------------------------------------
    with tab_batch:
        st.markdown("Upload a CSV or Excel file of customers to run bulk ML churn scoring and segment assignment.")
        if not df.empty:
            sample_csv = df.head(25)[
                [
                    "customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure",
                    "phone_service", "multiple_lines", "internet_service", "online_security",
                    "online_backup", "device_protection", "tech_support", "streaming_tv",
                    "streaming_movies", "contract", "paperless_billing", "payment_method",
                    "monthly_charges", "total_charges", "num_admin_tickets", "num_tech_tickets",
                ]
            ].to_csv(index=False)
            st.download_button(
                "Download Sample Input CSV (25 Customers)",
                data=sample_csv,
                file_name="sample_customers_for_scoring.csv",
                mime="text/csv",
            )

        uploaded = st.file_uploader("Upload Customer Dataset (.csv or .xlsx)", type=["csv", "xlsx"])
        if uploaded is not None:
            try:
                raw_in = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                norm_df = normalize_churn_dataframe(raw_in)
                scored = score_customers(norm_df, pipe=churn_pipe)
                scored = assign_behavioral_segment(scored, bundle=seg_bundle)

                b1, b2, b3 = st.columns(3)
                b1.metric("Scored Rows", f"{len(scored):,}")
                b2.metric("High-Risk Accounts", f"{int((scored['risk_level'] == 'High Risk').sum()):,}")
                b3.metric("Avg Churn Probability", f"{scored['churn_probability'].mean()*100:.1f}%")

                show_cols = [
                    "customer_id", "tenure", "contract", "monthly_charges", "num_tech_tickets",
                    "cluster_segment", "value_tier", "churn_probability", "predicted_churn", "risk_level"
                ]
                st.dataframe(scored[show_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Scored Predictions CSV",
                    data=scored.to_csv(index=False),
                    file_name="churn_scored_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
            except Exception as exc:
                st.error(f"Could not score uploaded file: {exc}")

    # ------------------------------------------------------------------
    # TAB 3: Model Leaderboard & Diagnostics
    # ------------------------------------------------------------------
    with tab_eval:
        st.markdown(f"**Active Champion Model:** `{churn_meta['best_model']}` | Train Rows: `{churn_meta['train_rows']:,}` | Test Holdout: `{churn_meta['test_rows']:,}`")
        lb_df = pd.DataFrame(churn_meta["leaderboard"])
        st.dataframe(
            lb_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                "f1_score": st.column_config.NumberColumn("F1 Score", format="%.4f"),
                "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
            },
        )

        col_roc, col_imp = st.columns(2)
        with col_roc:
            st.plotly_chart(chart_roc_curves(churn_meta["roc_curves"], churn_meta["leaderboard"]), use_container_width=True)
        with col_imp:
            st.plotly_chart(chart_feature_importance(churn_meta["top_features"]), use_container_width=True)

        selected_cm_model = st.selectbox(
            "Inspect Confusion Matrix for Model",
            options=[m["model"] for m in churn_meta["leaderboard"]],
        )
        cm = churn_meta["confusion_matrices"][selected_cm_model]
        st.plotly_chart(chart_confusion_matrix(cm, selected_cm_model), use_container_width=True)
