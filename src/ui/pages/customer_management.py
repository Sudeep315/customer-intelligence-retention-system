"""Page 5: Customer Management (Create, Read, Update, Delete + Live ML Re-Scoring)."""

from __future__ import annotations

import uuid
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.data_pipeline import compute_age_group, compute_value_tier
from src.database import (
    add_customer,
    delete_customer,
    get_cached_connection,
    get_customer_by_id,
    update_customer,
)
from src.ml_churn import load_churn_model, score_single_customer
from src.ml_segmentation import assign_behavioral_segment, load_behavioral_segmentation
from src.recommender import recommend_for_customer
from src.ui.styles import render_hero, risk_badge_html


def _enrich_and_score_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived metrics, run ML churn model + K-Means segmentation before DB persistence."""
    svc_cols = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    svc_cnt = sum(1 for c in svc_cols if str(payload.get(c, "No")) == "Yes")
    if str(payload.get("internet_service", "No")) != "No":
        svc_cnt += 1

    tenure = int(payload.get("tenure", 0))
    monthly = float(payload.get("monthly_charges", 0.0))
    total = float(payload.get("total_charges", 0.0))
    admin_t = int(payload.get("num_admin_tickets", 0))
    tech_t = int(payload.get("num_tech_tickets", 0))
    age = int(payload.get("age", 35))

    payload["service_count"] = svc_cnt
    payload["total_tickets"] = admin_t + tech_t
    payload["annual_spend"] = round(monthly * 12.0, 2)
    payload["value_tier"] = compute_value_tier(tenure, total)
    payload["age_group"] = compute_age_group(age)
    payload["senior_citizen"] = 1 if age >= 60 else int(payload.get("senior_citizen", 0))

    churn_pipe, _ = load_churn_model()
    seg_bundle, _ = load_behavioral_segmentation()
    pred = score_single_customer(payload, pipe=churn_pipe)
    payload["churn_probability"] = pred["churn_probability"]
    payload["predicted_churn"] = pred["predicted_churn"]
    payload["risk_level"] = pred["risk_level"]

    seg_row = assign_behavioral_segment(pd.DataFrame([payload]), bundle=seg_bundle).iloc[0]
    payload["cluster_id"] = int(seg_row["cluster_id"])
    payload["cluster_segment"] = str(seg_row["cluster_segment"])
    return payload


def render(df: pd.DataFrame) -> None:
    """Render Customer Management CRUD workspace."""
    render_hero(
        "Customer Management & 360° CRM Registry",
        "Browse, Search, Add, Update, and Delete Customer Records with Real-Time ML Risk & Segment Scoring",
    )

    engine, backend, _ = get_cached_connection()

    tab_browse, tab_add, tab_edit, tab_delete = st.tabs(
        ["Browse & 360° Profile", "Add New Customer", "Update Customer", "Delete Customer"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Browse, Search, Export, and Inspect 360° Profile
    # ------------------------------------------------------------------
    with tab_browse:
        s_col1, s_col2, s_col3 = st.columns([2, 1, 1])
        search_q = s_col1.text_input("Search by Customer ID, Name, Email, or Country", placeholder="e.g. 7590 or Jon Yang")
        sort_by = s_col2.selectbox("Sort By", ["churn_probability", "annual_spend", "tenure", "num_tech_tickets"])
        sort_asc = s_col3.selectbox("Order", ["Descending", "Ascending"]) == "Ascending"

        view_df = df.copy()
        if search_q.strip():
            q = search_q.strip().lower()
            mask = (
                view_df["customer_id"].astype(str).str.lower().str.contains(q, na=False)
                | view_df["customer_name"].astype(str).str.lower().str.contains(q, na=False)
                | view_df["email"].astype(str).str.lower().str.contains(q, na=False)
                | view_df["country"].astype(str).str.lower().str.contains(q, na=False)
            )
            view_df = view_df[mask]

        view_df = view_df.sort_values(sort_by, ascending=sort_asc)
        st.caption(f"Showing {len(view_df):,} matching customer records from {backend}")

        cols = [
            "customer_id", "customer_name", "country", "age", "contract", "tenure",
            "monthly_charges", "total_charges", "num_tech_tickets", "value_tier",
            "cluster_segment", "churn_probability", "risk_level", "actual_churn"
        ]
        st.dataframe(
            view_df[cols].head(500),
            use_container_width=True,
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn("Churn Risk", format="%.1f%%", min_value=0.0, max_value=1.0),
                "monthly_charges": st.column_config.NumberColumn("Monthly ($)", format="$%.2f"),
                "total_charges": st.column_config.NumberColumn("Total ($)", format="$%.2f"),
            },
        )
        st.download_button(
            "Export Filtered Customers CSV",
            data=view_df.to_csv(index=False),
            file_name="filtered_customers_export.csv",
            mime="text/csv",
        )

        # 360° Customer Card Inspector
        st.divider()
        st.markdown("#### Inspect 360° Customer Intelligence Card")
        if not view_df.empty:
            options = [f"{r.customer_id} — {r.customer_name} ({r.risk_level})" for r in view_df.head(200).itertuples()]
            picked = st.selectbox("Select Customer to Inspect", options)
            picked_id = picked.split(" — ")[0]
            cust = get_customer_by_id(engine, picked_id)
            if cust:
                rec = recommend_for_customer(cust)
                p1, p2, p3 = st.columns(3)
                with p1:
                    with st.container(border=True):
                        st.markdown(f"**{cust['customer_name']}** (`{cust['customer_id']}`)")
                        st.markdown(f"Email: `{cust.get('email', 'N/A')}`")
                        st.markdown(f"Country: **{cust['country']}** | Age: **{cust['age']}** ({cust['age_group']})")
                        st.markdown(f"Gender: **{cust['gender']}** | Partner: **{cust['partner']}** | Dependents: **{cust['dependents']}**")
                with p2:
                    with st.container(border=True):
                        st.markdown("**Subscription & Spend**")
                        st.markdown(f"Contract: **{cust['contract']}** | Tenure: **{cust['tenure']} mos**")
                        st.markdown(f"Internet: **{cust['internet_service']}** | Services: **{cust['service_count']}**")
                        st.markdown(f"Monthly: **${cust['monthly_charges']:.2f}** | Total: **${cust['total_charges']:,.2f}**")
                        st.markdown(f"Tickets: **{cust['num_tech_tickets']} Tech** / **{cust['num_admin_tickets']} Admin**")
                with p3:
                    with st.container(border=True):
                        st.markdown("**ML Risk & Retention Strategy**")
                        st.markdown(f"Risk: {risk_badge_html(cust['risk_level'])} (`{cust['churn_probability']*100:.1f}%`)", unsafe_allow_html=True)
                        st.markdown(f"Segment: **{cust['cluster_segment']}** (`{cust['value_tier']}`)")
                        st.markdown(f"Recommended Action: **{rec['recommended_strategy']}**")
                        st.markdown(f"Annual Risk: **${rec['revenue_at_risk']:,.0f}**")

    # ------------------------------------------------------------------
    # TAB 2: Add New Customer (Create)
    # ------------------------------------------------------------------
    with tab_add:
        st.markdown("Create a new customer record. Churn risk and segment are automatically computed via ML upon saving.")
        default_cid = f"NEW-{uuid.uuid4().hex[:5].upper()}"
        with st.form("add_customer_form"):
            a1, a2, a3 = st.columns(3)
            with a1:
                cid = st.text_input("Customer ID", value=default_cid)
                cname = st.text_input("Full Name", value="Priya Nair")
                cemail = st.text_input("Email", value="priya.nair@example.com")
                ccountry = st.selectbox("Country", ["United States", "United Kingdom", "Canada", "Australia", "Germany", "France", "India"])
                cgender = st.selectbox("Gender", ["Female", "Male"])
                cage = st.number_input("Age", min_value=18, max_value=100, value=34)
                cpartner = st.selectbox("Partner", ["Yes", "No"])
                cdep = st.selectbox("Dependents", ["No", "Yes"])
            with a2:
                ctenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=4)
                ccontract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                cinternet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                cpayment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                cpaper = st.selectbox("Paperless Billing", ["Yes", "No"])
                cmonthly = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=500.0, value=79.50)
                ctotal = st.number_input("Total Charges ($)", min_value=0.0, max_value=50000.0, value=318.00)
            with a3:
                cphone = st.selectbox("Phone Service", ["Yes", "No"])
                cmulti = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                csec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                cback = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                cprot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                ctech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                ctv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                cmov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                t1, t2 = st.columns(2)
                cnt_admin = t1.number_input("Admin Tickets", min_value=0, max_value=30, value=0)
                cnt_tech = t2.number_input("Tech Tickets", min_value=0, max_value=30, value=1)

            add_btn = st.form_submit_button("Add Customer & Score with ML", type="primary", use_container_width=True)

        if add_btn:
            cid_clean = cid.strip()
            if not cid_clean or not cname.strip():
                st.error("Customer ID and Full Name are required.")
            elif get_customer_by_id(engine, cid_clean) is not None:
                st.error(f"Customer ID `{cid_clean}` already exists. Please use a unique ID.")
            else:
                new_payload: Dict[str, Any] = {
                    "customer_id": cid_clean,
                    "customer_name": cname.strip(),
                    "email": cemail.strip(),
                    "country": ccountry,
                    "age": int(cage),
                    "marital_status": "Married" if cpartner == "Yes" else "Single",
                    "gender": cgender,
                    "senior_citizen": 1 if cage >= 60 else 0,
                    "partner": cpartner,
                    "dependents": cdep,
                    "tenure": int(ctenure),
                    "phone_service": cphone,
                    "multiple_lines": cmulti if cphone == "Yes" else "No phone service",
                    "internet_service": cinternet,
                    "online_security": csec if cinternet != "No" else "No internet service",
                    "online_backup": cback if cinternet != "No" else "No internet service",
                    "device_protection": cprot if cinternet != "No" else "No internet service",
                    "tech_support": ctech if cinternet != "No" else "No internet service",
                    "streaming_tv": ctv if cinternet != "No" else "No internet service",
                    "streaming_movies": cmov if cinternet != "No" else "No internet service",
                    "contract": ccontract,
                    "paperless_billing": cpaper,
                    "payment_method": cpayment,
                    "monthly_charges": float(cmonthly),
                    "total_charges": float(ctotal),
                    "num_admin_tickets": int(cnt_admin),
                    "num_tech_tickets": int(cnt_tech),
                    "actual_churn": "No",
                }
                scored_payload = _enrich_and_score_record(new_payload)
                add_customer(engine, scored_payload)
                st.cache_data.clear()
                st.success(
                    f"Added `{cid_clean}` ({cname}) — Churn Risk: **{scored_payload['risk_level']}** "
                    f"({scored_payload['churn_probability']*100:.1f}%), Segment: **{scored_payload['cluster_segment']}**."
                )

    # ------------------------------------------------------------------
    # TAB 3: Update Existing Customer
    # ------------------------------------------------------------------
    with tab_edit:
        if df.empty:
            st.info("No records available to edit.")
        else:
            edit_lookup = st.text_input("Filter Customer ID/Name to Edit", value="", key="edit_filter")
            edit_pool = df
            if edit_lookup.strip():
                q = edit_lookup.strip().lower()
                edit_pool = df[
                    df["customer_id"].astype(str).str.lower().str.contains(q, na=False)
                    | df["customer_name"].astype(str).str.lower().str.contains(q, na=False)
                ]
            if edit_pool.empty:
                st.warning("No matching customer found.")
            else:
                edit_options = [f"{r.customer_id} — {r.customer_name}" for r in edit_pool.head(150).itertuples()]
                chosen_edit = st.selectbox("Select Customer to Update", edit_options, key="edit_select")
                edit_cid = chosen_edit.split(" — ")[0]
                cur = get_customer_by_id(engine, edit_cid)
                if cur:
                    with st.form(f"edit_form_{edit_cid}"):
                        e1, e2, e3 = st.columns(3)
                        u_name = e1.text_input("Full Name", value=str(cur["customer_name"]))
                        u_country = e1.text_input("Country", value=str(cur["country"]))
                        u_age = e1.number_input("Age", min_value=18, max_value=100, value=int(cur["age"]))
                        u_tenure = e2.number_input("Tenure (Months)", min_value=0, max_value=120, value=int(cur["tenure"]))
                        contracts = ["Month-to-month", "One year", "Two year"]
                        u_contract = e2.selectbox("Contract", contracts, index=contracts.index(cur["contract"]) if cur["contract"] in contracts else 0)
                        pays = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
                        u_payment = e2.selectbox("Payment Method", pays, index=pays.index(cur["payment_method"]) if cur["payment_method"] in pays else 0)
                        u_monthly = e3.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=float(cur["monthly_charges"]))
                        u_total = e3.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=float(cur["total_charges"]))
                        u_tech_t = e3.number_input("Tech Tickets", min_value=0, max_value=50, value=int(cur["num_tech_tickets"]))
                        u_admin_t = e3.number_input("Admin Tickets", min_value=0, max_value=50, value=int(cur["num_admin_tickets"]))
                        yn_opts = ["No", "Yes", "No internet service"]
                        u_sec = e1.selectbox("Online Security", yn_opts, index=yn_opts.index(cur["online_security"]) if cur["online_security"] in yn_opts else 0)
                        u_tsup = e2.selectbox("Tech Support", yn_opts, index=yn_opts.index(cur["tech_support"]) if cur["tech_support"] in yn_opts else 0)
                        churn_opts = ["No", "Yes", "Unknown"]
                        u_actual = e3.selectbox("Actual Churn Status", churn_opts, index=churn_opts.index(cur["actual_churn"]) if cur["actual_churn"] in churn_opts else 0)

                        save_btn = st.form_submit_button("Save Changes & Re-Score with ML", type="primary", use_container_width=True)

                    if save_btn:
                        merged = dict(cur)
                        merged.update(
                            {
                                "customer_name": u_name.strip(),
                                "country": u_country.strip(),
                                "age": int(u_age),
                                "tenure": int(u_tenure),
                                "contract": u_contract,
                                "payment_method": u_payment,
                                "monthly_charges": float(u_monthly),
                                "total_charges": float(u_total),
                                "num_tech_tickets": int(u_tech_t),
                                "num_admin_tickets": int(u_admin_t),
                                "online_security": u_sec,
                                "tech_support": u_tsup,
                                "actual_churn": u_actual,
                            }
                        )
                        rescored = _enrich_and_score_record(merged)
                        update_customer(engine, edit_cid, rescored)
                        st.cache_data.clear()
                        st.success(
                            f"Updated `{edit_cid}` — New Churn Risk: **{rescored['risk_level']}** "
                            f"({rescored['churn_probability']*100:.1f}%), Segment: **{rescored['cluster_segment']}**."
                        )

    # ------------------------------------------------------------------
    # TAB 4: Delete Customer
    # ------------------------------------------------------------------
    with tab_delete:
        if df.empty:
            st.info("No records available.")
        else:
            del_lookup = st.text_input("Filter Customer ID/Name to Delete", value="", key="del_filter")
            del_pool = df
            if del_lookup.strip():
                q = del_lookup.strip().lower()
                del_pool = df[
                    df["customer_id"].astype(str).str.lower().str.contains(q, na=False)
                    | df["customer_name"].astype(str).str.lower().str.contains(q, na=False)
                ]
            if del_pool.empty:
                st.warning("No matching customer found.")
            else:
                del_options = [f"{r.customer_id} — {r.customer_name} ({r.risk_level})" for r in del_pool.head(150).itertuples()]
                chosen_del = st.selectbox("Select Customer to Delete", del_options, key="del_select")
                del_cid = chosen_del.split(" — ")[0]
                st.warning(f"Deleting `{del_cid}` permanently removes the customer and any linked retention logs.")
                confirm_del = st.checkbox(f"I confirm deletion of customer `{del_cid}`", key="confirm_del_chk")
                if st.button("Delete Customer Record", type="primary", disabled=not confirm_del):
                    if delete_customer(engine, del_cid):
                        st.cache_data.clear()
                        st.success(f"Customer `{del_cid}` deleted from database.")
                        st.rerun()
                    else:
                        st.error("Customer could not be found or deleted.")
