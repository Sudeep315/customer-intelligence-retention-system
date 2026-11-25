"""AI Retention Recommendation Engine, ROI Estimator, and Personalized Outreach Generator."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd


def recommend_for_customer(row: Dict[str, Any]) -> Dict[str, Any]:
    """Generate tailored retention strategies, expected ROI, and outreach email for a customer."""
    cid = str(row.get("customer_id", "CUST"))
    name = str(row.get("customer_name", "Valued Customer"))
    prob = float(row.get("churn_probability", 0.0) or 0.0)
    risk_level = str(row.get("risk_level", "Low Risk"))
    monthly = float(row.get("monthly_charges", 0.0) or 0.0)
    annual = float(row.get("annual_spend", monthly * 12.0) or 0.0)
    tenure = int(row.get("tenure", 0) or 0)
    contract = str(row.get("contract", "Month-to-month"))
    internet = str(row.get("internet_service", "DSL"))
    payment = str(row.get("payment_method", "Electronic check"))
    tech_tickets = int(row.get("num_tech_tickets", 0) or 0)
    admin_tickets = int(row.get("num_admin_tickets", 0) or 0)
    tech_support = str(row.get("tech_support", "No"))
    online_sec = str(row.get("online_security", "No"))
    value_tier = str(row.get("value_tier", "New"))
    segment = str(row.get("cluster_segment", "Core"))

    tactics: List[str] = []
    offer_headline = "Proactive Value Check-In"
    save_rate = 0.25

    # Determine dominant friction & tailored tactics
    if tech_tickets >= 2:
        offer_headline = "Priority Engineering Escalation + 2-Month SLA Credit"
        tactics.append(
            f"Assign dedicated Tier-2 technical specialist within 24 hours to resolve {tech_tickets} open/recent tech issues."
        )
        tactics.append("Issue proactive 20% service credit for 2 billing cycles until service stability KPIs recover.")
        save_rate = 0.52
    elif contract == "Month-to-month" and prob >= 0.55:
        offer_headline = "1-Year Contract Upgrade (15% Rate Lock + Free Add-On)"
        tactics.append(
            "Offer 15% discount to migrate from Month-to-month to a 1-year contract with zero switching fees."
        )
        save_rate = 0.48
    elif tenure < 12 and prob >= 0.45:
        offer_headline = "Early-Lifecycle Onboarding Concierge + Loyalty Bonus"
        tactics.append(
            f"Enroll in 90-day onboarding check-in program (tenure is {tenure} mos; highest early-churn window)."
        )
        tactics.append("Grant 3 months complimentary Tech Support to remove early setup friction.")
        save_rate = 0.45
    elif value_tier == "VIP" or annual >= 1000:
        offer_headline = "VIP Account Manager Review + Loyalty Perks"
        tactics.append("Schedule executive QBR and unlock VIP priority support queue.")
        save_rate = 0.55

    if internet == "Fiber optic" and (tech_support == "No" or online_sec == "No"):
        tactics.append(
            "Bundle Online Security + Tech Support at 50% off ($5/mo trial) to raise product stickiness on Fiber."
        )
        save_rate = min(0.65, save_rate + 0.06)

    if payment == "Electronic check":
        tactics.append(
            "Incentivize switch from Electronic check to Auto-Pay (Credit Card / Bank Transfer) with a $10 one-time bill credit."
        )
        save_rate = min(0.68, save_rate + 0.05)

    if admin_tickets >= 1:
        tactics.append(
            f"Audit past {admin_tickets} billing/admin ticket(s) and clarify invoice line items to prevent surprise-bill churn."
        )

    if monthly >= 80.0 and prob >= 0.50:
        tactics.append(
            f"Run plan-rightsizing review: current monthly spend (${monthly:.2f}) exceeds peer median; propose tailored bundle."
        )

    if not tactics:
        tactics.append("Maintain quarterly engagement newsletter and loyalty reward milestone tracking.")
        tactics.append("Offer referral bonus ($25 credit) to convert high satisfaction into advocacy.")

    if prob >= 0.75:
        priority = "Critical"
    elif prob >= 0.60:
        priority = "High"
    elif prob >= 0.35:
        priority = "Medium"
    else:
        priority = "Low"

    revenue_at_risk = round(annual * prob, 2)
    expected_saved = round(revenue_at_risk * save_rate, 2)

    first_name = name.split()[0] if name.strip() else "Customer"
    email_subject = f"Personal Account Review & Exclusive Loyalty Benefit for {first_name} ({cid})"
    email_body = (
        f"Hi {first_name},\n\n"
        f"Thank you for being part of our community ({tenure} months with us). "
        f"While reviewing your {internet if internet != 'No' else 'service'} account, we noticed "
        + (
            f"you recently experienced {tech_tickets} technical support interaction(s), and we want to make sure everything is running flawlessly.\n\n"
            if tech_tickets > 0
            else "there may be opportunities to optimize your current monthly plan and unlock extra protection.\n\n"
        )
        + f"We would like to extend our **{offer_headline}**:\n"
        + "\n".join(f"- {t}" for t in tactics[:3])
        + "\n\nReply directly to this message or click 'Claim Retention Offer' in your portal and our dedicated team will apply it immediately.\n\n"
        "Warm regards,\nCustomer Retention & Success Team"
    )

    return {
        "customer_id": cid,
        "customer_name": name,
        "risk_level": risk_level,
        "churn_probability": round(prob, 4),
        "priority": priority,
        "recommended_strategy": offer_headline,
        "estimated_save_rate": round(save_rate, 2),
        "revenue_at_risk": revenue_at_risk,
        "expected_saved_revenue": expected_saved,
        "tactics": tactics,
        "action_notes": " | ".join(tactics[:2]),
        "email_subject": email_subject,
        "email_body": email_body,
        "segment": segment,
        "value_tier": value_tier,
    }


def build_initial_retention_queue(customers_df: pd.DataFrame, top_n: int = 40) -> List[Dict[str, Any]]:
    """Build prioritized retention action entries for top high-risk, high-value customers."""
    high_risk = customers_df[customers_df["churn_probability"] >= 0.60].copy()
    if high_risk.empty:
        high_risk = customers_df.sort_values("churn_probability", ascending=False).head(top_n)
    else:
        high_risk["rev_risk_score"] = high_risk["annual_spend"] * high_risk["churn_probability"]
        high_risk = high_risk.sort_values("rev_risk_score", ascending=False).head(top_n)

    queue: List[Dict[str, Any]] = []
    for _, row in high_risk.iterrows():
        rec = recommend_for_customer(row.to_dict())
        queue.append(
            {
                "customer_id": rec["customer_id"],
                "risk_level": rec["risk_level"],
                "churn_probability": rec["churn_probability"],
                "recommended_strategy": rec["recommended_strategy"],
                "priority": rec["priority"],
                "estimated_save_rate": rec["estimated_save_rate"],
                "revenue_at_risk": rec["revenue_at_risk"],
                "action_notes": rec["action_notes"],
                "status": "Recommended",
            }
        )
    return queue


def generate_segment_playbooks(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategic retention playbooks across ML customer segments."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for seg, grp in df.groupby("cluster_segment"):
        avg_prob = float(grp["churn_probability"].mean())
        rev_risk = float((grp["annual_spend"] * grp["churn_probability"]).sum())
        avg_tickets = float(grp["total_tickets"].mean())
        avg_tenure = float(grp["tenure"].mean())
        pct_mtm = float((grp["contract"] == "Month-to-month").mean() * 100)

        if "At-Risk" in seg or avg_prob >= 0.50:
            playbook = "Immediate tech escalation + 1-year migration incentive + Auto-Pay credit"
            kpi_target = "Reduce tech friction & lock 35% into annual plans"
        elif "Explorers" in seg or avg_tenure < 15:
            playbook = "90-day guided onboarding + bundled security/support free trial"
            kpi_target = "Improve Month-12 retention by +18%"
        elif "Champions" in seg:
            playbook = "VIP concierge perks, multi-service loyalty discounts & referral rewards"
            kpi_target = "Maintain >95% annual retention & upsell streaming/backup"
        else:
            playbook = "Targeted upgrade nudges to Fiber/Security bundles + annual billing rewards"
            kpi_target = "Increase ARPU by +12% while keeping churn <10%"

        rows.append(
            {
                "Segment": seg,
                "Customers": int(len(grp)),
                "Avg Churn Risk (%)": round(avg_prob * 100, 1),
                "Avg Tenure (Mos)": round(avg_tenure, 1),
                "Month-to-Month (%)": round(pct_mtm, 1),
                "Avg Tickets": round(avg_tickets, 2),
                "Annual Revenue at Risk ($)": round(rev_risk, 0),
                "Recommended Playbook": playbook,
                "Target Outcome": kpi_target,
            }
        )
    return pd.DataFrame(rows).sort_values("Annual Revenue at Risk ($)", ascending=False)


def optional_llm_refine_strategy(customer_rec: Dict[str, Any]) -> str | None:
    """Optionally enrich strategy via OpenAI if OPENAI_API_KEY is configured; returns None otherwise."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            f"Customer {customer_rec['customer_name']} ({customer_rec['customer_id']}) has "
            f"{customer_rec['churn_probability']*100:.1f}% churn risk, annual revenue at risk "
            f"${customer_rec['revenue_at_risk']:.0f}, segment '{customer_rec['segment']}'. "
            f"Primary action: {customer_rec['recommended_strategy']}. Tactics: {customer_rec['tactics']}. "
            "Write 3 concise executive bullet points for an account manager retention call."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception:
        return None
