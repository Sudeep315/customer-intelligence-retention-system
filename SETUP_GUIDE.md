# Step-by-Step Installation, MySQL Setup & Execution Guide

This guide walks you through installing dependencies, setting up MySQL 8.0 (or running immediately in SQLite auto-fallback mode), initializing the ML models, and running the **Customer Intelligence & Retention System** for development or final-year project demonstration.

---

## 1. Prerequisites

- **Python 3.10+** (verified with Python 3.12)
- **MySQL Server 8.0+** (recommended for full relational DB mode; optional for quick preview because the app automatically falls back to a local SQLite mirror if MySQL is not running)
- **Git / VS Code / Terminal**

---

## 2. Python Environment Setup

Open a terminal inside the project directory (`CCP1`) and install all required dependencies:

```bash
# Optional: create & activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# Install packages
python -m pip install -r requirements.txt
```

---

## 3. MySQL Database Setup (Recommended)

### Option A: Automated Python + MySQL Setup
1. Ensure **MySQL 8.0 Server** is installed and running on `127.0.0.1:3306`.
2. Copy `.env.example` to `.env` and update your MySQL password:
   ```ini
   DB_BACKEND=mysql
   MYSQL_HOST=127.0.0.1
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_actual_password
   MYSQL_DATABASE=customer_intelligence_db
   ```
3. Run the full ML training + database seeding script:
   ```bash
   python scripts/init_project.py
   ```
   Or seed MySQL separately after training:
   ```bash
   python scripts/setup_mysql.py
   ```

### Option B: Manual DDL Import in MySQL Workbench / CLI
If you want to inspect or execute the raw SQL DDL first:
```bash
mysql -u root -p < database/schema_mysql.sql
python scripts/setup_mysql.py
```
You can also run the pre-written analytical SQL queries in MySQL Workbench from:
- [`database/analytics_queries_mysql.sql`](database/analytics_queries_mysql.sql)

### Option C: Instant Local Run (Zero-Config SQLite Fallback)
If MySQL Server is not installed on a demo laptop, keep `DB_BACKEND=auto` in `.env`:
- The system automatically tests MySQL connectivity and, if unavailable, seamlessly runs the exact same SQLAlchemy schema and CRUD operations against `data/customer_intelligence.db`.
- The active backend is clearly displayed in the Streamlit sidebar (`MySQL` or `SQLite (Fallback)`).

---

## 4. Train ML Models & Seed Database

Run the one-command initialization pipeline:
```bash
python scripts/init_project.py
```
This script performs five steps:
1. Cleans and enriches the **7,043** subscription accounts (`data/raw/customer_churn_dataset.xlsx`) and loads the **18,482** warehouse customer RFM profiles (`data/raw/sql_warehouse/gold.report_customers.csv`).
2. Trains and benchmarks four Scikit-learn classifiers (**Gradient Boosting**, **HistGradientBoosting**, **Random Forest**, **Logistic Regression**) and saves the champion pipeline to `models/churn_pipeline.joblib`.
3. Trains two **K-Means + PCA** segmentation models (Subscription Behavior & Warehouse RFM Purchasing) and validates `k=2..6` via Elbow & Silhouette scores.
4. Exports enriched datasets to `data/processed/`.
5. Creates all relational tables and seeds `customers`, `retention_actions`, `sales_products`, `sales_transactions`, and `customer_rfm_summary`.

---

## 5. Launch the Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 6. Run Automated Tests

Verify data cleaning, ML predictions, segmentation, retention recommendations, and database CRUD:
```bash
python -m pytest -v
```

---

## 7. College Project Presentation / Demo Flow

1. **Executive Overview:** Walk through the KPI cards (7,043 customers, 26.54% historical churn rate, \$16.06M lifetime revenue, Annual Revenue at Risk, and Tech/Admin ticket counts) and demonstrate live sidebar filtering.
2. **Churn Prediction Lab:** Show the live Risk Simulator by switching a customer from `Month-to-month` + `2 Tech Tickets` to `Two year` + `Tech Support = Yes` to demonstrate real-time risk drop, then show the Model Leaderboard & ROC curves.
3. **Segmentation & SQL Analytics:** Show Behavioral K-Means PCA clusters and switch to the **Purchasing Behavior RFM & SQL Analytics** tab to demonstrate retail revenue trends, part-to-whole category shares, and VIP/Regular/New SQL segmentation.
4. **Customer Management (CRUD):** Add a new customer record, show that ML automatically scores & segments them, edit their contract to reduce risk, and verify database persistence.
5. **AI Retention Advisor:** Select a high-risk customer, inspect the personalized outreach email & ROI save estimate, and update campaign status in the database-backed tracker.
