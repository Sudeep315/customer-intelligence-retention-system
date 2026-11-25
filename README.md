# Customer Intelligence & Retention System

An end-to-end AI-powered **Customer Intelligence & Retention System** combining **Machine Learning Churn Prediction**, **K-Means Customer Segmentation**, **SQL Data Warehouse Analytics**, **Interactive Plotly Dashboards**, **Full Customer CRUD Management**, and an **AI Retention Strategy Advisor**.

Designed as a production-style final-year engineering/data science capstone project synthesizing analytical concepts from both Power BI Churn Risk Analytics and SQL Data Warehouse Analytics into a unified **Python + Scikit-Learn + Streamlit + Plotly + MySQL** web application.

---

## Key Capabilities

1. **Customer Churn Prediction (Supervised ML)**
   - Benchmarks **Gradient Boosting**, **HistGradientBoosting**, **Random Forest**, and **Logistic Regression** on 7,043 customer records with engineered behavioral & support-ticket features.
   - Provides a **Live Single-Customer Risk Simulator**, **Batch CSV/Excel Prediction Upload**, **ROC Curve Comparison**, **Feature Importance**, **Confusion Matrices**, and per-customer **Risk Driver Explanations**.
2. **Customer Segmentation & Purchasing Analytics (Unsupervised ML + SQL)**
   - **Behavioral & Spend Clustering (K-Means + PCA):** Segments subscription customers into actionable personas (*Loyal High-Value Champions*, *At-Risk High-Friction*, *Budget Steady Core*, *New Digital Explorers*) validated via Elbow & Silhouette scores.
   - **Purchasing RFM & Warehouse Analytics:** Segments 18,482 retail accounts and 60,398 transactions into RFM clusters alongside **Monthly Revenue Trends**, **Cumulative Growth**, **Part-to-Whole Category Share**, and **VIP / Regular / New** value tiers.
3. **Executive Customer Dashboard**
   - Displays **Total Customers**, **Historical Churn Rate**, **High-Risk Predicted Count**, **Lifetime Revenue**, **Annual Revenue at Risk**, **Technical & Admin Support Tickets**, and **Automated Business Insights** with dynamic global cohort filtering.
4. **Interactive Data Visualization Studio**
   - Plotly interactive visualizations covering contract vulnerability, tenure cohorts (`<1 Year` to `5+ Years`), payment method risk, service adoption matrix, ticket escalation impact, geographic distribution, and demographic breakdowns.
5. **Customer Management (Full CRUD + 360° CRM)**
   - Browse, search, filter, export, inspect 360° customer profiles, **Add**, **Update**, and **Delete** customer records directly in the database with **real-time ML churn re-scoring & cluster re-assignment**.
6. **AI Retention Advisor & Campaign Tracker**
   - Recommends tailored retention strategies per high-risk account, computes **Revenue at Risk** and **Estimated Saveable Revenue**, generates personalized retention outreach emails, summarizes segment-level playbooks, and tracks campaign statuses (`Recommended`, `In Progress`, `Applied`, `Dismissed`) in the database.

---

## Project Folder Structure

```text
CCP1/
├── app.py                              # Main Streamlit dashboard entry point
├── requirements.txt                    # Python dependencies
├── .env.example                        # MySQL & app environment template
├── README.md                           # Project overview & documentation
├── SETUP_GUIDE.md                      # Detailed installation, MySQL & demo guide
├── .streamlit/
│   └── config.toml                     # Modern light enterprise theme
├── data/
│   ├── raw/
│   │   ├── customer_churn_dataset.xlsx # 7,043 subscription & churn records
│   │   └── sql_warehouse/              # Gold retail warehouse CSVs (60k+ orders)
│   └── processed/                      # Cleaned, enriched & scored datasets
├── database/
│   ├── schema_mysql.sql                # MySQL 8.0 DDL (tables, FKs, indexes, views)
│   └── analytics_queries_mysql.sql     # Analytical SQL queries (EDA + RFM + KPIs)
├── models/                             # Saved joblib pipelines, scalers & metrics JSON
├── scripts/
│   ├── init_project.py                 # Full pipeline: ETL + ML train + DB seed
│   └── setup_mysql.py                  # Dedicated MySQL schema & data seeder
├── src/
│   ├── config.py                       # Paths, risk thresholds, DB settings
│   ├── data_pipeline.py                # Cleaning, feature engineering, CRM enrichment
│   ├── database.py                     # SQLAlchemy MySQL + SQLite auto-fallback CRUD layer
│   ├── ml_churn.py                     # Multi-model churn training, scoring & explainability
│   ├── ml_segmentation.py              # Behavioral & RFM K-Means + PCA clustering
│   ├── recommender.py                  # AI retention strategy, ROI & outreach generator
│   ├── analytics.py                    # KPIs, cohort breakdowns & warehouse analytics
│   └── ui/
│       ├── styles.py                   # Custom enterprise CSS & KPI card components
│       ├── charts.py                   # Reusable Plotly chart builders
│       └── pages/                      # 6 modular Streamlit pages
└── tests/                              # Automated pytest suite
```

---

## Tech Stack

- **Language:** Python 3.12
- **Data Processing:** Pandas, NumPy, OpenPyXL
- **Machine Learning:** Scikit-learn (`GradientBoostingClassifier`, `HistGradientBoostingClassifier`, `RandomForestClassifier`, `LogisticRegression`, `KMeans`, `PCA`, `StandardScaler`, `ColumnTransformer`)
- **Database:** MySQL 8.0 (`mysql-connector-python`, `PyMySQL`, `SQLAlchemy`) with seamless zero-config SQLite fallback (`DB_BACKEND=auto`)
- **Frontend & Visualization:** Streamlit, Plotly

---

## Verified Model Benchmarks

| Model / Task | Accuracy | F1 Score | ROC-AUC | Silhouette Score |
|:--|:--:|:--:|:--:|:--:|
| **Gradient Boosting (Churn Champion)** | **86.23%** | **0.7252** | **0.9326** | — |
| HistGradientBoosting (Churn) | 86.52% | 0.7324 | 0.9322 | — |
| Random Forest (Churn) | 85.02% | 0.7577 | 0.9281 | — |
| Logistic Regression (Churn) | 83.25% | 0.7336 | 0.9274 | — |
| Behavioral K-Means (`k=4`, Subscription) | — | — | — | **0.3657** |
| Warehouse RFM K-Means (`k=4`, Retail) | — | — | — | **0.5230** |

---

## Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Configure environment (optional MySQL credentials; auto-falls back to SQLite if MySQL isn't running)
copy .env.example .env

# 3. Run ETL, train ML models, and seed the database
python scripts/init_project.py

# 4. Launch the dashboard
streamlit run app.py
```

For full MySQL Workbench/Server setup instructions, schema details, and presentation flow, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## Running Tests

```bash
python -m pytest -v
```
