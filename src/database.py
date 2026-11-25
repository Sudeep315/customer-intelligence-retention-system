"""Database abstraction layer supporting MySQL 8.0 (primary) with seamless SQLite fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config import SQL_WAREHOUSE_RAW_DIR, get_db_settings

metadata = MetaData()

customers_table = Table(
    "customers",
    metadata,
    Column("customer_id", String(20), primary_key=True),
    Column("customer_name", String(120), nullable=False),
    Column("email", String(150), nullable=True),
    Column("country", String(60), nullable=False, default="United States"),
    Column("age", Integer, nullable=False, default=35),
    Column("age_group", String(25), nullable=False, default="30-39"),
    Column("marital_status", String(20), nullable=False, default="Single"),
    Column("gender", String(15), nullable=False),
    Column("senior_citizen", Integer, nullable=False, default=0),
    Column("partner", String(10), nullable=False, default="No"),
    Column("dependents", String(10), nullable=False, default="No"),
    Column("tenure", Integer, nullable=False, default=0),
    Column("phone_service", String(10), nullable=False, default="Yes"),
    Column("multiple_lines", String(25), nullable=False, default="No"),
    Column("internet_service", String(25), nullable=False, default="DSL"),
    Column("online_security", String(25), nullable=False, default="No"),
    Column("online_backup", String(25), nullable=False, default="No"),
    Column("device_protection", String(25), nullable=False, default="No"),
    Column("tech_support", String(25), nullable=False, default="No"),
    Column("streaming_tv", String(25), nullable=False, default="No"),
    Column("streaming_movies", String(25), nullable=False, default="No"),
    Column("contract", String(30), nullable=False, default="Month-to-month"),
    Column("paperless_billing", String(10), nullable=False, default="Yes"),
    Column("payment_method", String(45), nullable=False, default="Electronic check"),
    Column("monthly_charges", Float, nullable=False, default=0.0),
    Column("total_charges", Float, nullable=False, default=0.0),
    Column("num_admin_tickets", Integer, nullable=False, default=0),
    Column("num_tech_tickets", Integer, nullable=False, default=0),
    Column("service_count", Integer, nullable=False, default=0),
    Column("total_tickets", Integer, nullable=False, default=0),
    Column("annual_spend", Float, nullable=False, default=0.0),
    Column("value_tier", String(20), nullable=False, default="New"),
    Column("actual_churn", String(10), nullable=False, default="No"),
    Column("churn_probability", Float, nullable=False, default=0.0),
    Column("predicted_churn", String(10), nullable=False, default="No"),
    Column("risk_level", String(20), nullable=False, default="Low Risk"),
    Column("cluster_id", Integer, nullable=False, default=0),
    Column("cluster_segment", String(60), nullable=False, default="Unassigned"),
    Column("created_at", DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)),
    Column("updated_at", DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)),
)

retention_actions_table = Table(
    "retention_actions",
    metadata,
    Column("action_id", Integer, primary_key=True, autoincrement=True),
    Column(
        "customer_id",
        String(20),
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("risk_level", String(20), nullable=False),
    Column("churn_probability", Float, nullable=False),
    Column("recommended_strategy", String(255), nullable=False),
    Column("priority", String(20), nullable=False, default="High"),
    Column("estimated_save_rate", Float, nullable=False, default=0.35),
    Column("revenue_at_risk", Float, nullable=False, default=0.0),
    Column("action_notes", Text, nullable=True),
    Column("status", String(30), nullable=False, default="Recommended"),
    Column("created_at", DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)),
)

sales_customers_table = Table(
    "sales_customers",
    metadata,
    Column("customer_key", Integer, primary_key=True),
    Column("customer_id", Integer, nullable=False),
    Column("customer_number", String(30), nullable=False),
    Column("first_name", String(60), nullable=True),
    Column("last_name", String(60), nullable=True),
    Column("country", String(60), nullable=True),
    Column("marital_status", String(20), nullable=True),
    Column("gender", String(20), nullable=True),
    Column("birthdate", String(20), nullable=True),
    Column("create_date", String(20), nullable=True),
)

sales_products_table = Table(
    "sales_products",
    metadata,
    Column("product_key", Integer, primary_key=True),
    Column("product_id", Integer, nullable=False),
    Column("product_number", String(40), nullable=False),
    Column("product_name", String(120), nullable=False),
    Column("category_id", String(20), nullable=True),
    Column("category", String(60), nullable=True),
    Column("subcategory", String(60), nullable=True),
    Column("maintenance", String(10), nullable=True),
    Column("cost", Float, nullable=False, default=0.0),
    Column("product_line", String(40), nullable=True),
    Column("start_date", String(20), nullable=True),
)

sales_transactions_table = Table(
    "sales_transactions",
    metadata,
    Column("tx_id", Integer, primary_key=True, autoincrement=True),
    Column("order_number", String(30), nullable=False),
    Column("product_key", Integer, nullable=False),
    Column("customer_key", Integer, nullable=False),
    Column("order_date", String(20), nullable=True),
    Column("shipping_date", String(20), nullable=True),
    Column("due_date", String(20), nullable=True),
    Column("sales_amount", Float, nullable=False, default=0.0),
    Column("quantity", Integer, nullable=False, default=1),
    Column("price", Float, nullable=False, default=0.0),
)

customer_rfm_summary_table = Table(
    "customer_rfm_summary",
    metadata,
    Column("customer_key", Integer, primary_key=True),
    Column("customer_number", String(30), nullable=False),
    Column("customer_name", String(120), nullable=True),
    Column("age", Integer, nullable=True),
    Column("age_group", String(25), nullable=True),
    Column("customer_segment", String(20), nullable=True),
    Column("last_order_date", String(20), nullable=True),
    Column("recency", Integer, nullable=False, default=0),
    Column("total_orders", Integer, nullable=False, default=0),
    Column("total_sales", Float, nullable=False, default=0.0),
    Column("total_quantity", Integer, nullable=False, default=0),
    Column("total_products", Integer, nullable=False, default=0),
    Column("lifespan", Integer, nullable=False, default=0),
    Column("avg_order_value", Float, nullable=False, default=0.0),
    Column("avg_monthly_spend", Float, nullable=False, default=0.0),
    Column("rfm_cluster_id", Integer, nullable=False, default=0),
    Column("rfm_segment", String(60), nullable=False, default="Unassigned"),
)


def _ensure_mysql_database() -> None:
    settings = get_db_settings()
    base_engine = create_engine(settings.mysql_url(include_db=False), pool_pre_ping=True, connect_args={"connect_timeout": 1})
    with base_engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
    base_engine.dispose()


def create_db_engine(force_sqlite_path: Optional[str] = None) -> Tuple[Engine, str, str]:
    """Create SQLAlchemy engine. Returns (engine, active_backend, detail_message)."""
    if force_sqlite_path:
        eng = create_engine(f"sqlite:///{force_sqlite_path}", future=True)
        return eng, "SQLite", f"Connected to SQLite ({force_sqlite_path})"

    settings = get_db_settings()
    if settings.backend in ("auto", "mysql"):
        try:
            _ensure_mysql_database()
            eng = create_engine(
                settings.mysql_url(include_db=True),
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"connect_timeout": 1},
                future=True,
            )
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return eng, "MySQL", f"MySQL connected ({settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database})"
        except Exception as exc:
            if settings.backend == "mysql":
                raise RuntimeError(f"MySQL connection failed: {exc}") from exc
            eng = create_engine(settings.sqlite_url(), future=True)
            return (
                eng,
                "SQLite (Fallback)",
                f"MySQL server unavailable locally; running on local SQLite mirror ({settings.sqlite_path.name}).",
            )

    eng = create_engine(settings.sqlite_url(), future=True)
    return eng, "SQLite", f"Connected to SQLite ({settings.sqlite_path.name})"


@lru_cache(maxsize=1)
def get_cached_connection() -> Tuple[Engine, str, str]:
    """Return cached database engine and connection status."""
    eng, backend, msg = create_db_engine()
    metadata.create_all(eng)
    return eng, backend, msg


def reset_db_cache() -> None:
    """Clear cached engine so updated .env credentials take effect immediately."""
    get_cached_connection.cache_clear()


def init_schema(engine: Engine) -> None:
    """Create all schema tables if they do not exist."""
    metadata.create_all(engine)


def table_row_count(engine: Engine, table: Table) -> int:
    """Return row count of a SQLAlchemy table."""
    with engine.connect() as conn:
        result = conn.execute(select(func.count()).select_from(table)).scalar()
    return int(result or 0)


def seed_customers(engine: Engine, customers_df: pd.DataFrame, replace: bool = False) -> int:
    """Seed master customers table cleanly."""
    init_schema(engine)
    existing = table_row_count(engine, customers_table)
    if existing > 0 and not replace:
        return existing

    valid_cols = {c.name for c in customers_table.columns} - {"created_at", "updated_at"}
    records_df = customers_df[[col for col in customers_df.columns if col in valid_cols]].copy()
    now = datetime.now(timezone.utc)
    records_df["created_at"] = now
    records_df["updated_at"] = now

    with engine.begin() as conn:
        if replace:
            conn.execute(delete(retention_actions_table))
            conn.execute(delete(customers_table))
        records_df.to_sql("customers", con=conn, if_exists="append", index=False, chunksize=500)
    return len(records_df)


def seed_warehouse_tables(engine: Engine, rfm_df: pd.DataFrame, replace: bool = False) -> Dict[str, int]:
    """Seed SQL analytics warehouse tables (RFM summary, products, sales transactions)."""
    init_schema(engine)
    counts: Dict[str, int] = {}

    rfm_existing = table_row_count(engine, customer_rfm_summary_table)
    if replace or rfm_existing == 0:
        valid_rfm = {c.name for c in customer_rfm_summary_table.columns}
        rfm_clean = rfm_df[[c for c in rfm_df.columns if c in valid_rfm]].copy()
        with engine.begin() as conn:
            if replace:
                conn.execute(delete(customer_rfm_summary_table))
            rfm_clean.to_sql("customer_rfm_summary", con=conn, if_exists="append", index=False, chunksize=1000)
        counts["customer_rfm_summary"] = len(rfm_clean)
    else:
        counts["customer_rfm_summary"] = rfm_existing

    prod_existing = table_row_count(engine, sales_products_table)
    prod_csv = SQL_WAREHOUSE_RAW_DIR / "gold.dim_products.csv"
    if (replace or prod_existing == 0) and prod_csv.exists():
        prod_df = pd.read_csv(prod_csv)
        with engine.begin() as conn:
            if replace:
                conn.execute(delete(sales_products_table))
            prod_df.to_sql("sales_products", con=conn, if_exists="append", index=False, chunksize=500)
        counts["sales_products"] = len(prod_df)
    else:
        counts["sales_products"] = prod_existing

    tx_existing = table_row_count(engine, sales_transactions_table)
    tx_csv = SQL_WAREHOUSE_RAW_DIR / "gold.fact_sales.csv"
    if (replace or tx_existing == 0) and tx_csv.exists():
        tx_df = pd.read_csv(tx_csv)
        with engine.begin() as conn:
            if replace:
                conn.execute(delete(sales_transactions_table))
            tx_df.to_sql("sales_transactions", con=conn, if_exists="append", index=False, chunksize=2000)
        counts["sales_transactions"] = len(tx_df)
    else:
        counts["sales_transactions"] = tx_existing

    return counts


def seed_initial_retention_actions(engine: Engine, customers_df: pd.DataFrame, actions_list: List[Dict[str, Any]]) -> int:
    """Populate initial AI retention queue for top high-risk customers if table is empty."""
    init_schema(engine)
    if table_row_count(engine, retention_actions_table) > 0:
        return table_row_count(engine, retention_actions_table)
    if not actions_list:
        return 0
    with engine.begin() as conn:
        conn.execute(insert(retention_actions_table), actions_list)
    return len(actions_list)


def fetch_customers(engine: Engine) -> pd.DataFrame:
    """Fetch all customer intelligence records as a DataFrame."""
    query = select(customers_table).order_by(customers_table.c.churn_probability.desc())
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def get_customer_by_id(engine: Engine, customer_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single customer record by customer_id."""
    query = select(customers_table).where(customers_table.c.customer_id == customer_id)
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    return dict(row) if row else None


def add_customer(engine: Engine, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a new customer record and return inserted data."""
    record = dict(payload)
    now = datetime.now(timezone.utc)
    record.setdefault("created_at", now)
    record["updated_at"] = now
    valid_cols = {c.name for c in customers_table.columns}
    filtered = {k: v for k, v in record.items() if k in valid_cols}
    with engine.begin() as conn:
        conn.execute(insert(customers_table).values(**filtered))
    return get_customer_by_id(engine, str(filtered["customer_id"])) or filtered


def update_customer(engine: Engine, customer_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update an existing customer record and return the refreshed row."""
    valid_cols = {c.name for c in customers_table.columns} - {"customer_id", "created_at"}
    filtered = {k: v for k, v in updates.items() if k in valid_cols}
    filtered["updated_at"] = datetime.now(timezone.utc)
    with engine.begin() as conn:
        res = conn.execute(
            update(customers_table)
            .where(customers_table.c.customer_id == customer_id)
            .values(**filtered)
        )
        if res.rowcount == 0:
            return None
    return get_customer_by_id(engine, customer_id)


def delete_customer(engine: Engine, customer_id: str) -> bool:
    """Delete a customer record (and cascaded retention actions)."""
    with engine.begin() as conn:
        conn.execute(delete(retention_actions_table).where(retention_actions_table.c.customer_id == customer_id))
        res = conn.execute(delete(customers_table).where(customers_table.c.customer_id == customer_id))
    return res.rowcount > 0


def fetch_retention_actions(engine: Engine) -> pd.DataFrame:
    """Fetch retention action log joined with customer name and contract."""
    stmt = (
        select(
            retention_actions_table,
            customers_table.c.customer_name,
            customers_table.c.contract,
            customers_table.c.monthly_charges,
            customers_table.c.cluster_segment,
        )
        .select_from(
            retention_actions_table.join(
                customers_table,
                retention_actions_table.c.customer_id == customers_table.c.customer_id,
            )
        )
        .order_by(retention_actions_table.c.action_id.desc())
    )
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def log_retention_action(engine: Engine, action: Dict[str, Any]) -> int:
    """Persist a new AI retention recommendation / campaign entry."""
    valid_cols = {c.name for c in retention_actions_table.columns} - {"action_id"}
    payload = {k: v for k, v in action.items() if k in valid_cols}
    payload.setdefault("created_at", datetime.now(timezone.utc))
    with engine.begin() as conn:
        res = conn.execute(insert(retention_actions_table).values(**payload))
        return int(res.inserted_primary_key[0]) if res.inserted_primary_key else 0


def update_retention_status(engine: Engine, action_id: int, new_status: str, notes: Optional[str] = None) -> bool:
    """Update status and optional notes on a retention action."""
    vals: Dict[str, Any] = {"status": new_status}
    if notes is not None:
        vals["action_notes"] = notes
    with engine.begin() as conn:
        res = conn.execute(
            update(retention_actions_table)
            .where(retention_actions_table.c.action_id == action_id)
            .values(**vals)
        )
    return res.rowcount > 0


def fetch_rfm_summary(engine: Engine) -> pd.DataFrame:
    """Fetch warehouse RFM customer summary table."""
    with engine.connect() as conn:
        return pd.read_sql(select(customer_rfm_summary_table), conn)


def fetch_warehouse_sales_joined(engine: Engine) -> pd.DataFrame:
    """Fetch sales transactions joined with product category for SQL-style analytics."""
    stmt = select(
        sales_transactions_table.c.order_number,
        sales_transactions_table.c.order_date,
        sales_transactions_table.c.customer_key,
        sales_transactions_table.c.sales_amount,
        sales_transactions_table.c.quantity,
        sales_products_table.c.product_name,
        sales_products_table.c.category,
        sales_products_table.c.subcategory,
        sales_products_table.c.cost,
    ).select_from(
        sales_transactions_table.outerjoin(
            sales_products_table,
            sales_transactions_table.c.product_key == sales_products_table.c.product_key,
        )
    )
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def fetch_warehouse_sql_aggregates(engine: Engine) -> Dict[str, Any]:
    """Execute SQL GROUP BY analytics directly on database for fast dashboard rendering."""
    month_expr = func.substr(sales_transactions_table.c.order_date, 1, 7)
    monthly_stmt = (
        select(
            month_expr.label("order_month"),
            func.sum(sales_transactions_table.c.sales_amount).label("total_revenue"),
            func.count(func.distinct(sales_transactions_table.c.order_number)).label("total_orders"),
            func.count(func.distinct(sales_transactions_table.c.customer_key)).label("active_customers"),
            func.sum(sales_transactions_table.c.quantity).label("total_units"),
        )
        .where(sales_transactions_table.c.order_date.is_not(None))
        .group_by(month_expr)
        .order_by(month_expr)
    )

    cat_stmt = (
        select(
            func.coalesce(sales_products_table.c.category, "Other").label("category"),
            func.sum(sales_transactions_table.c.sales_amount).label("total_sales"),
            func.count(func.distinct(sales_transactions_table.c.order_number)).label("total_orders"),
            func.sum(sales_transactions_table.c.quantity).label("total_units"),
        )
        .select_from(
            sales_transactions_table.outerjoin(
                sales_products_table,
                sales_transactions_table.c.product_key == sales_products_table.c.product_key,
            )
        )
        .group_by(func.coalesce(sales_products_table.c.category, "Other"))
        .order_by(func.sum(sales_transactions_table.c.sales_amount).desc())
    )

    prod_stmt = (
        select(
            sales_products_table.c.product_name,
            sales_products_table.c.category,
            func.sum(sales_transactions_table.c.sales_amount).label("total_sales"),
            func.sum(sales_transactions_table.c.quantity).label("units_sold"),
        )
        .select_from(
            sales_transactions_table.join(
                sales_products_table,
                sales_transactions_table.c.product_key == sales_products_table.c.product_key,
            )
        )
        .group_by(sales_products_table.c.product_name, sales_products_table.c.category)
        .order_by(func.sum(sales_transactions_table.c.sales_amount).desc())
        .limit(10)
    )

    with engine.connect() as conn:
        monthly_df = pd.read_sql(monthly_stmt, conn)
        cat_df = pd.read_sql(cat_stmt, conn)
        top_prod_df = pd.read_sql(prod_stmt, conn)
        total_tx = int(conn.execute(select(func.count()).select_from(sales_transactions_table)).scalar() or 0)
        total_orders = int(
            conn.execute(select(func.count(func.distinct(sales_transactions_table.c.order_number)))).scalar() or 0
        )

    if not monthly_df.empty:
        monthly_df["running_total"] = monthly_df["total_revenue"].cumsum()
    if not cat_df.empty:
        tot = float(cat_df["total_sales"].sum()) or 1.0
        cat_df["share_pct"] = (cat_df["total_sales"] / tot * 100.0).round(2)

    return {
        "monthly_trend": monthly_df,
        "category_sales": cat_df,
        "top_products": top_prod_df,
        "total_tx_lines": total_tx,
        "total_orders": total_orders,
    }


def test_mysql_connection(host: str, port: int, user: str, password: str, database: str) -> Tuple[bool, str]:
    """Test direct MySQL connectivity with provided credentials."""
    from urllib.parse import quote_plus

    try:
        pwd = quote_plus(password)
        auth = f"{user}:{pwd}" if password else user
        url = f"mysql+pymysql://{auth}@{host}:{port}?charset=utf8mb4"
        eng = create_engine(url, connect_args={"connect_timeout": 3})
        with eng.connect() as conn:
            ver = conn.execute(text("SELECT VERSION()")).scalar()
        eng.dispose()
        return True, f"Connected successfully to MySQL {ver} (target DB: {database})."
    except SQLAlchemyError as exc:
        return False, f"Connection failed: {exc}"
    except Exception as exc:
        return False, f"Connection error: {exc}"
