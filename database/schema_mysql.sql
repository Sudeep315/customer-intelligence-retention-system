-- ============================================================================
-- Customer Intelligence & Retention System - MySQL 8.0 Schema
-- Unifies Churn Prediction CRM, Retention Interventions, and SQL Analytics Warehouse
-- ============================================================================

CREATE DATABASE IF NOT EXISTS customer_intelligence_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE customer_intelligence_db;

-- ----------------------------------------------------------------------------
-- 1. Master Customer Intelligence Table (Churn + Segmentation + CRM Profile)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    customer_id         VARCHAR(20)     PRIMARY KEY,
    customer_name       VARCHAR(120)    NOT NULL,
    email               VARCHAR(150)    NULL,
    country             VARCHAR(60)     NOT NULL DEFAULT 'United States',
    age                 INT             NOT NULL DEFAULT 35,
    age_group           VARCHAR(25)     NOT NULL DEFAULT '30-39',
    marital_status      VARCHAR(20)     NOT NULL DEFAULT 'Single',
    gender              VARCHAR(15)     NOT NULL,
    senior_citizen      TINYINT         NOT NULL DEFAULT 0,
    partner             VARCHAR(10)     NOT NULL DEFAULT 'No',
    dependents          VARCHAR(10)     NOT NULL DEFAULT 'No',
    tenure              INT             NOT NULL DEFAULT 0,
    phone_service       VARCHAR(10)     NOT NULL DEFAULT 'Yes',
    multiple_lines      VARCHAR(25)     NOT NULL DEFAULT 'No',
    internet_service    VARCHAR(25)     NOT NULL DEFAULT 'DSL',
    online_security     VARCHAR(25)     NOT NULL DEFAULT 'No',
    online_backup       VARCHAR(25)     NOT NULL DEFAULT 'No',
    device_protection   VARCHAR(25)     NOT NULL DEFAULT 'No',
    tech_support        VARCHAR(25)     NOT NULL DEFAULT 'No',
    streaming_tv        VARCHAR(25)     NOT NULL DEFAULT 'No',
    streaming_movies    VARCHAR(25)     NOT NULL DEFAULT 'No',
    contract            VARCHAR(30)     NOT NULL DEFAULT 'Month-to-month',
    paperless_billing   VARCHAR(10)     NOT NULL DEFAULT 'Yes',
    payment_method      VARCHAR(45)     NOT NULL DEFAULT 'Electronic check',
    monthly_charges     DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    total_charges       DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    num_admin_tickets   INT             NOT NULL DEFAULT 0,
    num_tech_tickets    INT             NOT NULL DEFAULT 0,
    service_count       INT             NOT NULL DEFAULT 0,
    total_tickets       INT             NOT NULL DEFAULT 0,
    annual_spend        DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    value_tier          VARCHAR(20)     NOT NULL DEFAULT 'New',
    actual_churn        VARCHAR(10)     NOT NULL DEFAULT 'No',
    churn_probability   DECIMAL(6, 4)   NOT NULL DEFAULT 0.0000,
    predicted_churn     VARCHAR(10)     NOT NULL DEFAULT 'No',
    risk_level          VARCHAR(20)     NOT NULL DEFAULT 'Low Risk',
    cluster_id          INT             NOT NULL DEFAULT 0,
    cluster_segment     VARCHAR(60)     NOT NULL DEFAULT 'Unassigned',
    created_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_customers_risk (risk_level),
    INDEX idx_customers_segment (cluster_segment),
    INDEX idx_customers_contract (contract),
    INDEX idx_customers_country (country),
    INDEX idx_customers_value_tier (value_tier)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- 2. AI Retention Action & Campaign Tracking Table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retention_actions (
    action_id            INT             AUTO_INCREMENT PRIMARY KEY,
    customer_id          VARCHAR(20)     NOT NULL,
    risk_level           VARCHAR(20)     NOT NULL,
    churn_probability    DECIMAL(6, 4)   NOT NULL,
    recommended_strategy VARCHAR(255)    NOT NULL,
    priority             VARCHAR(20)     NOT NULL DEFAULT 'High',
    estimated_save_rate  DECIMAL(5, 2)   NOT NULL DEFAULT 0.35,
    revenue_at_risk      DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    action_notes         TEXT            NULL,
    status               VARCHAR(30)     NOT NULL DEFAULT 'Recommended',
    created_at           TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_retention_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON DELETE CASCADE,
    INDEX idx_retention_status (status),
    INDEX idx_retention_priority (priority)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- 3. Retail Sales & Purchasing Warehouse Tables (from SQL Data Analytics Ref)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sales_customers (
    customer_key    INT             PRIMARY KEY,
    customer_id     INT             NOT NULL,
    customer_number VARCHAR(30)     NOT NULL,
    first_name      VARCHAR(60)     NULL,
    last_name       VARCHAR(60)     NULL,
    country         VARCHAR(60)     NULL,
    marital_status  VARCHAR(20)     NULL,
    gender          VARCHAR(20)     NULL,
    birthdate       DATE            NULL,
    create_date     DATE            NULL,
    INDEX idx_sc_country (country)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sales_products (
    product_key     INT             PRIMARY KEY,
    product_id      INT             NOT NULL,
    product_number  VARCHAR(40)     NOT NULL,
    product_name    VARCHAR(120)    NOT NULL,
    category_id     VARCHAR(20)     NULL,
    category        VARCHAR(60)     NULL,
    subcategory     VARCHAR(60)     NULL,
    maintenance     VARCHAR(10)     NULL,
    cost            DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    product_line    VARCHAR(40)     NULL,
    start_date      DATE            NULL,
    INDEX idx_sp_category (category)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sales_transactions (
    tx_id           BIGINT          AUTO_INCREMENT PRIMARY KEY,
    order_number    VARCHAR(30)     NOT NULL,
    product_key     INT             NOT NULL,
    customer_key    INT             NOT NULL,
    order_date      DATE            NULL,
    shipping_date   DATE            NULL,
    due_date        DATE            NULL,
    sales_amount    DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    quantity        INT             NOT NULL DEFAULT 1,
    price           DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    INDEX idx_st_order_date (order_date),
    INDEX idx_st_customer (customer_key),
    INDEX idx_st_product (product_key)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS customer_rfm_summary (
    customer_key      INT             PRIMARY KEY,
    customer_number   VARCHAR(30)     NOT NULL,
    customer_name     VARCHAR(120)    NULL,
    age               INT             NULL,
    age_group         VARCHAR(25)     NULL,
    customer_segment  VARCHAR(20)     NULL,
    last_order_date   DATE            NULL,
    recency           INT             NOT NULL DEFAULT 0,
    total_orders      INT             NOT NULL DEFAULT 0,
    total_sales       DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    total_quantity    INT             NOT NULL DEFAULT 0,
    total_products    INT             NOT NULL DEFAULT 0,
    lifespan          INT             NOT NULL DEFAULT 0,
    avg_order_value   DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    avg_monthly_spend DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    rfm_cluster_id    INT             NOT NULL DEFAULT 0,
    rfm_segment       VARCHAR(60)     NOT NULL DEFAULT 'Unassigned',
    INDEX idx_rfm_segment (customer_segment),
    INDEX idx_rfm_cluster (rfm_segment)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- 4. Analytical SQL Views
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_customer_risk_summary AS
SELECT
    contract,
    internet_service,
    risk_level,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN actual_churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(AVG(churn_probability) * 100, 2) AS avg_churn_prob_pct,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue,
    ROUND(SUM(total_charges), 2) AS total_lifetime_revenue,
    SUM(num_tech_tickets) AS total_tech_tickets
FROM customers
GROUP BY contract, internet_service, risk_level;

CREATE OR REPLACE VIEW vw_monthly_revenue_trend AS
SELECT
    DATE_FORMAT(order_date, '%Y-%m') AS order_month,
    COUNT(DISTINCT order_number) AS total_orders,
    COUNT(DISTINCT customer_key) AS active_customers,
    SUM(sales_amount) AS total_revenue,
    SUM(quantity) AS total_units
FROM sales_transactions
WHERE order_date IS NOT NULL
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY order_month;
