-- ============================================================================
-- Customer Intelligence & Retention System - MySQL Analytical Queries
-- Adapted from sql-data-analytics-project (EDA + Advanced Analytics)
-- ============================================================================

USE customer_intelligence_db;

-- 1. Executive KPI Summary (Churn & Revenue Exposure)
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN actual_churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN actual_churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2) AS mrr,
    ROUND(SUM(annual_spend), 2) AS arr,
    ROUND(SUM(total_charges), 2) AS lifetime_charges,
    SUM(num_admin_tickets) AS admin_tickets,
    SUM(num_tech_tickets) AS tech_tickets
FROM customers;

-- 2. Churn Rate & Revenue by Contract Type (Magnitude Analysis)
SELECT
    contract,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN actual_churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND(100.0 * SUM(CASE WHEN actual_churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2) AS monthly_revenue
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- 3. Value Tier Segmentation (VIP / Regular / New)
SELECT
    value_tier,
    COUNT(*) AS total_customers,
    ROUND(AVG(tenure), 1) AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_spend,
    ROUND(SUM(total_charges), 2) AS total_revenue,
    ROUND(100.0 * SUM(CASE WHEN actual_churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY value_tier
ORDER BY total_revenue DESC;

-- 4. ML Behavioral Cluster Performance
SELECT
    cluster_segment,
    COUNT(*) AS customers,
    ROUND(AVG(tenure), 1) AS avg_tenure,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(service_count), 2) AS avg_services,
    ROUND(AVG(total_tickets), 2) AS avg_tickets,
    ROUND(100.0 * AVG(churn_probability), 2) AS avg_predicted_risk_pct
FROM customers
GROUP BY cluster_segment
ORDER BY avg_predicted_risk_pct DESC;

-- 5. Part-to-Whole Category Sales Contribution (Warehouse Analytics)
WITH category_sales AS (
    SELECT
        p.category,
        SUM(f.sales_amount) AS total_sales
    FROM sales_transactions f
    LEFT JOIN sales_products p
        ON p.product_key = f.product_key
    GROUP BY p.category
)
SELECT
    category,
    total_sales,
    SUM(total_sales) OVER () AS overall_sales,
    ROUND((total_sales / SUM(total_sales) OVER ()) * 100, 2) AS percentage_of_total
FROM category_sales
ORDER BY total_sales DESC;

-- 6. Monthly Revenue & Running Total Over Time (Cumulative Analysis)
WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(order_date, '%Y-%m') AS sales_month,
        SUM(sales_amount) AS total_sales,
        COUNT(DISTINCT customer_key) AS active_buyers
    FROM sales_transactions
    WHERE order_date IS NOT NULL
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT
    sales_month,
    total_sales,
    active_buyers,
    SUM(total_sales) OVER (ORDER BY sales_month) AS running_total_sales
FROM monthly_sales
ORDER BY sales_month;

-- 7. Top 10 High-Value At-Risk Customers Requiring Immediate Retention
SELECT
    customer_id,
    customer_name,
    country,
    contract,
    tenure,
    monthly_charges,
    annual_spend,
    num_tech_tickets,
    ROUND(churn_probability * 100, 1) AS churn_risk_pct,
    cluster_segment
FROM customers
WHERE risk_level = 'High Risk'
ORDER BY annual_spend DESC, churn_probability DESC
LIMIT 10;
