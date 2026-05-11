"""
Feature Engineering Module
==========================
Transforms raw transaction and engagement data into a unified
customer-level feature matrix for segmentation.

Features span four categories:
1. Purchase Behavior — orders, revenue, AOV, frequency, returns
2. Product Affinity — category spend percentages, diversity (Shannon entropy)
3. Engagement — Marketo activity rollups, engagement index, purchase intensity
4. Lifetime Value — simplified CLV estimation
"""

import numpy as np
import pandas as pd
from datetime import datetime


def rollup_transactions(
    orders: pd.DataFrame,
    ref_date: datetime = datetime(2024, 12, 31),
) -> pd.DataFrame:
    """
    Aggregate order data to customer level.

    Returns features: total_orders, total_items, total_revenue, total_profit,
    avg_order_value, unique_products, unique_categories, return_rate,
    days_since_last_order, order_frequency, avg_discount.
    """
    txn = orders.groupby("customer_id").agg(
        total_orders=("order_id", "nunique"),
        total_items=("quantity", "sum"),
        total_revenue=("line_total", "sum"),
        total_profit=("line_profit", "sum"),
        avg_order_value=("line_total", "mean"),
        unique_products=("product_name", "nunique"),
        unique_categories=("product_category", "nunique"),
        total_returns=("is_return", "sum"),
        last_order_date=("order_date", "max"),
        first_order_date=("order_date", "min"),
        avg_discount=("discount_pct", "mean"),
    ).reset_index()

    txn["return_rate"] = txn["total_returns"] / txn["total_orders"]
    txn["days_since_last_order"] = (ref_date - txn["last_order_date"]).dt.days
    txn["customer_tenure_days"] = (ref_date - txn["first_order_date"]).dt.days

    txn["order_frequency"] = np.where(
        txn["customer_tenure_days"] > 0,
        txn["total_orders"] / (txn["customer_tenure_days"] / 30),
        0,
    )

    return txn


def compute_product_affinity(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Compute category spend percentages and product diversity per customer.

    Returns a DataFrame with one column per category (pct_<category>)
    and a category_diversity column (Shannon entropy).
    """
    cat_spend = (
        orders.groupby(["customer_id", "product_category"])["line_total"]
        .sum()
        .unstack(fill_value=0)
    )
    cat_spend_pct = cat_spend.div(cat_spend.sum(axis=1), axis=0)
    cat_spend_pct.columns = [f"pct_{c.lower().replace(' ', '_')}" for c in cat_spend_pct.columns]

    # Shannon entropy for category diversity
    probs = cat_spend.div(cat_spend.sum(axis=1), axis=0).replace(0, np.nan)
    cat_spend_pct["category_diversity"] = -(probs * np.log2(probs)).sum(axis=1)
    cat_spend_pct["category_diversity"] = cat_spend_pct["category_diversity"].fillna(0)

    return cat_spend_pct.reset_index()


def rollup_engagement(
    activities: pd.DataFrame,
    ref_date: datetime = datetime(2024, 12, 31),
) -> pd.DataFrame:
    """
    Aggregate Marketo activity data to customer level.

    Returns engagement counts by type, total activities, recency,
    avg lead score, engagement index, and purchase intensity.
    """
    # Activity type counts
    act_pivot = (
        activities.groupby(["customer_id", "activity_type"])
        .size()
        .unstack(fill_value=0)
    )
    act_pivot.columns = [
        c.lower().replace(" ", "_") + "_count" for c in act_pivot.columns
    ]

    # Aggregate metrics
    agg = activities.groupby("customer_id").agg(
        total_activities=("activity_type", "count"),
        last_activity_date=("activity_date", "max"),
        avg_lead_score=("lead_score", "mean"),
    ).reset_index()

    agg["days_since_last_activity"] = (ref_date - agg["last_activity_date"]).dt.days

    # Engagement weights
    weights = {
        "email_opened_count": 1, "email_clicked_count": 3, "email_bounced_count": -2,
        "sms_delivered_count": 1, "sms_clicked_count": 3,
        "push_notification_opened_count": 2, "web_page_visited_count": 2,
        "cart_abandoned_count": 4, "product_reviewed_count": 5, "referral_sent_count": 5,
    }

    eng = act_pivot.reset_index().merge(agg, on="customer_id", how="outer")

    eng["engagement_index"] = sum(
        eng.get(col, 0) * w for col, w in weights.items()
    )
    eng["purchase_intensity"] = (
        eng.get("cart_abandoned_count", 0) * 3
        + eng.get("product_reviewed_count", 0) * 4
        + eng.get("web_page_visited_count", 0) * 1
    )

    return eng


def estimate_clv(
    txn_features: pd.DataFrame,
    profit_margin: float = 0.65,
) -> pd.Series:
    """
    Simplified CLV: Avg Monthly Revenue × Expected Lifetime Months × Profit Margin.

    Parameters
    ----------
    txn_features : pd.DataFrame
        Must contain: total_revenue, customer_tenure_days, order_frequency.
    profit_margin : float
        Gross profit margin.

    Returns
    -------
    pd.Series
        Estimated CLV per customer.
    """
    monthly_rev = np.where(
        txn_features["customer_tenure_days"] > 30,
        txn_features["total_revenue"] / (txn_features["customer_tenure_days"] / 30),
        txn_features["total_revenue"],
    )

    # Expected lifetime: base 12 months, scaled by frequency
    expected_months = np.clip(12 * (1 + txn_features["order_frequency"]), 6, 60)

    return pd.Series(
        np.round(monthly_rev * expected_months * profit_margin, 2),
        index=txn_features.index,
    )


def build_unified_matrix(
    customers: pd.DataFrame,
    txn_features: pd.DataFrame,
    engagement: pd.DataFrame,
    product_affinity: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all feature sets into a unified customer-level matrix."""
    unified = customers.merge(txn_features, on="customer_id", how="left")
    unified = unified.merge(engagement, on="customer_id", how="left")
    unified = unified.merge(product_affinity, on="customer_id", how="left")

    # Fill NaN for customers with no orders/engagement
    numeric_cols = unified.select_dtypes(include=[np.number]).columns
    unified[numeric_cols] = unified[numeric_cols].fillna(0)

    return unified
