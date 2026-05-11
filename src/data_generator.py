"""
Data Generator Module
=====================
Generates synthetic B2C data mimicking Salesforce Commerce Cloud (SFCC)
and Marketo exports for a D2C premium skincare brand.

Three datasets:
1. Customer Profiles — demographics, membership tier, channel preference
2. Order History — multi-product transactions with returns and profit
3. Marketo Activities — email, SMS, push, web, cart abandonment, reviews, referrals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ──────────────────────────────────────────────
# Product Catalog
# ──────────────────────────────────────────────

PRODUCT_CATALOG = {
    "Vitamin C Serum":        {"category": "Serums",       "price": 48.00, "cogs": 12.00},
    "Retinol Night Cream":    {"category": "Moisturizers", "price": 62.00, "cogs": 15.00},
    "Hyaluronic Acid Serum":  {"category": "Serums",       "price": 42.00, "cogs": 10.00},
    "SPF 50 Daily Sunscreen": {"category": "Sunscreen",    "price": 34.00, "cogs": 8.00},
    "Gentle Foaming Cleanser":{"category": "Cleansers",    "price": 28.00, "cogs": 7.00},
    "Niacinamide Toner":      {"category": "Serums",       "price": 36.00, "cogs": 9.00},
    "Overnight Recovery Mask":{"category": "Masks",        "price": 54.00, "cogs": 13.00},
    "Eye Cream Peptide":      {"category": "Moisturizers", "price": 58.00, "cogs": 14.00},
    "Vitamin E Body Lotion":  {"category": "Moisturizers", "price": 32.00, "cogs": 8.00},
    "Charcoal Detox Mask":    {"category": "Masks",        "price": 38.00, "cogs": 9.00},
    "AHA/BHA Exfoliant":      {"category": "Cleansers",    "price": 44.00, "cogs": 11.00},
    "Collagen Supplements":   {"category": "Supplements",  "price": 56.00, "cogs": 14.00},
    "Lip Repair Balm":        {"category": "Moisturizers", "price": 18.00, "cogs": 4.00},
    "Micellar Water":         {"category": "Cleansers",    "price": 22.00, "cogs": 5.00},
    "Anti-Aging Bundle":      {"category": "Serums",       "price": 120.00,"cogs": 30.00},
}


def generate_customers(n_customers: int = 15000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer profiles (Salesforce Commerce Cloud).

    Parameters
    ----------
    n_customers : int
        Number of customers to generate.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Customer profile table.
    """
    np.random.seed(seed)

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_weights = [0.12, 0.28, 0.25, 0.18, 0.12, 0.05]
    genders = ["Female", "Male", "Non-Binary"]
    gender_weights = [0.68, 0.28, 0.04]
    channels = ["Website", "Mobile App", "Social", "Retail Partner"]
    channel_weights = [0.40, 0.30, 0.20, 0.10]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    tier_weights = [0.40, 0.30, 0.20, 0.10]
    regions = ["Northeast", "Southeast", "Midwest", "West", "International"]
    region_weights = [0.25, 0.20, 0.20, 0.25, 0.10]

    customers = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(n_customers)],
        "age_group": np.random.choice(age_groups, n_customers, p=age_weights),
        "gender": np.random.choice(genders, n_customers, p=gender_weights),
        "preferred_channel": np.random.choice(channels, n_customers, p=channel_weights),
        "membership_tier": np.random.choice(tiers, n_customers, p=tier_weights),
        "region": np.random.choice(regions, n_customers, p=region_weights),
        "signup_date": [
            datetime(2021, 1, 1) + timedelta(days=int(np.random.uniform(0, 1400)))
            for _ in range(n_customers)
        ],
    })

    return customers


def generate_orders(
    customers: pd.DataFrame,
    seed: int = 42,
    ref_date: datetime = datetime(2024, 12, 31),
) -> pd.DataFrame:
    """
    Generate synthetic order history (Salesforce Commerce Cloud).

    Returns
    -------
    pd.DataFrame
        Order-line-level transaction data with product, quantity, revenue, profit.
    """
    np.random.seed(seed)

    products = list(PRODUCT_CATALOG.keys())
    records = []

    for _, cust in customers.iterrows():
        n_orders = max(1, int(np.random.exponential(scale=4)))
        for o in range(n_orders):
            order_date = cust["signup_date"] + timedelta(
                days=int(np.random.uniform(0, (ref_date - cust["signup_date"]).days + 1))
            )
            n_items = np.random.choice([1, 2, 3, 4], p=[0.40, 0.30, 0.20, 0.10])

            for _ in range(n_items):
                product = np.random.choice(products)
                info = PRODUCT_CATALOG[product]
                qty = np.random.choice([1, 2, 3], p=[0.70, 0.25, 0.05])
                discount = np.random.choice([0, 0.10, 0.15, 0.20, 0.25], p=[0.50, 0.20, 0.15, 0.10, 0.05])
                is_return = np.random.random() < 0.08

                line_total = round(info["price"] * qty * (1 - discount), 2)
                line_profit = round((info["price"] - info["cogs"]) * qty * (1 - discount), 2)

                records.append({
                    "customer_id": cust["customer_id"],
                    "order_id": f"ORD-{len(records):07d}",
                    "order_date": order_date,
                    "product_name": product,
                    "product_category": info["category"],
                    "quantity": qty,
                    "unit_price": info["price"],
                    "discount_pct": discount,
                    "line_total": -line_total if is_return else line_total,
                    "line_profit": -line_profit if is_return else line_profit,
                    "is_return": is_return,
                })

    return pd.DataFrame(records)


def generate_marketo_activities(
    customers: pd.DataFrame,
    seed: int = 42,
    ref_date: datetime = datetime(2024, 12, 31),
) -> pd.DataFrame:
    """
    Generate synthetic Marketo B2C engagement data.

    Activity types: Email, SMS, Push, Web, Cart Abandonment, Reviews, Referrals.
    """
    np.random.seed(seed)

    activity_types = [
        "Email Opened", "Email Clicked", "Email Bounced",
        "SMS Delivered", "SMS Clicked",
        "Push Notification Opened", "Web Page Visited",
        "Cart Abandoned", "Product Reviewed", "Referral Sent",
    ]
    activity_weights = [0.22, 0.14, 0.04, 0.12, 0.06, 0.08, 0.18, 0.08, 0.05, 0.03]

    records = []
    for _, cust in customers.iterrows():
        n_activities = max(1, int(np.random.exponential(scale=12)))
        for _ in range(n_activities):
            days_ago = np.random.randint(1, 365)
            records.append({
                "customer_id": cust["customer_id"],
                "activity_type": np.random.choice(activity_types, p=activity_weights),
                "activity_date": ref_date - timedelta(days=int(days_ago)),
                "lead_score": np.random.randint(10, 100),
            })

    return pd.DataFrame(records)
