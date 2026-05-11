"""
Persona Builder Module
======================
Assigns business-meaningful persona names to clusters, generates
profiles, and produces campaign playbooks per persona.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Default Persona Definitions
# ──────────────────────────────────────────────

DEFAULT_PERSONA_LABELS = [
    "💎 VIP Skincare Enthusiasts",
    "🌟 Loyal Routine Builders",
    "🔄 Bargain Hunters",
    "🌱 Promising Newcomers",
    "😴 Lapsed Customers",
    "🧪 Product Explorers",
]

DEFAULT_PLAYBOOKS = {
    "💎 VIP Skincare Enthusiasts": {
        "Email Strategy": "Exclusive early access, VIP-only launches, personalized routines",
        "SMS/Push": "Flash VIP sales, restock reminders, birthday rewards",
        "Loyalty": "Double points, free gifts with purchase, early sale access",
        "Retention Goal": "Maintain 95%+ retention, increase basket size",
    },
    "🌟 Loyal Routine Builders": {
        "Email Strategy": "Routine completion nudges, bundle recommendations, tips & tutorials",
        "SMS/Push": "Restock alerts based on purchase cadence, loyalty milestone updates",
        "Loyalty": "Points multipliers on bundles, referral bonuses",
        "Retention Goal": "Cross-sell into new categories, upgrade to VIP tier",
    },
    "🔄 Bargain Hunters": {
        "Email Strategy": "Flash sale alerts, clearance notifications, BOGO offers",
        "SMS/Push": "Time-limited discount codes, cart abandonment with incentive",
        "Loyalty": "Spend-based unlocks, cashback on full-price items",
        "Retention Goal": "Shift to value perception, reduce discount dependency",
    },
    "🌱 Promising Newcomers": {
        "Email Strategy": "Welcome series, starter kit offers, educational skincare content",
        "SMS/Push": "First-purchase thank you, 2nd order incentive, quiz invitation",
        "Loyalty": "Easy signup bonus, first review reward",
        "Retention Goal": "Drive 2nd purchase within 30 days, build routine habit",
    },
    "😴 Lapsed Customers": {
        "Email Strategy": "We miss you campaigns, new product announcements, win-back offers",
        "SMS/Push": "Exclusive comeback discount, limited-time free shipping",
        "Loyalty": "Reactivation bonus points, expiring rewards reminder",
        "Retention Goal": "Reactivate 15-20% within 60 days",
    },
    "🧪 Product Explorers": {
        "Email Strategy": "New arrival highlights, sample offers, discovery sets",
        "SMS/Push": "Limited edition alerts, ingredient spotlight",
        "Loyalty": "Points for trying new categories, review rewards",
        "Retention Goal": "Increase category penetration, find hero product",
    },
}


# ──────────────────────────────────────────────
# Persona Assignment
# ──────────────────────────────────────────────

def assign_personas(
    df: pd.DataFrame,
    cluster_col: str = "kmeans_cluster",
    rank_col: str = "estimated_clv",
    labels: list = None,
) -> pd.DataFrame:
    """
    Assign persona names to clusters, ranked by CLV (or another metric).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster labels.
    cluster_col : str
        Column containing cluster IDs.
    rank_col : str
        Column to rank clusters by (highest = best persona).
    labels : list, optional
        Persona names ordered best-to-worst. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'persona' column added.
    """
    if labels is None:
        labels = DEFAULT_PERSONA_LABELS

    # Rank clusters by mean of rank_col (descending)
    cluster_rank = (
        df.groupby(cluster_col)[rank_col]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    persona_map = {}
    for rank, cluster_id in enumerate(cluster_rank):
        persona_map[cluster_id] = labels[rank] if rank < len(labels) else f"Segment {rank + 1}"

    df = df.copy()
    df["persona"] = df[cluster_col].map(persona_map)
    return df


# ──────────────────────────────────────────────
# Profiling
# ──────────────────────────────────────────────

def profile_personas(
    df: pd.DataFrame,
    persona_col: str = "persona",
    profile_cols: list = None,
) -> pd.DataFrame:
    """Generate descriptive statistics per persona."""
    if profile_cols is None:
        profile_cols = [
            "total_revenue", "total_orders", "avg_order_value", "estimated_clv",
            "order_frequency", "engagement_index", "purchase_intensity",
            "days_since_last_order", "unique_products", "return_rate",
        ]

    available = [c for c in profile_cols if c in df.columns]
    return df.groupby(persona_col)[available].agg(["mean", "median", "count"])


def get_playbook(persona_name: str, custom: dict = None) -> dict:
    """Return campaign playbook for a given persona."""
    source = custom if custom else DEFAULT_PLAYBOOKS
    return source.get(persona_name, {"Note": "No playbook configured."})
