"""
Segmentation Module
===================
Multi-method B2C customer segmentation:

1. RFM — Recency/Frequency/Monetary scoring (B2C-adapted)
2. K-Means — centroid-based with optimal K selection
3. GMM — soft probabilistic clustering
4. DBSCAN — VIP and anomaly detection
5. Hierarchical — Ward linkage agglomerative clustering
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from kneed import KneeLocator


# ──────────────────────────────────────────────
# RFM Segmentation (B2C)
# ──────────────────────────────────────────────

def rfm_segmentation(
    df: pd.DataFrame,
    recency_col: str = "days_since_last_order",
    frequency_col: str = "total_orders",
    monetary_col: str = "total_revenue",
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    B2C RFM segmentation with consumer-specific segment labels.

    Returns DataFrame with R_Score, F_Score, M_Score, RFM_Score, RFM_Segment.
    """
    rfm = df[["customer_id", recency_col, frequency_col, monetary_col]].copy()
    rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]

    # Handle zero/constant values
    rfm["Frequency"] = rfm["Frequency"].clip(lower=1)
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)

    rfm["R_Score"] = pd.qcut(rfm["Recency"], n_bins, labels=range(n_bins, 0, -1), duplicates="drop")
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))

    for col in ["R_Score", "F_Score", "M_Score"]:
        rfm[col] = rfm[col].astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def assign_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "VIP Loyalists"
        elif r >= 4 and f >= 3:
            return "Active Regulars"
        elif r >= 4 and f <= 2:
            return "Promising Newcomers"
        elif r <= 2 and f >= 4:
            return "At-Risk High Value"
        elif r <= 2 and m >= 4:
            return "Big Spenders Fading"
        elif r <= 2 and f <= 2:
            return "Hibernating"
        else:
            return "Needs Nurturing"

    rfm["RFM_Segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


# ──────────────────────────────────────────────
# K-Means
# ──────────────────────────────────────────────

def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 12),
    random_state: int = 42,
) -> dict:
    """Evaluate K-Means across K values. Returns metrics and optimal K."""
    metrics = {"inertia": [], "silhouette": [], "calinski": [], "davies": []}

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        metrics["inertia"].append(km.inertia_)
        metrics["silhouette"].append(silhouette_score(X_scaled, labels))
        metrics["calinski"].append(calinski_harabasz_score(X_scaled, labels))
        metrics["davies"].append(davies_bouldin_score(X_scaled, labels))

    kl = KneeLocator(list(k_range), metrics["inertia"], curve="convex", direction="decreasing")
    optimal_elbow = kl.knee if kl.knee else 4
    optimal_sil = list(k_range)[np.argmax(metrics["silhouette"])]

    return {
        "k_range": list(k_range),
        **metrics,
        "optimal_k_elbow": optimal_elbow,
        "optimal_k_silhouette": optimal_sil,
    }


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple:
    """Fit K-Means. Returns (model, labels)."""
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=25, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    return km, labels


# ──────────────────────────────────────────────
# GMM
# ──────────────────────────────────────────────

def fit_gmm(X_scaled: np.ndarray, n_components: int, random_state: int = 42) -> tuple:
    """Fit GMM. Returns (model, labels, probabilities)."""
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=random_state, n_init=5)
    labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    return gmm, labels, probs


def select_gmm_components(X_scaled: np.ndarray, k_range: range = range(2, 10),
                          random_state: int = 42) -> dict:
    """Select optimal GMM components via BIC/AIC."""
    bic, aic = [], []
    for k in k_range:
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=random_state, n_init=5)
        g.fit(X_scaled)
        bic.append(g.bic(X_scaled))
        aic.append(g.aic(X_scaled))
    return {
        "k_range": list(k_range), "bic": bic, "aic": aic,
        "optimal_k_bic": list(k_range)[np.argmin(bic)],
    }


# ──────────────────────────────────────────────
# DBSCAN & Hierarchical
# ──────────────────────────────────────────────

def fit_dbscan(X_scaled: np.ndarray, eps: float = 1.5, min_samples: int = 10) -> np.ndarray:
    """Fit DBSCAN. Labels of -1 are outliers/VIPs."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)


def fit_hierarchical(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit Agglomerative Clustering with Ward linkage."""
    return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X_scaled)
