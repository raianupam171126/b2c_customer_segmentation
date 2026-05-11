from .data_generator import generate_customers, generate_orders, generate_marketo_activities
from .feature_engineering import (
    rollup_transactions, compute_product_affinity,
    rollup_engagement, estimate_clv, build_unified_matrix,
)
from .segmentation import (
    rfm_segmentation, find_optimal_k, fit_kmeans,
    fit_gmm, select_gmm_components,
    fit_dbscan, fit_hierarchical,
)
from .persona_builder import assign_personas, profile_personas, get_playbook

__all__ = [
    "generate_customers", "generate_orders", "generate_marketo_activities",
    "rollup_transactions", "compute_product_affinity",
    "rollup_engagement", "estimate_clv", "build_unified_matrix",
    "rfm_segmentation", "find_optimal_k", "fit_kmeans",
    "fit_gmm", "select_gmm_components", "fit_dbscan", "fit_hierarchical",
    "assign_personas", "profile_personas", "get_playbook",
]
