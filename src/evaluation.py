"""
Clustering evaluation metrics.

Internal metrics (no ground-truth needed):
  - Silhouette score
  - Davies-Bouldin index
  - Calinski-Harabasz index

External metrics (require ground-truth labels):
  - Adjusted Rand Index (ARI)
  - Adjusted Mutual Information (AMI)
  - Normalised Mutual Information (NMI)
  - F-score / F-measure
  - Homogeneity, Completeness, V-measure

External metrics requested explicitly in supervisor feedback (Jie, April 2026).

All metric implementations use scikit-learn under the hood.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute internal clustering quality metrics (no ground truth needed).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).

    Returns
    -------
    metrics : dict
        Keys: silhouette, davies_bouldin, calinski_harabasz
        - silhouette: range [-1, 1], higher is better
        - davies_bouldin: range [0, inf), lower is better
        - calinski_harabasz: range [0, inf), higher is better

    Notes
    -----
    Returns NaN values if fewer than 2 clusters are present (metrics undefined).
    DBSCAN's noise label (-1) is treated as a regular cluster by sklearn.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def compute_external_metrics(
    cluster_labels: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute external clustering metrics against ground-truth labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Predicted cluster labels of shape (n_samples,).
    ground_truth : np.ndarray
        Ground-truth labels of shape (n_samples,). Can be strings or integers.

    Returns
    -------
    metrics : dict
        Keys: ari, ami, nmi, f_score, homogeneity, completeness, v_measure
        - ari (Adjusted Rand Index): [-0.5, 1], higher is better, 0 = random
        - ami (Adjusted Mutual Information): [0, 1], higher is better, adjusted for chance
        - nmi (Normalised Mutual Information): [0, 1], higher is better
        - f_score: [0, 1], higher is better, computed via best cluster-to-class mapping
        - homogeneity: [0, 1], each cluster contains only members of a single class
        - completeness: [0, 1], all members of a class assigned to the same cluster
        - v_measure: [0, 1], harmonic mean of homogeneity and completeness

    Notes
    -----
    F-score requires aligning cluster IDs to ground-truth class IDs first.
    We use the Hungarian algorithm to find the optimal mapping.
    """
    # Encode ground-truth labels to integers if they're strings
    if ground_truth.dtype.kind in {"U", "O", "S"}:
        ground_truth_encoded = LabelEncoder().fit_transform(ground_truth)
    else:
        ground_truth_encoded = ground_truth

    metrics = {
        "ari": adjusted_rand_score(ground_truth_encoded, cluster_labels),
        "ami": adjusted_mutual_info_score(ground_truth_encoded, cluster_labels),
        "nmi": normalized_mutual_info_score(ground_truth_encoded, cluster_labels),
        "homogeneity": homogeneity_score(ground_truth_encoded, cluster_labels),
        "completeness": completeness_score(ground_truth_encoded, cluster_labels),
        "v_measure": v_measure_score(ground_truth_encoded, cluster_labels),
    }

    # F-score requires aligning cluster IDs to class IDs via best mapping
    metrics["f_score"] = _compute_f_score_with_alignment(
        cluster_labels, ground_truth_encoded
    )

    return metrics


def _compute_f_score_with_alignment(
    cluster_labels: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute F-score by first finding the optimal cluster-to-class mapping
    via the Hungarian algorithm.

    This is the standard approach for evaluating clustering against ground
    truth: cluster IDs are arbitrary, so we find the assignment that
    maximises agreement before computing F-score.
    """
    from scipy.optimize import linear_sum_assignment

    unique_clusters = np.unique(cluster_labels)
    unique_classes = np.unique(ground_truth)
    n_clusters = len(unique_clusters)
    n_classes = len(unique_classes)

    # Build cost matrix: -overlap (we want to maximise overlap = minimise -overlap)
    # Pad to square if dimensions differ
    size = max(n_clusters, n_classes)
    cost_matrix = np.zeros((size, size))

    for i, c in enumerate(unique_clusters):
        for j, t in enumerate(unique_classes):
            overlap = np.sum((cluster_labels == c) & (ground_truth == t))
            cost_matrix[i, j] = -overlap

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping from cluster ID to class ID
    cluster_to_class = {}
    for i, j in zip(row_ind, col_ind):
        if i < n_clusters and j < n_classes:
            cluster_to_class[unique_clusters[i]] = unique_classes[j]

    # Map cluster labels through the alignment
    aligned_labels = np.array([
        cluster_to_class.get(c, -999) for c in cluster_labels
    ])

    # Compute weighted F1
    return f1_score(ground_truth, aligned_labels, average="weighted", zero_division=0)


def evaluate_clustering(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the complete metric suite (internal + external if ground truth provided).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    cluster_labels : np.ndarray
        Predicted cluster labels of shape (n_samples,).
    ground_truth : np.ndarray, optional
        Ground-truth labels for external evaluation.

    Returns
    -------
    metrics : dict
        Combined dictionary of internal and (if applicable) external metrics.
    """
    metrics = compute_internal_metrics(X, cluster_labels)

    if ground_truth is not None:
        external = compute_external_metrics(cluster_labels, ground_truth)
        metrics.update(external)

    return metrics


def metrics_to_dataframe(
    results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Convert a dict of {algorithm_name: metrics_dict} into a tidy DataFrame.

    Parameters
    ----------
    results : dict
        Mapping from algorithm name to its metrics dict.

    Returns
    -------
    df : pd.DataFrame
        Algorithms as rows, metrics as columns.
    """
    return pd.DataFrame.from_dict(results, orient="index")
