"""
Clustering algorithms for sleep phenotyping.

Phase 1 baseline algorithms, ported into modular functions:
  - K-Means
  - Agglomerative Hierarchical (Ward linkage)
  - DBSCAN
  - Spectral Clustering
  - Gaussian Mixture Model
  - Density Peak Clustering (Rodriguez & Laio 2014)

Phase 2 additions (placeholders for future implementation):
  - Fuzzy Subspace Clustering (Gan & Wu 2008) — used by Ma et al.
  - Methodological contribution (TBD based on supervisor discussion)

All clustering functions follow a uniform interface:
  cluster_X(features, n_clusters=K, **kwargs) -> labels

This makes them swappable in evaluation pipelines.
"""

import numpy as np
from typing import Optional
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform


# ============================================================
# Standard clustering algorithms (sklearn-based)
# ============================================================


def cluster_kmeans(
    features: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """
    K-Means clustering.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters K. Default 5 (matches AASM sleep stages).
    random_state : int
        Random seed for reproducibility.
    n_init : int
        Number of random initialisations to try.

    Returns
    -------
    labels : np.ndarray
        Cluster labels of shape (n_samples,), values in [0, n_clusters).
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    return model.fit_predict(features)


def cluster_hierarchical(
    features: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "ward",
) -> np.ndarray:
    """
    Agglomerative hierarchical clustering with Ward linkage by default.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    linkage : str
        Linkage criterion. "ward" minimises within-cluster variance.

    Returns
    -------
    labels : np.ndarray
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(features)


def cluster_dbscan(
    features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """
    DBSCAN clustering. Discovers clusters automatically; noise points labelled -1.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    eps : float
        Neighbourhood radius. Critical parameter — tune via k-distance graph.
    min_samples : int
        Minimum points to form a dense region.

    Returns
    -------
    labels : np.ndarray
        Cluster labels; -1 indicates noise points.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(features)


def cluster_spectral(
    features: np.ndarray,
    n_clusters: int = 5,
    affinity: str = "rbf",
    random_state: int = 42,
) -> np.ndarray:
    """
    Spectral clustering using a graph Laplacian eigendecomposition.

    Parameters
    ----------
    features : np.ndarray
    n_clusters : int
    affinity : str
        Affinity kernel. "rbf" (Gaussian) is standard.
    random_state : int

    Returns
    -------
    labels : np.ndarray
    """
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        random_state=random_state,
        assign_labels="kmeans",
    )
    return model.fit_predict(features)


def cluster_gmm(
    features: np.ndarray,
    n_clusters: int = 5,
    covariance_type: str = "full",
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """
    Gaussian Mixture Model via Expectation-Maximisation.

    Parameters
    ----------
    features : np.ndarray
    n_clusters : int
        Number of mixture components.
    covariance_type : str
        "full", "tied", "diag", or "spherical".
    random_state : int
    n_init : int

    Returns
    -------
    labels : np.ndarray
    """
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )
    model.fit(features)
    return model.predict(features)


# ============================================================
# Density Peak Clustering (custom implementation)
# Rodriguez & Laio (2014), Science 344(6191), 1492-1496
# ============================================================


def cluster_density_peak(
    features: np.ndarray,
    n_clusters: int = 5,
    dc_percentile: float = 2.0,
) -> np.ndarray:
    """
    Density Peak Clustering (Rodriguez & Laio, 2014).

    For each point i, compute:
      rho_i: local density (Gaussian kernel with cutoff dc)
      delta_i: minimum distance to any point of higher density
    Cluster centres are points with high rho AND high delta.
    Remaining points assigned to the same cluster as their nearest higher-density neighbour.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    n_clusters : int
        Number of cluster centres to select (top-K by rho * delta).
    dc_percentile : float
        Percentile of pairwise distances used as the cutoff dc.
        Rodriguez & Laio recommend ~2% by default.

    Returns
    -------
    labels : np.ndarray
        Cluster labels of shape (n_samples,).
    """
    n = features.shape[0]

    # Pairwise distances
    distances = squareform(pdist(features))

    # Cutoff distance dc (percentile of all pairwise distances)
    dc = np.percentile(distances[distances > 0], dc_percentile)

    # Local density rho via Gaussian kernel
    rho = np.sum(np.exp(-((distances / dc) ** 2)), axis=1) - 1.0  # subtract self

    # Delta: min distance to a point of higher density
    # For the highest-density point, delta = max distance
    delta = np.zeros(n)
    sorted_idx = np.argsort(-rho)  # descending by rho

    delta[sorted_idx[0]] = distances[sorted_idx[0]].max()
    for i in range(1, n):
        higher_density_pts = sorted_idx[:i]
        delta[sorted_idx[i]] = distances[sorted_idx[i], higher_density_pts].min()

    # Cluster centres: top-K by rho * delta
    gamma = rho * delta
    centre_indices = np.argsort(-gamma)[:n_clusters]

    # Assign labels: each non-centre inherits from nearest higher-density neighbour
    labels = -1 * np.ones(n, dtype=int)
    for k, idx in enumerate(centre_indices):
        labels[idx] = k

    # Process points in descending density order
    for i in sorted_idx:
        if labels[i] != -1:
            continue
        # Find nearest higher-density point and inherit its label
        higher = np.where(rho > rho[i])[0]
        if len(higher) == 0:
            continue
        nearest = higher[np.argmin(distances[i, higher])]
        labels[i] = labels[nearest]

    return labels


# ============================================================
# Convenience: run all six baselines at once
# ============================================================


def run_all_baselines(
    features: np.ndarray,
    n_clusters: int = 5,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> dict:
    """
    Run all six Phase 1 baseline algorithms and return their labels.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features), preferably
        already normalised.
    n_clusters : int
        K for the algorithms that require a predefined cluster count.
    dbscan_eps : float
        Epsilon for DBSCAN. Tune separately via k-distance graph.
    dbscan_min_samples : int
        Min samples for DBSCAN.

    Returns
    -------
    results : dict
        Mapping from algorithm name to label array.
    """
    return {
        "kmeans": cluster_kmeans(features, n_clusters=n_clusters),
        "hierarchical": cluster_hierarchical(features, n_clusters=n_clusters),
        "dbscan": cluster_dbscan(features, eps=dbscan_eps, min_samples=dbscan_min_samples),
        "spectral": cluster_spectral(features, n_clusters=n_clusters),
        "gmm": cluster_gmm(features, n_clusters=n_clusters),
        "density_peak": cluster_density_peak(features, n_clusters=n_clusters),
    }
