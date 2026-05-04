"""
Reusable plotting functions for sleep phenotyping analysis.

All functions take data arrays as input and return matplotlib figures.
Consistent styling across Kaggle baseline and MESA analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Optional, Dict


def plot_pca_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA Scatter",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    alpha: float = 0.6,
    s: int = 30,
) -> plt.Figure:
    """
    PCA projection coloured by labels.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). PCA computed internally.
    labels : np.ndarray
        Labels for colouring (cluster IDs or ground-truth stage names).
    title : str
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)
    for label in sorted(set(labels)):
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(label), alpha=alpha, s=s)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pca_comparison(
    X: np.ndarray,
    ground_truth: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_name: str = "K-Means",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 6),
) -> plt.Figure:
    """
    Side-by-side PCA: ground truth (left) vs cluster labels (right).
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for label in sorted(set(ground_truth)):
        mask = ground_truth == label
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(label), alpha=0.4, s=10)
    axes[0].set_title('Ground Truth')
    axes[0].legend()

    for k in sorted(set(cluster_labels)):
        mask = cluster_labels == k
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {k}', alpha=0.4, s=10)
    axes[1].set_title(f'{cluster_name} Clusters')
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_metrics_bars(
    df_metrics: pd.DataFrame,
    metrics: list,
    titles: list,
    higher_is_better: list,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Bar chart comparing algorithms on specified metrics.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Algorithms as rows, metrics as columns.
    metrics : list of str
        Column names to plot.
    titles : list of str
        Title for each subplot.
    higher_is_better : list of bool
        For each metric, True if higher values are better.
    save_path : str, optional
    """
    n = len(metrics)
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, metric, title, hib in zip(axes, metrics, titles, higher_is_better):
        values = df_metrics[metric].dropna()
        best = values.max() if hib else values.min()
        colors = ['#2ecc71' if v == best else '#3498db' for v in values]
        ax.bar(values.index, values.values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_crosstab_heatmap(
    ground_truth: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_name: str = "Cluster",
    gt_name: str = "Ground Truth",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Heatmap of cross-tabulation between clusters and ground-truth labels.
    """
    ct = pd.crosstab(
        pd.Series(ground_truth, name=gt_name),
        pd.Series(cluster_labels, name=cluster_name),
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_title(f'{cluster_name} vs {gt_name}')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_feature_distributions(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    n_cols: int = 4,
    save_path: Optional[str] = None,
    figsize_per_subplot: tuple = (4, 3),
) -> plt.Figure:
    """
    Box plots of each feature grouped by label (sleep stage or cluster).
    Useful for inspecting whether features separate across groups.
    """
    n_features = X.shape[1]
    n_cols = min(n_cols, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    figsize = (figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = labels

    for i, feat in enumerate(feature_names):
        sns.boxplot(data=df, x='label', y=feat, ax=axes[i])
        axes[i].set_title(feat, fontsize=9)
        axes[i].tick_params(axis='x', rotation=45, labelsize=8)

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
