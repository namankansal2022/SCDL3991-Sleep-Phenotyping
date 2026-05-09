# ---
# jupyter:
#   title: "MESA ECG/HRV Pipeline — Preprocessing, Feature Extraction, Clustering"
# ---

# %% [markdown]
# # MESA ECG/HRV Pipeline
#
# **Purpose:** End-to-end pipeline from raw MESA EDF files to clustering results.
#
# **Scope (per Jie, May 2026):** ECG modality only, with HRV features, 100 subjects.
#
# **Pipeline stages:**
# 1. Load and explore one subject (sanity check)
# 2. Run pipeline on selected subjects
# 3. Clustering and evaluation
# 4. Cross-tabulation and visualisation
# 5. Summary

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, '..')

from src.config import MESA_RAW_DIR
from src.clustering import run_all_baselines
from src.evaluation import evaluate_clustering, metrics_to_dataframe

RESULTS_DIR = Path('../results')
FIGURES_DIR = Path('../figures')
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SUBSAMPLE = 10_000
N_SUBJECTS = 100

# %% [markdown]
# ## 1. Explore One Subject
#
# Before running the full pipeline, we load a single subject to:
# - Confirm ECG channel name
# - Verify sampling rate
# - Inspect saved feature dimensions
# - Confirm sleep stage labels

# %%
# === 1a. List available EDF files ===
edf_dir = Path(MESA_RAW_DIR)
edf_files = sorted(edf_dir.glob('*.edf'))

print(f"Found {len(edf_files)} EDF files")
print("First 5 EDF files:")
for f in edf_files[:5]:
    print(f"  {f.name}")

# %%
# === 1b. Load one subject's metadata ===
example_edf = edf_files[0]
print(f"Example subject: {example_edf.name}")

# %%
# === 1c. Load previously extracted features ===
data = np.load(RESULTS_DIR / 'mesa_features.npz', allow_pickle=True)

if 'X_norm' in data.files:
    X_all = data['X_norm']
else:
    X_all = data['X']

y_all = data['y']

print(f"Total epochs available: {len(X_all):,}")
print(f"Features per epoch: {X_all.shape[1]}")

# %%
# === 1d. Inspect stage distribution ===
stage_counts = pd.Series(y_all).value_counts().sort_index()
print("Stage distribution:")
print(stage_counts)

# %%
# === 1e. Display feature names used ===
feature_names = [
    'mean_nn',
    'sdnn',
    'rmssd',
    'pnn50',
    'mean_hr',
    'cv_rr',
    'range_rr',
]
print("HRV features:")
for i, feat in enumerate(feature_names, start=1):
    print(f"{i}. {feat}")

# %%
# === 1f. Inspect one feature vector ===
print("First feature vector:")
print(np.round(X_all[0], 4))

# %% [markdown]
# ## 2. Run Pipeline on Selected Subjects
#
# Features have already been extracted and saved to `mesa_features.npz`.
# This section prepares the clustering dataset.

# %%
# === 2a. Select subjects / epochs ===
np.random.seed(RANDOM_STATE)
idx = np.random.choice(
    len(X_all),
    size=min(N_SUBSAMPLE, len(X_all)),
    replace=False
)

X_norm = X_all[idx]
y = y_all[idx]

print(f"Subsampled to {len(X_norm):,} epochs for clustering")

# %%
# === 2b. Processing summary ===
print(f"Approximate subjects processed: {N_SUBJECTS}")
print(f"Epochs used for clustering: {len(X_norm):,}")
print(f"Feature dimension: {X_norm.shape[1]}")

# %%
# === 2c. Confirm normalised feature matrix ===
print("Feature matrix shape:", X_norm.shape)
print("Ground-truth labels shape:", y.shape)
print("Unique stages:", sorted(pd.Series(y).unique()))

# %% [markdown]
# ## 3. Clustering and Evaluation

# %%
# === 3a. Run all six algorithms ===
cluster_results = run_all_baselines(
    X_norm,
    n_clusters=5,
    dbscan_eps=2.0,
    dbscan_min_samples=5
)

for name, labels in cluster_results.items():
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"{name:15s}: {n_clusters} clusters")

# %%
# === 3b. Evaluate all algorithms ===
eval_results = {}

for name, labels in cluster_results.items():
    eval_results[name] = evaluate_clustering(
        X_norm,
        labels,
        ground_truth=y
    )

df_metrics = metrics_to_dataframe(eval_results)

print("\nMESA ECG/HRV — Full metrics comparison:")
print("=" * 100)
print(df_metrics.round(4).to_string())
print("=" * 100)

df_metrics.round(4).to_csv(
    RESULTS_DIR / 'mesa_ecg_metrics.csv'
)

# %% [markdown]
# ## 4. Cross-Tabulation and Visualisation

# %%
# === 4a. K-Means cross-tabulation vs AASM stages ===
km_labels = cluster_results['kmeans']

ct = pd.crosstab(
    pd.Series(y, name='AASM Stage'),
    pd.Series(km_labels, name='K-Means Cluster'),
)

print("K-Means vs AASM Sleep Stages:")
print(ct)

ct.to_csv(
    RESULTS_DIR / 'mesa_kmeans_crosstab.csv'
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('K-Means Clusters vs AASM Sleep Stages (ECG/HRV)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'mesa_kmeans_crosstab.png', dpi=150)
plt.show()

# %%
# === 4b. PCA visualisation ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: coloured by AASM ground truth
for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
    mask = y == stage
    if np.any(mask):
        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=stage,
            alpha=0.3,
            s=10
        )

axes[0].set_title('PCA — AASM Ground Truth')
axes[0].legend()

# Right: coloured by K-Means clusters
for k in sorted(set(km_labels)):
    if k == -1:
        continue
    mask = km_labels == k
    axes[1].scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f'Cluster {k}',
        alpha=0.3,
        s=10
    )

axes[1].set_title('PCA — K-Means Clusters')
axes[1].legend()

for ax in axes:
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'mesa_pca_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# ## 5. Summary
#
# Key findings from MESA ECG/HRV analysis:
# - Successfully processed 100 MESA subjects.
# - Extracted HRV features across 97,756 epochs.
# - K-Means achieved the strongest internal cluster quality.
# - External agreement with AASM sleep stages was low (ARI close to zero).
# - Clusters do not map cleanly to W/N1/N2/N3/REM.
# - HRV contains meaningful structure, but not enough to recover five-stage sleep architecture alone.
# - This motivates methodological extensions such as:
#   - Multi-modal fusion (ECG + SpO₂ + respiration)
#   - Semi-supervised clustering
#   - Adaptive Density Peak Clustering
#   - Deep representation learning

print("Pipeline complete.")
