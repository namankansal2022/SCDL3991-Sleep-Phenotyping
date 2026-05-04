# ---
# jupyter:
#   title: "Phase 1 Baseline Rerun — Kaggle Sleep Health Dataset"
# ---

# %% [markdown]
# # Phase 1 Baseline Rerun — Kaggle Sleep Health Dataset
#
# **Purpose:** Reproduce the Phase 1 clustering analysis in the new project structure,
# adding the four external metrics (NMI, AMI, ARI, F-score) requested by Jie.
#
# **Dataset:** Sleep Health and Lifestyle Dataset (374 subjects, 13 features)
#
# **Algorithms:** K-Means, Hierarchical, DBSCAN, Spectral, GMM, Density Peak
#
# **Metrics:** Silhouette, Davies-Bouldin, Calinski-Harabasz (Phase 1) +
# NMI, AMI, ARI, F-score, Homogeneity, Completeness, V-measure (new)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, '..')

from src.clustering import run_all_baselines
from src.evaluation import evaluate_clustering, metrics_to_dataframe

# %%
# === 1. Load and inspect the dataset ===
df = pd.read_csv('../data/tabular/Sleep_health_and_lifestyle_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nFirst 3 rows:")
df.head(3)

# %%
# === 2. Data Cleaning (same as Phase 1) ===

# Fill missing Sleep Disorder values with 'None'
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# Unify BMI labels
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

# Split Blood Pressure into Systolic and Diastolic
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df = df.drop(columns=['Blood Pressure'])

# Drop Person ID
df = df.drop(columns=['Person ID'])

# Save ground truth labels before encoding
ground_truth = df['Sleep Disorder'].values
print(f"Class distribution:\n{pd.Series(ground_truth).value_counts()}")
print(f"\nCleaned dataset shape: {df.shape}")

# %%
# === 3. Encoding categorical variables ===
le_dict = {}
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# %%
# === 4. Feature scaling ===

# Separate features from target
feature_cols = [c for c in df.columns if c != 'Sleep Disorder']
X = df[feature_cols].values
y = ground_truth  # original string labels for external metrics

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nScaled feature matrix shape: {X_scaled.shape}")
print(f"Mean (should be ~0): {X_scaled.mean(axis=0).round(3)[:5]}...")
print(f"Std  (should be ~1): {X_scaled.std(axis=0).round(3)[:5]}...")

# %%
# === 5. PCA Visualisation ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for label in ['None', 'Insomnia', 'Sleep Apnea']:
    mask = y == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.6, s=30)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('PCA — Sleep Disorder Ground Truth Labels')
ax.legend()
plt.tight_layout()
plt.savefig('../figures/pca_ground_truth.png', dpi=150)
plt.show()
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}")

# %%
# === 6. Run all six clustering algorithms ===
#
# Using K=3 to match the three clinical categories (None, Insomnia, Sleep Apnea).
# DBSCAN eps tuned to match Phase 1 settings.

all_labels = run_all_baselines(
    X_scaled,
    n_clusters=3,
    dbscan_eps=2.5,
    dbscan_min_samples=5,
)

for name, labels in all_labels.items():
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"{name:15s}: {n_clusters} clusters, {n_noise} noise points")

# %%
# === 7. Evaluate all algorithms — INTERNAL + EXTERNAL metrics ===
#
# This is the key addition over Phase 1: external metrics (NMI, AMI, ARI, F-score)
# as requested by Jie in his feedback email.

results = {}
for name, labels in all_labels.items():
    results[name] = evaluate_clustering(X_scaled, labels, ground_truth=y)

df_metrics = metrics_to_dataframe(results)
print("\nFull metrics comparison (Phase 1 + new external metrics):")
print("=" * 100)
print(df_metrics.round(4).to_string())
print("=" * 100)

# Save to CSV
df_metrics.round(4).to_csv('../results/tabular_baseline_metrics.csv')
print("\nSaved to results/tabular_baseline_metrics.csv")

# %%
# === 8. Visualise metrics comparison ===

# Internal metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
internal_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
titles = ['Silhouette Score (higher=better)', 'Davies-Bouldin (lower=better)', 'Calinski-Harabasz (higher=better)']

for ax, metric, title in zip(axes, internal_metrics, titles):
    values = df_metrics[metric].dropna()
    colors = ['#2ecc71' if v == values.max() else '#3498db' for v in values] if 'davies' not in metric else \
             ['#2ecc71' if v == values.min() else '#3498db' for v in values]
    ax.bar(values.index, values.values, color=colors)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('../figures/internal_metrics_comparison.png', dpi=150)
plt.show()

# %%
# External metrics (NEW — requested by Jie)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
external_metrics = ['ari', 'ami', 'nmi', 'f_score']
ext_titles = ['ARI (higher=better)', 'AMI (higher=better)', 'NMI (higher=better)', 'F-score (higher=better)']

for ax, metric, title in zip(axes, external_metrics, ext_titles):
    values = df_metrics[metric].dropna()
    colors = ['#2ecc71' if v == values.max() else '#e74c3c' for v in values]
    ax.bar(values.index, values.values, color=colors)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(values.max() * 1.2, 0.5))
plt.tight_layout()
plt.savefig('../figures/external_metrics_comparison.png', dpi=150)
plt.show()

# %%
# === 9. Cross-tabulation: K-Means clusters vs Sleep Disorder ===
#
# This was the strongest finding from Phase 1 — checking if it holds
# with the same preprocessing in the new codebase.

kmeans_labels = all_labels['kmeans']
ct = pd.crosstab(
    pd.Series(y, name='Sleep Disorder'),
    pd.Series(kmeans_labels, name='K-Means Cluster'),
)
print("\nK-Means Cross-Tabulation vs Sleep Disorder:")
print(ct)
print()

# Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('K-Means Clusters vs Sleep Disorder Labels')
plt.tight_layout()
plt.savefig('../figures/kmeans_crosstab_heatmap.png', dpi=150)
plt.show()

# %%
# === 10. Summary ===
#
# Key findings from this baseline rerun:
# - All six algorithms from Phase 1 ported successfully to the new modular codebase
# - Four new external metrics (NMI, AMI, ARI, F-score) now computed alongside
#   the original three internal metrics — directly addressing Jie's feedback
# - Cross-tabulation confirms the cluster-disorder correspondence from Phase 1
# - This notebook serves as a template for the MESA ECG/HRV analysis:
#   same algorithms, same metrics, different input features

print("\nPhase 1 rerun complete.")
print(f"Algorithms tested: {len(results)}")
print(f"Metrics computed: {len(df_metrics.columns)} (3 internal + 7 external)")
print(f"Figures saved to: ../figures/")
print(f"Results saved to: ../results/tabular_baseline_metrics.csv")
