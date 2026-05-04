# ---
# jupyter:
#   title: "MESA ECG/HRV Pipeline — Preprocessing, Feature Extraction, Clustering"
# ---

# %% [markdown]
# # MESA ECG/HRV Pipeline
#
# **Purpose:** End-to-end pipeline from raw MESA EDF files to clustering results.
#
# **Scope (per Jie, May 2026):** ECG modality only, with HRV features, ~200 subjects.
#
# **Pipeline stages:**
# 1. Load and explore one subject (sanity check)
# 2. Preprocess ECG signal (filter, R-peak detection)
# 3. Extract HRV features per 30-second epoch
# 4. Scale to ~200 subjects
# 5. Normalise features
# 6. Run six clustering algorithms
# 7. Evaluate with internal + external metrics
# 8. Cross-tabulate clusters vs AASM sleep stages
# 9. Generate plots and summary

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, '..')

from src.config import (
    MESA_RAW_DIR, PROCESSED_DIR, FEATURES_DIR,
    SAMPLING_RATES, EPOCH_LENGTH_SEC, NOTCH_FREQ_HZ,
    EEG_BANDPASS_HZ, SLEEP_STAGES,
)
from src.preprocessing import (
    load_ecg_from_edf, bandpass_filter, notch_filter,
    detect_r_peaks, segment_into_epochs, load_aasm_annotations,
    identify_bad_epochs,
)
from src.features import (
    extract_hrv_features_per_epoch, extract_hrv_features_all_epochs,
    normalise_features_srs, normalise_features_zscore,
    HRV_FEATURE_NAMES, N_HRV_FEATURES,
)
from src.clustering import run_all_baselines
from src.evaluation import evaluate_clustering, metrics_to_dataframe

# %% [markdown]
# ## 1. Explore One Subject
#
# Before running the full pipeline, we load a single subject to:
# - Confirm ECG channel name
# - Verify sampling rate is 256 Hz
# - Visually inspect the raw and filtered signal
# - Check that R-peak detection works
# - Parse the XML annotations and confirm sleep stage labels

# %%
# === 1a. List available EDF files ===
# TODO: Update this path once we know the exact directory structure
edf_dir = MESA_RAW_DIR
# edf_files = sorted(edf_dir.glob('*.edf'))
# print(f"Found {len(edf_files)} EDF files")
# print(f"First 5: {[f.name for f in edf_files[:5]]}")
print("TODO: Uncomment above once MESA data is on SSD")

# %%
# === 1b. Load one subject's ECG ===
# TODO: Fill in once channel name is confirmed
# example_edf = edf_files[0]
# ecg_signal, sr = load_ecg_from_edf(example_edf, ecg_channel_name="ECG")
# print(f"Subject: {example_edf.name}")
# print(f"Sampling rate: {sr} Hz")
# print(f"Signal length: {len(ecg_signal)} samples = {len(ecg_signal)/sr:.1f} seconds")
print("TODO: Uncomment above once MESA data is on SSD")

# %%
# === 1c. Visualise raw ECG (30 seconds) ===
# TODO: Uncomment once data is loaded
# fig, ax = plt.subplots(figsize=(14, 3))
# t = np.arange(30 * int(sr)) / sr
# ax.plot(t, ecg_signal[:30 * int(sr)], linewidth=0.5)
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Amplitude')
# ax.set_title(f'Raw ECG — first 30 seconds')
# plt.tight_layout()
# plt.savefig('../figures/raw_ecg_example.png', dpi=150)
# plt.show()
print("TODO: Uncomment above once MESA data is on SSD")

# %%
# === 1d. Filter the signal ===
# ecg_filtered = bandpass_filter(ecg_signal, sr, low_hz=0.5, high_hz=40.0)
# ecg_filtered = notch_filter(ecg_filtered, sr, notch_hz=60.0)
print("TODO: Uncomment above once preprocessing functions are implemented")

# %%
# === 1e. Detect R-peaks ===
# r_peaks, rr_intervals = detect_r_peaks(ecg_filtered, sr)
# print(f"Detected {len(r_peaks)} R-peaks")
# print(f"Mean HR: {60 / np.mean(rr_intervals):.1f} BPM")
# print(f"Mean RR: {np.mean(rr_intervals)*1000:.1f} ms")
print("TODO: Uncomment above once preprocessing functions are implemented")

# %%
# === 1f. Load AASM annotations ===
# xml_file = example_edf.with_suffix('.xml')  # or find the matching XML
# epoch_indices, sleep_stages = load_aasm_annotations(xml_file)
# print(f"Loaded {len(sleep_stages)} epoch annotations")
# print(f"Stage distribution: {pd.Series(sleep_stages).value_counts().to_dict()}")
print("TODO: Uncomment above once annotation parsing is implemented")

# %% [markdown]
# ## 2. Run Pipeline on ~200 Subjects
#
# Once the single-subject validation passes, scale to the full sample.

# %%
# === 2a. Select subjects ===
# np.random.seed(42)
# n_subjects = 200
# selected_files = np.random.choice(edf_files, size=min(n_subjects, len(edf_files)), replace=False)
# print(f"Selected {len(selected_files)} subjects for analysis")
print("TODO: Uncomment above once data is available")

# %%
# === 2b. Process all subjects ===
# all_features = []
# all_labels = []
# all_subject_ids = []
# failed_subjects = []
#
# for edf_file in tqdm(selected_files, desc="Processing subjects"):
#     try:
#         # Load
#         ecg_signal, sr = load_ecg_from_edf(edf_file)
#
#         # Filter
#         ecg_filtered = bandpass_filter(ecg_signal, sr)
#         ecg_filtered = notch_filter(ecg_filtered, sr)
#
#         # R-peaks
#         r_peaks, rr_intervals = detect_r_peaks(ecg_filtered, sr)
#
#         # Segment into epochs
#         epochs = segment_into_epochs(ecg_filtered, sr)
#
#         # Load annotations
#         xml_file = edf_file.with_suffix('.xml')
#         epoch_indices, sleep_stages = load_aasm_annotations(xml_file)
#
#         # Extract HRV features per epoch
#         # (need to split rr_intervals by epoch boundaries)
#         features = extract_hrv_features_all_epochs(rr_intervals_per_epoch)
#
#         # Identify and remove bad epochs
#         bad_mask = identify_bad_epochs(rr_intervals_per_epoch)
#         good_mask = ~bad_mask
#
#         all_features.append(features[good_mask])
#         all_labels.append(sleep_stages[good_mask])
#         all_subject_ids.append(np.full(good_mask.sum(), edf_file.stem))
#
#     except Exception as e:
#         failed_subjects.append((edf_file.name, str(e)))
#
# print(f"Processed: {len(selected_files) - len(failed_subjects)} subjects")
# print(f"Failed: {len(failed_subjects)} subjects")
# if failed_subjects:
#     for name, err in failed_subjects[:5]:
#         print(f"  {name}: {err}")
print("TODO: Uncomment above once all preprocessing functions are implemented")

# %%
# === 2c. Concatenate and normalise ===
# X = np.vstack(all_features)
# y = np.concatenate(all_labels)
# subject_ids = np.concatenate(all_subject_ids)
#
# print(f"Total epochs: {X.shape[0]}")
# print(f"Features per epoch: {X.shape[1]}")
# print(f"Stage distribution: {pd.Series(y).value_counts().to_dict()}")
#
# # Normalise using SRS (Ma et al. 2026)
# X_norm = normalise_features_srs(X)
print("TODO: Uncomment above once data is processed")

# %% [markdown]
# ## 3. Clustering and Evaluation

# %%
# === 3a. Run all six algorithms ===
# cluster_results = run_all_baselines(X_norm, n_clusters=5, dbscan_eps=2.0, dbscan_min_samples=5)
#
# for name, labels in cluster_results.items():
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     print(f"{name:15s}: {n_clusters} clusters")
print("TODO: Uncomment above once features are ready")

# %%
# === 3b. Evaluate all algorithms ===
# eval_results = {}
# for name, labels in cluster_results.items():
#     eval_results[name] = evaluate_clustering(X_norm, labels, ground_truth=y)
#
# df_metrics = metrics_to_dataframe(eval_results)
# print("\nMESA ECG/HRV — Full metrics comparison:")
# print("=" * 100)
# print(df_metrics.round(4).to_string())
# print("=" * 100)
#
# df_metrics.round(4).to_csv('../results/mesa_ecg_metrics.csv')
print("TODO: Uncomment above once clustering is done")

# %% [markdown]
# ## 4. Cross-Tabulation and Visualisation

# %%
# === 4a. K-Means cross-tabulation vs AASM stages ===
# ct = pd.crosstab(
#     pd.Series(y, name='AASM Stage'),
#     pd.Series(cluster_results['kmeans'], name='K-Means Cluster'),
# )
# print("K-Means vs AASM Sleep Stages:")
# print(ct)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
# ax.set_title('K-Means Clusters vs AASM Sleep Stages (ECG/HRV)')
# plt.tight_layout()
# plt.savefig('../figures/mesa_kmeans_crosstab.png', dpi=150)
# plt.show()
print("TODO: Uncomment above once clustering is done")

# %%
# === 4b. PCA visualisation ===
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_norm)
#
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#
# # Left: coloured by AASM ground truth
# for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
#     mask = y == stage
#     axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=stage, alpha=0.4, s=10)
# axes[0].set_title('PCA — AASM Ground Truth')
# axes[0].legend()
#
# # Right: coloured by K-Means clusters
# for k in sorted(set(cluster_results['kmeans'])):
#     mask = cluster_results['kmeans'] == k
#     axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {k}', alpha=0.4, s=10)
# axes[1].set_title('PCA — K-Means Clusters')
# axes[1].legend()
#
# for ax in axes:
#     ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
#     ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
#
# plt.tight_layout()
# plt.savefig('../figures/mesa_pca_comparison.png', dpi=150)
# plt.show()
print("TODO: Uncomment above once clustering is done")

# %% [markdown]
# ## 5. Summary
#
# Key findings from MESA ECG/HRV analysis:
# - TODO: fill in after running
# - Compare with Phase 1 tabular baseline results
# - Note which algorithms perform best on external metrics
# - Discuss what the cross-tabulation reveals about HRV vs sleep stages
