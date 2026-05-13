# SCDL3991 — Sleep Phenotyping via Multi-Modal Physiological Clustering

**Researcher:** Naman Kansal

**Supervisors:** Professor Wei Chen, Dr Jie Yang

**Institution:** School of Biomedical Engineering, The University of Sydney

**Phase:** 2 — Multi-modal physiological signal analysis (in progress)

---

## Project Overview

This project applies unsupervised clustering to multi-modal physiological sleep data from the MESA (Multi-Ethnic Study of Atherosclerosis) Sleep Study, distributed via the National Sleep Research Resource (NSRR).

Phase 1 (separate repository) established a six-algorithm clustering baseline on a tabular sleep health dataset. Phase 2 extends this to real polysomnography signals, beginning with ECG and HRV features for the upcoming progress meeting, and laying infrastructure for later extension to EEG, EOG, EMG, SpO2, and respiration.

### Research framing

- Adapt the unsupervised clustering framework of Ma et al. (2026, *Sleep*) — which used EEG/EOG/EMG only — to physiological modalities they did not consider, beginning with ECG.
- Add external clustering metrics (NMI, AMI, ARI, F-score) on top of internal ones, per supervisor feedback.
- Begin developing a methodological contribution that goes beyond applying off-the-shelf algorithms.

### Phase 1 reference

Earlier work on the Sleep Health and Lifestyle dataset is archived at:
https://github.com/namankansal2022/SCDL3991-Clustering-Analysis

---

## Repository Structure

| Folder / File | Description |
|---|---|
| `notebooks/` | Jupyter notebooks, numbered by stage |
| `src/` | Reusable Python modules |
| `src/config.py` | Paths and constants (SSD data root, sampling rates, filter settings) |
| `src/preprocessing.py` | ECG signal preprocessing pipeline (filter, R-peaks, epochs) |
| `src/features.py` | HRV feature extraction (11 time/frequency/nonlinear features) |
| `src/clustering.py` | Six clustering algorithms with uniform interface |
| `src/evaluation.py` | Internal metrics (Silhouette, DB, CH) + external metrics (NMI, AMI, ARI, F-score) |
| `src/plotting.py` | Reusable plotting functions (PCA, bar charts, heatmaps, boxplots) |
| `data/tabular/` | Small Kaggle dataset for Phase 1 baseline rerun |
| `results/` | Small CSVs of experimental results |
| `figures/` | PNG/PDF outputs from analysis |
| `docs/` | Project documentation (see below) |
| `README.md` | This file |
| `requirements.txt` | Python dependencies for reproducibility |
| `.gitignore` | Files excluded from version control |

The MESA dataset itself lives on an external SSD (not in this repo), with paths configured in `src/config.py`.

---

## Documentation

The `docs/` folder contains the design and reference documents for the project:

| Document | Description |
|---|---|
| `dataset_reference.md` | Full description of the MESA Sleep Study: 2,056 subjects, seven physiological modalities, file formats, annotations, NSRR pre-computed variables, citation requirements, and open questions |
| `preprocessing_plan.md` | Specification of the ECG/HRV preprocessing pipeline, adapted from Ma et al. (2026), with explicit notes on what is adopted, what is deliberately changed (60 Hz notch for US data, ECG-specific bandpass), and what extends the original framework |
| `novelty_directions.md` | Four candidate directions for a methodological contribution (multi-modal fusion, semi-supervised clustering, adaptive density peak clustering, deep representation learning). Discussion document for supervisor meeting |

---

## Setup

```bash
# Create the conda environment
conda create -n scdl3991-mesa python=3.11 -y
conda activate scdl3991-mesa

# Install dependencies
pip install -r requirements.txt
```

---

## Data

**Source:** MESA Sleep Study, distributed via the National Sleep Research Resource (NSRR) at https://sleepdata.org/datasets/mesa

**Required citations** (when publishing or submitting work using this dataset):
- Zhang GQ, Cui L, Mueller R, et al. (2018). The National Sleep Research Resource: towards a sleep data commons. *J Am Med Inform Assoc*, 25(10):1351-1358.
- Chen X, Wang R, Zee P, et al. (2015). Racial/Ethnic Differences in Sleep Disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA). *Sleep*, 38(6):877-88.

Detailed dataset description: see `docs/dataset_reference.md`.

---


## MESA ECG/HRV Baseline Summary (May 2026)

### Objective

Validate the complete Phase 2 pipeline on one physiological modality (ECG) using
heart rate variability (HRV) features extracted from the MESA Sleep dataset.

### Dataset and Experimental Setup

- Dataset: MESA Sleep Study (NSRR)
- Modality: ECG only
- Features: 7 time-domain HRV features
- Subjects processed: 100
- Total labelled epochs extracted: 97,756
- Sleep stages: W, N1, N2, N3, REM
- Clustering sample used for evaluation: 10,000 epochs

### Methods

Six baseline clustering algorithms were applied, including:

1. K-Means
2. Agglomerative (Hierarchical) Clustering
3. DBSCAN
4. Gaussian Mixture Models (GMM)
5. Spectral Clustering
6. Birch

Evaluation included:

- Internal metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
- External metrics: ARI, AMI, NMI, Homogeneity, Completeness, V-measure, F-score

### Key Results

- K-Means achieved the strongest overall internal cluster quality.
- All external agreement metrics (ARI, AMI, NMI) were close to zero.
- Cross-tabulation showed that each cluster contained a mixture of sleep stages.
- HRV features alone do not recover the five AASM sleep stages reliably.

### Scientific Interpretation

These findings are consistent with the sleep literature:

- ECG/HRV captures broad autonomic changes across sleep.
- HRV is useful for distinguishing sleep from wake and some REM/NREM differences.
- HRV alone is insufficient for accurate five-stage sleep phenotyping.
- EEG remains the most informative modality for fine-grained sleep staging.

### Research Implication

The weak correspondence between HRV-based clusters and AASM labels provides a
clear justification for methodological innovation, particularly:

- Multi-modal fusion (ECG + EEG + SpO₂ + respiration)
- Semi-supervised clustering
- Adaptive Density Peak Clustering
- Deep representation learning

### Repository Outputs

Results:
- `results/mesa_features.npz`
- `results/mesa_ecg_metrics.csv`
- `results/mesa_kmeans_crosstab.csv`

Figures:
- `figures/mesa_pca_ground_truth.png`
- `figures/mesa_pca_kmeans.png`
- `figures/mesa_external_metrics.png`
- `figures/mesa_kmeans_crosstab.png`
- `figures/mesa_feature_distributions.png`

Supporting documentation:
- `docs/novelty_directions.md`

### Conclusion

The complete ECG/HRV pipeline has been successfully validated on real MESA
polysomnography data. The baseline analysis demonstrates that ECG-derived HRV
features contain meaningful physiological structure, but are insufficient on
their own to reproduce the five canonical sleep stages. This establishes a
strong foundation for the next phase of the project, focused on developing
novel multi-modal clustering methods.


## Status

- [x] Project scaffolding and environment setup
- [x] Dataset reference document
- [x] Preprocessing plan document
- [x] Methodological novelty discussion document
- [x] Module skeletons with function signatures and docstrings
- [x] Clustering module implemented and tested (6 algorithms)
- [x] Evaluation module implemented and tested (10 metrics)
- [x] Plotting module implemented (5 reusable functions)
- [x] Phase 1 baseline rerun with new external metrics (results in `results/`)
- [x] MESA dataset acquired and configured on external SSD
- [x] Preprocessing pipeline implemented and validated on 100 subjects
- [x] HRV features extracted across 97,756 epochs
- [x] Six clustering algorithms applied to MESA ECG/HRV features
- [x] Cross-tabulation against AASM ground-truth sleep stages
- [x] Results and figures generated and committed to GitHub
- [ ] Methodological direction selected and prototyped

---

*Last updated: May 2026*



# Modality Ablation Summary

## Best Performing Combinations

| Metric | Best Combination | Score |
|------:|------------------|------:|
| ARI | EEG + EOG | 0.0342 |
| NMI | EEG | 0.1143 |
| F1 Score | EEG + EOG + EMG | 0.3621 |

## Main Findings

- EEG was the most informative single modality.
- EOG provided complementary information and improved ARI.
- EMG improved weighted F1.
- HRV consistently reduced clustering performance.

## Recommended Final Model

EEG + EOG is the recommended modality combination for unsupervised sleep-stage clustering because it achieved the highest adjusted Rand index (ARI).

## Detailed Ranked Results (Top 5 by ARI)

| Rank | Combination | ARI | NMI | F1 |
|-----:|-------------|----:|----:|----:|
| 1 | EEG + EOG | 0.0342 | 0.1009 | 0.3412 |
| 2 | EEG + EMG | 0.0249 | 0.0994 | 0.3502 |
| 3 | EEG + EOG + EMG | 0.0243 | 0.0959 | 0.3621 |
| 4 | EEG | 0.0227 | 0.1143 | 0.3211 |
| 5 | EEG + HRV | 0.0220 | 0.0878 | 0.3305 |

## Report Conclusion

EEG was the strongest single modality, achieving the highest normalized mutual information. Adding EOG improved the adjusted Rand index and produced the best overall clustering agreement. EMG increased the weighted F1 score but did not improve ARI further. HRV consistently reduced performance. Therefore, EEG and EOG form the most effective modality combination for unsupervised sleep-stage clustering in this study.
