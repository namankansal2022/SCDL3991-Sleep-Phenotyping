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

