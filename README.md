# SCDL3991 — Data-Driven Phenotyping of Sleep Signals via Clustering

**Researcher:** Naman Kansal

**Supervisors:** Professor Wei Chen, Dr Jie Yang

**Institution:** School of Biomedical Engineering, The University of Sydney

**Status:** Complete — final research report submitted

---

## Project Overview

This project investigates whether the five American Academy of Sleep Medicine
(AASM) sleep stages can be recovered from multi-modal physiological signals
**without expert labels**, using unsupervised clustering of polysomnography (PSG)
from the MESA (Multi-Ethnic Study of Atherosclerosis) Sleep Study, distributed via
the National Sleep Research Resource (NSRR).

The central finding is that **feature representation, not the choice of clustering
algorithm, is the dominant factor** in unsupervised sleep staging. The work
progresses along a "supervision ladder": from a heart-rate-variability baseline
that recovers essentially no stage structure, through richer EEG representations
that recover the stages without labels, to semi-supervised methods that use a small
labelling budget.

### Research framing

- Test whether the AASM five-stage paradigm emerges from unlabelled physiology,
  building on the unsupervised clustering framework of Ma et al. (2026, *Sleep*).
- Identify which feature representation and clustering algorithm maximise external
  agreement with expert AASM labels.
- Quantify the gain from temporal structure and a small labelling budget, and
  rigorously test whether deep representation learning helps at this dataset scale.

### Project lineage

Phase 1 (separate repository) established a six-algorithm clustering baseline on the
tabular Sleep Health and Lifestyle dataset:
https://github.com/namankansal2022/SCDL3991-Clustering-Analysis

This repository (Phase 2) extends that pipeline to real PSG signals and contains the
full final analysis and research report.

---

## Headline Results

All results are on 100 MESA subjects (127,401 thirty-second epochs), evaluated
against expert AASM labels. ARI = adjusted Rand index; metrics are calibrated
against a random baseline (ARI ≈ 0.000).

### The supervision ladder

| Regime | Method (labels) | ARI | Accuracy | Cohen's κ |
|---|---|---:|---:|---:|
| Baseline | Random assignment | 0.000 | 20.0% | 0.000 |
| Unsupervised | HRV + GMM (ECG only) | 0.025 | — | — |
| Unsupervised | Band-power EEG + DBSCAN | 0.210 | 57.7% | 0.333 |
| **Unsupervised** | **Rich EEG + DBSCAN** | **0.281** | **61.7%** | **0.395** |
| Unsupervised | Rich EEG + per-subject HMM | 0.198 | 45.3% | 0.309 |
| Semi-supervised | Self-training (10%) | 0.500 | 74.3% | 0.606 |
| Semi-supervised | Label spreading (10%, subject-CV) | 0.38 ± 0.04 | 66.3 ± 2.3% | 0.51 ± 0.03 |
| *(reference)* | *Supervised RF (subject-CV)* | *0.56 ± 0.07* | *77.1 ± 3.7%* | *0.65 ± 0.05* |

### Key findings

- **Representation dominates.** Replacing 15 band-power features with 39 rich
  spectral features (spectral entropy, Hjorth parameters, spectral edge frequency,
  etc.) raised unsupervised ARI from 0.21 to 0.28. Rich EEG alone outperformed rich
  EEG combined with EMG/SpO₂/respiration.
- **Temporal structure is the key unsupervised gain.** Adding temporal context
  improved ARI from 0.14 to 0.21; a per-subject hidden Markov model achieved the
  best minority-stage recovery (48.1% balanced accuracy).
- **A small labelling budget is highly cost-effective.** 10% of labels closes
  roughly half the gap between the unsupervised result and the supervised ceiling.
- **Per-stage gradient.** REM (F1 = 0.80) and N1 (F1 = 0.68) are recovered best,
  Wake (F1 = 0.14) worst, with ~40% of Wake epochs absorbed into N1.
- **Deep learning did not help at this scale.** Feature autoencoders (ARI 0.14),
  IDEC deep clustering (0.13), a raw-waveform 1D CNN (0.02) and consensus clustering
  (0.07) all underperformed the classical pipeline — an informative negative result.

The full analysis, figures and discussion are in the final report
(`SCDL3991_report.tex`).

---

## Repository Structure

| Folder / File | Description |
|---|---|
| `SCDL3991_report.tex` | Final research report (LaTeX, journal-article style) |
| `notebooks/` | Jupyter notebooks, numbered by stage |
| `src/` | Reusable Python modules (preprocessing, features, clustering, evaluation, plotting) |
| `results/` | CSVs of experimental results across all phases |
| `figures/report/` | Publication figures used in the final report |
| `figures/github/` | Baseline figures (HRV/ECG analysis) |
| `figures/` | Additional analysis outputs |
| `docs/` | Project documentation (dataset reference, preprocessing plan, novelty directions) |
| `build_report_assets.py` | Regenerates per-stage confusion matrix and PCA embedding |
| `generate_report_figures.py` | Regenerates the report summary figures |
| `make_kdistance_rich.py` | Regenerates the k-distance graph for the rich-feature space |
| `requirements.txt` | Python dependencies |

The MESA dataset itself lives on an external SSD (not in this repo); paths are
configured in `src/config.py`. Large feature files (`.npz`) are gitignored.

---

## Setup

```bash
conda create -n scdl3991-mesa python=3.11 -y
conda activate scdl3991-mesa
pip install -r requirements.txt
```

---

## Data

**Source:** MESA Sleep Study, via the National Sleep Research Resource (NSRR):
https://sleepdata.org/datasets/mesa

**Required citations:**
- Zhang GQ, Cui L, Mueller R, et al. (2018). The National Sleep Research Resource:
  towards a sleep data commons. *J Am Med Inform Assoc*, 25(10):1351-1358.
- Chen X, Wang R, Zee P, et al. (2015). Racial/Ethnic Differences in Sleep
  Disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA). *Sleep*,
  38(6):877-88.

Detailed dataset description: see `docs/dataset_reference.md`.

---

## Methodological Note

This README summarises final results. The project deliberately progressed from a
weak HRV baseline (ARI ≈ 0.025) to the rich-feature pipeline (ARI ≈ 0.28); the early
HRV results are retained in the project history and report as the honest starting
point that motivated the move to EEG-centred representations. The earlier modality
ablation (which used band-power features and reported ARI ≈ 0.03) has been
superseded by the rich-feature results above.

---

*Last updated: June 2026 — final report phase.*
