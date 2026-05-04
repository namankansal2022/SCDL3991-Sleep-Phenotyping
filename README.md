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

Earlier work on the Sleep Health & Lifestyle dataset is archived at:
https://github.com/namankansal2022/SCDL3991-Clustering-Analysis

---

## Repository Structure
SCDL3991-Sleep-Phenotyping/
├── notebooks/                    Jupyter notebooks, numbered by stage
├── src/                          Reusable Python modules
│   ├── config.py                 Paths and constants
│   ├── preprocessing.py          ECG signal preprocessing pipeline
│   ├── features.py               HRV feature extraction
│   ├── clustering.py             Clustering algorithms
│   └── evaluation.py             Internal and external metrics
├── data/tabular/                 Small Kaggle dataset (Phase 1 baseline rerun)
├── results/                      Small CSVs of experimental results
├── figures/                      PNG/PDF outputs
├── docs/                         Project documentation
│   ├── dataset_reference.md      MESA dataset structure and contents
│   ├── preprocessing_plan.md     ECG/HRV pipeline specification
│   └── novelty_directions.md     Candidate directions for methodological contribution
├── README.md
├── requirements.txt
└── .gitignore

The MESA dataset itself lives on an external SSD (not in this repo), with paths configured in `src/config.py`.

---

## Documentation

The `docs/` folder contains the design and reference documents for the project:

- **dataset_reference.md** — full description of the MESA Sleep Study: 2,056 subjects, seven physiological modalities, file formats, annotations, NSRR pre-computed variables, citation requirements, and open questions.
- **preprocessing_plan.md** — specification of the ECG/HRV preprocessing pipeline. Adapted from Ma et al. (2026), with explicit notes on what is adopted, what is deliberately changed (60 Hz notch for US data, ECG-specific bandpass), and what extends the original framework.
- **novelty_directions.md** — four candidate directions for a methodological contribution (multi-modal fusion, semi-supervised clustering, adaptive density peak clustering, deep representation learning). Discussion document for supervisor meeting.

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
- Zhang GQ, Cui L, Mueller R, et al. (2018). The National Sleep Research Resource: towards a sleep data commons. *J Am Med Inform Assoc*, 25(10):1351–1358.
- Chen X, Wang R, Zee P, et al. (2015). Racial/Ethnic Differences in Sleep Disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA). *Sleep*, 38(6):877–88.

Detailed dataset description: see `docs/dataset_reference.md`.

---

## Status

- [x] Project scaffolding and environment setup
- [x] Dataset reference document
- [x] Preprocessing plan document
- [x] Methodological novelty discussion document
- [x] Module skeletons with function signatures and docstrings
- [ ] MESA dataset acquired (in progress — in-person transfer scheduled)
- [ ] Preprocessing pipeline implemented and validated on a sample of subjects
- [ ] HRV features extracted across the sample
- [ ] Six clustering algorithms applied with internal and external metrics
- [ ] Cross-tabulation against AASM ground-truth sleep stages
- [ ] Methodological direction selected and prototyped

---

*Last updated: May 2026*
