# SCDL3991 — Sleep Phenotyping via Multi-Modal Physiological Clustering

**Researcher:** Naman Kansal
**Supervisors:** Professor Wei Chen, Dr Jie Yang
**Institution:** School of Biomedical Engineering, The University of Sydney
**Phase:** 2 — Multi-modal physiological signal analysis

---

## Project Overview

This project applies unsupervised clustering and representation learning to multi-modal physiological sleep data from the MESA (Multi-Ethnic Study of Atherosclerosis) Sleep Study. Building on the Phase 1 baseline (six clustering algorithms applied to a tabular sleep health dataset), this phase extends the work to raw polysomnography signals across seven modalities: EEG, EOG, EMG, ECG, SpO2, and thoracic/abdominal respiration.

### Phase 1 reference

Earlier work on the Sleep Health & Lifestyle dataset is archived at:
https://github.com/namankansal2022/SCDL3991-Clustering-Analysis

---

## Project Structure
SCDL3991-Sleep-Phenotyping/
├── notebooks/          Jupyter notebooks, numbered by stage
├── src/                Reusable Python modules
├── data/tabular/       Small Kaggle dataset (Phase 1 baseline rerun)
├── results/            Small CSVs of experimental results
├── figures/            PNG/PDF outputs
├── docs/               Markdown reference documents
├── README.md
├── requirements.txt
└── .gitignore

The MESA dataset itself lives on an external SSD (not in this repo). See `src/config.py` for the path configuration.

---

## Setup

```bash
conda create -n scdl3991-mesa python=3.11 -y
conda activate scdl3991-mesa
pip install -r requirements.txt
```

---

## Data

**Source:** MESA Sleep Study, distributed via the National Sleep Research Resource (NSRR).

Detailed dataset description: see `docs/dataset_reference.md`.

---

*Last updated: May 2026*
