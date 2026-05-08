"""
Project configuration — paths, constants, and shared settings.

Code lives on the laptop (~/Documents/SCDL3991-Sleep-Phenotyping/),
data lives on an external SSD (/Volumes/Expansion/SCDL3991-data/).

If the SSD mount point ever changes, update DATA_ROOT below.
"""

from pathlib import Path

# Project root on the laptop (resolved from this file's location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data root on the external SSD
DATA_ROOT = Path("/Volumes/Expansion/SCDL3991-data")

# SSD subdirectories
RAW_DIR = DATA_ROOT / "raw"
MESA_RAW_DIR = RAW_DIR / "mesa"
PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_DIR = DATA_ROOT / "features"

# Laptop subdirectories (small files only)
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
TABULAR_DATA_DIR = PROJECT_ROOT / "data" / "tabular"

# MESA sampling rates per channel
SAMPLING_RATES = {
    "EEG": 256,
    "EOG": 256,
    "EMG": 256,
    "ECG": 256,
    "SpO2": 1,
    "Thor": 32,
    "Abdo": 32,
}

# AASM convention: 30-second non-overlapping epochs
EPOCH_LENGTH_SEC = 30

# US power line frequency (MESA collected in the US — 60 Hz, not 50 Hz)
NOTCH_FREQ_HZ = 60

# Filter bands following Ma et al. (2026)
EEG_BANDPASS_HZ = (0.3, 35.0)
EOG_BANDPASS_HZ = (0.3, 35.0)
EMG_BANDPASS_HZ = (10.0, 100.0)

# AASM sleep stage labels in canonical order
SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]

# === CONFIRMED FROM ACTUAL MESA DATA (May 2026) ===

# ECG channel is labelled "EKG" in MESA EDFs (not "ECG")
ECG_CHANNEL_NAME = "EKG"

# MESA polysomnography directory structure
MESA_EDF_DIR = DATA_ROOT / "raw" / "mesa" / "polysomnography" / "edfs"
MESA_XML_DIR = DATA_ROOT / "raw" / "mesa" / "polysomnography" / "annotations-events-nsrr"

# Sleep stage label mapping from MESA XML to AASM convention
MESA_STAGE_MAP = {
    "Wake|0": "W",
    "Stage 1 sleep|1": "N1",
    "Stage 2 sleep|2": "N2",
    "Stage 3 sleep|3": "N3",
    "REM sleep|5": "REM",
}
