# ECG/HRV Preprocessing Plan — MESA Sleep Dataset

**Project:** SCDL3991 Sleep Phenotyping
**Phase:** 2 — Multi-modal physiological clustering
**Modality scope (per Jie, May 2026):** ECG only, with HRV features
**Last updated:** May 2026

---

## 1. Purpose

This document specifies the preprocessing pipeline for transforming raw ECG signals from MESA polysomnography recordings into per-epoch heart-rate-variability (HRV) feature vectors suitable for unsupervised clustering. The pipeline is designed to be modular and reusable, so the same scaffolding can later be extended to EEG, SpO2, and respiration modalities (future phases).

## 2. Relationship to Ma et al. (2026)

Our approach is informed by Ma et al. (2026), the paper Jie attached to his preparation email. We adopt their general framework but make several deliberate departures driven by our different scope and dataset.

**What we adopt from Ma et al.:**
- The high-level pipeline structure: filtering → bad-epoch identification → 30-second non-overlapping epoch segmentation aligned with AASM annotations → per-epoch feature extraction → normalisation → clustering
- The 30-second epoch convention aligned with the AASM annotation grid
- The Scaled Robust Sigmoid (SRS) normalisation approach using median and IQR (their Equation 1)
- The principle of mapping unsupervised clusters back to AASM ground-truth labels for evaluation
- The explicit handling of bad epochs (epochs with detected signal quality issues are discarded)

**What we deliberately deviate from:**
- *Modality:* Ma et al. used EEG, EOG, and EMG only — they did not use ECG, SpO2, or respiration. We focus on ECG, which is outside their scope. This is partly why Jie is interested in this direction: it extends their framework to physiological modalities they did not consider.
- *Power-line frequency:* Ma et al. used a 50 Hz notch filter because their data was Chinese (CNC) and Dutch (HMC, DHC). MESA was collected in the United States, so we use a 60 Hz notch filter.
- *Bandpass ranges:* Ma et al. used 0.3–35 Hz for EEG/EOG and 10–60 Hz for EMG. ECG has different filtering requirements (typically 0.5–40 Hz to preserve QRS morphology while removing baseline wander and high-frequency noise).
- *Ocular artefact removal:* Ma et al. used stationary wavelet transform with adaptive thresholding to remove eye-movement contamination from EEG. This step is not relevant for ECG and we omit it.
- *Sleep cycle / wake selection:* Ma et al. excluded prolonged wake epochs to avoid alpha-power contamination of EEG features. This concern is EEG-specific. For ECG/HRV, we include all sleep stages including wake epochs, since HRV during wake is itself diagnostically meaningful.
- *Feature space:* Ma et al. extracted 64–136 features mostly based on EEG band powers, cross-channel ratios, and functional connectivity. We extract a much smaller set (11 features) of standard HRV time-domain, frequency-domain, and nonlinear measures — the established literature standard for HRV analysis.
- *Clustering algorithm:* Ma et al. used Fuzzy Subspace Clustering (FSC). For our progress meeting, we will use the six clustering algorithms from Phase 1 (K-Means, Hierarchical, DBSCAN, Spectral, GMM, Density Peak) for direct comparison with our previous work. FSC is a candidate for future extension.

**What we extend beyond Ma et al.:**
- External evaluation metrics (NMI, AMI, ARI, F-score) explicitly requested by Jie. Ma et al. used a "mapping matrix" approach but did not report these standard external metrics directly.
- Multi-algorithm comparison (six algorithms vs. their one). This was an explicit limitation acknowledged in their Discussion section, where they wrote "future studies are expected to compare the results from multiple clustering algorithms".

## 3. Pipeline Overview

The ECG pipeline has six stages:

1. Load raw ECG signal from EDF
2. Bandpass filter to remove drift and high-frequency noise
3. Notch filter at 60 Hz to remove US power-line interference
4. R-peak detection
5. Segment into 30-second epochs aligned with AASM annotations
6. Extract HRV features per epoch

Output: a feature matrix of shape (n_epochs, n_features) per subject, plus a parallel array of AASM stage labels for ground-truth comparison.

## 4. Stage-by-Stage Specification

### Stage 1 — Loading

- Read EDF file using `mne.io.read_raw_edf`
- Extract the ECG channel only (channel name to be confirmed once data is available — likely "ECG" or "EKG")
- Confirmed sampling rate: 256 Hz (per MESA documentation)
- Load in `preload=True` mode for in-memory processing

### Stage 2 — Bandpass filtering

- Bandpass: 0.5 to 40 Hz, finite impulse response (FIR) filter
- Lower bound 0.5 Hz removes baseline drift from respiration and motion
- Upper bound 40 Hz removes muscle artefact and most high-frequency noise while preserving QRS morphology
- Implementation: `mne.filter.filter_data` or `scipy.signal.butter` + `filtfilt`

### Stage 3 — Notch filtering

- Notch at 60 Hz for US power-line removal (MESA was collected in the US — note this is the explicit deviation from Ma et al.'s 50 Hz)
- Quality factor Q ~30 (narrow notch to avoid distorting nearby cardiac frequency content)
- Implementation: `scipy.signal.iirnotch` + `filtfilt`

### Stage 4 — R-peak detection

- Use NeuroKit2's `nk.ecg_peaks` function with the default Pan-Tompkins-derived algorithm
- Returns indices of detected R-peaks
- Output: array of R-peak sample indices, plus RR intervals (differences between consecutive R-peaks, in seconds)
- Sanity-check pass: flag epochs with implausible RR intervals (under 0.3 s = >200 BPM, over 2.0 s = <30 BPM) as low-quality

### Stage 5 — Epoch segmentation

- Window length: 30 seconds (AASM convention, matches MESA annotation grid, aligned with Ma et al.)
- Step: 30 seconds (non-overlapping)
- Align epoch boundaries with the AASM annotation epochs from the XML file
- Each epoch labelled with its AASM stage (W / N1 / N2 / N3 / REM)
- Bad-epoch handling: discard epochs with fewer than 5 detected R-peaks (insufficient for HRV computation), or flagged as low-quality from Stage 4

### Stage 6 — HRV feature extraction

Compute, per 30-second epoch, the following feature set via NeuroKit2 (`nk.hrv_time`, `nk.hrv_frequency`, `nk.hrv_nonlinear`).

**Time-domain features (5):**
- Mean RR interval (mean NN)
- SDNN — standard deviation of NN intervals
- RMSSD — root mean square of successive NN differences
- pNN50 — proportion of NN intervals differing by more than 50 ms
- HR — mean heart rate (BPM)

**Frequency-domain features (4):**
- LF power — low-frequency power (0.04–0.15 Hz)
- HF power — high-frequency power (0.15–0.4 Hz)
- LF/HF ratio — sympathovagal balance proxy
- Total power — variance of the RR series

Note: standard HRV frequency analysis recommends 5-minute windows for stable frequency estimates. Within 30-second epochs, frequency-domain features will be noisier. We compute them anyway for consistency with the per-epoch design and acknowledge the limitation in writeup.

**Nonlinear features (2):**
- Sample entropy (SampEn)
- Detrended fluctuation analysis alpha-1 (DFA-alpha1)

**Total feature count per epoch: 11**

## 5. Output Specification

For each subject, the pipeline produces:

- `features`: numpy array of shape (n_valid_epochs, 11), dtype float
- `labels`: numpy array of shape (n_valid_epochs,), dtype string, values in {"W", "N1", "N2", "N3", "REM"}
- `epoch_indices`: numpy array of shape (n_valid_epochs,), original epoch indices in the recording
- `metadata`: dict containing subject ID, sampling rate, total recording duration, epochs discarded count, epochs discarded reasons

These are saved per-subject as `.npz` files in the SSD's `processed/` directory.

After all subjects are processed, we concatenate into a master matrix for clustering:
- `X`: shape (n_total_epochs, 11)
- `y`: shape (n_total_epochs,), AASM stage labels
- `subject_ids`: shape (n_total_epochs,), so we know which epoch belongs to which subject

## 6. Normalisation

After feature extraction across all subjects:
- Apply Scaled Robust Sigmoid (SRS) normalisation per feature, following Ma et al. (2026, Equation 1)
- This maps each feature into [0, 1] using the median m and IQR via: f(x) = 1 / (1 + exp(-(x - m) / (IQR / 1.35)))
- Robust to outliers compared to z-score normalisation
- Computed across the full dataset, not per-subject

Alternative: standard z-score normalisation. We will run both and compare.

## 7. Validation Checks

Before running on the full subject sample, validate the pipeline on:
- 1 subject end-to-end (sanity check, visual inspection of filtering and R-peak detection)
- 5 subjects (catch obvious failure modes — missing channels, very poor signal quality, extreme outliers)
- Then scale to the planned sample (~200 subjects)

Quality control metrics to track per-subject:
- Number of valid epochs / total epochs (should be > 90% for healthy recordings)
- Number of detected R-peaks total / expected total
- Distribution of mean HR across epochs (should be physiologically plausible, ~50–100 BPM for sleep)

## 8. Out of Scope (Future Phases)

- EEG processing — this is where the Ma et al. pipeline applies most directly (band powers, connectivity, wavelet artefact removal)
- EOG processing (eye movement features)
- EMG processing (muscle tone)
- SpO2 processing (desaturation events, ODI)
- Respiration processing (rate, variability, thoraco-abdominal asynchrony)
- Multi-modal fusion clustering (Phase 3 contribution)
- Fuzzy Subspace Clustering — Ma et al.'s algorithm of choice, candidate for Phase 3
- Subject-level (rather than epoch-level) phenotyping

## 9. References

1. Ma Y, Li C, Xu Y, Tan X, Yu X, Zhan CA. (2026). Unsupervised clustering of extensive physiological features substantiates five-stage sleep staging paradigm. *Sleep*, 49.
2. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *Circulation*, 93(5), 1043–1065.
3. Shaffer F, Ginsberg JP. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5:258.
4. Makowski D et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689–1696.
