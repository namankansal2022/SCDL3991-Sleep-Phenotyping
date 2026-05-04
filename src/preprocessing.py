"""
Signal preprocessing pipeline for MESA polysomnography data.

Implements the ECG/HRV preprocessing approach specified in
docs/preprocessing_plan.md. Adapted from Ma et al. (2026) framework
with key deviations for ECG-specific processing and US power-line frequency.

Functions are organised in pipeline order:
  Stage 1: load_ecg_from_edf
  Stage 2: bandpass_filter
  Stage 3: notch_filter
  Stage 4: detect_r_peaks
  Stage 5: segment_into_epochs
  Stage 6: see features.py for HRV extraction

Reference:
  Ma Y et al. (2026). Unsupervised clustering of extensive physiological
  features substantiates five-stage sleep staging paradigm. Sleep, 49.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_ecg_from_edf(edf_path: Path, ecg_channel_name: str = "ECG") -> Tuple[np.ndarray, float]:
    """
    Load the ECG channel from a MESA EDF file.

    Parameters
    ----------
    edf_path : Path
        Path to the EDF file for one subject.
    ecg_channel_name : str
        Name of the ECG channel in the EDF. Default "ECG".
        May need to be adjusted once we inspect a real MESA file
        (e.g. could be "EKG" or include lead designation).

    Returns
    -------
    ecg_signal : np.ndarray
        1D array of ECG samples in microvolts.
    sampling_rate : float
        Sampling rate in Hz (expected: 256 Hz for MESA).

    Notes
    -----
    Uses mne.io.read_raw_edf with preload=True for in-memory processing.
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
) -> np.ndarray:
    """
    Apply a bandpass filter to the ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    low_hz : float
        Lower cutoff. Default 0.5 Hz (removes baseline drift).
    high_hz : float
        Upper cutoff. Default 40 Hz (preserves QRS, removes muscle artefact).

    Returns
    -------
    filtered : np.ndarray
        Bandpass-filtered signal.

    Notes
    -----
    These cutoffs are ECG-appropriate. They differ from Ma et al.'s
    EEG settings (0.3-35 Hz). Use FIR filter via mne.filter.filter_data
    or scipy.signal.butter + filtfilt.
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def notch_filter(
    signal: np.ndarray,
    sampling_rate: float,
    notch_hz: float = 60.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Apply a notch filter to remove power-line interference.

    Parameters
    ----------
    signal : np.ndarray
        Bandpass-filtered ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    notch_hz : float
        Notch frequency. Default 60 Hz (US power line).
        Note: this differs from Ma et al.'s 50 Hz (their data was European/Chinese).
    quality_factor : float
        Q factor of the notch. Higher Q = narrower notch.

    Returns
    -------
    filtered : np.ndarray
        Notch-filtered signal.

    Notes
    -----
    Use scipy.signal.iirnotch + filtfilt.
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def detect_r_peaks(
    signal: np.ndarray,
    sampling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks in the filtered ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Filtered ECG signal.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    r_peak_indices : np.ndarray
        Sample indices of detected R-peaks.
    rr_intervals : np.ndarray
        RR intervals in seconds (length = len(r_peak_indices) - 1).

    Notes
    -----
    Uses NeuroKit2's nk.ecg_peaks with default Pan-Tompkins-derived algorithm.
    Apply sanity check: flag epochs with implausible RR intervals
    (under 0.3s or over 2.0s).
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def segment_into_epochs(
    signal: np.ndarray,
    sampling_rate: float,
    epoch_length_sec: int = 30,
    annotation_start_sec: float = 0.0,
) -> np.ndarray:
    """
    Segment the continuous signal into 30-second non-overlapping epochs
    aligned with the AASM annotation grid.

    Parameters
    ----------
    signal : np.ndarray
        Filtered ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    epoch_length_sec : int
        Length of each epoch in seconds. Default 30 (AASM convention).
    annotation_start_sec : float
        Start time of the first annotation epoch, in seconds.
        Used to align segmentation with the XML annotation grid.

    Returns
    -------
    epochs : np.ndarray
        2D array of shape (n_epochs, epoch_length_sec * sampling_rate).
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def load_aasm_annotations(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse the NSRR XML annotation file and extract per-epoch sleep stages.

    Parameters
    ----------
    xml_path : Path
        Path to the NSRR-format XML annotation file for one subject.

    Returns
    -------
    epoch_indices : np.ndarray
        1D array of epoch indices (0, 1, 2, ...).
    sleep_stages : np.ndarray
        1D array of sleep stage labels (strings: W, N1, N2, N3, REM).
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def identify_bad_epochs(
    rr_intervals_per_epoch: list,
    min_r_peaks: int = 5,
) -> np.ndarray:
    """
    Identify epochs that should be discarded as bad quality.

    Parameters
    ----------
    rr_intervals_per_epoch : list of np.ndarray
        RR intervals for each epoch.
    min_r_peaks : int
        Minimum number of R-peaks required for HRV computation.
        Epochs with fewer are discarded. Default 5.

    Returns
    -------
    is_bad : np.ndarray
        Boolean array of shape (n_epochs,). True = bad epoch.
    """
    raise NotImplementedError("To be implemented when MESA data is available")
