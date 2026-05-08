"""
Signal preprocessing pipeline for MESA polysomnography data.

Implements the ECG/HRV preprocessing approach specified in
docs/preprocessing_plan.md. Adapted from Ma et al. (2026) framework
with key deviations for ECG-specific processing and US power-line frequency.

Reference:
  Ma Y et al. (2026). Unsupervised clustering of extensive physiological
  features substantiates five-stage sleep staging paradigm. Sleep, 49.
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from scipy.signal import butter, filtfilt, iirnotch
import mne

from src.config import (
    ECG_CHANNEL_NAME, EPOCH_LENGTH_SEC, NOTCH_FREQ_HZ,
    MESA_STAGE_MAP, MESA_EDF_DIR, MESA_XML_DIR,
)


def load_ecg_from_edf(edf_path: Path, ecg_channel_name: str = None) -> Tuple[np.ndarray, float]:
    """
    Load the ECG channel from a MESA EDF file.

    Parameters
    ----------
    edf_path : Path
        Path to the EDF file for one subject.
    ecg_channel_name : str, optional
        Name of the ECG channel. Defaults to config value ("EKG" for MESA).

    Returns
    -------
    ecg_signal : np.ndarray
        1D array of ECG samples.
    sampling_rate : float
        Sampling rate in Hz (256 Hz for MESA).
    """
    if ecg_channel_name is None:
        ecg_channel_name = ECG_CHANNEL_NAME

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Pick only the ECG channel
    if ecg_channel_name not in raw.ch_names:
        raise ValueError(
            f"Channel '{ecg_channel_name}' not found. "
            f"Available channels: {raw.ch_names}"
        )

    raw.pick([ecg_channel_name])
    ecg_signal = raw.get_data()[0]  # shape (n_samples,)
    sampling_rate = raw.info['sfreq']

    return ecg_signal, sampling_rate


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
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
    order : int
        Filter order. Default 4.

    Returns
    -------
    filtered : np.ndarray
        Bandpass-filtered signal.
    """
    nyq = sampling_rate / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(
    signal: np.ndarray,
    sampling_rate: float,
    notch_hz: float = None,
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
    notch_hz : float, optional
        Notch frequency. Defaults to config value (60 Hz for US/MESA data).
    quality_factor : float
        Q factor of the notch.

    Returns
    -------
    filtered : np.ndarray
        Notch-filtered signal.
    """
    if notch_hz is None:
        notch_hz = NOTCH_FREQ_HZ

    b, a = iirnotch(notch_hz, quality_factor, sampling_rate)
    return filtfilt(b, a, signal)


def detect_r_peaks(
    signal: np.ndarray,
    sampling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks in the filtered ECG signal using NeuroKit2.

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
    """
    import neurokit2 as nk

    # Clean and detect R-peaks
    ecg_cleaned = nk.ecg_clean(signal, sampling_rate=int(sampling_rate))
    _, rpeaks_dict = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(sampling_rate))

    r_peak_indices = rpeaks_dict['ECG_R_Peaks']
    rr_intervals = np.diff(r_peak_indices) / sampling_rate  # in seconds

    return r_peak_indices, rr_intervals


def load_aasm_annotations(xml_path: Path) -> List[Dict]:
    """
    Parse the NSRR XML annotation file and extract sleep stage events.

    Parameters
    ----------
    xml_path : Path
        Path to the NSRR-format XML annotation file.

    Returns
    -------
    stage_events : list of dict
        Each dict has keys: 'stage' (AASM label), 'start' (seconds), 'duration' (seconds).
        Sorted by start time.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    events = root.findall('.//ScoredEvent')
    stage_events = []

    for e in events:
        concept = e.find('EventConcept')
        if concept is None:
            continue

        concept_text = concept.text
        if concept_text not in MESA_STAGE_MAP:
            continue

        start = float(e.find('Start').text)
        duration = float(e.find('Duration').text)
        stage = MESA_STAGE_MAP[concept_text]

        stage_events.append({
            'stage': stage,
            'start': start,
            'duration': duration,
        })

    # Sort by start time
    stage_events.sort(key=lambda x: x['start'])
    return stage_events


def expand_stages_to_epochs(
    stage_events: List[Dict],
    epoch_length: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand variable-duration stage events into fixed 30-second epochs.

    MESA annotations have variable durations (e.g., 90s of Stage 2).
    This function splits them into individual 30-second epochs.

    Parameters
    ----------
    stage_events : list of dict
        From load_aasm_annotations. Each has 'stage', 'start', 'duration'.
    epoch_length : int, optional
        Epoch length in seconds. Default from config (30s).

    Returns
    -------
    epoch_starts : np.ndarray
        Start time of each epoch in seconds.
    epoch_labels : np.ndarray
        AASM stage label for each epoch (W, N1, N2, N3, REM).
    """
    if epoch_length is None:
        epoch_length = EPOCH_LENGTH_SEC

    epoch_starts = []
    epoch_labels = []

    for event in stage_events:
        start = event['start']
        duration = event['duration']
        stage = event['stage']

        # How many full epochs fit in this event
        n_epochs = int(duration // epoch_length)

        for i in range(n_epochs):
            epoch_starts.append(start + i * epoch_length)
            epoch_labels.append(stage)

    return np.array(epoch_starts), np.array(epoch_labels)


def get_rr_intervals_per_epoch(
    r_peak_indices: np.ndarray,
    sampling_rate: float,
    epoch_starts: np.ndarray,
    epoch_length: int = None,
) -> List[np.ndarray]:
    """
    Split RR intervals into per-epoch arrays aligned with annotations.

    Parameters
    ----------
    r_peak_indices : np.ndarray
        Sample indices of detected R-peaks.
    sampling_rate : float
        Sampling rate in Hz.
    epoch_starts : np.ndarray
        Start time of each epoch in seconds (from expand_stages_to_epochs).
    epoch_length : int, optional
        Epoch length in seconds. Default from config (30s).

    Returns
    -------
    rr_per_epoch : list of np.ndarray
        Each element is the array of RR intervals (in seconds) for that epoch.
    """
    if epoch_length is None:
        epoch_length = EPOCH_LENGTH_SEC

    # Convert R-peak indices to times in seconds
    r_peak_times = r_peak_indices / sampling_rate

    rr_per_epoch = []
    for start in epoch_starts:
        end = start + epoch_length

        # Find R-peaks within this epoch
        mask = (r_peak_times >= start) & (r_peak_times < end)
        peaks_in_epoch = r_peak_times[mask]

        if len(peaks_in_epoch) > 1:
            rr = np.diff(peaks_in_epoch)
        else:
            rr = np.array([])

        rr_per_epoch.append(rr)

    return rr_per_epoch


def identify_bad_epochs(
    rr_per_epoch: List[np.ndarray],
    min_rr_count: int = 5,
    min_rr_sec: float = 0.3,
    max_rr_sec: float = 2.0,
) -> np.ndarray:
    """
    Identify epochs that should be discarded as bad quality.

    Parameters
    ----------
    rr_per_epoch : list of np.ndarray
        RR intervals per epoch.
    min_rr_count : int
        Minimum number of RR intervals required. Default 5.
    min_rr_sec : float
        Minimum plausible RR interval (0.3s = 200 BPM).
    max_rr_sec : float
        Maximum plausible RR interval (2.0s = 30 BPM).

    Returns
    -------
    is_bad : np.ndarray
        Boolean array. True = bad epoch (discard).
    """
    is_bad = np.zeros(len(rr_per_epoch), dtype=bool)

    for i, rr in enumerate(rr_per_epoch):
        # Too few RR intervals
        if len(rr) < min_rr_count:
            is_bad[i] = True
            continue

        # Any implausible RR intervals
        if np.any(rr < min_rr_sec) or np.any(rr > max_rr_sec):
            is_bad[i] = True

    return is_bad


def get_subject_ids(edf_dir: Path = None) -> List[str]:
    """
    Get list of all subject IDs from the EDF directory.

    Returns
    -------
    subject_ids : list of str
        e.g., ['0001', '0002', '0006', ...]
    """
    if edf_dir is None:
        edf_dir = MESA_EDF_DIR

    edf_files = sorted(edf_dir.glob('mesa-sleep-*.edf'))
    return [f.stem.replace('mesa-sleep-', '') for f in edf_files]


def get_edf_path(subject_id: str, edf_dir: Path = None) -> Path:
    """Get EDF file path for a subject ID."""
    if edf_dir is None:
        edf_dir = MESA_EDF_DIR
    return edf_dir / f'mesa-sleep-{subject_id}.edf'


def get_xml_path(subject_id: str, xml_dir: Path = None) -> Path:
    """Get NSRR XML annotation file path for a subject ID."""
    if xml_dir is None:
        xml_dir = MESA_XML_DIR
    return xml_dir / f'mesa-sleep-{subject_id}-nsrr.xml'
