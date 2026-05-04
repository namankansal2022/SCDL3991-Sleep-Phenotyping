"""
HRV feature extraction per 30-second epoch.

Implements 11 standard HRV features per the preprocessing plan:
  Time-domain (5):    mean_nn, sdnn, rmssd, pnn50, mean_hr
  Frequency-domain (4): lf_power, hf_power, lf_hf_ratio, total_power
  Nonlinear (2):      sample_entropy, dfa_alpha1

All features computed via NeuroKit2 (nk.hrv_time, nk.hrv_frequency, nk.hrv_nonlinear).

Note on frequency-domain features: standard HRV literature recommends 5-minute
windows for stable frequency estimates. Within 30-second epochs they will be
noisier; we compute them anyway for consistency with per-epoch design and
acknowledge the limitation in writeup.

Reference:
  Task Force (1996). Heart rate variability: standards of measurement,
  physiological interpretation, and clinical use. Circulation, 93(5).

  Shaffer F, Ginsberg JP. (2017). An overview of heart rate variability
  metrics and norms. Frontiers in Public Health, 5:258.
"""

import numpy as np
from typing import Dict


# Canonical feature names in fixed order — matches column order in feature matrix
HRV_FEATURE_NAMES = [
    "mean_nn",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_hr",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "total_power",
    "sample_entropy",
    "dfa_alpha1",
]
N_HRV_FEATURES = len(HRV_FEATURE_NAMES)


def compute_hrv_time_domain(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain HRV features for a single epoch.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds for one 30-second epoch.

    Returns
    -------
    features : dict
        Keys: mean_nn, sdnn, rmssd, pnn50, mean_hr
        - mean_nn (ms): mean of NN intervals
        - sdnn (ms): standard deviation of NN intervals
        - rmssd (ms): root mean square of successive differences
        - pnn50 (%): proportion of successive differences > 50 ms
        - mean_hr (BPM): mean heart rate
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def compute_hrv_frequency_domain(
    rr_intervals: np.ndarray,
    sampling_rate: float = 4.0,
) -> Dict[str, float]:
    """
    Compute frequency-domain HRV features for a single epoch.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds for one 30-second epoch.
    sampling_rate : float
        Resampling rate for the RR series before spectral analysis.
        Default 4 Hz (NeuroKit2 default).

    Returns
    -------
    features : dict
        Keys: lf_power, hf_power, lf_hf_ratio, total_power
        - lf_power (ms^2): low-frequency band power (0.04-0.15 Hz)
        - hf_power (ms^2): high-frequency band power (0.15-0.4 Hz)
        - lf_hf_ratio: ratio of LF to HF power
        - total_power (ms^2): variance of the RR series

    Notes
    -----
    Frequency-domain analysis is statistically noisy on short (30s) windows.
    Standard recommendation is 5-minute windows. We accept this limitation
    for per-epoch consistency.
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def compute_hrv_nonlinear(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute nonlinear HRV features for a single epoch.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds for one 30-second epoch.

    Returns
    -------
    features : dict
        Keys: sample_entropy, dfa_alpha1
        - sample_entropy: sample entropy (regularity)
        - dfa_alpha1: detrended fluctuation analysis alpha-1 exponent
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def extract_hrv_features_per_epoch(rr_intervals: np.ndarray) -> np.ndarray:
    """
    Compute the full HRV feature vector (11 features) for a single epoch.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds for one 30-second epoch.

    Returns
    -------
    feature_vector : np.ndarray
        1D array of shape (11,), in the order specified by HRV_FEATURE_NAMES.
        NaN for any feature that fails to compute (e.g. too few RR intervals).
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def extract_hrv_features_all_epochs(
    rr_intervals_per_epoch: list,
) -> np.ndarray:
    """
    Compute HRV feature matrix across all epochs for one subject.

    Parameters
    ----------
    rr_intervals_per_epoch : list of np.ndarray
        List of RR interval arrays, one per epoch.

    Returns
    -------
    features : np.ndarray
        2D array of shape (n_epochs, 11).
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def normalise_features_srs(features: np.ndarray) -> np.ndarray:
    """
    Apply Scaled Robust Sigmoid (SRS) normalisation per feature.

    Following Ma et al. (2026), Equation 1:
        f(x) = 1 / (1 + exp(-(x - m) / (IQR / 1.35)))

    where m is the median and IQR is the interquartile range of the feature
    across all epochs in the dataset. Maps each feature to [0, 1] and is
    robust to outliers.

    Parameters
    ----------
    features : np.ndarray
        2D array of shape (n_epochs, n_features). Computed across the full
        dataset, NOT per-subject.

    Returns
    -------
    normalised : np.ndarray
        2D array of shape (n_epochs, n_features), all values in [0, 1].
    """
    raise NotImplementedError("To be implemented when MESA data is available")


def normalise_features_zscore(features: np.ndarray) -> np.ndarray:
    """
    Standard z-score normalisation per feature, as an alternative to SRS.

    Parameters
    ----------
    features : np.ndarray
        2D array of shape (n_epochs, n_features).

    Returns
    -------
    normalised : np.ndarray
        2D array of shape (n_epochs, n_features), each feature with
        mean 0 and standard deviation 1.
    """
    raise NotImplementedError("To be implemented when MESA data is available")
