"""
HRV feature extraction per 30-second epoch.

Implements 11 standard HRV features:
  Time-domain (5):    mean_nn, sdnn, rmssd, pnn50, mean_hr
  Frequency-domain (4): lf_power, hf_power, lf_hf_ratio, total_power
  Nonlinear (2):      sample_entropy, dfa_alpha1

Reference:
  Task Force (1996). Heart rate variability. Circulation, 93(5).
  Shaffer F, Ginsberg JP. (2017). Frontiers in Public Health, 5:258.
"""

import numpy as np
import warnings
from typing import Dict, List

HRV_FEATURE_NAMES = [
    "mean_nn", "sdnn", "rmssd", "pnn50", "mean_hr",
    "lf_power", "hf_power", "lf_hf_ratio", "total_power",
    "sample_entropy", "dfa_alpha1",
]
N_HRV_FEATURES = len(HRV_FEATURE_NAMES)


def compute_hrv_features(rr_intervals: np.ndarray) -> np.ndarray:
    """
    Compute all 11 HRV features for a single epoch.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds for one epoch.

    Returns
    -------
    features : np.ndarray
        1D array of shape (11,). NaN for features that can't be computed.
    """
    features = np.full(N_HRV_FEATURES, np.nan)

    if len(rr_intervals) < 3:
        return features

    rr_ms = rr_intervals * 1000  # convert to milliseconds

    try:
        # Time-domain features
        features[0] = np.mean(rr_ms)                    # mean_nn
        features[1] = np.std(rr_ms, ddof=1)             # sdnn
        successive_diffs = np.diff(rr_ms)
        features[2] = np.sqrt(np.mean(successive_diffs**2))  # rmssd
        features[3] = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100  # pnn50
        features[4] = 60000 / np.mean(rr_ms)            # mean_hr (BPM)
    except Exception:
        pass

    try:
        # Frequency-domain features
        # Only compute if we have enough RR intervals
        if len(rr_intervals) >= 5:
            import neurokit2 as nk
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hrv_freq = nk.hrv_frequency(
                    {'ECG_R_Peaks': np.cumsum(np.insert(rr_intervals, 0, 0))},
                    sampling_rate=1.0 / np.mean(rr_intervals),
                    show=False,
                    normalize=False,
                )
                if hrv_freq is not None and len(hrv_freq) > 0:
                    features[5] = hrv_freq.get('HRV_LF', [np.nan])[0] if 'HRV_LF' in hrv_freq else np.nan
                    features[6] = hrv_freq.get('HRV_HF', [np.nan])[0] if 'HRV_HF' in hrv_freq else np.nan
                    lf = features[5]
                    hf = features[6]
                    features[7] = lf / hf if (not np.isnan(hf) and hf > 0) else np.nan
                    features[8] = hrv_freq.get('HRV_TP', [np.nan])[0] if 'HRV_TP' in hrv_freq else np.nan
    except Exception:
        pass

    try:
        # Nonlinear features
        if len(rr_intervals) >= 5:
            # Sample entropy
            features[9] = _sample_entropy(rr_ms, m=2, r=0.2 * np.std(rr_ms, ddof=1))

            # DFA alpha1
            features[10] = _dfa_alpha1(rr_ms)
    except Exception:
        pass

    return features


def _sample_entropy(data: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Compute sample entropy of a time series.

    Parameters
    ----------
    data : np.ndarray
        Input time series.
    m : int
        Embedding dimension.
    r : float
        Tolerance. If None, uses 0.2 * std(data).
    """
    N = len(data)
    if N < m + 2:
        return np.nan

    if r is None:
        r = 0.2 * np.std(data, ddof=1)
    if r == 0:
        return np.nan

    def _count_matches(template_len):
        count = 0
        templates = np.array([data[i:i+template_len] for i in range(N - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)

    if B == 0:
        return np.nan

    return -np.log(A / B) if A > 0 else np.nan


def _dfa_alpha1(data: np.ndarray) -> float:
    """
    Compute DFA alpha-1 (short-term fractal scaling exponent).
    Uses box sizes from 4 to 16 samples.
    """
    N = len(data)
    if N < 16:
        return np.nan

    # Integrate the mean-subtracted series
    y = np.cumsum(data - np.mean(data))

    # Box sizes
    box_sizes = [4, 6, 8, 10, 12, 14, 16]
    box_sizes = [b for b in box_sizes if b <= N // 2]

    if len(box_sizes) < 2:
        return np.nan

    fluctuations = []
    for box_size in box_sizes:
        n_boxes = N // box_size
        if n_boxes == 0:
            continue

        rms_list = []
        for i in range(n_boxes):
            segment = y[i*box_size:(i+1)*box_size]
            x = np.arange(box_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_list.append(np.sqrt(np.mean((segment - trend)**2)))

        fluctuations.append(np.mean(rms_list))

    if len(fluctuations) < 2:
        return np.nan

    # Log-log fit
    log_n = np.log(box_sizes[:len(fluctuations)])
    log_f = np.log(fluctuations)

    # Remove any inf/nan
    valid = np.isfinite(log_n) & np.isfinite(log_f)
    if valid.sum() < 2:
        return np.nan

    alpha, _ = np.polyfit(log_n[valid], log_f[valid], 1)
    return alpha


def extract_features_for_subject(
    rr_per_epoch: List[np.ndarray],
    bad_mask: np.ndarray,
) -> np.ndarray:
    """
    Extract HRV features for all good epochs of one subject.

    Parameters
    ----------
    rr_per_epoch : list of np.ndarray
        RR intervals per epoch.
    bad_mask : np.ndarray
        Boolean array. True = bad epoch (skip).

    Returns
    -------
    features : np.ndarray
        2D array of shape (n_good_epochs, 11).
    """
    good_indices = np.where(~bad_mask)[0]
    features = np.full((len(good_indices), N_HRV_FEATURES), np.nan)

    for i, idx in enumerate(good_indices):
        features[i] = compute_hrv_features(rr_per_epoch[idx])

    return features


def normalise_features_srs(features: np.ndarray) -> np.ndarray:
    """
    Scaled Robust Sigmoid normalisation (Ma et al. 2026, Equation 1).

    f(x) = 1 / (1 + exp(-(x - median) / (IQR / 1.35)))

    Parameters
    ----------
    features : np.ndarray
        2D array of shape (n_epochs, n_features).

    Returns
    -------
    normalised : np.ndarray
        Same shape, all values in [0, 1].
    """
    normalised = np.zeros_like(features)

    for j in range(features.shape[1]):
        col = features[:, j]
        valid = col[~np.isnan(col)]

        if len(valid) < 4:
            normalised[:, j] = np.nan
            continue

        median = np.median(valid)
        q75, q25 = np.percentile(valid, [75, 25])
        iqr = q75 - q25

        if iqr == 0:
            normalised[:, j] = 0.5
        else:
            normalised[:, j] = 1.0 / (1.0 + np.exp(-(col - median) / (iqr / 1.35)))

    return normalised


def normalise_features_zscore(features: np.ndarray) -> np.ndarray:
    """Standard z-score normalisation per feature."""
    normalised = np.zeros_like(features)

    for j in range(features.shape[1]):
        col = features[:, j]
        valid = col[~np.isnan(col)]

        if len(valid) < 2:
            normalised[:, j] = np.nan
            continue

        mean = np.mean(valid)
        std = np.std(valid, ddof=1)

        if std == 0:
            normalised[:, j] = 0.0
        else:
            normalised[:, j] = (col - mean) / std

    return normalised
