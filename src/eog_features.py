"""
EEG feature extraction for sleep staging.

For each 30-second epoch and each EEG channel, compute relative power in:
- Delta: 0.5–4 Hz
- Theta: 4–8 Hz
- Alpha: 8–12 Hz
- Sigma: 12–15 Hz
- Beta: 15–30 Hz

If 3 EEG channels are used (EEG1, EEG2, EEG3), this produces:
3 channels × 5 bands = 15 features per epoch.
"""

import numpy as np
from scipy.signal import welch

# Frequency bands (Hz)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}

EOG_FEATURE_NAMES = [
    f"{channel}_{band}"
    for channel in ["EOG-L", "EOG-R"]
    for band in EEG_BANDS.keys()
]

N_EOG_FEATURES = len(EOG_FEATURE_NAMES)  # 15


def compute_band_powers(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Compute relative band powers for one EEG signal.
    """
    if len(signal) == 0 or np.all(np.isnan(signal)):
        return np.full(len(EEG_BANDS), np.nan)

    freqs, psd = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(1024, len(signal)),
    )

    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = np.trapezoid(psd[total_mask], freqs[total_mask])

    if total_power <= 0 or np.isnan(total_power):
        return np.full(len(EEG_BANDS), np.nan)

    features = []

    for low, high in EEG_BANDS.values():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
        features.append(band_power / total_power)

    return np.array(features)


def compute_eog_features_epoch(
    epoch_data: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Compute EEG features for one epoch.
    epoch_data shape: (n_channels, n_samples)
    """
    all_features = []

    for ch in range(epoch_data.shape[0]):
        band_features = compute_band_powers(epoch_data[ch], sampling_rate)
        all_features.extend(band_features)

    return np.array(all_features)


def extract_eog_features_all_epochs(
    epochs: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Extract EEG features for all epochs.
    epochs shape: (n_epochs, n_channels, n_samples)
    """
    n_epochs = epochs.shape[0]
    features = np.full((n_epochs, N_EOG_FEATURES), np.nan)

    for i in range(n_epochs):
        features[i] = compute_eog_features_epoch(
            epochs[i],
            sampling_rate,
        )

    return features
