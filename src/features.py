"""
Feature extraction per modality.

Modalities and their feature sets:
  - EEG:  relative band powers (delta, theta, alpha, sigma),
          cross-frequency power ratios, cross-channel power ratios,
          functional connectivity (coherence, phase synchrony)
  - EOG:  slow eye movement (SEM), rapid eye movement (REM),
          general eye activity
  - EMG:  averaged EMG power
  - ECG:  HRV time-domain (SDNN, RMSSD, pNN50),
          HRV frequency-domain (LF, HF, LF/HF),
          nonlinear HRV (sample entropy)
  - SpO2: mean, min, time below 90%, ODI (oxygen desaturation index)
  - Resp: respiratory rate, respiratory variability,
          thoraco-abdominal asynchrony

All features computed per 30-second epoch unless noted otherwise.
"""

# To be implemented per modality as we work through the feature notebooks.
pass
