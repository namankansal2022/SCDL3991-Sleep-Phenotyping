# Modality Ablation Summary

## Best Performing Combinations

| Metric | Best Combination | Score |
|------:|------------------|------:|
| ARI | EEG + EOG | 0.0342 |
| NMI | EEG | 0.1143 |
| F1 Score | EEG + EOG + EMG | 0.3621 |

## Main Findings

- EEG was the most informative single modality.
- EOG provided complementary information and improved ARI.
- EMG improved weighted F1.
- HRV consistently reduced clustering performance.

## Recommended Final Model

EEG + EOG is the recommended modality combination for unsupervised sleep-stage clustering because it achieved the highest adjusted Rand index (ARI).

## Detailed Ranked Results (Top 5 by ARI)

| Rank | Combination | ARI | NMI | F1 |
|-----:|-------------|----:|----:|----:|
| 1 | EEG + EOG | 0.0342 | 0.1009 | 0.3412 |
| 2 | EEG + EMG | 0.0249 | 0.0994 | 0.3502 |
| 3 | EEG + EOG + EMG | 0.0243 | 0.0959 | 0.3621 |
| 4 | EEG | 0.0227 | 0.1143 | 0.3211 |
| 5 | EEG + HRV | 0.0220 | 0.0878 | 0.3305 |

## Report Conclusion

EEG was the strongest single modality, achieving the highest normalized mutual information. Adding EOG improved the adjusted Rand index and produced the best overall clustering agreement. EMG increased the weighted F1 score but did not improve ARI further. HRV consistently reduced performance. Therefore, EEG and EOG form the most effective modality combination for unsupervised sleep-stage clustering in this study.
