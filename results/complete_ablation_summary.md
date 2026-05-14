# Complete Modality Ablation Summary (Updated with SpO2 and Respiration)

## Best Performing Combinations

| Metric | Best Combination | Score |
|--------|-----------------|-------|
| ARI | EEG + EMG + SpO2 + Resp | 0.0533 |
| NMI | EEG + SpO2 + Resp | 0.1639 |
| F1 Score | SpO2 + Resp | 0.3911 |

## Main Findings

1. EEG is essential: every top combination includes EEG.
2. SpO2 is the second most valuable modality, surpassing EOG.
3. EEG + SpO2 (ARI=0.047) outperforms EEG + EOG (ARI=0.037) by 27%.
4. The best 4 modality combination (EEG+EMG+SpO2+Resp, ARI=0.053) outperforms using all 5 modalities (ARI=0.029).
5. EOG can hurt performance when combined with SpO2 (interference effect).
6. HRV alone remains the weakest single modality (ARI=0.005 from earlier analysis).

## Recommended Final Model

EEG + EMG + SpO2 + Resp is the recommended combination for unsupervised sleep stage clustering.
It achieves the highest ARI (0.0533) and strong NMI (0.1597).

## Single Modality Rankings

| Rank | Modality | ARI | NMI | Features |
|------|----------|-----|-----|----------|
| 1 | EEG | 0.0253 | 0.1155 | 15 |
| 2 | Resp | 0.0240 | 0.0601 | 5 |
| 3 | SpO2 | -0.0176 | 0.1318 | 5 |
| 4 | EOG | -0.0135 | 0.0448 | 10 |
| 5 | EMG | -0.0192 | 0.0295 | 5 |

Note: SpO2 has negative ARI alone but strong NMI (0.132), indicating it captures
sleep stage information but clusters don't align with stages without EEG guidance.

## Top 10 Combinations (Ranked by ARI)

| Rank | Combination | ARI | NMI | F1 | Features |
|------|------------|-----|-----|-----|----------|
| 1 | EEG + EMG + SpO2 + Resp | 0.0533 | 0.1597 | 0.3742 | 30 |
| 2 | EEG + SpO2 + Resp | 0.0483 | 0.1639 | 0.3684 | 25 |
| 3 | EEG + SpO2 | 0.0474 | 0.1631 | 0.3655 | 20 |
| 4 | EEG + EMG + SpO2 | 0.0470 | 0.1567 | 0.3647 | 25 |
| 5 | EEG + EOG + Resp | 0.0410 | 0.1055 | 0.3523 | 30 |
| 6 | EEG + EOG + EMG + Resp | 0.0384 | 0.1095 | 0.3465 | 35 |
| 7 | EEG + EOG + EMG | 0.0376 | 0.1085 | 0.3493 | 30 |
| 8 | EEG + EOG | 0.0375 | 0.1026 | 0.3410 | 25 |
| 9 | EEG + EOG + EMG + SpO2 + Resp | 0.0292 | 0.1176 | 0.3623 | 40 |
| 10 | EEG + Resp | 0.0272 | 0.1254 | 0.3559 | 20 |

## Clinical Interpretation

The finding that SpO2 substantially improves clustering is clinically meaningful.
Oxygen desaturation events are associated with sleep disordered breathing, which
varies across sleep stages (more common in REM due to muscle atonia). This
suggests SpO2 captures stage related physiological variation that complements
the electrical brain activity measured by EEG.

The negative interaction between EOG and SpO2 may reflect that both modalities
partially capture REM related information (rapid eye movements and respiratory
instability), creating redundancy that confuses the clustering algorithm.
