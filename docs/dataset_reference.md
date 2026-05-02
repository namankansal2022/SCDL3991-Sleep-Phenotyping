# MESA Sleep Dataset — Reference Document

**Project:** SCDL3991 Sleep Phenotyping
**Last updated:** May 2026
**Status:** Draft — to be refined once data is unzipped and explored

---

## 1. Overview

The Multi-Ethnic Study of Atherosclerosis (MESA) is an NHLBI-sponsored longitudinal cohort study of 6,814 men and women (initially aged 45–84) from black, white, Hispanic, and Chinese-American backgrounds, recruited in 2000–2002 across six US sites.

The MESA Sleep Ancillary Study (2010–2012) enrolled a subset of 2,237 participants for full overnight unattended polysomnography, 7-day wrist actigraphy, and a sleep questionnaire. Raw polysomnography data are available for **2,056 subjects**.

This project uses the polysomnography component, distributed via the National Sleep Research Resource (NSRR) at https://sleepdata.org/datasets/mesa.

## 2. Data Source

- **Distributed by:** National Sleep Research Resource (NSRR), https://sleepdata.org/datasets/mesa
- **Provided locally by:** Dr Zhiya Wang's group, USYD (compressed dataset, ~257 GB)
- **Storage location:** External SSD at /Volumes/SSD/SCDL3991-data/raw/mesa/
- **Compressed size:** ~257 GB (mesa.zip)
- **Expected uncompressed size:** ~400–500 GB

## 3. Signal Modalities and Sampling Rates

Each subject's recording contains seven physiological channels:

| Modality | Channel(s) | Sampling Rate | Purpose |
|----------|------------|---------------|---------|
| EEG | C4-M1 | 256 Hz | Primary sleep staging |
| EOG | Left | 256 Hz | Eye movement (REM, SEM) |
| EMG | Chin | 256 Hz | Muscle tone (REM atonia) |
| ECG | Single lead | 256 Hz | HRV, cardiac rhythm |
| SpO2 | Pulse oximeter | 1 Hz | Oxygen saturation |
| Respiration (Thorax) | Belt | 32 Hz | Respiratory effort |
| Respiration (Abdomen) | Belt | 32 Hz | Respiratory effort |

Sampling rates confirmed via Stein et al. 2024 (medRxiv) and NSRR documentation.

## 4. File Formats

Each subject contributes three files:

- **`.edf`** — European Data Format signal file, exported from Compumedics Profusion
- **`.xml` (Profusion)** — Annotation file in Compumedics Profusion format
- **`.xml` (NSRR)** — Annotation file processed via the EDF Editor and Translator tool

We will use the **NSRR XML** format for consistency with the broader NSRR ecosystem and EDF Viewer compatibility.

## 5. Annotations

The XML annotation files contain:

- **Sleep stages** — 30-second epoch labels (W, N1, N2, N3, REM) following AASM convention
- **Apnea/hypopnea events** — with two distinct hypopnea tags:
  - `Hypopnea` — airflow reduction 30–50% from baseline
  - `Unsure` — airflow reduction >50% from baseline
- **Arousal events** — cortical arousals during sleep
- **Limb movements** — periodic and isolated leg movements

## 6. Pre-Computed Variables on NSRR

Beyond raw signals, NSRR provides subject-level summary variables organised into:

- Administrative
- Apnea-Hypopnea Indices (AHI)
- Arousals
- Heart Rate
- Limb Movements
- Oxygen Saturation
- Quantitative Respiratory Analysis
- Respiratory Event Counts
- Respiratory Event Lengths
- Signal Quality
- Sleep Architecture

These can serve as auxiliary features or as additional ground-truth labels for evaluation.

## 7. Cohort Characteristics

- **Sample:** 2,056 subjects with valid PSG recordings
- **Age range:** Older adults (originally 45–84 at MESA baseline; older at sleep exam in 2010–2012)
- **Demographics:** Multi-ethnic — Black, White, Hispanic, Chinese-American
- **Clinical context:** Cardiovascular cohort (parent study focused on subclinical atherosclerosis)
- **Recording type:** Unattended at-home polysomnography

This is **not a healthy young cohort**. Subjects are older and clinically diverse, with a range of cardiovascular and sleep-disordered breathing comorbidities. This is a strength for clinical relevance but means we should expect signal artefacts and missing/poor-quality channels in some recordings.

## 8. Power Line Frequency

MESA was collected in the United States — power line frequency is **60 Hz**, not 50 Hz.

This is critical for preprocessing: the notch filter in our pipeline must target 60 Hz to remove powerline noise from EEG/EOG/EMG/ECG channels. The Ma et al. (2026) paper used 50 Hz notch (their data was from China and the Netherlands); we cannot copy that setting directly.

## 9. Required Citations

When publishing or submitting work using this dataset, the following citations are required:

1. **NSRR data commons paper:**
   Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D, Redline S. (2018). The National Sleep Research Resource: towards a sleep data commons. *J Am Med Inform Assoc*, 25(10):1351–1358. doi:10.1093/jamia/ocy064

2. **MESA Sleep cohort paper:**
   Chen X, Wang R, Zee P, Lutsey PL, Javaheri S, Alcántara C, Jackson CL, Williams MA, Redline S. (2015). Racial/Ethnic Differences in Sleep Disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA). *Sleep*, 38(6):877–88.

## 10. Open Questions (to resolve once data is browsed)

- [ ] Exact directory structure of mesa.zip when uncompressed
- [ ] File naming convention for subject IDs (zero-padding, prefix)
- [ ] Whether all 2,056 subjects have all seven modalities, or if some recordings have missing channels
- [ ] Presence and format of subject-level metadata files (demographics, comorbidities)
- [ ] Whether actigraphy data is included in this distribution or only PSG
- [ ] Whether the bundled XML annotations are Profusion-format, NSRR-format, or both
