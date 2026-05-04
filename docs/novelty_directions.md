# Methodological Novelty — Candidate Directions

**Project:** SCDL3991 Sleep Phenotyping
**Phase:** 2 — discussion document for Friday meeting with Jie
**Last updated:** May 2026
**Purpose:** to support a discussion of possible methodological contributions, not a finalised proposal

---

## Context

Jie's feedback on Progress Report 2 raised the question of methodological novelty: *"At this stage, you are still mainly applying existing clustering algorithms to physiological data. That is a good start, but I would also like to see you think about possible improvements to existing methods, with some methodological novelty."*

This document outlines four candidate directions for a methodological contribution, with brief notes on what each would involve, what makes it novel, and the relative tradeoffs. The intention is to discuss these with Jie at the Friday meeting and converge on a direction together.

All four directions assume the Phase 2 baseline (six clustering algorithms applied to ECG/HRV features on MESA, with internal and external evaluation) is already complete.

---

## Direction 1 — Multi-modal fusion clustering

**Idea.** Extend the analysis from ECG-only to a fusion of multiple modalities (ECG, SpO2, respiration), and develop a clustering approach that handles heterogeneous feature types coming from different physiological signals.

**Why this is novel.** Most existing sleep clustering work uses a single modality, typically EEG (e.g. Ma et al. 2026, Decat et al. 2022). Multi-modal fusion is genuinely under-explored despite being clinically motivated — consumer wearables can measure ECG, SpO2, and respiration but rarely EEG. A clustering method that recovers sleep structure from these "wearable-compatible" modalities would have real translational value.

**What it would look like.**
- Compute features per modality independently (HRV from ECG, ODI/desaturation features from SpO2, respiratory rate/variability from thorax/abdomen)
- Concatenate into a multi-modal feature vector per epoch
- Compare three fusion strategies: (a) naive concatenation, (b) per-modality normalisation then concatenation, (c) learned fusion weights (e.g. weighted distance metric)
- Evaluate against AASM ground truth, compare to single-modality baselines

**Tradeoffs.**
- Pro: directly extends the Ma et al. framework to a space they explicitly didn't cover; clinical relevance; aligns with Jie's email naming ECG/SpO2/respiration specifically
- Pro: requires building all the per-modality preprocessing pipelines anyway, which is useful infrastructure
- Con: not a new clustering algorithm per se — more a new application/integration; reviewers may consider this "engineering" rather than "methodological"
- Effort: medium. Bulk of work is preprocessing pipelines for additional modalities; the fusion step itself is conceptually straightforward.

---

## Direction 2 — Semi-supervised clustering with partial AASM constraints

**Idea.** Most sleep clustering is fully unsupervised. But MESA has rich ground-truth labels (per-epoch AASM stages from expert scoring). A semi-supervised approach uses a small fraction of labels — e.g. 5% of epochs — as constraints during clustering, and uses the rest of the data unsupervised. This explicitly models the realistic clinical scenario of "limited labelled data, abundant unlabelled data".

**Why this is novel.** Most clustering methods are either fully supervised or fully unsupervised. Constrained clustering (must-link / cannot-link constraints) and seeded clustering exist in the general ML literature but have been minimally applied to sleep data. Bridging the unsupervised-supervised gap with a small label budget is methodologically interesting.

**What it would look like.**
- Sample a small fraction (5%, 10%, 25%) of epochs and use their AASM labels
- Implement constrained K-Means or seeded K-Means with these labelled epochs as anchors
- Evaluate how performance scales with label fraction
- Compare against fully unsupervised (Phase 1 baseline) and fully supervised (e.g. simple classifier trained on the same fraction)

**Tradeoffs.**
- Pro: methodologically distinct from existing sleep clustering work
- Pro: connects to a real clinical question — how much label budget is needed
- Pro: links naturally to the project's stated framing of "low-label clinical settings" (from your repo README)
- Con: requires more careful experimental design (multiple label fractions, multiple random seeds for sampling) — more compute and more figures
- Effort: medium. Seeded K-Means and constrained K-Means have established implementations; main work is the experimental design.

---

## Direction 3 — Adaptive Density Peak Clustering for sleep data

**Idea.** Density Peak Clustering (Rodriguez & Laio 2014, used in Phase 1) requires manually tuning a cutoff distance dc, which is sensitive and dataset-dependent. An adaptive variant that selects dc automatically based on local density structure could improve robustness and reduce the need for manual tuning. We would propose, implement, and validate such a variant on sleep data.

**Why this is novel.** Algorithmic improvements to existing clustering methods are unambiguously "methodological" novelty. The dc tuning problem is well-known in the DPC literature and has multiple proposed solutions, but none specifically validated on sleep physiological data. There is room for a tailored variant.

**What it would look like.**
- Review existing adaptive-dc proposals in DPC literature (a few exist — e.g. based on k-nearest-neighbour distances, entropy criteria)
- Propose a variant suited to physiological data (perhaps using HRV-specific distance properties)
- Compare against fixed-dc DPC and the other five baselines
- Show robustness (less sensitive to the dc parameter) and at least comparable clustering quality

**Tradeoffs.**
- Pro: clearest "methodological novelty" framing — improving an algorithm
- Pro: connects directly to one of the algorithms already in our Phase 1 baseline
- Con: narrower contribution — improves one algorithm rather than addressing a broader problem
- Con: risk of being out-scoped by existing DPC variants in the broader ML literature; need to justify what's new about ours
- Effort: medium-high. Requires more theoretical reading on existing DPC variants, and implementation from scratch.

---

## Direction 4 — Representation learning via autoencoder for HRV time-series

**Idea.** Instead of computing handcrafted HRV features per epoch, train an autoencoder to learn a latent representation of each 30-second ECG epoch directly. Cluster in the learned latent space rather than in the handcrafted feature space. Compare to the handcrafted-feature baseline.

**Why this is novel.** Most sleep clustering uses handcrafted features (Ma et al. 2026, Decat et al. 2022). Deep representation learning is mainstream in supervised sleep staging but rarely combined with unsupervised clustering. This is a "deep clustering" approach in the spirit of the broader ML literature (e.g. DEC, IDEC) but applied specifically to physiological sleep data.

**What it would look like.**
- 1D-CNN or LSTM autoencoder trained on 30-second ECG epochs (input: 30s × 256Hz = 7680 samples)
- Latent dimension: ~16-32
- Cluster latent representations using the same six algorithms from Phase 1
- Compare to handcrafted-HRV baseline on the same evaluation metrics

**Tradeoffs.**
- Pro: most "modern ML" approach; aligns with where the broader sleep ML literature is heading
- Pro: connects to your stated research direction in the repo README ("representation learning to identify interpretable latent structure")
- Con: significantly more implementation effort — need to design and train the autoencoder
- Con: less interpretable than handcrafted features (Jie may value interpretability, given his framing)
- Con: requires more compute time, which we don't have a lot of given the timeline
- Effort: high. Autoencoder design, training, hyperparameter selection, all add up.

---

## Comparison summary

| Direction | Novelty type | Effort | Risk | Strategic fit |
|-----------|-------------|--------|------|---------------|
| 1. Multi-modal fusion | Application/integration | Medium | Low | High — aligns with Jie's named modalities |
| 2. Semi-supervised | Method | Medium | Low | High — aligns with project framing |
| 3. Adaptive DPC | Algorithm | Medium-high | Medium | Medium — narrow but cleanly methodological |
| 4. Deep representation | Method | High | Medium-high | Medium — modern but heavier |

## Recommended path for discussion

If we had to pick one going into the meeting, my current preference would be **Direction 1 (multi-modal fusion)** for these reasons:

- It directly extends Ma et al.'s framework to modalities they explicitly omitted, which is a defensible "novel contribution" framing
- It aligns precisely with the modalities Jie named in his preparation email (ECG, SpO2, respiration)
- The infrastructure work (per-modality preprocessing pipelines) is useful regardless of which direction we pick — so it is not wasted effort if we pivot
- Of all four options, it has the clearest path from current Phase 2 baseline to the contribution

A sensible secondary option would be **Direction 2 (semi-supervised)**, which is methodologically distinct and could be combined with Direction 1 if we wanted a more ambitious contribution.

Direction 3 (adaptive DPC) is technically interesting but narrower in impact.

Direction 4 (deep representation learning) is the most ambitious but carries the most timeline and interpretability risk.

These are starting points for discussion, not commitments — happy to be redirected.
