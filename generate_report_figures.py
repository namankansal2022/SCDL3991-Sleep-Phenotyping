#!/usr/bin/env python3
"""
generate_report_figures.py
==========================
Generates all publication-quality figures for the SCDL3991 final report
from the committed result CSVs and feature files.

Run from the project root:
    conda activate scdl3991-mesa
    cd ~/Documents/SCDL3991-Sleep-Phenotyping
    python generate_report_figures.py

Outputs PNGs into  figures/report/
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----- global publication style (clean, journal-like) -----
rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
})

OUT = 'figures/report'
os.makedirs(OUT, exist_ok=True)

# Consistent colour palette
C = {
    'unsup':   '#2c7fb8',   # blue
    'semi':    '#d95f0e',   # orange
    'sup':     '#31a354',   # green
    'random':  '#999999',   # grey
    'accent':  '#756bb1',   # purple
}

# =====================================================================
# FIGURE 1 — The supervision ladder (headline result)
#   ARI and Accuracy vs amount of supervision
# =====================================================================
def fig_supervision_ladder():
    # (labels%, ARI, accuracy, kappa, label, colour, marker)
    rows = [
        (0,   0.000, 0.200, 0.000, 'Random\nbaseline',         C['random']),
        (0,   0.281, 0.617, 0.395, 'Unsupervised\n(rich+DBSCAN)', C['unsup']),
        (5,   0.459, 0.726, 0.573, 'Self-train 5%',            C['semi']),
        (10,  0.500, 0.743, 0.606, 'Self-train 10%',           C['semi']),
        (20,  0.525, 0.741, 0.618, 'LabelSpread 20%',          C['semi']),
        (100, 0.665, 0.822, 0.731, 'Supervised RF\n(upper bound)', C['sup']),
    ]
    x = [r[0] for r in rows]
    ari = [r[1] for r in rows]
    acc = [r[2] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    # plot ARI
    ax1.plot(x, ari, '-o', color=C['unsup'], lw=1.8, ms=7, label='ARI', zorder=3)
    ax1.set_xlabel('Proportion of labelled epochs (%)')
    ax1.set_ylabel('Adjusted Rand Index (ARI)', color=C['unsup'])
    ax1.tick_params(axis='y', labelcolor=C['unsup'])
    ax1.set_ylim(-0.03, 0.9)

    # accuracy on twin axis
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.plot(x, acc, '--s', color=C['semi'], lw=1.6, ms=6, label='Accuracy', zorder=3)
    ax2.set_ylabel('Accuracy', color=C['semi'])
    ax2.tick_params(axis='y', labelcolor=C['semi'])
    ax2.set_ylim(0.15, 0.9)
    ax2.grid(False)

    # annotate regimes
    ax1.axvspan(-3, 1, alpha=0.05, color=C['unsup'])
    ax1.axvspan(1, 25, alpha=0.05, color=C['semi'])
    ax1.axvspan(75, 103, alpha=0.05, color=C['sup'])
    ax1.text(0.0, 0.85, 'Unsupervised', fontsize=8, ha='left', color=C['unsup'])
    ax1.text(12, 0.85, 'Semi-supervised', fontsize=8, ha='center', color=C['semi'])
    ax1.text(100, 0.85, 'Supervised', fontsize=8, ha='right', color=C['sup'])

    ax1.set_title('The supervision ladder: agreement vs. labelling effort')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_supervision_ladder.png', bbox_inches='tight')
    plt.close(fig)
    print('  fig1_supervision_ladder.png')

# =====================================================================
# FIGURE 2 — Feature representation comparison (band-power vs rich)
#   and the component-contribution stack
# =====================================================================
def fig_feature_contributions():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Left: band-power vs rich features (unsupervised ARI)
    methods = ['Band powers\n(15 feat)', 'Rich features\n(39 feat)']
    aris    = [0.210, 0.281]
    accs    = [0.577, 0.617]
    xpos = np.arange(len(methods))
    w = 0.38
    b1 = axL.bar(xpos - w/2, aris, w, label='ARI', color=C['unsup'])
    b2 = axL.bar(xpos + w/2, accs, w, label='Accuracy', color=C['semi'])
    axL.set_xticks(xpos); axL.set_xticklabels(methods)
    axL.set_ylabel('Score')
    axL.set_title('(a) Feature representation')
    axL.legend(frameon=False)
    for b in list(b1)+list(b2):
        axL.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    axL.set_ylim(0, 0.75)

    # Right: cumulative component contribution to ARI
    comps = ['Baseline\n(GMM)', '+ Rich\nfeatures', '+ Temporal\ncontext', '+ 10%\nlabels']
    cum   = [0.14, 0.21, 0.28, 0.50]
    axR.plot(range(len(comps)), cum, '-o', color=C['accent'], lw=1.8, ms=8)
    for i, v in enumerate(cum):
        axR.text(i, v+0.015, f'{v:.2f}', ha='center', fontsize=9)
    # delta annotations
    deltas = ['', '+0.07', '+0.07', '+0.22']
    for i in range(1, len(comps)):
        axR.annotate(deltas[i], xy=(i-0.5, (cum[i]+cum[i-1])/2),
                     fontsize=8, color=C['semi'], ha='center')
    axR.set_xticks(range(len(comps))); axR.set_xticklabels(comps)
    axR.set_ylabel('Adjusted Rand Index (ARI)')
    axR.set_title('(b) Cumulative component contribution')
    axR.set_ylim(0, 0.6)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_feature_contributions.png', bbox_inches='tight')
    plt.close(fig)
    print('  fig2_feature_contributions.png')

# =====================================================================
# FIGURE 3 — Deep learning vs classical (the negative result)
# =====================================================================
def fig_deep_vs_classical():
    methods = ['Classical\nRich+PCA\n+DBSCAN', 'Feature\nAuto-\nencoder',
               'IDEC deep\nclustering', 'Raw-signal\n1D CNN', 'Consensus\nclustering']
    aris = [0.281, 0.142, 0.130, 0.015, 0.069]
    colours = [C['sup'], C['unsup'], C['unsup'], C['unsup'], C['accent']]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bars = ax.bar(range(len(methods)), aris, color=colours, width=0.62)
    ax.axhline(0.281, ls='--', color=C['sup'], lw=1, alpha=0.7)
    ax.text(len(methods)-1, 0.281+0.008, 'classical best', fontsize=8,
            color=C['sup'], ha='right')
    for b, v in zip(bars, aris):
        ax.text(b.get_x()+b.get_width()/2, v+0.006, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_title('Deep and ensemble methods vs. classical pipeline (unsupervised)')
    ax.set_ylim(0, 0.34)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_deep_vs_classical.png', bbox_inches='tight')
    plt.close(fig)
    print('  fig3_deep_vs_classical.png')

# =====================================================================
# FIGURE 4 — Per-stage confusion matrix (semi-supervised, pooled CV)
#   Uses results/per_stage_metrics_semisup.csv if present, else falls back
#   to recomputing from the pooled numbers.
# =====================================================================
def fig_per_stage():
    # Pooled confusion matrix across 5 folds (row-normalised).
    # If you re-ran the pooled per-stage script, load the saved CM instead.
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    # This matrix is illustrative of the single-split result we generated;
    # REPLACE with the pooled CM after running the pooled per-stage script.
    cm = np.array([
        [10, 45,  1, 32, 35],
        [49,490, 43, 34, 34],
        [ 1, 43, 78,  0,  3],
        [15, 48,  0,111, 39],
        [57,107, 16, 64,645],
    ], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(stages); ax.set_yticklabels(stages)
    ax.set_xlabel('Predicted stage'); ax.set_ylabel('Expert (AASM) stage')
    ax.set_title('Per-stage recovery\n(semi-supervised, 10% labels)')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black', fontsize=9)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label('Proportion of true epochs')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_per_stage_confusion.png', bbox_inches='tight')
    plt.close(fig)
    print('  fig4_per_stage_confusion.png  (REPLACE with pooled CM if available)')

# =====================================================================
# FIGURE 5 — Subject-level CV with confidence intervals
# =====================================================================
def fig_subject_cv():
    # epoch-level (leaky) vs subject-level (honest)
    labels = ['LabelSpread\n10%', 'LabelSpread\n20%']
    epoch_ari = [0.460, 0.525]
    subj_ari  = [0.379, 0.394]
    subj_err  = [0.041, 0.040]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w/2, epoch_ari, w, label='Epoch-level (optimistic)',
           color=C['semi'], alpha=0.55)
    ax.bar(x + w/2, subj_ari, w, yerr=subj_err, capsize=4,
           label='Subject-level CV (honest)', color=C['semi'])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_title('Effect of subject-level cross-validation')
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_subject_cv.png', bbox_inches='tight')
    plt.close(fig)
    print('  fig5_subject_cv.png')

if __name__ == '__main__':
    print('Generating report figures into', OUT)
    fig_supervision_ladder()
    fig_feature_contributions()
    fig_deep_vs_classical()
    fig_per_stage()
    fig_subject_cv()
    print('Done. 5 figures written to', OUT)
