#!/usr/bin/env python3
"""
make_kdistance_rich.py
======================
Generates a k-distance graph on the ACTUAL rich-feature space (39 spectral
features + temporal context + PCA), so the epsilon = 2.0 claim in the report
is backed by the matching representation rather than the old band-power graph.

Run:
    conda activate scdl3991-mesa
    cd ~/Documents/SCDL3991-Sleep-Phenotyping
    python make_kdistance_rich.py

Writes: figures/report/fig_kdistance_rich.png
Prints the distance value at the elbow so you can confirm it sits near 2.0.
"""
import os, numpy as np, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from src.preprocessing import get_subject_ids, get_xml_path, load_aasm_annotations, expand_stages_to_epochs

rcParams.update({'font.family':'serif','font.size':11,'savefig.dpi':300,
                 'axes.spines.top':False,'axes.spines.right':False})
OUT='figures/report'; os.makedirs(OUT, exist_ok=True)

# ---- load rich features + rebuild subject ids (same recipe as everywhere) ----
eeg=np.load('results/mesa_eeg_rich_features.npz',allow_pickle=True)
X_rich=eeg['X']; y_all=eeg['y']
for j in range(X_rich.shape[1]):
    c=X_rich[:,j]; m=np.isnan(c)
    if m.any(): c[m]=np.nanmedian(c)

ids=get_subject_ids(); np.random.seed(42)
selected=np.random.choice(ids,size=100,replace=False)
slen=[]
for sid in selected:
    try:
        ev=load_aasm_annotations(get_xml_path(sid)); _,lab=expand_stages_to_epochs(ev); slen.append(len(lab))
    except: slen.append(0)
subj=np.concatenate([[i]*l for i,l in enumerate(slen)]); subj=subj[:len(X_rich)]

def ctx(X,subj,c):
    out=np.zeros((len(X),X.shape[1]*2)); h=c//2
    for s in np.unique(subj):
        idx=np.where(subj==s)[0]; Xs=X[idx]
        for k,i in enumerate(idx):
            a=max(0,k-h); b=min(len(idx),k+h+1)
            out[i,:X.shape[1]]=X[i]; out[i,X.shape[1]:]=Xs[a:b].mean(0)
    return out

Xn=(X_rich-X_rich.mean(0))/(X_rich.std(0)+1e-10)
Xc=ctx(Xn,subj,5); Xc=(Xc-Xc.mean(0))/(Xc.std(0)+1e-10)

# same 10k sample + PCA(20) as the champion DBSCAN config
np.random.seed(42); idx=np.random.choice(len(y_all),10000,replace=False)
Xs=Xc[idx]
Xp=PCA(n_components=20).fit_transform(Xs)

# ---- k-distance: distance to the min_samples-th nearest neighbour ----
min_samples=10
nn=NearestNeighbors(n_neighbors=min_samples).fit(Xp)
dist,_=nn.kneighbors(Xp)
kdist=np.sort(dist[:,-1])  # distance to the 10th neighbour, ascending

# crude elbow: point of maximum curvature on the upper tail
x=np.arange(len(kdist))
# normalise then find max distance from the chord (Kneedle-style)
x1,y1=x[0],kdist[0]; x2,y2=x[-1],kdist[-1]
num=np.abs((y2-y1)*x-(x2-x1)*kdist+x2*y1-y2*x1)
den=np.hypot(y2-y1,x2-x1)
elbow_idx=np.argmax(num/den)
elbow_val=kdist[elbow_idx]

fig,ax=plt.subplots(figsize=(6.4,4.2))
ax.plot(x,kdist,color='#2c7fb8',lw=1.6)
ax.axhline(elbow_val,ls='--',color='#d95f0e',lw=1,
           label=f'elbow $\\approx$ {elbow_val:.2f}')
ax.axhline(2.0,ls=':',color='#31a354',lw=1,label=r'$\varepsilon=2.0$ (used)')
ax.set_xlabel('Points sorted by distance')
ax.set_ylabel(f'Distance to {min_samples}th nearest neighbour')
ax.set_title('k-distance graph (rich features + temporal context, PCA-20)')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_kdistance_rich.png',bbox_inches='tight')
plt.close(fig)

print(f'Elbow distance (auto-detected): {elbow_val:.3f}')
print(f'epsilon used in report: 2.0')
print('Interpretation:')
if abs(elbow_val-2.0) < 0.4:
    print('  -> 2.0 is close to the detected elbow. The report claim is well supported.')
elif elbow_val < 2.0:
    print('  -> elbow is somewhat below 2.0; 2.0 sits on the early plateau.')
    print('     Consider softening the wording to "just above the elbow, on the')
    print('     low-density plateau" rather than "in the elbow region".')
else:
    print('  -> elbow is above 2.0; 2.0 sits just below it. Wording is fine.')
print('\nSaved figures/report/fig_kdistance_rich.png')
