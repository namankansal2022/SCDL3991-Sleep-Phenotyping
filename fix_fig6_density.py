#!/usr/bin/env python3
"""
fix_fig6_density.py
===================
Regenerates fig6 as a cleaner density-contour plot instead of an
overlapping scatter. Shows the SAME truth (stages overlap in 2D) but
far more legibly. Run after build_report_assets.py.

    conda activate scdl3991-mesa
    cd ~/Documents/SCDL3991-Sleep-Phenotyping
    python fix_fig6_density.py
"""
import os, numpy as np, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from src.preprocessing import get_subject_ids, get_xml_path, load_aasm_annotations, expand_stages_to_epochs

rcParams.update({'font.family':'serif','font.size':11,'savefig.dpi':300,
                 'axes.spines.top':False,'axes.spines.right':False})
OUT='figures/report'; os.makedirs(OUT,exist_ok=True)

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
stages=['W','N1','N2','N3','REM']
le=LabelEncoder(); le.fit(stages); yenc=le.transform(y_all)

np.random.seed(42); idx=np.random.choice(len(y_all),6000,replace=False)
Xs=Xc[idx]; ys=yenc[idx]
P=PCA(n_components=2).fit_transform(Xs)

# shared axis limits (trim 1-99 pct to avoid outlier stretch)
xlo,xhi=np.percentile(P[:,0],[1,99]); ylo,yhi=np.percentile(P[:,1],[1,99])
xx,yy=np.mgrid[xlo:xhi:120j, ylo:yhi:120j]
grid=np.vstack([xx.ravel(),yy.ravel()])
cmap=plt.cm.tab10

fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.6))

# (a) expert labels as density contours, one colour per stage
for k,s in enumerate(stages):
    pts=P[ys==k]
    if len(pts)<20: continue
    try:
        kde=gaussian_kde(pts.T)
        dens=kde(grid).reshape(xx.shape)
        a1.contour(xx,yy,dens,levels=4,colors=[cmap(k)],linewidths=1.2,alpha=0.9)
        a1.plot([],[],color=cmap(k),label=s)  # legend proxy
    except Exception:
        pass
a1.set_xlim(xlo,xhi); a1.set_ylim(ylo,yhi)
a1.set_title('(a) Expert AASM stages (density)'); a1.set_xlabel('PC1'); a1.set_ylabel('PC2')
a1.legend(frameon=False,loc='best')

# (b) discovered clusters via DBSCAN, filled density
Xp20=PCA(n_components=20).fit_transform(Xs)
db=DBSCAN(eps=2.0,min_samples=10).fit_predict(Xp20)
uc=[c for c in np.unique(db) if c!=-1]
# order clusters by size, show top 5
sizes=sorted(uc,key=lambda c:-(db==c).sum())[:5]
for n,c in enumerate(sizes):
    pts=P[db==c]
    if len(pts)<20: continue
    try:
        kde=gaussian_kde(pts.T)
        dens=kde(grid).reshape(xx.shape)
        a2.contourf(xx,yy,dens,levels=4,colors=[cmap(n)],alpha=0.25)
        a2.contour(xx,yy,dens,levels=4,colors=[cmap(n)],linewidths=0.8,alpha=0.8)
    except Exception:
        pass
a2.set_xlim(xlo,xhi); a2.set_ylim(ylo,yhi)
a2.set_title('(b) Discovered clusters (DBSCAN, top 5)'); a2.set_xlabel('PC1'); a2.set_ylabel('PC2')

fig.tight_layout(); fig.savefig(f'{OUT}/fig6_pca_embedding.png',bbox_inches='tight')
plt.close(fig)
print('Regenerated fig6_pca_embedding.png as density-contour version')
