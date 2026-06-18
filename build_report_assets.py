#!/usr/bin/env python3
"""
build_report_assets.py
======================
ONE script to produce every real asset the report needs:
  (1) Pooled-across-5-folds per-stage confusion matrix (semi-supervised)
  (2) PCA embedding scatter: expert labels vs discovered clusters
  (3) Saves the real confusion matrix to results/per_stage_cm_pooled.npy
Then call generate_report_figures.py for the summary bar/line charts.

Run:
    conda activate scdl3991-mesa
    cd ~/Documents/SCDL3991-Sleep-Phenotyping
    python build_report_assets.py
"""
import os, numpy as np, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from src.preprocessing import get_subject_ids, get_xml_path, load_aasm_annotations, expand_stages_to_epochs

rcParams.update({'font.family':'serif','font.size':11,'savefig.dpi':300,
                 'axes.spines.top':False,'axes.spines.right':False})
OUT='figures/report'; os.makedirs(OUT, exist_ok=True)
os.makedirs('results', exist_ok=True)

# ---------- load rich features + rebuild subject ids ----------
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

def ctx_per_subject(X,subj,c):
    out=np.zeros((len(X),X.shape[1]*2)); h=c//2
    for s in np.unique(subj):
        idx=np.where(subj==s)[0]; Xs=X[idx]
        for k,i in enumerate(idx):
            a=max(0,k-h); b=min(len(idx),k+h+1)
            out[i,:X.shape[1]]=X[i]; out[i,X.shape[1]:]=Xs[a:b].mean(0)
    return out

Xn=(X_rich-X_rich.mean(0))/(X_rich.std(0)+1e-10)
Xc=ctx_per_subject(Xn,subj,5); Xc=(Xc-Xc.mean(0))/(Xc.std(0)+1e-10)

stages=['W','N1','N2','N3','REM']
le=LabelEncoder(); le.fit(stages); yenc=le.transform(y_all)

# ===================================================================
# (1) POOLED PER-STAGE CONFUSION MATRIX (semi-supervised, 10% labels)
# ===================================================================
print('Computing pooled per-stage confusion matrix (5 folds)...')
uniq=np.unique(subj); np.random.seed(0); shuf=np.random.permutation(uniq)
folds=np.array_split(shuf,5)
all_t=[]; all_p=[]
for fi in range(5):
    ev_s=folds[fi]; tr_s=np.concatenate([folds[j] for j in range(5) if j!=fi])
    ev=np.where(np.isin(subj,ev_s))[0]; tr=np.where(np.isin(subj,tr_s))[0]
    np.random.seed(fi)
    if len(tr)>8000: tr=np.random.choice(tr,8000,replace=False)
    if len(ev)>2000: ev=np.random.choice(ev,2000,replace=False)
    ai=np.concatenate([tr,ev]); Xf=Xc[ai]; yf=yenc[ai]
    istr=np.array([True]*len(tr)+[False]*len(ev))
    Xp=PCA(n_components=15).fit_transform(Xf)
    yp=np.full(len(Xf),-1); tp=np.where(istr)[0]
    np.random.seed(fi); lp=np.random.choice(tp,int(0.10*len(tp)),replace=False)
    yp[lp]=yf[lp]
    ls=LabelSpreading(kernel='knn',n_neighbors=20,max_iter=40); ls.fit(Xp,yp)
    pred=ls.transduction_
    all_t.extend(yf[~istr]); all_p.extend(pred[~istr])
all_t=np.array(all_t); all_p=np.array(all_p)
cm=confusion_matrix(all_t,all_p,labels=range(5))
np.save('results/per_stage_cm_pooled.npy', cm)

prec=precision_score(all_t,all_p,labels=range(5),average=None,zero_division=0)
rec =recall_score(all_t,all_p,labels=range(5),average=None,zero_division=0)
f1  =f1_score(all_t,all_p,labels=range(5),average=None,zero_division=0)
print('\nPooled per-stage (semi-supervised 10%):')
print(f"{'Stage':<6}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Support':>9}")
for i,s in enumerate(stages):
    print(f"{s:<6}{prec[i]:>8.3f}{rec[i]:>8.3f}{f1[i]:>8.3f}{int((all_t==i).sum()):>9}")

# Figure: pooled confusion matrix
cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(5.6,5.0))
im=ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(stages); ax.set_yticklabels(stages)
ax.set_xlabel('Predicted stage'); ax.set_ylabel('Expert (AASM) stage')
ax.set_title('Per-stage recovery (semi-supervised, 10% labels,\npooled over 5 subject-level folds)')
for i in range(5):
    for j in range(5):
        ax.text(j,i,f'{cmn[i,j]:.2f}',ha='center',va='center',
                color='white' if cmn[i,j]>0.5 else 'black',fontsize=9)
cb=fig.colorbar(im,fraction=0.046,pad=0.04); cb.set_label('Proportion of true epochs')
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_per_stage_confusion.png',bbox_inches='tight')
plt.close(fig); print('  saved fig4_per_stage_confusion.png')

# ===================================================================
# (2) PCA EMBEDDING: expert labels vs discovered clusters
# ===================================================================
print('\nBuilding PCA embedding scatter...')
np.random.seed(42); idx=np.random.choice(len(y_all),8000,replace=False)
Xs=Xc[idx]; ys=yenc[idx]
P=PCA(n_components=2).fit_transform(Xs)
# discovered clusters via DBSCAN on PCA(15)+ctx (our champion)
Xp20=PCA(n_components=20).fit_transform(Xs)
db=DBSCAN(eps=2.0,min_samples=10).fit_predict(Xp20)

fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.6))
cmap=plt.cm.tab10
for k,s in enumerate(stages):
    m=ys==k
    a1.scatter(P[m,0],P[m,1],s=4,alpha=0.5,color=cmap(k),label=s)
a1.set_title('(a) Expert AASM labels'); a1.set_xlabel('PC1'); a1.set_ylabel('PC2')
a1.legend(markerscale=3,frameon=False,loc='best')
# discovered
uc=[c for c in np.unique(db) if c!=-1]
for n,c in enumerate(uc[:10]):
    m=db==c
    a2.scatter(P[m,0],P[m,1],s=4,alpha=0.5,color=cmap(n%10))
m=db==-1
a2.scatter(P[m,0],P[m,1],s=3,alpha=0.2,color='lightgrey',label='noise')
a2.set_title('(b) Discovered clusters (DBSCAN)'); a2.set_xlabel('PC1'); a2.set_ylabel('PC2')
fig.tight_layout(); fig.savefig(f'{OUT}/fig6_pca_embedding.png',bbox_inches='tight')
plt.close(fig); print('  saved fig6_pca_embedding.png')

print('\nAll real assets built. Now run: python generate_report_figures.py')
