#!/usr/bin/env python3
"""79 — HEADLINE figure: the transfer asymmetry. POOLED out-of-fold LOBO — catastrophe
OCCURRENCE prediction transfers (AUC 0.85-0.98) while error MAGNITUDE prediction does not
(Spearman 0.56-0.65), for all 5 arms. Honest, robust (pooled, not noisy per-benchmark)."""
import pickle, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (BAR_A, BAR_B, DPI, FS_LABEL, FS_TITLE, FS_LEGEND,
                          apply_rc)
apply_rc()

OUT=Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'; FIG=OUT/'figures'
I=pickle.load(open(CACHE/'v8_indist.pkl','rb')); arms=I['arms']
def auc(p,y):
    m=~np.isnan(y); p,y=p[m],y[m]; pos=y.sum(); neg=len(y)-pos
    if pos==0 or neg==0: return np.nan
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    return (r[y==1].sum()-pos*(pos+1)/2)/(pos*neg)

AUCs=[]; RHOs=[]; labels=[]
for arm in arms:
    j=arms.index(arm)
    pe=np.concatenate([c['preds'][j] for c in I['caches'].values()])
    pc=np.concatenate([c['pcat10'][j] for c in I['caches'].values()])
    te=np.concatenate([c['true_err'][j] for c in I['caches'].values()])
    m=~np.isnan(te)
    AUCs.append(auc(pc,(te>10).astype(float))); RHOs.append(spearmanr(pe[m],te[m]).correlation)
    labels.append(arm.replace('PBE-D3BJ_','PBE-').replace('CREST_xTB_DFT','CREST-DFT').replace('CREST_xTB','CREST'))

fig,ax=plt.subplots(figsize=(10,6))
x=np.arange(len(arms))
ax.bar(x-0.2,AUCs,0.4,color=BAR_A,edgecolor='black',linewidth=0.6,label='catastrophe OCCURRENCE prediction  (AUC, |err|>10 kcal/mol)')
ax.bar(x+0.2,RHOs,0.4,color=BAR_B,edgecolor='black',linewidth=0.6,label='error MAGNITUDE prediction  (Spearman ρ)')
ax.axhline(0.5,color='gray',ls=':',lw=1)
for i,(a,r) in enumerate(zip(AUCs,RHOs)):
    ax.text(i-0.2,a+0.01,f'{a:.2f}',ha='center',fontsize=9,fontweight='bold')
    ax.text(i+0.2,r+0.01,f'{r:.2f}',ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=FS_LABEL)
ax.set_ylabel('pooled out-of-fold LOBO transfer quality',fontsize=FS_LABEL); ax.set_ylim(0,1.05)
ax.set_xlabel('fidelity arm',fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND,loc='upper right',framealpha=0.85,edgecolor='gray',fancybox=False)
ax.set_title('The transfer asymmetry (paper LEAD): across chemical families, catastrophe OCCURRENCE is\npredictable (AUC 0.85–0.98) but error MAGNITUDE is not (ρ 0.56–0.65) — this gap enables safe cost-aware routing',
            fontsize=FS_TITLE,fontweight='bold')
ax.grid(axis='y',alpha=0.25,linewidth=0.5)
plt.tight_layout(); out=FIG/'transfer_asymmetry.png'; plt.savefig(out,dpi=DPI,bbox_inches='tight')
print(f"Saved -> {out}")
for l,a,r in zip(labels,AUCs,RHOs): print(f"  {l:<12} occurrence-AUC {a:.3f}  magnitude-ρ {r:.3f}")
