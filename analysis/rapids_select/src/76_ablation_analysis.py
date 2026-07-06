#!/usr/bin/env python3
"""76 — Ablation analysis: paired bootstrap CI per group vs full + figure."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (BAR_A, BAR_B, ARM_STEEL, DPI, FS_LABEL, FS_TITLE,
                          FS_SUPTITLE, apply_rc)
apply_rc()

OUT=Path(__file__).resolve().parents[1]; RES=OUT/'results'; FIG=OUT/'figures'
R=json.load(open(RES/'ablation_retrain.json'))
full_i=np.array(R['full']['indist_perbench'])   # [18,3] cost,err,cat
rng=np.random.default_rng(0)
def boot_diff(A,B):  # paired A-B over benchmarks, 95% CI on mean
    D=A-B; n=len(D); o=np.empty((5000,3))
    for b in range(5000): o[b]=D[rng.integers(0,n,n)].mean(0)
    return D.mean(0),np.percentile(o,2.5,0),np.percentile(o,97.5,0)

order=['drop-charged-aware','drop-WALLTIMES','drop-MLIP_STEPS','drop-CHEM','drop-SCAN_ECG',
       'drop-CHARGE','drop-ANCHORS','drop-GEOMETRY','drop-FROZEN','drop-OTHER']
print(f"{'ablation':<22}{'Δerr in-dist (95% CI)':<30}{'Δcat pp':<20}{'charged err':<12}")
rows=[]
for name in order:
    if name not in R: continue
    Ai=np.array(R[name]['indist_perbench']); mu,lo,hi=boot_diff(Ai,full_i)
    ch=np.array(R[name]['charged_mean'])
    sig='*' if (hi[1]<0 or lo[1]>0) else ''
    print(f"{name:<22}{mu[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]{sig:<8}{mu[2]*100:+.2f}pp{'':6}{ch[1]:.3f}")
    rows.append((name.replace('drop-',''),mu[1],lo[1],hi[1],ch[1]))

# figure: in-dist Δerr per dropped group (sorted), with CI
rows_feat=[r for r in rows if r[0]!='charged-aware']
rows_feat.sort(key=lambda r:-r[1])
fig,ax=plt.subplots(1,2,figsize=(15,5.5))
names=[r[0] for r in rows_feat]; mus=[r[1] for r in rows_feat]
los=[r[1]-r[2] for r in rows_feat]; his=[r[3]-r[1] for r in rows_feat]
ax[0].barh(range(len(names)),mus,xerr=[los,his],color=ARM_STEEL,edgecolor='black',linewidth=0.6,capsize=3)
ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names); ax[0].invert_yaxis()
ax[0].axvline(0,color='k',lw=0.8); ax[0].set_xlabel('Δ in-dist MAE vs full RAPIDS-Select (kcal/mol)  [drop-one]',fontsize=FS_LABEL)
ax[0].set_title('Feature-group importance for LOBO routing\n(no single group is critical; signal is redundant)',fontsize=FS_TITLE)
ax[0].grid(axis='x',alpha=0.25,linewidth=0.5)

# charged: charged-aware vs full (the big one)
labels=['full RAPIDS-Select','− charged-aware\ntraining']
ch_full=np.array(R['full']['charged_mean']); ch_dca=np.array(R['drop-charged-aware']['charged_mean'])
x=np.arange(2)
ax[1].bar(x-0.2,[ch_full[1],ch_dca[1]],0.4,label='charged MAE',color=BAR_B,edgecolor='black',linewidth=0.6)
ax[1].bar(x+0.2,[ch_full[2]*100,ch_dca[2]*100],0.4,label='charged catastrophic %',color=BAR_A,edgecolor='black',linewidth=0.6)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].legend(framealpha=0.85,edgecolor='gray',fancybox=False)
ax[1].set_title('Training-data ablation dominates:\ncharged-aware training is the critical factor',fontsize=FS_TITLE)
ax[1].set_ylabel('charged MAE (kcal/mol) / catastrophic %',fontsize=FS_LABEL)
for i,(e,c) in enumerate([(ch_full[1],ch_full[2]*100),(ch_dca[1],ch_dca[2]*100)]):
    ax[1].text(i-0.2,e+0.2,f'{e:.1f}',ha='center',fontweight='bold'); ax[1].text(i+0.2,c+0.5,f'{c:.0f}%',ha='center',fontweight='bold')
ax[1].grid(axis='y',alpha=0.25,linewidth=0.5)
plt.suptitle('RAPIDS-Select ablation: feature groups are redundant; charged-aware TRAINING DATA is what matters',fontweight='bold',fontsize=FS_SUPTITLE)
plt.tight_layout(); out=FIG/'V6_ablation.png'; plt.savefig(out,dpi=DPI,bbox_inches='tight'); print(f"\nSaved -> {out}")
