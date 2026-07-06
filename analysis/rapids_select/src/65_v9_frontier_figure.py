#!/usr/bin/env python3
"""65 — V9 cost-RISK frontier figure: selector dominates static baselines on the
practitioner-relevant (cost, catastrophic-rate) plane, both regimes."""
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (ARM_COLOR, ARM_MARKER, SELECTOR_C, SELECTOR_LW, DPI,
                          FS_LABEL, FS_TITLE, FS_SUPTITLE, FS_LEGEND, FS_ANNOT,
                          apply_rc, style_grid)
apply_rc()

OUT = Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'; FIG=OUT/'figures'; DATA=OUT/'data'
I=pickle.load(open(CACHE/'v8_indist.pkl','rb')); C=pickle.load(open(CACHE/'v8_charged.pkl','rb'))
arms=I['arms']; K=len(arms); Ic=list(I['caches'].values()); Cc=C['caches']

# charged eval must use DES370K gold ONLY (981 systems); pooling with IL174 silver
# inflates catastrophic-rate/MAE ~3x. Reconstruct the exact 5-fold split and mask to gold.
import pandas as pd
from sklearn.model_selection import KFold
_chg=pd.read_csv(DATA/'p02_selector_feature_matrix.csv',low_memory=False)
_chg=_chg[_chg['Reference'].notna()&_chg['RAPIDS'].notna()].reset_index(drop=True)
_tier=_chg['reference_tier'].values
GOLD=[(_tier[te]=='gold') for _,te in KFold(5,shuffle=True,random_state=0).split(np.arange(len(_chg)))]
N_GOLD=int(sum(m.sum() for m in GOLD))

def stack(folds,masks=None):
    if masks is None:
        sel=lambda k:[f[k] for f in folds]
    else:
        sel=lambda k:[f[k][:,m] for f,m in zip(folds,masks)]
    return (np.concatenate(sel('pcat10'),1),
            np.concatenate(sel('true_err'),1),
            np.concatenate(sel('cost'),1))
def curve(folds,masks=None):
    pc,te,co=stack(folds,masks); N=pc.shape[1]; xs=[];ys=[]
    for t in np.linspace(0,1,101):
        ok=pc<=t; first=np.where(ok.any(0),ok.argmax(0),pc.argmin(0))
        e=te[first,np.arange(N)]; c=co[first,np.arange(N)]
        xs.append(np.nanmean(c)); ys.append(np.nanmean(e>10)*100)
    return np.array(xs),np.array(ys)
def basepts(folds,masks=None):
    pc,te,co=stack(folds,masks); N=pc.shape[1]; pts={}
    for j,a in enumerate(arms): pts[a]=(np.nanmean(co[j]),np.nanmean(te[j]>10)*100)
    ba=np.nanargmin(te,0); pts['Oracle']=(np.nanmean(co[ba,np.arange(N)]),np.nanmean(te[ba,np.arange(N)]>10)*100)
    return pts

# literature selector baselines, evaluated on OUR 5-arm caches / benchmarks (apples-to-apples).
# in-dist = 18-benchmark (scripts 72/73/78); charged = DES370K gold 981 (scripts 82/83). cost/err/cat%.
LIT={  # name: {panel: (cost_s, cat_rate_pct)}, 's': method label, 'y': citation year
    'SATzilla-style (cost-blind)':{'in':(1027,5.30),'ch':(1861,4.91),'s':'SATzilla','y':2008},
    'FrugalML cascade':           {'in':(417,4.35), 'ch':(676,4.60), 's':'FrugalML','y':2020},
    'reject/defer rule':          {'in':(495,4.51), 'ch':(1636,5.22),'s':'defer','y':2020},
    'ALORS +cost':                {'in':(769,4.90), 'ch':(847,7.34), 's':'ALORS','y':2017},
    'Multiclass (Rice)':          {'in':(1066,6.16),'ch':(1903,5.00),'s':'Rice','y':1976},
}
LIT_C='#555555'
SHORT={'RAPIDS':'RAPIDS','PBE-D3BJ_SP':'PBE SP','PBE-D3BJ_GeoSP':'PBE GeoSP','CREST_xTB':'CREST','CREST_xTB_DFT':'CREST+DFT','Oracle':'Oracle'}
from adjustText import adjust_text
fig,ax=plt.subplots(1,2,figsize=(16,7.0))
for p,(folds,masks,title) in enumerate([
        (Ic,None,'In-distribution (18 benchmarks, 5,532 systems)'),
        (Cc,GOLD,f'Charged OOD (DES370K gold, {N_GOLD} systems)')]):
    a=ax[p]; xs,ys=curve(folds,masks); pts=basepts(folds,masks)
    texts=[]
    a.plot(xs,ys,'-',color=SELECTOR_C,lw=SELECTOR_LW,alpha=0.7,zorder=3,label='RAPIDS-Select cost–risk frontier')
    for name,(cx,cy) in pts.items():
        a.scatter(cx,cy,s=210 if name!='Oracle' else 340,marker=ARM_MARKER[name],color=ARM_COLOR[name],
                  edgecolors='black',lw=0.6,alpha=0.75 if name!='Oracle' else 0.85,zorder=8)
        texts.append(a.text(cx,cy,f'{SHORT[name]}',fontsize=FS_ANNOT+0.5,color=ARM_COLOR[name],fontweight='bold'))
    key='in' if p==0 else 'ch'
    for i,(name,d) in enumerate(LIT.items()):
        cx,cy=d[key]
        a.scatter(cx,cy,s=150,marker='X',color=LIT_C,edgecolors='black',lw=0.6,alpha=0.75,zorder=9,
                  label='literature selectors (same benchmark)' if i==0 else None)
        texts.append(a.text(cx,cy,f"{d['s']} ({d['y']})",fontsize=FS_ANNOT+0.5,color=LIT_C))
    a.set_xscale('log'); a.set_xlabel('mean per-system cost (s)',fontsize=FS_LABEL)
    a.set_ylabel('catastrophic rate (%, |err|>10 kcal/mol)',fontsize=FS_LABEL)
    a.set_title(title,fontsize=FS_TITLE,fontweight='bold'); style_grid(a)
    a.legend(fontsize=FS_LEGEND,loc='upper right',framealpha=0.85,edgecolor='gray',fancybox=False)
    adjust_text(texts,ax=a,expand=(1.15,1.4),force_text=(0.4,0.6),
                arrowprops=dict(arrowstyle='-',color='0.6',lw=0.5))
plt.suptitle('RAPIDS-Select — cost vs catastrophic-risk frontier dominates every static single-fidelity baseline',
             fontsize=FS_SUPTITLE,fontweight='bold')
plt.tight_layout(); out=FIG/'V9_cost_risk_frontier.png'; plt.savefig(out,dpi=DPI,bbox_inches='tight')
print(f"Saved -> {out}")
