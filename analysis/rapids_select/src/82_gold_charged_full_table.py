#!/usr/bin/env python3
"""82 — Complete GOLD-only (DES370K) charged table from v8 cache: selector + all
cache-derivable baselines. Corrects the gold+silver pooling error. No retraining."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
OUT=Path(__file__).resolve().parents[1]
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False); chg=chg[chg['Reference'].notna()&chg['RAPIDS'].notna()].reset_index(drop=True)
tier=chg['reference_tier'].values
C=pickle.load(open(OUT/'cache'/'v8_charged.pkl','rb')); arms=C['arms']; FIXED=C['fixed_cost']; idx={a:i for i,a in enumerate(arms)}; caches=C['caches']
folds=[te for _,te in KFold(5,shuffle=True,random_state=0).split(np.arange(len(chg)))]

def evalp(pick):
    r=[]
    for fi,c in enumerate(caches):
        msk=tier[folds[fi]]=='gold'; ch=pick(c); n=c['n']
        e=c['true_err'][ch,np.arange(n)][msk]; co=c['cost'][ch,np.arange(n)][msk]
        r.append((np.nanmean(co),np.nanmean(e),np.nanmean(e>10)))
    return np.array(r).mean(0)

pols={
 'V6 selector (argmin cost+λ·err)': lambda c: np.argmin(FIXED[:,None]+800*c['preds'].astype(float),0),
 'Cost-blind / SATzilla (argmin ê)': lambda c: np.argmin(c['preds'].astype(float),0),
 'FrugalML-style (pcat10<=0.1)': lambda c: (lambda ok: np.where(ok.any(0),ok.argmax(0),c['pcat10'].argmin(0)))(c['pcat10']<=0.1),
 'Always-RAPIDS': lambda c: np.full(c['n'],idx['RAPIDS']),
 'Always-PBE-D3BJ_SP': lambda c: np.full(c['n'],idx['PBE-D3BJ_SP']),
 'Always-GeoSP': lambda c: np.full(c['n'],idx['PBE-D3BJ_GeoSP']),
 'Always-CREST_xTB': lambda c: np.full(c['n'],idx['CREST_xTB']),
 'Always-CREST_xTB_DFT': lambda c: np.full(c['n'],idx['CREST_xTB_DFT']),
 'Oracle': lambda c: np.nanargmin(c['true_err'],0),
}
print("GOLD (DES370K, 981) charged — cost / MAE / catastrophic%:")
for name,pk in pols.items():
    m=evalp(pk); print(f"  {name:<36} {m[0]:>5.0f} / {m[1]:.2f} / {m[2]*100:.2f}%")
print("\n(silver IL174 = ~49 MAE / ~100% cat for ALL methods — reference-scale artifact, report separately)")
