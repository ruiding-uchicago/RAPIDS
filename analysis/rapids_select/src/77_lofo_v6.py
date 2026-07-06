#!/usr/bin/env python3
"""
77 — LOFO (Leave-One-Family-Out) on V6. The 4th generalization split: hold out a whole
chemistry family (all its benchmarks) — stronger than LOBO (which leaves related
benchmarks in training). V6 recipe: charged-aware training, argmin cost+λ·pred_err, λ=800.
Encode-once; per family fold train 5 arm regressors, route, eval. Error bars over folds.

Caveat: for the 'charged' family fold (IHB100+SSI_charged held out), P0.2 charged is a
DIFFERENT charged distribution (DES370K/IL174) so it remains a generalization test, noted.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]; RES=OUT/'results'
MDL=OUT/'models'/'rapids_select_v5_final'
manifest=json.load(open(MDL/'manifest.json')); feature_cols=manifest['feature_cols']; arm_costs=manifest['arm_costs']; ARMS=manifest['arms']
ARM_ENERGY_COL=b25c.ARM_ENERGY_COL; ARM_TIME_COL=b25c.ARM_TIME_COL
arms_sorted=sorted(ARMS,key=lambda a:arm_costs.get(a,1e9)); FIXED=np.array([arm_costs[a] for a in arms_sorted]); K=len(arms_sorted)
ERR_CAP=50.0; LAM=800
FAMILY={'A24':'mixed_small','S66':'mixed_small','BFDb_BBI':'protein','BFDb_HSG':'protein','BFDb_NBC1':'protein',
 'BFDb_SSI_dispersion':'dispersion','BFDb_SSI_mixed':'protein','BFDb_SSI_other':'protein','BFDb_SSI_polar':'polar',
 'D1200_HBCNO':'HBCNO','D1200_PS':'HBCNO','D1200_Halogens':'halogen','HB300SPX':'hbond','HB375':'hbond',
 'IHB100':'charged','BFDb_SSI_charged':'charged','SH250':'sigma_hole','X40':'halogen'}

ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False); chg=chg[chg['Reference'].notna()&chg['RAPIDS'].notna()].reset_index(drop=True)
n_ind=len(ind); pool=pd.concat([ind,chg],ignore_index=True)
is_chg=np.zeros(len(pool),bool); is_chg[n_ind:]=True
bench=pool['benchmark'].values; ref=pool['Reference'].values.astype(float)
fam=np.array([FAMILY.get(b,'?') for b in bench])
X=b25c.encode_features(pool,feature_cols).astype(np.float32)
true_err=np.full((K,len(pool)),np.nan); cost=np.full((K,len(pool)),np.nan)
for j,a in enumerate(arms_sorted):
    true_err[j]=np.minimum(np.abs(pool[ARM_ENERGY_COL[a]].values.astype(float)-ref),ERR_CAP)
    c=np.full(len(pool),FIXED[j])
    if ARM_TIME_COL[a] in pool:
        t=pool[ARM_TIME_COL[a]].values.astype(float); mm=~np.isnan(t)&~is_chg; c[mm]=t[mm]
    cost[j]=c
print(f"pool {len(pool)}  X {X.shape}", flush=True)

def train_pred(tr_mask,te_idx):
    preds=np.full((K,len(te_idx)),ERR_CAP)
    for j in range(K):
        y=true_err[j]; m=tr_mask&~np.isnan(y)
        if m.sum()<50: continue
        mod=xgb.XGBRegressor(n_estimators=120,max_depth=4,learning_rate=0.1,subsample=0.85,
                             colsample_bytree=0.7,random_state=0,objective='reg:absoluteerror',n_jobs=-1,verbosity=0)
        mod.fit(X[m],y[m]); preds[j]=mod.predict(X[te_idx])
    return preds
def route(p): return np.argmin(FIXED[:,None]+LAM*p.astype(float),0)
def metric(te,ch):
    e=true_err[ch,te]; c=cost[ch,te]; return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)

families=sorted(set(FAMILY.values()))
rows=[]
print(f"{'held-out family':<14}{'n_test':>7}{'cost':>8}{'err':>8}{'cat%':>7}")
for held in families:
    te=np.where((~is_chg)&(fam==held))[0]
    tr=np.ones(len(pool),bool); tr[(~is_chg)&(fam==held)]=False
    # for the charged family fold, P0.2 charged is a different charged distribution -> keep (noted)
    preds=train_pred(tr,te); m=metric(te,route(preds))
    rows.append({'family':held,'n':len(te),'cost':m[0],'err':m[1],'cat':m[2]})
    print(f"{held:<14}{len(te):>7}{m[0]:>8.0f}{m[1]:>8.3f}{m[2]*100:>7.2f}", flush=True)

dfr=pd.DataFrame(rows); dfr.to_csv(RES/'LOFO_v6_results.csv',index=False)
# macro-average over families + weighted
macro=dfr[['cost','err','cat']].mean()
w=dfr['n']/dfr['n'].sum(); wtd=(dfr[['cost','err','cat']].T*w.values).T.sum()
print(f"\nLOFO macro-avg (9 families): cost {macro['cost']:.0f}  err {macro['err']:.3f}  cat {macro['cat']*100:.2f}%")
print(f"LOFO system-weighted:        cost {wtd['cost']:.0f}  err {wtd['err']:.3f}  cat {wtd['cat']*100:.2f}%")
print(f"(ref) LOBO macro-avg: 375/2.44/4.89%")
print(f"Saved -> results/LOFO_v6_results.csv")
