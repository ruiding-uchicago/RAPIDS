#!/usr/bin/env python3
"""
74 — Exhaustive V6 ablation via retraining (feature-group drop-one + drop-charged-aware).
Encode-once; per config train 5 arm regressors, route with V6 rule (argmin cost+800*pred_err),
evaluate on in-dist 18 LOBO + charged 5-fold CV. Outputs per-benchmark values for error bars.

Configs:
  full                 — charged-aware, all 156 features (V6 reference)
  drop-charged-aware   — train on in-dist ONLY (exclude P0.2 charged), all features
  drop-<GROUP>         — charged-aware, that feature group masked to NaN (8 groups)
"""
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'; RES=OUT/'results'
MDL=OUT/'models'/'rapids_select_v5_final'
manifest=json.load(open(MDL/'manifest.json')); feature_cols=manifest['feature_cols']; arm_costs=manifest['arm_costs']; ARMS=manifest['arms']
ARM_ENERGY_COL=b25c.ARM_ENERGY_COL; ARM_TIME_COL=b25c.ARM_TIME_COL
arms_sorted=sorted(ARMS,key=lambda a:arm_costs.get(a,1e9)); FIXED=np.array([arm_costs[a] for a in arms_sorted]); K=len(arms_sorted)
ERR_CAP=50.0; LAM=800
groups=pickle.load(open(CACHE/'feature_groups.pkl','rb'))
GROUP_NAMES=sorted(set(groups.values()))
col_group=np.array([groups[c] for c in feature_cols])

ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False); chg=chg[chg['Reference'].notna()&chg['RAPIDS'].notna()].reset_index(drop=True)
n_ind=len(ind); pool=pd.concat([ind,chg],ignore_index=True)
is_chg=np.zeros(len(pool),bool); is_chg[n_ind:]=True
bench=pool['benchmark'].values; ref=pool['Reference'].values.astype(float)
X0=b25c.encode_features(pool,feature_cols).astype(np.float32)
true_err=np.full((K,len(pool)),np.nan); cost=np.full((K,len(pool)),np.nan)
for j,a in enumerate(arms_sorted):
    true_err[j]=np.minimum(np.abs(pool[ARM_ENERGY_COL[a]].values.astype(float)-ref),ERR_CAP)
    c=np.full(len(pool),FIXED[j])
    if ARM_TIME_COL[a] in pool:
        t=pool[ARM_TIME_COL[a]].values.astype(float); mm=~np.isnan(t)&~is_chg; c[mm]=t[mm]
    cost[j]=c
print(f"pool {len(pool)}  X {X0.shape}  groups {GROUP_NAMES}", flush=True)

def train_pred(X, tr_mask, te_idx):
    preds=np.full((K,len(te_idx)),ERR_CAP)
    for j in range(K):
        y=true_err[j]; m=tr_mask&~np.isnan(y)
        if m.sum()<50: continue
        mod=xgb.XGBRegressor(n_estimators=120,max_depth=4,learning_rate=0.1,subsample=0.85,
                             colsample_bytree=0.7,random_state=0,objective='reg:absoluteerror',n_jobs=-1,verbosity=0)
        mod.fit(X[m],y[m]); preds[j]=mod.predict(X[te_idx])
    return preds

def route(preds): return np.argmin(FIXED[:,None]+LAM*preds.astype(float),0)
def metric(te_idx,ch):
    e=true_err[ch,te_idx]; c=cost[ch,te_idx]; return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)

def eval_config(X, drop_charged=False):
    # in-dist LOBO (per-benchmark rows)
    ind_rows=[]
    for held in sorted(ind['benchmark'].unique()):
        te=np.where((~is_chg)&(bench==held))[0]
        if len(te)==0: continue
        tr=np.ones(len(pool),bool); tr[(~is_chg)&(bench==held)]=False
        if drop_charged: tr &= ~is_chg     # train on in-dist only
        preds=train_pred(X,tr,te); ind_rows.append(metric(te,route(preds)))
    ind_rows=np.array(ind_rows)
    # charged CV
    cidx=np.where(is_chg)[0]; kf=KFold(5,shuffle=True,random_state=0); ch_rows=[]
    for _,te_rel in kf.split(cidx):
        te=cidx[te_rel]; tr=np.ones(len(pool),bool); tr[te]=False
        if drop_charged: tr &= ~is_chg
        preds=train_pred(X,tr,te); ch_rows.append(metric(te,route(preds)))
    ch_rows=np.array(ch_rows)
    return ind_rows, ch_rows

results={}
configs=[('full',X0,False),('drop-charged-aware',X0,True)]
for G in GROUP_NAMES:
    Xg=X0.copy(); Xg[:,col_group==G]=np.nan; configs.append((f'drop-{G}',Xg,False))

for name,X,dc in configs:
    ir,cr=eval_config(X,dc)
    results[name]={'indist_perbench':ir.tolist(),'charged_perfold':cr.tolist(),
                   'indist_mean':ir.mean(0).tolist(),'charged_mean':cr.mean(0).tolist()}
    im=ir.mean(0); cm=cr.mean(0)
    print(f"  {name:<22} in-dist {im[0]:>5.0f}/{im[1]:.3f}/{im[2]*100:.2f}%   charged {cm[0]:>5.0f}/{cm[1]:.3f}/{cm[2]*100:.2f}%", flush=True)

json.dump(results, open(RES/'ablation_retrain.json','w'), indent=2)
print(f"\nSaved -> results/ablation_retrain.json", flush=True)
