#!/usr/bin/env python3
"""
62 — V8: per-arm catastrophe-aware routing caches.

Redirect from round-1 critique: error MAGNITUDE doesn't transfer across chemistry
(LOBO Spearman ~0.56 ceiling), but catastrophe OCCURRENCE does (LOBO AUC 0.94-0.97).
56% of routed error is the 7% catastrophic (>10) cases. So predict per-arm blow-up
probability and route to avoid it at least cost.

Per arm, per fold (charged-aware, encode-once): train
  - mean-err regressor (moderate d4/120, the transfer-robust config)  -> preds
  - catastrophe classifier P(|err_a| > 10)                            -> pcat10
  - catastrophe classifier P(|err_a| > 5)                             -> pcat5
Held-out caches: 18 in-dist LOBO + 5 charged CV.
Saved: cache/v8_indist.pkl + cache/v8_charged.pkl
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

OUT = Path(__file__).resolve().parents[1]
MDL = OUT/'models'/'rapids_select_v5_final'; CACHE = OUT/'cache'
ERR_CAP = 50.0
manifest = json.load(open(MDL/'manifest.json'))
feature_cols = manifest['feature_cols']; arm_costs = manifest['arm_costs']; ARMS = manifest['arms']
ARM_ENERGY_COL = b25c.ARM_ENERGY_COL; ARM_TIME_COL = b25c.ARM_TIME_COL
arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a,1e9))
FIXED = np.array([arm_costs[a] for a in arms_sorted]); K=len(arms_sorted)

ind = pd.read_csv(OUT/'data'/'selector_feature_matrix.csv', low_memory=False)
ind = ind[ind['Reference'].notna()].reset_index(drop=True)
chg = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
chg = chg[chg['Reference'].notna() & chg['RAPIDS'].notna()].reset_index(drop=True)
n_ind=len(ind); pool=pd.concat([ind,chg],ignore_index=True)
is_chg=np.zeros(len(pool),bool); is_chg[n_ind:]=True
bench=pool['benchmark'].values; ref=pool['Reference'].values.astype(float)
print(f"pool {len(pool)}", flush=True)
X=b25c.encode_features(pool,feature_cols).astype(np.float32)
print(f"encoded {X.shape}", flush=True)

true_err=np.full((K,len(pool)),np.nan,np.float32); cost=np.full((K,len(pool)),np.nan,np.float32)
for j,a in enumerate(arms_sorted):
    e=pool[ARM_ENERGY_COL[a]].values.astype(float); true_err[j]=np.minimum(np.abs(e-ref),ERR_CAP)
    c=np.full(len(pool),FIXED[j])
    if ARM_TIME_COL[a] in pool:
        t=pool[ARM_TIME_COL[a]].values.astype(float); mm=~np.isnan(t)&~is_chg; c[mm]=t[mm]
    cost[j]=c

def reg(): return dict(n_estimators=120,max_depth=4,learning_rate=0.1,subsample=0.85,
                       colsample_bytree=0.7,random_state=0,objective='reg:absoluteerror',n_jobs=-1,verbosity=0)
def clf(spw): return dict(n_estimators=200,max_depth=4,learning_rate=0.1,subsample=0.85,
                          colsample_bytree=0.7,random_state=0,scale_pos_weight=spw,
                          eval_metric='logloss',n_jobs=-1,verbosity=0)

def build(tr, te_idx):
    Xte=X[te_idx]; n=len(te_idx)
    preds=np.full((K,n),ERR_CAP,np.float32); pc10=np.zeros((K,n),np.float32); pc5=np.zeros((K,n),np.float32)
    for j in range(K):
        y=true_err[j]; m=tr&~np.isnan(y)
        if m.sum()>=50:
            r=xgb.XGBRegressor(**reg()); r.fit(X[m],y[m]); preds[j]=r.predict(Xte)
        for thr,out in [(10,pc10),(5,pc5)]:
            yy=(true_err[j]>thr).astype(int); mm=tr&~np.isnan(true_err[j])
            pos=yy[mm].sum(); neg=mm.sum()-pos
            if pos>=5:
                c=xgb.XGBClassifier(**clf(max(1.0,neg/max(pos,1)))); c.fit(X[mm],yy[mm])
                out[j]=c.predict_proba(Xte)[:,1]
    return {'preds':preds,'pcat10':pc10,'pcat5':pc5,
            'true_err':true_err[:,te_idx].copy(),'cost':cost[:,te_idx].copy(),'n':n}

print("in-dist LOBO...", flush=True); indist={}
for held in sorted(ind['benchmark'].unique()):
    te=np.where((~is_chg)&(bench==held))[0]
    if len(te)==0: continue
    tr=np.ones(len(pool),bool); tr[(~is_chg)&(bench==held)]=False
    indist[held]=build(tr,te); print(f"  {held}: n={indist[held]['n']}", flush=True)
pickle.dump({'arms':arms_sorted,'fixed_cost':FIXED,'caches':indist},open(CACHE/'v8_indist.pkl','wb'))

print("charged CV...", flush=True); cidx=np.where(is_chg)[0]; kf=KFold(5,shuffle=True,random_state=0); ch=[]
for fold,(_,te_rel) in enumerate(kf.split(cidx)):
    te=cidx[te_rel]; tr=np.ones(len(pool),bool); tr[te]=False
    ch.append(build(tr,te)); print(f"  fold {fold}: n={ch[-1]['n']}", flush=True)
pickle.dump({'arms':arms_sorted,'fixed_cost':FIXED,'caches':ch},open(CACHE/'v8_charged.pkl','wb'))
print("Saved -> cache/v8_indist.pkl + cache/v8_charged.pkl", flush=True)
