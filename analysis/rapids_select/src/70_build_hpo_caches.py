#!/usr/bin/env python3
"""
70 — Build router caches with the HPO-best regressor config (transfer-aware winner:
+5.6% LOBO MAE via heavy regularization). Decisive validation: does the transfer-metric
gain translate to actual routing (cost/err/cat)? Same encode-once structure as script 54.
Saved: cache/vhpo_indist.pkl + cache/vhpo_charged.pkl
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
MDL = OUT/'models'/'rapids_select_v5_final'; CACHE=OUT/'cache'
ERR_CAP=50.0
manifest=json.load(open(MDL/'manifest.json'))
feature_cols=manifest['feature_cols']; arm_costs=manifest['arm_costs']; ARMS=manifest['arms']
ARM_ENERGY_COL=b25c.ARM_ENERGY_COL; ARM_TIME_COL=b25c.ARM_TIME_COL
arms_sorted=sorted(ARMS,key=lambda a:arm_costs.get(a,1e9)); FIXED=np.array([arm_costs[a] for a in arms_sorted]); K=len(arms_sorted)

hpo=json.load(open(OUT/'models'/'hpo_best.json'))
REG=hpo['reg_best']; CLF=hpo['clf_best']
print(f"HPO reg params: {REG}", flush=True)

ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False); chg=chg[chg['Reference'].notna()&chg['RAPIDS'].notna()].reset_index(drop=True)
n_ind=len(ind); pool=pd.concat([ind,chg],ignore_index=True)
is_chg=np.zeros(len(pool),bool); is_chg[n_ind:]=True
bench=pool['benchmark'].values; ref=pool['Reference'].values.astype(float)
X=b25c.encode_features(pool,feature_cols).astype(np.float32)
print(f"encoded {X.shape}", flush=True)
true_err=np.full((K,len(pool)),np.nan,np.float32); cost=np.full((K,len(pool)),np.nan,np.float32)
for j,a in enumerate(arms_sorted):
    e=pool[ARM_ENERGY_COL[a]].values.astype(float); true_err[j]=np.minimum(np.abs(e-ref),ERR_CAP)
    c=np.full(len(pool),FIXED[j])
    if ARM_TIME_COL[a] in pool:
        t=pool[ARM_TIME_COL[a]].values.astype(float); mm=~np.isnan(t)&~is_chg; c[mm]=t[mm]
    cost[j]=c
rj=arms_sorted.index('RAPIDS'); rerr=true_err[rj]
y10=(rerr>10).astype(int)

def build(tr,te_idx):
    Xte=X[te_idx]; n=len(te_idx); preds=np.full((K,n),ERR_CAP,np.float32); pc10=np.zeros((K,n),np.float32)
    for j in range(K):
        y=true_err[j]; m=tr&~np.isnan(y)
        if m.sum()>=50:
            r=xgb.XGBRegressor(**REG,objective='reg:absoluteerror',n_jobs=-1,verbosity=0,random_state=0); r.fit(X[m],y[m]); preds[j]=r.predict(Xte)
        yy=(true_err[j]>10).astype(int); mm=tr&~np.isnan(true_err[j]); pos=yy[mm].sum(); neg=mm.sum()-pos
        if pos>=5:
            c=xgb.XGBClassifier(**CLF,scale_pos_weight=max(1.0,neg/max(pos,1)),eval_metric='logloss',n_jobs=-1,verbosity=0,random_state=0); c.fit(X[mm],yy[mm]); pc10[j]=c.predict_proba(Xte)[:,1]
    return {'preds':preds,'pcat10':pc10,'true_err':true_err[:,te_idx].copy(),'cost':cost[:,te_idx].copy(),'n':n}

print("in-dist LOBO...", flush=True); indist={}
for held in sorted(ind['benchmark'].unique()):
    te=np.where((~is_chg)&(bench==held))[0]
    if len(te)==0: continue
    tr=np.ones(len(pool),bool); tr[(~is_chg)&(bench==held)]=False
    indist[held]=build(tr,te); print(f"  {held}: n={indist[held]['n']}", flush=True)
pickle.dump({'arms':arms_sorted,'fixed_cost':FIXED,'caches':indist},open(CACHE/'vhpo_indist.pkl','wb'))
print("charged CV...", flush=True); cidx=np.where(is_chg)[0]; kf=KFold(5,shuffle=True,random_state=0); ch=[]
for fold,(_,te_rel) in enumerate(kf.split(cidx)):
    te=cidx[te_rel]; tr=np.ones(len(pool),bool); tr[te]=False
    ch.append(build(tr,te)); print(f"  fold {fold}: n={ch[-1]['n']}", flush=True)
pickle.dump({'arms':arms_sorted,'fixed_cost':FIXED,'caches':ch},open(CACHE/'vhpo_charged.pkl','wb'))
print("Saved -> cache/vhpo_indist.pkl + cache/vhpo_charged.pkl", flush=True)
