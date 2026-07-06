#!/usr/bin/env python3
"""
78 — Classification-based per-instance selector baseline (Rice 1976 algorithm-selection
counterpart to our regression approach). Closes the reviewer gap "why only regression?".

Two variants, both one-shot, charged-aware training, retrained per LOBO/charged fold:
  (a) multiclass-argmin: XGBoost multiclass classifier predicts the ORACLE-BEST arm
      (argmin true err) directly from features; pick predicted class. Cost-BLIND.
  (b) multiclass + cost tie-break: among classes with predicted prob >= τ·max_prob,
      pick the cheapest — a cost-aware version, for fair comparison to V6.
Encode-once. Reports both battlefields vs V6 / cost-blind-regression / oracle.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]; RES=OUT/'results'
MDL=OUT/'models'/'rapids_select_v5_final'
manifest=json.load(open(MDL/'manifest.json')); feature_cols=manifest['feature_cols']; arm_costs=manifest['arm_costs']; ARMS=manifest['arms']
ARM_ENERGY_COL=b25c.ARM_ENERGY_COL; ARM_TIME_COL=b25c.ARM_TIME_COL
arms_sorted=sorted(ARMS,key=lambda a:arm_costs.get(a,1e9)); FIXED=np.array([arm_costs[a] for a in arms_sorted]); K=len(arms_sorted)
ERR_CAP=50.0

ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False); chg=chg[chg['Reference'].notna()&chg['RAPIDS'].notna()].reset_index(drop=True)
n_ind=len(ind); pool=pd.concat([ind,chg],ignore_index=True)
is_chg=np.zeros(len(pool),bool); is_chg[n_ind:]=True
bench=pool['benchmark'].values; ref=pool['Reference'].values.astype(float)
X=b25c.encode_features(pool,feature_cols).astype(np.float32)
true_err=np.full((K,len(pool)),np.nan); cost=np.full((K,len(pool)),np.nan)
for j,a in enumerate(arms_sorted):
    true_err[j]=np.minimum(np.abs(pool[ARM_ENERGY_COL[a]].values.astype(float)-ref),ERR_CAP)
    c=np.full(len(pool),FIXED[j])
    if ARM_TIME_COL[a] in pool:
        t=pool[ARM_TIME_COL[a]].values.astype(float); mm=~np.isnan(t)&~is_chg; c[mm]=t[mm]
    cost[j]=c
best_arm=np.nanargmin(true_err,0)   # oracle-best label per system
print(f"pool {len(pool)}  X {X.shape}  label dist: {np.bincount(best_arm,minlength=K)}", flush=True)

def fit_clf(tr_mask):
    y=best_arm[tr_mask]
    m=xgb.XGBClassifier(objective='multi:softprob',num_class=K,n_estimators=200,max_depth=4,
                        learning_rate=0.1,subsample=0.85,colsample_bytree=0.7,random_state=0,n_jobs=-1,verbosity=0)
    m.fit(X[tr_mask],y); return m
def metric(te_idx,ch):
    e=true_err[ch,te_idx]; c=cost[ch,te_idx]; return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)

def eval_fold(tr_mask,te_idx):
    clf=fit_clf(tr_mask); P=clf.predict_proba(X[te_idx])   # [n,K]
    # (a) argmax prob = predicted best arm
    ch_a=P.argmax(1)
    # (b) cost tie-break: among classes with prob>=0.5*maxprob, pick cheapest (arms already cost-sorted)
    ch_b=np.empty(len(te_idx),int)
    for i in range(len(te_idx)):
        thr=0.5*P[i].max(); cand=np.where(P[i]>=thr)[0]; ch_b[i]=cand.min()  # cheapest qualifying (index=cost rank)
    return metric(te_idx,ch_a), metric(te_idx,ch_b)

# in-dist LOBO
ia=[];ib=[]
for held in sorted(ind['benchmark'].unique()):
    te=np.where((~is_chg)&(bench==held))[0]
    if len(te)==0: continue
    tr=np.ones(len(pool),bool); tr[(~is_chg)&(bench==held)]=False
    a,b=eval_fold(tr,te); ia.append(a); ib.append(b)
ia=np.array(ia).mean(0); ib=np.array(ib).mean(0)
# charged CV
ca=[];cb=[]
cidx=np.where(is_chg)[0]; kf=KFold(5,shuffle=True,random_state=0)
for _,te_rel in kf.split(cidx):
    te=cidx[te_rel]; tr=np.ones(len(pool),bool); tr[te]=False
    a,b=eval_fold(tr,te); ca.append(a); cb.append(b)
ca=np.array(ca).mean(0); cb=np.array(cb).mean(0)

print("\n=== Multiclass (classification-based) per-instance selector — Rice 1976 counterpart ===")
print(f"{'variant':<34}{'in-dist':^22}{'charged':^22}")
print(f"  {'multiclass argmax (cost-blind)':<32}{ia[0]:>6.0f}/{ia[1]:.3f}/{ia[2]*100:>4.1f}%   {ca[0]:>6.0f}/{ca[1]:.3f}/{ca[2]*100:>4.1f}%")
print(f"  {'multiclass + cost tie-break':<32}{ib[0]:>6.0f}/{ib[1]:.3f}/{ib[2]*100:>4.1f}%   {cb[0]:>6.0f}/{cb[1]:.3f}/{cb[2]*100:>4.1f}%")
print(f"\n(ref) V6 regression-select: in-dist 375/2.44/4.9%  charged 704/9.88/19.7%")
print(f"(ref) cost-blind regression (SATzilla-style): in-dist 1027/2.51/5.3%")
json.dump({'multiclass_argmax':{'indist':list(ia),'charged':list(ca)},
           'multiclass_costtiebreak':{'indist':list(ib),'charged':list(cb)}},
          open(RES/'multiclass_baseline.json','w'),indent=2)
print("Saved -> results/multiclass_baseline.json")
