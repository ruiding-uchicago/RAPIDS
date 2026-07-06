#!/usr/bin/env python3
"""
73 — ALORS baseline (Misir & Sebag 2017): algorithm selection via collaborative filtering.
Canonical one-shot per-instance algorithm selection with matrix factorization + cold-start.

Method:
  1. Training (systems x arms) error matrix E; low-rank factorize E ~ U V^T (TruncatedSVD).
  2. Cold-start regressor f: features -> U (system latent factors) [Ridge].
  3. Test x: Û=f(x); predicted per-arm error = Û V^T; select argmin (cost-blind, ALORS default)
     or cost-aware argmin cost + λ·pred (ALORS+cost, for fair comparison to V6).
Evaluated LOBO (18) + charged CV (5), charged-aware training. Encode-once for speed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]
MDL=OUT/'models'/'rapids_select_v5_final'
manifest=json.load(open(MDL/'manifest.json')); feature_cols=manifest['feature_cols']; arm_costs=manifest['arm_costs']; ARMS=manifest['arms']
ARM_ENERGY_COL=b25c.ARM_ENERGY_COL; ARM_TIME_COL=b25c.ARM_TIME_COL
arms_sorted=sorted(ARMS,key=lambda a:arm_costs.get(a,1e9)); FIXED=np.array([arm_costs[a] for a in arms_sorted]); K=len(arms_sorted)
ERR_CAP=50.0; RANK=3

ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False); chg=chg[chg['Reference'].notna()&chg['RAPIDS'].notna()].reset_index(drop=True)
n_ind=len(ind); pool=pd.concat([ind,chg],ignore_index=True)
is_chg=np.zeros(len(pool),bool); is_chg[n_ind:]=True
bench=pool['benchmark'].values; ref=pool['Reference'].values.astype(float)
X=b25c.encode_features(pool,feature_cols).astype(np.float32)
X=np.nan_to_num(X,nan=0.0)  # ALORS ridge needs finite features
true_err=np.full((K,len(pool)),np.nan)
cost=np.full((K,len(pool)),np.nan)
for j,a in enumerate(arms_sorted):
    true_err[j]=np.minimum(np.abs(pool[ARM_ENERGY_COL[a]].values.astype(float)-ref),ERR_CAP)
    c=np.full(len(pool),FIXED[j])
    if ARM_TIME_COL[a] in pool:
        t=pool[ARM_TIME_COL[a]].values.astype(float); mm=~np.isnan(t)&~is_chg; c[mm]=t[mm]
    cost[j]=c
print(f"pool {len(pool)}  X {X.shape}", flush=True)

from sklearn.preprocessing import StandardScaler
def alors_fit_predict(tr,te):
    E=true_err[:,tr].T.copy()          # [n_tr, K]
    col_mean=np.nanmean(E,0); inds=np.where(np.isnan(E)); E[inds]=np.take(col_mean,inds[1])
    svd=TruncatedSVD(n_components=RANK,random_state=0); U=svd.fit_transform(E); V=svd.components_  # U[n_tr,r], V[r,K]
    sc=StandardScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[te])
    reg=Ridge(alpha=10.0); reg.fit(Xtr,U)
    Ute=reg.predict(Xte); pred=Ute@V                  # [n_te, K]
    return pred.T                                     # [K, n_te]

def metrics(te_idx,chosen):
    e=true_err[chosen,te_idx]; c=cost[chosen,te_idx]
    return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)

def run(fold_iter, label):
    cb=[]; ca=[]
    for tr,te in fold_iter:
        pred=alors_fit_predict(tr,te)
        cb.append(metrics(te, np.argmin(pred,0)))
        ca.append(metrics(te, np.argmin(FIXED[:,None]+800*pred,0)))
    cb=np.array(cb).mean(0); ca=np.array(ca).mean(0)
    print(f"  {label} ALORS cost-blind : cost {cb[0]:.0f}  err {cb[1]:.3f}  cat {cb[2]*100:.2f}%")
    print(f"  {label} ALORS +cost(λ800): cost {ca[0]:.0f}  err {ca[1]:.3f}  cat {ca[2]*100:.2f}%")

# in-dist LOBO
def lobo_folds():
    for held in sorted(ind['benchmark'].unique()):
        te=np.where((~is_chg)&(bench==held))[0]
        if len(te)==0: continue
        tr=np.where(~((~is_chg)&(bench==held)))[0]
        yield tr,te
print("IN-DIST LOBO:")
run(lobo_folds(),"in-dist")

# charged CV
def chg_folds():
    cidx=np.where(is_chg)[0]; kf=KFold(5,shuffle=True,random_state=0)
    for _,te_rel in kf.split(cidx):
        te=cidx[te_rel]; tr=np.where(~np.isin(np.arange(len(pool)),te))[0]
        yield tr,te
print("CHARGED CV:")
run(chg_folds(),"charged")
print("\n(ref) V6 in-dist 375/2.44/4.9%  charged 704/9.88/19.7%")
