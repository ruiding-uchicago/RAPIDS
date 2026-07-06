#!/usr/bin/env python3
"""
69 — Paper-level charged CV error bars: repeat the 5-fold charged CV over multiple
seeds, report mean±std. Router = V6 rule argmin cost + lam*pred_err (only needs the
5 arm regressors). Optionally use HPO-best regressor config (arg 'hpo').

Usage: python3 69_charged_multiseed_cv.py [v6|hpo] [lam]
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT = Path(__file__).resolve().parents[1]
MDL = OUT/'models'/'rapids_select_v5_final'
feature_cols = json.load(open(MDL/'manifest.json'))['feature_cols']
arm_costs = json.load(open(MDL/'manifest.json'))['arm_costs']
ARMS = json.load(open(MDL/'manifest.json'))['arms']
ARM_ENERGY_COL = b25c.ARM_ENERGY_COL
arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a,1e9))
FIXED = np.array([arm_costs[a] for a in arms_sorted]); K=len(arms_sorted)
ERR_CAP=50.0

CONFIG = sys.argv[1] if len(sys.argv)>1 else 'v6'
LAM = float(sys.argv[2]) if len(sys.argv)>2 else 800.0
if CONFIG=='hpo':
    reg_params = json.load(open(OUT/'models'/'hpo_best.json'))['reg_best']
else:
    reg_params = dict(n_estimators=120,max_depth=4,learning_rate=0.1,subsample=0.85,colsample_bytree=0.7)
print(f"config={CONFIG} lam={LAM} reg_params={reg_params}", flush=True)

ind = pd.read_csv(OUT/'data'/'selector_feature_matrix.csv', low_memory=False)
ind = ind[ind['Reference'].notna()].reset_index(drop=True)
chg = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
chg = chg[chg['Reference'].notna() & chg['RAPIDS'].notna()].reset_index(drop=True)
pool_ind = ind
Xind = b25c.encode_features(ind, feature_cols).astype(np.float32)
Xchg = b25c.encode_features(chg, feature_cols).astype(np.float32)
refc = chg['Reference'].values.astype(float)
true_err_c = np.full((K,len(chg)),np.nan)
for j,a in enumerate(arms_sorted):
    true_err_c[j]=np.minimum(np.abs(chg[ARM_ENERGY_COL[a]].values.astype(float)-refc),ERR_CAP)

def arm_err(df,a,r): return np.minimum(np.abs(df[ARM_ENERGY_COL[a]].values.astype(float)-r),ERR_CAP)
refi = ind['Reference'].values.astype(float)

SEEDS=[0,1,2,3,4]
per_seed=[]
for seed in SEEDS:
    kf=KFold(5,shuffle=True,random_state=seed)
    fold_metrics=[]
    for tr_rel,te_rel in kf.split(np.arange(len(chg))):
        # train on all in-dist + charged-train
        Xtr=np.vstack([Xind, Xchg[tr_rel]])
        preds=np.full((K,len(te_rel)),ERR_CAP)
        for j,a in enumerate(arms_sorted):
            ytr=np.concatenate([arm_err(ind,a,refi), arm_err(chg.iloc[tr_rel],a,refc[tr_rel])])
            m=~np.isnan(ytr)
            mod=xgb.XGBRegressor(**reg_params,objective='reg:absoluteerror',n_jobs=-1,verbosity=0,random_state=0)
            mod.fit(Xtr[m],ytr[m]); preds[j]=mod.predict(Xchg[te_rel])
        chosen=np.argmin(FIXED[:,None]+LAM*preds.astype(float),0)
        te=true_err_c[:,te_rel]; e=te[chosen,np.arange(len(te_rel))]; c=FIXED[chosen]
        fold_metrics.append((np.nanmean(c),np.nanmean(e),np.nanmean(e>10)))
    fm=np.array(fold_metrics).mean(0)
    per_seed.append(fm)
    print(f"  seed {seed}: cost {fm[0]:.0f}  err {fm[1]:.3f}  cat {fm[2]*100:.2f}%", flush=True)

P=np.array(per_seed)
print(f"\n=== charged CV over {len(SEEDS)} seeds (config={CONFIG}) ===")
for k,name in enumerate(['cost(s)','err(kcal/mol)','catastrophic']):
    v=P[:,k]*(100 if k==2 else 1)
    print(f"  {name:<16} {v.mean():.3f} ± {v.std():.3f}  (min {v.min():.3f}, max {v.max():.3f})")
# GeoSP reference (fixed arm, no training)
gj=arms_sorted.index('PBE-D3BJ_GeoSP')
print(f"\n  Always-GeoSP charged: cost {FIXED[gj]:.0f}  err {np.nanmean(true_err_c[gj]):.3f}  cat {np.nanmean(true_err_c[gj]>10)*100:.2f}%")
