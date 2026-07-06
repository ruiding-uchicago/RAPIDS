#!/usr/bin/env python3
"""
54 (v2) — Build V6 prediction caches, FAST: encode the full pool once, slice per fold.

Charged-aware training, no charge short-circuit. Two held-out caches:
  (1) in-dist LOBO (18 folds): train = (17 in-dist benchmarks + ALL P0.2 charged), test = held-out benchmark.
  (2) charged CV (5 folds):    train = (ALL in-dist + 4/5 charged), test = 1/5 charged.
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
MDL = OUT/'models'/'rapids_select_v5_final'
CACHE = OUT/'cache'; CACHE.mkdir(exist_ok=True)
ERR_CAP = 50.0

manifest = json.load(open(MDL/'manifest.json'))
feature_cols = manifest['feature_cols']
arm_costs = manifest['arm_costs']
ARMS = manifest['arms']
ARM_ENERGY_COL = b25c.ARM_ENERGY_COL
ARM_TIME_COL = b25c.ARM_TIME_COL
arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a, 1e9))
FIXED = np.array([arm_costs[a] for a in arms_sorted])
K = len(arms_sorted)

ind = pd.read_csv(OUT/'data'/'selector_feature_matrix.csv', low_memory=False)
ind = ind[ind['Reference'].notna()].reset_index(drop=True)
chg = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
chg = chg[chg['Reference'].notna() & chg['RAPIDS'].notna()].reset_index(drop=True)
n_ind, n_chg = len(ind), len(chg)
print(f"in-dist {n_ind}  charged {n_chg}  pool {n_ind+n_chg}", flush=True)

pool = pd.concat([ind, chg], ignore_index=True)
is_chg = np.zeros(len(pool), dtype=bool); is_chg[n_ind:] = True
bench = pool['benchmark'].values if 'benchmark' in pool else np.array(['?']*len(pool))
ref = pool['Reference'].values.astype(float)

# encode ONCE
print("Encoding full pool once...", flush=True)
X = b25c.encode_features(pool, feature_cols).astype(np.float32)
print(f"  X {X.shape}", flush=True)

# true err + cost per arm for whole pool
true_err = np.full((K, len(pool)), np.nan, dtype=np.float32)
cost = np.full((K, len(pool)), np.nan, dtype=np.float32)
for j,a in enumerate(arms_sorted):
    e = pool[ARM_ENERGY_COL[a]].values.astype(float)
    true_err[j] = np.minimum(np.abs(e - ref), ERR_CAP)
    # in-dist: real walltime if present; charged: fixed cost
    c = np.full(len(pool), FIXED[j], dtype=float)
    if ARM_TIME_COL[a] in pool:
        t = pool[ARM_TIME_COL[a]].values.astype(float)
        m = ~np.isnan(t) & ~is_chg
        c[m] = t[m]
    cost[j] = c

# RAPIDS-error labels (RAPIDS = arms_sorted[0]? verify)
rj = arms_sorted.index('RAPIDS')
rerr = true_err[rj]
y5  = (rerr > 5).astype(int)
y10 = (rerr > 10).astype(int)
y20 = (rerr > 20).astype(int)

def fit_regressor(tr_mask, j):
    y = true_err[j]; m = tr_mask & ~np.isnan(y)
    if m.sum() < 50: return None
    mod = xgb.XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.1,
                           subsample=0.85, colsample_bytree=0.7, random_state=0,
                           objective='reg:absoluteerror', n_jobs=-1, verbosity=0)
    mod.fit(X[m], y[m]); return mod

def fit_clf(tr_mask, y):
    m = tr_mask & ~np.isnan(rerr)  # need valid RAPIDS label
    yy = y[m]; pos = yy.sum(); neg = len(yy)-pos
    if pos < 5: return None
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            subsample=0.85, colsample_bytree=0.7, random_state=0,
                            scale_pos_weight=max(1.0, neg/max(pos,1)),
                            eval_metric='logloss', n_jobs=-1, verbosity=0)
    clf.fit(X[m], yy); return clf

def build(tr_mask, te_idx):
    arm_m = [fit_regressor(tr_mask, j) for j in range(K)]
    d5 = fit_clf(tr_mask, y5); d10 = fit_clf(tr_mask, y10); d20 = fit_clf(tr_mask, y20)
    Xte = X[te_idx]; n = len(te_idx)
    preds = np.full((K, n), ERR_CAP, dtype=np.float32)
    for j in range(K):
        if arm_m[j] is not None: preds[j] = arm_m[j].predict(Xte)
    def pp(c): return c.predict_proba(Xte)[:,1].astype(np.float32) if c is not None else np.zeros(n, np.float32)
    return {'preds':preds,'p5':pp(d5),'p10':pp(d10),'p20':pp(d20),
            'true_err':true_err[:,te_idx].copy(),'cost':cost[:,te_idx].copy(),'n':n}

# ---- (1) in-dist LOBO ----
print("\nin-dist LOBO folds...", flush=True)
indist = {}
for held in sorted(ind['benchmark'].unique()):
    te_idx = np.where((~is_chg) & (bench==held))[0]
    if len(te_idx)==0: continue
    tr_mask = np.ones(len(pool), bool)
    tr_mask[(~is_chg) & (bench==held)] = False  # drop held-out neutral benchmark; keep all charged
    indist[held] = build(tr_mask, te_idx)
    print(f"  {held}: n={indist[held]['n']}", flush=True)
pickle.dump({'arms':arms_sorted,'fixed_cost':FIXED,'caches':indist}, open(CACHE/'v6_indist.pkl','wb'))

# ---- (2) charged CV ----
print("\ncharged CV folds...", flush=True)
chg_pool_idx = np.where(is_chg)[0]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
charged = []
for fold,(tr_rel, te_rel) in enumerate(kf.split(chg_pool_idx)):
    te_idx = chg_pool_idx[te_rel]
    tr_mask = np.ones(len(pool), bool)
    tr_mask[te_idx] = False  # drop this charged fold; keep all in-dist + other charged
    charged.append(build(tr_mask, te_idx))
    print(f"  fold {fold}: n={charged[-1]['n']}", flush=True)
pickle.dump({'arms':arms_sorted,'fixed_cost':FIXED,'caches':charged}, open(CACHE/'v6_charged.pkl','wb'))

print(f"\nSaved -> {CACHE/'v6_indist.pkl'} + {CACHE/'v6_charged.pkl'}")
