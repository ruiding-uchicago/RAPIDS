#!/usr/bin/env python3
"""
56 — Train FINAL V6 on the full pool (ALL in-dist + ALL P0.2 charged).

V6 = arm-regressor greedy, charged-aware, NO charge short-circuit.
  arm*(x) = cheapest arm a with pred_err_a(x) <= eps   (default eps set by script 55)
Also saves p5/p10/p20 detectors for optional catastrophic guard.
Dump to models/rapids_select_v6_final/.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT = Path(__file__).resolve().parents[1]
MDL5 = OUT/'models'/'rapids_select_v5_final'
MDL = OUT/'models'/'rapids_select_v6_final'; MDL.mkdir(exist_ok=True, parents=True)
ERR_CAP = 50.0

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 800.0

manifest5 = json.load(open(MDL5/'manifest.json'))
feature_cols = manifest5['feature_cols']
arm_costs = manifest5['arm_costs']
ARMS = manifest5['arms']
ARM_ENERGY_COL = b25c.ARM_ENERGY_COL
arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a, 1e9))

ind = pd.read_csv(OUT/'data'/'selector_feature_matrix.csv', low_memory=False)
ind = ind[ind['Reference'].notna()].reset_index(drop=True)
chg = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
chg = chg[chg['Reference'].notna() & chg['RAPIDS'].notna()].reset_index(drop=True)
pool = pd.concat([ind, chg], ignore_index=True)
print(f"Training V6 on full pool: {len(pool)} ({len(ind)} in-dist + {len(chg)} charged), lambda={LAM}", flush=True)

X = b25c.encode_features(pool, feature_cols)
ref = pool['Reference'].values.astype(float)
rerr = np.minimum(np.abs(pool['RAPIDS'].values.astype(float) - ref), ERR_CAP)

def err_arm(a):
    return np.minimum(np.abs(pool[ARM_ENERGY_COL[a]].values.astype(float) - ref), ERR_CAP)

for a in arms_sorted:
    y = err_arm(a); m = ~np.isnan(y)
    mod = xgb.XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.1,
                           subsample=0.85, colsample_bytree=0.7, random_state=0,
                           objective='reg:absoluteerror', n_jobs=-1, verbosity=0)
    mod.fit(X[m], y[m]); mod.save_model(str(MDL/f'arm_{a}.json'))
    print(f"  arm {a}: n={m.sum()}", flush=True)

for name, thr in [('cat_p5',5),('cat_p10',10),('cat_p20',20)]:
    y = (rerr > thr).astype(int); pos=y.sum(); neg=len(y)-pos
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            subsample=0.85, colsample_bytree=0.7, random_state=0,
                            scale_pos_weight=max(1.0, neg/max(pos,1)),
                            eval_metric='logloss', n_jobs=-1, verbosity=0)
    clf.fit(X, y); clf.save_model(str(MDL/f'{name}.json'))
    print(f"  {name}: pos={pos}", flush=True)

json.dump({
    'feature_cols': feature_cols, 'arm_costs': arm_costs, 'arms': arms_sorted,
    'policy': 'cost_aware_lambda_no_charge_shortcircuit',
    'lambda': LAM,
    'rule': 'arm*(x) = argmin_a [ cost(a) + lambda * pred_err_a(x) ]; NO charge special-casing',
    'note': 'charged-aware training (in-dist + P0.2 charged). Dual-win holds for lambda in [300,3000].',
    'trained_on': {'in_dist': len(ind), 'charged': len(chg), 'total': len(pool)},
}, open(MDL/'manifest.json','w'), indent=2)
print(f"\nSaved V6 -> {MDL}  (lambda={LAM})", flush=True)
