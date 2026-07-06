#!/usr/bin/env python3
"""
67 — Transfer-aware HPO (goal: search XGBoost hyperparams until the LOBO ceiling).

Key design: the objective is GroupKFold with groups=benchmark, i.e. every validation
fold is UNSEEN chemistry (LOBO-style). This makes the search itself transfer-aware, so
a config that merely overfits training benchmarks scores badly — the honest way to ask
"can HPO push the LOBO number, or is 0.56/2.44 a real ceiling?"

Two Optuna studies:
  (1) arm error regressor — minimize mean GroupKFold-5 MAE over the 3 routing-critical
      arms (RAPIDS, PBE-D3BJ_SP, PBE-D3BJ_GeoSP). If HPO can't lower LOBO MAE, routing won't improve.
  (2) catastrophe classifier P(RAPIDS err>10) — maximize GroupKFold-5 AUC (the transferable signal).
Saves best configs to models/hpo_best.json. Baseline (V6 config) scored for reference.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT = Path(__file__).resolve().parents[1]
MDL = OUT/'models'/'rapids_select_v5_final'
feature_cols = json.load(open(MDL/'manifest.json'))['feature_cols']
ARM_ENERGY_COL = b25c.ARM_ENERGY_COL
ERR_CAP = 50.0
optuna.logging.set_verbosity(optuna.logging.WARNING)

ind = pd.read_csv(OUT/'data'/'selector_feature_matrix.csv', low_memory=False)
ind = ind[ind['Reference'].notna()].reset_index(drop=True)
chg = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
chg = chg[chg['Reference'].notna() & chg['RAPIDS'].notna()].reset_index(drop=True)
pool = pd.concat([ind, chg], ignore_index=True)
groups = pool['benchmark'].astype(str).values
ref = pool['Reference'].values.astype(float)
X = b25c.encode_features(pool, feature_cols).astype(np.float32)
print(f"pool {len(pool)}  groups {len(np.unique(groups))}  X {X.shape}", flush=True)

def arm_err(a):
    return np.minimum(np.abs(pool[ARM_ENERGY_COL[a]].values.astype(float) - ref), ERR_CAP)

ROUTING_ARMS = ['RAPIDS','PBE-D3BJ_SP','PBE-D3BJ_GeoSP']
Y = {a: arm_err(a) for a in ROUTING_ARMS}
ycat = (arm_err('RAPIDS') > 10).astype(int)
gkf = GroupKFold(n_splits=5)

def cv_mae(params):
    maes=[]
    for a in ROUTING_ARMS:
        y=Y[a]; m=~np.isnan(y)
        Xa,ya,ga=X[m],y[m],groups[m]
        fold_mae=[]
        for tr,va in gkf.split(Xa,ya,ga):
            mod=xgb.XGBRegressor(**params,objective='reg:absoluteerror',n_jobs=-1,verbosity=0,random_state=0)
            mod.fit(Xa[tr],ya[tr]); p=mod.predict(Xa[va])
            fold_mae.append(np.mean(np.abs(p-ya[va])))
        maes.append(np.mean(fold_mae))
    return np.mean(maes)

def cv_auc(params):
    aucs=[]
    for tr,va in gkf.split(X,ycat,groups):
        pos=ycat[tr].sum(); neg=len(tr)-pos
        mod=xgb.XGBClassifier(**params,scale_pos_weight=max(1.0,neg/max(pos,1)),
                              eval_metric='logloss',n_jobs=-1,verbosity=0,random_state=0)
        mod.fit(X[tr],ycat[tr]); p=mod.predict_proba(X[va])[:,1]
        if len(np.unique(ycat[va]))>1: aucs.append(roc_auc_score(ycat[va],p))
    return np.mean(aucs)

# baseline V6 config
V6_REG=dict(n_estimators=120,max_depth=4,learning_rate=0.1,subsample=0.85,colsample_bytree=0.7)
V6_CLF=dict(n_estimators=200,max_depth=4,learning_rate=0.1,subsample=0.85,colsample_bytree=0.7)
base_mae=cv_mae(V6_REG); base_auc=cv_auc(V6_CLF)
print(f"\nV6 baseline: LOBO-GroupKFold MAE {base_mae:.4f}   catastrophe AUC {base_auc:.4f}", flush=True)

def reg_space(t):
    return dict(
        n_estimators=t.suggest_int('n_estimators',60,600),
        max_depth=t.suggest_int('max_depth',2,8),
        learning_rate=t.suggest_float('learning_rate',0.01,0.3,log=True),
        subsample=t.suggest_float('subsample',0.5,1.0),
        colsample_bytree=t.suggest_float('colsample_bytree',0.4,1.0),
        reg_lambda=t.suggest_float('reg_lambda',0.0,10.0),
        reg_alpha=t.suggest_float('reg_alpha',0.0,5.0),
        min_child_weight=t.suggest_float('min_child_weight',1.0,20.0),
        gamma=t.suggest_float('gamma',0.0,5.0),
    )

print("\n=== Study 1: arm error regressor (minimize LOBO MAE) ===", flush=True)
s1=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=0))
s1.optimize(lambda t: cv_mae(reg_space(t)), n_trials=120, show_progress_bar=False)
print(f"  best LOBO MAE {s1.best_value:.4f}  (V6 {base_mae:.4f}, delta {base_mae-s1.best_value:+.4f} = {(base_mae-s1.best_value)/base_mae*100:+.1f}%)")
print(f"  best params: {s1.best_params}")

print("\n=== Study 2: catastrophe classifier (maximize LOBO AUC) ===", flush=True)
s2=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=0))
s2.optimize(lambda t: cv_auc(reg_space(t)), n_trials=80, show_progress_bar=False)
print(f"  best LOBO AUC {s2.best_value:.4f}  (V6 {base_auc:.4f}, delta {s2.best_value-base_auc:+.4f})")
print(f"  best params: {s2.best_params}")

json.dump({'reg_best':s1.best_params,'reg_best_mae':s1.best_value,'reg_base_mae':base_mae,
           'clf_best':s2.best_params,'clf_best_auc':s2.best_value,'clf_base_auc':base_auc},
          open(OUT/'models'/'hpo_best.json','w'),indent=2)
print("\nSaved -> models/hpo_best.json", flush=True)
print(f"\nVERDICT: regressor HPO buys {(base_mae-s1.best_value)/base_mae*100:+.1f}% LOBO MAE; "
      f"classifier HPO buys {s2.best_value-base_auc:+.4f} AUC.", flush=True)
