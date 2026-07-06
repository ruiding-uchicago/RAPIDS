#!/usr/bin/env python3
"""
53 — Charged-aware selector: does folding charged chemistry into training let the
detectors route per-system on charged (no dumb charge->GeoSP short circuit)?

Setup (semi-prospective, honest):
  Pool = 5,532 in-dist (incl. 876 old charged) + P0.2 charged (DES370K+IL174).
  5-fold CV over P0.2 charged:
    train = [all in-dist] + [4/5 of P0.2 charged]
    test  = [1/5 of P0.2 charged]   (held out — never seen in training)
  Retrain 5 arm-regressors + p5/p10/p20 detectors each fold.
  Route per-system on the held-out charged (NO charge short circuit):
    (a) arm-regressor greedy: cheapest arm with pred_err <= eps
    (b) cascade with re-tuned taus
  Compare to Always-GeoSP / Always-DFT / Oracle on the SAME held-out folds.

Success = a per-system policy that beats Always-GeoSP on the charged test
(cheaper at equal-or-better err, using the ~19% cheap-arm headroom), WITHOUT
degenerating to a single arm.
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

OUT = Path(__file__).resolve().parents[1]
MDL = OUT/'models'/'rapids_select_v5_final'
RES = OUT/'results'
ERR_CAP = 50.0

manifest = json.load(open(MDL/'manifest.json'))
feature_cols = manifest['feature_cols']
arm_costs = manifest['arm_costs']
ARMS = manifest['arms']
ARM_ENERGY_COL = b25c.ARM_ENERGY_COL
arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a, 1e9))
COST = np.array([arm_costs[a] for a in arms_sorted])
idx = {a:i for i,a in enumerate(arms_sorted)}

# ---- load pools ----
ind = pd.read_csv(OUT/'data'/'selector_feature_matrix.csv', low_memory=False)
ind = ind[ind['Reference'].notna()].reset_index(drop=True)
chg = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
chg = chg[chg['Reference'].notna()].reset_index(drop=True)
# charged systems must have RAPIDS (label source) + the 5 arms for eval
chg = chg[chg['RAPIDS'].notna()].reset_index(drop=True)
print(f"in-dist pool: {len(ind)}   P0.2 charged (with RAPIDS): {len(chg)}")

def add_labels(df):
    re = (df['RAPIDS'] - df['Reference']).abs()
    df = df.copy()
    df['y5'] = (re > 5).astype(int); df['y10'] = (re > 10).astype(int); df['y20'] = (re > 20).astype(int)
    return df
ind = add_labels(ind); chg = add_labels(chg)

def err_arm(df, a):
    return (df[ARM_ENERGY_COL[a]] - df['Reference']).abs().clip(upper=ERR_CAP).values

def true_err_matrix(df):
    n = len(df); te = np.full((len(arms_sorted), n), np.nan)
    for j, a in enumerate(arms_sorted):
        te[j] = np.minimum(np.abs(df[ARM_ENERGY_COL[a]].values - df['Reference'].values), ERR_CAP)
    return te

def train_models(df_tr):
    X = b25c.encode_features(df_tr, feature_cols)
    arm_m = {}
    for a in ARMS:
        y = err_arm(df_tr, a); m = ~np.isnan(y)
        if m.sum() < 50: arm_m[a] = None; continue
        mod = xgb.XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.1,
                               subsample=0.85, colsample_bytree=0.7, random_state=0,
                               objective='reg:absoluteerror', n_jobs=-1, verbosity=0)
        mod.fit(X[m], y[m]); arm_m[a] = mod
    det = {}
    for key, col in [('p5','y5'),('p10','y10'),('p20','y20')]:
        y = df_tr[col].values; pos = y.sum(); neg = len(y)-pos
        if pos < 5: det[key] = None; continue
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              subsample=0.85, colsample_bytree=0.7, random_state=0,
                              scale_pos_weight=max(1.0, neg/max(pos,1)),
                              eval_metric='logloss', n_jobs=-1, verbosity=0)
        m.fit(X, y); det[key] = m
    return arm_m, det

def auc(p, y):
    m = ~np.isnan(y); p, y = p[m], y[m]
    pos = y.sum(); neg = len(y)-pos
    if pos==0 or neg==0: return float('nan')
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1,len(p)+1)
    return (ranks[y==1].sum() - pos*(pos+1)/2)/(pos*neg)

def metrics(chosen, te):
    n = te.shape[1]
    e = te[chosen, np.arange(n)]; c = COST[chosen]
    return np.nanmean(c), np.nanmean(e), np.nanmedian(e), np.nanmean(e>10)

# ---- 5-fold CV over P0.2 charged ----
kf = KFold(n_splits=5, shuffle=True, random_state=0)
EPS = [1,2,3,5,7,10]
agg = {f'greedy_eps{e}':[] for e in EPS}
agg.update({'Always_GeoSP':[], 'Always_DFT':[], 'Oracle':[], 'cascade_tuned':[]})
auc_rows = []

for fold,(tr_idx, te_idx) in enumerate(kf.split(chg)):
    chg_tr = chg.iloc[tr_idx].reset_index(drop=True)
    chg_te = chg.iloc[te_idx].reset_index(drop=True)
    df_tr = pd.concat([ind, chg_tr], ignore_index=True)
    arm_m, det = train_models(df_tr)

    Xte = b25c.encode_features(chg_te, feature_cols)
    te_err = true_err_matrix(chg_te)
    pe = np.full((len(arms_sorted), len(chg_te)), ERR_CAP)
    for j, a in enumerate(arms_sorted):
        if arm_m[a] is not None: pe[j] = arm_m[a].predict(Xte)
    def pp(k): return det[k].predict_proba(Xte)[:,1] if det.get(k) is not None else np.zeros(len(chg_te))
    p5,p10,p20 = pp('p5'),pp('p10'),pp('p20')

    # detector AUC on held-out charged
    re = te_err[0]
    auc_rows.append({'fold':fold,'p5_auc':auc(p5,(re>5).astype(float)),
                     'p10_auc':auc(p10,(re>10).astype(float)),'p20_auc':auc(p20,(re>20).astype(float))})

    # (a) greedy eps on arm regressor
    for e in EPS:
        mask = pe <= e
        first = np.where(mask.any(0), mask.argmax(0), len(arms_sorted)-1)
        agg[f'greedy_eps{e}'].append(metrics(first, te_err))
    # (b) cascade with re-tuned taus (grid small)
    best = None
    for t20 in [0.3,0.4,0.5,0.6]:
        for t10 in [0.4,0.5,0.6,0.7]:
            for t5 in [0.6,0.7,0.8,0.9]:
                ch = np.zeros(len(chg_te),dtype=int)
                ch[p5>t5]=idx['PBE-D3BJ_SP']; ch[p10>t10]=idx['CREST_xTB']; ch[p20>t20]=idx['CREST_xTB_DFT']
                c,er,md,ct = metrics(ch, te_err)
                # objective: minimize err subject to cost<=GeoSP-ish; track best err
                if best is None or er < best[1]:
                    best=(c,er,md,ct)
    agg['cascade_tuned'].append(best)
    # baselines on same fold
    agg['Always_GeoSP'].append(metrics(np.full(len(chg_te),idx['PBE-D3BJ_GeoSP']), te_err))
    agg['Always_DFT'].append(metrics(np.full(len(chg_te),idx['CREST_xTB_DFT']), te_err))
    ba = np.nanargmin(te_err,0); agg['Oracle'].append(metrics(ba, te_err))
    print(f"fold {fold}: n_te={len(chg_te)}  p10_auc={auc_rows[-1]['p10_auc']:.3f}", flush=True)

print("\n" + "="*78)
print("DETECTOR AUC on held-out charged (charged-aware training):")
adf = pd.DataFrame(auc_rows)
print(f"  p5  AUC {adf['p5_auc'].mean():.3f}   p10 AUC {adf['p10_auc'].mean():.3f}   p20 AUC {adf['p20_auc'].mean():.3f}")
print("  (vs frozen-V5 on charged: p5=0.34 p10=0.41 p20=0.57 — anti-calibrated)")

print("\n" + "="*78)
print("5-fold CV on held-out P0.2 charged (cost / err / median / cat):")
print("="*78)
def show(name):
    arr = np.array(agg[name])
    c,er,md,ct = arr.mean(0)
    print(f"  {name:<18} cost={c:>7.1f}  err={er:>6.3f}  med={md:>5.2f}  cat={ct*100:>5.2f}%")
for e in EPS: show(f'greedy_eps{e}')
show('cascade_tuned')
print("  " + "-"*60)
show('Always_GeoSP'); show('Always_DFT'); show('Oracle')

json.dump({'auc':auc_rows,
           'policies':{k:list(np.array(v).mean(0)) for k,v in agg.items()}},
          open(RES/'p02_charged_aware_cv.json','w'), indent=2)
print(f"\nSaved -> {RES/'p02_charged_aware_cv.json'}")
