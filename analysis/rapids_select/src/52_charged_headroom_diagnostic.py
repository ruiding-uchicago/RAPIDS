#!/usr/bin/env python3
"""
52 — Charged headroom diagnostic: is there per-system signal to route on, or is
always-escalate genuinely near-optimal?

Answers:
 (A) On P0.2 charged, how often is each cheap arm (RAPIDS/SP/CREST_xTB) actually
     accurate? -> is there anything for a selector to gain by NOT always escalating?
 (B) Oracle best-arm ceiling on P0.2 (per-system min true err) -> the max a perfect
     selector could achieve, at what cost.
 (C) Oracle cost-aware ceiling: cheapest arm with true err <= tau -> realistic target.
 (D) Do the frozen p5/p10/p20 detectors have signal on charged? (calibration by decile)
 (E) What does V5 give if we REMOVE the charge->GeoSP short circuit and let the
     4-tier cascade actually route each charged system? (this is the honest 'selector')

No retraining. Uses frozen V5 models + P0.2 feature matrix.
Cost uses fixed in-dist per-arm walltime (arm_costs.json) since P0.2 lacks per-system times.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

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
print("Arms by cost:", [(a, round(arm_costs[a])) for a in arms_sorted])

df = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
df = df[df['Reference'].notna()].reset_index(drop=True)
ref = df['Reference'].values
n = len(df)
print(f"P0.2 systems: {n}\n")

# true error per arm (NaN where arm missing)
true_err = np.full((len(arms_sorted), n), np.nan)
for j, a in enumerate(arms_sorted):
    e = df[ARM_ENERGY_COL[a]].values
    true_err[j] = np.minimum(np.abs(e - ref), ERR_CAP)

# ---------- (A) how good is each cheap arm on charged ----------
print("="*70)
print("(A) Per-arm accuracy on P0.2 charged (true err distribution)")
print("="*70)
print(f"{'arm':<18}{'cov%':>6}{'mean':>8}{'median':>8}{'<2kc%':>7}{'<5kc%':>7}{'<10kc%':>8}{'cost':>7}")
for j, a in enumerate(arms_sorted):
    e = true_err[j]; m = ~np.isnan(e)
    print(f"{a:<18}{m.mean()*100:>6.1f}{np.nanmean(e):>8.2f}{np.nanmedian(e):>8.2f}"
          f"{np.nanmean(e[m]<2)*100:>7.1f}{np.nanmean(e[m]<5)*100:>7.1f}{np.nanmean(e[m]<10)*100:>8.1f}{arm_costs[a]:>7.0f}")

# ---------- (B) oracle best-arm ceiling ----------
print("\n" + "="*70)
print("(B) ORACLE best-arm ceiling (per-system pick min true err)")
print("="*70)
best_arm = np.nanargmin(true_err, axis=0)
oracle_err = true_err[best_arm, np.arange(n)]
oracle_cost = COST[best_arm]
print(f"  err  mean={np.nanmean(oracle_err):.3f}  median={np.nanmedian(oracle_err):.3f}  cat={np.nanmean(oracle_err>10)*100:.2f}%")
print(f"  cost mean={np.nanmean(oracle_cost):.1f}s")
print("  oracle arm distribution:")
for j, a in enumerate(arms_sorted):
    print(f"    {a:<18} {(best_arm==j).sum():>4} ({(best_arm==j).mean()*100:.1f}%)")

# ---------- (C) oracle cost-aware ceiling ----------
print("\n" + "="*70)
print("(C) ORACLE cost-aware: cheapest arm with true err <= tau")
print("="*70)
for tau in [1.0, 2.0, 3.0, 5.0]:
    chosen = np.full(n, len(arms_sorted)-1)  # default most expensive
    for i in range(n):
        for j in range(len(arms_sorted)):
            if not np.isnan(true_err[j, i]) and true_err[j, i] <= tau:
                chosen[i] = j; break
    ce = true_err[chosen, np.arange(n)]
    cc = COST[chosen]
    print(f"  tau={tau:>4.1f}: cost={np.nanmean(cc):>7.1f}s  err={np.nanmean(ce):>6.3f}  cat={np.nanmean(ce>10)*100:>5.2f}%"
          f"  | RAPIDS picked {np.mean(chosen==0)*100:>4.1f}%")

# ---------- load frozen models, predict ----------
arm_models = {}
for a in ARMS:
    p = MDL/f'arm_{a}.json'
    m = xgb.XGBRegressor(); m.load_model(str(p)); arm_models[a] = m
cat_models = {}
for name in ['cat_p5','cat_p10','cat_p20']:
    m = xgb.XGBClassifier(); m.load_model(str(MDL/f'{name}.json')); cat_models[name] = m

X = b25c.encode_features(df, feature_cols)
pred_err = np.full((len(arms_sorted), n), ERR_CAP, dtype=np.float64)
for j, a in enumerate(arms_sorted):
    pred_err[j] = arm_models[a].predict(X)
p5  = cat_models['cat_p5'].predict_proba(X)[:,1]
p10 = cat_models['cat_p10'].predict_proba(X)[:,1]
p20 = cat_models['cat_p20'].predict_proba(X)[:,1]

# ---------- (D) detector calibration on charged ----------
print("\n" + "="*70)
print("(D) Detector calibration on P0.2 charged")
print("="*70)
# ground-truth: does RAPIDS actually miss by >5/>10/>20?
rerr = true_err[0]  # RAPIDS is arms_sorted[0]
gt5 = (rerr > 5).astype(float); gt10 = (rerr > 10).astype(float); gt20 = (rerr > 20).astype(float)
from numpy import isnan
def auc(p, y):
    m = ~isnan(y)
    p, y = p[m], y[m]
    pos = y.sum(); neg = len(y)-pos
    if pos==0 or neg==0: return float('nan')
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p)+1)
    return (ranks[y==1].sum() - pos*(pos+1)/2) / (pos*neg)
print(f"  p5  AUC (vs RAPIDS err>5):  {auc(p5,  gt5):.3f}   base rate {np.nanmean(gt5)*100:.1f}%")
print(f"  p10 AUC (vs RAPIDS err>10): {auc(p10, gt10):.3f}   base rate {np.nanmean(gt10)*100:.1f}%")
print(f"  p20 AUC (vs RAPIDS err>20): {auc(p20, gt20):.3f}   base rate {np.nanmean(gt20)*100:.1f}%")
print("\n  p10 decile calibration (predicted prob -> actual RAPIDS cat rate):")
dec = pd.qcut(p10, 10, labels=False, duplicates='drop')
for d in sorted(pd.unique(dec)):
    msk = dec==d
    print(f"    decile {d}: p10∈[{p10[msk].min():.2f},{p10[msk].max():.2f}]  n={msk.sum():>4}  actual RAPIDS cat={np.nanmean(rerr[msk]>10)*100:>5.1f}%")

# ---------- (E) V5 WITHOUT charge short-circuit (honest cascade routing) ----------
print("\n" + "="*70)
print("(E) V5 cascade WITHOUT charge->GeoSP short circuit (per-system routing)")
print("="*70)
tau5, tau10, tau20 = manifest['thresholds']['tau5'], manifest['thresholds']['tau10'], manifest['thresholds']['tau20']
ARM5, ARM10, ARM20 = manifest['routing']['arm5'], manifest['routing']['arm10'], manifest['routing']['arm20']
idx = {a:i for i,a in enumerate(arms_sorted)}

def eval_pick(chosen, label):
    e = true_err[chosen, np.arange(n)]
    c = COST[chosen]
    picks = {arms_sorted[j]: int((chosen==j).sum()) for j in range(len(arms_sorted)) if (chosen==j).any()}
    print(f"  {label:<32} cost={np.nanmean(c):>7.1f}  err={np.nanmean(e):>6.3f}  med={np.nanmedian(e):>5.2f}  cat={np.nanmean(e>10)*100:>5.2f}%")
    return picks

# pure cascade (no charge rule)
casc = np.zeros(n, dtype=int)  # default RAPIDS=0
casc[p5  > tau5]  = idx[ARM5]
casc[p10 > tau10] = idx[ARM10]
casc[p20 > tau20] = idx[ARM20]
picks_casc = eval_pick(casc, "cascade (frozen taus, no charge)")

# reference points
eval_pick(np.zeros(n, dtype=int), "Always-RAPIDS")
eval_pick(np.full(n, idx['PBE-D3BJ_GeoSP']), "Always-GeoSP")
eval_pick(np.full(n, idx['CREST_xTB_DFT']), "Always-CREST_xTB_DFT")
print(f"\n  cascade arm picks: {picks_casc}")

# ---------- (F) arm-regressor greedy (cost-aware, no charge rule) ----------
print("\n" + "="*70)
print("(F) Arm-regressor greedy: cheapest arm with pred_err <= eps (no charge rule)")
print("="*70)
for eps in [1,2,3,5,7,10]:
    mask = pred_err <= eps
    first = np.where(mask.any(0), mask.argmax(0), len(arms_sorted)-1)
    e = true_err[first, np.arange(n)]; c = COST[first]
    print(f"  eps={eps:>2}: cost={np.nanmean(c):>7.1f}  err={np.nanmean(e):>6.3f}  cat={np.nanmean(e>10)*100:>5.2f}%  RAPIDS%={np.mean(first==0)*100:>4.1f}")

np.savez(RES/'p02_diagnostic_arrays.npz',
         true_err=true_err, pred_err=pred_err, p5=p5, p10=p10, p20=p20,
         cost=COST, arms=np.array(arms_sorted, dtype=object))
print(f"\nSaved arrays -> {RES/'p02_diagnostic_arrays.npz'}")
