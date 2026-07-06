#!/usr/bin/env python3
"""
51 — P0.2 prospective evaluation: apply FROZEN V5 policy to 1,291 charged systems.

Load final V5 (trained on all 5,567 in-dist), extract predictions on P0.2 features,
apply the frozen decision rule, report cost/err/cat vs Always-* baselines.
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
RES = OUT/'results'; RES.mkdir(exist_ok=True)

ARMS = b25c.ARMS
ARM_ENERGY_COL = b25c.ARM_ENERGY_COL
ARM_TIME_COL = b25c.ARM_TIME_COL
ERR_CAP = 50.0

# --- Load frozen V5 policy ---
manifest = json.load(open(MDL/'manifest.json'))
feature_cols = manifest['feature_cols']
arm_costs = manifest['arm_costs']
tau5, tau10, tau20 = manifest['thresholds']['tau5'], manifest['thresholds']['tau10'], manifest['thresholds']['tau20']
CHARGE_ARM = manifest['routing']['charge_arm']
ARM5, ARM10, ARM20 = manifest['routing']['arm5'], manifest['routing']['arm10'], manifest['routing']['arm20']
arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a, 1e9))
arm_idx = {a:i for i,a in enumerate(arms_sorted)}

print("=" * 70)
print("FROZEN V5 POLICY (loaded from manifest, NOT retuned):")
print(f"  charge: |q|>=1 -> {CHARGE_ARM}")
print(f"  p20 > {tau20} -> {ARM20}")
print(f"  p10 > {tau10} -> {ARM10}")
print(f"  p5  > {tau5}  -> {ARM5}")
print(f"  else          -> RAPIDS")
print("=" * 70)

# --- Load arm regressors + cat detectors ---
arm_models = {}
for a in ARMS:
    p = MDL/f'arm_{a}.json'
    if p.exists():
        m = xgb.XGBRegressor(); m.load_model(str(p)); arm_models[a] = m
    else:
        arm_models[a] = None

cat_models = {}
for name in ['cat_p5','cat_p10','cat_p20']:
    p = MDL/f'{name}.json'
    if p.exists():
        m = xgb.XGBClassifier(); m.load_model(str(p)); cat_models[name] = m
    else:
        cat_models[name] = None

# --- Load P0.2 feature matrix ---
df = pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv', low_memory=False)
print(f"\nP0.2 feature matrix: {len(df)} rows × {df.shape[1]} cols")
print(f"  DES370K gold: {(df['reference_tier']=='gold').sum() if 'reference_tier' in df else 'n/a'}")
print(f"  IL174 silver: {(df['reference_tier']=='silver').sum() if 'reference_tier' in df else 'n/a'}")

# Charged systems — apply the |charge|>=1 rule
def _abs_charge(row):
    for c in ('oracle_complex_charge','complex_charge','monA_charge','monB_charge','Charge'):
        v = row.get(c) if c in row else np.nan
        if v is not None and not pd.isna(v):
            try:
                if abs(int(v)) >= 1: return abs(int(v))
            except: pass
    return 0
df['_abs_charge'] = df.apply(_abs_charge, axis=1)

# --- Predict ---
X = b25c.encode_features(df, feature_cols)
preds = np.full((len(arms_sorted), len(df)), ERR_CAP, dtype=np.float32)
for j, a in enumerate(arms_sorted):
    if arm_models[a] is not None:
        preds[j] = arm_models[a].predict(X)

def pp(m): return m.predict_proba(X)[:,1] if m is not None else np.zeros(len(df))
p5  = pp(cat_models['cat_p5'])
p10 = pp(cat_models['cat_p10'])
p20 = pp(cat_models['cat_p20'])

# --- Apply frozen policy ---
def apply_frozen_v5():
    n = len(df); abs_q = df['_abs_charge'].values
    chosen = np.full(n, arm_idx['RAPIDS'], dtype=np.int32)
    chosen[p5  > tau5]  = arm_idx[ARM5]
    chosen[p10 > tau10] = arm_idx[ARM10]
    chosen[p20 > tau20] = arm_idx[ARM20]
    chosen[abs_q > 0]   = arm_idx[CHARGE_ARM]
    return chosen

chosen_v5 = apply_frozen_v5()

# --- Metrics per system (accounting for missing arm data) ---
DEFAULT_WALLTIME = {a: arm_costs[a] for a in ARMS}  # fallback if arm_t is NaN

def metrics_for_policy(chosen, name):
    n = len(df)
    cost = np.full(n, np.nan)
    err = np.full(n, np.nan)
    for i in range(n):
        a = arms_sorted[chosen[i]]
        ecol = ARM_ENERGY_COL[a]
        tcol = ARM_TIME_COL[a]
        r = df[ecol].iloc[i]; ref = df['Reference'].iloc[i]
        if pd.notna(r) and pd.notna(ref):
            err[i] = min(abs(r - ref), ERR_CAP)
        t = df[tcol].iloc[i] if tcol in df else np.nan
        cost[i] = t if pd.notna(t) else DEFAULT_WALLTIME.get(a, np.nan)
    return {
        'policy': name,
        'n_eval': int((~np.isnan(err)).sum()),
        'mean_cost_s': float(np.nanmean(cost)),
        'mean_err_kcal': float(np.nanmean(err)),
        'median_err_kcal': float(np.nanmedian(err)),
        'catastrophic_rate': float(np.nanmean(err > 10)),
        'arm_picks': dict(zip(*np.unique([arms_sorted[c] for c in chosen], return_counts=True))),
    }

# --- Baselines ---
def always(arm, name):
    n = len(df); chosen = np.full(n, arm_idx[arm], dtype=np.int32)
    return metrics_for_policy(chosen, name)

def charge_rule():
    n = len(df); chosen = np.full(n, arm_idx['RAPIDS'], dtype=np.int32)
    chosen[df['_abs_charge'].values > 0] = arm_idx['PBE-D3BJ_GeoSP']
    return metrics_for_policy(chosen, 'Charge_rule')

results = [
    always('RAPIDS', 'Always_RAPIDS'),
    always('CREST_xTB', 'Always_CREST_xTB'),
    always('PBE-D3BJ_SP', 'Always_PBE-D3BJ_SP'),
    always('PBE-D3BJ_GeoSP', 'Always_GeoSP'),
    always('CREST_xTB_DFT', 'Always_CREST_xTB_DFT'),
    charge_rule(),
    metrics_for_policy(chosen_v5, 'FROZEN_V5'),
]

dfr = pd.DataFrame([{k:v for k,v in r.items() if k != 'arm_picks'} for r in results])
dfr.to_csv(RES/'p02_prospective_eval.csv', index=False)
print(f"\n{'='*90}")
print(dfr.to_string(index=False))
print(f"{'='*90}\n")

# --- Focused comparison V5 vs Always-GeoSP ---
v5 = next(r for r in results if r['policy']=='FROZEN_V5')
g  = next(r for r in results if r['policy']=='Always_GeoSP')
print("=== FROZEN V5 vs Always-GeoSP on P0.2 (prospective OOD) ===")
print(f"{'':<25}{'V5':>12}{'Always-GeoSP':>15}{'Δ':>12}")
print(f"{'cost (s)':<25}{v5['mean_cost_s']:>12.1f}{g['mean_cost_s']:>15.1f}{(1-v5['mean_cost_s']/g['mean_cost_s'])*100:>+11.1f}%")
print(f"{'err (kcal/mol)':<25}{v5['mean_err_kcal']:>12.3f}{g['mean_err_kcal']:>15.3f}{(v5['mean_err_kcal']-g['mean_err_kcal']):>+12.3f}")
print(f"{'cat (>10 kcal/mol)':<25}{v5['catastrophic_rate']*100:>11.2f}%{g['catastrophic_rate']*100:>14.2f}%{(v5['catastrophic_rate']-g['catastrophic_rate'])*100:>+11.2f}pp")

print(f"\n=== V5 arm-selection distribution on P0.2 ===")
for a, c in v5['arm_picks'].items():
    print(f"  {a:<20} {c:>4} / {v5['n_eval']} ({c/v5['n_eval']*100:.1f}%)")

# Also compare vs Always-CREST_xTB_DFT (strongest single arm on charged)
cd = next(r for r in results if r['policy']=='Always_CREST_xTB_DFT')
print("\n=== FROZEN V5 vs Always-CREST_xTB_DFT (strongest OOD baseline) ===")
print(f"{'':<25}{'V5':>12}{'Always-CREST-DFT':>18}{'Δ':>12}")
print(f"{'cost (s)':<25}{v5['mean_cost_s']:>12.1f}{cd['mean_cost_s']:>18.1f}{(1-v5['mean_cost_s']/cd['mean_cost_s'])*100:>+11.1f}%")
print(f"{'err (kcal/mol)':<25}{v5['mean_err_kcal']:>12.3f}{cd['mean_err_kcal']:>18.3f}{(v5['mean_err_kcal']-cd['mean_err_kcal']):>+12.3f}")
print(f"{'cat rate':<25}{v5['catastrophic_rate']*100:>11.2f}%{cd['catastrophic_rate']*100:>17.2f}%{(v5['catastrophic_rate']-cd['catastrophic_rate'])*100:>+11.2f}pp")

print(f"\nSaved → {RES/'p02_prospective_eval.csv'}")
