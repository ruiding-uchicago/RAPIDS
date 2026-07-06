#!/usr/bin/env python3
"""
25c — RAPIDS-Select policy + baselines implementation.

Defines:
  POLICIES = {name: function(row) → arm_name}

Each policy takes a per-system Series of features (including oracle E for Oracle
baseline; features-only for others) and returns one of the 5 arm names.

Baselines:
  1. Oracle              — picks arm with smallest |E - Ref| (unattainable lower bound)
  2. Always-RAPIDS
  3. Always-CREST_xTB    (cheap CREST option)
  4. Always-PBE-D3BJ_GeoSP
  5. Always-CREST_xTB_DFT
  6. Charge-rule         — |charge|>=1 → GeoSP, else RAPIDS
  7. Guard-rule          — RAPIDS_status==flagged → GeoSP, else RAPIDS
  8. Random              — uniform arm
  9. RAPIDS-Select-ε=2   — our trained selector with tolerance 2 kcal/mol
 10. RAPIDS-Select-ε=1   — tighter tolerance

Note: bandit-style strategies (UCB, LinUCB, Meta-GBM, MFMI-Greedy) live in
offline_replay/sequential_bandit.py and need stateful sequence access — they're
not single-system policies, so they're handled separately in 25d when we run
the full sequential evaluation. Here we only encode stateless single-system
policies.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = OUT_ROOT / 'models' / 'rapids_select_v1'

ARMS = ['RAPIDS', 'CREST_xTB', 'PBE-D3BJ_SP', 'PBE-D3BJ_GeoSP', 'CREST_xTB_DFT']
ARM_ENERGY_COL = {
    'RAPIDS': 'RAPIDS',
    'CREST_xTB': 'CREST_xTB',
    'PBE-D3BJ_SP': 'PBE-D3BJ_SP',
    'PBE-D3BJ_GeoSP': 'PBE-D3BJ_GeoSP',
    'CREST_xTB_DFT': 'CREST_xTB_DFT',
}
ARM_TIME_COL = {a: f'{ARM_ENERGY_COL[a]}_time' for a in ARMS}


# -------- load trained selector --------
def load_selector():
    """Load the 5 per-arm error predictors + feature column list + cost table."""
    import xgboost as xgb
    feature_cols = json.load(open(MODEL_DIR / 'feature_columns.json'))
    arm_costs = json.load(open(MODEL_DIR / 'arm_costs.json'))
    models = {}
    for a in ARMS:
        m = xgb.XGBRegressor()
        m.load_model(str(MODEL_DIR / f'{a}_error_predictor.json'))
        models[a] = m
    return models, feature_cols, arm_costs


def encode_features(df, feature_cols):
    """Same encoding used in 25b: factorize object cols, replace inf, keep NaN."""
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c not in df.columns:
            X[c] = np.nan; continue
        s = df[c]
        if s.dtype == 'object':
            codes, _ = pd.factorize(s, sort=True)
            codes = codes.astype(float)
            codes[s.isna().values] = np.nan
            X[c] = codes
        else:
            X[c] = s.astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.values


# -------- stateless policies --------

def policy_oracle(row):
    errs = {a: abs(row[ARM_ENERGY_COL[a]] - row['Reference'])
            for a in ARMS if pd.notna(row.get(ARM_ENERGY_COL[a]))}
    if not errs: return 'RAPIDS'
    return min(errs, key=errs.get)

def policy_always(arm):
    def fn(row): return arm
    fn.__name__ = f'Always_{arm}'
    return fn

def policy_charge_rule(row):
    # use complex_charge (from per-bench CSV, may be NaN for neutral)
    q = row.get('oracle_complex_charge') or row.get('Charge') or 0
    try: q = abs(int(q))
    except: q = 0
    return 'PBE-D3BJ_GeoSP' if q >= 1 else 'RAPIDS'

def policy_guard_rule(row):
    status = row.get('RAPIDS_status', '')
    if isinstance(status, str) and status.lower() == 'flagged':
        return 'PBE-D3BJ_GeoSP'
    return 'RAPIDS'

def policy_random(rng):
    def fn(row): return ARMS[rng.integers(len(ARMS))]
    fn.__name__ = 'Random'
    return fn


# -------- RAPIDS-Select trained policy --------
def make_rapids_select(epsilon=2.0):
    """Build the RAPIDS-Select policy. epsilon = error tolerance in kcal/mol."""
    models, feature_cols, arm_costs = load_selector()
    arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a, 1e9))
    def fn(row, _x_cache=[None]):
        # vectorized encode
        df_one = pd.DataFrame([row])
        X = encode_features(df_one, feature_cols)
        # predict |err| for each arm
        preds = {a: float(models[a].predict(X)[0]) for a in ARMS}
        # pick cheapest arm with predicted error <= epsilon
        for a in arms_sorted:
            if preds[a] <= epsilon:
                return a
        # nothing satisfies → escalate to highest-tier
        return arms_sorted[-1]
    fn.__name__ = f'RAPIDS_Select_eps_{epsilon:.1f}'
    return fn


# -------- batch-vectorized selector for speed --------
def make_rapids_select_batch(epsilon=2.0):
    """Vectorized version that picks arm for a whole DataFrame at once."""
    models, feature_cols, arm_costs = load_selector()
    arms_sorted = sorted(ARMS, key=lambda a: arm_costs.get(a, 1e9))
    def fn(df):
        X = encode_features(df, feature_cols)
        preds = {a: models[a].predict(X) for a in ARMS}
        chosen = []
        for i in range(len(df)):
            arm_pick = arms_sorted[-1]
            for a in arms_sorted:
                if preds[a][i] <= epsilon:
                    arm_pick = a; break
            chosen.append(arm_pick)
        return chosen
    fn.__name__ = f'RAPIDS_Select_eps_{epsilon:.1f}_batch'
    return fn


# -------- evaluate policy on a dataframe → per-system (arm, cost, error) --------
def evaluate_policy(df, policy_fn, batch_mode=False, rng=None):
    """Return per-system DataFrame: system_id, chosen_arm, cost_s, abs_err_kcal."""
    if batch_mode:
        chosen = policy_fn(df)
    else:
        chosen = [policy_fn(df.iloc[i]) for i in range(len(df))]
    out = pd.DataFrame({
        'benchmark': df['benchmark'].values,
        'system_id': df['system_id'].values,
        'chosen_arm': chosen,
    })
    # lookup per-arm energy + time
    e_pick = np.array([df.iloc[i].get(ARM_ENERGY_COL[chosen[i]], np.nan) for i in range(len(df))])
    t_pick = np.array([df.iloc[i].get(ARM_TIME_COL[chosen[i]], np.nan) for i in range(len(df))])
    ref = df['Reference'].values
    out['arm_energy_kcal'] = e_pick
    out['cost_s'] = t_pick
    out['abs_err_kcal'] = np.minimum(np.abs(e_pick - ref), 50.0)  # cap at 50 kcal/mol
    return out


if __name__ == '__main__':
    # quick smoke test: load + run on a small sample
    df = pd.read_csv(OUT_ROOT / 'data' / 'selector_feature_matrix.csv', low_memory=False)
    df = df[df['Reference'].notna()].head(200).reset_index(drop=True)
    print(f"Smoke test on {len(df)} systems")
    rng = np.random.default_rng(0)
    policies = {
        'Oracle': (policy_oracle, False),
        'Always_RAPIDS': (policy_always('RAPIDS'), False),
        'Always_PBE-D3BJ_GeoSP': (policy_always('PBE-D3BJ_GeoSP'), False),
        'Always_CREST_xTB_DFT': (policy_always('CREST_xTB_DFT'), False),
        'Charge_rule': (policy_charge_rule, False),
        'Guard_rule': (policy_guard_rule, False),
        'Random': (policy_random(rng), False),
        'RAPIDS-Select-eps2': (make_rapids_select_batch(2.0), True),
        'RAPIDS-Select-eps1': (make_rapids_select_batch(1.0), True),
    }
    for name, (fn, batch) in policies.items():
        r = evaluate_policy(df, fn, batch_mode=batch, rng=rng)
        print(f"  {name:<25}  mean_cost={np.nanmean(r['cost_s']):.0f}s  "
              f"mean_err={np.nanmean(r['abs_err_kcal']):.3f}  "
              f"median_err={np.nanmedian(r['abs_err_kcal']):.3f}  "
              f"P50_err_capped@50={float(np.nanpercentile(r['abs_err_kcal'], 50)):.3f}")
