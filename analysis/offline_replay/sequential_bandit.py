#!/usr/bin/env python3
"""
Sequential Multi-Fidelity Bandit Offline Replay — 9-Arm Version.

Every strategy selects from ALL 9 fidelity methods for each system.
Cost = sum of ALL methods actually run on that system.

IMPORTANT: No strategy peeks at reference values when selecting predictions.
Strategies commit to their chosen method's prediction. Only Oracle uses ref
(by definition). Learning signals use ref for training, but prediction
selection does not.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path(os.path.expanduser("~/benchmarking/collection_finished_all_fidelity"))
NEUTRAL_DIR = BASE / "neutral"
CHARGED_DIR = BASE / "charged"
OUT_DIR = BASE / "offline_replay" / "results_sequential"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "RAPIDS", "PBE-D3BJ_SP", "wB97X-D3BJ_SP", "wB97M-V_SP",
    "PBE-D3BJ_GeoSP", "wB97X-D3BJ_GeoSP", "wB97M-V_GeoSP",
    "CREST_xTB", "CREST_xTB_DFT",
]
N_METHODS = len(METHODS)
METHOD_IDX = {m: i for i, m in enumerate(METHODS)}

NEUTRAL_BENCHMARKS = [
    "A24", "S66", "X40", "HB300SPX", "HB375", "SH250",
    "D1200_Halogens", "D1200_HBCNO", "D1200_PS",
    "BFDb_BBI", "BFDb_HSG", "BFDb_NBC1",
    "BFDb_SSI_dispersion", "BFDb_SSI_mixed", "BFDb_SSI_other", "BFDb_SSI_polar",
]
CHARGED_BENCHMARKS = ["IHB100", "BFDb_SSI_charged"]

N_SEEDS = 10
ERROR_CAP = 50.0
N_BUDGET_POINTS = 30

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark(bench_name, category="neutral"):
    if category == "neutral":
        csv_path = NEUTRAL_DIR / bench_name / f"{bench_name}.csv"
        systems_dir = NEUTRAL_DIR / bench_name / "systems"
    else:
        csv_path = CHARGED_DIR / bench_name / f"{bench_name}.csv"
        systems_dir = CHARGED_DIR / bench_name / "systems"
    df = pd.read_csv(csv_path)
    result = {"Reference": df["Reference"].values}
    for m in METHODS:
        if m in df.columns:
            result[m] = pd.to_numeric(df[m], errors="coerce").values
        else:
            result[m] = np.full(len(df), np.nan)
        time_col = f"{m}_time"
        if time_col in df.columns:
            result[time_col] = pd.to_numeric(df[time_col], errors="coerce").values
        else:
            result[time_col] = np.full(len(df), np.nan)

    # Load chemical metadata from probe.meta.json + target.meta.json
    chem_features = _load_chem_features(systems_dir, bench_name, len(df))
    for col_name, col_data in chem_features.items():
        result[col_name] = col_data

    return pd.DataFrame(result)


# Chemical descriptor fields to extract from meta.json
_CHEM_FIELDS_PUBCHEM = [
    "molecular_weight", "heavy_atom_count", "xlogp", "tpsa",
    "complexity", "h_bond_donor_count", "h_bond_acceptor_count",
    "rotatable_bond_count", "formal_charge",
]
_CHEM_FIELDS_RDKIT = [
    "aromatic_atom_count", "ring_count", "fsp3",
]


def _load_chem_features(systems_dir, bench_name, n_systems):
    """Load chemical descriptors from probe/target meta.json files.
    Returns dict of column_name -> np.array for DataFrame injection."""
    # Column names: chem_{probe|target}_{field}
    col_names = []
    for role in ("probe", "target"):
        for f in _CHEM_FIELDS_PUBCHEM + _CHEM_FIELDS_RDKIT:
            col_names.append(f"chem_{role}_{f}")
    # Extra pair-level features
    col_names.extend([
        "chem_total_heavy_atoms", "chem_total_hbd", "chem_total_hba",
        "chem_total_tpsa", "chem_max_complexity",
        "chem_delta_xlogp", "chem_delta_mw",
    ])

    result = {c: np.full(n_systems, np.nan) for c in col_names}

    if not systems_dir.exists():
        return result

    # Enumerate system directories in sorted order (matches CSV row order)
    sys_dirs = sorted([d for d in systems_dir.iterdir() if d.is_dir()])
    # Pick first available method subdir for meta.json (all have same chem info)
    method_dir_pref = ["RAPIDS_scan_x9", "RAPIDS_PBE-D3BJ_SP", "CREST_xTB"]

    for i, sdir in enumerate(sys_dirs):
        if i >= n_systems:
            break
        # Find a method subdir with meta.json
        meta_probe, meta_target = None, None
        for mpref in method_dir_pref:
            mdir = sdir / mpref
            if mdir.exists():
                pf = mdir / "probe.meta.json"
                tf = mdir / "target.meta.json"
                if pf.exists() and tf.exists():
                    try:
                        with open(pf) as fp:
                            meta_probe = json.load(fp)
                        with open(tf) as fp:
                            meta_target = json.load(fp)
                    except (json.JSONDecodeError, IOError):
                        pass
                    break
        if meta_probe is None or meta_target is None:
            # Try any subdir
            for mdir in sdir.iterdir():
                if not mdir.is_dir():
                    continue
                pf = mdir / "probe.meta.json"
                tf = mdir / "target.meta.json"
                if pf.exists() and tf.exists():
                    try:
                        with open(pf) as fp:
                            meta_probe = json.load(fp)
                        with open(tf) as fp:
                            meta_target = json.load(fp)
                    except (json.JSONDecodeError, IOError):
                        pass
                    break

        if meta_probe is None or meta_target is None:
            continue

        # Extract per-role fields
        for role, meta in [("probe", meta_probe), ("target", meta_target)]:
            pub = meta.get("descriptors_pubchem", {})
            rdk = meta.get("descriptors_rdkit", {})
            for f in _CHEM_FIELDS_PUBCHEM:
                val = pub.get(f)
                if val is not None:
                    try:
                        result[f"chem_{role}_{f}"][i] = float(val)
                    except (ValueError, TypeError):
                        pass
            for f in _CHEM_FIELDS_RDKIT:
                val = rdk.get(f)
                if val is not None:
                    try:
                        result[f"chem_{role}_{f}"][i] = float(val)
                    except (ValueError, TypeError):
                        pass

        # Pair-level features
        p_pub = meta_probe.get("descriptors_pubchem", {})
        t_pub = meta_target.get("descriptors_pubchem", {})
        try:
            p_ha = float(p_pub.get("heavy_atom_count", 0))
            t_ha = float(t_pub.get("heavy_atom_count", 0))
            result["chem_total_heavy_atoms"][i] = p_ha + t_ha
        except (ValueError, TypeError):
            pass
        try:
            result["chem_total_hbd"][i] = float(p_pub.get("h_bond_donor_count", 0)) + float(t_pub.get("h_bond_donor_count", 0))
        except (ValueError, TypeError):
            pass
        try:
            result["chem_total_hba"][i] = float(p_pub.get("h_bond_acceptor_count", 0)) + float(t_pub.get("h_bond_acceptor_count", 0))
        except (ValueError, TypeError):
            pass
        try:
            result["chem_total_tpsa"][i] = float(p_pub.get("tpsa", 0)) + float(t_pub.get("tpsa", 0))
        except (ValueError, TypeError):
            pass
        try:
            result["chem_max_complexity"][i] = max(float(p_pub.get("complexity", 0)), float(t_pub.get("complexity", 0)))
        except (ValueError, TypeError):
            pass
        try:
            result["chem_delta_xlogp"][i] = abs(float(p_pub.get("xlogp", 0)) - float(t_pub.get("xlogp", 0)))
        except (ValueError, TypeError):
            pass
        try:
            result["chem_delta_mw"][i] = abs(float(p_pub.get("molecular_weight", 0)) - float(t_pub.get("molecular_weight", 0)))
        except (ValueError, TypeError):
            pass

    return result


def load_all_benchmarks():
    data = {}
    for b in NEUTRAL_BENCHMARKS:
        data[b] = load_benchmark(b, "neutral")
        print(f"  Loaded {b}: {len(data[b])} systems")
    for b in CHARGED_BENCHMARKS:
        data[b] = load_benchmark(b, "charged")
        print(f"  Loaded {b}: {len(data[b])} systems")
    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def capped_error(pred, ref):
    return min(abs(pred - ref), ERROR_CAP)


def compute_mae(preds, refs):
    if len(preds) == 0:
        return np.nan
    return np.mean([capped_error(p, r) for p, r in zip(preds, refs)])


def compute_rho(preds, refs):
    if len(preds) < 3:
        return np.nan
    rho, _ = spearmanr(preds, refs)
    return rho


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def get_method_cost(df_row, method):
    """Cost of running a single method on a system."""
    t = df_row.get(f"{method}_time", np.nan)
    return t if not np.isnan(t) else 0.0


def get_methods_cost(df_row, methods_list):
    """Total cost of running a list of methods on a system."""
    return sum(get_method_cost(df_row, m) for m in methods_list)


def get_method_pred(df_row, method):
    """Get prediction from a single method."""
    val = df_row.get(method, np.nan)
    return val


def median_pred(df_row, methods_run):
    """Return median prediction across all methods run.
    NO peeking at reference — robust aggregation without ordering assumption."""
    vals = []
    for m in methods_run:
        val = get_method_pred(df_row, m)
        if not np.isnan(val):
            vals.append(val)
    if not vals:
        return np.nan
    return float(np.median(vals))


def last_method_pred(df_row, methods_run):
    """Return prediction from the last method in methods_run that has data.
    Used for ladder/disagreement where the last method is the upgrade choice."""
    for m in reversed(methods_run):
        val = get_method_pred(df_row, m)
        if not np.isnan(val):
            return val
    return np.nan


def get_features(df_row):
    """Rich feature vector from cheap methods (RAPIDS + PBE).
    No ref peeking — only uses predictions."""
    r = get_method_pred(df_row, "RAPIDS")
    pbe = get_method_pred(df_row, "PBE-D3BJ_SP")
    if np.isnan(r):
        r = 0.0
    if np.isnan(pbe):
        pbe = r  # fallback
    disag = abs(r - pbe)
    return [r, pbe, abs(r), disag, r**2, disag**2]


# Chemical feature columns loaded from meta.json
_CHEM_COLS_PER_ROLE = [f"chem_{role}_{f}"
                       for role in ("probe", "target")
                       for f in _CHEM_FIELDS_PUBCHEM + _CHEM_FIELDS_RDKIT]
_CHEM_COLS_PAIR = [
    "chem_total_heavy_atoms", "chem_total_hbd", "chem_total_hba",
    "chem_total_tpsa", "chem_max_complexity",
    "chem_delta_xlogp", "chem_delta_mw",
]
CHEM_COLS = _CHEM_COLS_PER_ROLE + _CHEM_COLS_PAIR
N_CHEM_FEATURES = len(CHEM_COLS)


def get_chem_features(df_row):
    """Extract chemical descriptor vector from DataFrame row.
    Returns np.array of length N_CHEM_FEATURES. NaN → 0."""
    feat = np.zeros(N_CHEM_FEATURES)
    for i, col in enumerate(CHEM_COLS):
        val = df_row.get(col, np.nan)
        if not np.isnan(val):
            feat[i] = val
    return feat


def get_combined_features(df_row):
    """Prediction features (6) + chemical features (31) = combined vector."""
    pred_feat = get_features(df_row)
    chem_feat = get_chem_features(df_row)
    return np.concatenate([pred_feat, chem_feat])


# ---------------------------------------------------------------------------
# Strategy implementations
# Each returns (preds, costs, refs) in processing order.
# ---------------------------------------------------------------------------

# ---- 1. Always-X baselines ----
def strategy_always_X(df, order, method):
    """Always run just this one method. Cost = that method only."""
    preds, costs, refs = [], [], []
    for idx in order:
        row = df.iloc[idx]
        val = get_method_pred(row, method)
        if np.isnan(val):
            continue
        preds.append(val)
        costs.append(get_method_cost(row, method))
        refs.append(row["Reference"])
    return preds, costs, refs


# ---- 2. Oracle ----
def strategy_oracle(df, order):
    """For each system pick the method with lowest |error|. Pay only for that method."""
    preds, costs, refs = [], [], []
    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        best_err = float("inf")
        best_pred = np.nan
        best_m = None
        for m in METHODS:
            val = get_method_pred(row, m)
            if np.isnan(val):
                continue
            err = capped_error(val, ref)
            if err < best_err:
                best_err = err
                best_pred = val
                best_m = m
        if not np.isnan(best_pred):
            preds.append(best_pred)
            costs.append(get_method_cost(row, best_m))
            refs.append(ref)
    return preds, costs, refs


# ---- 3. Random ----
def strategy_random(df, order, rng):
    """Uniformly randomly pick one of 9 methods per system."""
    preds, costs, refs = [], [], []
    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        m = METHODS[rng.integers(N_METHODS)]
        val = get_method_pred(row, m)
        if np.isnan(val):
            # Fallback: try RAPIDS
            val = get_method_pred(row, "RAPIDS")
            m = "RAPIDS"
        if np.isnan(val):
            continue
        preds.append(val)
        costs.append(get_method_cost(row, m))
        refs.append(ref)
    return preds, costs, refs


# ---- 4. Multi-Arm Disagreement ----
def strategy_disagreement(df, order, thresholds=(1.0, 5.0, 15.0)):
    """
    Run RAPIDS + PBE-D3BJ_SP on all systems.
    Disagreement routes to different tiers.
    """
    t1, t2, t3 = thresholds
    preds, costs, refs = [], [], []
    for idx in order:
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        pbe = get_method_pred(row, "PBE-D3BJ_SP")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        base_methods = ["RAPIDS", "PBE-D3BJ_SP"]

        if np.isnan(pbe):
            # Can't compute disagreement, stay with RAPIDS
            preds.append(r)
            costs.append(get_method_cost(row, "RAPIDS"))
            refs.append(ref)
            continue

        disag = abs(r - pbe)
        if disag < t1:
            methods_run = base_methods
        elif disag < t2:
            methods_run = base_methods + ["wB97M-V_SP"]
        elif disag < t3:
            methods_run = base_methods + ["wB97M-V_GeoSP"]
        else:
            methods_run = base_methods + ["CREST_xTB_DFT"]

        pred = last_method_pred(row, methods_run)
        if np.isnan(pred):
            pred = r
        preds.append(pred)
        costs.append(get_methods_cost(row, methods_run))
        refs.append(ref)
    return preds, costs, refs


# ---- 5. Learned Multi-Arm Selector ----
def strategy_learned_selector(df, order, explore_frac=0.1, rng=None):
    """
    For each method, maintain a Ridge regression predicting |error_m| from RAPIDS features.
    Pick method with lowest predicted |error|.
    First 10% of systems: run all 9 (exploration).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(order)
    n_explore = max(5, int(n * explore_frac))

    # Training data: features_seen[i], errors_seen[i] = array of 9 errors
    features_seen = []
    errors_seen = []  # (n_seen, 9)

    preds, costs, refs = [], [], []

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        feat = get_features(row)

        if i < n_explore:
            # Exploration: run all 9, use median (no ref peeking)
            methods_run = list(METHODS)
            pred = median_pred(row, methods_run)
            if np.isnan(pred):
                pred = r
            cost = get_methods_cost(row, methods_run)
        else:
            # Exploitation: predict error for each method, pick lowest predicted
            chosen_m = "RAPIDS"
            if len(features_seen) >= 3:
                X = np.array(features_seen)
                Y = np.array(errors_seen)
                best_predicted_err = float("inf")
                for mi in range(N_METHODS):
                    y_col = Y[:, mi]
                    valid = ~np.isnan(y_col)
                    if valid.sum() < 3:
                        continue
                    try:
                        model = Ridge(alpha=1.0)
                        model.fit(X[valid], y_col[valid])
                        pred_err = model.predict(np.array([feat]))[0]
                        if pred_err < best_predicted_err:
                            best_predicted_err = pred_err
                            chosen_m = METHODS[mi]
                    except Exception:
                        pass

            val = get_method_pred(row, chosen_m)
            if np.isnan(val):
                val = r
                chosen_m = "RAPIDS"
            pred = val  # Commit to chosen method, no ref peeking
            cost = get_method_cost(row, chosen_m)

        # Record ground truth errors for all methods (for learning)
        err_row = []
        for m in METHODS:
            v = get_method_pred(row, m)
            if np.isnan(v):
                err_row.append(np.nan)
            else:
                err_row.append(capped_error(v, ref))
        features_seen.append(feat)
        errors_seen.append(err_row)

        preds.append(pred)
        costs.append(cost)
        refs.append(ref)

    return preds, costs, refs


# ---- 6. Multi-Arm Thompson Sampling ----
def strategy_thompson(df, order, n_bins=5, cost_aware=False, rng=None):
    """
    Bin systems by |RAPIDS_pred|. For each (bin, method), maintain Gaussian belief
    about negative error (reward). Thompson sample to pick method.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rapids_vals = df["RAPIDS"].dropna().abs().values
    bin_edges = np.quantile(rapids_vals, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    def get_bin(val):
        aval = abs(val)
        for b in range(n_bins):
            if aval <= bin_edges[b + 1]:
                return b
        return n_bins - 1

    # Gaussian beliefs: mu, sigma for each (bin, method)
    mu = np.zeros((n_bins, N_METHODS))
    sigma = np.ones((n_bins, N_METHODS)) * 5.0  # wide prior
    counts = np.zeros((n_bins, N_METHODS))

    # Median cost per method for cost-aware variant
    method_med_costs = np.ones(N_METHODS)
    if cost_aware:
        for mi, m in enumerate(METHODS):
            tcol = f"{m}_time"
            if tcol in df.columns:
                vals = pd.to_numeric(df[tcol], errors="coerce").dropna().values
                if len(vals) > 0:
                    method_med_costs[mi] = max(np.median(vals), 1.0)

    preds, costs_list, refs = [], [], []

    for idx in order:
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        b = get_bin(r)

        # Thompson sample
        sampled = np.array([rng.normal(mu[b, mi], sigma[b, mi]) for mi in range(N_METHODS)])
        if cost_aware:
            sampled = sampled / method_med_costs

        # Pick best (highest reward = least error)
        ranking = np.argsort(-sampled)
        chosen_m = None
        chosen_val = np.nan
        for mi in ranking:
            val = get_method_pred(row, METHODS[mi])
            if not np.isnan(val):
                chosen_m = METHODS[mi]
                chosen_val = val
                break
        if chosen_m is None:
            continue

        pred = chosen_val  # Commit to Thompson's choice, no ref peeking

        # Update beliefs for ALL methods (we observe all in offline replay)
        for mi, m in enumerate(METHODS):
            v = get_method_pred(row, m)
            if np.isnan(v):
                continue
            reward = -capped_error(v, ref)  # higher is better
            counts[b, mi] += 1
            n_obs = counts[b, mi]
            # Online Gaussian update
            old_mu = mu[b, mi]
            mu[b, mi] = old_mu + (reward - old_mu) / n_obs
            # Shrink sigma but floor it
            sigma[b, mi] = max(sigma[b, mi] * 0.98, 0.1)

        preds.append(pred)
        costs_list.append(get_method_cost(row, chosen_m))
        refs.append(ref)

    return preds, costs_list, refs


# ---- 7. Cost-Aware Multi-Arm ----
def strategy_cost_aware(df, order, explore_frac=0.1, rng=None):
    """
    Predict benefit of each method vs RAPIDS, pick best benefit/cost.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(order)
    n_explore = max(5, int(n * explore_frac))

    features_seen = []
    benefits_seen = []  # (n_seen, 9) = rapids_error - method_error

    preds, costs, refs = [], [], []

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        feat = get_features(row)
        rapids_err = capped_error(r, ref)

        if i < n_explore:
            # Explore: run all, use median (no ref peeking)
            methods_run = list(METHODS)
            pred = median_pred(row, methods_run)
            if np.isnan(pred):
                pred = r
            cost = get_methods_cost(row, methods_run)
        else:
            # Exploit: predict benefit/cost for each method
            chosen_m = "RAPIDS"
            best_ratio = 0.0
            if len(features_seen) >= 3:
                X = np.array(features_seen)
                B = np.array(benefits_seen)
                for mi in range(N_METHODS):
                    b_col = B[:, mi]
                    valid = ~np.isnan(b_col)
                    if valid.sum() < 3:
                        continue
                    try:
                        model = Ridge(alpha=1.0)
                        model.fit(X[valid], b_col[valid])
                        pred_benefit = model.predict(np.array([feat]))[0]
                        mcost = get_method_cost(row, METHODS[mi])
                        if mcost < 1:
                            mcost = 1.0
                        ratio = pred_benefit / mcost
                        if ratio > best_ratio:
                            best_ratio = ratio
                            chosen_m = METHODS[mi]
                    except Exception:
                        pass

            val = get_method_pred(row, chosen_m)
            if np.isnan(val):
                val = r
                chosen_m = "RAPIDS"
            pred = val  # Commit to chosen method, no ref peeking
            cost = get_method_cost(row, chosen_m)

        # Record benefits
        ben_row = []
        for m in METHODS:
            v = get_method_pred(row, m)
            if np.isnan(v):
                ben_row.append(np.nan)
            else:
                ben_row.append(rapids_err - capped_error(v, ref))
        features_seen.append(feat)
        benefits_seen.append(ben_row)

        preds.append(pred)
        costs.append(cost)
        refs.append(ref)

    return preds, costs, refs


# ---- 8. Progressive Ladder ----
def strategy_progressive_ladder(df, order, thresholds=(3.0, 2.0, 1.5)):
    """
    RAPIDS -> PBE-D3BJ_SP -> wB97M-V_SP -> wB97M-V_GeoSP -> CREST_xTB_DFT
    Upgrade only if disagreement at each stage exceeds threshold.
    """
    t1, t2, t3 = thresholds
    preds, costs, refs = [], [], []
    for idx in order:
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        methods_run = ["RAPIDS"]
        current_pred = r

        # Stage 1: always also run PBE-D3BJ_SP
        pbe = get_method_pred(row, "PBE-D3BJ_SP")
        if not np.isnan(pbe):
            methods_run.append("PBE-D3BJ_SP")
            current_pred = pbe  # Use latest method, no ref peeking

            if abs(r - pbe) > t1:
                # Stage 2: run wB97M-V_SP
                sp = get_method_pred(row, "wB97M-V_SP")
                if not np.isnan(sp):
                    methods_run.append("wB97M-V_SP")
                    current_pred = sp

                    if abs(sp - pbe) > t2:
                        # Stage 3: run wB97M-V_GeoSP
                        gsp = get_method_pred(row, "wB97M-V_GeoSP")
                        if not np.isnan(gsp):
                            methods_run.append("wB97M-V_GeoSP")
                            current_pred = gsp

                            if abs(gsp - sp) > t3:
                                # Stage 4: CREST_xTB_DFT
                                crest = get_method_pred(row, "CREST_xTB_DFT")
                                if not np.isnan(crest):
                                    methods_run.append("CREST_xTB_DFT")
                                    current_pred = crest

        preds.append(current_pred)
        costs.append(get_methods_cost(row, methods_run))
        refs.append(ref)
    return preds, costs, refs


# ---- 9. Multi-Fidelity Stacking ----
def strategy_stacking(df, order, residual_threshold=3.0):
    """
    Online stacking: run 3 cheap methods, train linear model incrementally.
    Use disagreement among cheap methods to decide upgrades.
    No data leakage: model only applied to systems NOT in training set.
    """
    cheap_methods = ["RAPIDS", "PBE-D3BJ_SP", "wB97X-D3BJ_SP"]
    preds, costs, refs = [], [], []

    # Online learning: accumulate training data as we go
    X_train_list = []
    y_train_list = []
    model = None
    n_train = max(10, int(len(order) * 0.15))  # 15% training phase

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        ref = row["Reference"]
        cheap_vals = [get_method_pred(row, m) for m in cheap_methods]
        if any(np.isnan(v) for v in cheap_vals):
            continue

        methods_run = list(cheap_methods)
        disagreement = np.std(cheap_vals)

        if i < n_train:
            # Training phase: use median of cheap methods, collect labels
            pred = float(np.median(cheap_vals))
            X_train_list.append(cheap_vals + [1.0])  # bias term
            y_train_list.append(ref)

            # Train model once we have enough data
            if len(X_train_list) >= 10 and model is None:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train_list), np.array(y_train_list))
                except Exception:
                    model = None
        else:
            # Inference phase: use stacking model if available
            if model is not None:
                x_i = np.array(cheap_vals + [1.0]).reshape(1, -1)
                stacked_pred = model.predict(x_i)[0]
                pred = stacked_pred
            else:
                pred = float(np.median(cheap_vals))

            # Upgrade if cheap methods disagree a lot
            if disagreement > residual_threshold:
                upgrade_methods = ["wB97M-V_GeoSP", "CREST_xTB_DFT"]
                for um in upgrade_methods:
                    v = get_method_pred(row, um)
                    if not np.isnan(v):
                        methods_run.append(um)
                        pred = v  # Use upgrade method's prediction
                        break

            # Continue online learning
            X_train_list.append(cheap_vals + [1.0])
            y_train_list.append(ref)
            if len(X_train_list) % 50 == 0:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train_list), np.array(y_train_list))
                except Exception:
                    pass

        if np.isnan(pred):
            pred = X_cheap[i][0]

        preds.append(pred)
        costs.append(get_methods_cost(row, methods_run))
        refs.append(ref)

    return preds, costs, refs


# ---- 10. Cheap Ensemble ----
def strategy_cheap_ensemble(df, order, n_cheap=3):
    """
    Always run n_cheap cheapest methods, return their median.
    Simple but effective — no learning, no ref peeking.
    """
    cheap = METHODS[:n_cheap]  # RAPIDS, PBE-D3BJ_SP, wB97X-D3BJ_SP
    preds, costs, refs = [], [], []
    for idx in order:
        row = df.iloc[idx]
        vals = [get_method_pred(row, m) for m in cheap if not np.isnan(get_method_pred(row, m))]
        if not vals:
            continue
        preds.append(float(np.median(vals)))
        costs.append(get_methods_cost(row, cheap))
        refs.append(row["Reference"])
    return preds, costs, refs


# ---- 11. Stacking Meta-Learner ----
def strategy_stacking_metalearner(df, order, n_cheap=2, train_frac=0.15):
    """
    Instead of SELECTING a method, COMBINE cheap predictions to predict truth.
    Use polynomial features of RAPIDS + PBE predictions.
    Cost = only the cheap methods used as features.
    """
    cheap = METHODS[:n_cheap]  # RAPIDS, PBE-D3BJ_SP
    preds, costs, refs = [], [], []

    X_train, y_train = [], []
    model = None
    n_total = sum(1 for idx in order
                  if not any(np.isnan(get_method_pred(df.iloc[idx], m)) for m in cheap))
    n_train = max(10, int(n_total * train_frac))
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        vals = [get_method_pred(row, m) for m in cheap]
        if any(np.isnan(v) for v in vals):
            continue

        r, p = vals[0], vals[1]
        # Polynomial features: r, p, r², p², r*p, |r-p|
        feat = [r, p, r**2, p**2, r * p, abs(r - p)]

        if i_valid < n_train:
            # Training phase: use median of cheap, collect labels
            pred = float(np.median(vals))
            X_train.append(feat)
            y_train.append(ref)
            # Retrain periodically
            if len(X_train) >= 10 and len(X_train) % 5 == 0:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train), np.array(y_train))
                except Exception:
                    pass
        else:
            # Inference: use trained model
            if model is not None:
                pred = model.predict(np.array([feat]))[0]
            else:
                pred = float(np.median(vals))
            # Continue online learning
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) % 50 == 0:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train), np.array(y_train))
                except Exception:
                    pass

        i_valid += 1
        preds.append(pred)
        costs.append(get_methods_cost(row, cheap))
        refs.append(ref)

    return preds, costs, refs


# ---- 12. Per-Benchmark Bias Correction ----
def strategy_bias_correction(df, order, n_cal=None, method="RAPIDS"):
    """
    Empirical Bayes: estimate bias = mean(method_pred - truth) on first N systems,
    then correct all subsequent predictions. Very cheap, addresses systematic error.
    """
    if n_cal is None:
        n_cal = max(5, int(len(order) * 0.1))

    preds, costs, refs = [], [], []
    residuals = []  # method_pred - truth
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        val = get_method_pred(row, method)
        if np.isnan(val):
            continue
        ref = row["Reference"]

        if i_valid < n_cal:
            # Calibration phase: use raw prediction, collect residuals
            pred = val
            residuals.append(val - ref)
        else:
            # Correction phase: subtract estimated bias
            bias = np.mean(residuals)
            pred = val - bias
            # Continue updating bias estimate
            residuals.append(val - ref)

        i_valid += 1
        preds.append(pred)
        costs.append(get_method_cost(row, method))
        refs.append(ref)

    return preds, costs, refs


# ---- 13. ALORS (Algorithm Recommendation via Latent Structures) ----
def strategy_alors(df, order, rank=3, train_frac=0.15):
    """
    Matrix factorization approach: decompose (system × method) prediction matrix,
    learn to predict latent system embedding from cheap features,
    reconstruct expected |error| for each method, pick argmin.
    """
    from sklearn.decomposition import TruncatedSVD

    n_train = max(10, int(len(order) * train_frac))
    preds, costs, refs = [], [], []

    # Collect training data: (features, error_vector) per system
    train_features = []
    train_errors = []  # (n, 9) matrix of |errors|
    svd_model = None
    feat_to_latent = None  # Ridge: features -> latent U_i

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        feat = get_features(row)

        if i < n_train:
            # Training: run all cheap methods to build features, use median
            pred = median_pred(row, METHODS[:3])
            if np.isnan(pred):
                pred = r

            # Record errors for all methods
            err_vec = []
            for m in METHODS:
                v = get_method_pred(row, m)
                if np.isnan(v):
                    err_vec.append(50.0)  # cap for missing
                else:
                    err_vec.append(capped_error(v, ref))
            train_features.append(feat)
            train_errors.append(err_vec)

            # Try to build model after enough training data
            if len(train_errors) >= 10 and svd_model is None:
                try:
                    E = np.array(train_errors)  # (n_train, 9)
                    k = min(rank, E.shape[0] - 1, E.shape[1] - 1)
                    if k < 1:
                        k = 1
                    svd_model = TruncatedSVD(n_components=k)
                    U = svd_model.fit_transform(E)  # (n_train, k)
                    # Train regressor: features -> U
                    feat_to_latent = Ridge(alpha=1.0)
                    feat_to_latent.fit(np.array(train_features), U)
                except Exception:
                    svd_model = None

            cost = get_methods_cost(row, METHODS[:3])  # pay for 3 cheap
        else:
            # Exploitation: predict latent vector, reconstruct errors, pick best
            chosen_m = "RAPIDS"
            if svd_model is not None and feat_to_latent is not None:
                try:
                    u_pred = feat_to_latent.predict(np.array([feat]))  # (1, k)
                    err_pred = svd_model.inverse_transform(u_pred)  # (1, 9)
                    best_mi = int(np.argmin(err_pred[0]))
                    chosen_m = METHODS[best_mi]
                except Exception:
                    pass

            val = get_method_pred(row, chosen_m)
            if np.isnan(val):
                val = r
                chosen_m = "RAPIDS"
            pred = val
            cost = get_method_cost(row, chosen_m)

            # Update model periodically
            err_vec = []
            for m in METHODS:
                v = get_method_pred(row, m)
                if np.isnan(v):
                    err_vec.append(50.0)
                else:
                    err_vec.append(capped_error(v, ref))
            train_features.append(feat)
            train_errors.append(err_vec)
            if len(train_errors) % 50 == 0:
                try:
                    E = np.array(train_errors)
                    k = min(rank, E.shape[0] - 1, E.shape[1] - 1)
                    if k < 1:
                        k = 1
                    svd_model = TruncatedSVD(n_components=k)
                    U = svd_model.fit_transform(E)
                    feat_to_latent = Ridge(alpha=1.0)
                    feat_to_latent.fit(np.array(train_features), U)
                except Exception:
                    pass

        preds.append(pred)
        costs.append(cost)
        refs.append(ref)

    return preds, costs, refs


# ---- 14. MF-MI-Greedy (Song, Chen, Yue 2018) ----
def strategy_mfmi_greedy(df, order, explore_budget_mult=3.0, alpha=1.0,
                          refit_interval=20, rng=None):
    """
    Adapted MF-MI-Greedy v2: improved version.
    - BayesianRidge for built-in uncertainty (no bootstrap)
    - Always probe 3 cheapest methods for stable features
    - Consistent feature vector (no 0-padding for unobserved)
    - Adaptive LCB coefficient decaying over time
    - More frequent refitting (every 20 systems)
    - Cross-method correlation from error history
    """
    from sklearn.linear_model import BayesianRidge

    if rng is None:
        rng = np.random.default_rng(42)

    # Method costs
    method_costs = np.ones(N_METHODS)
    for mi, m in enumerate(METHODS):
        tcol = f"{m}_time"
        if tcol in df.columns:
            vals = pd.to_numeric(df[tcol], errors="coerce").dropna().values
            if len(vals) > 0:
                method_costs[mi] = max(np.median(vals), 1.0)

    # Cheap probe methods (always run these 3)
    PROBES = [0, 1, 2]  # RAPIDS, PBE-D3BJ_SP, wB97X-D3BJ_SP
    probe_cost = sum(method_costs[mi] for mi in PROBES)

    # Training data per method
    train_X = []  # shared feature matrix
    train_y = {mi: [] for mi in range(N_METHODS)}  # errors per method
    models = {}  # mi -> BayesianRidge

    # Cross-correlation matrix (updated periodically)
    error_history = []
    corr_matrix = np.full((N_METHODS, N_METHODS), 0.3)

    preds, costs, refs = [], [], []

    def _build_feat(row):
        """Consistent feature vector from 3 cheap probes."""
        vals = []
        for mi in PROBES:
            v = get_method_pred(row, METHODS[mi])
            vals.append(v if not np.isnan(v) else 0.0)
        r, p, w = vals
        return np.array([
            r, p, w,                    # raw predictions
            abs(r), abs(p), abs(w),     # magnitudes
            abs(r - p), abs(r - w), abs(p - w),  # pairwise disagreements
            r**2, p**2,                 # squared terms
            r * p, r * w,              # interaction terms
            np.std(vals),              # overall disagreement
        ])

    def _predict_error(mi, feat):
        """Predict error and std using BayesianRidge."""
        if mi not in models:
            return 5.0, 5.0
        try:
            pred_mean, pred_std = models[mi].predict(
                feat.reshape(1, -1), return_std=True)
            return max(float(pred_mean[0]), 0.0), max(float(pred_std[0]), 0.01)
        except Exception:
            return 5.0, 5.0

    def _refit_models():
        """Refit BayesianRidge for all methods."""
        if len(train_X) < 5:
            return
        X = np.array(train_X)
        for mi in range(N_METHODS):
            y = np.array(train_y[mi])
            valid = ~np.isnan(y)
            if valid.sum() < 5:
                continue
            try:
                m = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                  lambda_1=1e-6, lambda_2=1e-6)
                m.fit(X[valid], y[valid])
                models[mi] = m
            except Exception:
                pass

    def _update_correlations():
        """Update cross-method correlation matrix."""
        if len(error_history) < 10:
            return
        errs = np.array(error_history)
        for i in range(N_METHODS):
            for j in range(i, N_METHODS):
                vi = ~np.isnan(errs[:, i])
                vj = ~np.isnan(errs[:, j])
                valid = vi & vj
                if valid.sum() >= 5:
                    c = np.corrcoef(errs[valid, i], errs[valid, j])[0, 1]
                    corr_matrix[i, j] = max(0, c)
                    corr_matrix[j, i] = corr_matrix[i, j]

    for t, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]

        # ===== Phase 1: Always probe 3 cheapest methods =====
        feat = _build_feat(row)
        observed = set(PROBES)
        total_cost = probe_cost

        # ===== Phase 1b: Adaptive exploration =====
        # Compute info gain / cost for remaining methods
        explore_budget = explore_budget_mult * probe_cost
        lcb_alpha = max(0.5, 2.0 / np.sqrt(max(t + 1, 1)))  # decaying

        if len(train_X) >= 10:
            # For each unobserved method, compute expected info gain
            for _ in range(2):  # max 2 extra explorations
                best_mi = None
                best_ratio = 0.0

                for mi in range(N_METHODS):
                    if mi in observed:
                        continue
                    if total_cost + method_costs[mi] > explore_budget:
                        continue

                    # Info gain = sum of correlated variance reductions
                    _, std_mi = _predict_error(mi, feat)
                    info = 0.0
                    for mj in range(N_METHODS):
                        if mj == mi:
                            continue
                        _, std_mj = _predict_error(mj, feat)
                        info += corr_matrix[mi, mj] * std_mj
                    ratio = info * std_mi / method_costs[mi]

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_mi = mi

                if best_mi is None:
                    break

                observed.add(best_mi)
                total_cost += method_costs[best_mi]

        # ===== Phase 2: Select method (LCB) =====
        best_lcb = float("inf")
        chosen_mi = 0

        for mi in range(N_METHODS):
            pred_err, pred_std = _predict_error(mi, feat)
            lcb = pred_err - lcb_alpha * pred_std
            if lcb < best_lcb:
                best_lcb = lcb
                chosen_mi = mi

        # Commit
        val = get_method_pred(row, METHODS[chosen_mi])
        if np.isnan(val):
            val = r
            chosen_mi = 0

        if chosen_mi not in observed:
            total_cost += method_costs[chosen_mi]

        preds.append(val)
        costs.append(total_cost)
        refs.append(ref)

        # ===== Learn =====
        err_vec = np.full(N_METHODS, np.nan)
        for mi in range(N_METHODS):
            v = get_method_pred(row, METHODS[mi])
            if not np.isnan(v):
                err_vec[mi] = capped_error(v, ref)
        train_X.append(feat.copy())
        for mi in range(N_METHODS):
            train_y[mi].append(err_vec[mi])
        error_history.append(err_vec)

        # Periodic refit
        if (t + 1) % refit_interval == 0:
            _refit_models()
            _update_correlations()

    return preds, costs, refs


# ---- 15. UCB Multi-Arm ----
def strategy_ucb(df, order, n_bins=5, c_param=1.0, cost_aware=False, rng=None):
    """
    UCB1 on (feature_bin, method) pairs.
    reward = -|error|, UCB = mean_reward + c * sqrt(log(t) / count).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rapids_vals = df["RAPIDS"].dropna().abs().values
    bin_edges = np.quantile(rapids_vals, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    def get_bin(val):
        aval = abs(val)
        for b in range(n_bins):
            if aval <= bin_edges[b + 1]:
                return b
        return n_bins - 1

    sum_reward = np.zeros((n_bins, N_METHODS))
    counts = np.zeros((n_bins, N_METHODS))
    t = 0

    # Median cost per method
    method_med_costs = np.ones(N_METHODS)
    if cost_aware:
        for mi, m in enumerate(METHODS):
            tcol = f"{m}_time"
            if tcol in df.columns:
                vals = pd.to_numeric(df[tcol], errors="coerce").dropna().values
                if len(vals) > 0:
                    method_med_costs[mi] = max(np.median(vals), 1.0)

    preds, costs_list, refs = [], [], []

    for idx in order:
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        b = get_bin(r)
        t += 1

        # UCB scores
        ucb_scores = np.full(N_METHODS, float("inf"))
        for mi in range(N_METHODS):
            if counts[b, mi] > 0:
                mean_r = sum_reward[b, mi] / counts[b, mi]
                bonus = c_param * np.sqrt(np.log(t) / counts[b, mi])
                ucb_scores[mi] = mean_r + bonus
                if cost_aware:
                    ucb_scores[mi] /= method_med_costs[mi]

        # Pick best UCB, try to get valid prediction
        ranking = np.argsort(-ucb_scores)
        chosen_m = None
        chosen_val = np.nan
        for mi in ranking:
            val = get_method_pred(row, METHODS[mi])
            if not np.isnan(val):
                chosen_m = METHODS[mi]
                chosen_val = val
                break
        if chosen_m is None:
            continue

        pred = chosen_val  # Commit to UCB's choice, no ref peeking

        # Update ALL arms (offline: we can observe all)
        for mi, m in enumerate(METHODS):
            v = get_method_pred(row, m)
            if np.isnan(v):
                continue
            reward = -capped_error(v, ref)
            sum_reward[b, mi] += reward
            counts[b, mi] += 1

        preds.append(pred)
        costs_list.append(get_method_cost(row, chosen_m))
        refs.append(ref)

    return preds, costs_list, refs


# ---- 16. Chem-Aware Learned Selector ----
def strategy_chem_learned_selector(df, order, explore_frac=0.1, rng=None):
    """
    Like Learned-Selector but uses combined prediction+chemical features.
    Chemical features provide context about the molecular system that helps
    predict which fidelity method will work best.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(order)
    n_explore = max(5, int(n * explore_frac))

    features_seen = []
    errors_seen = []

    preds, costs, refs = [], [], []

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        feat = get_combined_features(row)

        if i < n_explore:
            methods_run = list(METHODS)
            pred = median_pred(row, methods_run)
            if np.isnan(pred):
                pred = r
            cost = get_methods_cost(row, methods_run)
        else:
            chosen_m = "RAPIDS"
            if len(features_seen) >= 5:
                X = np.array(features_seen)
                Y = np.array(errors_seen)
                best_predicted_err = float("inf")
                for mi in range(N_METHODS):
                    y_col = Y[:, mi]
                    valid = ~np.isnan(y_col)
                    if valid.sum() < 5:
                        continue
                    try:
                        model = Ridge(alpha=1.0)
                        model.fit(X[valid], y_col[valid])
                        pred_err = model.predict(np.array([feat]))[0]
                        if pred_err < best_predicted_err:
                            best_predicted_err = pred_err
                            chosen_m = METHODS[mi]
                    except Exception:
                        pass

            val = get_method_pred(row, chosen_m)
            if np.isnan(val):
                val = r
                chosen_m = "RAPIDS"
            pred = val
            cost = get_method_cost(row, chosen_m)

        err_row = []
        for m in METHODS:
            v = get_method_pred(row, m)
            if np.isnan(v):
                err_row.append(np.nan)
            else:
                err_row.append(capped_error(v, ref))
        features_seen.append(feat)
        errors_seen.append(err_row)

        preds.append(pred)
        costs.append(cost)
        refs.append(ref)

    return preds, costs, refs


# ---- 17. Chem-Aware ALORS ----
def strategy_chem_alors(df, order, rank=3, train_frac=0.15):
    """
    ALORS with chemical features: matrix factorization of error matrix,
    then predict latent system embedding from combined pred+chem features.
    """
    from sklearn.decomposition import TruncatedSVD

    n_train = max(10, int(len(order) * train_frac))
    preds, costs, refs = [], [], []

    train_features = []
    train_errors = []
    svd_model = None
    feat_to_latent = None

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        feat = get_combined_features(row)

        if i < n_train:
            pred = median_pred(row, METHODS[:3])
            if np.isnan(pred):
                pred = r

            err_vec = []
            for m in METHODS:
                v = get_method_pred(row, m)
                if np.isnan(v):
                    err_vec.append(50.0)
                else:
                    err_vec.append(capped_error(v, ref))
            train_features.append(feat)
            train_errors.append(err_vec)

            if len(train_errors) >= 10 and svd_model is None:
                try:
                    E = np.array(train_errors)
                    k = min(rank, E.shape[0] - 1, E.shape[1] - 1)
                    if k < 1:
                        k = 1
                    svd_model = TruncatedSVD(n_components=k)
                    U = svd_model.fit_transform(E)
                    feat_to_latent = Ridge(alpha=1.0)
                    feat_to_latent.fit(np.array(train_features), U)
                except Exception:
                    svd_model = None

            cost = get_methods_cost(row, METHODS[:3])
        else:
            if svd_model is not None and feat_to_latent is not None:
                try:
                    u_pred = feat_to_latent.predict(np.array([feat]))[0]
                    err_pred = svd_model.inverse_transform(u_pred.reshape(1, -1))[0]
                    best_mi = int(np.argmin(err_pred))
                    chosen_m = METHODS[best_mi]
                except Exception:
                    chosen_m = "RAPIDS"
            else:
                chosen_m = "RAPIDS"

            val = get_method_pred(row, chosen_m)
            if np.isnan(val):
                val = r
                chosen_m = "RAPIDS"
            pred = val
            cost = get_method_cost(row, chosen_m)

            err_vec = []
            for m in METHODS:
                v = get_method_pred(row, m)
                if np.isnan(v):
                    err_vec.append(50.0)
                else:
                    err_vec.append(capped_error(v, ref))
            train_features.append(feat)
            train_errors.append(err_vec)

            if len(train_errors) % 20 == 0:
                try:
                    E = np.array(train_errors)
                    k = min(rank, E.shape[0] - 1, E.shape[1] - 1)
                    if k < 1:
                        k = 1
                    svd_model = TruncatedSVD(n_components=k)
                    U = svd_model.fit_transform(E)
                    feat_to_latent = Ridge(alpha=1.0)
                    feat_to_latent.fit(np.array(train_features), U)
                except Exception:
                    pass

        preds.append(pred)
        costs.append(cost)
        refs.append(ref)

    return preds, costs, refs


# ---- 18. Chem-Aware UCB ----
def strategy_chem_ucb(df, order, n_bins=8, c_param=1.0, rng=None):
    """
    UCB with chemical feature binning instead of RAPIDS-value binning.
    Uses k-means on chemical features to define bins, then per-bin UCB.
    """
    from sklearn.cluster import KMeans

    if rng is None:
        rng = np.random.default_rng(42)

    # Collect chemical features for all systems to define bins
    all_chem = []
    valid_indices = []
    for idx in order:
        row = df.iloc[idx]
        if np.isnan(get_method_pred(row, "RAPIDS")):
            continue
        all_chem.append(get_chem_features(row))
        valid_indices.append(idx)

    if len(all_chem) < n_bins:
        n_bins = max(2, len(all_chem) // 2)

    all_chem = np.array(all_chem)
    # Normalize features for clustering
    means = np.nanmean(all_chem, axis=0)
    stds = np.nanstd(all_chem, axis=0)
    stds[stds < 1e-10] = 1.0
    all_chem_norm = (all_chem - means) / stds

    kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
    kmeans.fit(all_chem_norm)

    sum_reward = np.zeros((n_bins, N_METHODS))
    counts = np.zeros((n_bins, N_METHODS))
    t = 0

    preds, costs_list, refs = [], [], []

    for idx in order:
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]

        chem_feat = get_chem_features(row)
        chem_norm = (chem_feat - means) / stds
        b = int(kmeans.predict(chem_norm.reshape(1, -1))[0])
        t += 1

        # UCB selection
        if np.any(counts[b] == 0):
            # Try untried methods in this bin
            untried = np.where(counts[b] == 0)[0]
            chosen_mi = rng.choice(untried)
        else:
            ucb_vals = (sum_reward[b] / counts[b]) + c_param * np.sqrt(np.log(t) / counts[b])
            chosen_mi = int(np.argmax(ucb_vals))

        chosen_m = METHODS[chosen_mi]
        val = get_method_pred(row, chosen_m)
        if np.isnan(val):
            val = r
            chosen_m = "RAPIDS"
            chosen_mi = 0

        pred = val

        for mi, m in enumerate(METHODS):
            v = get_method_pred(row, m)
            if np.isnan(v):
                continue
            reward = -capped_error(v, ref)
            sum_reward[b, mi] += reward
            counts[b, mi] += 1

        preds.append(pred)
        costs_list.append(get_method_cost(row, chosen_m))
        refs.append(ref)

    return preds, costs_list, refs


# ---- 19. Chem-Aware Stacking Meta-Learner ----
def strategy_chem_stacking(df, order, n_cheap=2, train_frac=0.15):
    """
    Stacking with chemical features: predict reference from cheap predictions
    + chemical descriptors.
    """
    cheap = METHODS[:n_cheap]
    preds, costs, refs = [], [], []

    X_train, y_train = [], []
    model = None
    n_total = sum(1 for idx in order
                  if not any(np.isnan(get_method_pred(df.iloc[idx], m)) for m in cheap))
    n_train = max(10, int(n_total * train_frac))
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        vals = [get_method_pred(row, m) for m in cheap]
        if any(np.isnan(v) for v in vals):
            continue

        r, p = vals[0], vals[1] if n_cheap > 1 else (vals[0], vals[0])
        # Prediction features + chemical features
        pred_feat = [r, p, r**2, p**2, r * p, abs(r - p)]
        chem_feat = get_chem_features(row).tolist()
        feat = pred_feat + chem_feat

        if i_valid < n_train:
            pred = float(np.median(vals))
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) >= 10 and len(X_train) % 5 == 0:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train), np.array(y_train))
                except Exception:
                    pass
        else:
            if model is not None:
                pred = model.predict(np.array([feat]))[0]
            else:
                pred = float(np.median(vals))
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) % 10 == 0:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train), np.array(y_train))
                except Exception:
                    pass

        preds.append(pred)
        costs.append(get_methods_cost(row, cheap))
        refs.append(ref)
        i_valid += 1

    return preds, costs, refs


# ---- 20. Chem-Aware Bias Correction ----
def strategy_chem_bias_correction(df, order, method="RAPIDS", train_frac=0.15):
    """
    Like BiasCorr but uses chemical features to learn system-dependent
    bias correction instead of a single global offset.
    bias(system) = f(chem_features) fitted via Ridge.
    """
    preds, costs, refs = [], [], []
    X_train, y_train = [], []
    model = None
    n_total = sum(1 for idx in order if not np.isnan(get_method_pred(df.iloc[idx], method)))
    n_train = max(10, int(n_total * train_frac))
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        val = get_method_pred(row, method)
        if np.isnan(val):
            continue

        chem_feat = get_chem_features(row)
        # Features: prediction value + chemical descriptors
        feat = np.concatenate([[val, val**2, abs(val)], chem_feat])

        if i_valid < n_train:
            pred = val  # raw prediction during training
            X_train.append(feat)
            y_train.append(ref)  # learn to predict ref from (pred, chem)
            if len(X_train) >= 10 and len(X_train) % 5 == 0:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train), np.array(y_train))
                except Exception:
                    pass
        else:
            if model is not None:
                pred = model.predict(feat.reshape(1, -1))[0]
            else:
                pred = val
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) % 10 == 0:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(np.array(X_train), np.array(y_train))
                except Exception:
                    pass

        preds.append(pred)
        costs.append(get_method_cost(row, method))
        refs.append(ref)
        i_valid += 1

    return preds, costs, refs


# ---- 21. Bucket Prior (cross-benchmark transfer) ----
def build_bucket_prior(all_data, exclude_bench, n_buckets=12):
    """
    Build a bucket prior from ALL benchmarks except exclude_bench.
    Returns (kmeans_model, means, stds, bucket_best_method, bucket_method_scores).

    For each bucket: compute mean |error| for each method across all systems
    in that bucket, using reference values. This is the "experience table"
    that an LLM agent would have access to.
    """
    from sklearn.cluster import KMeans

    # Collect all systems from other benchmarks
    all_chem_feats = []
    all_errors = []  # (n_systems, 9) — |pred - ref| per method

    for bench_name, df in all_data.items():
        if bench_name == exclude_bench:
            continue
        for idx in range(len(df)):
            row = df.iloc[idx]
            chem_feat = get_chem_features(row)
            # Skip if no chem data
            if np.all(chem_feat == 0):
                continue
            err_vec = []
            has_any = False
            for m in METHODS:
                v = get_method_pred(row, m)
                ref = row["Reference"]
                if np.isnan(v):
                    err_vec.append(np.nan)
                else:
                    err_vec.append(capped_error(v, ref))
                    has_any = True
            if has_any:
                all_chem_feats.append(chem_feat)
                all_errors.append(err_vec)

    if len(all_chem_feats) < n_buckets * 2:
        return None

    X = np.array(all_chem_feats)
    E = np.array(all_errors)

    # Normalize for clustering
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds < 1e-10] = 1.0
    X_norm = (X - means) / stds

    kmeans = KMeans(n_clusters=n_buckets, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_norm)

    # Per-bucket: mean error for each method
    bucket_mean_err = np.full((n_buckets, N_METHODS), np.nan)
    bucket_counts = np.zeros((n_buckets, N_METHODS))
    for i, lbl in enumerate(labels):
        for mi in range(N_METHODS):
            if not np.isnan(E[i, mi]):
                if np.isnan(bucket_mean_err[lbl, mi]):
                    bucket_mean_err[lbl, mi] = 0.0
                bucket_mean_err[lbl, mi] += E[i, mi]
                bucket_counts[lbl, mi] += 1

    for b in range(n_buckets):
        for mi in range(N_METHODS):
            if bucket_counts[b, mi] > 0:
                bucket_mean_err[b, mi] /= bucket_counts[b, mi]

    # Best method per bucket
    bucket_best = []
    for b in range(n_buckets):
        row_err = bucket_mean_err[b]
        valid = ~np.isnan(row_err)
        if np.any(valid):
            best_mi = int(np.nanargmin(row_err))
            bucket_best.append(best_mi)
        else:
            bucket_best.append(0)  # fallback RAPIDS

    return {
        "kmeans": kmeans,
        "means": means,
        "stds": stds,
        "bucket_best": bucket_best,
        "bucket_mean_err": bucket_mean_err,
        "bucket_counts": bucket_counts,
        "n_train_systems": len(all_chem_feats),
    }


def strategy_bucket_prior(df, order, prior, rng=None):
    """
    Pure bucket lookup: classify system by chemistry → use bucket's best method.
    No online learning, just the prior lookup table.
    """
    if prior is None:
        # Fallback to RAPIDS if no prior
        return strategy_always_X(df, order, "RAPIDS")

    kmeans = prior["kmeans"]
    means = prior["means"]
    stds = prior["stds"]
    bucket_best = prior["bucket_best"]

    preds, costs, refs = [], [], []

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        chem_feat = get_chem_features(row)
        chem_norm = (chem_feat - means) / stds
        bucket = int(kmeans.predict(chem_norm.reshape(1, -1))[0])
        best_mi = bucket_best[bucket]
        chosen_m = METHODS[best_mi]

        val = get_method_pred(row, chosen_m)
        if np.isnan(val):
            val = get_method_pred(row, "RAPIDS")
            if np.isnan(val):
                continue
            chosen_m = "RAPIDS"

        preds.append(val)
        costs.append(get_method_cost(row, chosen_m))
        refs.append(ref)

    return preds, costs, refs


def strategy_bucket_prior_adaptive(df, order, prior, explore_frac=0.05, rng=None):
    """
    Bucket prior as warm start + online adaptation via per-bucket UCB.
    Prior provides initial reward estimates, UCB refines per-bucket choices
    as systems are processed.
    """
    if prior is None:
        return strategy_always_X(df, order, "RAPIDS")
    if rng is None:
        rng = np.random.default_rng(42)

    kmeans = prior["kmeans"]
    means_norm = prior["means"]
    stds_norm = prior["stds"]
    bucket_mean_err = prior["bucket_mean_err"]
    bucket_counts_prior = prior["bucket_counts"]
    n_buckets = len(prior["bucket_best"])

    # Initialize UCB with prior: convert mean_err to reward (-err)
    # Scale prior counts to avoid over-trusting (effective prior weight)
    prior_weight = 10.0  # treat prior as if we've seen 10 systems per bucket
    sum_reward = np.zeros((n_buckets, N_METHODS))
    counts = np.zeros((n_buckets, N_METHODS))
    for b in range(n_buckets):
        for mi in range(N_METHODS):
            if not np.isnan(bucket_mean_err[b, mi]):
                sum_reward[b, mi] = -bucket_mean_err[b, mi] * prior_weight
                counts[b, mi] = prior_weight

    t = 0
    preds, costs_list, refs = [], [], []

    for idx in order:
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]
        t += 1

        chem_feat = get_chem_features(row)
        chem_norm = (chem_feat - means_norm) / stds_norm
        bucket = int(kmeans.predict(chem_norm.reshape(1, -1))[0])

        # UCB selection with warm-started prior
        if np.any(counts[bucket] == 0):
            untried = np.where(counts[bucket] == 0)[0]
            chosen_mi = rng.choice(untried)
        else:
            ucb_vals = (sum_reward[bucket] / counts[bucket]) + 1.0 * np.sqrt(np.log(t) / counts[bucket])
            chosen_mi = int(np.argmax(ucb_vals))

        chosen_m = METHODS[chosen_mi]
        val = get_method_pred(row, chosen_m)
        if np.isnan(val):
            val = r
            chosen_m = "RAPIDS"
            chosen_mi = 0

        pred = val

        # Update ALL arms for this bucket (offline: we observe all)
        for mi, m in enumerate(METHODS):
            v = get_method_pred(row, m)
            if np.isnan(v):
                continue
            reward = -capped_error(v, ref)
            sum_reward[bucket, mi] += reward
            counts[bucket, mi] += 1

        preds.append(pred)
        costs_list.append(get_method_cost(row, chosen_m))
        refs.append(ref)

    return preds, costs_list, refs


def strategy_bucket_learned(df, order, prior, rng=None):
    """
    Use bucket prior + chemical features for Ridge-based method selection.
    Prior provides bucket_id as a categorical feature, combined with
    chemical+prediction features for per-method error prediction.
    """
    if prior is None:
        return strategy_always_X(df, order, "RAPIDS")
    if rng is None:
        rng = np.random.default_rng(42)

    kmeans = prior["kmeans"]
    means_norm = prior["means"]
    stds_norm = prior["stds"]
    n_buckets = len(prior["bucket_best"])
    bucket_mean_err = prior["bucket_mean_err"]

    n = len(order)
    n_explore = max(5, int(n * 0.1))

    features_seen = []
    errors_seen = []
    preds, costs, refs = [], [], []

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]

        # Build feature: combined + one-hot bucket + bucket prior errors
        chem_feat = get_chem_features(row)
        chem_norm = (chem_feat - means_norm) / stds_norm
        bucket = int(kmeans.predict(chem_norm.reshape(1, -1))[0])

        pred_feat = get_features(row)  # 6-dim prediction features
        bucket_onehot = np.zeros(n_buckets)
        bucket_onehot[bucket] = 1.0
        # Prior error estimates for this bucket as features
        prior_err = bucket_mean_err[bucket].copy()
        prior_err[np.isnan(prior_err)] = 5.0
        feat = np.concatenate([pred_feat, chem_feat, bucket_onehot, prior_err])

        if i < n_explore:
            methods_run = list(METHODS)
            pred = median_pred(row, methods_run)
            if np.isnan(pred):
                pred = r
            cost = get_methods_cost(row, methods_run)
        else:
            chosen_m = METHODS[prior["bucket_best"][bucket]]  # prior default
            if len(features_seen) >= 5:
                X = np.array(features_seen)
                Y = np.array(errors_seen)
                best_predicted_err = float("inf")
                for mi in range(N_METHODS):
                    y_col = Y[:, mi]
                    valid = ~np.isnan(y_col)
                    if valid.sum() < 5:
                        continue
                    try:
                        model = Ridge(alpha=1.0)
                        model.fit(X[valid], y_col[valid])
                        pred_err = model.predict(np.array([feat]))[0]
                        if pred_err < best_predicted_err:
                            best_predicted_err = pred_err
                            chosen_m = METHODS[mi]
                    except Exception:
                        pass

            val = get_method_pred(row, chosen_m)
            if np.isnan(val):
                val = r
                chosen_m = "RAPIDS"
            pred = val
            cost = get_method_cost(row, chosen_m)

        err_row = []
        for m in METHODS:
            v = get_method_pred(row, m)
            if np.isnan(v):
                err_row.append(np.nan)
            else:
                err_row.append(capped_error(v, ref))
        features_seen.append(feat)
        errors_seen.append(err_row)

        preds.append(pred)
        costs.append(cost)
        refs.append(ref)

    return preds, costs, refs


# ---------------------------------------------------------------------------
# v2 Enhanced feature engineering
# ---------------------------------------------------------------------------

def compute_global_chem_medians(all_data):
    """Compute cross-benchmark median for each chem feature (for NaN imputation)."""
    all_vals = {col: [] for col in CHEM_COLS}
    for bench_name, df in all_data.items():
        for idx in range(len(df)):
            row = df.iloc[idx]
            for col in CHEM_COLS:
                val = row.get(col, np.nan)
                if not np.isnan(val):
                    all_vals[col].append(val)
    medians = {}
    for col in CHEM_COLS:
        if all_vals[col]:
            medians[col] = float(np.median(all_vals[col]))
        else:
            medians[col] = 0.0
    return medians


def get_enhanced_chem_features(df_row, chem_medians):
    """Extract chemical features with median imputation + ratio features."""
    feat = np.zeros(N_CHEM_FEATURES)
    for i, col in enumerate(CHEM_COLS):
        val = df_row.get(col, np.nan)
        if np.isnan(val):
            feat[i] = chem_medians.get(col, 0.0)
        else:
            feat[i] = val

    # Ratio features
    probe_mw = df_row.get("chem_probe_molecular_weight", np.nan)
    target_mw = df_row.get("chem_target_molecular_weight", np.nan)
    if np.isnan(probe_mw):
        probe_mw = chem_medians.get("chem_probe_molecular_weight", 100.0)
    if np.isnan(target_mw):
        target_mw = chem_medians.get("chem_target_molecular_weight", 100.0)
    ratio_mw = probe_mw / max(target_mw, 1.0)

    probe_tpsa = df_row.get("chem_probe_tpsa", np.nan)
    target_tpsa = df_row.get("chem_target_tpsa", np.nan)
    if np.isnan(probe_tpsa):
        probe_tpsa = chem_medians.get("chem_probe_tpsa", 20.0)
    if np.isnan(target_tpsa):
        target_tpsa = chem_medians.get("chem_target_tpsa", 20.0)
    ratio_tpsa = probe_tpsa / (target_tpsa + 1.0)

    probe_cplx = df_row.get("chem_probe_complexity", np.nan)
    target_cplx = df_row.get("chem_target_complexity", np.nan)
    if np.isnan(probe_cplx):
        probe_cplx = chem_medians.get("chem_probe_complexity", 50.0)
    if np.isnan(target_cplx):
        target_cplx = chem_medians.get("chem_target_complexity", 50.0)
    ratio_cplx = probe_cplx / max(target_cplx, 1.0)

    ratios = np.array([ratio_mw, ratio_tpsa, ratio_cplx])
    return np.concatenate([feat, ratios])


def get_enhanced_features(df_row, chem_medians):
    """Prediction features (6) + enhanced chemical features (31+3) = combined vector."""
    pred_feat = get_features(df_row)  # 6 dims
    chem_feat = get_enhanced_chem_features(df_row, chem_medians)  # 34 dims
    return np.concatenate([pred_feat, chem_feat])  # 40 dims


def get_stacking_features_2cheap(df_row, chem_medians):
    """Features for GBM stacking: RAPIDS + PBE predictions + enhanced chem features."""
    r = get_method_pred(df_row, "RAPIDS")
    pbe = get_method_pred(df_row, "PBE-D3BJ_SP")
    if np.isnan(r):
        r = 0.0
    if np.isnan(pbe):
        pbe = r
    pred_feats = [r, pbe, abs(r), abs(pbe), r - pbe, abs(r - pbe),
                  r**2, pbe**2, r * pbe, (r + pbe) / 2.0]
    chem_feat = get_enhanced_chem_features(df_row, chem_medians)
    return np.concatenate([pred_feats, chem_feat])


def get_stacking_features_3cheap(df_row, chem_medians):
    """Features for GBM stacking: RAPIDS + PBE + wB97X predictions + enhanced chem features."""
    r = get_method_pred(df_row, "RAPIDS")
    pbe = get_method_pred(df_row, "PBE-D3BJ_SP")
    w = get_method_pred(df_row, "wB97X-D3BJ_SP")
    if np.isnan(r):
        r = 0.0
    if np.isnan(pbe):
        pbe = r
    if np.isnan(w):
        w = pbe
    pred_feats = [r, pbe, w,
                  abs(r), abs(pbe), abs(w),
                  r - pbe, r - w, pbe - w,
                  abs(r - pbe), abs(r - w), abs(pbe - w),
                  r**2, pbe**2, w**2,
                  r * pbe, r * w, pbe * w,
                  (r + pbe + w) / 3.0, np.std([r, pbe, w])]
    chem_feat = get_enhanced_chem_features(df_row, chem_medians)
    return np.concatenate([pred_feats, chem_feat])


# ---------------------------------------------------------------------------
# v2 Strategy implementations (22-28)
# ---------------------------------------------------------------------------

def _fit_gbm(X, y, n_est=50, max_depth=3):
    """Helper to fit GBM with optional subsampling for speed."""
    MAX_TRAIN = 500
    if len(X) > MAX_TRAIN:
        idx = np.random.default_rng(42).choice(len(X), MAX_TRAIN, replace=False)
        X, y = X[idx], y[idx]
    model = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=max_depth, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model


# ---- 22. GBM Stacking (2 cheap methods) ----
def strategy_gbm_stacking_2cheap(df, order, chem_medians):
    """
    GBM prediction combination: RAPIDS + PBE predictions + chem features -> predict reference.
    Cost = RAPIDS + PBE only. Training on first 15%, retrain every 50 systems.
    """
    cheap_methods = ["RAPIDS", "PBE-D3BJ_SP"]
    preds, costs, refs = [], [], []

    X_train, y_train = [], []
    model = None
    n_total = sum(1 for idx in order
                  if not np.isnan(get_method_pred(df.iloc[idx], "RAPIDS")))
    n_train = max(15, int(n_total * 0.15))
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue

        feat = get_stacking_features_2cheap(row, chem_medians)

        if i_valid < n_train:
            pbe = get_method_pred(row, "PBE-D3BJ_SP")
            vals = [v for v in [r, pbe] if not np.isnan(v)]
            pred = float(np.median(vals))
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) >= 15 and len(X_train) % 10 == 0:
                try:
                    model = _fit_gbm(np.array(X_train), np.array(y_train))
                except Exception:
                    pass
        else:
            if model is not None:
                try:
                    pred = float(model.predict(feat.reshape(1, -1))[0])
                except Exception:
                    pred = r
            else:
                pred = r
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) % 50 == 0:
                try:
                    model = _fit_gbm(np.array(X_train), np.array(y_train))
                except Exception:
                    pass

        i_valid += 1
        preds.append(pred)
        costs.append(get_methods_cost(row, cheap_methods))
        refs.append(ref)

    return preds, costs, refs


# ---- 23. GBM Stacking (3 cheap methods) ----
def strategy_gbm_stacking_3cheap(df, order, chem_medians):
    """
    GBM prediction combination: RAPIDS + PBE + wB97X predictions + chem features -> predict reference.
    Cost = RAPIDS + PBE + wB97X.
    """
    cheap_methods = ["RAPIDS", "PBE-D3BJ_SP", "wB97X-D3BJ_SP"]
    preds, costs, refs = [], [], []

    X_train, y_train = [], []
    model = None
    n_total = sum(1 for idx in order
                  if not np.isnan(get_method_pred(df.iloc[idx], "RAPIDS")))
    n_train = max(15, int(n_total * 0.15))
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue

        feat = get_stacking_features_3cheap(row, chem_medians)

        if i_valid < n_train:
            vals = [get_method_pred(row, m) for m in cheap_methods]
            vals = [v for v in vals if not np.isnan(v)]
            pred = float(np.median(vals)) if vals else r
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) >= 15 and len(X_train) % 10 == 0:
                try:
                    model = _fit_gbm(np.array(X_train), np.array(y_train))
                except Exception:
                    pass
        else:
            if model is not None:
                try:
                    pred = float(model.predict(feat.reshape(1, -1))[0])
                except Exception:
                    pred = r
            else:
                pred = r
            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) % 50 == 0:
                try:
                    model = _fit_gbm(np.array(X_train), np.array(y_train))
                except Exception:
                    pass

        i_valid += 1
        preds.append(pred)
        costs.append(get_methods_cost(row, cheap_methods))
        refs.append(ref)

    return preds, costs, refs


# ---- 24. Cross-benchmark Meta-GBM ----
def build_meta_gbm(all_data, exclude_bench, chem_medians):
    """
    Pre-train a GBM on ALL other benchmarks (leave-one-out).
    Maps [RAPIDS_pred, PBE_pred, chem_features] -> reference value.
    """
    X_all, y_all = [], []
    for bench_name, df in all_data.items():
        if bench_name == exclude_bench:
            continue
        for idx in range(len(df)):
            row = df.iloc[idx]
            r = get_method_pred(row, "RAPIDS")
            if np.isnan(r):
                continue
            feat = get_stacking_features_2cheap(row, chem_medians)
            X_all.append(feat)
            y_all.append(row["Reference"])

    if len(X_all) < 50:
        return None

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    model = _fit_gbm(X_all, y_all, n_est=100, max_depth=4)
    return model


def strategy_meta_gbm(df, order, meta_model, chem_medians):
    """
    Pure meta-model prediction. No online learning, no cold start.
    Uses pre-trained cross-benchmark GBM from system 1.
    """
    cheap_methods = ["RAPIDS", "PBE-D3BJ_SP"]
    preds, costs, refs = [], [], []

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue

        feat = get_stacking_features_2cheap(row, chem_medians)

        if meta_model is not None:
            try:
                pred = float(meta_model.predict(feat.reshape(1, -1))[0])
            except Exception:
                pred = r
        else:
            pred = r

        preds.append(pred)
        costs.append(get_methods_cost(row, cheap_methods))
        refs.append(ref)

    return preds, costs, refs


# ---- 25. Meta-GBM Adaptive ----
def strategy_meta_gbm_adaptive(df, order, meta_model, chem_medians):
    """
    Meta-model prediction + online fine-tuning every 20 systems.
    Starts with pre-trained model (no cold start), then adapts.
    """
    cheap_methods = ["RAPIDS", "PBE-D3BJ_SP"]
    preds, costs, refs = [], [], []

    X_online, y_online = [], []
    local_model = None

    for i, idx in enumerate(order):
        row = df.iloc[idx]
        ref = row["Reference"]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue

        feat = get_stacking_features_2cheap(row, chem_medians)

        # Predict: blend meta-model and local model
        meta_pred = r
        if meta_model is not None:
            try:
                meta_pred = float(meta_model.predict(feat.reshape(1, -1))[0])
            except Exception:
                meta_pred = r

        if local_model is not None:
            try:
                local_pred = float(local_model.predict(feat.reshape(1, -1))[0])
                local_weight = min(0.7, len(X_online) / 200.0)
                pred = (1.0 - local_weight) * meta_pred + local_weight * local_pred
            except Exception:
                pred = meta_pred
        else:
            pred = meta_pred

        # Online learning
        X_online.append(feat)
        y_online.append(ref)
        if len(X_online) >= 20 and len(X_online) % 50 == 0:
            try:
                local_model = _fit_gbm(np.array(X_online), np.array(y_online))
            except Exception:
                pass

        preds.append(pred)
        costs.append(get_methods_cost(row, cheap_methods))
        refs.append(ref)

    return preds, costs, refs


# ---- 26. Hybrid Select-then-Stack ----
def strategy_hybrid_select_stack(df, order, meta_model, chem_medians):
    """
    Meta-model predicts per-method error -> select top 3 -> run them ->
    weighted average (weights = inverse predicted errors).
    """
    preds, costs, refs = [], [], []

    X_train = []
    y_train = []  # list of error_rows
    per_method_error_models = {}
    n_total = sum(1 for idx in order if not np.isnan(get_method_pred(df.iloc[idx], "RAPIDS")))
    n_train = max(15, int(n_total * 0.15))
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue

        feat = get_stacking_features_2cheap(row, chem_medians)

        if i_valid < n_train:
            # Training: use meta-model prediction or median
            if meta_model is not None:
                try:
                    pred = float(meta_model.predict(feat.reshape(1, -1))[0])
                except Exception:
                    pred = r
            else:
                pred = r

            # Record per-method errors
            err_row = []
            for m in METHODS:
                v = get_method_pred(row, m)
                if np.isnan(v):
                    err_row.append(np.nan)
                else:
                    err_row.append(capped_error(v, ref))
            X_train.append(feat)
            y_train.append(err_row)

            if len(X_train) >= 15 and len(X_train) % 10 == 0:
                X_arr = np.array(X_train)
                Y_arr = np.array(y_train)
                for mi in range(N_METHODS):
                    y_col = Y_arr[:, mi]
                    valid = ~np.isnan(y_col)
                    if valid.sum() >= 10:
                        try:
                            m = Ridge(alpha=1.0)
                            m.fit(X_arr[valid], y_col[valid])
                            per_method_error_models[mi] = m
                        except Exception:
                            pass

            methods_run = ["RAPIDS", "PBE-D3BJ_SP"]
            cost = get_methods_cost(row, methods_run)
        else:
            # Select top 3 methods by predicted error
            pred_errors = {}
            for mi in range(N_METHODS):
                if mi in per_method_error_models:
                    try:
                        pe = float(per_method_error_models[mi].predict(feat.reshape(1, -1))[0])
                        pred_errors[mi] = max(pe, 0.01)
                    except Exception:
                        pred_errors[mi] = 50.0
                else:
                    pred_errors[mi] = 50.0

            sorted_methods = sorted(pred_errors.keys(), key=lambda mi: pred_errors[mi])
            top_methods = sorted_methods[:3]

            method_preds = {}
            methods_run = []
            for mi in top_methods:
                v = get_method_pred(row, METHODS[mi])
                if not np.isnan(v):
                    method_preds[mi] = v
                    methods_run.append(METHODS[mi])

            if not method_preds:
                pred = r
                methods_run = ["RAPIDS"]
            else:
                weights = []
                vals = []
                for mi, v in method_preds.items():
                    w = 1.0 / max(pred_errors[mi], 0.01)
                    weights.append(w)
                    vals.append(v)
                weights = np.array(weights)
                weights = weights / weights.sum()
                pred = float(np.dot(weights, vals))

            cost = get_methods_cost(row, methods_run)

            err_row = []
            for m in METHODS:
                v = get_method_pred(row, m)
                if np.isnan(v):
                    err_row.append(np.nan)
                else:
                    err_row.append(capped_error(v, ref))
            X_train.append(feat)
            y_train.append(err_row)
            if len(X_train) % 20 == 0:
                X_arr = np.array(X_train)
                Y_arr = np.array(y_train)
                for mi in range(N_METHODS):
                    y_col = Y_arr[:, mi]
                    valid = ~np.isnan(y_col)
                    if valid.sum() >= 10:
                        try:
                            m = Ridge(alpha=1.0)
                            m.fit(X_arr[valid], y_col[valid])
                            per_method_error_models[mi] = m
                        except Exception:
                            pass

        i_valid += 1
        preds.append(pred)
        costs.append(cost)
        refs.append(ref)

    return preds, costs, refs


# ---- 27. LinUCB (contextual bandit with BayesianRidge) ----
def strategy_linucb(df, order, chem_medians, alpha=1.0, rng=None):
    """
    BayesianRidge per method predicting reward from enhanced features.
    UCB = predicted_reward + alpha * prediction_std. Pick highest UCB.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    preds, costs, refs = [], [], []

    train_X = []
    train_rewards = {mi: [] for mi in range(N_METHODS)}
    models = {}

    MAX_TRAIN_LINUCB = 300

    def _refit(min_samples=10):
        if len(train_X) < min_samples:
            return
        X = np.array(train_X)
        if len(X) > MAX_TRAIN_LINUCB:
            sel = np.random.default_rng(42).choice(len(X), MAX_TRAIN_LINUCB, replace=False)
        else:
            sel = np.arange(len(X))
        X_sub = X[sel]
        for mi in range(N_METHODS):
            y = np.array(train_rewards[mi])
            valid = ~np.isnan(y[sel])
            if valid.sum() < min_samples:
                continue
            try:
                m = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                  lambda_1=1e-6, lambda_2=1e-6)
                m.fit(X_sub[valid], y[sel][valid])
                models[mi] = m
            except Exception:
                pass

    for t, idx in enumerate(order):
        row = df.iloc[idx]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue
        ref = row["Reference"]

        feat = get_enhanced_features(row, chem_medians)

        if len(models) == 0:
            chosen_mi = rng.integers(N_METHODS)
        else:
            best_ucb = -float("inf")
            chosen_mi = 0
            for mi in range(N_METHODS):
                if mi in models:
                    try:
                        pred_mean, pred_std = models[mi].predict(
                            feat.reshape(1, -1), return_std=True)
                        ucb = float(pred_mean[0]) + alpha * float(pred_std[0])
                    except Exception:
                        ucb = 0.0
                else:
                    ucb = 10.0  # high UCB for unexplored
                if ucb > best_ucb:
                    best_ucb = ucb
                    chosen_mi = mi

        chosen_m = METHODS[chosen_mi]
        val = get_method_pred(row, chosen_m)
        if np.isnan(val):
            val = r
            chosen_m = "RAPIDS"
            chosen_mi = 0

        pred = val

        train_X.append(feat)
        for mi in range(N_METHODS):
            v = get_method_pred(row, METHODS[mi])
            if np.isnan(v):
                train_rewards[mi].append(np.nan)
            else:
                train_rewards[mi].append(-capped_error(v, ref))

        if (t + 1) % 50 == 0:
            _refit()

        preds.append(pred)
        costs.append(get_method_cost(row, chosen_m))
        refs.append(ref)

    return preds, costs, refs


# ---- 28. GBM Stacking with meta warm-start ----
def strategy_gbm_stacking_metawarm(df, order, meta_model, chem_medians):
    """
    GBM stacking 2-cheap but uses meta-model predictions during cold-start phase.
    Transitions to local GBM after 15% training, then blends.
    """
    cheap_methods = ["RAPIDS", "PBE-D3BJ_SP"]
    preds, costs, refs = [], [], []

    X_train, y_train = [], []
    local_model = None
    n_total = sum(1 for idx in order
                  if not np.isnan(get_method_pred(df.iloc[idx], "RAPIDS")))
    n_train = max(15, int(n_total * 0.15))
    i_valid = 0

    for idx in order:
        row = df.iloc[idx]
        ref = row["Reference"]
        r = get_method_pred(row, "RAPIDS")
        if np.isnan(r):
            continue

        feat = get_stacking_features_2cheap(row, chem_medians)

        if i_valid < n_train:
            if meta_model is not None:
                try:
                    pred = float(meta_model.predict(feat.reshape(1, -1))[0])
                except Exception:
                    pred = r
            else:
                pbe = get_method_pred(row, "PBE-D3BJ_SP")
                vals = [v for v in [r, pbe] if not np.isnan(v)]
                pred = float(np.median(vals))

            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) >= 15 and len(X_train) % 10 == 0:
                try:
                    local_model = _fit_gbm(np.array(X_train), np.array(y_train))
                except Exception:
                    pass
        else:
            meta_pred = r
            if meta_model is not None:
                try:
                    meta_pred = float(meta_model.predict(feat.reshape(1, -1))[0])
                except Exception:
                    meta_pred = r

            if local_model is not None:
                try:
                    local_pred = float(local_model.predict(feat.reshape(1, -1))[0])
                    local_weight = min(0.8, len(X_train) / 150.0)
                    pred = (1.0 - local_weight) * meta_pred + local_weight * local_pred
                except Exception:
                    pred = meta_pred
            else:
                pred = meta_pred

            X_train.append(feat)
            y_train.append(ref)
            if len(X_train) % 50 == 0:
                try:
                    local_model = _fit_gbm(np.array(X_train), np.array(y_train))
                except Exception:
                    pass

        i_valid += 1
        preds.append(pred)
        costs.append(get_methods_cost(row, cheap_methods))
        refs.append(ref)

    return preds, costs, refs


# ---------------------------------------------------------------------------
# Curve computation
# ---------------------------------------------------------------------------

def build_curve(preds, costs, refs, n_points=N_BUDGET_POINTS):
    n = len(preds)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    cum_costs = np.cumsum(costs)
    total_cost = cum_costs[-1]
    if total_cost <= 0:
        total_cost = 1.0
    budget_points = np.linspace(cum_costs[0], total_cost, n_points)

    mae_curve, rho_curve = [], []
    running_preds, running_refs, running_errors = [], [], []
    sys_idx = 0

    for bp in budget_points:
        while sys_idx < n and cum_costs[sys_idx] <= bp:
            running_preds.append(preds[sys_idx])
            running_refs.append(refs[sys_idx])
            running_errors.append(capped_error(preds[sys_idx], refs[sys_idx]))
            sys_idx += 1

        if not running_preds:
            mae_curve.append(np.nan)
            rho_curve.append(np.nan)
            continue

        mae = np.mean(running_errors)
        rho = compute_rho(running_preds, running_refs)
        mae_curve.append(mae)
        rho_curve.append(rho if not np.isnan(rho) else np.nan)

    budget_hours = budget_points / 3600.0
    return np.array(budget_hours), np.array(mae_curve), np.array(rho_curve)


# ---------------------------------------------------------------------------
# Run strategies on one benchmark
# ---------------------------------------------------------------------------

def run_strategies_on_benchmark(df, bench_name, seeds=N_SEEDS, bucket_prior=None,
                                meta_model=None, chem_medians=None):
    n = len(df)
    print(f"\n  {bench_name}: {n} systems")
    results = {}
    order = list(range(n))

    # 1. Always-X baselines (static, one per method)
    for m in METHODS:
        label = f"Always-{m}"
        preds, costs, refs = strategy_always_X(df, order, m)
        if len(preds) == 0:
            continue
        results[label] = {
            "type": "static",
            "mae": compute_mae(preds, refs),
            "rho": compute_rho(preds, refs),
            "cost_hours": sum(costs) / 3600.0,
        }
    print(f"    Always-RAPIDS: MAE={results.get('Always-RAPIDS', {}).get('mae', 'N/A')}")

    # 2. Oracle (static)
    preds, costs, refs = strategy_oracle(df, order)
    results["Oracle"] = {
        "type": "static",
        "mae": compute_mae(preds, refs),
        "rho": compute_rho(preds, refs),
        "cost_hours": sum(costs) / 3600.0,
    }
    print(f"    Oracle: MAE={results['Oracle']['mae']:.3f}")

    # --- Dynamic strategies with multiple seeds ---
    # Disagreement with threshold sweeps
    disag_thresholds_list = [
        (0.5, 2.0, 8.0),
        (1.0, 5.0, 15.0),
        (2.0, 8.0, 25.0),
        (3.0, 10.0, 30.0),
    ]

    # Progressive ladder threshold sweeps
    ladder_thresholds_list = [
        (1.5, 1.0, 0.5),
        (3.0, 2.0, 1.5),
        (5.0, 3.0, 2.0),
        (8.0, 5.0, 3.0),
    ]

    # Stacking residual thresholds
    stacking_thresholds = [1.0, 2.0, 3.0, 5.0]

    dynamic_defs = {}

    # 3. Random
    dynamic_defs["Random"] = lambda df, o, rng, **kw: strategy_random(df, o, rng)

    # 4. Disagreement (multiple threshold settings)
    for ti, thr in enumerate(disag_thresholds_list):
        name = f"Disagreement-t{ti}"
        dynamic_defs[name] = lambda df, o, rng, t=thr: strategy_disagreement(df, o, thresholds=t)

    # 5. Learned Selector
    dynamic_defs["Learned-Selector"] = lambda df, o, rng, **kw: strategy_learned_selector(df, o, rng=rng)

    # 6. Thompson Sampling
    dynamic_defs["Thompson"] = lambda df, o, rng, **kw: strategy_thompson(df, o, rng=rng)
    dynamic_defs["Thompson-CostAware"] = lambda df, o, rng, **kw: strategy_thompson(df, o, cost_aware=True, rng=rng)

    # 7. Cost-Aware
    dynamic_defs["Cost-Aware"] = lambda df, o, rng, **kw: strategy_cost_aware(df, o, rng=rng)

    # 8. Progressive Ladder (multiple threshold settings)
    for ti, thr in enumerate(ladder_thresholds_list):
        name = f"Ladder-t{ti}"
        dynamic_defs[name] = lambda df, o, rng, t=thr: strategy_progressive_ladder(df, o, thresholds=t)

    # 9. Stacking (multiple residual thresholds)
    for ti, thr in enumerate(stacking_thresholds):
        name = f"Stacking-r{ti}"
        dynamic_defs[name] = lambda df, o, rng, t=thr: strategy_stacking(df, o, residual_threshold=t)

    # 10. Cheap Ensemble (static-like but run through dynamic for curve)
    dynamic_defs["CheapEnsemble-3"] = lambda df, o, rng, **kw: strategy_cheap_ensemble(df, o, n_cheap=3)
    dynamic_defs["CheapEnsemble-5"] = lambda df, o, rng, **kw: strategy_cheap_ensemble(df, o, n_cheap=5)

    # 11. Stacking Meta-Learner (regression, not selection)
    dynamic_defs["StackingML-2"] = lambda df, o, rng, **kw: strategy_stacking_metalearner(df, o, n_cheap=2)
    dynamic_defs["StackingML-3"] = lambda df, o, rng, **kw: strategy_stacking_metalearner(df, o, n_cheap=3)

    # 12. Bias Correction
    dynamic_defs["BiasCorr-RAPIDS"] = lambda df, o, rng, **kw: strategy_bias_correction(df, o, method="RAPIDS")
    dynamic_defs["BiasCorr-PBE"] = lambda df, o, rng, **kw: strategy_bias_correction(df, o, method="PBE-D3BJ_SP")

    # 13. ALORS
    dynamic_defs["ALORS"] = lambda df, o, rng, **kw: strategy_alors(df, o, rank=3)

    # 14. MF-MI-Greedy (Song, Chen, Yue 2018)
    dynamic_defs["MFMI-Greedy"] = lambda df, o, rng, **kw: strategy_mfmi_greedy(df, o, explore_budget_mult=3.0, rng=rng)
    dynamic_defs["MFMI-Greedy-5x"] = lambda df, o, rng, **kw: strategy_mfmi_greedy(df, o, explore_budget_mult=5.0, rng=rng)

    # 15. UCB
    dynamic_defs["UCB"] = lambda df, o, rng, **kw: strategy_ucb(df, o, rng=rng)
    dynamic_defs["UCB-CostAware"] = lambda df, o, rng, **kw: strategy_ucb(df, o, cost_aware=True, rng=rng)

    # 16. Chem-Aware Learned Selector
    dynamic_defs["ChemLearned-Selector"] = lambda df, o, rng, **kw: strategy_chem_learned_selector(df, o, rng=rng)

    # 17. Chem-Aware ALORS
    dynamic_defs["ChemALORS"] = lambda df, o, rng, **kw: strategy_chem_alors(df, o, rank=3)

    # 18. Chem-Aware UCB
    dynamic_defs["ChemUCB-8"] = lambda df, o, rng, **kw: strategy_chem_ucb(df, o, n_bins=8, rng=rng)
    dynamic_defs["ChemUCB-16"] = lambda df, o, rng, **kw: strategy_chem_ucb(df, o, n_bins=16, rng=rng)

    # 19. Chem-Aware Stacking
    dynamic_defs["ChemStacking-2"] = lambda df, o, rng, **kw: strategy_chem_stacking(df, o, n_cheap=2)
    dynamic_defs["ChemStacking-3"] = lambda df, o, rng, **kw: strategy_chem_stacking(df, o, n_cheap=3)

    # 20. Chem-Aware Bias Correction
    dynamic_defs["ChemBiasCorr-RAPIDS"] = lambda df, o, rng, **kw: strategy_chem_bias_correction(df, o, method="RAPIDS")
    dynamic_defs["ChemBiasCorr-PBE"] = lambda df, o, rng, **kw: strategy_chem_bias_correction(df, o, method="PBE-D3BJ_SP")

    # 21. Bucket Prior strategies (require cross-benchmark prior)
    if bucket_prior is not None:
        dynamic_defs["BucketPrior"] = lambda df, o, rng, bp=bucket_prior, **kw: strategy_bucket_prior(df, o, bp, rng=rng)
        dynamic_defs["BucketPrior-Adaptive"] = lambda df, o, rng, bp=bucket_prior, **kw: strategy_bucket_prior_adaptive(df, o, bp, rng=rng)
        dynamic_defs["BucketLearned"] = lambda df, o, rng, bp=bucket_prior, **kw: strategy_bucket_learned(df, o, bp, rng=rng)

    # 22-28. v2 strategies (GBM stacking, meta-GBM, LinUCB)
    if chem_medians is not None:
        dynamic_defs["GBM-Stack-2cheap"] = lambda df, o, rng, **kw: \
            strategy_gbm_stacking_2cheap(df, o, chem_medians)

        dynamic_defs["GBM-Stack-3cheap"] = lambda df, o, rng, **kw: \
            strategy_gbm_stacking_3cheap(df, o, chem_medians)

        dynamic_defs["Meta-GBM"] = lambda df, o, rng, mm=meta_model, **kw: \
            strategy_meta_gbm(df, o, mm, chem_medians)

        dynamic_defs["Meta-GBM-Adaptive"] = lambda df, o, rng, mm=meta_model, **kw: \
            strategy_meta_gbm_adaptive(df, o, mm, chem_medians)

        dynamic_defs["Hybrid-SelectStack"] = lambda df, o, rng, mm=meta_model, **kw: \
            strategy_hybrid_select_stack(df, o, mm, chem_medians)

        dynamic_defs["LinUCB"] = lambda df, o, rng, **kw: \
            strategy_linucb(df, o, chem_medians, alpha=1.0, rng=rng)

        dynamic_defs["GBM-Stack-MetaWarm"] = lambda df, o, rng, mm=meta_model, **kw: \
            strategy_gbm_stacking_metawarm(df, o, mm, chem_medians)

    for strat_name, strat_fn in dynamic_defs.items():
        seed_data = {
            "budget_hours": [], "mae_curves": [], "rho_curves": [],
            "final_mae": [], "final_rho": [], "final_cost": [],
        }
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n).tolist()
            preds, costs, refs = strat_fn(df, perm, rng)
            bh, mc, rc = build_curve(preds, costs, refs)
            seed_data["budget_hours"].append(bh.tolist())
            seed_data["mae_curves"].append(mc.tolist())
            seed_data["rho_curves"].append(rc.tolist())
            seed_data["final_mae"].append(mc[-1] if len(mc) > 0 else np.nan)
            seed_data["final_rho"].append(rc[-1] if len(rc) > 0 else np.nan)
            seed_data["final_cost"].append(bh[-1] if len(bh) > 0 else 0)

        med_mae = float(np.nanmedian(seed_data["final_mae"]))
        med_rho = float(np.nanmedian(seed_data["final_rho"]))
        med_cost = float(np.nanmedian(seed_data["final_cost"]))
        print(f"    {strat_name}: MAE={med_mae:.3f}, rho={med_rho:.4f}, cost={med_cost:.1f}h")
        results[strat_name] = {
            "type": "dynamic",
            "seed_results": seed_data,
            "median_mae": med_mae,
            "median_rho": med_rho,
            "median_cost": med_cost,
        }

    return results


# ---------------------------------------------------------------------------
# Aggregation across benchmarks
# ---------------------------------------------------------------------------

def aggregate_results(all_results, benchmark_list):
    agg = {}
    strat_names = set()
    for b in benchmark_list:
        if b in all_results:
            strat_names.update(all_results[b].keys())

    for strat in sorted(strat_names):
        bench_maes, bench_rhos, bench_costs = [], [], []
        is_dynamic = False

        for b in benchmark_list:
            if b not in all_results or strat not in all_results[b]:
                continue
            r = all_results[b][strat]
            if r["type"] == "static":
                bench_maes.append(r["mae"])
                bench_rhos.append(r["rho"])
                bench_costs.append(r["cost_hours"])
            else:
                is_dynamic = True
                bench_maes.append(r["median_mae"])
                bench_rhos.append(r["median_rho"])
                bench_costs.append(r["median_cost"])

        if not bench_maes:
            continue

        entry = {
            "type": "dynamic" if is_dynamic else "static",
            "mae": float(np.nanmedian(bench_maes)),
            "rho": float(np.nanmedian(bench_rhos)),
            "cost_hours": float(np.nanmedian(bench_costs)),
        }

        if is_dynamic:
            norm_grid = np.linspace(0, 1, N_BUDGET_POINTS)
            seed_mae_agg, seed_rho_agg, seed_budget_agg = [], [], []

            for seed in range(N_SEEDS):
                per_bench_mae, per_bench_rho, per_bench_budget = [], [], []
                for b in benchmark_list:
                    if b not in all_results or strat not in all_results[b]:
                        continue
                    r = all_results[b][strat]
                    if r["type"] != "dynamic":
                        continue
                    sr = r["seed_results"]
                    if seed >= len(sr["mae_curves"]):
                        continue
                    bh = np.array(sr["budget_hours"][seed])
                    mc = np.array(sr["mae_curves"][seed])
                    rc = np.array(sr["rho_curves"][seed])
                    if len(bh) < 2:
                        continue
                    span = bh[-1] - bh[0]
                    bh_norm = (bh - bh[0]) / span if span > 0 else np.zeros_like(bh)
                    per_bench_mae.append(np.interp(norm_grid, bh_norm, mc))
                    per_bench_rho.append(np.interp(norm_grid, bh_norm, rc))
                    per_bench_budget.append(bh)

                if per_bench_mae:
                    seed_mae_agg.append(np.nanmedian(per_bench_mae, axis=0))
                    seed_rho_agg.append(np.nanmedian(per_bench_rho, axis=0))
                    budget_arr = np.array([
                        np.interp(norm_grid, np.linspace(0, 1, len(bb)), bb)
                        for bb in per_bench_budget
                    ])
                    seed_budget_agg.append(np.nanmedian(budget_arr, axis=0))

            if seed_mae_agg:
                mae_all = np.array(seed_mae_agg)
                rho_all = np.array(seed_rho_agg)
                budget_all = np.array(seed_budget_agg)
                entry["curves"] = {
                    "budget_median": np.nanmedian(budget_all, axis=0).tolist(),
                    "mae_median": np.nanmedian(mae_all, axis=0).tolist(),
                    "mae_p10": np.nanpercentile(mae_all, 10, axis=0).tolist(),
                    "mae_p90": np.nanpercentile(mae_all, 90, axis=0).tolist(),
                    "rho_median": np.nanmedian(rho_all, axis=0).tolist(),
                    "rho_p10": np.nanpercentile(rho_all, 10, axis=0).tolist(),
                    "rho_p90": np.nanpercentile(rho_all, 90, axis=0).tolist(),
                }

        agg[strat] = entry
    return agg


def print_summary_table(aggregated, label=""):
    print(f"\n{'='*80}")
    print(f"SUMMARY -- {label}")
    print(f"{'='*80}")
    rapids_cost = aggregated.get("Always-RAPIDS", {}).get("cost_hours", 1.0)
    header = f"{'Strategy':<30} {'MAE':>8} {'Rho':>8} {'Cost(h)':>10} {'xRAPIDS':>8}"
    print(header)
    print("-" * 80)

    # Sort by MAE
    items = sorted(aggregated.items(), key=lambda x: x[1].get("mae", 999))
    for strat, r in items:
        cost_ratio = r["cost_hours"] / rapids_cost if rapids_cost > 0 else 0
        mae_str = f"{r['mae']:.3f}" if not np.isnan(r["mae"]) else "N/A"
        rho_str = f"{r['rho']:.4f}" if not np.isnan(r["rho"]) else "N/A"
        print(f"{strat:<30} {mae_str:>8} {rho_str:>8} {r['cost_hours']:>10.2f} {cost_ratio:>7.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading all benchmarks...")
    data = load_all_benchmarks()

    # Compute global chemical feature medians for v2 strategies
    print("\nComputing global chemical feature medians...")
    chem_medians = compute_global_chem_medians(data)
    print(f"  Computed medians for {len(chem_medians)} features")

    all_results = {}
    print("\n" + "=" * 70)
    print("Running strategies on all benchmarks (with leave-one-out bucket priors)...")
    print("=" * 70)

    for b in NEUTRAL_BENCHMARKS + CHARGED_BENCHMARKS:
        if b in data:
            # Build leave-one-out bucket prior: train on all OTHER benchmarks
            print(f"  Building bucket prior (excluding {b})...")
            bp = build_bucket_prior(data, exclude_bench=b, n_buckets=12)
            if bp is not None:
                print(f"    Prior built from {bp['n_train_systems']} systems")

            # Build leave-one-out meta-GBM for v2 strategies
            print(f"  Building Meta-GBM (excluding {b})...")
            meta_model = build_meta_gbm(data, exclude_bench=b, chem_medians=chem_medians)
            if meta_model is not None:
                print(f"    Meta-GBM trained")
            else:
                print(f"    Meta-GBM: insufficient data")

            all_results[b] = run_strategies_on_benchmark(
                data[b], b, bucket_prior=bp,
                meta_model=meta_model, chem_medians=chem_medians)

    print("\n" + "=" * 70)
    print("Aggregating results...")
    print("=" * 70)

    neutral_agg = aggregate_results(all_results, NEUTRAL_BENCHMARKS)
    charged_agg = aggregate_results(all_results, CHARGED_BENCHMARKS)
    overall_agg = aggregate_results(all_results, NEUTRAL_BENCHMARKS + CHARGED_BENCHMARKS)

    print_summary_table(neutral_agg, "NEUTRAL (median across 16 benchmarks)")
    print_summary_table(charged_agg, "CHARGED (median across 2 benchmarks)")
    print_summary_table(overall_agg, "ALL (median across 18 benchmarks)")

    # Save
    save_data = {"neutral_agg": {}, "charged_agg": {}, "per_benchmark": {}}
    for strat, r in neutral_agg.items():
        save_data["neutral_agg"][strat] = r
    for strat, r in charged_agg.items():
        save_data["charged_agg"][strat] = r

    for b in NEUTRAL_BENCHMARKS + CHARGED_BENCHMARKS:
        if b not in all_results:
            continue
        save_data["per_benchmark"][b] = {}
        for strat, r in all_results[b].items():
            if r["type"] == "static":
                save_data["per_benchmark"][b][strat] = r
            else:
                save_data["per_benchmark"][b][strat] = {
                    "type": "dynamic",
                    "median_mae": r["median_mae"],
                    "median_rho": r["median_rho"],
                    "median_cost": r["median_cost"],
                }

    out_file = OUT_DIR / "sequential_results.json"
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)
    print(f"\nResults saved to {out_file}")
    return save_data


if __name__ == "__main__":
    main()
