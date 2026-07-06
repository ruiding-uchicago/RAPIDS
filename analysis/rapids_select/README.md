# RAPIDS-Select — one-shot cost-aware fidelity selector

One-shot (committing) framing of fidelity selection. For each system, exactly one
arm is chosen up front from cheap features, by minimizing

```
arm*(x) = argmin_a [ cost(a) + lambda * pred_err_a(x) ],   lambda = 800
```

over the 5 arms `RAPIDS, PBE-D3BJ_SP, PBE-D3BJ_GeoSP, CREST_xTB, CREST_xTB_DFT`.
`pred_err_a(x)` comes from 5 per-arm XGBoost error regressors trained on 156 cheap
features (UMA scan/variance-guard descriptors, geometry descriptors, RDKit chem
descriptors). Training is charged-aware with no charge short-circuit.

Scientific result — **transfer asymmetry**: catastrophe *occurrence* is
predictable (per-arm classifier AUC 0.85–0.98) but error *magnitude* is not
(Spearman rho 0.56–0.65).

## Directory layout the scripts assume

Every script resolves the package root as `Path(__file__).resolve().parents[1]`,
i.e. the parent of `src/`, and reads/writes sibling folders. Keep this layout:

```
rapids_select/
├── src/            <- the code in this package (all 34 scripts)
├── models/         <- SHIPPED config JSONs (see below) + YOU add trained weights
├── data/           <- YOU supply: feature-matrix CSVs (see below)
├── cache/          <- auto-created: per-fold prediction caches (*.pkl)
├── results/        <- auto-created: JSON/CSV metric dumps
└── figures/        <- auto-created: PNGs (optional figure scripts)
```

`cache/`, `results/`, `figures/` are created on demand by the scripts. The small
**config** JSONs under `models/` are shipped with this package; you must supply
`data/` (feature matrices) and the trained model weights — see "Required inputs".

## Script map

Filenames keep their internal numeric/iteration tags because scripts reference
each other and the model directory by these literal names; `25c_baselines.py` in
particular is imported by ~everything via its exact filename. Do **not** rename.
`v6` = the router, `v9` = the conformal risk-control analysis layer — all part of
RAPIDS-Select (see naming note at the bottom).

**Shared core (imported everywhere)**
- `25c_baselines.py` — policy definitions, selector loader (`load_selector`,
  `make_rapids_select`), feature encoding (`encode_features`), and
  `evaluate_policy`. Loaded dynamically by many scripts via `importlib`.

**Feature building (needs upstream harvest — see inputs)**
- `25a_build_feature_matrix.py` — in-distribution matrix (~5.5k systems x 156 feat).
- `49_build_p02_feature_matrix.py` — charged ("P0.2") matrix (~1.3k systems).
  Imports a module **`compute_geometry_descriptors`** that is part of the upstream
  harvest tooling and is **not** included here (see inputs).

**RAPIDS-Select training + policy**
- `54_build_v6_caches.py` — per-fold (LOBO in-dist / CV charged) prediction caches → `cache/v6_*.pkl`.
- `56_train_final_v6.py` — trains the 5 per-arm regressors → `models/rapids_select_v6_final/`.
- `55_v6_policy_sweep.py`, `57_v6_lambda_finetune.py` — cost-aware `cost + lambda*err` argmin sweep / lambda tuning off the caches.

**Per-arm catastrophe layer (dependency of baselines + risk control)**
- `62_v8_catastrophe_caches.py` — per-arm catastrophe classifiers + regressors → `cache/v8_*.pkl`. (This is a cache builder, not a routing method.)

**Science / diagnostics**
- `52_charged_headroom_diagnostic.py` — magnitude-vs-occurrence (the transfer asymmetry).
- `53_charged_aware_selector.py` — charged-aware detector (AUC lift).

**Generalization evaluations (reproduce the method-comparison table)**
- `68_indist_bootstrap_ci.py` — leave-one-benchmark-out + bootstrap CI.
- `77_lofo_v6.py` — leave-one-feature-group-out (retrains).
- `69_charged_multiseed_cv.py` — charged multi-seed CV.
- `51_p02_prospective_eval.py` — prospective eval with the frozen policy (reads manifest, does not retune).

**Literature / competitor baselines**
- `72_oneshot_baselines_cached.py` — SATzilla-style, FrugalML-style, defer, global-best (off `cache/v8_*.pkl`).
- `73_alors_baseline.py` — ALORS (low-rank algorithm selection).
- `78_multiclass_selector_baseline.py` — direct multiclass arm classifier.

**Ablations**
- `74_ablation_retrain.py` — feature-group ablation (retrains; needs `cache/feature_groups.pkl`).
- `75_mechanism_ablation_cached.py` — mechanism ablation off caches.
- `76_ablation_analysis.py` — ablation table + figure (reads `results/ablation_retrain.json`).

**HPO ceiling / rigor**
- `67_hpo_transfer_aware.py` — Optuna transfer-aware search → `models/hpo_best.json`.
- `70_build_hpo_caches.py` — caches under the HPO config → `cache/vhpo_*.pkl`.
- `71_hpo_router_eval.py` — HPO-vs-default router comparison.

**Conformal risk-control layer**
- `64_v9_conformal_risk_control.py` — conformal risk control off `cache/v8_*.pkl`.
- `66_v10_group_conditional_crc.py` — group-conditional (charge-conditioned) CRC.

**Charged "gold" subset evaluation**
- `80_charged_gold_eval.py`, `81_charged_gold_from_cache.py`,
  `82_gold_charged_full_table.py`, `83_gold_retrained_baselines.py`.

**Figures (OPTIONAL — visualization only; need `matplotlib`, some need `adjustText`)**
- `_paper_style.py` (shared style, imported by the figure scripts),
  `58_v6_figure.py`, `65_v9_frontier_figure.py`, `79_transfer_asymmetry_figure.py`,
  `76_ablation_analysis.py` (also emits a figure), `90_composite_main_figure.py`.

## Entry points

```bash
# Train the product model (config JSON is shipped; you supply data/):
python3 src/56_train_final_v6.py

# Build the prediction caches, then evaluate policies / baselines off them:
python3 src/54_build_v6_caches.py
python3 src/62_v8_catastrophe_caches.py
python3 src/68_indist_bootstrap_ci.py        # LOBO + CI
python3 src/72_oneshot_baselines_cached.py   # literature baselines
python3 src/77_lofo_v6.py                     # LOFO (retrains, ~minutes)
```

## Required inputs (NOT bundled — supply your own)

### 1. Feature matrices → `data/`
- `data/selector_feature_matrix.csv` — in-distribution (neutral+charged in-dist),
  ~5.5k rows. Read by nearly every script.
- `data/p02_selector_feature_matrix.csv` — held-out charged ("P0.2"), ~1.3k rows.

Both are wide tables with one row per system. Expected columns:
- `system_id`, `benchmark` — identifiers.
- `Reference` — ground-truth interaction energy (kcal/mol); rows with NaN are dropped.
- Per-arm energy columns named exactly by arm: `RAPIDS`, `CREST_xTB`,
  `PBE-D3BJ_SP`, `PBE-D3BJ_GeoSP`, `CREST_xTB_DFT` (kcal/mol), plus matching
  `<arm>_time` wall-clock columns (seconds).
- The 156 feature columns listed in the manifest (below): UMA scan/variance-guard
  descriptors (`UMA__*`), geometry descriptors, and RDKit chem descriptors.

To (re)build these from raw harvest, `25a_` / `49_` are provided, but they read
absolute upstream harvest paths (e.g. `~/benchmarking/scan_summary_harvest_*`,
`RAPIDS_NCI_charged_rescue_*`) and `49_` imports the harvest-side
`compute_geometry_descriptors` module. That upstream tooling is out of scope for
this package; if you already have the two CSVs, you can skip the build scripts.

### 2. Config JSONs → `models/`  (SHIPPED with this package)

Three small **config** JSONs are included in this package (they are configuration
— feature-column list, per-arm cost table, thresholds/routing — **not** trained
weights):

- `models/rapids_select_v5_final/manifest.json` — read by 13 scripts.
- `models/rapids_select_v1/feature_columns.json` + `arm_costs.json` — read by the
  smoke test in `25c_baselines.py`.

Schema of `manifest.json`:

```jsonc
{
  "feature_cols": ["UMA__n_anchors", ...],          // 156 feature column names, in order
  "arm_costs":  {"RAPIDS":230.35,"CREST_xTB":1549.6,"PBE-D3BJ_SP":239.5,
                 "PBE-D3BJ_GeoSP":479.45,"CREST_xTB_DFT":2041.3},  // seconds
  "arms":       ["RAPIDS","CREST_xTB","PBE-D3BJ_SP","PBE-D3BJ_GeoSP","CREST_xTB_DFT"],
  "thresholds": {"tau5":0.85,"tau10":0.5,"tau20":0.42},            // used by frozen policy
  "routing":    {"charge_arm":"PBE-D3BJ_GeoSP","arm5":"PBE-D3BJ_SP",
                 "arm10":"CREST_xTB","arm20":"CREST_xTB_DFT"}
}
```

The **trained model weights are NOT shipped** — only the config JSONs above.
`56_train_final_v6.py` writes the actual trained regressors (`arm_<ARM>.json`,
XGBoost) to `models/rapids_select_v6_final/`; the catastrophe classifiers
(`cat_p{5,10,20}.json`) come from `62_`. Regenerate these by running training,
or drop your own in place.

### 3. Prediction caches → `cache/` (auto-generated)
`cache/v6_*.pkl` (from `54_`), `cache/v8_*.pkl` (from `62_`), `cache/vhpo_*.pkl`
(from `70_`), `cache/feature_groups.pkl` (feature-group map for `74_`). Each pkl is
`{'arms': [...], 'fixed_cost': {...}, 'caches': {fold: {...per-arm preds, true_err, cost...}}}`.
The evaluation scripts read these so they run in seconds without retraining.

## Shipped vs excluded

**Shipped:** all 34 `src/` scripts, this README, and the 3 config JSONs under
`models/` (`rapids_select_v5_final/manifest.json`,
`rapids_select_v1/{feature_columns,arm_costs}.json`).

**Excluded:** `data/` feature-matrix CSVs; all **trained model weights**
(`models/**/arm_*.json`, `cat_*.json`) and any `models/rapids_select_v6_final/`
contents; `cache/*.pkl`; `results/` and `figures/` dumps; `__pycache__/`; and the
upstream harvest tooling (`compute_geometry_descriptors` and the raw harvest dirs).

## Naming note
The public method is **RAPIDS-Select**. Internal tags in filenames/caches
(`v5/v6/v8/v9/v10/vhpo`) are load-bearing cross-references, not separate methods.
The shipped `models/rapids_select_v1` and `_v5_final` JSONs supply only the shared
feature-column list + cost table + thresholds (config); the product model weights
live in `models/rapids_select_v6_final/` and are produced by running training.
