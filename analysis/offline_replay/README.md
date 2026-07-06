# Offline replay — sequential multi-fidelity bandit

Sequential (online) framing of fidelity selection. A stream of molecular-dimer
systems is presented one at a time; each strategy must pick a computational
fidelity *arm* for the current system before seeing the next, under a growing
cost budget. Everything is **offline replay**: results at all 9 fidelity levels
are pre-computed and stored per system, so a strategy "runs" an arm by looking up
its stored energy + wall-clock time. This gives a controlled comparison of ~56
strategies over 18 benchmarks without re-running any chemistry.

## Files (core code)

| File | Purpose |
|------|---------|
| `sequential_bandit.py` | The replay engine (~3.1k lines). Benchmark loading, all ~56 strategies (Oracle + Always-X baselines; UCB / Thompson / cost-aware bandits; Learned Selector; Disagreement; Progressive Ladder; ALORS; Stacking / Cheap-Ensemble / Bias-Correction; ChemUCB and other chemistry-aware variants; Bucket-Prior; Meta-GBM / GBM-Stacking; LinUCB), leave-one-benchmark-out priors/meta-models, 10-seed replays, 30-point learning curves, and result aggregation. |
| `plot_sequential.py` | Reproduces the paper's replay figures from the aggregated results JSON. Optional (visualization only). |

No local module imports; both files are standalone. Dependencies: `numpy`,
`pandas`, `scipy`, `scikit-learn` (engine); `matplotlib` (plots only).

## Entry points

```bash
python sequential_bandit.py      # full replay, ~30 min; writes results JSON + CSV
python plot_sequential.py        # 6 figures from the merged results JSON
```

## Required external inputs (NOT bundled)

`sequential_bandit.py` reads a benchmark data root, hard-coded near the top as:

```python
BASE = Path("~/benchmarking/collection_finished_all_fidelity")   # expanduser'd
NEUTRAL_DIR = BASE / "neutral"
CHARGED_DIR = BASE / "charged"
OUT_DIR     = BASE / "offline_replay" / "results_sequential"      # outputs
```

Point `BASE` at your own data root. Expected layout, per benchmark:

```
<BASE>/<neutral|charged>/<BENCH>/<BENCH>.csv       # per-system energies + times
<BASE>/<neutral|charged>/<BENCH>/systems/...       # probe.meta.json + target.meta.json per system
```

- **`<BENCH>.csv`** — one row per system. Required column `Reference` (CCSD(T)/CBS
  or equivalent, kcal/mol). Then, for each of the 9 methods
  (`RAPIDS`, `PBE-D3BJ_SP`, `wB97X-D3BJ_SP`, `wB97M-V_SP`, `PBE-D3BJ_GeoSP`,
  `wB97X-D3BJ_GeoSP`, `wB97M-V_GeoSP`, `CREST_xTB`, `CREST_xTB_DFT`): an energy
  column `<method>` (kcal/mol) and a wall-clock column `<method>_time` (seconds).
  Missing methods/columns are tolerated and filled with NaN.
- **`systems/.../probe.meta.json` + `target.meta.json`** — per-fragment chemical
  descriptors used by the chemistry-aware strategies. Fields read: PubChem
  `molecular_weight, heavy_atom_count, xlogp, tpsa, complexity,
  h_bond_donor_count, h_bond_acceptor_count, rotatable_bond_count, formal_charge`
  and RDKit `aromatic_atom_count, ring_count, fsp3`. Missing files degrade the
  chemistry-aware strategies gracefully (median imputation).

Benchmark names are fixed in the script: 16 neutral (A24, S66, X40, HB300SPX,
HB375, SH250, D1200_Halogens, D1200_HBCNO, D1200_PS, BFDb_BBI, BFDb_HSG,
BFDb_NBC1, BFDb_SSI_{dispersion,mixed,other,polar}) and 2 charged (IHB100,
BFDb_SSI_charged).

`plot_sequential.py` reads `results_sequential/sequential_results_merged.json`
(relative to the script) — i.e. the aggregated output of a replay run. Produce it
first (or drop your own in place); it is intentionally not shipped.

## Key knobs (top of `sequential_bandit.py`)

`N_SEEDS = 10` (random system orderings), `ERROR_CAP = 50.0` kcal/mol (per-system
abs-error cap), `N_BUDGET_POINTS = 30` (learning-curve checkpoints).

## Excluded from this package

`legacy/`, `results_sequential/` (JSON/CSV dumps), `plots_sequential/` (PNGs),
`__pycache__/`. Regenerate all of these by running the two scripts against your data.
