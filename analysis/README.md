# RAPIDS fidelity-selection analysis code

Two self-contained analysis blocks that support the RAPIDS paper's
multi-fidelity method-selection study. Both sit *downstream* of the core RAPIDS
engine: RAPIDS (and the reference DFT / CREST pathways) produce per-system
interaction energies and wall-clock times; the code here consumes those
pre-computed tables and asks **"which fidelity should we have paid for on each
system?"** Neither block runs an MLIP or a DFT calculation — they operate on
offline tables — so they do **not** depend on `fairchem-core`, `ase`, `pyscf`,
etc.

The five fidelity *arms* studied here are:
`RAPIDS`, `PBE-D3BJ_SP`, `PBE-D3BJ_GeoSP`, `CREST_xTB`, `CREST_xTB_DFT`
(the offline-replay block additionally splits DFT into PBE / wB97X-D3BJ / wB97M-V
single-point and geo+SP variants, for 9 arms total).

## The two blocks

| Folder | What it is | Entry points |
|--------|-----------|--------------|
| [`offline_replay/`](offline_replay/) | **Sequential multi-fidelity bandit replay.** A stream of dimer systems arrives one at a time; each of ~56 strategies (bandits, stacking, chemistry-aware, meta-GBM, ALORS, and static baselines) picks a fidelity arm per system under a cost budget, using only offline-stored results. Produces learning curves + Pareto frontiers. | `sequential_bandit.py` (run), `plot_sequential.py` (figures) |
| [`rapids_select/`](rapids_select/) | **RAPIDS-Select**: a *one-shot*, cost-aware fidelity selector. For each system it commits the single arm minimizing `cost(a) + lambda * predicted_error_a(x)` (lambda = 800), using 5 per-arm XGBoost error regressors over 156 cheap features. Includes training, the cost-aware argmin policy, literature baselines (SATzilla / FrugalML / ALORS / multiclass), generalization evaluations, ablations, and a conformal risk-control analysis layer. | `src/56_train_final_v6.py` (train), `src/72_oneshot_baselines_cached.py` (compare) — see its README |

**Relationship.** The offline replay is the *sequential/online* framing (a strategy
adapts as it sees more systems and may pay for several arms per system).
RAPIDS-Select is the *one-shot/committing* framing (exactly one arm per system,
chosen up front from cheap features). They share the same underlying per-system
energy/time tables and the same set of physical fidelity arms; they are reported
side by side in the paper.

## Headline scientific finding (RAPIDS-Select)

A **transfer asymmetry**: whether a cheap arm will *catastrophically* fail on a
system is highly predictable (per-arm catastrophe-classifier AUC 0.85–0.98), but
the *magnitude* of its error is not (Spearman rho 0.56–0.65). Catastrophe
**occurrence** transfers across chemistry; error **size** does not. This is why
RAPIDS-Select routes on predicted risk rather than trying to regress exact errors.

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+. See per-block READMEs for which packages each script actually needs.

## Data / artifacts are NOT bundled (config JSONs are)

No result dumps, caches, trained model weights, figures, or the large
feature-matrix / benchmark CSVs are included — only code, plus three small
**config** JSONs for RAPIDS-Select (feature-column list, per-arm cost table,
thresholds/routing under `rapids_select/models/`; these are configuration, **not**
trained weights). Each block's README documents the **schema and expected
location** of every remaining input so you can supply your own harvested tables
and produce the trained weights by running the training script. Nothing here runs
end-to-end until you drop in those inputs; the code is provided for inspection and
reproduction, not as a turn-key binary.

## Naming note

The public method is simply **"RAPIDS-Select"**. Some script filenames and cached
artifact names carry internal iteration tags (`v5`, `v6`, `v8`, `v9`, `v10`,
`vhpo`). These are **kept only because scripts reference each other and the model
directory by those literal names** — renaming would break the cross-file wiring.
They do not denote separate public methods. In particular `v6` is the router and
`v9` is the conformal cost-risk analysis layer; treat both as parts of
RAPIDS-Select.
