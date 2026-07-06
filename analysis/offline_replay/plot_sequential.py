#!/usr/bin/env python3
"""
Plot sequential multi-fidelity bandit replay results — 61 strategies.
Clean, publication-quality figures.

Naming rules:
  - RAPIDS-based methods: "RAPIDS PBE-D3BJ SP", "RAPIDS wB97M-V GeoSP", etc.
  - CREST methods keep original: "CREST xTB", "CREST xTB+DFT"
  - Same-family strategies: only show best hyperparameter, no suffix
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path(__file__).parent
RESULTS_FILE = BASE / "results_sequential" / "sequential_results_merged.json"
PLOTS_DIR = BASE / "plots_sequential"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Display name mapping: internal key -> clean label
# ---------------------------------------------------------------------------
DISPLAY_MAP = {
    # Always-X baselines
    "Always-RAPIDS":            "RAPIDS",
    "Always-PBE-D3BJ_SP":      "RAPIDS PBE-D3BJ SP",
    "Always-wB97X-D3BJ_SP":    "RAPIDS wB97X-D3BJ SP",
    "Always-wB97M-V_SP":       "RAPIDS wB97M-V SP",
    "Always-PBE-D3BJ_GeoSP":   "RAPIDS PBE-D3BJ GeoSP",
    "Always-wB97X-D3BJ_GeoSP": "RAPIDS wB97X-D3BJ GeoSP",
    "Always-wB97M-V_GeoSP":    "RAPIDS wB97M-V GeoSP",
    "Always-CREST_xTB":        "CREST xTB",
    "Always-CREST_xTB_DFT":    "CREST xTB+DFT",
    "Oracle":                   "Oracle",
    # Dynamic strategies (best of each family)
    "Learned-Selector":         "Learned Selector",
    "ChemUCB-8":                "ChemUCB",
    "ChemLearned-Selector":     "Chem Learned Selector",
    "BucketPrior-Adaptive":     "Bucket Prior Adaptive",
    "BucketLearned":            "Bucket Learned",
    "UCB":                      "UCB",
    "Thompson":                 "Thompson",
    "Cost-Aware":               "Cost-Aware",
    "ALORS":                    "ALORS",
    "ChemALORS":                "ChemALORS",
    "Random":                   "Random",
    "Disagreement-t1":          "Disagreement",
    "Ladder-t1":                "Progressive Ladder",
    "Stacking-r0":              "Stacking",
    "StackingML-2":             "Stacking ML",
    "CheapEnsemble-3":          "Cheap Ensemble",
    "BiasCorr-RAPIDS":          "Bias Correction",
    "ChemBiasCorr-RAPIDS":      "Chem Bias Correction",
    "ChemBiasCorr-PBE":         "Chem Bias Correction",
    "BiasCorr-PBE":             "Bias Correction",
    "ChemStacking-3":           "Chem Stacking",
    "MFMI-Greedy":              "MF-MI Greedy",
    "BucketPrior":              "Bucket Prior",
    "GBM-Stack-2cheap":         "GBM Stacking",
    "GBM-Stack-MetaWarm":       "GBM Meta-Warm",
    "Meta-GBM-Adaptive":        "Meta-GBM Adaptive",
    "Hybrid-SelectStack":       "Hybrid Select+Stack",
    "LinUCB":                   "LinUCB",
    "Meta-GBM":                 "Meta-GBM",
}

def display_name(key):
    return DISPLAY_MAP.get(key, key)


# Duplicate hyperparams to skip (keep only best per family)
SKIP_DUPLICATES = {
    "ChemUCB-16", "ChemUCB-4",
    "GBM-Stack-3cheap",
    "Stacking-r1", "Stacking-r2", "Stacking-r3",
    "StackingML-3",
    "CheapEnsemble-2", "CheapEnsemble-4",
    "BiasCorr-PBE-D3BJ_SP", "BiasCorr-wB97X-D3BJ_SP", "BiasCorr-PBE",
    "ChemBiasCorr-PBE-D3BJ_SP", "ChemBiasCorr-RAPIDS",
    "ChemStacking-2",
    "Disagreement-t0", "Disagreement-t2", "Disagreement-t3",
    "Ladder-t0", "Ladder-t2", "Ladder-t3",
    "BucketPrior",
}


# ---------------------------------------------------------------------------
# Strategy categorization (no version numbers)
# ---------------------------------------------------------------------------
CAT_STYLES = {
    "Oracle":       {"c": "#2ca02c", "marker": "*", "s": 250, "z": 10},
    "Always-X":     {"c": "#7f7f7f", "marker": "s", "s": 50,  "z": 3},
    "Bandit":       {"c": "#1f77b4", "marker": "o", "s": 70,  "z": 5},
    "Chem/Bucket":  {"c": "#ff7f0e", "marker": "D", "s": 70,  "z": 6},
    "GBM/Meta":     {"c": "#d62728", "marker": "^", "s": 90,  "z": 7},
}

def categorize(strat):
    if strat == "Oracle":
        return "Oracle"
    if strat.startswith("Always-"):
        return "Always-X"
    if strat.startswith("Chem") or strat.startswith("Bucket"):
        return "Chem/Bucket"
    if any(strat.startswith(p) for p in ("GBM", "Meta-GBM", "Hybrid", "LinUCB")):
        return "GBM/Meta"
    return "Bandit"

def get_val(r, key):
    return r.get(key, r.get(f"median_{key}", np.nan))

LEGEND_ELS = [Line2D([0], [0], marker=st["marker"], color="w",
                     markerfacecolor=st["c"], markersize=8, label=cat)
              for cat, st in CAT_STYLES.items()]


# ---------------------------------------------------------------------------
# 2x2 learning curves
# ---------------------------------------------------------------------------

STATIC_2x2 = {
    "Always-RAPIDS":            {"color": "#2196F3", "marker": "o",  "ms": 7,  "label": "RAPIDS"},
    "Always-PBE-D3BJ_SP":      {"color": "#4CAF50", "marker": "s",  "ms": 5,  "label": "RAPIDS PBE-D3BJ SP"},
    "Always-wB97X-D3BJ_SP":    {"color": "#8BC34A", "marker": "s",  "ms": 5,  "label": "RAPIDS wB97X-D3BJ SP"},
    "Always-wB97M-V_SP":       {"color": "#CDDC39", "marker": "s",  "ms": 5,  "label": "RAPIDS wB97M-V SP"},
    "Always-PBE-D3BJ_GeoSP":   {"color": "#FF9800", "marker": "D",  "ms": 5,  "label": "RAPIDS PBE-D3BJ GeoSP"},
    "Always-wB97X-D3BJ_GeoSP": {"color": "#FF5722", "marker": "D",  "ms": 5,  "label": "RAPIDS wB97X-D3BJ GeoSP"},
    "Always-wB97M-V_GeoSP":    {"color": "#F44336", "marker": "D",  "ms": 7,  "label": "RAPIDS wB97M-V GeoSP"},
    "Always-CREST_xTB":        {"color": "#9C27B0", "marker": "^",  "ms": 7,  "label": "CREST xTB"},
    "Always-CREST_xTB_DFT":    {"color": "#673AB7", "marker": "P",  "ms": 7,  "label": "CREST xTB+DFT"},
    "Oracle":                   {"color": "#2ca02c", "marker": "*",  "ms": 12, "label": "Oracle"},
}

DYNAMIC_2x2 = {
    # Top 3 (emphasized)
    "ChemUCB-8":                {"color": "#e41a1c", "lw": 3.0, "ls": "-",  "label": "ChemUCB"},
    "BucketPrior-Adaptive":     {"color": "#ff7f00", "lw": 3.0, "ls": "-",  "label": "Bucket Prior Adaptive"},
    "GBM-Stack-2cheap":         {"color": "#984ea3", "lw": 3.0, "ls": "-",  "label": "GBM Stacking"},
    # Other representative strategies
    "Learned-Selector":         {"color": "#1f77b4", "lw": 2.0, "ls": "-",  "label": "Learned Selector"},
    "UCB":                      {"color": "#ff9896", "lw": 1.5, "ls": "-",  "label": "UCB"},
    "Thompson":                 {"color": "#d62728", "lw": 1.5, "ls": "--", "label": "Thompson"},
    "ALORS":                    {"color": "#756bb1", "lw": 1.5, "ls": "-",  "label": "ALORS"},
    "Meta-GBM-Adaptive":        {"color": "#e7298a", "lw": 2.0, "ls": "-.", "label": "Meta-GBM Adaptive"},
    "Disagreement-t1":          {"color": "#e377c2", "lw": 1.5, "ls": "--", "label": "Disagreement"},
    "Random":                   {"color": "#bcbd22", "lw": 1.0, "ls": ":",  "label": "Random"},
}

BAR_STRATEGIES = [
    ("Always-RAPIDS",           "#2196F3"),
    ("Always-wB97M-V_GeoSP",   "#F44336"),
    ("Always-CREST_xTB_DFT",   "#673AB7"),
    ("Oracle",                  "#2ca02c"),
    ("ChemUCB-8",               "#e41a1c"),
    ("BucketPrior-Adaptive",    "#ff7f00"),
    ("GBM-Stack-2cheap",        "#984ea3"),
    ("Learned-Selector",        "#1f77b4"),
    ("Meta-GBM-Adaptive",       "#e7298a"),
    ("UCB",                     "#ff9896"),
]

# Which strategies get text labels in Pareto
TOP_LABEL = {"Oracle", "Always-RAPIDS", "Always-wB97M-V_GeoSP",
             "ChemUCB-8", "Learned-Selector", "BucketPrior-Adaptive",
             "GBM-Stack-2cheap", "Meta-GBM-Adaptive", "GBM-Stack-MetaWarm"}


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_learning_panel(ax, agg, metric, ylabel, title=None, show_legend=True):
    x_vals_all = []

    for strat, style in STATIC_2x2.items():
        if strat not in agg:
            continue
        r = agg[strat]
        val = get_val(r, metric)
        cost = r.get("cost_hours", 0)
        if np.isnan(val) or cost <= 0:
            continue
        ax.plot(cost, val, marker=style["marker"], color=style["color"],
                ms=style["ms"], zorder=5, markeredgecolor="black",
                markeredgewidth=0.4, label=style["label"], alpha=0.85)
        x_vals_all.append(cost)

    for strat, style in DYNAMIC_2x2.items():
        if strat not in agg:
            continue
        r = agg[strat]
        if "curves" not in r:
            continue
        curves = r["curves"]
        budget = np.array(curves["budget_median"])
        med = np.array(curves[f"{metric}_median"])
        p10 = np.array(curves[f"{metric}_p10"])
        p90 = np.array(curves[f"{metric}_p90"])
        valid = ~np.isnan(med) & ~np.isnan(budget) & (budget > 0)
        if not np.any(valid):
            continue
        budget, med, p10, p90 = budget[valid], med[valid], p10[valid], p90[valid]
        ax.plot(budget, med, color=style["color"], lw=style["lw"],
                ls=style["ls"], label=style["label"], zorder=3)
        ax.fill_between(budget, p10, p90, color=style["color"], alpha=0.10, zorder=2)
        x_vals_all.extend([budget[0], budget[-1]])

    if x_vals_all:
        ax.set_xscale("log")
        ax.set_xlim(1.0, max(x_vals_all) * 1.3)

    ax.set_xlabel("Cumulative Cost (hours)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, which="both", linewidth=0.5)
    ax.tick_params(labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    if show_legend and handles:
        ax.legend(handles, labels, fontsize=6, loc="best",
                 framealpha=0.85, edgecolor="gray", fancybox=False, ncol=1)


def plot_2x2(neutral_agg, charged_agg):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=200)
    fig.suptitle("Sequential Multi-Fidelity Offline Replay — 9 Arms, 56 Strategies\n"
                 "(median across benchmarks, 10 seeds, 10th-90th percentile bands)",
                 fontsize=14, fontweight="bold", y=0.98)

    plot_learning_panel(axes[0, 0], neutral_agg, "mae",
                        "MAE (kcal/mol)", "Neutral (16 benchmarks) — MAE")
    plot_learning_panel(axes[0, 1], neutral_agg, "rho",
                        "Spearman rho", "Neutral (16 benchmarks) — Spearman rho")
    plot_learning_panel(axes[1, 0], charged_agg, "mae",
                        "MAE (kcal/mol)", "Charged (2 benchmarks) — MAE")
    plot_learning_panel(axes[1, 1], charged_agg, "rho",
                        "Spearman rho", "Charged (2 benchmarks) — Spearman rho")

    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0)
    out = PLOTS_DIR / "sequential_bandit_2x2.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_replay_frontier(agg, subset_label, n_bench, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.6), dpi=200)

    plot_learning_panel(
        axes[0], agg, "mae", "MAE (kcal/mol)",
        None,
        show_legend=False,
    )
    plot_learning_panel(
        axes[1], agg, "rho", "Spearman rho",
        None,
        show_legend=True,
    )

    plt.tight_layout(pad=1.2, w_pad=1.4)
    out = PLOTS_DIR / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out}")
    plt.close()


def plot_per_benchmark_bars(per_bench):
    if not per_bench:
        return
    strats = [s for s, _ in BAR_STRATEGIES]
    colors = [c for _, c in BAR_STRATEGIES]
    labels = [display_name(s) for s in strats]

    bench_names = sorted(per_bench.keys())
    n_bench = len(bench_names)
    n_strat = len(strats)
    width = 0.8 / n_strat
    x = np.arange(n_bench)

    fig, ax = plt.subplots(figsize=(18, 7), dpi=150)

    for i, (strat, col) in enumerate(zip(strats, colors)):
        vals = []
        for b in bench_names:
            r = per_bench[b].get(strat, {})
            vals.append(get_val(r, "mae"))
        offset = (i - n_strat / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=labels[i],
               color=col, alpha=0.85, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (kcal/mol)", fontsize=12)
    ax.set_title("Final MAE by Strategy and Benchmark", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=5, loc="upper left", framealpha=0.9)
    ax.set_ylim(0, min(30, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "per_benchmark_mae_bars.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def _plot_pareto_impl(agg, subset_label, mae_cap, filename):
    fig, (ax_cost, ax_rho) = plt.subplots(1, 2, figsize=(18, 8), dpi=150)

    for strat, r in agg.items():
        if strat in SKIP_DUPLICATES:
            continue
        mae = get_val(r, "mae")
        rho = get_val(r, "rho")
        cost = r.get("cost_hours", r.get("median_cost", 0))
        if np.isnan(mae) or mae > mae_cap:
            continue

        cat = categorize(strat)
        st = CAT_STYLES[cat]

        ax_cost.scatter(cost, mae, c=st["c"], marker=st["marker"], s=st["s"],
                       zorder=st["z"], edgecolors="black", linewidth=0.3, alpha=0.8)
        if strat in TOP_LABEL:
            ax_cost.annotate(display_name(strat), (cost, mae), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, alpha=0.85)

        if not np.isnan(rho):
            ax_rho.scatter(rho, mae, c=st["c"], marker=st["marker"], s=st["s"],
                          zorder=st["z"], edgecolors="black", linewidth=0.3, alpha=0.8)
            if strat in TOP_LABEL:
                ax_rho.annotate(display_name(strat), (rho, mae), textcoords="offset points",
                               xytext=(5, 4), fontsize=7, alpha=0.85)

    for ax, xlabel, title, loc in [
        (ax_cost, "Aggregate Cost (hours)", f"{subset_label}: MAE vs Aggregate Cost", "upper right"),
        (ax_rho,  "Spearman rho",           f"{subset_label}: MAE vs Spearman rho",   "upper left"),
    ]:
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("MAE (kcal/mol)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(handles=LEGEND_ELS, loc=loc, fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_pareto(neutral_agg):
    _plot_pareto_impl(neutral_agg, "Neutral", mae_cap=2.5, filename="pareto_neutral.png")


def plot_pareto_charged(charged_agg):
    _plot_pareto_impl(charged_agg, "Charged", mae_cap=30, filename="pareto_charged.png")


def _plot_top_impl(agg, subset_label, n_bench, filename, top_n=15):
    items = []
    for strat, r in agg.items():
        if strat in SKIP_DUPLICATES:
            continue
        mae = get_val(r, "mae")
        rho = get_val(r, "rho")
        cost = r.get("cost_hours", r.get("median_cost", 0))
        cat = categorize(strat)
        if cat == "Always-X":
            continue
        items.append((strat, mae, rho, cost, cat))

    items.sort(key=lambda x: x[1])
    top = items[:top_n]
    if not top:
        return

    fig, ax = plt.subplots(figsize=(18, 7), dpi=150)
    labels = [display_name(x[0]) for x in top]
    maes = [x[1] for x in top]
    rhos = [x[2] for x in top]
    cats = [x[4] for x in top]
    bar_colors = [CAT_STYLES[c]["c"] for c in cats]

    bars = ax.bar(range(len(labels)), maes, color=bar_colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)

    for i, (bar, mae, rho) in enumerate(zip(bars, maes, rhos)):
        ha = "right" if i >= len(bars) - 2 else "center"
        rho_str = f"rho={rho:.3f}" if not np.isnan(rho) else "rho=N/A"
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"MAE={mae:.3f}\n{rho_str}", ha=ha, va="bottom", fontsize=6.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE (kcal/mol)", fontsize=12)
    ax.set_title(f"Top {top_n} Dynamic Strategies — {subset_label} Benchmarks (median across {n_bench})",
                 fontsize=13, fontweight="bold")
    oracle_mae = top[0][1]
    ax.axhline(y=oracle_mae, color="#2ca02c", ls="--", alpha=0.5)
    ax.legend(handles=LEGEND_ELS, loc="upper right", fontsize=8)
    ax.set_ylim(0, max(maes) * 1.25)
    ax.set_xlim(-0.6, len(labels) + 0.8)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {out}")
    plt.close()


def plot_top15(neutral_agg):
    _plot_top_impl(neutral_agg, "Neutral", 16, "top15_neutral.png")


def plot_top15_charged(charged_agg):
    _plot_top_impl(charged_agg, "Charged", 2, "top15_charged.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    neutral_agg = data.get("neutral_agg", {})
    charged_agg = data.get("charged_agg", {})
    per_bench = data.get("per_benchmark", {})

    plot_2x2(neutral_agg, charged_agg)
    plot_replay_frontier(neutral_agg, "Neutral", 16, "neutral_replay_frontier.png")
    plot_replay_frontier(charged_agg, "Charged", 2, "charged_replay_frontier.png")
    plot_per_benchmark_bars(per_bench)
    plot_pareto(neutral_agg)
    plot_pareto_charged(charged_agg)
    plot_top15(neutral_agg)
    plot_top15_charged(charged_agg)


if __name__ == "__main__":
    main()
