"""Shared paper-figure style for the RAPIDS-Select V6 supplementary figures.

Matches the main-paper figures (offline_replay/plot_sequential.py and
other_figures/plot_fidelity_overview.py): Material-Design arm palette,
consistent markers, dpi=200, and a small set of font sizes.

Usage:
    from _paper_style import ARM_COLOR, ARM_MARKER, apply_rc, DPI, SELECTOR_C
    apply_rc()
    ...
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
"""
import matplotlib as mpl

# ---- canonical arm palette (identical to the main-paper figures) ----
ARM_COLOR = {
    "RAPIDS":         "#2196F3",
    "PBE-D3BJ_SP":    "#4CAF50",
    "PBE-D3BJ_GeoSP": "#FF9800",
    "CREST_xTB":      "#9C27B0",
    "CREST_xTB_DFT":  "#673AB7",
    "Oracle":         "#2ca02c",
}
ARM_MARKER = {
    "RAPIDS":         "o",
    "PBE-D3BJ_SP":    "s",
    "PBE-D3BJ_GeoSP": "D",
    "CREST_xTB":      "^",
    "CREST_xTB_DFT":  "P",
    "Oracle":         "*",
}

# selector frontier / highlight accents
SELECTOR_C = "#d62728"   # crimson, matches the main-paper "GBM/Meta" family accent
SELECTOR_LW = 2.5

# figure defaults
DPI = 200

# font sizes (kept consistent with plot_sequential.py)
FS_LABEL = 11
FS_TITLE = 12
FS_SUPTITLE = 13
FS_TICK = 9
FS_LEGEND = 8
FS_ANNOT = 8

# bar-chart accent pair (occurrence vs magnitude / MAE vs catastrophic)
BAR_A = "#2ca02c"   # seagreen-ish -> aligned to Oracle green
BAR_B = "#d62728"   # crimson
ARM_STEEL = "#1f77b4"   # neutral steel-blue for single-series bars (matches main "Bandit" blue)


def apply_rc():
    """Apply global rcParams so every panel inherits the paper look."""
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })


def style_grid(ax):
    """Consistent both-axes grid used across the paper panels."""
    ax.grid(True, alpha=0.25, which="both", linewidth=0.5)
