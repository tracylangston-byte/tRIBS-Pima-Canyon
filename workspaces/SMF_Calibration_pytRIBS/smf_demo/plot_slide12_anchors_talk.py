"""
plot_slide12_anchors_talk.py
==============================
A presentation-only figure for the AHS talk (slide 12, "Scenario 3"):
the Series 100 KGE surface from slide 10, with the six completed Series
101 anchors marked on it. NOT a replacement for any technical script --
plot_lhs_Ks_f_100.py and analyze_series101_cvrn_identifiability.py
remain the authoritative analysis artifacts. This script only draws the
one bridging figure that argues, visually, for why routing-parameter
identifiability was tested at six different points instead of one.

Carried over exactly from plot_slide10_kge_talk.py:
  - Same KGE surface (Series 100, 400-run Ks x f LHS), same normalized-
    coordinate triangulation (Ks and log10(f) min-max normalized before
    griddata runs -- avoids banding from the >100x raw scale mismatch
    between the two axes), same YlGnBu colormap, same log-f-axis
    treatment (custom ticks at 0.005/0.01/0.02/0.03, "log scale" label,
    no "RS soil"), same red star for true parameters, same wide/
    rectangular aspect ratio and font sizes.

New to this script:
  - Six anchor markers (Oasis Blue diamonds), hardcoded from
    ridge_width_vs_f.csv / the Series 101 handoff table -- each anchor
    is a FIXED (Ks, f) point where cv/flowexp-r/n_e were then swept via
    LHS (100 runs/anchor: 50 matched-seed + 50 independent-seed, per
    Tracy's count).
  - The seventh anchor (f0p012_true, nearest the true point) is
    intentionally NOT shown -- it hung on all 15 attempts and never
    completed. Per Tracy: not addressed in the talk; she'll explain
    verbally if asked. If you want it back in (e.g. as a hollow marker
    with a "blocked" label), that was tried and dropped for label-
    placement/legibility reasons -- ask before re-adding rather than
    reintroducing the same clutter.
  - No per-anchor LHS scatter (the cv/r/n draws) -- deliberately
    excluded. Those parameters aren't spatial axes on this Ks-f plot,
    that identifiability content belongs on slide 13, and Tracy
    confirmed it would just be noise here.

Usage (run from smf_demo/):
    python plot_slide12_anchors_talk.py

Output:
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_f_100/
        fig_slide12_anchors_talk.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.interpolate import griddata

# ======================================================================
# CONFIG
# ======================================================================
RESULTS_CSV   = "lhs_results_synth_Ks_f_100_RESCORED.csv"
OUTPUT_SUBDIR = "lhs_Ks_f_100"
SECOND_COL    = "f_RS_abs"
YLABEL        = "Hydraulic conductivity decay f (mm$^{-1}$), log scale"
XLABEL        = "Ks multiplier"

KGE_CLIP = (-0.3, 1.0)
TRUE_KS, TRUE_F = 7.0, 0.012
Y_TICKS = [0.005, 0.01, 0.02, 0.03]

# The six COMPLETED Series 101 anchors: (label, f, Ks-ridge-peak).
# Source: ridge_width_vs_f.csv / Handoff_Series101_CvRnIdentifiabilityRebuild_v1.md.
# f0p012_true (f=0.012, Ks=6.932) intentionally omitted -- see docstring.
ANCHORS = [
    ("f0p006", 0.006, 4.394),
    ("f0p008", 0.008, 5.234),
    ("f0p010", 0.010, 6.193),
    ("f0p015", 0.015, 8.051),
    ("f0p02",  0.020, 8.191),
    ("f0p03",  0.030, 7.172),
]
RUNS_PER_ANCHOR = 100   # 50 matched-seed + 50 independent-seed

RED = "#AB0520"
WHITE = "#FFFFFF"
OASIS = "#378DBD"

SHOW_TITLE = True

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / OUTPUT_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD RESULTS (Series 100 -- same surface as slide 10)
# -----------------------------------------------------------------------
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(f"LHS results not found: {results_path}")

df = pd.read_csv(results_path)
required_cols = ["Ks_mult", SECOND_COL, "kge_2012"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results CSV: {missing}")

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"Loaded {len(df)} LHS runs (after dropping NaN rows)")

ks_pts     = df["Ks_mult"].values
second_pts = df[SECOND_COL].values
kge_vals   = df["kge_2012"].values

# -----------------------------------------------------------------------
# LOG-F GRID
# -----------------------------------------------------------------------
N_GRID = 200
ks_grid_1d = np.linspace(ks_pts.min(), ks_pts.max(), N_GRID)
f_log_1d = np.logspace(np.log10(second_pts.min()), np.log10(second_pts.max()), N_GRID)
KS_GRID_LOG, F_GRID_LOG = np.meshgrid(ks_grid_1d, f_log_1d)

# -----------------------------------------------------------------------
# NORMALIZED-COORDINATE TRIANGULATION -- see docstring
# -----------------------------------------------------------------------
def normalize_ks(ks):
    return (ks - ks_pts.min()) / (ks_pts.max() - ks_pts.min())

def normalize_logf(f):
    log_lo, log_hi = np.log10(second_pts.min()), np.log10(second_pts.max())
    return (np.log10(f) - log_lo) / (log_hi - log_lo)

points_norm = np.column_stack([normalize_ks(ks_pts), normalize_logf(second_pts)])
KS_GRID_LOG_NORM = normalize_ks(KS_GRID_LOG)
F_GRID_LOG_NORM  = normalize_logf(F_GRID_LOG)

kge_surface = griddata(points_norm, kge_vals, (KS_GRID_LOG_NORM, F_GRID_LOG_NORM), method='cubic')

# -----------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------
kge_norm = mcolors.Normalize(vmin=KGE_CLIP[0], vmax=KGE_CLIP[1])

fig, ax = plt.subplots(figsize=(14, 7.5))
cf = ax.contourf(KS_GRID_LOG, F_GRID_LOG, kge_surface, levels=20,
                  cmap='YlGnBu', norm=kge_norm, extend='min')

for label, f, ks in ANCHORS:
    ax.scatter([ks], [f], s=260, marker='D', color=OASIS, edgecolors=WHITE,
               linewidths=1.8, zorder=9)

ax.scatter([TRUE_KS], [TRUE_F], s=550, marker='*', color=RED,
           edgecolors=WHITE, linewidths=1.5, zorder=10)

ax.set_yscale('log')
ax.set_yticks(Y_TICKS)
ax.set_yticklabels([str(v) for v in Y_TICKS])
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.set_xlabel(XLABEL, fontsize=17)
ax.set_ylabel(YLABEL, fontsize=17)
ax.tick_params(labelsize=14)

cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
cbar.set_label("KGE", fontsize=16)
cbar.ax.tick_params(labelsize=13)

handles = [
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=OASIS, markeredgecolor=WHITE,
               markersize=14, linestyle='None', label='Anchor (cv/r/n tested here)'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=RED, markeredgecolor=WHITE,
               markersize=18, linestyle='None', label='True parameters'),
]
ax.legend(handles=handles, loc='upper right', fontsize=13, framealpha=0.9)

if SHOW_TITLE:
    ax.set_title(
        f"Six anchors along the Ks \u00d7 f ridge\n"
        f"SMF Aug 12, 2014 (synthetic truth Ks=7.0x/f=0.012)  |  Each anchor: cv, flowexp r, "
        f"n\u2091 swept, {RUNS_PER_ANCHOR} runs/anchor",
        fontsize=15)

out_path = plot_dir / "fig_slide12_anchors_talk.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out_path}")
