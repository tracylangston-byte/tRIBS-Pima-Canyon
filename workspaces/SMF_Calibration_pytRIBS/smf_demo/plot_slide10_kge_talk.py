"""
plot_slide10_kge_talk.py
==========================
A presentation-oriented derivative of plot_lhs_Ks_f_100.py -- NOT a
replacement for it. That script's full 10-figure technical output is
untouched and remains the authoritative analysis artifact; this script
produces ONE figure for the AHS talk (slide 10), simplified in a few
specific ways but keeping most of the real technical content.

Carried over EXACTLY from plot_lhs_Ks_f_100.py:
  - Normalized-coordinate triangulation (Ks and f min-max-normalized, f
    in log10 space, before griddata runs) for BOTH the KGE and PBIAS
    surfaces -- this is what keeps the PBIAS=0 boundary smooth instead
    of banding. See that script's comments for the full explanation.

Deliberately different, per Tracy's requests this round:
  - YlGnBu colormap instead of RdYlGn (red is the deck's UI accent color).
  - Log10 f-axis ONLY, but wider aspect ratio (rectangular slide space).
  - f-axis label drops "RS soil", adds "log scale", custom tick labels
    at 0.005 / 0.01 / 0.03 instead of matplotlib's default 10^-2 style.
  - PBIAS=0 dashed white line: back in.
  - LHS sample points: back in, colored by their own KGE value.
  - Star marks true parameters; if the true parameters and the
    best-scoring run are close enough to be visually the same point,
    the label says so explicitly instead of just "true parameters".
  - Title: short, plain-language KGE label (no "2012"), includes the
    event/date/truth-value line and the PBIAS-line explanation.

Usage (run from smf_demo/):
    python plot_slide10_kge_talk.py

Output:
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_f_100/
        fig_slide10_kge_talk.png
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

# Tolerance for treating the true parameters and the best-scoring run as
# "the same point" for labeling purposes -- adjust if this reads wrong
# once you see it against your actual best-run value.
KS_PROXIMITY_PCT = 0.05      # 5% relative difference in Ks
LOGF_PROXIMITY   = 0.08      # absolute difference in log10(f)

RED = "#AB0520"
WHITE = "#FFFFFF"

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
# LOAD RESULTS
# -----------------------------------------------------------------------
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(f"LHS results not found: {results_path}")

df = pd.read_csv(results_path)
required_cols = ["Ks_mult", SECOND_COL, "kge_2012", "pbias_pct"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results CSV: {missing}")

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"Loaded {len(df)} LHS runs (after dropping NaN rows)")

ks_pts     = df["Ks_mult"].values
second_pts = df[SECOND_COL].values
kge_vals   = df["kge_2012"].values
pbias_vals = df["pbias_pct"].values

best_idx = int(np.argmax(kge_vals))
best_ks, best_f, best_kge = ks_pts[best_idx], second_pts[best_idx], kge_vals[best_idx]
print(f"Best run: Ks={best_ks:.3f}x  f={best_f:.4g}  KGE={best_kge:.3f}")
print(f"True:     Ks={TRUE_KS}x  f={TRUE_F}")

# -----------------------------------------------------------------------
# LOG-F GRID ONLY
# -----------------------------------------------------------------------
N_GRID = 200
ks_grid_1d = np.linspace(ks_pts.min(), ks_pts.max(), N_GRID)
f_log_1d = np.logspace(np.log10(second_pts.min()), np.log10(second_pts.max()), N_GRID)
KS_GRID_LOG, F_GRID_LOG = np.meshgrid(ks_grid_1d, f_log_1d)

# -----------------------------------------------------------------------
# NORMALIZED-COORDINATE TRIANGULATION -- see module docstring
# -----------------------------------------------------------------------
def normalize_ks(ks):
    return (ks - ks_pts.min()) / (ks_pts.max() - ks_pts.min())

def normalize_logf(f):
    log_lo, log_hi = np.log10(second_pts.min()), np.log10(second_pts.max())
    return (np.log10(f) - log_lo) / (log_hi - log_lo)

points_norm = np.column_stack([normalize_ks(ks_pts), normalize_logf(second_pts)])
KS_GRID_LOG_NORM = normalize_ks(KS_GRID_LOG)
F_GRID_LOG_NORM  = normalize_logf(F_GRID_LOG)

kge_surface   = griddata(points_norm, kge_vals,   (KS_GRID_LOG_NORM, F_GRID_LOG_NORM), method='cubic')
pbias_surface = griddata(points_norm, pbias_vals, (KS_GRID_LOG_NORM, F_GRID_LOG_NORM), method='cubic')

# -----------------------------------------------------------------------
# DETERMINE STAR LABEL -- combined if true params and best run coincide
# -----------------------------------------------------------------------
ks_close = abs(best_ks - TRUE_KS) / TRUE_KS <= KS_PROXIMITY_PCT
f_close = abs(np.log10(best_f) - np.log10(TRUE_F)) <= LOGF_PROXIMITY
if ks_close and f_close:
    star_label = "True parameters\n\u2248 best-scoring run"
else:
    star_label = "True parameters"
    print("NOTE: best run and true parameters are NOT close by the "
          "current tolerance -- labeling star as 'True parameters' only. "
          "Consider whether the best-run point should be shown separately.")

# -----------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------
kge_norm = mcolors.Normalize(vmin=KGE_CLIP[0], vmax=KGE_CLIP[1])

fig, ax = plt.subplots(figsize=(14, 7.5))

cf = ax.contourf(KS_GRID_LOG, F_GRID_LOG, kge_surface, levels=20,
                  cmap='YlGnBu', norm=kge_norm, extend='min')

# PBIAS zero-crossing boundary
ax.contour(KS_GRID_LOG, F_GRID_LOG, pbias_surface, levels=[0], colors=[WHITE],
           linewidths=2.2, linestyles='--', zorder=6)

# LHS sample points, colored by their own KGE value
ax.scatter(ks_pts, second_pts, c=kge_vals, cmap='YlGnBu', norm=kge_norm,
           s=18, edgecolors='white', linewidths=0.4, alpha=0.75, zorder=5)

# True-parameter star
ax.scatter([TRUE_KS], [TRUE_F], s=550, marker='*', color=RED,
           edgecolors=WHITE, linewidths=1.5, zorder=10)
ax.annotate(star_label, xy=(TRUE_KS, TRUE_F), xytext=(14, -45),
            textcoords="offset points", fontsize=17, fontweight="bold",
            color=WHITE, zorder=11, va="top")

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

if SHOW_TITLE:
    ax.set_title(
        "KGE \u2014 Ks \u00d7 f joint sensitivity\n"
        "SMF Aug 12, 2014 (synthetic truth Ks=7.0x/f=0.012)  |  "
        "White dashed = perfect volume match",
        fontsize=15)

out_path = plot_dir / "fig_slide10_kge_talk.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out_path}")