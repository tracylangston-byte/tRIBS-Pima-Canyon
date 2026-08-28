"""
plot_slide11_joint_feasibility_talk.py
=========================================
A presentation-oriented derivative of plot_storm_magnitude_comparison_100.py's
Figure D (joint feasibility narrowing) -- NOT a replacement for it. That
script's full comparison output (Figures A-D, both summary CSVs) is
untouched and remains the authoritative analysis artifact; this script
produces ONE figure for the AHS talk (slide 11): does testing across
three storm magnitudes narrow the Ks-f equifinal region beyond what any
single storm shows?

Carried over from plot_storm_magnitude_comparison_100.py:
  - Same three input CSVs (storm080, 100_narrow, storm125), same shared
    grid (intersection of each series' sampled Ks/f range), same
    normalized-coordinate triangulation before interpolating (Ks and
    log10(f) both min-max normalized -- avoids the banding artifact a
    raw-coordinate triangulation produces on this scale-mismatched data).
  - Same feasibility definition: |PBIAS| <= 2% at a grid cell counts as
    feasible for that storm. Joint feasibility = feasible in all three.
  - Same underlying math for area-fraction stats (kept precise, since
    Tracy wants exact fractions in the legend, not simplified labels).

Deliberately different, to match slide 10's presentation styling and
this round's color decisions:
  - Okabe-Ito colorblind-safe qualitative palette for the three storms
    (sky blue / bluish green / orange) -- chosen after the original
    sequential-blues attempt turned out to risk confusing the darkest
    storm color with the joint region's dark fill. Okabe-Ito stays
    distinguishable under deuteranopia, protanopia, and tritanopia, not
    just approximately safe. Joint-feasible region: deep blue, a fourth
    distinct hue rather than a shade of any storm color. Red is NOT used
    anywhere in this palette -- reserved exclusively for the
    true-parameter marker, matching slide 10's convention.
  - True-parameter marker: red star with white outline, sized and
    styled to match slide 10 exactly, instead of the original white/
    black X -- so the same symbol means the same thing across both
    slides.
  - Single log-f-axis panel, wide/rectangular aspect ratio, larger
    fonts, and the same custom f-axis treatment as slide 10: "log
    scale" in the label (not "RS soil"), ticks at 0.005/0.01/0.02/0.03
    instead of matplotlib's default 10^-2 style.
  - Legend keeps exact feasibility fractions per Tracy's request.

Usage (run from smf_demo/):
    python plot_slide11_joint_feasibility_talk.py

Requires all three storm CSVs to already exist in
calibration_work/03_comparisons/summary_tables/ -- this script does not
degrade gracefully to 2-way like the original comparison script does,
since the whole point of this figure is the 3-way joint narrowing.

Output:
    calibration_work/03_comparisons/sensitivity_plots/Comparisons/
        fig_slide11_joint_feasibility_talk.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import griddata

# ======================================================================
# CONFIG
# ======================================================================
SERIES = {
    "080":  {"label": "0.8\u00d7 storm",  "csv": "lhs_results_synth_Ks_f_100_storm080_RESCORED.csv",
             "color": "#56B4E9"},                    # Okabe-Ito sky blue
    "100n": {"label": "1.0\u00d7 storm",  "csv": "lhs_results_synth_Ks_f_100_narrow_RESCORED.csv",
             "color": "#009E73"},                    # Okabe-Ito bluish green
    "125":  {"label": "1.25\u00d7 storm", "csv": "lhs_results_synth_Ks_f_100_storm125_RESCORED.csv",
             "color": "#E69F00"},                    # Okabe-Ito orange
}
JOINT_COLOR = "#0072B2"   # Okabe-Ito deep blue -- red stays reserved for true params only
RED = "#AB0520"
WHITE = "#FFFFFF"

TRUTH_KS, TRUTH_F = 7.0, 0.012
PBIAS_TOL = 2.0     # |PBIAS| <= this (%) counts as "feasible", matches original script
N_GRID = 200
Y_TICKS = [0.005, 0.01, 0.02, 0.03]
YLABEL = "Hydraulic conductivity decay f (mm$^{-1}$), log scale"
XLABEL = "Ks multiplier"

COMPARISON_SUBDIR = "Comparisons"
REQUIRED_COLS = ["Ks_mult", "f_RS_abs", "pbias_pct"]

SHOW_TITLE = True

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / COMPARISON_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD ALL THREE SERIES -- this figure requires all three, no graceful
# 2-way degradation (unlike the original comparison script), since the
# whole point is the 3-way joint narrowing.
# -----------------------------------------------------------------------
data = {}
for key, cfg in SERIES.items():
    path = summary_dir / cfg["csv"]
    if not path.exists():
        sys.exit(f"Missing required input: {path}\n"
                  f"This figure needs all three storm sweeps -- run the "
                  f"missing one first, or check the filename in SERIES.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"{cfg['csv']} is missing columns: {missing}")
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    data[key] = df
    print(f"Loaded {len(df)} rows: {cfg['label']} ({cfg['csv']})")

# -----------------------------------------------------------------------
# SHARED GRID -- intersection of each series' actual sampled range
# -----------------------------------------------------------------------
ks_lo = max(df["Ks_mult"].min() for df in data.values())
ks_hi = min(df["Ks_mult"].max() for df in data.values())
f_lo  = max(df["f_RS_abs"].min() for df in data.values())
f_hi  = min(df["f_RS_abs"].max() for df in data.values())
print(f"Shared grid domain: Ks [{ks_lo:.3f}, {ks_hi:.3f}]  f [{f_lo:.5f}, {f_hi:.5f}]")
if ks_hi <= ks_lo or f_hi <= f_lo:
    sys.exit("Series ranges don't overlap -- check LHS bounds match across sweeps.")

ks_grid_1d = np.linspace(ks_lo, ks_hi, N_GRID)
f_log_1d = np.logspace(np.log10(f_lo), np.log10(f_hi), N_GRID)
KS_GRID, F_GRID = np.meshgrid(ks_grid_1d, f_log_1d)

# -----------------------------------------------------------------------
# NORMALIZED-COORDINATE TRIANGULATION -- same fix as slide 10 and the
# original comparison script: Ks and log10(f) both min-max normalized
# to [0,1] before griddata triangulates, avoiding banding from the
# >100x raw scale mismatch between the two axes.
# -----------------------------------------------------------------------
def norm_ks(ks):
    return (ks - ks_lo) / (ks_hi - ks_lo)

def norm_logf(f):
    return (np.log10(f) - np.log10(f_lo)) / (np.log10(f_hi) - np.log10(f_lo))

KS_NORM = norm_ks(KS_GRID)
F_NORM = norm_logf(F_GRID)

feasible_masks = {}
valid_masks = {}
for key, df in data.items():
    points_norm = np.column_stack([norm_ks(df["Ks_mult"].values), norm_logf(df["f_RS_abs"].values)])
    pbias_surf = griddata(points_norm, df["pbias_pct"].values, (KS_NORM, F_NORM), method="cubic")
    valid = ~np.isnan(pbias_surf)
    feasible = (np.abs(pbias_surf) <= PBIAS_TOL) & valid
    feasible_masks[key] = feasible
    valid_masks[key] = valid
    frac = feasible.sum() / valid.sum() if valid.sum() else np.nan
    print(f"  {SERIES[key]['label']}: feasible area fraction = {frac:.3f}")

valid_all = np.ones_like(KS_GRID, dtype=bool)
for v in valid_masks.values():
    valid_all &= v
joint_mask = np.ones_like(KS_GRID, dtype=bool)
for m in feasible_masks.values():
    joint_mask &= m
joint_frac = joint_mask[valid_all].sum() / valid_all.sum() if valid_all.sum() else np.nan
print(f"  Joint (all three): feasible area fraction = {joint_frac:.3f}")

# -----------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7.5))

legend_handles = []
for key, cfg in SERIES.items():
    mask = feasible_masks[key].astype(float)
    ax.contourf(KS_GRID, F_GRID, mask, levels=[0.5, 1.5], colors=[cfg["color"]], alpha=0.35)
    ax.contour(KS_GRID, F_GRID, mask, levels=[0.5], colors=[cfg["color"]], linewidths=1.5)
    frac = feasible_masks[key].sum() / valid_masks[key].sum()
    legend_handles.append(plt.Line2D([0], [0], color=cfg["color"], lw=8,
                                      label=f"{cfg['label']} (feasible: {frac:.1%})"))

ax.contourf(KS_GRID, F_GRID, joint_mask.astype(float), levels=[0.5, 1.5],
            colors=[JOINT_COLOR], alpha=0.65)
ax.contour(KS_GRID, F_GRID, joint_mask.astype(float), levels=[0.5],
           colors=[JOINT_COLOR], linewidths=2.2)
legend_handles.append(plt.Line2D([0], [0], color=JOINT_COLOR, lw=8,
                                  label=f"Joint \u2014 all three (feasible: {joint_frac:.1%})"))

ax.scatter([TRUTH_KS], [TRUTH_F], s=550, marker="*", color=RED,
           edgecolors=WHITE, linewidths=1.5, zorder=10)

ax.set_yscale("log")
ax.set_yticks(Y_TICKS)
ax.set_yticklabels([str(v) for v in Y_TICKS])
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.set_xlabel(XLABEL, fontsize=17)
ax.set_ylabel(YLABEL, fontsize=17)
ax.tick_params(labelsize=14)

ax.legend(handles=legend_handles, loc="upper right", fontsize=13, framealpha=0.9)

if SHOW_TITLE:
    ax.set_title(
        "Joint feasibility \u2014 Ks \u00d7 f, three storm magnitudes\n"
        "SMF Aug 12, 2014 (synthetic truth Ks=7.0x/f=0.012)  |  "
        "Shaded = |PBIAS| \u2264 2% (acceptable fit) per storm magnitude",
        fontsize=15)

out_path = plot_dir / "fig_slide11_joint_feasibility_talk.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
