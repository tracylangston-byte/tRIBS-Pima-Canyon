"""
plot_slide13_findings_heatmap.py
===================================
The slide 13 "findings" figure for the AHS talk: for each of the three
routing parameters (flowexp r, n, cv), does composite KGE identify it,
does a timing-specific metric identify it, or both/neither? Six rows,
one per independent Series 101 anchor.

This is a presentation-simplified VIEW into the comprehensive analysis,
not a separate computation -- it reads the correlations that
analyze_series101_kge_components.py already computed and filters down
to just 2 of its 11 metrics (composite KGE and first-arrival error),
rather than recomputing anything. If those two scripts haven't been run
yet (or need re-running because the underlying anchor data changed),
run them first -- see "Requires" below.

Pipeline this script sits at the end of:
    1. merge_series101_anchors.py
       -> lhs_results_anchor_ALL_101_MERGED.csv (6 anchors, n=100 each)
    2. analyze_series101_kge_components.py
       -> series101_kge_component_correlations.csv (33 metric columns)
    3. THIS SCRIPT
       -> fig_slide13_findings_heatmap.png (2 of those 33 columns)

Styling decisions from this round of review, all deliberate:
  - Only 2 metrics shown (KGE 2012, first-arrival error) out of the 11
    available -- cv/n/r's other 9 metric correlations are real and
    checked (see the FULL heatmap from analyze_series101_kge_components.py)
    but not needed to make this slide's specific point.
  - Anchor rows ordered by DESCENDING f (largest at top) to match the
    y-axis orientation of the slide 10/11/12 contour plots.
  - Black column-group separators (not white).
  - No column-group header labels (Ks/f description, etc.) -- Tracy is
    adding her own labels in PowerPoint below the figure.
  - Title: only the first line ("Which metric identifies which
    parameter?") is large; the second and third lines match the same
    size as the figure's other text (tick labels), not the big headline
    size.

Requires:
    calibration_work/03_comparisons/summary_tables/
        series101_kge_component_correlations.csv

Usage (run from smf_demo/):
    python plot_slide13_findings_heatmap.py

Output:
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_f_100/
        fig_slide13_findings_heatmap.png
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# CONFIG
# ======================================================================
CORRELATIONS_CSV = "series101_kge_component_correlations.csv"
OUTPUT_SUBDIR = "lhs_Ks_f_100"

PARAM_ORDER = ["flowexp", "channelroughness", "kinemvelcoef"]
METRIC_KGE = "kge_2012"
METRIC_TIMING = "first_arrival_error_min"

# f-value for each anchor, used only for sort order (largest f at top,
# matching the slide 10/11/12 contour plots' y-axis orientation).
ANCHOR_F_VALUES = {
    "f0p006": 0.006, "f0p008": 0.008, "f0p010": 0.010,
    "f0p015": 0.015, "f0p02": 0.020, "f0p03": 0.030,
}
ANCHOR_DISPLAY = {k: f"f={v}" for k, v in ANCHOR_F_VALUES.items()}

BODY_FONTSIZE = 13   # matches tick labels
TITLE_FONTSIZE = 19.5  # 1.5x body -- top title line and cell numbers both use this
CELL_FONTSIZE = TITLE_FONTSIZE

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
# LOAD
# -----------------------------------------------------------------------
corr_path = summary_dir / CORRELATIONS_CSV
if not corr_path.exists():
    raise FileNotFoundError(
        f"{corr_path} not found. Run merge_series101_anchors.py then "
        f"analyze_series101_kge_components.py first."
    )

df = pd.read_csv(corr_path)

missing_anchors = set(ANCHOR_F_VALUES) - set(df["anchor_label"].unique())
if missing_anchors:
    print(f"WARNING: expected anchors not found in {CORRELATIONS_CSV}: {missing_anchors}")

anchors_present = [a for a in ANCHOR_F_VALUES if a in df["anchor_label"].unique()]
anchors_sorted = sorted(anchors_present, key=lambda a: ANCHOR_F_VALUES[a], reverse=True)

# -----------------------------------------------------------------------
# BUILD GRID -- 6 columns (KGE, Timing) x 3 params, ordered rows by anchor
# -----------------------------------------------------------------------
cols = [(p, m) for p in PARAM_ORDER for m in (METRIC_KGE, METRIC_TIMING)]
col_labels = ["KGE", "Timing"] * len(PARAM_ORDER)

grid = np.full((len(anchors_sorted), len(cols)), np.nan)
for i, anchor in enumerate(anchors_sorted):
    for j, (param, metric) in enumerate(cols):
        match = df[(df["anchor_label"] == anchor) &
                    (df["parameter"] == param) & (df["metric"] == metric)]
        if len(match):
            grid[i, j] = match["pearson_r"].values[0]
        else:
            print(f"  WARNING: no data for anchor={anchor}, param={param}, metric={metric}")

# -----------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(col_labels, fontsize=BODY_FONTSIZE)
ax.set_yticks(range(len(anchors_sorted)))
ax.set_yticklabels([ANCHOR_DISPLAY[a] for a in anchors_sorted], fontsize=BODY_FONTSIZE)
ax.tick_params(axis="both", which="both", length=0)

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid[i, j]
        if np.isnan(v):
            continue
        color = "white" if abs(v) > 0.55 else "black"
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=CELL_FONTSIZE,
                 color=color, fontweight="bold")

# black separators between parameter groups (every 2 columns)
for k in range(1, len(PARAM_ORDER)):
    ax.axvline(2 * k - 0.5, color="black", linewidth=2.5)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Pearson r vs. parameter value", fontsize=BODY_FONTSIZE)
cbar.ax.tick_params(labelsize=11, length=0)

# Title, positioned relative to the AXES (not the whole figure) so it's
# centered over the heatmap itself rather than shifted left by the
# colorbar's width. Line 1 large; line 2 combined, body-text size.
ax.text(0.5, 1.09, "Which metric identifies which parameter?",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=TITLE_FONTSIZE, fontweight="bold", clip_on=False)
ax.text(0.5, 1.02,
        "Pearson r, six anchors (n=100 each)  |  Timing metric shown: first-arrival error",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=BODY_FONTSIZE, clip_on=False)

out_path = plot_dir / "fig_slide13_findings_heatmap.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")