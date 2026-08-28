"""
analyze_series101_kge_components.py
======================================
Extends analyze_series101_cvrn_identifiability.py's correlation analysis
to include the individual KGE components (r, alpha, beta, gamma), not
just composite KGE and the phase-specific metrics. Purpose: check
whether cv/flowexp-r/n's correlation with composite KGE runs through a
specific component (e.g. does flowexp's effect show up specifically in
KGE's timing-correlation term, matching physical expectation for a
routing parameter?) -- a candidate figure for slide 13, not yet decided
whether it's worth including.

Does NOT require new tRIBS runs. kge_r/kge_alpha/kge_beta/kge_gamma are
already computed and stored for every run by run_sensitivity_single.py
(confirmed in rescore_series100_family_interp.py's metrics dict) -- this
script only adds a new correlation pass against columns that already
exist in your combined anchor results file.

NAMING, per Tracy's decision: KGE's own correlation component and the
flowexp routing parameter are BOTH conventionally symbol "r" (confirmed
in parameter_key.py) -- an unavoidable collision if shown side by side.
Every label in this script's output uses "KGE r" and "flowexp r"
explicitly, never a bare "r", specifically to avoid that collision.

FIRST PASS, DELIBERATELY COMPREHENSIVE: per Tracy's instruction, this
includes every available metric (all 4 KGE components + composite KGE
under both formulations + all 5 phase-specific metrics = 11 metrics x 3
parameters = 33 columns) rather than pre-narrowing. Expect a dense,
messy heatmap -- that's intentional for this pass. Narrow the METRICS
list below once you've seen which columns are actually load-bearing.

Usage (run from smf_demo/):
    python analyze_series101_kge_components.py

Requires:
    calibration_work/03_comparisons/summary_tables/lhs_results_anchor_ALL_101.csv
    (the same combined-anchor file analyze_series101_cvrn_identifiability.py
    reads -- if your independent-seed replication batch lives in a
    separate file rather than already merged into this one, you'll need
    to point LOAD_PATHS below at both and concat before the correlation
    loop; I don't know your exact post-replication file layout, so this
    defaults to the single combined file the original script expects.)

Output:
    calibration_work/03_comparisons/summary_tables/
        series101_kge_component_correlations.csv
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_f_100/
        fig_series101_kge_components_heatmap_FULL.png   (comprehensive, messy)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ======================================================================
# CONFIG
# ======================================================================
COMBINED_RESULTS_CSV = "lhs_results_anchor_ALL_101_MERGED.csv"   # see merge_series101_anchors.py --
                                                                    # the original ALL_101.csv turned out
                                                                    # to be a single overwritten anchor,
                                                                    # not a real merge. Run that script
                                                                    # first if this file doesn't exist yet.

PARAMS = ["kinemvelcoef", "flowexp", "channelroughness"]
PARAM_LABELS = {
    "kinemvelcoef":     "cv",
    "flowexp":          "flowexp r",     # never bare "r" -- see docstring
    "channelroughness": "n",
}

# Deliberately comprehensive for this first pass -- 11 metrics.
METRICS = [
    "kge_2012",
    "kge",
    "kge_r",
    "kge_alpha",
    "kge_beta",
    "kge_gamma",
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    "duration_above_thresh_error_min",
    "recession_rate_ratio",
]
METRIC_LABELS = {
    "kge_2012":                          "KGE (2012)",
    "kge":                               "KGE (2009)",
    "kge_r":                             "KGE r",        # never bare "r"
    "kge_alpha":                         "KGE \u03b1",
    "kge_beta":                          "KGE \u03b2",
    "kge_gamma":                         "KGE \u03b3",
    "first_arrival_error_min":           "First-arrival error",
    "rising_limb_steepness_ratio":       "Rising-limb steepness",
    "time_to_peak_from_exc_min":         "Time-to-peak",
    "duration_above_thresh_error_min":   "Duration-above-thresh error",
    "recession_rate_ratio":              "Recession-rate ratio",
}

OUTPUT_SUBDIR = "lhs_Ks_f_100"

# -----------------------------------------------------------------------
# PATHS -- matches analyze_series101_cvrn_identifiability.py's convention
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / OUTPUT_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD -- same combined-anchor file the original Series 101 analysis reads
# -----------------------------------------------------------------------
combined_path = summary_dir / COMBINED_RESULTS_CSV
if not combined_path.exists():
    raise FileNotFoundError(
        f"{combined_path} not found. Run run_lhs_nanchor_cvrn_101.py first "
        f"(or point COMBINED_RESULTS_CSV at wherever your post-replication "
        f"combined results actually live -- see docstring)."
    )

df = pd.read_csv(combined_path)
print(f"Loaded {len(df)} rows from {combined_path.name} "
      f"across {df['anchor_label'].nunique()} anchors: "
      f"{sorted(df['anchor_label'].unique())}")

missing_metrics = [m for m in METRICS if m not in df.columns]
if missing_metrics:
    print(f"\nWARNING: these metrics aren't in the results file and will be "
          f"skipped: {missing_metrics}")
    print("(kge_r/kge_alpha/kge_beta/kge_gamma should already be present if "
          "this data was scored by the current run_sensitivity_single.py -- "
          "if they're missing, the anchor runs may predate that patch.)")
    METRICS = [m for m in METRICS if m in df.columns]

# -----------------------------------------------------------------------
# CORRELATIONS -- every (anchor, parameter, metric) triple, Pearson + Spearman
# -----------------------------------------------------------------------
rows = []
for anchor in sorted(df["anchor_label"].unique()):
    sub = df[df["anchor_label"] == anchor]
    for param in PARAMS:
        if param not in sub.columns:
            continue
        for metric in METRICS:
            valid = sub[[param, metric]].dropna()
            if len(valid) < 4:
                continue
            r_p, p_p = pearsonr(valid[param], valid[metric])
            r_s, p_s = spearmanr(valid[param], valid[metric])
            rows.append({
                "anchor_label": anchor,
                "parameter": param,
                "parameter_label": PARAM_LABELS[param],
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_rho": r_s,
                "spearman_p": p_s,
                "n": len(valid),
            })

corr_df = pd.DataFrame(rows)
out_csv = summary_dir / "series101_kge_component_correlations.csv"
corr_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv.name}  ({len(corr_df)} rows)")

# -----------------------------------------------------------------------
# COMPREHENSIVE HEATMAP -- all columns, intentionally dense for this pass
# -----------------------------------------------------------------------
anchors = sorted(df["anchor_label"].unique())
cols = [(p, m) for p in PARAMS for m in METRICS]
col_labels = [f"{PARAM_LABELS[p]}\n{METRIC_LABELS.get(m, m)}" for p, m in cols]

grid = np.full((len(anchors), len(cols)), np.nan)
for i, anchor in enumerate(anchors):
    for j, (p, m) in enumerate(cols):
        match = corr_df[(corr_df["anchor_label"] == anchor) &
                         (corr_df["parameter"] == p) & (corr_df["metric"] == m)]
        if len(match):
            grid[i, j] = match["pearson_r"].values[0]

fig_w = max(16, 0.55 * len(cols))
fig, ax = plt.subplots(figsize=(fig_w, max(5, 0.6 * len(anchors))))
im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
ax.set_yticks(range(len(anchors)))
ax.set_yticklabels(anchors, fontsize=10)

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid[i, j]
        if np.isnan(v):
            continue
        color = "white" if abs(v) > 0.55 else "black"
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6, color=color)

# separators between parameter groups
n_metrics = len(METRICS)
for k in range(1, len(PARAMS)):
    ax.axvline(k * n_metrics - 0.5, color="white", linewidth=2.5)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Pearson r", fontsize=12)

ax.set_title("Series 101: cv / flowexp r / n vs. KGE components + phase metrics, by anchor\n"
             "FULL comprehensive pass -- expect this to be dense; narrow METRICS once reviewed",
             fontsize=13)
fig.tight_layout()

out_fig = plot_dir / "fig_series101_kge_components_heatmap_FULL.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_fig.name}")

# -----------------------------------------------------------------------
# CONSOLE SUMMARY -- flag anything |r| >= 0.4 for a quick first read
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Correlations with |Pearson r| >= 0.4 (quick-scan candidates):")
print("=" * 70)
notable = corr_df[corr_df["pearson_r"].abs() >= 0.4].sort_values(
    ["parameter_label", "metric_label", "anchor_label"])
if notable.empty:
    print("  (none at this threshold)")
else:
    for _, row in notable.iterrows():
        print(f"  {row['parameter_label']:<12} vs {row['metric_label']:<26} "
              f"@ {row['anchor_label']:<10} r={row['pearson_r']:+.3f}  p={row['pearson_p']:.2e}")