"""
plot_pearson_nanchor_99.py
===========================
Series 99 -- cv/r/n identifiability trend across 9 volume-matched
Ks_mult/f_RS_abs anchors spanning 4.25x-8.25x.

Replaces the never-built Series 98 per-anchor heatmap plan (unreadable at
9 anchors x 9 metrics x 3 params). Instead: a 3x3 grid of metric panels,
one trend line per routing parameter (cv, r, n) per panel, plotted across
anchor Ks_mult position on the x-axis. A companion long-format CSV has
the full numeric Pearson r table underneath the figure.

Anchor x-axis and lo/hi branches
---------------------------------
Two Ks_mult positions (7.25x, 8.25x) each have BOTH a low-f and a high-f
volume-matched solution (the Ks-f equifinality "swoosh" -- see
run_lhs_nanchor_cvrn_99.py). Both are plotted at the same x-position;
marker fill distinguishes them (filled = lo-f/single anchor, open =
hi-f branch). Each parameter's trend line connects points in
(Ks_mult, f_RS_abs) order, so at 7.25x and 8.25x it visibly steps from
the filled to the open point before continuing -- that step is a real
result (does picking a different volume-matched f at fixed Ks change
routing-parameter identifiability?), not a plotting artifact.

Ks6p25hi has no lo-f counterpart in this run: Ks6p25lo sits ~0.4%/~7%
away from the Series 98 hang point and is deliberately excluded pending
probe_anchor_Ks6p25lo_99.py clearing it (see run_lhs_nanchor_cvrn_99.py
docstring). It still gets an open marker for labeling consistency, just
without a paired filled point beside it.

Partial-completion behavior
----------------------------
The Series 99 sweep runs 9 anchors sequentially and can take a long time.
This script does NOT require all 9 anchor CSVs to exist -- any anchor
whose lhs_results_anchor_<label>_99.csv is missing is skipped with a
console warning, and the figure/CSV are built from whatever anchors ARE
done. Re-run again later as more anchors finish.

Failure/hang counts
--------------------
Per-anchor HANG/FAILED draw counts (from lhs_results_anchor_FAILED_99.csv)
are NOT annotated on the figure -- at the scale seen so far (1 hang out
of ~450 draws) it isn't visually meaningful, and labeling counts under
9 panels x up to 9 x-ticks would hurt readability more than it helps.
Counts are instead carried as columns in the companion CSV
(n_completed / n_hang / n_failed per anchor) so the audit trail travels
with the correlation numbers without appearing on the plot itself.

Output:
    calibration_work/03_comparisons/sensitivity_plots/NAnchor_99/
        fig_pearson_nanchor_trend_99.png
        pearson_nanchor_trend_99.csv

Usage (run from smf_demo):
    python plot_pearson_nanchor_99.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import pearsonr

from parameter_key import PARAM_KEY, METRIC_KEY

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =======================================================================
# CONFIG
# =======================================================================
LHS_SERIES = "99"

# Mirrors ANCHORS in run_lhs_nanchor_cvrn_99.py exactly (Ks6p25lo stays
# commented out there and is therefore absent here too). If you uncomment
# it there and rerun, add the matching entry here.
ANCHORS = [
    {"label": "Ks4p25",   "Ks_mult": 4.25, "f_RS_abs": 0.0055, "branch": None},
    {"label": "anchorA",  "Ks_mult": 5.0,  "f_RS_abs": 0.0075, "branch": None},
    {"label": "Ks5p25",   "Ks_mult": 5.25, "f_RS_abs": 0.008,  "branch": None},
    {"label": "Ks6p25hi", "Ks_mult": 6.25, "f_RS_abs": 0.0403, "branch": "hi"},
    {"label": "anchorB",  "Ks_mult": 6.5,  "f_RS_abs": 0.011,  "branch": None},
    {"label": "Ks7p25lo", "Ks_mult": 7.25, "f_RS_abs": 0.0125, "branch": "lo"},
    {"label": "Ks7p25hi", "Ks_mult": 7.25, "f_RS_abs": 0.0288, "branch": "hi"},
    {"label": "Ks8p25lo", "Ks_mult": 8.25, "f_RS_abs": 0.0155, "branch": "lo"},
    {"label": "Ks8p25hi", "Ks_mult": 8.25, "f_RS_abs": 0.0188, "branch": "hi"},
]

TRUTH_KS_MULT = 8.5   # reference line -- synthetic truth Ks_mult

# Routing parameters -- one trend line per panel, per parameter
ROUTING_PARAMS = ["kinemvelcoef", "flowexp", "channelroughness"]
PARAM_COLOR = {
    "kinemvelcoef":     "#e76f51",
    "flowexp":          "#e9c46a",
    "channelroughness": "#457b9d",
}

# 9 metrics, phase-grouped, filling the 3x3 grid row-major -- identical
# set/order to plot_pearson_comparison.py's METRIC_KEYS.
METRICS_3x3 = [
    "first_arrival_error_min", "rising_limb_steepness_ratio", "time_to_peak_from_exc_min",
    "peak_error_pct",          "peak_timing_error_hr",         "pbias_pct",
    "duration_above_thresh_error_min", "recession_rate_ratio", "kge",
]

MIN_VALID = 10   # minimum valid (non-NaN) rows required to compute Pearson r

OUTPUT_SUBDIR = "NAnchor_99"

# =======================================================================
# PATHS
# =======================================================================
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
summary_dir  = project_root / "calibration_work" / "03_comparisons" / "summary_tables"
plot_dir     = project_root / "calibration_work" / "03_comparisons" / "sensitivity_plots" / OUTPUT_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)


# =======================================================================
# LOAD DATA -- opportunistic: skip anchors whose CSV isn't there yet
# =======================================================================
def load_anchor(label):
    path = summary_dir / f"lhs_results_anchor_{label}_{LHS_SERIES}.csv"
    if not path.exists():
        print(f"  SKIP '{label}': {path.name} not found yet.")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded '{label}': {len(df)} rows from {path.name}")
    return df


def load_failed_counts():
    """Per-anchor HANG/FAILED/BUILD_ERROR counts from the audit CSV, if it
    exists. Returns dict label -> {'n_hang': int, 'n_failed': int}."""
    path = summary_dir / f"lhs_results_anchor_FAILED_{LHS_SERIES}.csv"
    counts = {}
    if not path.exists():
        return counts
    fdf = pd.read_csv(path)
    for label, sub in fdf.groupby("anchor_label"):
        counts[label] = {
            "n_hang":   int((sub["status"] == "HANG").sum()),
            "n_failed": int((sub["status"] != "HANG").sum()),
        }
    return counts


print("\nLoading Series 99 anchor results...")
anchor_dfs = {}
for a in ANCHORS:
    df = load_anchor(a["label"])
    if df is not None:
        anchor_dfs[a["label"]] = df

if not anchor_dfs:
    raise FileNotFoundError(
        "No anchor CSVs found yet -- run run_lhs_nanchor_cvrn_99.py first."
    )

available_anchors = [a for a in ANCHORS if a["label"] in anchor_dfs]
print(f"\n{len(available_anchors)}/{len(ANCHORS)} anchors available: "
      f"{[a['label'] for a in available_anchors]}")
if len(available_anchors) < len(ANCHORS):
    missing = [a["label"] for a in ANCHORS if a["label"] not in anchor_dfs]
    print(f"  Missing (skipped -- rerun later once these finish): {missing}")

failed_counts = load_failed_counts()


# =======================================================================
# COMPUTE PEARSON r -- one row per (anchor, param, metric)
# =======================================================================
def pearson_r(df, param, metric):
    if param not in df.columns or metric not in df.columns:
        return np.nan, 0
    x = df[param].values
    y = df[metric].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    n_valid = int(mask.sum())
    if n_valid < MIN_VALID:
        return np.nan, n_valid
    r, _ = pearsonr(x[mask], y[mask])
    return r, n_valid


print("\nComputing Pearson r (param x metric) per anchor...")
rows = []
for a in available_anchors:
    label = a["label"]
    df    = anchor_dfs[label]
    fc    = failed_counts.get(label, {"n_hang": 0, "n_failed": 0})
    for metric in METRICS_3x3:
        for param in ROUTING_PARAMS:
            r, n_valid = pearson_r(df, param, metric)
            rows.append({
                "anchor_label":  label,
                "branch":        a["branch"] if a["branch"] else "single",
                "Ks_mult":       a["Ks_mult"],
                "f_RS_abs":      a["f_RS_abs"],
                "param":         param,
                "param_label":   PARAM_KEY[param]["symbol"],
                "metric":        metric,
                "metric_label":  METRIC_KEY[metric]["display_name"],
                "phase":         METRIC_KEY[metric]["phase"],
                "pearson_r":     r,
                "n_valid":       n_valid,
                "n_completed":   len(df),
                "n_hang":        fc["n_hang"],
                "n_failed":      fc["n_failed"],
            })
results = pd.DataFrame(rows)


# =======================================================================
# FIGURE -- 3x3 grid of metric panels, one trend line per routing param
# =======================================================================
print("\nBuilding figure...")
fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)

for ax, metric in zip(axes.flat, METRICS_3x3):
    sub = results[results["metric"] == metric]

    for param in ROUTING_PARAMS:
        psub = sub[sub["param"] == param].sort_values(["Ks_mult", "f_RS_abs"])
        color = PARAM_COLOR[param]

        # Trend line through every anchor point in (Ks_mult, f_RS_abs)
        # order -- the step between lo/hi branches at the same Ks_mult
        # is a real signal (branch sensitivity), not noise, so the line
        # is drawn straight through it rather than averaged or split.
        ax.plot(psub["Ks_mult"], psub["pearson_r"], "-", color=color,
                 linewidth=1.3, alpha=0.7, zorder=1)

        lo_pts = psub[psub["branch"] != "hi"]
        hi_pts = psub[psub["branch"] == "hi"]
        ax.scatter(lo_pts["Ks_mult"], lo_pts["pearson_r"], s=42,
                   facecolors=color, edgecolors=color, linewidths=1.3, zorder=2)
        ax.scatter(hi_pts["Ks_mult"], hi_pts["pearson_r"], s=42,
                   facecolors="none", edgecolors=color, linewidths=1.3, zorder=2)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=0)
    ax.axvline(TRUTH_KS_MULT, color="black", linewidth=0.8,
               linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(METRIC_KEY[metric]["display_name"], fontsize=9.5)
    ax.tick_params(labelsize=8)

for ax in axes[-1, :]:
    ax.set_xlabel("Anchor Ks_mult", fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("Pearson r", fontsize=9)

# Legend: parameter colors + branch marker meaning, built manually since
# each panel only plots a subset of (param, branch) combinations.
legend_handles = [
    plt.Line2D([0], [0], color=PARAM_COLOR[p], marker="o",
               markerfacecolor=PARAM_COLOR[p], markeredgecolor=PARAM_COLOR[p],
               linewidth=1.3, label=PARAM_KEY[p]["symbol"])
    for p in ROUTING_PARAMS
] + [
    plt.Line2D([0], [0], color="gray", marker="o", markerfacecolor="gray",
               markeredgecolor="gray", linestyle="None",
               label="lo-f branch / single anchor"),
    plt.Line2D([0], [0], color="gray", marker="o", markerfacecolor="none",
               markeredgecolor="gray", linestyle="None",
               label="hi-f branch (same Ks_mult)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
          fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

n_total_completed = sum(len(df) for df in anchor_dfs.values())
n_total_hang   = sum(v["n_hang"]   for v in failed_counts.values())
n_total_failed = sum(v["n_failed"] for v in failed_counts.values())
fig.suptitle(
    f"Series 99 \u2014 cv/r/n identifiability across {len(available_anchors)} "
    f"volume-matched Ks/f anchors\n"
    f"{n_total_completed} completed draws"
    + (f"  ({n_total_hang} hung, {n_total_failed} failed \u2014 see audit CSV)"
       if (n_total_hang or n_total_failed) else "")
    + f"  |  dashed line = synthetic truth Ks_mult ({TRUTH_KS_MULT}x)",
    fontsize=11, y=1.02)

fig.tight_layout(rect=[0, 0.04, 1, 0.97])

fig_path = plot_dir / f"fig_pearson_nanchor_trend_{LHS_SERIES}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to:\n  {fig_path}")
plt.close(fig)


# =======================================================================
# COMPANION CSV
# =======================================================================
csv_path = plot_dir / f"pearson_nanchor_trend_{LHS_SERIES}.csv"
results.sort_values(["metric", "param", "Ks_mult", "f_RS_abs"]).to_csv(csv_path, index=False)
print(f"Companion CSV saved to:\n  {csv_path}\n")


# =======================================================================
# ANCHORA vs ANCHORB DELTA TABLE -- color-coded companion to the trend
# grid above, giving back the precise pairwise comparison the Series 96
# heatmap provided. Pulled straight out of the `results` dataframe
# already computed for the 3x3 grid -- these Pearson r values are the
# same anchorA/anchorB points sitting on the trend lines, just isolated
# and differenced. Note these are freshly computed under Series 99's
# shared seed (n=50 by default), not reused from the archived Series 96
# CSVs (n=100), so small numeric differences from the old plot are
# expected even though both are estimating the same thing.
# =======================================================================
DELTA_ANCHOR_A = "anchorA"
DELTA_ANCHOR_B = "anchorB"

if DELTA_ANCHOR_A in anchor_dfs and DELTA_ANCHOR_B in anchor_dfs:
    print(f"Building {DELTA_ANCHOR_A} vs {DELTA_ANCHOR_B} delta table...")

    ks_a = next(a["Ks_mult"] for a in ANCHORS if a["label"] == DELTA_ANCHOR_A)
    ks_b = next(a["Ks_mult"] for a in ANCHORS if a["label"] == DELTA_ANCHOR_B)

    delta_rows = []
    for metric in METRICS_3x3:
        row = {"metric": metric, "metric_label": METRIC_KEY[metric]["display_name"],
               "phase": METRIC_KEY[metric]["phase"]}
        for param in ROUTING_PARAMS:
            r_a_vals = results[(results["anchor_label"] == DELTA_ANCHOR_A) &
                                (results["metric"] == metric) &
                                (results["param"] == param)]["pearson_r"].values
            r_b_vals = results[(results["anchor_label"] == DELTA_ANCHOR_B) &
                                (results["metric"] == metric) &
                                (results["param"] == param)]["pearson_r"].values
            r_a = r_a_vals[0] if len(r_a_vals) else np.nan
            r_b = r_b_vals[0] if len(r_b_vals) else np.nan
            row[f"r_{DELTA_ANCHOR_A}_{param}"] = r_a
            row[f"r_{DELTA_ANCHOR_B}_{param}"] = r_b
            row[f"delta_{param}"] = r_b - r_a
        delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)

    delta_matrix = delta_df[[f"delta_{p}" for p in ROUTING_PARAMS]].values.astype(float)

    fig2, ax2 = plt.subplots(figsize=(6.2, 0.55 * len(METRICS_3x3) + 2.0))
    vmax = max(0.05, np.nanmax(np.abs(delta_matrix)))
    im = ax2.imshow(delta_matrix, cmap="PiYG", vmin=-vmax, vmax=vmax, aspect="auto")

    ax2.set_xticks(range(len(ROUTING_PARAMS)))
    ax2.set_xticklabels([PARAM_KEY[p]["symbol"] for p in ROUTING_PARAMS], fontsize=10)
    ax2.set_yticks(range(len(METRICS_3x3)))
    ax2.set_yticklabels([METRIC_KEY[m]["display_name"] for m in METRICS_3x3], fontsize=9)

    for i in range(delta_matrix.shape[0]):
        for j in range(delta_matrix.shape[1]):
            v = delta_matrix[i, j]
            txt = "NaN" if np.isnan(v) else f"{v:+.3f}"
            text_color = "white" if (not np.isnan(v) and abs(v) > vmax * 0.6) else "black"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=text_color)

    cbar = fig2.colorbar(im, ax=ax2, fraction=0.08, pad=0.03)
    cbar.set_label(f"\u0394 Pearson r  ({DELTA_ANCHOR_B} \u2212 {DELTA_ANCHOR_A})", fontsize=9)

    ax2.set_title(
        f"{DELTA_ANCHOR_A} (Ks={ks_a}x) vs {DELTA_ANCHOR_B} (Ks={ks_b}x)\n"
        f"cv / r / n \u2013 metric Pearson r shift",
        fontsize=10.5)

    fig2.tight_layout()
    delta_fig_path = plot_dir / f"fig_pearson_{DELTA_ANCHOR_A}_vs_{DELTA_ANCHOR_B}_delta_{LHS_SERIES}.png"
    fig2.savefig(delta_fig_path, dpi=150, bbox_inches="tight")
    print(f"Delta table figure saved to:\n  {delta_fig_path}")
    plt.close(fig2)

    delta_csv_path = plot_dir / f"pearson_{DELTA_ANCHOR_A}_vs_{DELTA_ANCHOR_B}_delta_{LHS_SERIES}.csv"
    delta_df.to_csv(delta_csv_path, index=False)
    print(f"Delta table CSV saved to:\n  {delta_csv_path}\n")
else:
    missing = [l for l in (DELTA_ANCHOR_A, DELTA_ANCHOR_B) if l not in anchor_dfs]
    print(f"Skipping {DELTA_ANCHOR_A}/{DELTA_ANCHOR_B} delta table -- missing: {missing}")


# =======================================================================
# ALL-PAIRS DELTA TABLE -- pairwise Pearson r deltas between every
# available anchor (not just anchorA/anchorB), for every metric x param
# combination. This is an analysis reference, not a presentation figure:
# with up to 9 anchors that's up to C(9,2)=36 pairwise deltas per
# (metric, param) cell, so annotation font is small by necessity.
#
# Anchors are ordered by (Ks_mult, f_RS_abs) ascending -- the same order
# as the trend-line x-axis above -- and only the upper triangle of each
# 9x9 anchor-pair matrix is drawn. The lower triangle is exactly the
# negative mirror (delta(A,B) = -delta(B,A)) and would only double the
# ink without adding information.
#
# delta_2_minus_1 = r(anchor_2) - r(anchor_1), where anchor_1 always
# precedes anchor_2 in Ks-position order (anchor_1 is the row, anchor_2
# is the column, in both the CSV and the figure).
# =======================================================================
pair_anchor_labels = [a["label"] for a in available_anchors]
anchor_ks = {a["label"]: a["Ks_mult"] for a in available_anchors}
n_anchors = len(pair_anchor_labels)

if n_anchors < 2:
    print("Skipping all-pairs delta table -- need at least 2 anchors, "
          f"only {n_anchors} available.")
else:
    print(f"Building all-pairs delta table ({n_anchors} anchors, "
          f"{n_anchors * (n_anchors - 1) // 2} pairs per metric x param cell)...")

    pivot_r = results.pivot_table(index=["metric", "param"], columns="anchor_label",
                                   values="pearson_r")

    pair_rows = []
    delta_matrices = {}   # (metric, param) -> n_anchors x n_anchors array, upper tri filled
    for metric in METRICS_3x3:
        for param in ROUTING_PARAMS:
            if (metric, param) in pivot_r.index:
                vals = pivot_r.loc[(metric, param)].reindex(pair_anchor_labels).values.astype(float)
            else:
                vals = np.full(n_anchors, np.nan)

            mat = np.full((n_anchors, n_anchors), np.nan)
            for i in range(n_anchors):
                for j in range(i + 1, n_anchors):
                    d = vals[j] - vals[i]
                    mat[i, j] = d
                    pair_rows.append({
                        "anchor_1":      pair_anchor_labels[i],
                        "anchor_2":      pair_anchor_labels[j],
                        "Ks_mult_1":     anchor_ks[pair_anchor_labels[i]],
                        "Ks_mult_2":     anchor_ks[pair_anchor_labels[j]],
                        "metric":        metric,
                        "metric_label":  METRIC_KEY[metric]["display_name"],
                        "param":         param,
                        "param_label":   PARAM_KEY[param]["symbol"],
                        "r_1":           vals[i],
                        "r_2":           vals[j],
                        "delta_2_minus_1": d,
                    })
            delta_matrices[(metric, param)] = mat

    pairs_df = pd.DataFrame(pair_rows)
    pairs_csv_path = plot_dir / f"pearson_allpairs_delta_{LHS_SERIES}.csv"
    pairs_df.to_csv(pairs_csv_path, index=False)
    print(f"All-pairs delta CSV saved to:\n  {pairs_csv_path}  ({len(pairs_df)} rows)")

    all_deltas = np.concatenate([m[~np.isnan(m)] for m in delta_matrices.values()])
    vmax_global = max(0.05, np.nanmax(np.abs(all_deltas))) if all_deltas.size else 0.05
    cmap3 = plt.get_cmap("PiYG")
    norm3 = mcolors.TwoSlopeNorm(vmin=-vmax_global, vcenter=0.0, vmax=vmax_global)

    fig3, axes3 = plt.subplots(
        len(METRICS_3x3), len(ROUTING_PARAMS),
        figsize=(len(ROUTING_PARAMS) * 2.4, len(METRICS_3x3) * 2.2),
    )

    for i, metric in enumerate(METRICS_3x3):
        for j, param in enumerate(ROUTING_PARAMS):
            ax  = axes3[i, j]
            mat = delta_matrices[(metric, param)]
            ax.imshow(mat, cmap=cmap3, norm=norm3, aspect="equal")

            for r in range(n_anchors):
                for c in range(n_anchors):
                    v = mat[r, c]
                    if np.isnan(v):
                        continue
                    txt_color = "white" if abs(v) > vmax_global * 0.6 else "black"
                    ax.text(c, r, f"{v:+.2f}", ha="center", va="center",
                            fontsize=3.6, color=txt_color)

            ax.set_xticks(range(n_anchors))
            ax.set_yticks(range(n_anchors))
            if i == len(METRICS_3x3) - 1:
                ax.set_xticklabels(pair_anchor_labels, fontsize=5, rotation=90)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels(pair_anchor_labels, fontsize=5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=0)

            if i == 0:
                ax.set_title(PARAM_KEY[param]["symbol"], fontsize=10)
            if j == 0:
                ax.set_ylabel(METRIC_KEY[metric]["display_name"], fontsize=6.5)

            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

    cbar_ax = fig3.add_axes([0.92, 0.15, 0.015, 0.7])
    fig3.colorbar(plt.cm.ScalarMappable(cmap=cmap3, norm=norm3), cax=cbar_ax,
                  label="\u0394 Pearson r  (column anchor \u2212 row anchor)")

    fig3.suptitle(
        f"Series 99 \u2014 all-pairs Pearson r delta matrix "
        f"({n_anchors} anchors, upper triangle only)\n"
        f"Analysis reference, not presentation scale \u2014 see companion CSV for exact values",
        fontsize=10, y=0.995)

    fig3.subplots_adjust(left=0.14, right=0.90, top=0.94, bottom=0.05, wspace=0.15, hspace=0.30)

    allpairs_fig_path = plot_dir / f"fig_pearson_allpairs_delta_{LHS_SERIES}.png"
    fig3.savefig(allpairs_fig_path, dpi=200, bbox_inches="tight")
    print(f"All-pairs delta figure saved to:\n  {allpairs_fig_path}")
    plt.close(fig3)
