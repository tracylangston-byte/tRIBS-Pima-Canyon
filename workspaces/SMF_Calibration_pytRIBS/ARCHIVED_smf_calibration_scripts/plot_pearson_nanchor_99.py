"""
plot_pearson_nanchor_99.py
===========================
Series 99 -- cv/r/n identifiability across 9 volume-matched Ks_mult/f_RS_abs
anchors spanning the Ks-f equifinality "swoosh" (4.25x-8.25x).

Produces four outputs:

  1. Trend grid, swoosh order (fig_pearson_nanchor_trend_99.png)
     4x3 grid: the original 9 metrics plus kge_r, kge_alpha, kge_beta,
     one line per routing parameter (cv, r, n), plotted across all
     available anchors in swoosh-path order. Legend above the grid,
     Ks/f value table below it.

  2. Trend grid, Ks order (fig_pearson_nanchor_trend_ks_order_99.png)
     Same 12-panel content, but plotted on a continuous Ks_mult x-axis
     (the original style) instead of the swoosh path -- Ks7.25lo/hi and
     Ks8.25lo/hi share an x position, distinguished only by marker fill.
     Includes the synthetic-truth Ks_mult=8.5x reference line, which is
     only meaningful on a real numeric Ks axis (dropped from the swoosh
     version for that reason).

  3. Two-anchor comparison (fig_pearson_<label1>_vs_<label2>_delta_99.png)
     Color-coded Pearson r delta heatmap between any two anchors you pick
     via COMPARE_ANCHOR_1 / COMPARE_ANCHOR_2 below. Still the original
     9-metric set -- not expanded to the KGE components.

  4. All-pairs delta matrix (fig_pearson_allpairs_delta_99.png)
     27 small panels (9 metrics x 3 params), each a 9x9 anchor-pair delta
     matrix, ordered along the swoosh path on both axes. Still the
     original 9-metric set. Analysis reference, not presentation scale.

One companion CSV backs both trend-grid figures and holds all 12 metrics
(pearson_nanchor_trend_99.csv) -- the two figures are just two different
views of the same underlying data.

Anchor labels and the "p" convention
--------------------------------------
Internally each anchor keeps its raw `label` (e.g. "Ks4p25", "anchorA")
because that's what matches the actual CSV filenames written by
run_lhs_nanchor_cvrn_99.py. Every anchor also has a `display_label`
(e.g. "Ks4.25", "Ks5.0", "Ks6.5lo") used everywhere a person actually
reads the anchor name. Companion CSVs keep the raw label column(s) for
joins/lookups and add a matching *_display column alongside for
readability.

The Ks-f "swoosh" path
------------------------
Two anchor positions (Ks7.25x, Ks8.25x) each have a low-f and a high-f
volume-matched solution. The swoosh path orders anchors along the actual
equifinality curve instead of plain Ks-ascending, so path-adjacent
anchors stay next to each other:

    Ks4.25 -> Ks5.0 -> Ks5.25 -> Ks6.5lo -> Ks7.25lo -> Ks8.25lo
           -> Ks8.25hi -> Ks7.25hi -> Ks6.25hi

This drives the swoosh trend grid and the all-pairs matrix. The Ks-order
trend grid instead sorts by (Ks_mult, f_RS_abs) ascending and plots on a
real numeric Ks_mult axis, so Ks7.25lo/hi and Ks8.25lo/hi land on top of
each other -- the vertical jump at those x positions is a real branch
effect, not a plotting artifact.

Partial-completion behavior
------------------------------
Every section skips anchors whose CSV isn't there yet, with a console
warning, rather than requiring all 9 to be present.

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

# Mirrors ANCHORS in run_lhs_nanchor_cvrn_99.py (Ks6p25lo stays excluded
# there and is therefore absent here too). `label` must match the run
# script's naming exactly -- it's used to find lhs_results_anchor_<label>_99.csv.
ANCHORS = [
    {"label": "Ks4p25",   "display_label": "Ks4.25",   "Ks_mult": 4.25, "f_RS_abs": 0.0055, "branch": None},
    {"label": "anchorA",  "display_label": "Ks5.0",    "Ks_mult": 5.0,  "f_RS_abs": 0.0075, "branch": None},
    {"label": "Ks5p25",   "display_label": "Ks5.25",   "Ks_mult": 5.25, "f_RS_abs": 0.008,  "branch": None},
    {"label": "Ks6p25hi", "display_label": "Ks6.25hi", "Ks_mult": 6.25, "f_RS_abs": 0.0403, "branch": "hi"},
    {"label": "anchorB",  "display_label": "Ks6.5lo",  "Ks_mult": 6.5,  "f_RS_abs": 0.011,  "branch": "lo"},
    {"label": "Ks7p25lo", "display_label": "Ks7.25lo", "Ks_mult": 7.25, "f_RS_abs": 0.0125, "branch": "lo"},
    {"label": "Ks7p25hi", "display_label": "Ks7.25hi", "Ks_mult": 7.25, "f_RS_abs": 0.0288, "branch": "hi"},
    {"label": "Ks8p25lo", "display_label": "Ks8.25lo", "Ks_mult": 8.25, "f_RS_abs": 0.0155, "branch": "lo"},
    {"label": "Ks8p25hi", "display_label": "Ks8.25hi", "Ks_mult": 8.25, "f_RS_abs": 0.0188, "branch": "hi"},
]
ANCHOR_LOOKUP = {a["label"]: a for a in ANCHORS}

# Anchor order along the Ks-f equifinality "swoosh" (not plain Ks-ascending):
# up the lo-f branch, across the peak at Ks8.25, back down the hi-f branch.
# Used for the swoosh trend grid and the all-pairs matrix row/column order.
SWOOSH_ORDER = ["Ks4p25", "anchorA", "Ks5p25", "anchorB",
                "Ks7p25lo", "Ks8p25lo", "Ks8p25hi", "Ks7p25hi", "Ks6p25hi"]

TRUTH_KS_MULT = 8.5   # synthetic-truth reference -- only meaningful on a real Ks_mult axis

# Routing parameters -- one trend line per panel, per parameter
ROUTING_PARAMS = ["kinemvelcoef", "flowexp", "channelroughness"]
PARAM_COLOR = {
    "kinemvelcoef":     "#e76f51",
    "flowexp":          "#e9c46a",
    "channelroughness": "#457b9d",
}

# Original 9 metrics -- still used by the two-anchor comparison and the
# all-pairs delta matrix (Sections 2 and 3), unchanged.
METRICS_3x3 = [
    "first_arrival_error_min", "rising_limb_steepness_ratio", "time_to_peak_from_exc_min",
    "peak_error_pct",          "peak_timing_error_hr",         "pbias_pct",
    "duration_above_thresh_error_min", "recession_rate_ratio", "kge",
]

# Trend-grid metric set: the original 9 plus the three KGE components,
# filling a 4x3 grid. Only used by the two trend-grid figures (Section 1).
METRICS_TREND_GRID = METRICS_3x3 + ["kge_r", "kge_alpha", "kge_beta"]

# METRIC_KEY (from parameter_key.py) may not define the three KGE
# components -- these overrides guarantee the exact labels requested,
# and metric_display()/metric_phase() fall back to METRIC_KEY for
# everything else.
METRIC_DISPLAY_OVERRIDE = {
    "kge_r":     "KGE r (correlation)",
    "kge_alpha": "KGE alpha (flashiness)",
    "kge_beta":  "KGE beta (flow volume)",
}
METRIC_PHASE_OVERRIDE = {
    "kge_r": "summary", "kge_alpha": "summary", "kge_beta": "summary",
}

def metric_display(metric):
    return METRIC_DISPLAY_OVERRIDE.get(metric) or METRIC_KEY[metric]["display_name"]

def metric_phase(metric):
    return METRIC_PHASE_OVERRIDE.get(metric) or METRIC_KEY[metric]["phase"]

MIN_VALID = 10   # minimum valid (non-NaN) rows required to compute Pearson r

# Two-anchor focused comparison (Section 2) -- swap either value for any
# other anchor's raw `label` from ANCHORS above to compare a different pair.
COMPARE_ANCHOR_1 = "Ks8p25lo"
COMPARE_ANCHOR_2 = "Ks8p25hi"

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
    print(f"  Loaded '{label}' ({ANCHOR_LOOKUP[label]['display_label']}): "
          f"{len(df)} rows from {path.name}")
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

available_labels = [a["label"] for a in ANCHORS if a["label"] in anchor_dfs]
print(f"\n{len(available_labels)}/{len(ANCHORS)} anchors available: "
      f"{[ANCHOR_LOOKUP[l]['display_label'] for l in available_labels]}")
if len(available_labels) < len(ANCHORS):
    missing = [a["label"] for a in ANCHORS if a["label"] not in anchor_dfs]
    print(f"  Missing (skipped -- rerun later once these finish): "
          f"{[ANCHOR_LOOKUP[l]['display_label'] for l in missing]}")

failed_counts = load_failed_counts()

# Swoosh-path order, filtered down to whatever anchors are actually present.
swoosh_available = [lbl for lbl in SWOOSH_ORDER if lbl in anchor_dfs]
n_anchors = len(swoosh_available)


# =======================================================================
# COMPUTE PEARSON r -- one row per (anchor, param, metric). Uses the
# 12-metric superset (METRICS_TREND_GRID) so the trend grid has what it
# needs; Sections 2/3 simply filter this down to METRICS_3x3.
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
for label in available_labels:
    a  = ANCHOR_LOOKUP[label]
    df = anchor_dfs[label]
    fc = failed_counts.get(label, {"n_hang": 0, "n_failed": 0})
    for metric in METRICS_TREND_GRID:
        for param in ROUTING_PARAMS:
            r, n_valid = pearson_r(df, param, metric)
            rows.append({
                "anchor_label":   label,
                "anchor_display": a["display_label"],
                "branch":         a["branch"] if a["branch"] else "single",
                "Ks_mult":        a["Ks_mult"],
                "f_RS_abs":       a["f_RS_abs"],
                "param":          param,
                "param_label":    PARAM_KEY[param]["symbol"],
                "metric":         metric,
                "metric_label":   metric_display(metric),
                "phase":          metric_phase(metric),
                "pearson_r":      r,
                "n_valid":        n_valid,
                "n_completed":    len(df),
                "n_hang":         fc["n_hang"],
                "n_failed":       fc["n_failed"],
            })
results = pd.DataFrame(rows)

csv_path = plot_dir / f"pearson_nanchor_trend_{LHS_SERIES}.csv"
results.sort_values(["metric", "param", "Ks_mult", "f_RS_abs"]).to_csv(csv_path, index=False)
print(f"Companion CSV saved to:\n  {csv_path}  ({len(results)} rows, {len(METRICS_TREND_GRID)} metrics)\n")


# =======================================================================
# SECTION 1 -- TREND GRID (built twice: swoosh order and Ks order)
# 4x3 panels (12 metrics), legend above the grid, Ks/f value table below.
# =======================================================================
n_metric_rows = len(METRICS_TREND_GRID) // 3

n_total_completed = sum(len(df) for df in anchor_dfs.values())
n_total_hang   = sum(v["n_hang"]   for v in failed_counts.values())
n_total_failed = sum(v["n_failed"] for v in failed_counts.values())

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
               label="hi-f branch (paired lo/hi, same Ks_mult)"),
]


def fmt_ks(k):
    s = f"{k:.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def build_trend_grid(mode):
    """mode: 'swoosh' (categorical, swoosh-path x-axis) or
    'ks_continuous' (real numeric Ks_mult x-axis, original style)."""
    assert mode in ("swoosh", "ks_continuous")

    if mode == "swoosh":
        order_labels = swoosh_available
        x_of         = {lbl: i for i, lbl in enumerate(order_labels)}
        sort_cols    = ["x"]
        xlabel_text  = "Anchor (Ks_mult, swoosh order)"
        caption_text = "Anchor Ks_mult / f_RS_abs values (swoosh order, left to right)"
        mode_title   = "swoosh path order"
        out_stub     = f"fig_pearson_nanchor_trend_{LHS_SERIES}"
    else:
        order_labels = sorted(available_labels,
                               key=lambda l: (ANCHOR_LOOKUP[l]["Ks_mult"], ANCHOR_LOOKUP[l]["f_RS_abs"]))
        x_of         = {lbl: ANCHOR_LOOKUP[lbl]["Ks_mult"] for lbl in order_labels}
        sort_cols    = ["Ks_mult", "f_RS_abs"]
        xlabel_text  = "Anchor Ks_mult"
        caption_text = "Anchor Ks_mult / f_RS_abs values (Ks order, low to high)"
        mode_title   = "Ks order, low to high"
        out_stub     = f"fig_pearson_nanchor_trend_ks_order_{LHS_SERIES}"

    n = len(order_labels)
    table_display = [ANCHOR_LOOKUP[l]["display_label"] for l in order_labels]

    # ---- fixed inch budgets, converted to figure fractions below -------
    fig_h = 3.6 * n_metric_rows + 4.3

    suptitle_from_top_in = 0.10
    legend_from_top_in   = 0.85
    top_budget_in        = 1.70   # total space reserved above the grid

    table_bottom_in   = 0.15
    table_height_in   = 0.45
    caption_gap_in     = 0.15
    caption_height_in  = 0.20
    grid_gap_in        = 0.45   # gap between last row's xlabel and the caption -- widened per feedback
    xlabel_ticks_in    = 0.55   # space consumed by tick labels + axis label below each panel
    bottom_budget_in = (table_bottom_in + table_height_in + caption_gap_in
                         + caption_height_in + grid_gap_in + xlabel_ticks_in)

    fig, axes = plt.subplots(n_metric_rows, 3, figsize=(13, fig_h), sharey=True)

    for ax, metric in zip(axes.flat, METRICS_TREND_GRID):
        sub = results[results["metric"] == metric]

        for param in ROUTING_PARAMS:
            psub = sub[sub["param"] == param].copy()
            psub["x"] = psub["anchor_label"].map(x_of)
            psub = psub.sort_values(sort_cols)
            color = PARAM_COLOR[param]

            # Trend line through every anchor -- the step between lo/hi
            # branches (same x in ks_continuous mode, adjacent-ish in
            # swoosh mode) is a real branch-sensitivity signal, not noise.
            ax.plot(psub["x"], psub["pearson_r"], "-", color=color,
                     linewidth=1.3, alpha=0.7, zorder=1)

            lo_pts = psub[psub["branch"] != "hi"]
            hi_pts = psub[psub["branch"] == "hi"]
            ax.scatter(lo_pts["x"], lo_pts["pearson_r"], s=42,
                       facecolors=color, edgecolors=color, linewidths=1.3, zorder=2)
            ax.scatter(hi_pts["x"], hi_pts["pearson_r"], s=42,
                       facecolors="none", edgecolors=color, linewidths=1.3, zorder=2)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=0)
        ax.set_ylim(-1.05, 1.05)

        if mode == "swoosh":
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_xticks(range(n))
            ax.set_xticklabels(table_display, fontsize=6.5, rotation=90)
        else:
            ks_vals = sorted({ANCHOR_LOOKUP[l]["Ks_mult"] for l in order_labels})
            ax.set_xlim(min(ks_vals) - 0.5, max(TRUTH_KS_MULT, max(ks_vals)) + 0.5)
            ax.set_xticks(ks_vals)
            ax.set_xticklabels([fmt_ks(k) for k in ks_vals], fontsize=8, rotation=0)
            ax.axvline(TRUTH_KS_MULT, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.4, zorder=0)

        title_text = metric_display(metric)
        if metric == "kge_beta":
            title_text = f"{title_text}\n(same signal as PBIAS)"
        ax.set_title(title_text, fontsize=11, fontweight="semibold", pad=9)
        ax.set_xlabel(xlabel_text, fontsize=8.5)
        ax.tick_params(labelsize=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("Pearson r", fontsize=9)

    top_frac    = 1 - top_budget_in / fig_h
    bottom_frac = bottom_budget_in / fig_h
    fig.subplots_adjust(left=0.06, right=0.98, top=top_frac, bottom=bottom_frac,
                         hspace=0.95, wspace=0.15)

    fig.legend(handles=legend_handles, loc="upper center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1 - legend_from_top_in / fig_h))

    fig.suptitle(
        f"Series 99 \u2014 cv/r/n identifiability across {n} "
        f"volume-matched Ks/f anchors ({mode_title})\n"
        f"{n_total_completed} completed draws"
        + (f"  ({n_total_hang} hung, {n_total_failed} failed \u2014 see audit CSV)"
           if (n_total_hang or n_total_failed) else ""),
        fontsize=12.5, fontweight="bold", y=1 - suptitle_from_top_in / fig_h)

    caption_y = (table_bottom_in + table_height_in + caption_gap_in) / fig_h
    fig.text(0.5, caption_y, caption_text, ha="center", va="bottom",
              fontsize=8.5, fontweight="bold")

    table_ax = fig.add_axes([0.06, table_bottom_in / fig_h, 0.90, table_height_in / fig_h])
    table_ax.axis("off")
    cell_text = [[f"{ANCHOR_LOOKUP[l]['Ks_mult']:.2f} / {ANCHOR_LOOKUP[l]['f_RS_abs']:.4f}"
                  for l in order_labels]]
    tbl = table_ax.table(cellText=cell_text, colLabels=table_display,
                          loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)

    fig_path = plot_dir / f"{out_stub}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to:\n  {fig_path}")
    plt.close(fig)


print("Building trend grid (swoosh path order)...")
build_trend_grid("swoosh")

print("\nBuilding trend grid (Ks order, continuous axis)...")
build_trend_grid("ks_continuous")


# =======================================================================
# SECTION 2 -- TWO-ANCHOR COMPARISON: pick any pair via COMPARE_ANCHOR_1/2
# in CONFIG above. Defaults to Ks8.25lo vs Ks8.25hi. Original 9-metric set.
# =======================================================================
def build_anchor_pair_delta(label_1, label_2):
    if label_1 not in anchor_dfs or label_2 not in anchor_dfs:
        missing = [l for l in (label_1, label_2) if l not in anchor_dfs]
        print(f"Skipping two-anchor comparison -- missing: "
              f"{[ANCHOR_LOOKUP[l]['display_label'] for l in missing]}")
        return

    a1, a2 = ANCHOR_LOOKUP[label_1], ANCHOR_LOOKUP[label_2]
    d1, d2 = a1["display_label"], a2["display_label"]
    print(f"Building {d1} vs {d2} comparison...")

    delta_rows = []
    for metric in METRICS_3x3:
        row = {
            "anchor_1_label": label_1, "anchor_1_display": d1,
            "anchor_2_label": label_2, "anchor_2_display": d2,
            "metric": metric, "metric_label": metric_display(metric),
            "phase": metric_phase(metric),
        }
        for param in ROUTING_PARAMS:
            r_1_vals = results[(results["anchor_label"] == label_1) &
                                (results["metric"] == metric) &
                                (results["param"] == param)]["pearson_r"].values
            r_2_vals = results[(results["anchor_label"] == label_2) &
                                (results["metric"] == metric) &
                                (results["param"] == param)]["pearson_r"].values
            r_1 = r_1_vals[0] if len(r_1_vals) else np.nan
            r_2 = r_2_vals[0] if len(r_2_vals) else np.nan
            row[f"r_{param}_{d1}"] = r_1
            row[f"r_{param}_{d2}"] = r_2
            row[f"delta_{param}"]  = r_2 - r_1
        delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)

    delta_matrix = delta_df[[f"delta_{p}" for p in ROUTING_PARAMS]].values.astype(float)

    fig2, ax2 = plt.subplots(figsize=(6.2, 0.55 * len(METRICS_3x3) + 2.0))
    vmax = max(0.05, np.nanmax(np.abs(delta_matrix)))
    im = ax2.imshow(delta_matrix, cmap="PiYG", vmin=-vmax, vmax=vmax, aspect="auto")

    ax2.set_xticks(range(len(ROUTING_PARAMS)))
    ax2.set_xticklabels([PARAM_KEY[p]["symbol"] for p in ROUTING_PARAMS], fontsize=10)
    ax2.set_yticks(range(len(METRICS_3x3)))
    ax2.set_yticklabels([metric_display(m) for m in METRICS_3x3], fontsize=9)

    for i in range(delta_matrix.shape[0]):
        for j in range(delta_matrix.shape[1]):
            v = delta_matrix[i, j]
            txt = "NaN" if np.isnan(v) else f"{v:+.3f}"
            text_color = "white" if (not np.isnan(v) and abs(v) > vmax * 0.6) else "black"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=text_color)

    cbar = fig2.colorbar(im, ax=ax2, fraction=0.08, pad=0.03)
    cbar.set_label(f"\u0394 Pearson r  ({d2} \u2212 {d1})", fontsize=9)

    ax2.set_title(
        f"{d1} (Ks={a1['Ks_mult']:.2f}x, f={a1['f_RS_abs']:.4f}) vs "
        f"{d2} (Ks={a2['Ks_mult']:.2f}x, f={a2['f_RS_abs']:.4f})\n"
        f"cv / r / n \u2013 metric Pearson r shift",
        fontsize=10.5)

    fig2.tight_layout()
    fig2_path = plot_dir / f"fig_pearson_{label_1}_vs_{label_2}_delta_{LHS_SERIES}.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved to:\n  {fig2_path}")
    plt.close(fig2)

    delta_csv_path = plot_dir / f"pearson_{label_1}_vs_{label_2}_delta_{LHS_SERIES}.csv"
    delta_df.to_csv(delta_csv_path, index=False)
    print(f"Comparison CSV saved to:\n  {delta_csv_path}\n")


build_anchor_pair_delta(COMPARE_ANCHOR_1, COMPARE_ANCHOR_2)


# =======================================================================
# SECTION 3 -- ALL-PAIRS DELTA MATRIX: pairwise Pearson r deltas between
# every available anchor, for every metric x param combination, ordered
# along the swoosh path (not Ks-ascending) on both axes. Original
# 9-metric set. Analysis reference, not presentation scale -- with 9
# anchors that's up to C(9,2)=36 pairwise deltas per (metric, param)
# cell, so annotation font is small by necessity. Only the upper
# triangle is drawn (the lower triangle is exactly the negative mirror).
#
# delta_2_minus_1 = r(anchor_2) - r(anchor_1), where anchor_1 always
# precedes anchor_2 along the swoosh path (anchor_1 is the row, anchor_2
# is the column, in both the CSV and the figure).
# =======================================================================
if n_anchors < 2:
    print("Skipping all-pairs delta table -- need at least 2 anchors, "
          f"only {n_anchors} available.")
else:
    print(f"Building all-pairs delta table ({n_anchors} anchors, swoosh order, "
          f"{n_anchors * (n_anchors - 1) // 2} pairs per metric x param cell)...")

    pivot_r = results.pivot_table(index=["metric", "param"], columns="anchor_label",
                                   values="pearson_r")

    swoosh_display = [ANCHOR_LOOKUP[l]["display_label"] for l in swoosh_available]

    pair_rows = []
    delta_matrices = {}   # (metric, param) -> n_anchors x n_anchors array, upper tri filled
    for metric in METRICS_3x3:
        for param in ROUTING_PARAMS:
            if (metric, param) in pivot_r.index:
                vals = pivot_r.loc[(metric, param)].reindex(swoosh_available).values.astype(float)
            else:
                vals = np.full(n_anchors, np.nan)

            mat = np.full((n_anchors, n_anchors), np.nan)
            for i in range(n_anchors):
                for j in range(i + 1, n_anchors):
                    d = vals[j] - vals[i]
                    mat[i, j] = d
                    pair_rows.append({
                        "anchor_1":         swoosh_available[i],
                        "anchor_1_display": swoosh_display[i],
                        "anchor_2":         swoosh_available[j],
                        "anchor_2_display": swoosh_display[j],
                        "Ks_mult_1":        ANCHOR_LOOKUP[swoosh_available[i]]["Ks_mult"],
                        "Ks_mult_2":        ANCHOR_LOOKUP[swoosh_available[j]]["Ks_mult"],
                        "metric":           metric,
                        "metric_label":     metric_display(metric),
                        "param":            param,
                        "param_label":      PARAM_KEY[param]["symbol"],
                        "r_1":              vals[i],
                        "r_2":              vals[j],
                        "delta_2_minus_1":  d,
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
        figsize=(len(ROUTING_PARAMS) * 2.6, len(METRICS_3x3) * 2.6),
    )

    for i, metric in enumerate(METRICS_3x3):
        for j, param in enumerate(ROUTING_PARAMS):
            ax  = axes3[i, j]
            mat = delta_matrices[(metric, param)]
            ax.imshow(mat, cmap=cmap3, norm=norm3, aspect="equal")

            for ri in range(n_anchors):
                for ci in range(n_anchors):
                    v = mat[ri, ci]
                    if np.isnan(v):
                        continue
                    txt_color = "white" if abs(v) > vmax_global * 0.6 else "black"
                    ax.text(ci, ri, f"{v:+.2f}", ha="center", va="center",
                            fontsize=3.6, color=txt_color)

            ax.set_xticks(range(n_anchors))
            ax.set_yticks(range(n_anchors))
            ax.set_xticklabels(swoosh_display, fontsize=4.2, rotation=90)
            if j == 0:
                ax.set_yticklabels(swoosh_display, fontsize=5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=0)

            if i == 0:
                ax.set_title(PARAM_KEY[param]["symbol"], fontsize=10, fontweight="semibold")
            if j == 0:
                ax.set_ylabel(metric_display(metric), fontsize=6.5)

            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

    cbar_ax = fig3.add_axes([0.92, 0.15, 0.015, 0.7])
    fig3.colorbar(plt.cm.ScalarMappable(cmap=cmap3, norm=norm3), cax=cbar_ax,
                  label="\u0394 Pearson r  (column anchor \u2212 row anchor)")

    fig3.suptitle(
        f"Series 99 \u2014 all-pairs Pearson r delta matrix "
        f"({n_anchors} anchors, swoosh path order, upper triangle only)\n"
        f"Analysis reference, not presentation scale \u2014 see companion CSV for exact values",
        fontsize=10, y=0.995)

    fig3.subplots_adjust(left=0.14, right=0.90, top=0.93, bottom=0.06,
                          wspace=0.15, hspace=0.55)

    allpairs_fig_path = plot_dir / f"fig_pearson_allpairs_delta_{LHS_SERIES}.png"
    fig3.savefig(allpairs_fig_path, dpi=200, bbox_inches="tight")
    print(f"All-pairs delta figure saved to:\n  {allpairs_fig_path}")
    plt.close(fig3)
