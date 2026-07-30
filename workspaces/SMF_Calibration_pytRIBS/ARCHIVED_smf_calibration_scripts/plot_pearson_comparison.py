"""
plot_pearson_comparison.py
==========================
Compare the Pearson r (parameter × metric) matrix between two LHS sweep
series using side-by-side heatmaps with a difference panel.

Output:
    fig_pearson_comparison_{SERIES_A_LABEL}_vs_{SERIES_B_LABEL}.png

Usage (run from smf_demo):
    python plot_pearson_comparison.py

To adapt for a different pair of runs, edit the CONFIG block only.

Change log
----------
- Layout rebuilt on one explicit GridSpec: category strip | panel A |
  panel B | panel Delta | main colorbar | delta colorbar. Every colorbar
  now lives in its own dedicated axis (fig.colorbar(..., cax=...)) instead
  of the ax=[...] shorthand, which matplotlib does not lay out correctly
  together with tight_layout (it was emitting "This figure includes Axes
  that are not compatible with tight_layout" and could clip the delta
  colorbar's label). tight_layout() has been dropped entirely in favor of
  the explicit GridSpec width ratios, which is the layout-stable approach
  for a figure with manually placed colorbar axes.
- The bottom-of-figure phase-color legend is gone. Category is now shown
  as a colored, labeled strip immediately to the left of panel A's
  y-axis, spanning exactly the rows in that phase -- read directly off
  the figure instead of decoded from a legend key.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr

from parameter_key import PARAM_KEY, METRIC_KEY

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =======================================================================
# CONFIG — edit this block to change which series are compared
# =======================================================================

SERIES = {
    "A": {
        "label":       "Series 97 log",         # display name in figure title
        "csv":         "lhs_results_synth_Ks_f_97log.csv",
        # Series 97log was scored against the OLD synthetic truth
        # (retired). kinemvelcoef/flowexp/channelroughness are held fixed
        # at these values in every row -- they are NOT swept in this
        # series, hence PARAM_KEYS below uses Ks_mult/f_RS_abs instead.
        "true_values": {
            "Ks_mult":          8.50,
            "f_RS_abs":         0.020,
            "kinemvelcoef":     4.50,
            "flowexp":          0.24,
            "channelroughness": 0.026,
        },
    },
    "B": {
        "label":       "Series 100",
        "csv":         "lhs_results_synth_Ks_f_100.csv",
        # Series 100 is scored against the NEW synthetic truth
        # (Ks_mult=7.0x, f_RS_abs=0.012); cv/r/n unchanged from Series 97log.
        "true_values": {
            "Ks_mult":          7.00,
            "f_RS_abs":         0.012,
            "kinemvelcoef":     4.50,
            "flowexp":          0.24,
            "channelroughness": 0.026,
        },
    },
    # To add Series 96 Anchor A/B once needed again, swap an entry above to:
    # "A": {
    #     "label":       "Series 96 - Anchor A",
    #     "csv":         "lhs_results_anchor_anchorA_96.csv",
    #     "true_values": { "kinemvelcoef": 4.50, "flowexp": 0.24, "channelroughness": 0.026 },
    # },
    # -- and set PARAM_KEYS below to the routing params (cv/r/n), since
    # those are what's actually swept in anchor-based series like 96/99.
}

# Parameters to include (must be columns in both CSVs).
# Order here controls column order in the heatmap.
# IMPORTANT: this must match what's actually swept in the series being
# compared. Ks x f sweeps (93, 97, 97log, 100, ...) vary Ks_mult/f_RS_abs
# and hold cv/r/n fixed; anchor-based cv/r/n sweeps (96, 99) do the
# opposite. A fixed (zero-variance) column produces an undefined
# correlation -- see the variance guard in compute_pearson_matrix, which
# will now warn loudly instead of silently returning NaN if this list is
# set wrong for the series being loaded.
PARAM_KEYS = [
    "Ks_mult",
    "f_RS_abs",
]

# Metrics to include (must be columns in both CSVs).
# Order here controls row order in the heatmap.
METRIC_KEYS = [
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    "peak_error_pct",
    "peak_timing_error_hr",
    "pbias_pct",
    "duration_above_thresh_error_min",
    "recession_rate_ratio",
    "kge",
]

# Minimum valid rows required to compute a correlation for a given cell.
MIN_VALID = 10

# Output directory label (subfolder under sensitivity_plots/)
COMPARISON_SUBDIR = "Comparisons"


# =======================================================================
# DISPLAY METADATA
# Derived from parameter_key.py — add/remove entries to match PARAM_KEYS
# and METRIC_KEYS above.
# =======================================================================

PARAM_META = {
    "Ks_mult":          {"label": PARAM_KEY["Ks_mult"]["symbol"],          "color": "#2a9d8f", "family": "Soil"},
    "f_RS_abs":         {"label": PARAM_KEY["f_RS_abs"]["symbol"],         "color": "#e63946", "family": "Soil"},
    "kinemvelcoef":     {"label": PARAM_KEY["kinemvelcoef"]["symbol"],     "color": "#e76f51", "family": "Routing"},
    "flowexp":          {"label": PARAM_KEY["flowexp"]["symbol"],           "color": "#e9c46a", "family": "Routing"},
    "channelroughness": {"label": PARAM_KEY["channelroughness"]["symbol"], "color": "#457b9d", "family": "Routing"},
}
# Both parameter families are kept here (not just the active PARAM_KEYS)
# so this same dict serves either a Ks x f comparison or an anchor-based
# cv/r/n comparison -- only PARAM_KEYS above needs to change between uses.

# Phase colors mirror plot_lhs_synth_permetric.py conventions
METRIC_META = {
    "first_arrival_error_min":        {"label": "First arrival\nerror (min)",          "phase": "pre-peak",  "phase_color": "#5b9bd5"},
    "rising_limb_steepness_ratio":    {"label": "Rising limb\nsteepness ratio",        "phase": "pre-peak",  "phase_color": "#5b9bd5"},
    "time_to_peak_from_exc_min":      {"label": "Time-to-peak\nfrom exc. (min)",       "phase": "pre-peak",  "phase_color": "#5b9bd5"},
    "peak_error_pct":                 {"label": "Peak discharge\nerror (%)",           "phase": "peak",      "phase_color": "#f4a261"},
    "peak_timing_error_hr":           {"label": "Peak timing\nerror (hr)",             "phase": "peak",      "phase_color": "#f4a261"},
    "pbias_pct":                      {"label": "Volume bias\n(PBIAS %)",              "phase": "volume",    "phase_color": "#2a9d8f"},
    "duration_above_thresh_error_min":{"label": "Duration above\nthreshold error (min)","phase": "volume",   "phase_color": "#2a9d8f"},
    "recession_rate_ratio":           {"label": "Recession rate\nratio",               "phase": "recession", "phase_color": "#8ecae6"},
    "kge":                            {"label": "KGE\n(summary)",                      "phase": "summary",   "phase_color": "#6a4c93"},
}

PARAM_LABELS  = [PARAM_META[k]["label"] for k in PARAM_KEYS]


# =======================================================================
# PATHS
# =======================================================================
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
summary_dir  = project_root / "calibration_work" / "03_comparisons" / "summary_tables"
plot_dir     = project_root / "calibration_work" / "03_comparisons" / "sensitivity_plots" / COMPARISON_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)


# =======================================================================
# HELPERS
# =======================================================================

def load_series(cfg: dict) -> pd.DataFrame:
    path = summary_dir / cfg["csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Run the corresponding run_lhs_synth_4param*.py first."
        )
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path.name}")
    return df


def compute_pearson_matrix(df: pd.DataFrame,
                            param_keys: list,
                            metric_keys: list,
                            series_label: str = "") -> np.ndarray:
    """Return (n_metrics × n_params) array of Pearson r values.

    Variance guard: a parameter or metric column that's held fixed
    (zero variance) in this series produces an undefined correlation --
    pearsonr raises, the old code silently caught it and left NaN, and
    an all-NaN heatmap gave no clue why. That's almost always a sign
    that PARAM_KEYS/METRIC_KEYS in CONFIG doesn't match what was
    actually swept in this CSV (e.g. comparing a Ks x f sweep while
    PARAM_KEYS is still set to the routing params cv/r/n, or vice
    versa). This now raises a visible warning naming the offending
    column(s) instead of failing silently.
    """
    n_m = len(metric_keys)
    n_p = len(param_keys)
    arr = np.full((n_m, n_p), np.nan)

    tag = f"[{series_label}] " if series_label else ""

    zero_var_params = [
        p for p in param_keys
        if p in df.columns and np.nanstd(df[p].values) < 1e-12
    ]
    zero_var_metrics = [
        m for m in metric_keys
        if m in df.columns and np.nanstd(df[m].values) < 1e-12
    ]
    if zero_var_params:
        warnings.warn(
            f"{tag}Zero-variance parameter column(s) {zero_var_params} -- "
            f"held fixed in this series, not swept. Their row/column will "
            f"be all-NaN. Check PARAM_KEYS in CONFIG matches what this "
            f"CSV actually varies.",
            stacklevel=2,
        )
    if zero_var_metrics:
        warnings.warn(
            f"{tag}Zero-variance metric column(s) {zero_var_metrics} -- "
            f"correlations against these will be all-NaN.",
            stacklevel=2,
        )

    for i, m in enumerate(metric_keys):
        if m not in df.columns or m in zero_var_metrics:
            continue
        for j, p in enumerate(param_keys):
            if p not in df.columns or p in zero_var_params:
                continue
            x = df[p].values
            y = df[m].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < MIN_VALID:
                continue
            try:
                r, _ = pearsonr(x[mask], y[mask])
                arr[i, j] = r
            except Exception:
                pass
    return arr


def phase_groups(metric_keys):
    """Collapse a list of metric keys into contiguous (phase, color, start,
    end) groups, in the order the metrics appear."""
    groups = []
    current_phase = None
    prev_color = "gray"
    start = 0
    for i, m in enumerate(metric_keys):
        phase = METRIC_META.get(m, {}).get("phase", "other")
        color = METRIC_META.get(m, {}).get("phase_color", "gray")
        if phase != current_phase:
            if current_phase is not None:
                groups.append((current_phase, prev_color, start, i - 1))
            current_phase = phase
            prev_color = color
            start = i
    groups.append((current_phase, prev_color, start, len(metric_keys) - 1))
    return groups


def draw_category_strip(ax, metric_keys, ylim):
    """Narrow labeled strip naming the hydrograph phase each block of rows
    belongs to. Sits to the left of panel A's y-axis and replaces the
    bottom-of-figure phase-color legend."""
    ax.set_xlim(0, 1)
    ax.set_ylim(ylim)
    ax.axis("off")
    for phase, color, start, end in phase_groups(metric_keys):
        height = (end - start + 1)
        ax.add_patch(mpatches.Rectangle(
            (0.15, start - 0.5), 0.7, height,
            facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.0))
        rotation = 90 if height >= 2 else 0
        ax.text(0.5, start + height / 2.0 - 0.5, phase.upper(),
                ha="center", va="center", rotation=rotation,
                fontsize=8.5, fontweight="bold", color="#333333")


def draw_heatmap(ax, arr: np.ndarray, metric_keys: list, param_labels: list,
                 title: str, cmap, norm, annotate: bool = True,
                 show_ylabel: bool = True):
    """Render one Pearson r heatmap panel onto ax."""
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")

    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v):
                    ax.text(j, i, "NaN", ha="center", va="center",
                            fontsize=7.5, color="gray")
                else:
                    text_col = "white" if abs(v) > 0.60 else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color=text_col)

    ax.set_xticks(range(len(param_labels)))
    ax.set_xticklabels(param_labels, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Parameter", fontsize=10)

    if show_ylabel:
        metric_display_labels = [
            METRIC_META[m]["label"] if m in METRIC_META else m
            for m in metric_keys
        ]
        ax.set_yticks(range(len(metric_keys)))
        ax.set_yticklabels(metric_display_labels, fontsize=8.5)
    else:
        ax.set_yticks(range(len(metric_keys)))
        ax.set_yticklabels([], fontsize=8.5)

    current_phase = None
    for i, m in enumerate(metric_keys):
        if m not in METRIC_META:
            continue
        phase = METRIC_META[m]["phase"]
        if phase != current_phase and i > 0:
            ax.axhline(i - 0.5, color="black", linewidth=1.5, alpha=0.7)
        current_phase = phase

    ax.tick_params(left=False)
    return im


# =======================================================================
# LOAD DATA
# =======================================================================
print("\nLoading series data...")
dfs = {}
for key in ("A", "B"):
    print(f"  Series {key} ({SERIES[key]['label']}):")
    dfs[key] = load_series(SERIES[key])

# Restrict to metrics present in BOTH dataframes
available_metrics = [
    m for m in METRIC_KEYS
    if m in dfs["A"].columns and m in dfs["B"].columns
]
missing = [m for m in METRIC_KEYS if m not in available_metrics]
if missing:
    print(f"\n  WARNING: metrics absent in one or both CSVs (skipped): {missing}")
if not available_metrics:
    raise ValueError("No metric columns found in both CSVs.")

available_params = [
    p for p in PARAM_KEYS
    if p in dfs["A"].columns and p in dfs["B"].columns
]
if not available_params:
    raise ValueError("No parameter columns found in both CSVs.")

n_m = len(available_metrics)
n_p = len(available_params)
param_labels_avail = [PARAM_META[k]["label"] if k in PARAM_META else k
                      for k in available_params]

print(f"\n  Using {n_m} metrics × {n_p} parameters")


# =======================================================================
# COMPUTE PEARSON r MATRICES
# =======================================================================
print("\nComputing Pearson r matrices...")
pearson = {}
for key in ("A", "B"):
    pearson[key] = compute_pearson_matrix(
        dfs[key], available_params, available_metrics,
        series_label=SERIES[key]["label"],
    )
    print(f"  {SERIES[key]['label']}: done")

delta = pearson["B"] - pearson["A"]   # Δ = B minus A


# =======================================================================
# PRINT TABLES TO CONSOLE
# =======================================================================
for key in ("A", "B"):
    label = SERIES[key]["label"]
    arr   = pearson[key]
    header = f"\nPEARSON r — {label}\n" + "-"*60
    print(header)
    col_head = f"{'Metric':<38s}" + "".join(f"{l:>8s}" for l in param_labels_avail)
    print(col_head)
    for i, m in enumerate(available_metrics):
        ml = (METRIC_META[m]["label"] if m in METRIC_META else m).replace("\n", " ")
        row = f"{ml:<38s}"
        for j in range(n_p):
            v = arr[i, j]
            row += f"  {v:+.3f}  " if not np.isnan(v) else f"   NaN    "
        print(row)

print("\nΔ (B − A) matrix:")
header = f"{'Metric':<38s}" + "".join(f"{l:>8s}" for l in param_labels_avail)
print(header)
for i, m in enumerate(available_metrics):
    ml = (METRIC_META[m]["label"] if m in METRIC_META else m).replace("\n", " ")
    row = f"{ml:<38s}"
    for j in range(n_p):
        v = delta[i, j]
        row += f"  {v:+.3f}  " if not np.isnan(v) else f"   NaN    "
    print(row)


# =======================================================================
# FIGURE: category strip | Series A | Series B | Delta | cbar | cbar
#
# One explicit GridSpec row, six axes. Both colorbars get their own fixed
# -width axis (cax=...) rather than the ax=[...] shorthand -- that
# shorthand does not cooperate with tight_layout (matplotlib warns "This
# figure includes Axes that are not compatible with tight_layout") and
# could squeeze or clip the delta colorbar's label. Using explicit width
# ratios instead of tight_layout keeps the whole row stable.
# =======================================================================
print("\nBuilding figure...")

label_a = SERIES["A"]["label"]
label_b = SERIES["B"]["label"]

panel_w   = 4.6
fig_height = max(5.0, 0.65 * n_m + 2.6)
fig_width  = 0.55 + panel_w * 3 + 0.35 + 0.35 + 1.6   # strip+3 panels+2 cbars+margins

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(
    1, 6,
    width_ratios=[0.55, panel_w, panel_w, panel_w, 0.30, 0.30],
    wspace=0.12,
)

ax_cat    = fig.add_subplot(gs[0, 0])
ax_a      = fig.add_subplot(gs[0, 1])
ax_b      = fig.add_subplot(gs[0, 2])
ax_d      = fig.add_subplot(gs[0, 3])
ax_cbar_m = fig.add_subplot(gs[0, 4])
ax_cbar_d = fig.add_subplot(gs[0, 5])

# Shared colormap for the two primary panels
cmap_main = plt.get_cmap("RdBu_r")
norm_main = mcolors.Normalize(vmin=-1.0, vmax=1.0)

# Diverging colormap for the delta panel — centered at 0
delta_max  = max(0.1, np.nanmax(np.abs(delta)))   # never let vmin=vmax
cmap_delta = plt.get_cmap("PiYG")
norm_delta = mcolors.TwoSlopeNorm(vmin=-delta_max, vcenter=0.0, vmax=delta_max)

draw_heatmap(ax_a, pearson["A"], available_metrics, param_labels_avail,
             label_a, cmap_main, norm_main, show_ylabel=True)

im_b = draw_heatmap(ax_b, pearson["B"], available_metrics, param_labels_avail,
             label_b, cmap_main, norm_main, show_ylabel=False)

im_d = draw_heatmap(ax_d, delta, available_metrics, param_labels_avail,
             f"\u0394 ({label_b} \u2212 {label_a})",
             cmap_delta, norm_delta, show_ylabel=False)

# Category strip shares panel A's y-limits exactly
draw_category_strip(ax_cat, available_metrics, ax_a.get_ylim())

# Dedicated colorbar axes -- fixed width, can't overlap or get squeezed
cbar_main = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap_main, norm=norm_main), cax=ax_cbar_m)
cbar_main.set_label("Pearson r", fontsize=10)

cbar_delta = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap_delta, norm=norm_delta), cax=ax_cbar_d)
cbar_delta.set_label("\u0394 Pearson r", fontsize=10)
cbar_delta.ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

# Sample sizes in title
n_a = len(dfs["A"])
n_b = len(dfs["B"])
fig.suptitle(
    f"Pearson r matrix comparison — {label_a} (n={n_a}) vs {label_b} (n={n_b})\n"
    f"Rows = metrics  |  Cols = swept parameters  |  "
    f"Right panel = shift in parameter–metric sensitivity structure",
    fontsize=11, y=1.04)

out_fname = (
    f"fig_pearson_comparison_"
    f"{label_a.replace(' ', '').replace('/', '_')}_vs_"
    f"{label_b.replace(' ', '').replace('/', '_')}.png"
)
out_path = plot_dir / out_fname
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to:\n  {out_path}")
plt.close(fig)


# =======================================================================
# SAVE DELTA TABLE TO CSV
# =======================================================================
rows = []
for i, m in enumerate(available_metrics):
    for j, p in enumerate(available_params):
        rows.append({
            "metric":       m,
            "metric_label": (METRIC_META[m]["label"] if m in METRIC_META else m).replace("\n", " "),
            "phase":        METRIC_META[m]["phase"] if m in METRIC_META else "",
            "parameter":    p,
            "param_label":  PARAM_META[p]["label"] if p in PARAM_META else p,
            f"pearson_r_{label_a}": pearson["A"][i, j],
            f"pearson_r_{label_b}": pearson["B"][i, j],
            "delta_B_minus_A":      delta[i, j],
        })
csv_fname = out_fname.replace(".png", ".csv")
csv_path  = plot_dir / csv_fname
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"Delta table saved to:\n  {csv_path}\n")
