"""
plot_pearson_comparison.py
==========================
Compare the Pearson r (parameter × metric) matrix between two LHS sweep
series using side-by-side heatmaps with a difference panel.

Output:
    /workspaces/tRIBS-Pima-Canyon/workspaces/SMF_Calibration_pytRIBS/calibration_work/03_comparisons/sensitivity_plots/Comparisons
    fig_pearson_comparison_{SERIES_A_LABEL}_vs_{SERIES_B_LABEL}.png

Usage (run from smf_demo):
    python plot_pearson_comparison.py

To adapt for a different pair of runs, edit the CONFIG block only.
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
        "label":       "Series 96 - Anchor A",         # display name in figure title
        "csv":         "lhs_results_anchor_anchorA_96.csv",
        "true_values": {
            "kinemvelcoef":     4.50,
            "flowexp":          0.24,
            "channelroughness": 0.026,
        },
    },
    "B": {
        "label":       "Series 96 - Anchor B",
        "csv":         "lhs_results_anchor_anchorB_96.csv",
        "true_values": {
            "kinemvelcoef":     4.50,
            "flowexp":          0.24,
            "channelroughness": 0.026,
        },
    },
    # To add Series 93 once it exists, change one entry above to:
    # "B": {
    #     "label":       "Series 93",
    #     "csv":         "lhs_results_synth_4param_93.csv",
    #     "true_values": { ... },
    # },
}

# Parameters to include (must be columns in both CSVs).
# Order here controls column order in the heatmap.
PARAM_KEYS = [
    "kinemvelcoef",
    "flowexp",
    "channelroughness",
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
    "kinemvelcoef":     {"label": PARAM_KEY["kinemvelcoef"]["symbol"],     "color": "#e76f51", "family": "Routing"},
    "flowexp":          {"label": PARAM_KEY["flowexp"]["symbol"],           "color": "#e9c46a", "family": "Routing"},
    "channelroughness": {"label": PARAM_KEY["channelroughness"]["symbol"], "color": "#457b9d", "family": "Routing"},
}

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
            f"Run run_lhs_anchor_cvrn.py first."
        )
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path.name}")
    return df


def compute_pearson_matrix(df: pd.DataFrame,
                            param_keys: list,
                            metric_keys: list) -> np.ndarray:
    """Return (n_metrics × n_params) array of Pearson r values."""
    n_m = len(metric_keys)
    n_p = len(param_keys)
    arr = np.full((n_m, n_p), np.nan)
    for i, m in enumerate(metric_keys):
        if m not in df.columns:
            continue
        for j, p in enumerate(param_keys):
            if p not in df.columns:
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


def phase_separators(ax, metric_keys: list):
    """Draw horizontal lines between metric phase groups."""
    current_phase = None
    for i, m in enumerate(metric_keys):
        if m not in METRIC_META:
            continue
        phase = METRIC_META[m]["phase"]
        if phase != current_phase and i > 0:
            ax.axhline(i - 0.5, color="black", linewidth=1.5, alpha=0.7)
        current_phase = phase


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
        phase_colors_ordered = [
            METRIC_META[m]["phase_color"] if m in METRIC_META else "black"
            for m in metric_keys
        ]
        ax.set_yticks(range(len(metric_keys)))
        ax.set_yticklabels(metric_display_labels, fontsize=8.5)
        for tick, col in zip(ax.get_yticklabels(), phase_colors_ordered):
            tick.set_color(col)
    else:
        ax.set_yticks(range(len(metric_keys)))
        ax.set_yticklabels([], fontsize=8.5)

    phase_separators(ax, metric_keys)
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
    pearson[key] = compute_pearson_matrix(dfs[key], available_params, available_metrics)
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
# FIGURE: 3-panel heatmap  [Series A | Series B | Δ = B − A]
# =======================================================================
print("\nBuilding figure...")

label_a = SERIES["A"]["label"]
label_b = SERIES["B"]["label"]

fig_width  = 5.5 * 3 + 1.5          # 3 panels + colorbar space
fig_height = max(5.0, 0.65 * n_m + 2.5)
fig, axes  = plt.subplots(1, 3, figsize=(fig_width, fig_height),
                           gridspec_kw={"wspace": 0.08})

ax_a, ax_b, ax_d = axes

# Shared colormap for the two primary panels
cmap_main = plt.get_cmap("RdBu_r")
norm_main = mcolors.Normalize(vmin=-1.0, vmax=1.0)

# Diverging colormap for the delta panel — centered at 0
delta_max  = max(0.1, np.nanmax(np.abs(delta)))   # never let vmin=vmax
cmap_delta = plt.get_cmap("PiYG")
norm_delta = mcolors.TwoSlopeNorm(vmin=-delta_max, vcenter=0.0, vmax=delta_max)

draw_heatmap(ax_a, pearson["A"], available_metrics, param_labels_avail,
             label_a, cmap_main, norm_main, show_ylabel=True)

draw_heatmap(ax_b, pearson["B"], available_metrics, param_labels_avail,
             label_b, cmap_main, norm_main, show_ylabel=False)

draw_heatmap(ax_d, delta, available_metrics, param_labels_avail,
             f"Δ ({label_b} − {label_a})",
             cmap_delta, norm_delta, show_ylabel=False)

# Colorbar for primary panels (shared)
sm_main = plt.cm.ScalarMappable(cmap=cmap_main, norm=norm_main)
sm_main.set_array([])
cbar_main = fig.colorbar(sm_main, ax=[ax_a, ax_b],
                          fraction=0.022, pad=0.02, shrink=0.85)
cbar_main.set_label("Pearson r", fontsize=10)

# Colorbar for delta panel
sm_delta = plt.cm.ScalarMappable(cmap=cmap_delta, norm=norm_delta)
sm_delta.set_array([])
cbar_delta = fig.colorbar(sm_delta, ax=ax_d,
                           fraction=0.035, pad=0.03, shrink=0.85)
cbar_delta.set_label("Δ Pearson r", fontsize=10)
cbar_delta.ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

# Phase legend
phase_handles = {}
for m in available_metrics:
    if m not in METRIC_META:
        continue
    ph = METRIC_META[m]["phase"]
    if ph not in phase_handles:
        phase_handles[ph] = mpatches.Patch(
            facecolor=METRIC_META[m]["phase_color"],
            label=ph.capitalize())
fig.legend(handles=list(phase_handles.values()), fontsize=9,
           loc="lower center", ncol=len(phase_handles),
           bbox_to_anchor=(0.5, -0.04),
           facecolor="white", framealpha=0.9)

# Sample sizes in title
n_a = len(dfs["A"])
n_b = len(dfs["B"])
fig.suptitle(
    f"Pearson r matrix comparison — {label_a} (n={n_a}) vs {label_b} (n={n_b})\n"
    f"Rows = metrics  |  Cols = swept parameters  |  "
    f"Right panel = shift in parameter–metric sensitivity structure",
    fontsize=11, y=1.02)

fig.tight_layout()

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