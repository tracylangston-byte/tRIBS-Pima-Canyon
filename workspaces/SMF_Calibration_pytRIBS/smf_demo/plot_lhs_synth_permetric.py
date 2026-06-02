"""
plot_lhs_synth_permetric.py
============================
Phase 3 per-metric sensitivity analysis for the Series 91 synthetic
inversion LHS sweep.

For each of the 9 hydrograph metrics scored by run_sensitivity_single.py,
this script computes both Pearson r and Spearman ρ between every swept
parameter and that metric, then produces:

    fig10_permetric_correlation_bars.png
        Nine side-by-side bar clusters — one per metric — showing
        Pearson r (solid) and Spearman ρ (hatched) for each parameter.
        Bars are colored by parameter family.

    fig11_param_metric_heatmap.png
        Full parameter × metric correlation heatmap (Pearson r),
        annotated with numeric values. True-value proximity is also
        encoded as a dot-size overlay on a companion Spearman panel.

    fig12_metric_vs_param_grids/
        One 2×2 scatter grid per metric: each panel shows that metric
        as a function of one swept parameter, with the true value marked
        as a red dashed line and a rolling-median smoothing curve.
        Only generated for metrics with at least one |Spearman ρ| > 0.10.

    fig13_top_bottom_comparison.png
        For each metric, a horizontal stripplot comparing the parameter
        distributions of the top-20 and bottom-20 runs.  Reveals whether
        the best and worst performers are distinguishable in parameter
        space — the key identifiability test.

    fig14_metric_correlation_matrix.png
        Spearman ρ heatmap between the nine metrics themselves, to reveal
        which metrics carry redundant information and which are orthogonal.

    console output (and per_metric_correlations_91.csv):
        Full correlation table: one row per metric, one column per
        parameter, values = Pearson r  |  Spearman ρ.

Usage (run from the smf_demo directory):
    python plot_lhs_synth_permetric.py

Input:
    calibration_work/03_comparisons/summary_tables/lhs_results_synth_4param_91.csv

Output directory:
    calibration_work/03_comparisons/sensitivity_plots/Series91_SynthInversion/

Requires:
    scipy  (for Spearman correlation)
    matplotlib, numpy, pandas
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =======================================================================
# CONFIG
# =======================================================================
RESULTS_CSV  = "lhs_results_synth_4param_91.csv"
SERIES_LABEL = "Series91_SynthInversion"

KGE_CEILING = 0.912     # truth self-score ceiling

TRUE_VALUES = {
    "Ks_mult":          8.50,
    "kinemvelcoef":     4.50,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}

# Swept parameter metadata
PARAMS = {
    "Ks_mult":          {"label": "Ks mult",    "color": "#2a9d8f", "family": "Soil"},
    "kinemvelcoef":     {"label": "cv",          "color": "#e76f51", "family": "Routing"},
    "flowexp":          {"label": "r",           "color": "#e9c46a", "family": "Routing"},
    "channelroughness": {"label": "n (channel)", "color": "#457b9d", "family": "Routing"},
}
PARAM_KEYS   = list(PARAMS.keys())
PARAM_LABELS = [PARAMS[k]["label"] for k in PARAM_KEYS]

# Metric metadata: column name → display properties
# ideal_val: what "perfect" looks like (for axis reference lines)
# direction: +1 = higher is better (e.g. KGE), -1 = lower absolute is better (errors)
# phase: for grouping in the heatmap
METRICS = {
    # --- pre-peak ---
    "first_arrival_error_min": {
        "label":     "First arrival\nerror (min)",
        "ideal":     0.0,
        "direction": -1,   # zero error is best
        "phase":     "pre-peak",
        "phase_color": "#5b9bd5",
    },
    "rising_limb_steepness_ratio": {
        "label":     "Rising limb\nsteepness ratio",
        "ideal":     1.0,
        "direction": -1,   # ratio of 1.0 is best
        "phase":     "pre-peak",
        "phase_color": "#5b9bd5",
    },
    "time_to_peak_from_exc_min": {
        "label":     "Time-to-peak\nfrom exc. (min)",
        "ideal":     0.0,
        "direction": -1,
        "phase":     "pre-peak",
        "phase_color": "#5b9bd5",
    },
    # --- peak ---
    "peak_error_pct": {
        "label":     "Peak discharge\nerror (%)",
        "ideal":     0.0,
        "direction": -1,
        "phase":     "peak",
        "phase_color": "#f4a261",
    },
    "peak_timing_error_hr": {
        "label":     "Peak timing\nerror (hr)",
        "ideal":     0.0,
        "direction": -1,
        "phase":     "peak",
        "phase_color": "#f4a261",
    },
    # --- volume ---
    "pbias_pct": {
        "label":     "Volume bias\n(PBIAS %)",
        "ideal":     0.0,
        "direction": -1,
        "phase":     "volume",
        "phase_color": "#2a9d8f",
    },
    "duration_above_thresh_error_min": {
        "label":     "Duration above\nthreshold error (min)",
        "ideal":     0.0,
        "direction": -1,
        "phase":     "volume",
        "phase_color": "#2a9d8f",
    },
    # --- recession ---
    "recession_rate_ratio": {
        "label":     "Recession rate\nratio",
        "ideal":     1.0,
        "direction": -1,
        "phase":     "recession",
        "phase_color": "#8ecae6",
    },
    # --- summary ---
    "kge": {
        "label":     "KGE\n(summary)",
        "ideal":     KGE_CEILING,
        "direction": +1,  # higher is better
        "phase":     "summary",
        "phase_color": "#6a4c93",
    },
}
METRIC_KEYS = list(METRICS.keys())


# =======================================================================
# PATHS
# =======================================================================
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / SERIES_LABEL
grid_dir     = plot_dir / "fig12_metric_vs_param_grids"
plot_dir.mkdir(parents=True, exist_ok=True)
grid_dir.mkdir(parents=True, exist_ok=True)


# =======================================================================
# LOAD DATA
# =======================================================================
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(
        f"LHS results not found: {results_path}\n"
        f"Run run_lhs_synth_4param.py first."
    )

df = pd.read_csv(results_path)
print(f"\nLoaded {len(df)} LHS runs from {results_path.name}")

# Check which metric columns actually exist
available_metrics = [m for m in METRIC_KEYS if m in df.columns]
missing_metrics   = [m for m in METRIC_KEYS if m not in df.columns]
if missing_metrics:
    print(f"  WARNING: these metrics are absent from the CSV and will be skipped:")
    for m in missing_metrics:
        print(f"    {m}")
if not available_metrics:
    raise ValueError("No metric columns found in the results CSV. "
                     "Ensure run_sensitivity_single.py computed phase metrics.")

# Drop rows where ALL parameters are NaN
df = df.dropna(subset=PARAM_KEYS, how="all").reset_index(drop=True)
print(f"  {len(df)} runs after dropping parameter-NaN rows\n")

# Parameter arrays
param_arrays = {k: df[k].values for k in PARAM_KEYS}

# Metric arrays — keep NaN, handle per-metric
metric_arrays = {}
for m in available_metrics:
    metric_arrays[m] = df[m].values


def save_fig(fig, filename):
    path = plot_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


# =======================================================================
# CORRELATION COMPUTATION
# =======================================================================
print("Computing Pearson r and Spearman ρ for all parameter × metric pairs...")

# corr_df: rows = metrics, columns = params, values = (pearson_r, spearman_rho)
pearson_table  = {}
spearman_table = {}

for m in available_metrics:
    y_raw = metric_arrays[m]
    pearson_row  = {}
    spearman_row = {}
    for p in PARAM_KEYS:
        x = param_arrays[p]
        # Mask NaN in metric
        mask = ~np.isnan(y_raw) & ~np.isnan(x)
        if mask.sum() < 10:
            pearson_row[p]  = np.nan
            spearman_row[p] = np.nan
            continue
        xv = x[mask]
        yv = y_raw[mask]
        try:
            pr, _  = pearsonr(xv, yv)
        except Exception:
            pr = np.nan
        try:
            sr, _  = spearmanr(xv, yv)
        except Exception:
            sr = np.nan
        pearson_row[p]  = pr
        spearman_row[p] = sr

    pearson_table[m]  = pearson_row
    spearman_table[m] = spearman_row

pearson_df  = pd.DataFrame(pearson_table).T   # shape: metrics × params
spearman_df = pd.DataFrame(spearman_table).T

# Add metric labels for display
pearson_df.index.name  = "metric"
spearman_df.index.name = "metric"

# Print full table
print("\n" + "="*70)
print("PEARSON r  (parameter vs metric)")
print("="*70)
header = f"{'Metric':<38s}" + "".join(f"{PARAMS[p]['label']:>10s}" for p in PARAM_KEYS)
print(header)
print("-"*70)
for m in available_metrics:
    row_str = f"{METRICS[m]['label'].replace(chr(10), ' '):<38s}"
    for p in PARAM_KEYS:
        v = pearson_table[m].get(p, np.nan)
        row_str += f"  {v:+6.3f}  " if not np.isnan(v) else f"   {'NaN':>6s}  "
    print(row_str)

print("\n" + "="*70)
print("SPEARMAN ρ  (parameter vs metric)")
print("="*70)
print(header)
print("-"*70)
for m in available_metrics:
    row_str = f"{METRICS[m]['label'].replace(chr(10), ' '):<38s}"
    for p in PARAM_KEYS:
        v = spearman_table[m].get(p, np.nan)
        row_str += f"  {v:+6.3f}  " if not np.isnan(v) else f"   {'NaN':>6s}  "
    print(row_str)
print("="*70 + "\n")

# Save correlation tables to CSV
combined_rows = []
for m in available_metrics:
    for p in PARAM_KEYS:
        combined_rows.append({
            "metric":       m,
            "metric_label": METRICS[m]["label"].replace("\n", " "),
            "phase":        METRICS[m]["phase"],
            "parameter":    p,
            "param_label":  PARAMS[p]["label"],
            "pearson_r":    pearson_table[m].get(p, np.nan),
            "spearman_rho": spearman_table[m].get(p, np.nan),
        })
corr_csv_path = summary_dir / "per_metric_correlations_91.csv"
pd.DataFrame(combined_rows).to_csv(corr_csv_path, index=False)
print(f"Correlation table saved to: {corr_csv_path.name}\n")


# =======================================================================
# FIG 10: Per-metric correlation bar chart
# Nine panels (one per metric), each showing Pearson r and Spearman ρ
# for all four parameters side by side.
# =======================================================================
print("Figure 10: Per-metric correlation bars...")

n_metrics = len(available_metrics)
n_cols    = 3
n_rows    = int(np.ceil(n_metrics / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows),
                          sharey=False)
axes_flat = axes.flatten() if n_metrics > 1 else [axes]

x_pos    = np.arange(len(PARAM_KEYS))
bar_w    = 0.35
colors   = [PARAMS[p]["color"] for p in PARAM_KEYS]

for ax_idx, m in enumerate(available_metrics):
    ax = axes_flat[ax_idx]

    pr_vals = [pearson_table[m].get(p, np.nan)  for p in PARAM_KEYS]
    sr_vals = [spearman_table[m].get(p, np.nan) for p in PARAM_KEYS]

    bars_p = ax.bar(x_pos - bar_w/2, pr_vals, width=bar_w,
                    color=colors, edgecolor="white", linewidth=0.7,
                    label="Pearson r", alpha=0.9)
    bars_s = ax.bar(x_pos + bar_w/2, sr_vals, width=bar_w,
                    color=colors, edgecolor="white", linewidth=0.7,
                    label="Spearman ρ", alpha=0.55, hatch="///")

    # Value labels on bars
    for bar in list(bars_p) + list(bars_s):
        h = bar.get_height()
        if np.isnan(h):
            continue
        va = "bottom" if h >= 0 else "top"
        offset = 0.02 if h >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2,
                h + offset, f"{h:+.2f}",
                ha="center", va=va, fontsize=7, rotation=90,
                color="black")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline( 0.3, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.axhline(-0.3, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)

    phase_col = METRICS[m]["phase_color"]
    ax.set_facecolor(phase_col + "22")   # subtle phase tint
    ax.set_title(METRICS[m]["label"], fontsize=9, fontweight="bold",
                 color="#333333", pad=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PARAM_LABELS, fontsize=8)
    ax.set_ylabel("Correlation", fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="y", labelsize=8)

    # Phase label in corner
    phase_label = METRICS[m]["phase"].upper()
    ax.text(0.98, 0.97, phase_label, transform=ax.transAxes,
            fontsize=7, color=METRICS[m]["phase_color"],
            ha="right", va="top", fontweight="bold", alpha=0.8)

# Legend (one shared legend on the first panel)
legend_patches = [
    mpatches.Patch(facecolor="gray", edgecolor="white", label="Pearson r (solid)"),
    mpatches.Patch(facecolor="gray", edgecolor="white", hatch="///", alpha=0.55,
                   label="Spearman ρ (hatched)"),
]
axes_flat[0].legend(handles=legend_patches, fontsize=8, loc="lower right",
                    facecolor="white", framealpha=0.9)

# Hide unused panels
for ax_idx in range(n_metrics, len(axes_flat)):
    axes_flat[ax_idx].axis("off")

fig.suptitle(
    f"Parameter–metric correlations — Series 91  |  Synthetic Inversion  (n={len(df)})\n"
    f"Solid bars = Pearson r  |  Hatched bars = Spearman ρ  |  "
    f"Gray dotted lines = |r| = 0.3 threshold",
    fontsize=11, y=1.01)
fig.tight_layout()
save_fig(fig, "fig10_permetric_correlation_bars.png")


# =======================================================================
# FIG 11: Correlation heatmap (Pearson and Spearman side by side)
# =======================================================================
print("Figure 11: Correlation heatmaps...")

# Build display arrays: rows = metrics, cols = params
pearson_arr  = np.array([[pearson_table[m].get(p, np.nan)  for p in PARAM_KEYS]
                          for m in available_metrics])
spearman_arr = np.array([[spearman_table[m].get(p, np.nan) for p in PARAM_KEYS]
                          for m in available_metrics])

metric_display_labels = [METRICS[m]["label"] for m in available_metrics]
phase_colors_ordered  = [METRICS[m]["phase_color"] for m in available_metrics]

fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(14, 0.65 * n_metrics + 2.5),
                                   sharey=True)

cmap = plt.get_cmap("RdBu_r")
norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

for ax, arr, title in [
    (ax_p, pearson_arr,  "Pearson r"),
    (ax_s, spearman_arr, "Spearman ρ"),
]:
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")

    # Annotate each cell
    for i in range(len(available_metrics)):
        for j in range(len(PARAM_KEYS)):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "NaN", ha="center", va="center",
                        fontsize=8, color="gray")
            else:
                text_col = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=text_col)

    ax.set_xticks(range(len(PARAM_KEYS)))
    ax.set_xticklabels(PARAM_LABELS, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Parameter", fontsize=10)

    # Colored row labels by phase
    ax.set_yticks(range(len(available_metrics)))
    ax.set_yticklabels(metric_display_labels, fontsize=8.5)
    for i, (tick, col) in enumerate(zip(ax.get_yticklabels(), phase_colors_ordered)):
        tick.set_color(col)

    # Horizontal separators between phases
    current_phase = None
    for i, m in enumerate(available_metrics):
        if METRICS[m]["phase"] != current_phase:
            if i > 0:
                ax.axhline(i - 0.5, color="black", linewidth=1.5, alpha=0.7)
            current_phase = METRICS[m]["phase"]

    ax.tick_params(left=False)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_p, ax_s], fraction=0.025, pad=0.03)
cbar.set_label("Correlation coefficient", fontsize=10)

# Phase legend
phase_handles = {}
for m in available_metrics:
    ph = METRICS[m]["phase"]
    if ph not in phase_handles:
        phase_handles[ph] = mpatches.Patch(
            facecolor=METRICS[m]["phase_color"], label=ph.capitalize())
fig.legend(handles=list(phase_handles.values()), fontsize=9, loc="lower center",
           ncol=len(phase_handles), bbox_to_anchor=(0.5, -0.04),
           facecolor="white", framealpha=0.9)

fig.suptitle(
    f"Parameter × metric correlation heatmap — Series 91  |  Synthetic Inversion  "
    f"(n={len(df)})\n"
    f"True values: Ks=8.5x  cv=4.5  r=0.24  n=0.026  |  f fixed at 0.020 mm⁻¹",
    fontsize=11, y=1.02)
fig.tight_layout()
save_fig(fig, "fig11_param_metric_heatmap.png")


# =======================================================================
# FIG 12: Metric vs parameter scatter grids (one figure per metric)
# Produced only for metrics with at least one |Spearman ρ| > 0.10.
# =======================================================================
print("Figure 12: Per-metric × parameter scatter grids...")

for m in available_metrics:
    # Check if this metric has any meaningful signal
    sr_vals = [abs(spearman_table[m].get(p, 0.0) or 0.0) for p in PARAM_KEYS]
    if max(sr_vals) < 0.10:
        print(f"  Skipping {m} (max |Spearman ρ| = {max(sr_vals):.3f} < 0.10)")
        continue

    y_raw = metric_arrays[m]
    ideal = METRICS[m]["ideal"]
    phase = METRICS[m]["phase"]
    mlabel = METRICS[m]["label"].replace("\n", " ")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, p in zip(axes.flatten(), PARAM_KEYS):
        x = param_arrays[p]
        mask = ~np.isnan(y_raw) & ~np.isnan(x)
        xv = x[mask]
        yv = y_raw[mask]

        pr = pearson_table[m].get(p, np.nan)
        sr = spearman_table[m].get(p, np.nan)

        # Scatter, colored by KGE
        kge_v = df["kge"].values[mask]
        kge_norm = mcolors.Normalize(
            vmin=np.nanpercentile(kge_v, 5),
            vmax=np.nanpercentile(kge_v, 95), clip=True)

        sc = ax.scatter(xv, yv, c=kge_v, cmap="plasma", norm=kge_norm,
                        s=30, edgecolors="white", linewidths=0.4, alpha=0.8, zorder=3)

        # Rolling median smooth
        if len(xv) >= 10:
            sort_order = np.argsort(xv)
            xs_s  = xv[sort_order]
            ys_s  = yv[sort_order]
            window = max(5, len(xv) // 8)
            ys_sm  = (pd.Series(ys_s)
                      .rolling(window, center=True, min_periods=1)
                      .median().values)
            ax.plot(xs_s, ys_sm, color="navy", linewidth=1.8,
                    linestyle="--", alpha=0.75, label="Rolling median", zorder=4)

        # True value vertical line
        tv = TRUE_VALUES[p]
        ax.axvline(tv, color="red", linewidth=1.8, linestyle="--", alpha=0.85,
                   label=f"True = {tv}", zorder=5)

        # Ideal horizontal line
        ax.axhline(ideal, color="green", linewidth=1.2, linestyle="-.",
                   alpha=0.7, label=f"Ideal = {ideal}", zorder=5)

        plab = PARAMS[p]["label"]
        corr_str = (f"r={pr:+.3f}" if not np.isnan(pr) else "r=NaN") + "  "
        corr_str += (f"ρ={sr:+.3f}" if not np.isnan(sr) else "ρ=NaN")
        ax.set_title(f"{plab}  |  {corr_str}", fontsize=9, fontweight="bold")
        ax.set_xlabel(PARAMS[p]["label"], fontsize=9)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.legend(fontsize=7, loc="best", facecolor="white", framealpha=0.85)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)

        fig.colorbar(sc, ax=ax, label="KGE", fraction=0.04, pad=0.03)

    fig.suptitle(
        f"{mlabel}  vs each swept parameter\n"
        f"Series 91  |  Synthetic Inversion  |  Phase: {phase}  (n={len(df)})\n"
        f"Red dashed = true value  |  Green dash-dot = ideal  |  Navy dashed = rolling median",
        fontsize=10, y=1.01)
    fig.tight_layout()

    safe_name = m.replace("/", "_").replace(" ", "_")
    save_path  = grid_dir / f"fig12_{safe_name}_vs_params.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path.name}")
    plt.close(fig)


# =======================================================================
# FIG 13: Top-20 vs Bottom-20 parameter stripplot
# The key identifiability test: can we distinguish high-scoring from
# low-scoring runs in parameter space?
# =======================================================================
print("Figure 13: Top-20 vs Bottom-20 parameter stripplot...")

N_STRIP   = 20
top20     = df.nlargest(N_STRIP, "kge")
bottom20  = df.nsmallest(N_STRIP, "kge")

n_metrics_strip = len(available_metrics)
n_cols_strip    = 3
n_rows_strip    = int(np.ceil(n_metrics_strip / n_cols_strip))

fig, axes = plt.subplots(n_rows_strip, n_cols_strip,
                          figsize=(14, 3.2 * n_rows_strip), sharey=False)
axes_flat = axes.flatten() if n_metrics_strip > 1 else [axes]

jitter_std = 0.08

rng = np.random.default_rng(0)

for ax_idx, m in enumerate(available_metrics):
    ax = axes_flat[ax_idx]

    for group_idx, (group_df, group_label, group_color) in enumerate([
        (top20,    f"Top {N_STRIP}",    "#2a9d8f"),
        (bottom20, f"Bottom {N_STRIP}", "#e76f51"),
    ]):
        for p_idx, p in enumerate(PARAM_KEYS):
            x_base = p_idx + group_idx * 0.25 - 0.125
            vals_g = group_df[p].values
            y_jitter = rng.normal(x_base, jitter_std * 0.4, size=len(vals_g))

            # Normalize parameter to [0,1] for display on shared axis
            lo = df[p].min()
            hi = df[p].max()
            vals_norm = (vals_g - lo) / (hi - lo) if hi > lo else vals_g * 0

            ax.scatter(y_jitter, vals_norm,
                       color=group_color, alpha=0.7, s=25,
                       edgecolors="white", linewidths=0.3,
                       label=group_label if p_idx == 0 else None,
                       zorder=3)

            # True value normalized
            tv_norm = (TRUE_VALUES[p] - lo) / (hi - lo) if hi > lo else 0.5
            ax.axhline(tv_norm, color="red", linewidth=0.9, linestyle=":",
                       alpha=0.5, zorder=2)

    mlabel = METRICS[m]["label"].replace("\n", " ")
    # subtitle: best/worst metric value
    top_metric    = top20[m].median() if m in top20.columns else np.nan
    bottom_metric = bottom20[m].median() if m in bottom20.columns else np.nan
    stat_str = ""
    if not np.isnan(top_metric) and not np.isnan(bottom_metric):
        stat_str = f"  top median={top_metric:.2f}  bot median={bottom_metric:.2f}"

    ax.set_title(mlabel + stat_str, fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(PARAM_KEYS)))
    ax.set_xticklabels(PARAM_LABELS, fontsize=7.5)
    ax.set_ylabel("Normalized param value\n(0=lo, 1=hi)", fontsize=7.5)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)
    ax.axhline(1, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=7.5)

    phase_label = METRICS[m]["phase"].upper()
    ax.text(0.99, 0.99, phase_label, transform=ax.transAxes,
            fontsize=7, color=METRICS[m]["phase_color"],
            ha="right", va="top", fontweight="bold")

# Shared legend
handles = [
    mpatches.Patch(facecolor="#2a9d8f", label=f"Top {N_STRIP} runs by KGE"),
    mpatches.Patch(facecolor="#e76f51", label=f"Bottom {N_STRIP} runs by KGE"),
    plt.Line2D([0], [0], color="red", linestyle=":", linewidth=1.2,
               label="True value (normalized)"),
]
fig.legend(handles=handles, fontsize=9, loc="upper center",
           ncol=3, bbox_to_anchor=(0.5, 1.0), facecolor="white", framealpha=0.9)

# Hide unused
for ax_idx in range(n_metrics_strip, len(axes_flat)):
    axes_flat[ax_idx].axis("off")

fig.suptitle(
    f"Top {N_STRIP} vs Bottom {N_STRIP} parameter distributions — "
    f"Series 91  |  Synthetic Inversion  (n={len(df)})\n"
    f"Convergence near true value (red dots) = parameter is identifiable",
    fontsize=11, y=1.04)
fig.tight_layout()
save_fig(fig, "fig13_top_bottom_comparison.png")


# =======================================================================
# FIG 14: Metric–metric Spearman ρ heatmap
# Which metrics carry redundant vs. orthogonal information?
# =======================================================================
print("Figure 14: Metric–metric correlation heatmap...")

n_m = len(available_metrics)
mm_corr = np.full((n_m, n_m), np.nan)

for i, m1 in enumerate(available_metrics):
    for j, m2 in enumerate(available_metrics):
        y1 = metric_arrays[m1]
        y2 = metric_arrays[m2]
        mask = ~np.isnan(y1) & ~np.isnan(y2)
        if mask.sum() < 10:
            continue
        try:
            sr, _ = spearmanr(y1[mask], y2[mask])
            mm_corr[i, j] = sr
        except Exception:
            pass

fig, ax = plt.subplots(figsize=(9, 7.5))
cmap2 = plt.get_cmap("RdBu_r")
norm2 = mcolors.Normalize(vmin=-1, vmax=1)

im = ax.imshow(mm_corr, cmap=cmap2, norm=norm2, aspect="auto")

for i in range(n_m):
    for j in range(n_m):
        v = mm_corr[i, j]
        if np.isnan(v):
            ax.text(j, i, "NaN", ha="center", va="center", fontsize=7.5, color="gray")
        else:
            text_col = "white" if abs(v) > 0.65 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_col)

short_labels = [METRICS[m]["label"].replace("\n", "\n") for m in available_metrics]
ax.set_xticks(range(n_m))
ax.set_xticklabels(short_labels, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(n_m))
ax.set_yticklabels(short_labels, fontsize=8)

# Color x-tick labels by phase
for i, m in enumerate(available_metrics):
    ax.get_xticklabels()[i].set_color(METRICS[m]["phase_color"])
    ax.get_yticklabels()[i].set_color(METRICS[m]["phase_color"])

# Phase separators
current_phase = None
for i, m in enumerate(available_metrics):
    if METRICS[m]["phase"] != current_phase:
        if i > 0:
            ax.axhline(i - 0.5, color="black", linewidth=1.5)
            ax.axvline(i - 0.5, color="black", linewidth=1.5)
        current_phase = METRICS[m]["phase"]

cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
cbar.set_label("Spearman ρ", fontsize=10)

ax.set_title(
    f"Metric–metric Spearman ρ — Series 91  |  Synthetic Inversion  (n={len(df)})\n"
    f"High |ρ| = redundant metrics  |  Low |ρ| = orthogonal information",
    fontsize=11, pad=10)
fig.tight_layout()
save_fig(fig, "fig14_metric_correlation_matrix.png")


# =======================================================================
# CONSOLE SUMMARY
# =======================================================================
print(f"\n{'='*65}")
print("PHASE 3 SUMMARY  —  Series 91 Per-Metric Sensitivity")
print(f"{'='*65}")

print("\nStrongest parameter–metric signals (|Spearman ρ| > 0.20):")
printed_any = False
for m in available_metrics:
    for p in PARAM_KEYS:
        sr = spearman_table[m].get(p, np.nan)
        if not np.isnan(sr) and abs(sr) > 0.20:
            mlabel = METRICS[m]["label"].replace("\n", " ")
            plabel = PARAMS[p]["label"]
            print(f"  {mlabel:<40s}  ←  {plabel:<15s}  ρ = {sr:+.3f}")
            printed_any = True
if not printed_any:
    print("  None exceeded the 0.20 threshold — "
          "equifinality is high or sample size too small.")

print(f"\nFigures saved to:\n  {plot_dir}")
print(f"\nCorrelation CSV:\n  {corr_csv_path}")
print(f"{'='*65}\n")
