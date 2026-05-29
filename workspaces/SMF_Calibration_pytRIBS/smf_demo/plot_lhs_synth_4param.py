"""
plot_lhs_5param.py
==================
Generates all diagnostic figures from a 5-parameter LHS sweep.
Easily switchable between series 80 (KsLo) and series 81 (KsHi)
by editing the two lines in the CONFIG block below.

Usage (run from the smf_demo directory):
    python plot_lhs_5param.py

Produces 8 figures saved to:
    calibration_work/03_comparisons/sensitivity_plots/{SERIES_LABEL}/

Figure list
-----------
    fig1_hydrograph_envelope_all.png   — All runs: envelope + median + best run
    fig2_hydrograph_envelope_kge0.png  — KGE > 0 runs only + filtered median
    fig3_correlation_bar.png           — Pearson r of each swept param vs KGE
    fig4_parallel_coordinates.png      — All 4 params as vertical axes, lines colored by KGE
    fig5_pairwise_scatter.png          — 6-panel pairwise scatter matrix, colored by KGE
    fig6_kge_vs_each_param.png         — 4-panel KGE vs each parameter (1D response curves)
    fig7_pbias_vs_kge.png              — PBIAS vs KGE scatter
    fig8_top15_table.png               — Top-15 runs by KGE as formatted table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from itertools import combinations
from pathlib import Path

# =======================================================================
# CONFIG — edit these two lines to switch between series
# =======================================================================
RESULTS_CSV  = "lhs_results_synth_4param_90.csv"
SERIES_LABEL = "Series90_SynthInversion"
# =======================================================================

EVENT_LABEL      = "SMF Aug 12, 2014"
EVENT_CROP_START = "2014-08-12 17:30"
EVENT_CROP_END   = "2014-08-12 21:00"

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
csv_dir      = calib_dir / "03_comparisons" / "csv_exports"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / SERIES_LABEL
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(
        f"LHS results not found: {results_path}\n"
        f"Run the appropriate run_lhs_5param.py first."
    )

df = pd.read_csv(results_path)
print(f"Loaded {len(df)} LHS runs from {results_path.name}")

required_cols = ["run_id", "Ks_mult", "kinemvelcoef", "flowexp", "channelroughness",
                 "kge", "nse", "pbias_pct", "kge_r", "kge_alpha", "kge_beta"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results CSV: {missing}\n"
                     f"Available: {list(df.columns)}")

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"  {len(df)} runs after dropping NaN rows")

# SERIES_TITLE built from actual data — Ks range is always accurate
ks_lo = df["Ks_mult"].min()
ks_hi = df["Ks_mult"].max()
SERIES_TITLE = f"Series 90  |  Synthetic Inversion  |  Ks {ks_lo:.1f}-{ks_hi:.1f}x"

print(f"Series: {SERIES_TITLE}")
print(f"Plots -> {plot_dir}")

# Convenience arrays
ks_vals    = df["Ks_mult"].values
cv_vals    = df["kinemvelcoef"].values
r_vals     = df["flowexp"].values
n_vals     = df["channelroughness"].values
kge_vals   = df["kge"].values
nse_vals   = df["nse"].values
pbias_vals = df["pbias_pct"].values

best_idx    = int(np.argmax(kge_vals))
best_run_id = df["run_id"].iloc[best_idx]
best_kge    = kge_vals[best_idx]

print(f"\n  Best run: {best_run_id}")
print(f"  Best KGE: {best_kge:.3f}")
print(f"  KGE range:   {kge_vals.min():.3f} to {kge_vals.max():.3f}")
print(f"  PBIAS range: {pbias_vals.min():.1f}% to {pbias_vals.max():.1f}%")

# Swept parameter metadata
PARAMS = {
    "Ks_mult":          {"label": "Ks multiplier",               "vals": ks_vals, "fmt": "{:.2f}x"},
    "kinemvelcoef":     {"label": "Hillslope velocity coef. cv", "vals": cv_vals, "fmt": "{:.2f}"},
    "flowexp":          {"label": "Hillslope velocity exp. r",   "vals": r_vals,  "fmt": "{:.3f}"},
    "channelroughness": {"label": "Channel roughness n",         "vals": n_vals,  "fmt": "{:.4f}"},
}
PARAM_KEYS   = list(PARAMS.keys())
PARAM_LABELS = [PARAMS[k]["label"] for k in PARAM_KEYS]

TRUE_VALUES = {
    "Ks_mult":          8.50,
    "kinemvelcoef":     5.75,
    "flowexp":          0.23,
    "channelroughness": 0.02,
}

# KGE colormap — plasma has no white/near-white values on a white background.
# Norm uses 5th-95th percentile of actual data with clip=True so:
#   (a) colour range reflects the real spread of results
#   (b) norm_val is always in [0,1], preventing alpha ValueError
KGE_CMAP = plt.get_cmap("plasma")
kge_p05  = np.percentile(kge_vals, 5)
kge_p95  = np.percentile(kge_vals, 95)
KGE_NORM = mcolors.Normalize(vmin=kge_p05, vmax=kge_p95, clip=True)


def save_fig(fig, filename):
    path = plot_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


# -----------------------------------------------------------------------
# LOAD ALL HYDROGRAPH CSVs
# -----------------------------------------------------------------------
def load_all_hydrographs(run_ids):
    hydrographs   = {}
    missing_count = 0
    for run_id in run_ids:
        csv_path = csv_dir / f"{run_id}_compare_obs_sim.csv"
        if csv_path.exists():
            try:
                hdf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                hydrographs[run_id] = hdf
            except Exception as e:
                print(f"  Warning: could not read {csv_path.name}: {e}")
                missing_count += 1
        else:
            missing_count += 1
    if missing_count:
        print(f"  Warning: {missing_count} hydrograph CSVs not found.")
    print(f"  Loaded {len(hydrographs)} hydrograph CSVs.")
    return hydrographs


print("\nLoading hydrograph CSVs...")
all_hydros = load_all_hydrographs(df["run_id"].values)

obs_series = None
for hdf in all_hydros.values():
    if "Observed" in hdf.columns:
        obs_series = hdf["Observed"]
        break

if obs_series is None:
    print("  WARNING: No observed series found. Hydrograph figures will be skipped.")


# -----------------------------------------------------------------------
# FIGURES 1 & 2: Hydrograph uncertainty envelopes
# Cropped to event window for readability.
# -----------------------------------------------------------------------
def plot_hydrograph_envelope(hydros_subset, df_subset, title_suffix, filename,
                              envelope_color, median_color, filter_label):
    if obs_series is None or len(hydros_subset) == 0:
        print(f"  Skipping {filename} — insufficient data.")
        return

    common_idx = obs_series.index
    sim_matrix = pd.DataFrame(index=common_idx)

    best_run_in_subset = df_subset.loc[df_subset["kge"].idxmax(), "run_id"]

    for run_id, hdf in hydros_subset.items():
        sim_matrix[run_id] = hdf["Simulated"].reindex(common_idx)

    sim_matrix = sim_matrix.dropna(how="all")
    if sim_matrix.empty:
        print(f"  Skipping {filename} — sim_matrix empty after dropna.")
        return

    # Crop to event window
    sim_matrix  = sim_matrix.loc[EVENT_CROP_START:EVENT_CROP_END]
    obs_cropped = obs_series.reindex(sim_matrix.index)

    p05 = sim_matrix.quantile(0.05, axis=1)
    p25 = sim_matrix.quantile(0.25, axis=1)
    p50 = sim_matrix.quantile(0.50, axis=1)
    p75 = sim_matrix.quantile(0.75, axis=1)
    p95 = sim_matrix.quantile(0.95, axis=1)

    best_sim      = hydros_subset.get(best_run_in_subset)
    n_runs        = len(hydros_subset)
    best_kge_here = df_subset.loc[df_subset["run_id"] == best_run_in_subset, "kge"].values[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#f5f5f5")

    for run_id, hdf in hydros_subset.items():
        sim = hdf["Simulated"].reindex(sim_matrix.index)
        ax.plot(sim_matrix.index, sim, color=envelope_color, alpha=0.08,
                linewidth=0.5, zorder=1)

    ax.fill_between(sim_matrix.index, p05, p95,
                    color=envelope_color, alpha=0.25, label="5th-95th percentile", zorder=2)
    ax.fill_between(sim_matrix.index, p25, p75,
                    color=envelope_color, alpha=0.45, label="25th-75th percentile", zorder=3)
    ax.plot(sim_matrix.index, p50, color=median_color, linewidth=1.8,
            label=f"{filter_label} Median", zorder=4)

    if best_sim is not None:
        ax.plot(sim_matrix.index, best_sim["Simulated"].reindex(sim_matrix.index),
                color="crimson", linewidth=1.6, linestyle="--",
                label=f"Best run (KGE={best_kge_here:.3f})", zorder=5)

    ax.plot(sim_matrix.index, obs_cropped,
            color="black", linewidth=2.2, label="Observed", zorder=6)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Discharge (m3/s)", fontsize=11)
    ax.set_title(
        f"Hydrograph uncertainty — {title_suffix}  (n={n_runs})\n"
        f"5-parameter LHS  |  {SERIES_TITLE}  |  {EVENT_LABEL}",
        fontsize=12
    )
    ax.legend(fontsize=9, loc="upper right", facecolor="white", framealpha=0.9)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    save_fig(fig, filename)


print("\nFigure 1: Hydrograph envelope — all runs")
all_hydros_dict = {rid: all_hydros[rid] for rid in df["run_id"].values if rid in all_hydros}
plot_hydrograph_envelope(
    hydros_subset=all_hydros_dict,
    df_subset=df[df["run_id"].isin(all_hydros_dict)].copy(),
    title_suffix="All runs",
    filename="fig1_hydrograph_envelope_all.png",
    envelope_color="#4c8cbf",
    median_color="#1a5fa8",
    filter_label="All runs"
)

print("Figure 2: Hydrograph envelope — KGE > 0 runs only")
df_kge0     = df[df["kge"] > 0].copy()
hydros_kge0 = {rid: all_hydros[rid] for rid in df_kge0["run_id"].values if rid in all_hydros}
plot_hydrograph_envelope(
    hydros_subset=hydros_kge0,
    df_subset=df_kge0,
    title_suffix="KGE > 0 runs",
    filename="fig2_hydrograph_envelope_kge0.png",
    envelope_color="#e8833a",
    median_color="#b85c00",
    filter_label="KGE > 0"
)


# -----------------------------------------------------------------------
# FIGURE 3: Pearson correlation bar chart
# -----------------------------------------------------------------------
print("Figure 3: Correlation bar chart")

correlations = {key: np.corrcoef(df[key].values, kge_vals)[0, 1] for key in PARAM_KEYS}

fig, ax = plt.subplots(figsize=(8, 4.5))
bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in correlations.values()]
bars = ax.barh(PARAM_LABELS, list(correlations.values()),
               color=bar_colors, edgecolor="white", height=0.55)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson r  (parameter value vs KGE)", fontsize=11)
ax.set_title(
    f"Parameter-KGE correlations — 5-param LHS  {SERIES_TITLE}  (n={len(df)})\n"
    f"{EVENT_LABEL}  |  Green = higher value -> better KGE  |  Red = opposite",
    fontsize=11
)
for bar, val in zip(bars, correlations.values()):
    x_pos = val + 0.01 if val >= 0 else val - 0.01
    ha    = "left"      if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"r = {val:+.3f}", va="center", ha=ha, fontsize=9)
ax.set_xlim(-1.05, 1.05)
ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax.grid(axis="x", which="major", alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig3_correlation_bar.png")


# -----------------------------------------------------------------------
# FIGURE 4: Parallel coordinates
# -----------------------------------------------------------------------
print("Figure 4: Parallel coordinates")

param_arrays = np.column_stack([df[k].values for k in PARAM_KEYS])
param_mins   = param_arrays.min(axis=0)
param_maxs   = param_arrays.max(axis=0)
param_norm   = (param_arrays - param_mins) / (param_maxs - param_mins + 1e-12)

n_params    = len(PARAM_KEYS)
x_positions = np.arange(n_params)

fig, ax = plt.subplots(figsize=(12, 6))

for i in np.argsort(kge_vals):
    color    = KGE_CMAP(KGE_NORM(kge_vals[i]))
    norm_val = float(KGE_NORM(kge_vals[i]))   # clip=True guarantees [0,1]; float() avoids masked array issues
    alpha    = 0.35 + 0.50 * norm_val
    ax.plot(x_positions, param_norm[i], color=color, alpha=alpha,
            linewidth=1.4, zorder=int(norm_val * 100))

# Best run in red so it stands out against the plasma colormap
ax.plot(x_positions, param_norm[best_idx],
        color="red", linewidth=2.5, zorder=200,
        label=f"Best run  KGE={best_kge:.3f}")

for j, (key, label) in enumerate(zip(PARAM_KEYS, PARAM_LABELS)):
    ax.axvline(j, color="gray", linewidth=0.8, alpha=0.5)
    ax.text(j, -0.04, PARAMS[key]["fmt"].format(param_mins[j]),
            ha="center", va="top", fontsize=7.5, color="gray",
            transform=ax.get_xaxis_transform())
    ax.text(j,  1.04, PARAMS[key]["fmt"].format(param_maxs[j]),
            ha="center", va="bottom", fontsize=7.5, color="gray",
            transform=ax.get_xaxis_transform())

ax.set_xticks(x_positions)
ax.set_xticklabels(PARAM_LABELS, fontsize=10)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["Min", "Mid", "Max"], fontsize=9)
ax.set_ylabel("Normalised parameter value", fontsize=10)
ax.set_title(
    f"Parallel coordinates — 5-param LHS  {SERIES_TITLE}  (n={len(df)})\n"
    f"Lines colored by KGE  |  {EVENT_LABEL}  |  f fixed at 0.020 mm^-1",
    fontsize=12
)
sm = plt.cm.ScalarMappable(cmap=KGE_CMAP, norm=KGE_NORM)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="KGE", shrink=0.7, pad=0.02)
ax.legend(fontsize=9, loc="upper left", facecolor="white", framealpha=0.9)
fig.tight_layout()
save_fig(fig, "fig4_parallel_coordinates.png")


# -----------------------------------------------------------------------
# FIGURE 5: Pairwise scatter matrix
# -----------------------------------------------------------------------
print("Figure 5: Pairwise scatter matrix")

pairs  = list(combinations(range(n_params), 2))
n_cols = 3
n_rows = (len(pairs) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
axes_flat = axes.flat

for ax_i, (pi, pj) in enumerate(pairs):
    ax = axes_flat[ax_i]
    ki, kj = PARAM_KEYS[pi], PARAM_KEYS[pj]
    ax.scatter(df[ki].values, df[kj].values,
               c=kge_vals, cmap=KGE_CMAP, norm=KGE_NORM,
               s=30, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)
    ax.scatter(df[ki].iloc[best_idx], df[kj].iloc[best_idx],
               s=150, marker="*", color="white", edgecolors="black",
               linewidths=1.2, zorder=5, label=f"Best KGE={best_kge:.3f}")
    ax.set_xlabel(PARAMS[ki]["label"], fontsize=9)
    ax.set_ylabel(PARAMS[kj]["label"], fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right", facecolor="white", framealpha=0.8)
    ax.grid(alpha=0.25)

for ax_i in range(len(pairs), n_rows * n_cols):
    axes_flat[ax_i].set_visible(False)

sm = plt.cm.ScalarMappable(cmap=KGE_CMAP, norm=KGE_NORM)
sm.set_array([])
fig.colorbar(sm, ax=axes.ravel().tolist(), label="KGE", shrink=0.6, pad=0.02)
fig.suptitle(
    f"Pairwise parameter scatter — 5-param LHS  {SERIES_TITLE}  (n={len(df)})\n"
    f"Colored by KGE  |  {EVENT_LABEL}  |  star = best run",
    fontsize=13
)
fig.tight_layout()
save_fig(fig, "fig5_pairwise_scatter.png")


# -----------------------------------------------------------------------
# FIGURE 6: KGE vs each parameter — 1D response curves
# -----------------------------------------------------------------------
print("Figure 6: KGE vs each parameter (1D response)")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, key in zip(axes.flat, PARAM_KEYS):
    xi = df[key].values
    ax.scatter(xi, kge_vals, c=kge_vals, cmap=KGE_CMAP, norm=KGE_NORM,
               s=28, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)
    ax.scatter(df[key].iloc[best_idx], best_kge,
               s=160, marker="*", color="white", edgecolors="black",
               linewidths=1.2, zorder=5, label=f"Best KGE={best_kge:.3f}")
    sort_order = np.argsort(xi)
    xs_sorted  = xi[sort_order]
    ys_sorted  = kge_vals[sort_order]
    window     = max(5, len(df) // 8)
    ys_smooth  = pd.Series(ys_sorted).rolling(window, center=True,
                                               min_periods=1).median().values
    ax.plot(xs_sorted, ys_smooth, color="navy", linewidth=1.6,
            linestyle="--", alpha=0.7, label="Rolling median")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axvline(TRUE_VALUES[key], color="red", linewidth=1.8,
               linestyle="--", alpha=0.85, label=f"True = {TRUE_VALUES[key]}")
    ax.set_xlabel(PARAMS[key]["label"], fontsize=10)
    ax.set_ylabel("KGE", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", facecolor="white", framealpha=0.85)
    ax.grid(alpha=0.25)

fig.suptitle(
    f"KGE response to each parameter — Synthetic Inversion  {SERIES_TITLE}  (n={{len(df)}})\n"
    f"Red dashed line = true parameter value  |  f fixed at 0.020 mm^-1",
    fontsize=12
)
fig.tight_layout()
save_fig(fig, "fig6_kge_vs_each_param.png")


# -----------------------------------------------------------------------
# FIGURE 7: PBIAS vs KGE scatter
# -----------------------------------------------------------------------
print("Figure 7: PBIAS vs KGE scatter")

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(pbias_vals, kge_vals, c=ks_vals, cmap="plasma",
                s=35, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)
ax.scatter(pbias_vals[best_idx], kge_vals[best_idx],
           s=200, marker="*", color="white", edgecolors="black",
           linewidths=1.5, zorder=5, label=f"Best run  KGE={best_kge:.3f}")
ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5,
           label="PBIAS = 0 (no volume bias)")
ax.axhline(0, color="gray",  linewidth=0.8, linestyle=":",  alpha=0.5,
           label="KGE = 0 (no skill)")
ax.set_xlabel("PBIAS (%)  — positive = over-predict volume", fontsize=11)
ax.set_ylabel("KGE", fontsize=11)
ax.set_title(
    f"PBIAS vs KGE — 5-param LHS  {SERIES_TITLE}  (n={len(df)})\n"
    f"Points colored by Ks multiplier  |  {EVENT_LABEL}",
    fontsize=12
)
fig.colorbar(sc, ax=ax, label="Ks multiplier")
ax.legend(fontsize=9, loc="lower right", facecolor="white", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
save_fig(fig, "fig7_pbias_vs_kge.png")


# -----------------------------------------------------------------------
# FIGURE 8: Top-15 runs as formatted table
# -----------------------------------------------------------------------
print("Figure 8: Top-15 table")

top_cols_display = {
    "Ks_mult":          "Ks x",
    "kinemvelcoef":     "cv",
    "flowexp":          "r",
    "channelroughness": "n",
    "kge":              "KGE",
    "nse":              "NSE",
    "pbias_pct":        "PBIAS %",
    "kge_r":            "r (KGE)",
    "kge_alpha":        "alpha",
    "kge_beta":         "beta",
}
top_cols_available = [c for c in top_cols_display if c in df.columns]
top15 = df.sort_values("kge", ascending=False).head(15)[top_cols_available].copy()
top15 = top15.rename(columns=top_cols_display)

fmt_map = {
    "Ks x":    "{:.2f}",
    "cv":      "{:.2f}",
    "r":       "{:.3f}",
    "n":       "{:.4f}",
    "KGE":     "{:.3f}",
    "NSE":     "{:.3f}",
    "PBIAS %": "{:+.1f}",
    "r (KGE)": "{:.3f}",
    "alpha":   "{:.3f}",
    "beta":    "{:.3f}",
}
for col, fmt in fmt_map.items():
    if col in top15.columns:
        top15[col] = top15[col].apply(lambda v: fmt.format(v))

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.axis("off")
tbl = ax.table(cellText=top15.values, colLabels=top15.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.55)

for j in range(len(top15.columns)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(top15) + 1):
    bg = "#eaf4ea" if i == 1 else ("#f7f7f7" if i % 2 == 0 else "white")
    for j in range(len(top15.columns)):
        tbl[i, j].set_facecolor(bg)

ax.set_title(
    f"Top 15 runs by KGE — Synthetic Inversion  {SERIES_TITLE}  (n={{len(df)}})\n"
    f"True values: Ks=8.5x  cv=5.75  r=0.23  n=0.02  |  Green row = best run",
    fontsize=11, pad=12
)
fig.tight_layout()
save_fig(fig, "fig8_top15_table.png")

# -----------------------------------------------------------------------
# FIGURE 9: KGE component decomposition vs Ks
# 3-panel figure: r, alpha, beta each vs Ks multiplier, colored by KGE.
# Shows which KGE component is limiting performance and how it responds to Ks.
# -----------------------------------------------------------------------
print("Figure 9: KGE component decomposition vs Ks")

kge_r_vals     = df["kge_r"].values
kge_alpha_vals = df["kge_alpha"].values
kge_beta_vals  = df["kge_beta"].values

components = [
    {"col": kge_r_vals,     "label": "r  (timing correlation)",  "ideal": 1.0, "color": "#2a9d8f"},
    {"col": kge_alpha_vals, "label": "\u03b1  (variability ratio)", "ideal": 1.0, "color": "#e9c46a"},
    {"col": kge_beta_vals,  "label": "\u03b2  (bias ratio)",        "ideal": 1.0, "color": "#e76f51"},
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, comp in zip(axes, components):
    sc = ax.scatter(ks_vals, comp["col"],
                    c=kge_vals, cmap=KGE_CMAP, norm=KGE_NORM,
                    s=35, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)

    # Best run star
    ax.scatter(ks_vals[best_idx], comp["col"][best_idx],
               s=180, marker="*", color="white", edgecolors="black",
               linewidths=1.2, zorder=5, label=f"Best KGE={best_kge:.3f}")

    # Rolling median trend line
    sort_order = np.argsort(ks_vals)
    xs_sorted  = ks_vals[sort_order]
    ys_sorted  = comp["col"][sort_order]
    window     = max(5, len(df) // 8)
    ys_smooth  = pd.Series(ys_sorted).rolling(window, center=True,
                                               min_periods=1).median().values
    ax.plot(xs_sorted, ys_smooth, color="navy", linewidth=1.6,
            linestyle="--", alpha=0.7, label="Rolling median")

    # Ideal value reference line
    ax.axhline(comp["ideal"], color=comp["color"], linewidth=1.2,
               linestyle="-", alpha=0.6, label=f"Ideal = {comp['ideal']:.1f}")

    ax.set_xlabel("Ks multiplier", fontsize=10)
    ax.set_ylabel(comp["label"], fontsize=10)
    ax.set_title(comp["label"], fontsize=11)
    ax.legend(fontsize=8, loc="best", facecolor="white", framealpha=0.85)
    ax.grid(alpha=0.25)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=KGE_CMAP, norm=KGE_NORM)
sm.set_array([])
fig.colorbar(sm, ax=axes.tolist(), label="KGE", shrink=0.7, pad=0.02)

fig.suptitle(
    f"KGE component decomposition vs Ks — 5-param LHS  {SERIES_TITLE}  (n={len(df)})\n"
    f"{EVENT_LABEL}  |  Dashed = rolling median  |  Colored line = ideal value",
    fontsize=12
)
fig.tight_layout()
save_fig(fig, "fig9_kge_components_vs_ks.png")

# -----------------------------------------------------------------------
# CONSOLE SUMMARY
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"All figures saved to:\n  {plot_dir}")
print(f"\nParameter-KGE correlations (Pearson r):")
for key, label in zip(PARAM_KEYS, PARAM_LABELS):
    r_corr = np.corrcoef(df[key].values, kge_vals)[0, 1]
    print(f"  {label:<35s}  r = {r_corr:+.3f}")
print(f"\nTop 5 runs by KGE:")
top5_cols  = ["run_id", "Ks_mult", "kinemvelcoef", "flowexp",
              "channelroughness", "kge", "pbias_pct"]
top5_avail = [c for c in top5_cols if c in df.columns]
print(df.sort_values("kge", ascending=False).head(5)[top5_avail]
        .to_string(index=False, float_format="%.4f"))
print(f"{'='*60}\n")
