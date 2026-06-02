"""
plot_lhs_synth_4param.py
========================
Generates all diagnostic figures from the Series 91 synthetic inversion
LHS sweep. Includes true value reference lines on all 1D response plots
and pairwise scatter panels to assess parameter identifiability.

Usage (run from the smf_demo directory):
    python plot_lhs_synth_4param.py

Produces 9 figures saved to:
    calibration_work/03_comparisons/sensitivity_plots/Series91_SynthInversion/

Figure list
-----------
    fig1_hydrograph_envelope_all.png   — All runs: envelope + median + best
    fig2_hydrograph_envelope_kge0.png  — KGE > 0 runs only
    fig3_correlation_bar.png           — Pearson r of each param vs KGE
    fig4_parallel_coordinates.png      — 4 params as vertical axes, colored by KGE
    fig5_pairwise_scatter.png          — 6-panel pairwise scatter, colored by KGE
    fig6_kge_vs_each_param.png         — 4-panel KGE vs each param + true value line
    fig7_pbias_vs_kge.png              — PBIAS vs KGE scatter
    fig8_top15_table.png               — Top-15 runs by KGE
    fig9_kge_components_vs_ks.png      — KGE r/alpha/beta decomposition vs Ks

KGE ceiling note
----------------
The self-score of the truth run is KGE = 0.912 due to resampling
asymmetry between sim (.mean()) and obs (.interpolate()). The ceiling
annotation is drawn on Fig 6 and Fig 7. Interpret recovery as top runs
clustering near the true value lines, not as KGE approaching 1.0.
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
RESULTS_CSV  = "lhs_results_synth_4param_91.csv"
SERIES_LABEL = "Series91_SynthInversion"
# =======================================================================

EVENT_LABEL      = "SMF Aug 12, 2014  |  Synthetic Inversion"
EVENT_CROP_START = "2014-08-12 17:30"
EVENT_CROP_END   = "2014-08-12 21:00"

# KGE ceiling from truth self-score (resampling asymmetry)
KGE_CEILING = 0.912

# True parameter values — drawn as red dashed lines on Fig 6
TRUE_VALUES = {
    "Ks_mult":          8.50,
    "kinemvelcoef":     4.50,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}

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
        f"Run run_lhs_synth_4param.py first."
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

ks_lo = df["Ks_mult"].min()
ks_hi = df["Ks_mult"].max()
SERIES_TITLE = (f"Series 91  |  Synthetic Inversion  |  "
                f"Ks {ks_lo:.1f}-{ks_hi:.1f}x  |  KGE ceiling={KGE_CEILING}")

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

print(f"\n  Best run:  {best_run_id}")
print(f"  Best KGE:  {best_kge:.3f}  (ceiling = {KGE_CEILING})")
print(f"  KGE range: {kge_vals.min():.3f} to {kge_vals.max():.3f}")
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

# KGE colormap — plasma; norm on 5th–95th percentile with clip
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
    print("  WARNING: No observed series found — hydrograph figures will be skipped.")


# -----------------------------------------------------------------------
# FIGURES 1 & 2: Hydrograph uncertainty envelopes
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

    sim_matrix  = sim_matrix.dropna(how="all")
    if sim_matrix.empty:
        print(f"  Skipping {filename} — sim_matrix empty after dropna.")
        return

    sim_matrix  = sim_matrix.loc[EVENT_CROP_START:EVENT_CROP_END]
    obs_cropped = obs_series.reindex(sim_matrix.index)

    sim_min    = sim_matrix.min(axis=1)
    sim_max    = sim_matrix.max(axis=1)
    sim_median = sim_matrix.median(axis=1)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(sim_matrix.index, sim_min, sim_max,
                    alpha=0.25, color=envelope_color, label="Ensemble envelope")
    ax.plot(sim_matrix.index, sim_median,
            color=median_color, linewidth=2.0, linestyle="--", label="Ensemble median")

    if best_run_in_subset in hydros_subset:
        best_sim = hydros_subset[best_run_in_subset]["Simulated"].reindex(sim_matrix.index)
        ax.plot(sim_matrix.index, best_sim,
                color="gold", linewidth=2.2, label=f"Best run (KGE={best_kge:.3f})")

    ax.plot(sim_matrix.index, obs_cropped,
            color="black", linewidth=2.5, label="Synthetic truth")

    # KGE ceiling annotation
    ax.axhline(obs_cropped.max() * 0.0, color="none")
    ax.text(0.02, 0.97, f"KGE ceiling = {KGE_CEILING}",
            transform=ax.transAxes, fontsize=9, va="top",
            color="gray", style="italic")

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (Aug 12, 2014)", fontsize=11)
    ax.set_ylabel("Discharge (m³/s)", fontsize=11)
    ax.set_title(
        f"Hydrograph ensemble — {filter_label} {title_suffix}\n"
        f"{SERIES_TITLE}  |  n={len(hydros_subset)}  |  {EVENT_LABEL}",
        fontsize=11)
    ax.legend(fontsize=9, facecolor="white", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, filename)


print("Figure 1: Hydrograph envelope — all runs")
plot_hydrograph_envelope(
    all_hydros, df,
    title_suffix="— all runs",
    filename="fig1_hydrograph_envelope_all.png",
    envelope_color="steelblue", median_color="navy",
    filter_label="All runs")

print("Figure 2: Hydrograph envelope — KGE > 0 runs")
df_pos    = df[df["kge"] > 0].reset_index(drop=True)
hydros_pos = {rid: all_hydros[rid] for rid in df_pos["run_id"] if rid in all_hydros}
plot_hydrograph_envelope(
    hydros_pos, df_pos,
    title_suffix="— KGE > 0 runs",
    filename="fig2_hydrograph_envelope_kge0.png",
    envelope_color="darkorange", median_color="saddlebrown",
    filter_label="KGE > 0 runs")


# -----------------------------------------------------------------------
# FIGURE 3: Pearson r bar chart
# -----------------------------------------------------------------------
print("Figure 3: Correlation bar chart")

correlations = {key: np.corrcoef(df[key].values, kge_vals)[0, 1]
                for key in PARAM_KEYS}
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in correlations.values()]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(PARAM_LABELS, list(correlations.values()),
               color=colors, edgecolor="white", height=0.55)
ax.axvline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, correlations.values()):
    ax.text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)
ax.set_xlabel("Pearson r with KGE", fontsize=11)
ax.set_title(
    f"Parameter–KGE correlations — {SERIES_TITLE}\n"
    f"{EVENT_LABEL}  |  Green = positive, Red = negative",
    fontsize=11)
ax.set_xlim(-1, 1)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig3_correlation_bar.png")


# -----------------------------------------------------------------------
# FIGURE 4: Parallel coordinates
# -----------------------------------------------------------------------
print("Figure 4: Parallel coordinates")

fig, ax = plt.subplots(figsize=(11, 6))
x_pos = np.arange(len(PARAM_KEYS))

for idx, row in df.iterrows():
    vals_norm = []
    for key in PARAM_KEYS:
        lo  = df[key].min()
        hi  = df[key].max()
        rng = hi - lo if hi > lo else 1.0
        vals_norm.append((row[key] - lo) / rng)
    color = KGE_CMAP(KGE_NORM(row["kge"]))
    ax.plot(x_pos, vals_norm, color=color, alpha=0.4, linewidth=0.8)

# True value lines
for xi, key in enumerate(PARAM_KEYS):
    lo  = df[key].min()
    hi  = df[key].max()
    rng = hi - lo if hi > lo else 1.0
    tv_norm = (TRUE_VALUES[key] - lo) / rng
    ax.plot([xi - 0.08, xi + 0.08], [tv_norm, tv_norm],
            color="red", linewidth=3.0, zorder=5,
            label="True value" if xi == 0 else "")

ax.set_xticks(x_pos)
ax.set_xticklabels(PARAM_LABELS, fontsize=10)
ax.set_ylabel("Normalised parameter value", fontsize=10)
ax.set_title(
    f"Parallel coordinates — {SERIES_TITLE}\n"
    f"Lines colored by KGE  |  Red tick = true value  |  {EVENT_LABEL}",
    fontsize=11)
sm = plt.cm.ScalarMappable(cmap=KGE_CMAP, norm=KGE_NORM)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="KGE", fraction=0.03, pad=0.02)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig4_parallel_coordinates.png")


# -----------------------------------------------------------------------
# FIGURE 5: Pairwise scatter matrix
# -----------------------------------------------------------------------
print("Figure 5: Pairwise scatter matrix")

pairs     = list(combinations(PARAM_KEYS, 2))
n_pairs   = len(pairs)
n_cols    = 3
n_rows    = (n_pairs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes_flat = axes.flatten()

for pi, (k1, k2) in enumerate(pairs):
    ax = axes_flat[pi]
    sc = ax.scatter(df[k1].values, df[k2].values,
                    c=kge_vals, cmap=KGE_CMAP, norm=KGE_NORM,
                    s=28, edgecolors="white", linewidths=0.4, alpha=0.85)
    ax.scatter(df[k1].iloc[best_idx], df[k2].iloc[best_idx],
               s=180, marker="*", color="white", edgecolors="black",
               linewidths=1.2, zorder=5, label=f"Best KGE={best_kge:.3f}")
    # True value crosshair
    ax.axvline(TRUE_VALUES[k1], color="red", linewidth=1.5,
               linestyle="--", alpha=0.8, label="True value")
    ax.axhline(TRUE_VALUES[k2], color="red", linewidth=1.5,
               linestyle="--", alpha=0.8)
    ax.set_xlabel(PARAMS[k1]["label"], fontsize=9)
    ax.set_ylabel(PARAMS[k2]["label"], fontsize=9)
    ax.legend(fontsize=7, loc="upper right", facecolor="white", framealpha=0.85)
    ax.grid(alpha=0.2)

for pi in range(n_pairs, len(axes_flat)):
    axes_flat[pi].set_visible(False)

fig.suptitle(
    f"Pairwise parameter scatter — {SERIES_TITLE}\n"
    f"Colored by KGE  |  Red dashed = true values  |  ★ = best run",
    fontsize=12)
fig.tight_layout()
save_fig(fig, "fig5_pairwise_scatter.png")


# -----------------------------------------------------------------------
# FIGURE 6: KGE vs each parameter — 1D response curves
# Key identifiability figure — does rolling median peak near true value?
# -----------------------------------------------------------------------
print("Figure 6: KGE vs each parameter")

fig, axes = plt.subplots(1, len(PARAM_KEYS),
                          figsize=(5 * len(PARAM_KEYS), 5), sharey=False)

for ax, key in zip(axes, PARAM_KEYS):
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
    ys_smooth  = pd.Series(ys_sorted).rolling(
        window, center=True, min_periods=1).median().values
    ax.plot(xs_sorted, ys_smooth, color="navy", linewidth=1.6,
            linestyle="--", alpha=0.7, label="Rolling median")

    # True value line
    ax.axvline(TRUE_VALUES[key], color="red", linewidth=1.8,
               linestyle="--", alpha=0.85,
               label=f"True = {TRUE_VALUES[key]}")

    # KGE ceiling line
    ax.axhline(KGE_CEILING, color="gray", linewidth=1.0,
               linestyle=":", alpha=0.7,
               label=f"KGE ceiling = {KGE_CEILING}")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel(PARAMS[key]["label"], fontsize=10)
    ax.set_ylabel("KGE", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", facecolor="white", framealpha=0.85)
    ax.grid(alpha=0.25)

fig.suptitle(
    f"KGE response to each parameter — {SERIES_TITLE}  (n={len(df)})\n"
    f"Red dashed = true value  |  Gray dotted = KGE ceiling ({KGE_CEILING})  "
    f"|  f fixed at {0.020} mm⁻¹",
    fontsize=12)
fig.tight_layout()
save_fig(fig, "fig6_kge_vs_each_param.png")


# -----------------------------------------------------------------------
# FIGURE 7: PBIAS vs KGE
# -----------------------------------------------------------------------
print("Figure 7: PBIAS vs KGE scatter")

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(pbias_vals, kge_vals, c=ks_vals, cmap="plasma",
                s=35, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)
ax.scatter(pbias_vals[best_idx], kge_vals[best_idx],
           s=200, marker="*", color="white", edgecolors="black",
           linewidths=1.5, zorder=5, label=f"Best run  KGE={best_kge:.3f}")
ax.axvline(0,           color="black", linewidth=0.9, linestyle="--",
           alpha=0.5, label="PBIAS = 0")
ax.axhline(0,           color="gray",  linewidth=0.8, linestyle=":",
           alpha=0.5, label="KGE = 0")
ax.axhline(KGE_CEILING, color="gray",  linewidth=1.2, linestyle=":",
           alpha=0.8, label=f"KGE ceiling = {KGE_CEILING}")
ax.set_xlabel("PBIAS (%)  — positive = over-predict volume", fontsize=11)
ax.set_ylabel("KGE", fontsize=11)
ax.set_title(
    f"PBIAS vs KGE — {SERIES_TITLE}  (n={len(df)})\n"
    f"Points colored by Ks multiplier  |  {EVENT_LABEL}",
    fontsize=12)
fig.colorbar(sc, ax=ax, label="Ks multiplier", fraction=0.046, pad=0.04)
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
top15 = (df.sort_values("kge", ascending=False)
           .head(15)[top_cols_available].copy())
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
    f"Top 15 runs by KGE — {SERIES_TITLE}  (n={len(df)})\n"
    f"True values: Ks=8.5x  cv=4.5  r=0.24  n=0.026  "
    f"|  KGE ceiling={KGE_CEILING}  |  Green row = best run",
    fontsize=11, pad=12)
fig.tight_layout()
save_fig(fig, "fig8_top15_table.png")


# -----------------------------------------------------------------------
# FIGURE 9: KGE component decomposition vs Ks
# -----------------------------------------------------------------------
print("Figure 9: KGE component decomposition vs Ks")

kge_r_vals     = df["kge_r"].values
kge_alpha_vals = df["kge_alpha"].values
kge_beta_vals  = df["kge_beta"].values

components = [
    {"col": kge_r_vals,     "label": "r  (timing correlation)",    "ideal": 1.0, "color": "#2a9d8f"},
    {"col": kge_alpha_vals, "label": "\u03b1  (variability ratio)", "ideal": 1.0, "color": "#e9c46a"},
    {"col": kge_beta_vals,  "label": "\u03b2  (bias ratio)",        "ideal": 1.0, "color": "#e76f51"},
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, comp in zip(axes, components):
    sc = ax.scatter(ks_vals, comp["col"],
                    c=kge_vals, cmap=KGE_CMAP, norm=KGE_NORM,
                    s=35, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)
    ax.scatter(ks_vals[best_idx], comp["col"][best_idx],
               s=180, marker="*", color="white", edgecolors="black",
               linewidths=1.2, zorder=5, label=f"Best KGE={best_kge:.3f}")

    sort_order = np.argsort(ks_vals)
    xs_sorted  = ks_vals[sort_order]
    ys_sorted  = comp["col"][sort_order]
    window     = max(5, len(df) // 8)
    ys_smooth  = pd.Series(ys_sorted).rolling(
        window, center=True, min_periods=1).median().values
    ax.plot(xs_sorted, ys_smooth, color="navy", linewidth=1.6,
            linestyle="--", alpha=0.7, label="Rolling median")
    ax.axhline(comp["ideal"], color=comp["color"], linewidth=1.2,
               linestyle="-", alpha=0.6, label=f"Ideal = {comp['ideal']:.1f}")
    ax.axvline(TRUE_VALUES["Ks_mult"], color="red", linewidth=1.5,
               linestyle="--", alpha=0.8,
               label=f"True Ks = {TRUE_VALUES['Ks_mult']}")

    ax.set_xlabel("Ks multiplier", fontsize=10)
    ax.set_ylabel(comp["label"], fontsize=10)
    ax.set_title(comp["label"], fontsize=11)
    ax.legend(fontsize=8, loc="best", facecolor="white", framealpha=0.85)
    ax.grid(alpha=0.25)

sm = plt.cm.ScalarMappable(cmap=KGE_CMAP, norm=KGE_NORM)
sm.set_array([])
fig.suptitle(
    f"KGE component decomposition vs Ks — {SERIES_TITLE}  (n={len(df)})\n"
    f"{EVENT_LABEL}  |  Dashed navy = rolling median  |  Red = true Ks value",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 0.88, 1.0])
fig.colorbar(sm, ax=axes.tolist(), label="KGE", shrink=0.8, pad=0.02)
save_fig(fig, "fig9_kge_components_vs_ks.png")


# -----------------------------------------------------------------------
# CONSOLE SUMMARY
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"All figures saved to:\n  {plot_dir}")
print(f"\nTrue values for reference:")
for key, val in TRUE_VALUES.items():
    print(f"  {key:<22s}  true = {val}")
print(f"\nParameter-KGE correlations (Pearson r):")
for key, label in zip(PARAM_KEYS, PARAM_LABELS):
    r_corr = np.corrcoef(df[key].values, kge_vals)[0, 1]
    print(f"  {label:<35s}  r = {r_corr:+.3f}")
print(f"\nTop 5 runs by KGE (ceiling = {KGE_CEILING}):")
top5_cols  = ["run_id", "Ks_mult", "kinemvelcoef", "flowexp",
              "channelroughness", "kge", "pbias_pct"]
top5_avail = [c for c in top5_cols if c in df.columns]
print(df.sort_values("kge", ascending=False).head(5)[top5_avail]
        .to_string(index=False, float_format="%.4f"))
print(f"{'='*60}\n")