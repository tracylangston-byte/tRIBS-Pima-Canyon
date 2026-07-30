"""
plot_doublepeak_r_hypothesis.py
================================
Tests the hypothesis that the double-peaked hydrograph seen in the Series 92
ensemble envelope is produced by runs with low flowexp (r) values, not by
all runs.

The physical argument: low r → fast hillslope routing → the SMPHQ-forced
northeastern watershed delivers runoff to the outlet before the larger
SMF-forced signal arrives, producing a visible first pulse. At the true
r=0.24, routing is slow enough that the two signals merge into a single peak.

This script:
    1. Splits the ensemble into low-r / mid-r / high-r terciles
    2. Plots one hydrograph envelope per tercile, side by side and overlaid
    3. Applies a simple double-peak detector to every run and reports the
       fraction of double-peaked runs in each r tercile
    4. Produces a scatter plot of r vs double-peak score, colored by KGE

Usage (run from smf_demo directory):
    python plot_doublepeak_r_hypothesis.py

Input:
    calibration_work/03_comparisons/summary_tables/lhs_results_synth_4param_92.csv
    calibration_work/03_comparisons/csv_exports/*_compare_obs_sim.csv

Output:
    calibration_work/03_comparisons/sensitivity_plots/Series92_SynthInversion/
        fig15_doublepeak_envelopes_by_r_tercile.png
        fig16_doublepeak_overlaid_envelopes.png
        fig17_doublepeak_score_vs_r.png
        fig18_doublepeak_fraction_by_r_bin.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from pathlib import Path
from scipy.signal import argrelextrema

# =======================================================================
# CONFIG
# =======================================================================
RESULTS_CSV  = "lhs_results_synth_4param_92.csv"
SERIES_LABEL = "Series92_SynthInversion"

KGE_CEILING = 0.912

TRUE_VALUES = {
    "Ks_mult":          8.50,
    "kinemvelcoef":     4.50,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}

EVENT_CROP_START = "2014-08-12 17:30"
EVENT_CROP_END   = "2014-08-12 21:00"
EVENT_LABEL      = "SMF Aug 12, 2014  |  Synthetic Inversion"

# Double-peak detector settings
# A "second peak" is detected if, after the global peak, discharge rises again
# by more than REPEAK_FRAC of the global peak before the end of the event.
# Alternatively, we look for a local maximum before the global peak that
# exceeds PRE_PEAK_FRAC of the global peak.
PRE_PEAK_FRAC  = 0.25   # pre-peak bump must be at least 25% of global peak
REPEAK_FRAC    = 0.15   # post-peak rise must be at least 15% of global peak
SMOOTH_WINDOW  = 3      # rolling mean window (timesteps) before peak detection

# Number of r bins for the fraction plot
N_R_BINS = 5

# =======================================================================
# PATHS
# =======================================================================
script_dir = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
csv_dir      = calib_dir / "03_comparisons" / "csv_exports"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / SERIES_LABEL
plot_dir.mkdir(parents=True, exist_ok=True)


# =======================================================================
# LOAD RESULTS
# =======================================================================
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(f"Results not found: {results_path}")

df = pd.read_csv(results_path)
df = df.dropna(subset=["flowexp", "kge"]).reset_index(drop=True)
print(f"Loaded {len(df)} runs from {results_path.name}")
print(f"  flowexp range: {df['flowexp'].min():.3f} – {df['flowexp'].max():.3f}")
print(f"  True r = {TRUE_VALUES['flowexp']}")


def save_fig(fig, filename):
    path = plot_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


# =======================================================================
# LOAD HYDROGRAPH CSVs
# =======================================================================
print("\nLoading hydrograph CSVs...")
all_hydros    = {}
missing_count = 0
for run_id in df["run_id"]:
    csv_path = csv_dir / f"{run_id}_compare_obs_sim.csv"
    if csv_path.exists():
        try:
            hdf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            all_hydros[run_id] = hdf
        except Exception as e:
            print(f"  Warning: could not read {csv_path.name}: {e}")
            missing_count += 1
    else:
        missing_count += 1

if missing_count:
    print(f"  Warning: {missing_count} CSVs not found.")
print(f"  Loaded {len(all_hydros)} hydrograph CSVs.")

# Extract observed series from first available hydro
obs_series = None
for hdf in all_hydros.values():
    if "Observed" in hdf.columns:
        obs_series = hdf["Observed"]
        break

if obs_series is None:
    raise ValueError("No observed series found in any hydrograph CSV.")


# =======================================================================
# DOUBLE-PEAK DETECTOR
# =======================================================================
def detect_double_peak(sim: pd.Series,
                       pre_peak_frac=PRE_PEAK_FRAC,
                       repeak_frac=REPEAK_FRAC,
                       smooth_window=SMOOTH_WINDOW) -> dict:
    """
    Detect whether a simulated hydrograph has a double peak.

    Strategy:
      1. Smooth the series with a rolling mean to suppress noise.
      2. Find all local maxima.
      3. Flag as double-peaked if:
         (a) There is a local maximum BEFORE the global peak that exceeds
             pre_peak_frac * global_peak  (early pre-peak bump), OR
         (b) There is a local maximum AFTER the global peak that exceeds
             repeak_frac * global_peak  (post-peak resurgence).

    Returns
    -------
    dict with keys:
        double_peak     : bool
        n_local_maxima  : int  (total local maxima found)
        pre_peak_bump   : float  (height of largest pre-peak local max / global peak)
        post_peak_bump  : float  (height of largest post-peak local max / global peak)
        double_peak_score : float  (max of pre_peak_bump, post_peak_bump) — 0=clean, 1=equal peaks
    """
    if sim.empty or sim.max() <= 0:
        return {"double_peak": False, "n_local_maxima": 0,
                "pre_peak_bump": 0.0, "post_peak_bump": 0.0,
                "double_peak_score": 0.0}

    # Smooth
    smoothed = sim.rolling(smooth_window, center=True, min_periods=1).mean()
    vals     = smoothed.values
    global_peak_idx = int(np.argmax(vals))
    global_peak_val = vals[global_peak_idx]

    # Local maxima (order=2: must be larger than 2 neighbors on each side)
    local_max_idxs = argrelextrema(vals, np.greater_equal, order=2)[0]

    # Remove the global peak itself from the local maxima list for separate analysis
    other_maxima = [i for i in local_max_idxs if i != global_peak_idx]

    pre_peak_maxima  = [i for i in other_maxima if i < global_peak_idx]
    post_peak_maxima = [i for i in other_maxima if i > global_peak_idx]

    pre_peak_bump  = (max(vals[pre_peak_maxima])  / global_peak_val
                      if pre_peak_maxima  else 0.0)
    post_peak_bump = (max(vals[post_peak_maxima]) / global_peak_val
                      if post_peak_maxima else 0.0)

    double_peak_score = max(pre_peak_bump, post_peak_bump)
    double_peak       = (pre_peak_bump  >= pre_peak_frac or
                         post_peak_bump >= repeak_frac)

    return {
        "double_peak":        double_peak,
        "n_local_maxima":     len(local_max_idxs),
        "pre_peak_bump":      pre_peak_bump,
        "post_peak_bump":     post_peak_bump,
        "double_peak_score":  double_peak_score,
    }


# =======================================================================
# APPLY DETECTOR TO ALL RUNS
# =======================================================================
print("\nApplying double-peak detector to all runs...")
dp_records = []
for _, row in df.iterrows():
    run_id = row["run_id"]
    if run_id not in all_hydros:
        dp_records.append({"run_id": run_id, "double_peak": False,
                            "double_peak_score": 0.0,
                            "pre_peak_bump": 0.0, "post_peak_bump": 0.0})
        continue
    hdf    = all_hydros[run_id]
    sim    = hdf["Simulated"]
    # Crop to event window
    try:
        sim_crop = sim.loc[EVENT_CROP_START:EVENT_CROP_END]
    except Exception:
        sim_crop = sim
    result = detect_double_peak(sim_crop)
    result["run_id"] = run_id
    dp_records.append(result)

dp_df = pd.DataFrame(dp_records)
df    = df.merge(dp_df, on="run_id", how="left")

n_double = df["double_peak"].sum()
print(f"  Double-peaked runs: {n_double} / {len(df)} "
      f"({100*n_double/len(df):.1f}%)")
print(f"  Using thresholds: pre_peak_frac={PRE_PEAK_FRAC}, "
      f"repeak_frac={REPEAK_FRAC}")

# Save augmented results
augmented_path = summary_dir / "lhs_results_synth_4param_92_doublepeak.csv"
df.to_csv(augmented_path, index=False)
print(f"  Augmented results saved: {augmented_path.name}")


# =======================================================================
# SPLIT INTO r TERCILES
# =======================================================================
r_vals    = df["flowexp"].values
r_terciles = np.percentile(r_vals, [33.3, 66.7])

df["r_tercile"] = pd.cut(
    df["flowexp"],
    bins=[r_vals.min() - 1e-9, r_terciles[0], r_terciles[1], r_vals.max() + 1e-9],
    labels=["Low r", "Mid r", "High r"]
)

tercile_info = {
    "Low r":  {"color": "#e76f51", "range": f"r < {r_terciles[0]:.3f}"},
    "Mid r":  {"color": "#e9c46a", "range": f"{r_terciles[0]:.3f} ≤ r < {r_terciles[1]:.3f}"},
    "High r": {"color": "#2a9d8f", "range": f"r ≥ {r_terciles[1]:.3f}"},
}

print(f"\nr tercile boundaries: {r_terciles[0]:.3f} | {r_terciles[1]:.3f}")
for label in ["Low r", "Mid r", "High r"]:
    sub    = df[df["r_tercile"] == label]
    n_dp   = sub["double_peak"].sum()
    print(f"  {label} ({tercile_info[label]['range']}): "
          f"n={len(sub)}, double-peaked={n_dp} ({100*n_dp/max(len(sub),1):.1f}%)")


# =======================================================================
# HELPER: build sim matrix for a subset of runs
# =======================================================================
def build_sim_matrix(run_ids, crop_start, crop_end):
    common_idx = obs_series.loc[crop_start:crop_end].index
    mat = pd.DataFrame(index=common_idx)
    for rid in run_ids:
        if rid in all_hydros:
            mat[rid] = all_hydros[rid]["Simulated"].reindex(common_idx)
    mat = mat.dropna(how="all")
    return mat


# =======================================================================
# FIG 15: Side-by-side envelopes for each r tercile
# =======================================================================
print("\nFigure 15: Envelopes by r tercile (side by side)...")

fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)

obs_crop = obs_series.loc[EVENT_CROP_START:EVENT_CROP_END]

for ax, label in zip(axes, ["Low r", "Mid r", "High r"]):
    sub      = df[df["r_tercile"] == label]
    col      = tercile_info[label]["color"]
    rng      = tercile_info[label]["range"]
    run_ids  = sub["run_id"].values
    mat      = build_sim_matrix(run_ids, EVENT_CROP_START, EVENT_CROP_END)

    if mat.empty:
        ax.set_title(f"{label}\n(no data)")
        continue

    sim_min    = mat.min(axis=1)
    sim_max    = mat.max(axis=1)
    sim_median = mat.median(axis=1)

    ax.fill_between(mat.index, sim_min, sim_max,
                    alpha=0.30, color=col, label="Ensemble envelope")
    ax.plot(mat.index, sim_median,
            color=col, linewidth=2.0, linestyle="--", label="Ensemble median")
    ax.plot(obs_crop.index, obs_crop,
            color="black", linewidth=2.5, label="Synthetic truth")

    # Mark true r value
    ax.axvline(pd.Timestamp("2014-08-12 19:00"), color="none")  # spacer

    n_dp  = sub["double_peak"].sum()
    pct   = 100 * n_dp / max(len(sub), 1)
    r_med = sub["flowexp"].median()
    kge_med = sub["kge"].median()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (Aug 12, 2014)", fontsize=10)
    ax.set_ylabel("Discharge (m³/s)", fontsize=10)
    ax.set_title(
        f"{label}  |  {rng}\n"
        f"n={len(sub)}  |  median r={r_med:.3f}  |  median KGE={kge_med:.3f}\n"
        f"Double-peaked: {n_dp}/{len(sub)} ({pct:.0f}%)",
        fontsize=9.5)
    ax.legend(fontsize=8, facecolor="white", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.35)

    # Shade background by tercile color
    ax.set_facecolor(col + "11")

fig.suptitle(
    f"Hydrograph envelopes split by r (flowexp) tercile\n"
    f"Series 92  |  Synthetic Inversion  |  n={len(df)}  |  True r = {TRUE_VALUES['flowexp']}\n"
    f"Hypothesis: low-r runs produce double peak via faster hillslope routing",
    fontsize=11)
fig.tight_layout()
save_fig(fig, "fig15_doublepeak_envelopes_by_r_tercile.png")


# =======================================================================
# FIG 16: Overlaid envelopes — all three terciles on one axis
# Shows the shift in hydrograph shape with r
# =======================================================================
print("Figure 16: Overlaid envelopes by r tercile...")

fig, ax = plt.subplots(figsize=(12, 6))

for label in ["Low r", "Mid r", "High r"]:
    sub     = df[df["r_tercile"] == label]
    col     = tercile_info[label]["color"]
    rng     = tercile_info[label]["range"]
    run_ids = sub["run_id"].values
    mat     = build_sim_matrix(run_ids, EVENT_CROP_START, EVENT_CROP_END)

    if mat.empty:
        continue

    sim_median = mat.median(axis=1)
    sim_p25    = mat.quantile(0.25, axis=1)
    sim_p75    = mat.quantile(0.75, axis=1)

    n_dp = sub["double_peak"].sum()
    pct  = 100 * n_dp / max(len(sub), 1)

    ax.fill_between(mat.index, sim_p25, sim_p75,
                    alpha=0.20, color=col)
    ax.plot(mat.index, sim_median,
            color=col, linewidth=2.2,
            label=f"{label} ({rng})  |  {pct:.0f}% double-peaked")

ax.plot(obs_crop.index, obs_crop,
        color="black", linewidth=2.8, label="Synthetic truth")

# True value annotation
ax.axvline(pd.Timestamp(TRUE_VALUES["flowexp"]), color="none")  # spacer

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Time (Aug 12, 2014)", fontsize=11)
ax.set_ylabel("Discharge (m³/s)", fontsize=11)
ax.set_title(
    f"Ensemble medians by r tercile — Series 92  |  Synthetic Inversion\n"
    f"Shaded band = 25th–75th percentile  |  True r = {TRUE_VALUES['flowexp']}  |  "
    f"n={len(df)}",
    fontsize=11)
ax.legend(fontsize=9, facecolor="white", framealpha=0.9)
ax.grid(linestyle=":", alpha=0.35)
fig.tight_layout()
save_fig(fig, "fig16_doublepeak_overlaid_envelopes.png")


# =======================================================================
# FIG 17: Double-peak score vs r, colored by KGE
# =======================================================================
print("Figure 17: Double-peak score vs r...")

r_plot    = df["flowexp"].values
dp_score  = df["double_peak_score"].values
kge_plot  = df["kge"].values
dp_flag   = df["double_peak"].values.astype(bool)

kge_norm = mcolors.Normalize(
    vmin=np.nanpercentile(kge_plot, 5),
    vmax=np.nanpercentile(kge_plot, 95), clip=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Non-double-peaked runs
mask_single = ~dp_flag
ax.scatter(r_plot[mask_single], dp_score[mask_single],
           c=kge_plot[mask_single], cmap="plasma", norm=kge_norm,
           s=40, edgecolors="white", linewidths=0.4, alpha=0.8,
           marker="o", label="Single-peaked", zorder=3)

# Double-peaked runs
mask_double = dp_flag
sc = ax.scatter(r_plot[mask_double], dp_score[mask_double],
                c=kge_plot[mask_double], cmap="plasma", norm=kge_norm,
                s=80, edgecolors="black", linewidths=0.8, alpha=0.9,
                marker="D", label="Double-peaked", zorder=4)

# True r line
ax.axvline(TRUE_VALUES["flowexp"], color="red", linewidth=1.8,
           linestyle="--", alpha=0.85,
           label=f"True r = {TRUE_VALUES['flowexp']}")

# Threshold lines
ax.axhline(PRE_PEAK_FRAC, color="gray", linewidth=1.0, linestyle=":",
           alpha=0.7, label=f"Pre-peak threshold = {PRE_PEAK_FRAC}")
ax.axhline(REPEAK_FRAC,   color="silver", linewidth=1.0, linestyle=":",
           alpha=0.7, label=f"Re-peak threshold = {REPEAK_FRAC}")

# Rolling median
sort_idx   = np.argsort(r_plot)
r_sorted   = r_plot[sort_idx]
dp_sorted  = dp_score[sort_idx]
window     = max(5, len(df) // 8)
dp_smooth  = (pd.Series(dp_sorted)
              .rolling(window, center=True, min_periods=1)
              .median().values)
ax.plot(r_sorted, dp_smooth, color="navy", linewidth=1.8,
        linestyle="--", alpha=0.75, label="Rolling median", zorder=5)

fig.colorbar(sc, ax=ax, label="KGE", fraction=0.04, pad=0.03)
ax.set_xlabel("Hillslope velocity exponent r (flowexp)", fontsize=11)
ax.set_ylabel("Double-peak score\n(secondary peak height / global peak)", fontsize=11)
ax.set_title(
    f"Double-peak score vs r — Series 92  |  Synthetic Inversion  (n={len(df)})\n"
    f"Diamonds = flagged double-peaked  |  Circles = single-peaked  |  "
    f"Colored by KGE\n"
    f"Thresholds: pre-peak={PRE_PEAK_FRAC}, re-peak={REPEAK_FRAC}",
    fontsize=11)
ax.legend(fontsize=8, facecolor="white", framealpha=0.9, loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
save_fig(fig, "fig17_doublepeak_score_vs_r.png")


# =======================================================================
# FIG 18: Double-peak fraction by r bin
# Bar chart: what fraction of runs in each r bin are double-peaked?
# =======================================================================
print("Figure 18: Double-peak fraction by r bin...")

r_bin_edges  = np.linspace(df["flowexp"].min(), df["flowexp"].max(), N_R_BINS + 1)
r_bin_labels = [f"{r_bin_edges[i]:.3f}–{r_bin_edges[i+1]:.3f}"
                for i in range(N_R_BINS)]
df["r_bin"] = pd.cut(df["flowexp"], bins=r_bin_edges, labels=r_bin_labels,
                      include_lowest=True)

bin_stats = []
for lbl in r_bin_labels:
    sub    = df[df["r_bin"] == lbl]
    n_tot  = len(sub)
    n_dp   = sub["double_peak"].sum()
    frac   = n_dp / max(n_tot, 1)
    r_mid  = sub["flowexp"].mean()
    kge_md = sub["kge"].median()
    bin_stats.append({"bin": lbl, "n": n_tot, "n_dp": n_dp,
                       "frac_dp": frac, "r_mid": r_mid, "kge_median": kge_md})

bin_df = pd.DataFrame(bin_stats)

# Color bars by whether they're above/below true r
bar_colors = ["#e76f51" if r < TRUE_VALUES["flowexp"] else "#2a9d8f"
              for r in bin_df["r_mid"]]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8),
                                      gridspec_kw={"height_ratios": [2, 1]})

# Top: double-peak fraction bars
bars = ax_top.bar(range(N_R_BINS), bin_df["frac_dp"],
                   color=bar_colors, edgecolor="white", linewidth=0.8, alpha=0.85)
for bar, row in zip(bars, bin_df.itertuples()):
    ax_top.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{row.n_dp}/{row.n}\n({100*row.frac_dp:.0f}%)",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# True r vertical reference — find which bin contains true r
true_r_bin = None
for i, row in bin_df.iterrows():
    lo, hi = r_bin_edges[i], r_bin_edges[i+1]
    if lo <= TRUE_VALUES["flowexp"] <= hi:
        true_r_bin = i
        break
if true_r_bin is not None:
    ax_top.axvline(true_r_bin, color="red", linewidth=1.8,
                   linestyle="--", alpha=0.8,
                   label=f"Bin containing true r={TRUE_VALUES['flowexp']}")

ax_top.set_xticks(range(N_R_BINS))
ax_top.set_xticklabels(bin_df["bin"], fontsize=8.5, rotation=15, ha="right")
ax_top.set_ylabel("Fraction double-peaked", fontsize=10)
ax_top.set_ylim(0, 1.0)
ax_top.set_title(
    f"Double-peak fraction by r bin — Series 92  |  n={len(df)}\n"
    f"Red = r < true value ({TRUE_VALUES['flowexp']})  |  "
    f"Teal = r ≥ true value",
    fontsize=11)
ax_top.legend(fontsize=9)
ax_top.grid(axis="y", alpha=0.3)

# Bottom: median KGE per bin
ax_bot.bar(range(N_R_BINS), bin_df["kge_median"],
           color=bar_colors, edgecolor="white", linewidth=0.8, alpha=0.75)
ax_bot.axhline(KGE_CEILING, color="gray", linewidth=1.2, linestyle=":",
               alpha=0.8, label=f"KGE ceiling = {KGE_CEILING}")
if true_r_bin is not None:
    ax_bot.axvline(true_r_bin, color="red", linewidth=1.8,
                   linestyle="--", alpha=0.8)
ax_bot.set_xticks(range(N_R_BINS))
ax_bot.set_xticklabels(bin_df["bin"], fontsize=8.5, rotation=15, ha="right")
ax_bot.set_ylabel("Median KGE", fontsize=10)
ax_bot.set_xlabel("r (flowexp) bin", fontsize=10)
ax_bot.legend(fontsize=9)
ax_bot.grid(axis="y", alpha=0.3)

fig.tight_layout()
save_fig(fig, "fig18_doublepeak_fraction_by_r_bin.png")


# =======================================================================
# CONSOLE SUMMARY
# =======================================================================
print(f"\n{'='*60}")
print("DOUBLE-PEAK HYPOTHESIS TEST — Series 92 Summary")
print(f"{'='*60}")
print(f"Total runs:          {len(df)}")
print(f"Double-peaked:       {n_double} ({100*n_double/len(df):.1f}%)")
print(f"True r value:        {TRUE_VALUES['flowexp']}")
print(f"\nDouble-peak fraction by r tercile:")
for label in ["Low r", "Mid r", "High r"]:
    sub  = df[df["r_tercile"] == label]
    n_dp = sub["double_peak"].sum()
    pct  = 100 * n_dp / max(len(sub), 1)
    rng  = tercile_info[label]["range"]
    print(f"  {label} ({rng}):  {n_dp}/{len(sub)} = {pct:.1f}%")

print(f"\nPearson r (flowexp vs double_peak_score): "
      f"{np.corrcoef(df['flowexp'], df['double_peak_score'])[0,1]:+.3f}")
print(f"Pearson r (flowexp vs double_peak flag):  "
      f"{np.corrcoef(df['flowexp'], df['double_peak'].astype(float))[0,1]:+.3f}")
print(f"\nAugmented CSV: {augmented_path.name}")
print(f"Figures:       {plot_dir}")
print(f"{'='*60}\n")
