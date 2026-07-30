"""
compare_interp_vs_mean.py
==========================
A/B comparison for the resampling-method hypothesis (see
Handoff_Series100_TruthPointAnomaly_v4.md, Sections 7.7 and 8):

  MEAN:   truth_point_validation_100_storms.csv        (validate_truth_point_100_storms.py,
                                                          sim resampled via .resample('5min').mean())
  INTERP: truth_point_validation_100_storms_interp.csv  (validate_truth_point_100_storms_v2.py,
                                                          sim resampled via .resample('5min').interpolate('time'))

Both score tRIBS run AT the exact true (Ks_mult=7.0x, f_RS_abs=0.012,
cv=4.5, r=0.24, n=0.026) point, under each of the three storm forcings,
against that storm's own synthetic truth. Section 7.7 localized the
entire truth-point PBIAS residual to a ~45-minute window at and
immediately after the peak, with everything else (15+ hours of recession
and low-flow tail) matching almost perfectly -- exactly the signature
you'd expect from a resampling method that smooths/bins across a fast
transition. This script settles whether that hypothesis holds:

  1. STORM-SUMMARY COMPARISON (always runs): merges the two 3-storm
     summary CSVs on storm_label and reports, per storm, how much PBIAS/
     KGE/peak_error_pct moved when switching resample methods, plus a
     plain-language verdict.

  2. PER-STORM TIMESTEP DIAGNOSTIC (runs if the underlying
     *_compare_obs_sim.csv files are present for a given storm): overlays
     Observed / Simulated-mean / Simulated-interp, plots the point-wise
     (interp - mean) residual, and compares cumulative-volume-error
     curves for both methods against Observed -- with the peak window
     (obs peak +/- --peak-window-min minutes) shaded on both time panels
     so the divergence can be read directly against where Section 7.7
     located the mismatch.

Neither the mean nor the interp CSVs are modified; this script only
reads and compares them.

Usage (run from the smf_demo directory, after both validate_truth_point_
100_storms.py and validate_truth_point_100_storms_v2.py have been run):
    python compare_interp_vs_mean.py
    python compare_interp_vs_mean.py --labels storm080,100_narrow
    python compare_interp_vs_mean.py --no-per-storm-detail
    python compare_interp_vs_mean.py --peak-window-min 45

Output:
    calibration_work/03_comparisons/summary_tables/
        interp_vs_mean_comparison_100_storms.csv
    calibration_work/03_comparisons/sensitivity_plots/truth_comparison/
        interp_vs_mean_diagnostic_<label>.png   (one per storm, if that
        storm's per-timestep CSVs are found on both sides)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mean-csv", default=None,
                     help="Explicit path to the mean-aggregation summary CSV. "
                          "Default: calibration_work/03_comparisons/summary_tables/"
                          "truth_point_validation_100_storms.csv")
parser.add_argument("--interp-csv", default=None,
                     help="Explicit path to the interp-resample summary CSV. "
                          "Default: calibration_work/03_comparisons/summary_tables/"
                          "truth_point_validation_100_storms_interp.csv")
parser.add_argument("--labels", default="storm080,100_narrow,storm125",
                     help="Comma-separated storm labels to attempt per-storm "
                          "timestep diagnostics for. Default: all three.")
parser.add_argument("--per-storm-detail", dest="per_storm_detail",
                     action="store_true", default=True,
                     help="Produce the per-storm timestep diagnostic plots "
                          "(default: on).")
parser.add_argument("--no-per-storm-detail", dest="per_storm_detail",
                     action="store_false",
                     help="Skip the per-storm timestep diagnostic plots; "
                          "storm-summary comparison only.")
parser.add_argument("--peak-window-min", type=float, default=30.0,
                     help="Half-width (minutes) of the shaded peak window on "
                          "the per-storm diagnostic plots, centered on each "
                          "storm's own Observed peak time. Default: 30.")
parser.add_argument("--outdir", default=None,
                     help="Directory to save per-storm diagnostic figures. "
                          "Default: calibration_work/03_comparisons/"
                          "sensitivity_plots/truth_comparison/")
args = parser.parse_args()

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
csv_dir      = calib_dir / "03_comparisons" / "csv_exports"

mean_csv_path   = Path(args.mean_csv)   if args.mean_csv   else \
    summary_dir / "truth_point_validation_100_storms.csv"
interp_csv_path = Path(args.interp_csv) if args.interp_csv else \
    summary_dir / "truth_point_validation_100_storms_interp.csv"

plot_dir = Path(args.outdir) if args.outdir else \
    calib_dir / "03_comparisons" / "sensitivity_plots" / "truth_comparison"

for p, tag in [(mean_csv_path, "mean-aggregation"), (interp_csv_path, "interp-resample")]:
    if not p.exists():
        sys.exit(f"ERROR: {tag} summary CSV not found: {p}\n"
                  f"Run validate_truth_point_100_storms.py (mean) and "
                  f"validate_truth_point_100_storms_v2.py (interp) first.")

# -----------------------------------------------------------------------
# 1. STORM-SUMMARY COMPARISON
# -----------------------------------------------------------------------
mean_df   = pd.read_csv(mean_csv_path)
interp_df = pd.read_csv(interp_csv_path)

print(f"Mean-aggregation summary:  {mean_csv_path.name}  ({len(mean_df)} storms)")
print(f"Interp-resample summary:   {interp_csv_path.name}  ({len(interp_df)} storms)\n")

missing_mean   = set(interp_df["storm_label"]) - set(mean_df["storm_label"])
missing_interp = set(mean_df["storm_label"])   - set(interp_df["storm_label"])
if missing_mean:
    print(f"WARNING: storms in interp CSV but not mean CSV: {sorted(missing_mean)}")
if missing_interp:
    print(f"WARNING: storms in mean CSV but not interp CSV: {sorted(missing_interp)}")

merge_cols = ["storm_label", "storm_scale", "pbias_pct", "kge", "kge_r",
              "kge_alpha", "kge_beta", "nse", "peak_error_pct", "volume_error_pct"]
merge_cols = [c for c in merge_cols if c in mean_df.columns and c in interp_df.columns]

merged = mean_df[merge_cols].merge(
    interp_df[merge_cols], on=["storm_label", "storm_scale"],
    suffixes=("_mean", "_interp"), how="inner",
)

merged["d_pbias_pct"]      = merged["pbias_pct_interp"]      - merged["pbias_pct_mean"]
merged["d_kge"]            = merged["kge_interp"]            - merged["kge_mean"]
merged["d_peak_error_pct"] = merged["peak_error_pct_interp"] - merged["peak_error_pct_mean"]
merged["d_volume_error_pct"] = merged["volume_error_pct_interp"] - merged["volume_error_pct_mean"]

# How much closer to zero did PBIAS get? 100% = fully closed, 0% = no
# change, negative = interp made it WORSE. Guards div-by-zero (shouldn't
# happen -- PBIAS at this anomaly is never exactly 0 in the mean version).
merged["pct_pbias_anomaly_closed"] = 100.0 * (
    1.0 - merged["pbias_pct_interp"].abs() / merged["pbias_pct_mean"].abs()
)

out_cols = ["storm_label", "storm_scale",
            "pbias_pct_mean", "pbias_pct_interp", "d_pbias_pct", "pct_pbias_anomaly_closed",
            "kge_mean", "kge_interp", "d_kge",
            "peak_error_pct_mean", "peak_error_pct_interp", "d_peak_error_pct",
            "volume_error_pct_mean", "volume_error_pct_interp", "d_volume_error_pct"]
out_cols = [c for c in out_cols if c in merged.columns]

print("=== STORM-SUMMARY COMPARISON: interp vs. mean resample, at the true point ===")
print(merged[out_cols].round(4).to_string(index=False))
print()

out_path = summary_dir / "interp_vs_mean_comparison_100_storms.csv"
merged[out_cols].to_csv(out_path, index=False)
print(f"Saved: {out_path}\n")

# ---- verdict ----
closed_vals = merged["pct_pbias_anomaly_closed"]
if (closed_vals > 50).all():
    verdict = ("SUPPORTS the resample-artifact hypothesis: PBIAS moved more than "
               "halfway to zero for every storm under time-interpolation. Worth "
               "re-scoring completed series (97, 97log, 99, 100, storm080/100_narrow/"
               "125) with the interp method before treating any of their displacement "
               "findings as real equifinality.")
elif (closed_vals > 50).any():
    verdict = ("MIXED: PBIAS moved substantially closer to zero for some storms but "
               "not others. The resample method may be a partial contributor, not the "
               "sole explanation -- still worth discussing with Josh alongside the "
               "remaining equifinality / solver-behavior hypotheses.")
elif (closed_vals < -10).any():
    verdict = ("RULED OUT, and inconsistent: interp resampling made PBIAS worse for "
               "at least one storm. The peak-window volume residual is not primarily "
               "a resampling artifact.")
else:
    verdict = ("RULED OUT: PBIAS is essentially unchanged (or only marginally "
               "improved) under time-interpolation for all three storms. The "
               "resample-method hypothesis does not explain the Section 7 anomaly -- "
               "the two remaining live hypotheses (real KGE-formula equifinality, or "
               "a tRIBS-internal behavior near the peak invisible at standard logging "
               "verbosity) stand as-is.")

print("=== VERDICT ===")
print(f"  {verdict}\n")

if not args.per_storm_detail:
    sys.exit(0)

# -----------------------------------------------------------------------
# 2. PER-STORM TIMESTEP DIAGNOSTIC
# -----------------------------------------------------------------------
def compute_metrics(obs, sim):
    r     = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    pbias = 100 * (np.sum(sim - obs) / np.sum(obs))
    # Kling et al. (2012) modified KGE, added alongside 2009 per
    # Handoff_KGE2012Transition_v1.md. gamma = alpha/beta exactly.
    gamma    = alpha / beta
    kge_2012 = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
    return dict(r=r, alpha=alpha, beta=beta, kge=kge, pbias=pbias,
                gamma=gamma, kge_2012=kge_2012)


labels = [s.strip() for s in args.labels.split(",") if s.strip()]
plot_dir.mkdir(parents=True, exist_ok=True)

for label in labels:
    mean_ts_path   = csv_dir / f"SMF_20140812_100_truthcheck_{label}_compare_obs_sim.csv"
    interp_ts_path = csv_dir / f"SMF_20140812_100_truthcheck_interp_{label}_compare_obs_sim.csv"

    missing = [p for p in (mean_ts_path, interp_ts_path) if not p.exists()]
    if missing:
        print(f"--- {label}: SKIPPING per-storm detail (missing "
              f"{', '.join(p.name for p in missing)}) ---\n")
        continue

    mean_ts   = pd.read_csv(mean_ts_path,   index_col=0, parse_dates=True)
    interp_ts = pd.read_csv(interp_ts_path, index_col=0, parse_dates=True)

    if len(mean_ts) != len(interp_ts) or not mean_ts.index.equals(interp_ts.index):
        print(f"--- {label}: WARNING -- mean and interp timestep CSVs do not share "
              f"an identical timestamp index. Aligning on the intersection. ---")
        common = mean_ts.index.intersection(interp_ts.index)
        mean_ts, interp_ts = mean_ts.loc[common], interp_ts.loc[common]

    obs        = mean_ts["Observed"]          # same truth file both sides
    sim_mean   = mean_ts["Simulated"]
    sim_interp = interp_ts["Simulated"]
    dt_min     = (obs.index[1] - obs.index[0]).total_seconds() / 60.0

    obs_vs_interp = (mean_ts["Observed"] - interp_ts["Observed"]).abs()
    if obs_vs_interp.max() > 1e-6:
        print(f"--- {label}: WARNING -- Observed columns differ between the mean and "
              f"interp CSVs (max abs diff {obs_vs_interp.max():.2e} m3/s). Unexpected, "
              f"since both should read the identical truth .qout; investigate before "
              f"trusting this comparison. ---")

    m_mean   = compute_metrics(obs, sim_mean)
    m_interp = compute_metrics(obs, sim_interp)

    print(f"--- {label}: per-storm timestep comparison ---")
    print(f"  MEAN:    PBIAS={m_mean['pbias']:+.4f}%  KGE={m_mean['kge']:.4f}  "
          f"peak={sim_mean.max():.3f} m3/s @ {sim_mean.idxmax()}")
    print(f"  INTERP:  PBIAS={m_interp['pbias']:+.4f}%  KGE={m_interp['kge']:.4f}  "
          f"peak={sim_interp.max():.3f} m3/s @ {sim_interp.idxmax()}")
    print(f"  Observed peak: {obs.max():.3f} m3/s @ {obs.idxmax()}")

    resid = sim_interp - sim_mean
    print(f"  (interp - mean) residual: max abs={resid.abs().max():.4f} m3/s at "
          f"{resid.abs().idxmax()}, mean abs={resid.abs().mean():.4f} m3/s\n")

    # ---- cumulative volume-error curves, each method vs. Observed ----
    cum_obs_vol_final = obs.sum() * dt_min * 60.0
    cum_pct_mean   = 100.0 * (sim_mean   - obs).cumsum() * dt_min * 60.0 / cum_obs_vol_final
    cum_pct_interp = 100.0 * (sim_interp - obs).cumsum() * dt_min * 60.0 / cum_obs_vol_final

    # ---- peak window shading, centered on Observed peak ----
    obs_tpeak = obs.idxmax()
    win_start = obs_tpeak - pd.Timedelta(minutes=args.peak_window_min)
    win_end   = obs_tpeak + pd.Timedelta(minutes=args.peak_window_min)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))

    # Panel 1: hydrograph overlay
    ax = axes[0]
    ax.axvspan(win_start, win_end, color="gray", alpha=0.15,
               label=f"peak window (obs peak \u00b1{args.peak_window_min:.0f} min)")
    ax.plot(obs.index, obs.values, color="black", linewidth=2.0, label="Observed (truth)")
    ax.plot(sim_mean.index, sim_mean.values, color="#1976D2", linewidth=1.6,
            linestyle="--", label=f"Simulated -- MEAN resample (PBIAS={m_mean['pbias']:+.2f}%)")
    ax.plot(sim_interp.index, sim_interp.values, color="#E65100", linewidth=1.6,
            linestyle="-.", label=f"Simulated -- INTERP resample (PBIAS={m_interp['pbias']:+.2f}%)")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(f"{label}: true-point validation -- mean vs. interp resample")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # Panel 2: point-wise residual between the two resample methods
    ax = axes[1]
    ax.axvspan(win_start, win_end, color="gray", alpha=0.15)
    ax.axhline(0, color="gray", linewidth=1)
    ax.plot(resid.index, resid.values, color="#6A1B9A", linewidth=1.3)
    ax.fill_between(resid.index, resid.values, 0, color="#6A1B9A", alpha=0.25)
    ax.set_ylabel("INTERP \u2212 MEAN\nSimulated (m3/s)")
    ax.set_title("Where the two resample methods diverge")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # Panel 3: cumulative volume-error comparison
    ax = axes[2]
    ax.axvspan(win_start, win_end, color="gray", alpha=0.15)
    ax.axhline(0, color="gray", linewidth=1)
    ax.plot(cum_pct_mean.index, cum_pct_mean.values, color="#1976D2", linewidth=1.6,
            linestyle="--", label=f"MEAN (final = {m_mean['pbias']:+.2f}%)")
    ax.plot(cum_pct_interp.index, cum_pct_interp.values, color="#E65100", linewidth=1.6,
            linestyle="-.", label=f"INTERP (final = {m_interp['pbias']:+.2f}%)")
    ax.set_ylabel("Cumulative vol. error\n(% of final Observed total)")
    ax.set_title("Where each method's volume discrepancy accumulates")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.tight_layout()
    out_fig = plot_dir / f"interp_vs_mean_diagnostic_{label}.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_fig}\n")
    plt.close(fig)
