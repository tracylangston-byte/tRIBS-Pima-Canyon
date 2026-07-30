"""
plot_truth_hydrograph_comparison.py
=====================================
Diagnostic hydrograph-shape comparison for the Section 7 truth-point
validation anomaly (see "Handoff: Series 100 Storm-Magnitude Investigation
+ Truth-Point Validation Anomaly").

REPURPOSED (2026-07) from its original form, which overlaid several
synthetic-truth candidate hydrographs (S93/S94/S95/truth100) against the
real SMF gauge -- that comparison is superseded by the truth reset. This
version instead answers the open Section 7.6 question directly:

  validate_truth_point_100_storms.py builds and runs tRIBS AT the exact
  true (Ks_mult, f_RS_abs, cv, r, n) point and scores it against that
  storm's own synthetic-truth .qout. Because both "Observed" (=truth) and
  "Simulated" (=validation run) come from running the identical model at
  identical parameters and forcing, they should reproduce each other
  almost exactly (PBIAS~0%, KGE~1.0) -- but they do not (Section 7.3:
  e.g. 100_narrow scores PBIAS=-6.92%, KGE=0.9211). Every input-side
  explanation checked so far (SPOPINTRVL, soil table, mesh/met/rasters,
  binary) has been ruled out (Section 7.4). This script reads one of the
  *_compare_obs_sim.csv files written by run_sensitivity_single.py for a
  validation run and asks: WHERE in the event window does the volume
  discrepancy accumulate, and is the pattern consistent with a fixed
  timing/phase offset between the two pipelines (as opposed to a real
  shape/process difference)?

Four diagnostics, one figure:
  1. Hydrograph overlay, full event window
  2. Point-wise residual (Simulated - Observed) over time
  3. Cumulative volume-error curve (running % of final PBIAS) over time
  4. Lag-scan: RMSE and Pearson r of Simulated-shifted-by-lag vs Observed,
     across a range of +/-N min lags -- tests whether a fixed timing
     offset between the two pipelines would resolve most of the mismatch

Usage (run from smf_demo/, or pass --csv directly):
    python plot_truth_hydrograph_comparison.py
    python plot_truth_hydrograph_comparison.py --label storm080
    python plot_truth_hydrograph_comparison.py --csv /path/to/some_compare_obs_sim.csv --label custom_run

Output:
    calibration_work/03_comparisons/sensitivity_plots/truth_comparison/
        truthcheck_hydrograph_diagnostic_<label>.png
    Prints a metrics sanity-check table (compare against Section 7.3
    documented values) and a lag-scan table to stdout.
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
parser.add_argument("--label", default="100_narrow",
                     help="Validation run label (100_narrow, storm080, storm125, or any "
                          "custom string used for titles/filenames when --csv is given "
                          "explicitly). Default: 100_narrow")
parser.add_argument("--csv", default=None,
                     help="Explicit path to a *_compare_obs_sim.csv. If omitted, defaults "
                          "to the standard truthcheck path for --label.")
parser.add_argument("--outdir", default=None,
                     help="Directory to save the output figure. If omitted, uses the "
                          "standard calibration_work plot directory.")
parser.add_argument("--lag-range-min", type=int, default=30,
                     help="Max lag (minutes, each direction) to scan. Default 30.")
args = parser.parse_args()

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"

if args.csv is not None:
    csv_path = Path(args.csv)
else:
    csv_path = (calib_dir / "03_comparisons" / "csv_exports"
                / f"SMF_20140812_100_truthcheck_{args.label}_compare_obs_sim.csv")

if args.outdir is not None:
    plot_dir = Path(args.outdir)
else:
    plot_dir = calib_dir / "03_comparisons" / "sensitivity_plots" / "truth_comparison"
plot_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    sys.exit(f"ERROR: {csv_path} not found.")

# -----------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------
df  = pd.read_csv(csv_path, index_col=0, parse_dates=True)
obs = df["Observed"]
sim = df["Simulated"]
dt_min = (df.index[1] - df.index[0]).total_seconds() / 60.0

# -----------------------------------------------------------------------
# STANDARD METRICS -- exact formulas from run_sensitivity_single.py, used
# here as a sanity check that this CSV matches the documented Section 7.3
# numbers before trusting any downstream diagnostic on it.
# -----------------------------------------------------------------------
def compute_metrics(obs, sim):
    r     = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nse   = 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    pbias = 100 * (np.sum(sim - obs) / np.sum(obs))
    rmse  = np.sqrt(np.mean((sim - obs) ** 2))
    obs_peak, sim_peak   = obs.max(), sim.max()
    obs_tpeak, sim_tpeak = obs.idxmax(), sim.idxmax()
    peak_error_pct       = (sim_peak - obs_peak) / obs_peak * 100
    peak_timing_error_hr = (sim_tpeak - obs_tpeak).total_seconds() / 3600.0
    # Kling et al. (2012) modified KGE, added alongside 2009 per
    # Handoff_KGE2012Transition_v1.md. gamma = alpha/beta exactly.
    gamma    = alpha / beta
    kge_2012 = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
    return dict(r=r, alpha=alpha, beta=beta, kge=kge, nse=nse, pbias=pbias,
                rmse=rmse, obs_peak=obs_peak, sim_peak=sim_peak,
                obs_tpeak=obs_tpeak, sim_tpeak=sim_tpeak,
                peak_error_pct=peak_error_pct,
                peak_timing_error_hr=peak_timing_error_hr,
                gamma=gamma, kge_2012=kge_2012)

m = compute_metrics(obs, sim)
print(f"=== {args.label}: metrics sanity check (from {csv_path.name}) ===")
print(f"  PBIAS                 {m['pbias']:+.4f} %")
print(f"  KGE                   {m['kge']:.4f}   (r={m['r']:.4f}, alpha={m['alpha']:.4f}, beta={m['beta']:.4f})")
print(f"  KGE_2012              {m['kge_2012']:.4f}   (r={m['r']:.4f}, gamma={m['gamma']:.4f}, beta={m['beta']:.4f})")
print(f"  NSE                   {m['nse']:.4f}")
print(f"  RMSE                  {m['rmse']:.4f} m3/s")
print(f"  Peak: obs={m['obs_peak']:.3f} @ {m['obs_tpeak']}   sim={m['sim_peak']:.3f} @ {m['sim_tpeak']}")
print(f"  Peak error             {m['peak_error_pct']:+.2f} %")
print(f"  Peak timing error      {m['peak_timing_error_hr']*60:+.1f} min")
print()

# -----------------------------------------------------------------------
# WHERE DOES THE VOLUME ERROR ACCUMULATE?
# Cumulative (Sim - Obs) volume, expressed as % of the FINAL cumulative
# observed volume -- i.e. this curve's endpoint equals total PBIAS by
# construction, and its shape shows where along the event window that
# total gets built up.
# -----------------------------------------------------------------------
resid              = sim - obs
cum_resid_vol       = resid.cumsum() * dt_min * 60.0     # m3/s * sec = m3
cum_obs_vol_final    = obs.sum() * dt_min * 60.0
cum_pct_of_final_pbias = 100.0 * cum_resid_vol / cum_obs_vol_final

# -----------------------------------------------------------------------
# LAG SCAN -- does shifting Simulated in time reduce the mismatch?
# Positive lag = delay Simulated (shift later in time); tests whether
# Simulated is systematically running "ahead" of Observed.
# -----------------------------------------------------------------------
lag_steps = max(1, int(round(args.lag_range_min / dt_min)))
rows = []
for k in range(-lag_steps, lag_steps + 1):
    sim_shifted = sim.shift(k)
    valid = sim_shifted.notna() & obs.notna()
    if valid.sum() < 10:
        continue
    o, s = obs[valid], sim_shifted[valid]
    rows.append({
        "lag_min":   k * dt_min,
        "rmse":      np.sqrt(np.mean((s - o) ** 2)),
        "r":         np.corrcoef(s, o)[0, 1],
        "pbias_pct": 100 * np.sum(s - o) / np.sum(o),
    })
lag_df   = pd.DataFrame(rows)
zero_row = lag_df.loc[lag_df["lag_min"] == 0].iloc[0]
best_row = lag_df.loc[lag_df["rmse"].idxmin()]

print("=== Lag scan (shifting Simulated relative to Observed) ===")
print(lag_df.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
print(f"\n  At lag=0:      RMSE={zero_row['rmse']:.4f}  r={zero_row['r']:.4f}  PBIAS={zero_row['pbias_pct']:+.2f}%")
print(f"  Best-fit lag:  {best_row['lag_min']:+.0f} min   RMSE={best_row['rmse']:.4f}  r={best_row['r']:.4f}  PBIAS={best_row['pbias_pct']:+.2f}%")
print()

# -----------------------------------------------------------------------
# PLOT -- 4-panel diagnostic
# -----------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(11, 15))

# Panel 1: hydrograph overlay
ax = axes[0]
ax.plot(obs.index, obs.values, color="black", linewidth=2.2,
        label=f"Observed (truth) — peak {m['obs_peak']:.2f} m3/s @ {m['obs_tpeak'].strftime('%m-%d %H:%M')}")
ax.plot(sim.index, sim.values, color="#C62828", linewidth=1.8, linestyle="--",
        label=f"Simulated (validation run) — peak {m['sim_peak']:.2f} m3/s @ {m['sim_tpeak'].strftime('%m-%d %H:%M')}")
ax.set_ylabel("Discharge (m3/s)")
ax.set_title(f"{args.label}: truth-point validation — Observed vs Simulated  "
             f"(PBIAS={m['pbias']:+.2f}%, KGE={m['kge']:.4f})")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

# Panel 2: residual over time
ax = axes[1]
ax.axhline(0, color="gray", linewidth=1)
ax.plot(resid.index, resid.values, color="#1976D2", linewidth=1.3)
ax.fill_between(resid.index, resid.values, 0, color="#1976D2", alpha=0.25)
ax.set_ylabel("Simulated − Observed\n(m3/s)")
ax.set_title("Point-wise residual over time")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

# Panel 3: cumulative volume error (% of final PBIAS)
ax = axes[2]
ax.axhline(0, color="gray", linewidth=1)
ax.plot(cum_pct_of_final_pbias.index, cum_pct_of_final_pbias.values,
        color="#6A1B9A", linewidth=1.6)
ax.axhline(m["pbias"], color="#6A1B9A", linestyle=":", linewidth=1,
           label=f"Final PBIAS = {m['pbias']:+.2f}%")
ax.set_ylabel("Cumulative vol. error\n(% of final total)")
ax.set_title("Where the volume discrepancy accumulates")
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

# Panel 4: lag scan
ax  = axes[3]
ax2 = ax.twinx()
l1, = ax.plot(lag_df["lag_min"], lag_df["rmse"], color="#E65100", marker="o",
              markersize=3.5, label="RMSE")
l2, = ax2.plot(lag_df["lag_min"], lag_df["r"], color="#2E7D32", marker="s",
               markersize=3.5, label="Pearson r")
ax.axvline(0, color="gray", linewidth=1)
ax.axvline(best_row["lag_min"], color="#E65100", linestyle=":", linewidth=1.3)
ax.set_xlabel("Lag applied to Simulated (min); positive = Simulated delayed")
ax.set_ylabel("RMSE (m3/s)", color="#E65100")
ax2.set_ylabel("Pearson r", color="#2E7D32")
ax.set_title(f"Lag scan — best fit at lag = {best_row['lag_min']:+.0f} min "
             f"(RMSE {zero_row['rmse']:.3f} → {best_row['rmse']:.3f})")
ax.legend(handles=[l1, l2], fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

fig.tight_layout()
out_path = plot_dir / f"truthcheck_hydrograph_diagnostic_{args.label}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)
