"""
rescore_series100_family_interp.py
====================================
Full re-score of the Series 100 baseline sweep AND its three storm-
magnitude siblings (storm080, 100_narrow, storm125), using the interp
resampling fix now patched into run_sensitivity_single.py -- WITHOUT
re-running tRIBS.

Context: Handoff_Series100_TruthPointAnomaly_v5.md, section 8.1 step 2
("decide with Josh which completed series need re-scoring" -- for now,
scope is Series 100 + its storm siblings; 97/97log/99 are TBD) and step 3
("re-run Section 6's storm-magnitude sweep comparison ... once the LHS
sweeps are re-scored"). Also closes the loop opened by
spotcheck_interp_rescore_100.py, which confirmed the interp fix shifts
PBIAS in a consistent direction (but variable magnitude, correlated with
Ks/f position) across non-true-point draws -- meaning a full re-score
changes rank order, not just reported numbers, and is necessary before
trusting any "top cluster near truth" or swoosh-shape finding.

WHAT THIS DOES
--------------
For every run_id in a series' existing results CSV:
  1. Loads the already-completed raw tRIBS output from disk (pytRIBS
     Project/Results, get_qout_results()) -- tRIBS is NOT re-run.
  2. Loads the correct truth file for that series (baseline truth100 for
     "100"/"narrow"; the storm-specific truth subdirectory for
     storm080/storm125), via the same _find_synth_truth_file() used by
     run_sensitivity_single.py.
  3. Resamples BOTH obs and sim via .resample('5min').interpolate(method=
     'time') -- the patched method -- and re-computes every metric in the
     standard metrics_summary schema, using the *same* helper function
     (_compute_phase_metrics) imported directly from the now-patched
     run_sensitivity_single.py, so phase-specific metrics (first arrival,
     rising limb, recession ratio, etc.) are recomputed with the identical
     formulas, not just PBIAS/KGE.
  4. All non-obs/sim-derived columns (Ks_mult, f_RS_abs, kinemvelcoef,
     flowexp, channelroughness, channelwidthcoeff, optpercolation,
     channelconductivity_mmhr, channelporosity, As_value, Au_value,
     thetaS_mult, swept_param, swept_value) are carried through unchanged
     from the original CSV row -- these are run *inputs*, unaffected by
     the resampling bug.
  5. Writes a new "_RESCORED" CSV per series, full original schema plus
     audit columns (old mean-method PBIAS/KGE side by side, deltas, and a
     per-row status flag), sorted by kge descending like the originals.
     Saves incrementally every CHECKPOINT_EVERY rows so a long run isn't
     lost if interrupted.
  6. Prints a rank-stability summary: Spearman correlation of old vs new
     KGE ranking, and how much the top-20 (by KGE) set changed.

SCOPE (per 2026-07-21 decision with Josh)
------------------------------------------
Series 97, 97log, 99 are explicitly NOT covered here (raw outputs for
those are permanently deleted -- re-scoring is impossible without a full
re-run, which is a separate, TBD decision). This script covers only:
  - "100"       Series 100 baseline (n=400, full bounds)
  - "storm080"  storm-magnitude sibling, 0.8x rain (n=200, narrowed bounds)
  - "narrow"    storm-magnitude sibling, 1.0x rain (n=200, narrowed bounds)
  - "storm125"  storm-magnitude sibling, 1.25x rain (n=200, narrowed bounds)

Usage (run from the smf_demo directory, AFTER confirming
run_sensitivity_single.py's patch, e.g. via spotcheck_interp_rescore_100.py):
    python rescore_series100_family_interp.py --series 100
    python rescore_series100_family_interp.py --series storm080
    python rescore_series100_family_interp.py --series all
    python rescore_series100_family_interp.py --series 100 --limit 10   # test on a subset first

Output (per series, in calibration_work/03_comparisons/summary_tables/):
    lhs_results_synth_Ks_f_100_RESCORED.csv
    lhs_results_synth_Ks_f_100_storm080_RESCORED.csv
    lhs_results_synth_Ks_f_100_narrow_RESCORED.csv
    lhs_results_synth_Ks_f_100_storm125_RESCORED.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
SCRIPT_DIR   = Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "smf_demo" else SCRIPT_DIR
CALIB_DIR    = PROJECT_ROOT / "calibration_work"
SUMMARY_DIR  = CALIB_DIR / "03_comparisons" / "summary_tables"

LOCATION = "SMF"
EPSG     = 26912

CHECKPOINT_EVERY = 25   # rows between incremental saves

# -----------------------------------------------------------------------
# SERIES CONFIG -- one entry per series this script covers. category and
# truth_override are pulled directly from the corresponding
# run_lhs_synth_Ks_f_100*.py sweep script's own constants.
# -----------------------------------------------------------------------
SERIES_CONFIG = {
    "100": {
        "category":       "100_lhs_synth_Ks_f",
        "results_csv":    "lhs_results_synth_Ks_f_100.csv",
        "truth_override": None,
        "n_expected":     400,
    },
    "storm080": {
        "category":       "100_storm080_lhs_synth_Ks_f",
        "results_csv":    "lhs_results_synth_Ks_f_100_storm080.csv",
        "truth_override": "synth_truth/storm080",
        "n_expected":     200,
    },
    "narrow": {
        "category":       "100_narrow_lhs_synth_Ks_f",
        "results_csv":    "lhs_results_synth_Ks_f_100_narrow.csv",
        "truth_override": None,   # reuses baseline truth100, same as Series 100
        "n_expected":     200,
    },
    "storm125": {
        "category":       "100_storm125_lhs_synth_Ks_f",
        "results_csv":    "lhs_results_synth_Ks_f_100_storm125.csv",
        "truth_override": "synth_truth/storm125",
        "n_expected":     200,
    },
}

# Columns carried through unchanged from the original row -- run inputs,
# not affected by the resampling bug.
PASSTHROUGH_COLS = [
    "Ks_mult", "f_RS_abs", "As_value", "Au_value", "thetaS_mult",
    "optpercolation", "channelconductivity_mmhr", "channelporosity",
    "kinemvelcoef", "flowexp", "channelroughness", "channelwidthcoeff",
    "swept_param", "swept_value", "obs_mode",
]


def rescore_one_run(run_id, category, truth_override, event_start, event_end):
    """Re-score one already-completed run's raw tRIBS output using the
    interp method, WITHOUT re-running tRIBS. Returns (metrics_dict, None)
    on success or (None, error_string) on failure."""
    from pytRIBS.classes import Project, Results
    from run_sensitivity_single import _compute_phase_metrics, _find_synth_truth_file

    run_results_dir    = CALIB_DIR / "02_results" / category / run_id
    output_prefix_abs  = run_results_dir / run_id
    input_file_abs      = CALIB_DIR / "01_run_inputs" / category / f"{run_id}.in"
    qout_path           = Path(str(output_prefix_abs) + "_Outlet.qout")

    if not input_file_abs.exists():
        return None, f"input file not found: {input_file_abs}"
    if not qout_path.exists():
        return None, f"raw output not found: {qout_path}"

    input_file = os.path.relpath(input_file_abs, SCRIPT_DIR)

    proj = Project(os.getcwd(), LOCATION, EPSG)
    results = Results(input_file, meta=proj.meta)
    results.get_mrf_results()
    results.get_element_results()
    strmflw_sim_raw = results.get_qout_results()

    synth_file = _find_synth_truth_file(CALIB_DIR, truth_override)
    if synth_file is None:
        return None, "no synthetic truth file found"

    obs_raw = pd.read_csv(
        synth_file, sep=r'\s+', skiprows=1,
        names=['Time_hr', 'Qstrm_m3s', 'Hlev_m']
    )
    obs_raw['datetime'] = pd.to_datetime(
        obs_raw['Time_hr'] * 3600, unit='s', origin=pd.Timestamp('2014-08-01')
    )
    obs_raw.set_index('datetime', inplace=True)
    obs_raw['Observed_CMS'] = obs_raw['Qstrm_m3s']
    obs_df = obs_raw

    strmflw_sim = strmflw_sim_raw.copy()
    strmflw_sim['Time'] = pd.to_datetime(strmflw_sim['Time'])
    strmflw_sim.set_index('Time', inplace=True)

    # INTERP METHOD -- matches the now-patched run_sensitivity_single.py
    obs_resampled = obs_df['Observed_CMS'].resample('5min').interpolate(method='time')
    sim_resampled = strmflw_sim['Qstrm_m3s'].resample('5min').interpolate(method='time')

    compare_df = pd.DataFrame({
        'Observed':  obs_resampled,
        'Simulated': sim_resampled,
    }).dropna()

    event_df = compare_df.loc[event_start:event_end].copy()
    if event_df.empty:
        return None, "event_df empty after alignment"

    obs = event_df['Observed']
    sim = event_df['Simulated']

    obs_peak, sim_peak   = obs.max(), sim.max()
    obs_tpeak, sim_tpeak = obs.idxmax(), sim.idxmax()

    dt_seconds    = (event_df.index[1] - event_df.index[0]).total_seconds()
    obs_vol_m3    = obs.sum() * dt_seconds
    sim_vol_m3    = sim.sum() * dt_seconds
    vol_error_pct = ((sim_vol_m3 - obs_vol_m3) / obs_vol_m3) * 100

    peak_error_m3s = sim_peak - obs_peak
    peak_error_pct = (peak_error_m3s / obs_peak) * 100

    rmse  = np.sqrt(np.mean((sim - obs) ** 2))
    nse   = 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    pbias = 100 * (np.sum(sim - obs) / np.sum(obs))

    r     = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim)  / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # Kling et al. (2012) modified KGE: gamma (CV ratio) replaces alpha to
    # decouple "shape" from bias. gamma = alpha/beta exactly. Added
    # alongside the 2009 formula per Handoff_KGE2012Transition_v1.md.
    gamma    = alpha / beta
    kge_2012 = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)

    phase_metrics = _compute_phase_metrics(obs, sim)

    metrics = {
        "obs_peak_m3s":        obs_peak,
        "sim_peak_m3s":        sim_peak,
        "peak_error_m3s":      peak_error_m3s,
        "peak_error_pct":      peak_error_pct,
        "obs_peak_time":       str(obs_tpeak),
        "sim_peak_time":       str(sim_tpeak),
        "peak_timing_error_hr": (sim_tpeak - obs_tpeak).total_seconds() / 3600,
        "obs_volume_m3":       obs_vol_m3,
        "sim_volume_m3":       sim_vol_m3,
        "volume_error_pct":    vol_error_pct,
        "rmse_m3s":            rmse,
        "nse":                 nse,
        "pbias_pct":           pbias,
        "kge":                 kge,
        "kge_r":               r,
        "kge_alpha":           alpha,
        "kge_beta":            beta,
        "kge_gamma":           gamma,
        "kge_2012":            kge_2012,
        "threshold_m3s":                   phase_metrics["threshold_m3s"],
        "first_arrival_error_min":         phase_metrics["first_arrival_error_min"],
        "rising_limb_steepness_ratio":     phase_metrics["rising_limb_steepness_ratio"],
        "time_to_peak_from_exc_min":       phase_metrics["time_to_peak_from_exc_min"],
        "duration_above_thresh_error_min": phase_metrics["duration_above_thresh_error_min"],
        "recession_rate_ratio":            phase_metrics["recession_rate_ratio"],
    }
    return metrics, None


def rescore_series(series_key, limit=None):
    cfg = SERIES_CONFIG[series_key]
    csv_path = SUMMARY_DIR / cfg["results_csv"]
    out_path = SUMMARY_DIR / cfg["results_csv"].replace(".csv", "_RESCORED.csv")

    if not csv_path.exists():
        print(f"  SKIP series '{series_key}': {csv_path} not found.")
        return

    orig_df = pd.read_csv(csv_path)
    if limit:
        orig_df = orig_df.head(limit)

    print(f"\n{'='*80}")
    print(f"Series '{series_key}'  ({len(orig_df)} rows, expected ~{cfg['n_expected']})")
    print(f"  results CSV:  {csv_path.name}")
    print(f"  category:     {cfg['category']}")
    print(f"  truth override: {cfg['truth_override']}")
    print(f"{'='*80}")

    new_rows  = []
    n_ok, n_skip = 0, 0
    t0 = time.time()

    for i, (_, row) in enumerate(orig_df.iterrows()):
        run_id = row["run_id"]
        event_start = row.get("event_start", "2014-08-12 16:00")
        event_end   = row.get("event_end",   "2014-08-13 12:00")

        metrics, err = rescore_one_run(
            run_id, cfg["category"], cfg["truth_override"], event_start, event_end
        )

        new_row = {"run_id": run_id}
        for col in PASSTHROUGH_COLS:
            if col in row:
                new_row[col] = row[col]
        new_row["event_start"] = event_start
        new_row["event_end"]   = event_end

        if err:
            new_row["rescore_status"] = f"SKIPPED: {err}"
            n_skip += 1
        else:
            new_row.update(metrics)
            new_row["rescore_status"]  = "OK"
            new_row["pbias_pct_meanmethod"] = row.get("pbias_pct", np.nan)
            new_row["kge_meanmethod"]       = row.get("kge", np.nan)
            new_row["peak_error_pct_meanmethod"] = row.get("peak_error_pct", np.nan)
            new_row["delta_pbias_pct"] = metrics["pbias_pct"] - row.get("pbias_pct", np.nan)
            new_row["delta_kge"]       = metrics["kge"]       - row.get("kge", np.nan)
            n_ok += 1

        new_rows.append(new_row)

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(orig_df):
            pd.DataFrame(new_rows).to_csv(out_path, index=False)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = ((len(orig_df) - (i + 1)) / rate / 60) if rate > 0 else 0
            print(f"  [{i+1:>4}/{len(orig_df)}]  ok={n_ok}  skipped={n_skip}  "
                  f"ETA {eta_min:.1f} min  (checkpoint saved)")

    result_df = pd.DataFrame(new_rows)
    if "kge" in result_df.columns:
        result_df = result_df.sort_values("kge", ascending=False, na_position="last")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(result_df)} rows, {n_ok} rescored, {n_skip} skipped)")

    # ------------------------------------------------------------------
    # Rank-stability summary
    # ------------------------------------------------------------------
    scored = result_df[result_df["rescore_status"] == "OK"].copy()
    merged = scored.merge(
        orig_df[["run_id", "kge"]].rename(columns={"kge": "kge_orig"}),
        on="run_id", how="left"
    )
    if len(merged) >= 2:
        spearman_r = merged["kge"].rank().corr(merged["kge_orig"].rank(), method="pearson")
        old_top20 = set(orig_df.sort_values("kge", ascending=False).head(20)["run_id"])
        new_top20 = set(merged.sort_values("kge", ascending=False).head(20)["run_id"])
        overlap = len(old_top20 & new_top20)
        print(f"\n  Rank stability (this series):")
        print(f"    Spearman rank corr (old KGE vs new KGE): {spearman_r:.4f}")
        print(f"    Top-20 by KGE overlap: {overlap}/20 runs unchanged")
        print(f"    mean delta_pbias_pct: {merged['pbias_pct'].sub(merged['pbias_pct_meanmethod']).mean():+.4f}")
        print(f"    mean delta_kge:       {merged['kge'].sub(merged['kge_meanmethod']).mean():+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Full re-score of Series 100 + storm siblings with the interp fix (no tRIBS re-run)."
    )
    parser.add_argument("--series", choices=list(SERIES_CONFIG.keys()) + ["all"],
                         default="all", help="Which series to re-score (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only re-score the first N rows (for testing)")
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    series_list = list(SERIES_CONFIG.keys()) if args.series == "all" else [args.series]
    for series_key in series_list:
        rescore_series(series_key, limit=args.limit)

    print(f"\n{'='*80}")
    print("Done. Next: re-run plot_storm_magnitude_comparison_100.py and the Ks_f plotting")
    print("scripts pointed at the *_RESCORED.csv files (handoff section 8.1 step 3).")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
