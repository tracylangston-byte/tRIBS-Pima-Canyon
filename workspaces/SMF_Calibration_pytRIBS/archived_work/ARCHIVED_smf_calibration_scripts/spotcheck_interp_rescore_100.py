"""
spotcheck_interp_rescore_100.py
================================
Spot-checks whether the interp-resample fix (Handoff_Series100_
TruthPointAnomaly_v5.md, section 7.20 / section 8.1) behaves consistently
on NON-true-point Series 100 LHS draws, before committing to a full
re-score of the 400-run sweep.

WHAT THIS DOES
--------------
For a handful of already-completed Series 100 runs, this script:
  1. Loads the existing raw tRIBS output from disk (calibration_work/
     02_results/100_lhs_synth_Ks_f/{run_id}/) via pytRIBS's Results class.
     tRIBS is NOT re-run -- this mirrors score_notebook_run_against_truth.py's
     "score an already-executed run" pattern, generalized to loop over
     several run_ids instead of one.
  2. Re-computes metrics using the INTERP method now patched into
     run_sensitivity_single.py (both obs and sim resampled via
     `.resample('5min').interpolate(method='time')`).
  3. Compares the re-scored metrics against the ORIGINAL row for that
     run_id in lhs_results_synth_Ks_f_100.csv (which was scored under the
     old sim=.mean() bug).
  4. Prints a comparison table and a verdict, and saves the full comparison
     to CSV.

WHY THIS MATTERS
----------------
The true-point anomaly (PBIAS -6.9% to -9.6%, scaling with storm magnitude)
was resolved by this same interp fix -- but that was tested only AT the
true point, where the hydrograph is a single clean isolated peak. Ordinary
LHS draws away from the true point can have different peak shapes/timing,
so the resampling-bias direction and magnitude might not transfer 1:1.
This script checks that before trusting a full re-score (handoff section
8.3, item 6).

SELECTION STRATEGY (spot-check points, not exhaustive)
--------------------------------------------------------
Pulled from the existing results CSV, no re-running required to select:
  - best-KGE run       (top of the original ranking)
  - worst-KGE run      (bottom of the original ranking)
  - up to 2 runs inside/near the documented hang-risk zone
    (Ks 5.75-6.75x, f 0.008-0.014), if any completed runs landed there
  - the runs with min and max Ks_mult (range extremes)
  - N_RANDOM additional runs, seeded for reproducibility

Usage (run from the smf_demo directory, AFTER the patched
run_sensitivity_single.py has been confirmed correct):
    python spotcheck_interp_rescore_100.py
    python spotcheck_interp_rescore_100.py --n_random 4 --seed 7

Output:
    calibration_work/03_comparisons/summary_tables/
        spotcheck_interp_vs_mean_100.csv
    (one row per spot-checked run_id: old metrics, new metrics, deltas)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# PATHS -- same convention as run_lhs_synth_Ks_f_100.py / run_sensitivity_single.py
# -----------------------------------------------------------------------
SCRIPT_DIR   = Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "smf_demo" else SCRIPT_DIR
CALIB_DIR    = PROJECT_ROOT / "calibration_work"

LHS_CATEGORY  = "100_lhs_synth_Ks_f"
RESULTS_CSV   = CALIB_DIR / "03_comparisons" / "summary_tables" / "lhs_results_synth_Ks_f_100.csv"
RUN_INPUT_DIR = CALIB_DIR / "01_run_inputs" / LHS_CATEGORY
RUN_RESULTS_DIR_ROOT = CALIB_DIR / "02_results" / LHS_CATEGORY
OUT_CSV       = CALIB_DIR / "03_comparisons" / "summary_tables" / "spotcheck_interp_vs_mean_100.csv"

LOCATION = "SMF"
EPSG     = 26912
EVENT_START = "2014-08-12 16:00"
EVENT_END   = "2014-08-13 12:00"

# Documented tRIBS hang-risk zone, for spot-check point selection only
HANG_ZONE_KS_LO, HANG_ZONE_KS_HI = 5.75, 6.75
HANG_ZONE_F_LO,  HANG_ZONE_F_HI  = 0.008, 0.014


def select_spotcheck_run_ids(df, n_random=2, seed=42):
    """Pick a small, informative subset of run_ids to re-score, from the
    already-loaded original results CSV. No raw output is touched here --
    just selection logic on the summary table."""
    picks = {}

    picks["best_kge"] = df.loc[df["kge"].idxmax(), "run_id"]
    picks["worst_kge"] = df.loc[df["kge"].idxmin(), "run_id"]

    zone = df[
        (df["Ks_mult"]  >= HANG_ZONE_KS_LO) & (df["Ks_mult"]  <= HANG_ZONE_KS_HI) &
        (df["f_RS_abs"] >= HANG_ZONE_F_LO)  & (df["f_RS_abs"] <= HANG_ZONE_F_HI)
    ]
    for i, run_id in enumerate(zone["run_id"].head(2)):
        picks[f"hang_zone_{i+1}"] = run_id

    picks["min_ks"] = df.loc[df["Ks_mult"].idxmin(), "run_id"]
    picks["max_ks"] = df.loc[df["Ks_mult"].idxmax(), "run_id"]

    rng = np.random.default_rng(seed)
    remaining = df[~df["run_id"].isin(picks.values())]
    if len(remaining) > 0 and n_random > 0:
        n = min(n_random, len(remaining))
        random_rows = remaining.sample(n=n, random_state=rng.integers(0, 2**31 - 1))
        for i, run_id in enumerate(random_rows["run_id"]):
            picks[f"random_{i+1}"] = run_id

    # de-duplicate while keeping the first (most informative) label per run_id
    seen = {}
    for label, run_id in picks.items():
        if run_id not in seen.values():
            seen[label] = run_id
    return seen


def rescore_run_interp(run_id):
    """Re-score one already-completed run's raw tRIBS output using the
    interp method, WITHOUT re-running tRIBS. Mirrors
    score_notebook_run_against_truth.py's no-rerun loading pattern."""
    from pytRIBS.classes import Project, Results
    from run_sensitivity_single import _find_synth_truth_file

    run_results_dir = RUN_RESULTS_DIR_ROOT / run_id
    output_prefix_abs = run_results_dir / run_id
    input_file_abs = RUN_INPUT_DIR / f"{run_id}.in"
    qout_path = Path(str(output_prefix_abs) + "_Outlet.qout")

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

    synth_file = _find_synth_truth_file(CALIB_DIR, None)
    if synth_file is None:
        return None, "no synthetic truth file found (real-gauge mode unexpected here)"

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

    # INTERP METHOD -- matches the now-patched run_sensitivity_single.py:
    # both obs and sim resampled via .resample('5min').interpolate(method='time')
    obs_resampled = obs_df['Observed_CMS'].resample('5min').interpolate(method='time')
    sim_resampled = strmflw_sim['Qstrm_m3s'].resample('5min').interpolate(method='time')

    compare_df = pd.DataFrame({
        'Observed':  obs_resampled,
        'Simulated': sim_resampled,
    }).dropna()

    event_df = compare_df.loc[EVENT_START:EVENT_END].copy()
    if event_df.empty:
        return None, "event_df empty after alignment"

    obs = event_df['Observed']
    sim = event_df['Simulated']

    obs_peak, sim_peak = obs.max(), sim.max()
    peak_error_pct = (sim_peak - obs_peak) / obs_peak * 100

    r     = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    pbias = 100 * (np.sum(sim - obs) / np.sum(obs))
    # Kling et al. (2012) modified KGE, added alongside 2009 per
    # Handoff_KGE2012Transition_v1.md. gamma = alpha/beta exactly.
    gamma    = alpha / beta
    kge_2012 = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)

    return {
        "run_id":               run_id,
        "n_timesteps":          len(event_df),
        "pbias_pct_interp":     pbias,
        "kge_interp":           kge,
        "kge_r_interp":         r,
        "kge_alpha_interp":     alpha,
        "kge_beta_interp":      beta,
        "kge_gamma_interp":     gamma,
        "kge_2012_interp":      kge_2012,
        "peak_error_pct_interp": peak_error_pct,
    }, None


def main():
    parser = argparse.ArgumentParser(
        description="Spot-check the interp resampling fix on non-true-point Series 100 LHS draws."
    )
    parser.add_argument("--n_random", type=int, default=2,
                         help="Number of additional random draws to include (default 2)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for random draw selection (default 42)")
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        sys.exit(f"ERROR: {RESULTS_CSV} not found. Run from the smf_demo directory.")

    orig_df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(orig_df)} original (mean-resample) results from {RESULTS_CSV.name}")

    picks = select_spotcheck_run_ids(orig_df, n_random=args.n_random, seed=args.seed)
    print(f"\nSelected {len(picks)} spot-check points:")
    for label, run_id in picks.items():
        print(f"  {label:14s} -> {run_id}")

    rows = []
    print(f"\n{'='*100}")
    for label, run_id in picks.items():
        print(f"\n[{label}] {run_id}")
        new_metrics, err = rescore_run_interp(run_id)
        if err:
            print(f"  SKIP: {err}")
            continue

        old_row = orig_df.loc[orig_df["run_id"] == run_id].iloc[0]
        row = {
            "label":               label,
            "run_id":              run_id,
            "Ks_mult":             old_row["Ks_mult"],
            "f_RS_abs":            old_row["f_RS_abs"],
            "pbias_pct_mean":      old_row["pbias_pct"],
            "pbias_pct_interp":    new_metrics["pbias_pct_interp"],
            "delta_pbias_pct":     new_metrics["pbias_pct_interp"] - old_row["pbias_pct"],
            "kge_mean":            old_row["kge"],
            "kge_interp":          new_metrics["kge_interp"],
            "delta_kge":           new_metrics["kge_interp"] - old_row["kge"],
            "peak_error_pct_mean":   old_row["peak_error_pct"],
            "peak_error_pct_interp": new_metrics["peak_error_pct_interp"],
            "delta_peak_error_pct":  new_metrics["peak_error_pct_interp"] - old_row["peak_error_pct"],
        }
        rows.append(row)

        print(f"  PBIAS:  mean={old_row['pbias_pct']:+.4f}%   interp={new_metrics['pbias_pct_interp']:+.4f}%   "
              f"delta={row['delta_pbias_pct']:+.4f}%")
        print(f"  KGE:    mean={old_row['kge']:.4f}          interp={new_metrics['kge_interp']:.4f}          "
              f"delta={row['delta_kge']:+.4f}")
        print(f"  Peak err: mean={old_row['peak_error_pct']:+.4f}%   interp={new_metrics['peak_error_pct_interp']:+.4f}%   "
              f"delta={row['delta_peak_error_pct']:+.4f}%")

    if not rows:
        sys.exit("\nNo spot-check points could be re-scored -- check that raw outputs exist on disk.")

    result_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*100}")
    print(f"Saved: {OUT_CSV}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n=== Verdict ===")
    print(f"  n points scored:        {len(result_df)}")
    print(f"  mean delta_pbias_pct:   {result_df['delta_pbias_pct'].mean():+.4f}")
    print(f"  std  delta_pbias_pct:   {result_df['delta_pbias_pct'].std():.4f}")
    print(f"  mean delta_kge:         {result_df['delta_kge'].mean():+.4f}")
    print(f"  std  delta_kge:         {result_df['delta_kge'].std():.4f}")

    same_sign = np.sign(result_df['delta_pbias_pct']).nunique() == 1
    print(f"  delta_pbias_pct all same sign: {same_sign}")

    if result_df['delta_pbias_pct'].std() < 1.0 and same_sign:
        print("\n  --> CONSISTENT: interp fix shifts PBIAS by a similar amount/direction "
              "across these draws. A full sweep re-score is likely safe to trust directly.")
    elif same_sign:
        print("\n  --> DIRECTIONALLY CONSISTENT but variable magnitude: fix moves things the "
              "same way everywhere, but the size of the shift depends on the run. Sweep-wide "
              "SHAPE conclusions (e.g. swoosh curvature) may shift slightly on re-score even "
              "though the true point itself is now exact -- worth a visual check post-rescore.")
    else:
        print("\n  --> INCONSISTENT: sign of the shift varies across draws away from the true "
              "point. Do not assume the true-point fix generalizes uniformly -- investigate "
              "further (e.g. plot delta_pbias_pct vs peak timing/shape) before trusting a full "
              "re-score.")


if __name__ == "__main__":
    main()
