"""
run_sensitivity_sweep.py
========================
Runs the full single-parameter sensitivity sweep for all parameters.
For each value it calls build_sensitivity_run.py then run_sensitivity_single.py,
then assembles a combined results table.

All parameters accumulate in one CSV — running a new parameter sweep appends
to (or updates) sensitivity_results_all.csv rather than overwriting it. This
means you can run Ks_mult, then f_RS_abs, then f_RS_abs_Ks1, and all results
are preserved in one place for cross-parameter comparison and plotting.

Usage (run from the smf_demo directory):
    python run_sensitivity_sweep.py                         # runs all parameters
    python run_sensitivity_sweep.py --param Ks_mult         # runs one parameter only
    python run_sensitivity_sweep.py --param f_RS_abs        # runs f sweep at Ks=7x
    python run_sensitivity_sweep.py --param f_RS_abs_Ks1    # runs f sweep at Ks=1x
    python run_sensitivity_sweep.py --skip_existing         # skips runs whose CSV already exists

Output:
    calibration_work/03_comparisons/summary_tables/sensitivity_results_all.csv

Parameter notes:
    f_RS_abs      — ABSOLUTE f for RS soil only, Ks fixed at best calibrated value (7x)
    f_RS_abs_Ks1  — ABSOLUTE f for RS soil only, Ks fixed at uncalibrated baseline (1x)
                    Use to test whether the f response curve shifts with Ks.
"""

import argparse
import os
import sys
import time
import pandas as pd
from pathlib import Path

import build_sensitivity_run as builder
import run_sensitivity_single as runner

# ------------------------------------------------------------------
# SWEEP DEFINITIONS
# Edit these lists to change the values that are tested.
# ------------------------------------------------------------------
SWEEP_VALUES = {
    # Multipliers applied to baseline Ks per soil class
    "Ks_mult": [
        0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0,
        10.0, 15.0, 20.0, 30.0, 50.0, 100.0
    ],
    # PREVIOUS RUN - ABSOLUTE f for RS soil only — Ks fixed at 7x (best calibrated)
    "f_RS_abs": [
        0.001, 0.002, 0.005, 0.010, 0.020,
        0.025, 0.030, 0.035, 0.040, 0.050,
        0.060, 0.070, 0.080, 0.090, 0.100,
        0.110, 0.120, 0.150, 0.200, 0.500, 1.000
    ],
    # ABSOLUTE f for RS soil only — Ks fixed at 1x (uncalibrated baseline)
    # Focused range around the transition zone observed in f_RS_abs sweep
    "f_RS_abs_Ks1": [
        0.005, 0.010, 0.015, 0.020,
        0.025, 0.030, 0.035, 0.040,
        0.050, 0.060, 0.070, 0.080,
        0.090, 0.100, 0.110,
    ],
    # Absolute values for hillslope velocity coefficient (baseline = 3)
    "kinemvelcoef": [
        1.0, 2.0, 3.0, 5.0, 8.0, 12.0,
        17.0, 23.0, 30.0, 38.0, 47.0,
        57.0, 68.0, 80.0, 93.0, 100.0,
    ],
    # Absolute values for hillslope velocity exponent (baseline = 0.3)
    "flowexp": [
        0.05, 0.1, 0.15, 0.2, 0.25, 
        0.3, 0.35, 0.4, 0.5, 0.6,
        0.8, 1.0, 1.2, 1.5, 2.0, 3.0
    ],
    # Absolute values for Manning's channel roughness (baseline = 0.04)
    "channelroughness": [
        0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
        0.07, 0.09, 0.12, 0.15, 0.20, 0.30, 0.50, 0.80
    ],
}


def csv_already_exists(param_name, value, calib_dir):
    """Check if the compare_obs_sim CSV for this run already exists."""
    run_id, _ = builder.build_run_id(param_name, value)
    csv_path = calib_dir / "03_comparisons" / "csv_exports" / f"{run_id}_compare_obs_sim.csv"
    return csv_path.exists()


def load_existing_results(out_path):
    """Load existing sensitivity_results_all.csv if it exists, else return empty DataFrame."""
    if out_path.exists():
        try:
            df = pd.read_csv(out_path)
            print(f"  Loaded existing results: {len(df)} rows from {out_path.name}")
            return df
        except Exception as e:
            print(f"  Warning: could not load existing results ({e}). Starting fresh.")
    return pd.DataFrame()


def merge_results(existing_df, new_results, params_being_swept):
    """
    Merge new results into the existing DataFrame.
    Drops existing rows only for the parameters being swept in this run,
    then appends new results and sorts. Other parameters are untouched.
    """
    if existing_df.empty:
        merged = pd.DataFrame(new_results)
    else:
        mask = existing_df["swept_param"].isin(params_being_swept)
        kept = existing_df[~mask].copy()
        n_dropped = mask.sum()
        if n_dropped > 0:
            print(f"  Replacing {n_dropped} existing rows for: {params_being_swept}")
        merged = pd.concat([kept, pd.DataFrame(new_results)], ignore_index=True)

    merged = merged.sort_values(["swept_param", "swept_value"], ignore_index=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity sweep; results accumulate in one CSV.")
    parser.add_argument("--param", default=None,
                        choices=list(SWEEP_VALUES.keys()),
                        help="Run only this parameter (default: all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip runs whose compare CSV already exists")
    args = parser.parse_args()

    notebook_dir = Path.cwd()
    project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
    calib_dir    = project_root / "calibration_work"
    summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "sensitivity_results_all.csv"

    params_to_run = [args.param] if args.param else list(SWEEP_VALUES.keys())

    existing_df = load_existing_results(out_path)

    new_results = []
    total_runs  = sum(len(SWEEP_VALUES[p]) for p in params_to_run)
    completed   = 0
    skipped     = 0
    failed      = 0
    sweep_start = time.time()

    print(f"\n{'='*60}")
    print(f"Sensitivity sweep: {len(params_to_run)} parameter(s), {total_runs} total runs")
    print(f"{'='*60}\n")

    for param_name in params_to_run:
        values = SWEEP_VALUES[param_name]
        print(f"\n--- {param_name} ({len(values)} values) ---")
        print(f"    Baseline = {builder.BASELINE[param_name]}")
        if param_name == "f_RS_abs":
            print(f"    Mode: ABSOLUTE f for RS soil only | Ks_mult = {builder.BASELINE['Ks_mult']}x (best calibrated)")
        
        elif param_name == "f_RS_abs_Ks1":
            print(f"    Mode: ABSOLUTE f for RS soil only | Ks_mult = 1x (uncalibrated baseline)")
        print(f"    Values:   {values}\n")

        for value in values:
            run_id, _ = builder.build_run_id(param_name, value)

            if args.skip_existing and csv_already_exists(param_name, value, calib_dir):
                print(f"  SKIP (exists): {run_id}")
                skipped += 1
                metrics_file = summary_dir / f"{run_id}_metrics_summary.csv"
                if metrics_file.exists():
                    df = pd.read_csv(metrics_file)
                    new_results.append(df.iloc[0].to_dict())
                continue

            print(f"\n[{completed + 1}/{total_runs}] {param_name} = {value}")
            t0 = time.time()

            try:
                builder.build_input_file(param_name, value)
                metrics = runner.run_and_score()
                new_results.append(metrics)
                completed += 1

            except Exception as e:
                print(f"  FAILED: {run_id}")
                print(f"  Error: {e}")
                failed += 1

            elapsed = time.time() - t0
            total_elapsed = time.time() - sweep_start
            remaining = total_runs - completed - skipped - failed
            if completed > 0:
                avg_time = total_elapsed / completed
                eta_min  = (avg_time * remaining) / 60
                print(f"  Run time: {elapsed/60:.1f} min  |  ETA: {eta_min:.0f} min remaining")

    # ------------------------------------------------------------------
    # Merge and save
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Sweep complete:  {completed} ran,  {skipped} skipped,  {failed} failed")
    print(f"{'='*60}\n")

    if new_results:
        final_df = merge_results(existing_df, new_results, params_to_run)
        final_df.to_csv(out_path, index=False)
        print(f"Combined results saved to:\n  {out_path}")
        print(f"  Total rows in file: {len(final_df)}")

        param_counts = final_df.groupby("swept_param").size()
        print(f"\n  Parameters in combined CSV:")
        for p, n in param_counts.items():
            print(f"    {p}: {n} runs")

        print(f"\n  Results from this sweep:")
        cols = ["swept_param", "swept_value", "kge", "nse", "rmse_m3s",
                "pbias_pct", "peak_timing_error_hr", "sim_peak_m3s"]
        session_df = final_df[final_df["swept_param"].isin(params_to_run)]
        available  = [c for c in cols if c in session_df.columns]
        print(session_df[available].to_string(index=False, float_format="%.3f"))
    else:
        print("No new results to save.")
        if not existing_df.empty:
            print(f"  Existing CSV unchanged: {len(existing_df)} rows")


if __name__ == "__main__":
    main()