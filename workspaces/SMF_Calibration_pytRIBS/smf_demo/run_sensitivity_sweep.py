"""
run_sensitivity_sweep.py
========================
Runs the full single-parameter sensitivity sweep for all parameters.
For each value it calls build_sensitivity_run.py then run_sensitivity_single.py,
then assembles a combined results table.

All parameters accumulate in one CSV — running a new parameter sweep appends
to (or updates) sensitivity_results_all.csv rather than overwriting it. This
means you can run Ks_mult, then f_RS_abs, then thetaS_mult, and all results
are preserved in one place for cross-parameter comparison and plotting.

Usage (run from the smf_demo directory):
    python run_sensitivity_sweep.py                              # runs all parameters
    python run_sensitivity_sweep.py --param Ks_mult             # runs one parameter only
    python run_sensitivity_sweep.py --param thetaS_mult         # runs thetaS sweep
    python run_sensitivity_sweep.py --param channelwidthcoeff   # runs channel width sweep
    python run_sensitivity_sweep.py --param psiB_mult           # runs psiB sweep
    python run_sensitivity_sweep.py --param As_value            # runs saturated anisotropy sweep
    python run_sensitivity_sweep.py --param Au_value            # runs unsaturated anisotropy sweep
    python run_sensitivity_sweep.py --param AsAu_value          # runs combined As=Au sweep
    python run_sensitivity_sweep.py --skip_existing             # skips runs whose CSV already exists

Output:
    calibration_work/03_comparisons/summary_tables/sensitivity_results_all.csv

Parameter notes:
    f_RS_abs         — ABSOLUTE f for RS soil only, Ks fixed at best calibrated value (6.1x)
    thetaS_mult      — multiplier applied uniformly to all soil class thetaS values
    channelwidthcoeff— absolute channel width coefficient (hydraulic geometry scaling)
    psiB_mult        — multiplier applied uniformly to all soil class PsiB values (series 59)
    As_value         — absolute saturated anisotropy ratio, applied to all soil classes (Au held at 1.0)
    Au_value         — absolute unsaturated anisotropy ratio, applied to all soil classes (As held at 1.0)
    AsAu_value       — both As and Au set to the same swept value simultaneously
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
    # ABSOLUTE f for RS soil only — Ks fixed at best calibrated value (6.1x)
    "f_RS_abs": [
        0.001, 0.002, 0.005, 0.010, 0.020,
        0.025, 0.030, 0.035, 0.040, 0.050,
        0.060, 0.070, 0.080, 0.090, 0.100,
        0.110, 0.120, 0.150, 0.200, 0.500, 1.000
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
        0.001, 0.002, 0.003, 0.005,
        0.0075, 0.01, 0.015, 0.02,
        0.03, 0.04, 0.05, 0.07,
        0.10, 0.15, 0.20
    ],
    # Absolute values for channel width coefficient (baseline = 2.33)
    # Range spans from very narrow (0.25) through calibrated (2.33) to wide (5.0)
    "channelwidthcoeff": [
        0.25, 0.50, 0.75, 1.00, 1.25, 1.50,
        1.75, 2.00, 2.33, 2.75, 3.25, 4.00, 5.00
    ],
    # Multipliers on per-class baseline PsiB (baseline mult = 1.0; series 59)
    # Range 0.8–1.25 based on single-param sweep: KGE is positive only in this
    # zone; below 0.8x and above 1.25x model performance degrades sharply.
    # Applied uniformly across all soil classes (same pattern as thetaS_mult).
    "psiB_mult": [
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
        0.70, 0.80, 0.90, 1.00, 1.25, 1.50,
        2.00, 3.00, 5.00,
    ],
    # Multipliers on per-class baseline thetaS (baseline mult = 1.0)
    # Covers Luke's LHS range (0.38–0.45) and extends slightly beyond
    "thetaS_mult": [
        0.85, 0.88, 0.91, 0.94, 0.97,
        1.00, 1.03, 1.06, 1.09, 1.12, 1.15
    ],
    # Absolute saturated anisotropy ratio, all soil classes (baseline = 1.0)
    # 1.0 = isotropic; higher = stronger lateral saturated flow
    "As_value": [
        1.0, 2.0, 5.0, 10.0, 20.0,
        50.0, 100.0, 200.0, 500.0, 1000.0
    ],
    # Absolute unsaturated anisotropy ratio, all soil classes (baseline = 1.0)
    "Au_value": [
        1.0, 2.0, 5.0, 10.0, 20.0,
        50.0, 100.0, 200.0, 500.0, 1000.0
    ],
    # Combined sweep: As and Au set to the same value simultaneously (series 66)
    # Use to test the joint response when both lateral flow pathways are active together.
    "AsAu_value": [
        1.0, 2.0, 5.0, 10.0, 20.0,
        50.0, 100.0, 200.0, 500.0, 1000.0
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
        elif param_name == "thetaS_mult":
            print(f"    Mode: uniform multiplier on per-class thetaS | 1.0 = no change")
        elif param_name in ("As_value", "Au_value"):
            print(f"    Mode: absolute value applied to all soil classes | 1.0 = isotropic")
        elif param_name == "AsAu_value":
            print(f"    Mode: As AND Au set to the same value | 1.0 = isotropic (series 66)")
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