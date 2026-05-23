"""
run_sensitivity_single.py
=========================
Runs tRIBS for one sensitivity run (reads current_run_config.json written
by build_sensitivity_run.py), then exports:
  - *_compare_obs_sim.csv
  - *_metrics_summary.csv

Mirrors Run_Model.ipynb exactly.

Usage (run from the smf_demo directory):
    python run_sensitivity_single.py

Called automatically by run_sensitivity_sweep.py after each build step.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path


def run_and_score():
    from pytRIBS.classes import Project, Results

    # ------------------------------------------------------------------
    # Load run config written by build_sensitivity_run.py
    # ------------------------------------------------------------------
    notebook_dir = Path.cwd()
    project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
    calib_dir    = project_root / "calibration_work"
    config_path  = calib_dir / "current_run_config.json"

    with open(config_path) as f:
        run_config = json.load(f)

    run_id        = run_config["run_id"]
    input_file    = run_config["input_file"]
    log_file      = run_config["log_file"]
    event_start   = run_config["event_start"]
    event_end     = run_config["event_end"]

    csv_export_dir     = Path(run_config["csv_export_dir"])
    summary_export_dir = Path(run_config["summary_export_dir"])

    # ------------------------------------------------------------------
    # Create output folders if needed
    # ------------------------------------------------------------------
    results_folder = Path(run_config["output_prefix"]).parent
    log_folder     = Path(log_file).parent
    results_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)
    csv_export_dir.mkdir(parents=True, exist_ok=True)
    summary_export_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run tRIBS
    # ------------------------------------------------------------------
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    print(f"  Running tRIBS: {run_id}")
    t0 = time.time()
    exit_code = os.system(f"tRIBS {input_file} > {log_file} 2>&1")
    duration  = (time.time() - t0) / 60

    if exit_code != 0:
        print(f"  WARNING: tRIBS may have failed (exit code {exit_code}). Check {log_file}")
    else:
        print(f"  tRIBS finished in {duration:.2f} min")

    # ------------------------------------------------------------------
    # Load results via pytRIBS
    # ------------------------------------------------------------------
    name = run_config["location"]
    proj = Project(os.getcwd(), name, 26912)

    results = Results(input_file, meta=proj.meta)
    results.get_mrf_results()
    results.get_element_results()
    strmflw_sim_raw = results.get_qout_results()

    # ------------------------------------------------------------------
    # Load observed discharge
    # ------------------------------------------------------------------
    obs_filepath = '../smf_init_data/met/SMF_Observations_1993-2025.xlsx'
    obs_df = pd.read_excel(obs_filepath, sheet_name='Discharge', skiprows=6)
    obs_df['datetime'] = pd.to_datetime(
        obs_df['Date'].astype(str) + ' ' + obs_df['Time'].astype(str)
    )
    obs_df.set_index('datetime', inplace=True)
    obs_df['Observed_CMS'] = obs_df['cfs'] * 0.0283168

    # ------------------------------------------------------------------
    # Align simulated and observed to 5-minute intervals
    # ------------------------------------------------------------------
    strmflw_sim = strmflw_sim_raw.copy()
    strmflw_sim['Time'] = pd.to_datetime(strmflw_sim['Time'])
    strmflw_sim.set_index('Time', inplace=True)

    obs_resampled = obs_df['Observed_CMS'].resample('5min').mean()
    sim_resampled = strmflw_sim['Qstrm_m3s'].resample('5min').mean()

    compare_df = pd.DataFrame({
        'Observed':  obs_resampled,
        'Simulated': sim_resampled,
    }).dropna()

    event_df = compare_df.loc[event_start:event_end].copy()

    if event_df.empty:
        print(f"  ERROR: event_df is empty for {run_id}. Check event window and outputs.")
        sys.exit(1)

    print(f"  Aligned event timesteps: {len(event_df)}")

    # ------------------------------------------------------------------
    # Save comparison CSV
    # ------------------------------------------------------------------
    compare_csv = csv_export_dir / f"{run_id}_compare_obs_sim.csv"
    event_df.to_csv(compare_csv, index=True)
    print(f"  Saved: {compare_csv.name}")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    obs = event_df['Observed']
    sim = event_df['Simulated']

    obs_peak  = obs.max()
    sim_peak  = sim.max()
    obs_tpeak = obs.idxmax()
    sim_tpeak = sim.idxmax()

    dt_seconds = (event_df.index[1] - event_df.index[0]).total_seconds()
    obs_vol_m3 = obs.sum() * dt_seconds
    sim_vol_m3 = sim.sum() * dt_seconds
    vol_error_pct = ((sim_vol_m3 - obs_vol_m3) / obs_vol_m3) * 100

    rmse  = np.sqrt(np.mean((sim - obs) ** 2))
    nse   = 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    pbias = 100 * (np.sum(sim - obs) / np.sum(obs))

    r     = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim)  / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    metrics_summary = {
        "run_id":              run_id,
        "swept_param":         run_config.get("swept_param", ""),
        "swept_value":         run_config.get("swept_value", np.nan),
        "event_start":         event_start,
        "event_end":           event_end,
        "obs_peak_m3s":        obs_peak,
        "sim_peak_m3s":        sim_peak,
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
        # all parameter values for completeness
        "Ks_mult":             run_config["Ks_mult"],
        "f_RS_abs":            run_config.get("f_RS_abs", np.nan),
        "As_value":            run_config.get("As_value", np.nan),
        "Au_value":            run_config.get("Au_value", np.nan),
        "thetaS_mult":         run_config.get("thetaS_mult", np.nan),
        "optpercolation":      run_config["optpercolation"],
        "channelconductivity_mmhr": run_config["channelconductivity_mmhr"],
        "channelporosity":     run_config["channelporosity"],
        "kinemvelcoef":        run_config["kinemvelcoef"],
        "flowexp":             run_config["flowexp"],
        "channelroughness":    run_config["channelroughness"],
        "channelwidthcoeff":   run_config["channelwidthcoeff"],
    }

    summary_file = summary_export_dir / f"{run_id}_metrics_summary.csv"
    pd.DataFrame([metrics_summary]).to_csv(summary_file, index=False)
    print(f"  KGE={kge:.3f}  NSE={nse:.3f}  RMSE={rmse:.3f}  PBIAS={pbias:+.1f}%")
    print(f"  Saved: {summary_file.name}")

    return metrics_summary


if __name__ == "__main__":
    run_and_score()
