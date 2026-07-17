"""
run_sensitivity_single.py
=========================
Runs tRIBS for one sensitivity/LHS run (reads current_run_config.json written
by build_sensitivity_run.py or an LHS script), then exports:
  - *_compare_obs_sim.csv
  - *_metrics_summary.csv

Mirrors Run_Model.ipynb exactly.

OBSERVED DATA MODE
------------------
This script supports two observed-data sources, selected automatically by
checking for a synthetic truth file in calibration_work/synth_truth/:

  REAL GAUGE MODE (default):
    Reads ../smf_init_data/met/SMF_Observations_1993-2025.xlsx
    Used for all standard calibration series (81, 82, 83, …)

  SYNTHETIC TRUTH MODE (automatic when synth_truth/ file exists):
    Reads calibration_work/synth_truth/synth_truth_*.qout
    Used for synthetic inversion series (90+)
    The .qout file uses fractional hours from 2014-08-01 00:00;
    the flood peak for Aug 12 falls near hour 282–283.

By default, synthetic mode activates when exactly one *.qout file exists
directly in calibration_work/synth_truth/ (real-gauge mode is used
otherwise). A build step can override this by writing a "truth_file" key
into current_run_config.json -- either a specific .qout path, or a
subdirectory containing exactly one .qout file (e.g. "synth_truth/
storm080"). This is how the storm080/storm125 sibling sweeps select their
own truth without disturbing the baseline auto-detect. See
_find_synth_truth_file() for details.

METRICS COMPUTED
----------------
Standard (all runs):
  KGE, KGE components (r, alpha, beta)
  NSE, RMSE, PBIAS
  Peak discharge error (m³/s and %)
  Peak timing error (hr)
  Total volume error (m³ and %)

Phase-specific (5 new metrics, from hydrograph_metrics_table):
  All five use a threshold = 5% of observed peak discharge.

  PRE-PEAK
    first_arrival_error_min   : sim threshold-crossing time minus obs
                                threshold-crossing time (positive = model late)
    rising_limb_steepness_ratio : (sim dQ/dt from threshold to peak) /
                                  (obs dQ/dt from threshold to peak)
                                  1.0 = perfect; >1 = too steep; <1 = too flat
    time_to_peak_from_exc_min : sim duration (threshold→peak) minus obs
                                duration (positive = model takes longer)

  VOLUME
    duration_above_thresh_error_min : sim duration above threshold minus obs
                                      duration (positive = model lingers longer)

  RECESSION
    recession_rate_ratio      : sim log-linear recession slope / obs slope
                                (computed on post-peak falling limb above thresh)
                                1.0 = perfect; >1 = model recedes faster

Usage (run from the smf_demo directory):
    python run_sensitivity_single.py

Called automatically by run_sensitivity_sweep.py and all run_lhs_*.py scripts
after each build step.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------------------------------------------------
# THRESHOLD FRACTION for phase-specific metrics
# 5% of observed peak discharge.  Adjust here if needed.
# -----------------------------------------------------------------------
THRESHOLD_FRAC = 0.05


# -----------------------------------------------------------------------
# HELPER: phase-specific metrics
# -----------------------------------------------------------------------
def _compute_phase_metrics(obs: pd.Series, sim: pd.Series) -> dict:
    """
    Compute the five hydrograph-phase metrics defined in
    hydrograph_metrics_table.html.  All use threshold = THRESHOLD_FRAC
    of observed peak.

    Parameters
    ----------
    obs, sim : pd.Series with a DatetimeIndex at uniform 5-min intervals,
               covering the event window (already clipped to event_start/end).

    Returns
    -------
    dict of metric name → float.  NaN is returned for any metric that
    cannot be computed (e.g. threshold never crossed by sim).
    """
    threshold = THRESHOLD_FRAC * obs.max()
    dt_min    = (obs.index[1] - obs.index[0]).total_seconds() / 60.0

    # ---- observed threshold crossing and peak ----
    obs_above     = obs >= threshold
    obs_cross_idx = obs_above.idxmax() if obs_above.any() else None   # first True
    obs_tpeak     = obs.idxmax()

    # ---- simulated threshold crossing ----
    sim_above     = sim >= threshold
    sim_cross_idx = sim_above.idxmax() if sim_above.any() else None

    # ------------------------------------------------------------------
    # 1. First arrival error (pre-peak)
    #    Positive → model arrives late
    # ------------------------------------------------------------------
    if obs_cross_idx is not None and sim_cross_idx is not None:
        first_arrival_error_min = (
            (sim_cross_idx - obs_cross_idx).total_seconds() / 60.0
        )
    else:
        first_arrival_error_min = np.nan

    # ------------------------------------------------------------------
    # 2. Rising limb steepness ratio (pre-peak)
    #    dQ/dt = (peak - threshold_value) / time_to_peak_from_exceedance
    #    Ratio = sim slope / obs slope.  1.0 is perfect.
    # ------------------------------------------------------------------
    if obs_cross_idx is not None and sim_cross_idx is not None:
        obs_rise_min = max(
            (obs_tpeak - obs_cross_idx).total_seconds() / 60.0, dt_min
        )
        obs_slope = (obs.max() - threshold) / obs_rise_min

        sim_tpeak    = sim.idxmax()
        sim_rise_min = max(
            (sim_tpeak - sim_cross_idx).total_seconds() / 60.0, dt_min
        )
        sim_slope = (sim.max() - threshold) / sim_rise_min

        rising_limb_steepness_ratio = (
            sim_slope / obs_slope if obs_slope > 0 else np.nan
        )
    else:
        rising_limb_steepness_ratio = np.nan

    # ------------------------------------------------------------------
    # 3. Time-to-peak from first exceedance (pre-peak)
    #    Positive → model takes longer to go from threshold to peak
    # ------------------------------------------------------------------
    if obs_cross_idx is not None and sim_cross_idx is not None:
        obs_ttp = (obs_tpeak - obs_cross_idx).total_seconds() / 60.0
        sim_ttp = (sim.idxmax() - sim_cross_idx).total_seconds() / 60.0
        time_to_peak_from_exc_min = sim_ttp - obs_ttp
    else:
        time_to_peak_from_exc_min = np.nan

    # ------------------------------------------------------------------
    # 4. Duration above threshold error (volume)
    #    Count timesteps above threshold × dt_min.
    #    Positive → model sustains elevated flow longer than observed.
    # ------------------------------------------------------------------
    obs_dur_min = obs_above.sum() * dt_min
    sim_dur_min = sim_above.sum() * dt_min
    duration_above_thresh_error_min = sim_dur_min - obs_dur_min

    # ------------------------------------------------------------------
    # 5. Recession rate ratio (recession)
    #    Log-linear slope of falling limb above threshold.
    #    slope = d(ln Q)/dt, fitted by least-squares.
    #    Ratio = sim_slope / obs_slope.  1.0 is perfect.
    #    Positive slopes indicate rising flow; we expect negative slopes
    #    on the recession, so the ratio of two negative numbers is positive
    #    when both recede in the same direction.
    # ------------------------------------------------------------------
    def _recession_slope(series: pd.Series, tpeak_idx) -> float:
        """Log-linear slope of the falling limb above threshold."""
        rec = series.loc[tpeak_idx:]          # post-peak
        rec = rec[rec >= threshold]            # above threshold only
        rec = rec[rec > 0]                     # guard log(0)
        if len(rec) < 3:
            return np.nan
        t_numeric = np.arange(len(rec), dtype=float)
        log_q     = np.log(rec.values)
        slope     = np.polyfit(t_numeric, log_q, 1)[0]
        return slope

    obs_rec_slope = _recession_slope(obs, obs_tpeak)
    sim_rec_slope = _recession_slope(sim, sim.idxmax())

    if (obs_rec_slope is not np.nan and sim_rec_slope is not np.nan
            and not np.isnan(obs_rec_slope) and not np.isnan(sim_rec_slope)
            and obs_rec_slope != 0):
        recession_rate_ratio = sim_rec_slope / obs_rec_slope
    else:
        recession_rate_ratio = np.nan

    return {
        # pre-peak
        "threshold_m3s":                  threshold,
        "first_arrival_error_min":        first_arrival_error_min,
        "rising_limb_steepness_ratio":    rising_limb_steepness_ratio,
        "time_to_peak_from_exc_min":      time_to_peak_from_exc_min,
        # volume
        "duration_above_thresh_error_min": duration_above_thresh_error_min,
        # recession
        "recession_rate_ratio":           recession_rate_ratio,
    }


# -----------------------------------------------------------------------
# HELPER: detect synthetic truth mode
# -----------------------------------------------------------------------
def _find_synth_truth_file(calib_dir: Path, truth_file_override=None):
    """
    Return the path to the synthetic truth .qout file to score against.

    truth_file_override may be (read from run_config["truth_file"], written
    by the calling build step -- e.g. build_only() in a run_lhs_*.py sweep
    script):

      - None (default): fall back to the original auto-detect -- exactly
        one *.qout file directly in calibration_work/synth_truth/, else
        None (real-gauge mode). This is what the baseline Series 100 sweep
        uses, and is unchanged from the original single-truth convention.

      - A directory path (relative to calib_dir, or absolute): the same
        "exactly one *.qout file" auto-detect is applied *within that
        directory* instead. This is what the storm080/storm125 sibling
        sweeps use (e.g. "synth_truth/storm080") -- callers never need to
        hardcode the exact truth filename, only which storm-magnitude
        subdirectory to look in.

      - A file path: used directly, if it exists.

    With three truth files now live simultaneously (baseline in
    synth_truth/, storm080 and storm125 each in their own subdirectory),
    an explicit override that fails to resolve raises FileNotFoundError
    rather than silently falling back to real-gauge mode -- a silent
    fallback would let a whole sweep run and complete against the wrong
    (or no) truth without any error, which is a much worse failure mode
    than stopping immediately.
    """
    if truth_file_override:
        override_path = Path(truth_file_override)
        if not override_path.is_absolute():
            override_path = calib_dir / override_path

        if override_path.is_dir():
            qout_files = list(override_path.glob("*.qout"))
            if len(qout_files) == 1:
                return qout_files[0]
            raise FileNotFoundError(
                f"truth_file override directory {override_path} does not "
                f"contain exactly one *.qout file (found "
                f"{[f.name for f in qout_files]}). Check that only the "
                f"intended storm's truth file lives in this subdirectory."
            )

        if override_path.exists():
            return override_path

        raise FileNotFoundError(f"truth_file override does not exist: {override_path}")

    synth_dir = calib_dir / "synth_truth"
    if not synth_dir.exists():
        return None
    qout_files = list(synth_dir.glob("*.qout"))
    if len(qout_files) == 1:
        return qout_files[0]
    if len(qout_files) > 1:
        print(f"  WARNING: multiple .qout files in synth_truth/; "
              f"falling back to real gauge. Files found: "
              f"{[f.name for f in qout_files]}")
    return None


# -----------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------
def run_and_score():
    from pytRIBS.classes import Project, Results

    # ------------------------------------------------------------------
    # Load run config written by build_sensitivity_run.py
    # ------------------------------------------------------------------
    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
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
    t0        = time.time()
    exit_code = os.system(f"tRIBS {input_file} > {log_file} 2>&1")
    duration  = (time.time() - t0) / 60

    if exit_code != 0:
        print(f"  WARNING: tRIBS may have failed (exit code {exit_code}). "
              f"Check {log_file}")
    else:
        print(f"  tRIBS finished in {duration:.2f} min")

    # ------------------------------------------------------------------
    # Load simulated streamflow via pytRIBS
    # ------------------------------------------------------------------
    name = run_config["location"]
    proj = Project(os.getcwd(), name, 26912)

    results         = Results(input_file, meta=proj.meta)
    results.get_mrf_results()
    results.get_element_results()
    strmflw_sim_raw = results.get_qout_results()

    # ------------------------------------------------------------------
    # Load observed discharge
    # Automatically switches between real-gauge and synthetic-truth mode.
    # ------------------------------------------------------------------
    truth_file_override = run_config.get("truth_file")
    synth_file = _find_synth_truth_file(calib_dir, truth_file_override)

    if synth_file is not None:
        # ---- SYNTHETIC TRUTH MODE ----
        print(f"  [SYNTH MODE] Reading observed from: {synth_file.name}")
        obs_raw = pd.read_csv(
            synth_file, sep=r'\s+', skiprows=1,
            names=['Time_hr', 'Qstrm_m3s', 'Hlev_m']
        )
        # .qout Time column is fractional hours from model start 2014-08-01 00:00
        obs_raw['datetime'] = pd.to_datetime(
            obs_raw['Time_hr'] * 3600, unit='s',
            origin=pd.Timestamp('2014-08-01')
        )
        obs_raw.set_index('datetime', inplace=True)
        obs_raw['Observed_CMS'] = obs_raw['Qstrm_m3s']
        obs_df = obs_raw

    else:
        # ---- REAL GAUGE MODE ----
        print(f"  [GAUGE MODE] Reading observed from SMF_Observations xlsx")
        obs_filepath = '../smf_init_data/met/SMF_Observations_1993-2025.xlsx'
        obs_df = pd.read_excel(
            obs_filepath, sheet_name='Discharge', skiprows=6
        )
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

    # Synthetic .qout timestep is 3.75 min — use interpolation, not mean,
    # to avoid NaN dropout on a 5-min grid.
    if synth_file is not None:
        obs_resampled = (obs_df['Observed_CMS']
                         .resample('5min')
                         .interpolate(method='time'))
    else:
        obs_resampled = obs_df['Observed_CMS'].resample('5min').mean()

    sim_resampled = strmflw_sim['Qstrm_m3s'].resample('5min').mean()

    compare_df = pd.DataFrame({
        'Observed':  obs_resampled,
        'Simulated': sim_resampled,
    }).dropna()

    event_df = compare_df.loc[event_start:event_end].copy()

    if event_df.empty:
        print(f"  ERROR: event_df is empty for {run_id}. "
              f"Check event window and outputs.")
        sys.exit(1)

    print(f"  Aligned event timesteps: {len(event_df)}")

    # ------------------------------------------------------------------
    # Save comparison CSV
    # ------------------------------------------------------------------
    compare_csv = csv_export_dir / f"{run_id}_compare_obs_sim.csv"
    event_df.to_csv(compare_csv, index=True)
    print(f"  Saved: {compare_csv.name}")

    # ------------------------------------------------------------------
    # Compute standard metrics
    # ------------------------------------------------------------------
    obs = event_df['Observed']
    sim = event_df['Simulated']

    obs_peak  = obs.max()
    sim_peak  = sim.max()
    obs_tpeak = obs.idxmax()
    sim_tpeak = sim.idxmax()

    dt_seconds    = (event_df.index[1] - event_df.index[0]).total_seconds()
    obs_vol_m3    = obs.sum() * dt_seconds
    sim_vol_m3    = sim.sum() * dt_seconds
    vol_error_pct = ((sim_vol_m3 - obs_vol_m3) / obs_vol_m3) * 100

    peak_error_m3s  = sim_peak - obs_peak
    peak_error_pct  = (peak_error_m3s / obs_peak) * 100

    rmse  = np.sqrt(np.mean((sim - obs) ** 2))
    nse   = 1 - (np.sum((sim - obs) ** 2)
                 / np.sum((obs - obs.mean()) ** 2))
    pbias = 100 * (np.sum(sim - obs) / np.sum(obs))

    r     = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim)  / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # ------------------------------------------------------------------
    # Compute phase-specific metrics (5 new)
    # ------------------------------------------------------------------
    phase_metrics = _compute_phase_metrics(obs, sim)

    # ------------------------------------------------------------------
    # Assemble metrics summary
    # ------------------------------------------------------------------
    metrics_summary = {
        # --- identifiers ---
        "run_id":              run_id,
        "swept_param":         run_config.get("swept_param", ""),
        "swept_value":         run_config.get("swept_value", np.nan),
        "obs_mode":            "synth" if synth_file is not None else "gauge",
        "event_start":         event_start,
        "event_end":           event_end,

        # --- peak ---
        "obs_peak_m3s":        obs_peak,
        "sim_peak_m3s":        sim_peak,
        "peak_error_m3s":      peak_error_m3s,
        "peak_error_pct":      peak_error_pct,
        "obs_peak_time":       str(obs_tpeak),
        "sim_peak_time":       str(sim_tpeak),
        "peak_timing_error_hr": (sim_tpeak - obs_tpeak).total_seconds() / 3600,

        # --- volume ---
        "obs_volume_m3":       obs_vol_m3,
        "sim_volume_m3":       sim_vol_m3,
        "volume_error_pct":    vol_error_pct,

        # --- standard scalar metrics ---
        "rmse_m3s":            rmse,
        "nse":                 nse,
        "pbias_pct":           pbias,
        "kge":                 kge,
        "kge_r":               r,
        "kge_alpha":           alpha,
        "kge_beta":            beta,

        # --- phase-specific metrics (hydrograph_metrics_table) ---
        "threshold_m3s":                   phase_metrics["threshold_m3s"],
        # pre-peak
        "first_arrival_error_min":         phase_metrics["first_arrival_error_min"],
        "rising_limb_steepness_ratio":     phase_metrics["rising_limb_steepness_ratio"],
        "time_to_peak_from_exc_min":       phase_metrics["time_to_peak_from_exc_min"],
        # volume
        "duration_above_thresh_error_min": phase_metrics["duration_above_thresh_error_min"],
        # recession
        "recession_rate_ratio":            phase_metrics["recession_rate_ratio"],

        # --- parameter values (all series) ---
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

    print(f"  KGE={kge:.3f}  NSE={nse:.3f}  RMSE={rmse:.3f}  "
          f"PBIAS={pbias:+.1f}%  peak_err={peak_error_pct:+.1f}%")
    print(f"  arrival_err={phase_metrics['first_arrival_error_min']:+.1f} min  "
          f"rise_ratio={phase_metrics['rising_limb_steepness_ratio']:.3f}  "
          f"rec_ratio={phase_metrics['recession_rate_ratio']:.3f}")
    print(f"  Saved: {summary_file.name}")

    return metrics_summary


if __name__ == "__main__":
    run_and_score()