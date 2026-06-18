"""
run_lhs_synth_4param.py
=======================
Latin Hypercube Sampling sweep for the Series 95 synthetic inversion
experiment. Scores simulated hydrographs against a synthetic truth
.qout file (not the real gauge) to test parameter identifiability and
equifinality.

TO CHANGE, ADJUST SERIES NUMBER AND PARAMETER VALUES & SWEPT VALUES

Parameters swept (4 free dimensions): DOCSTRING NOT ADJUSTED FOR SERIES 95
    Ks_mult:          7.5 – 9.5   (true = 8.5,  at 50th percentile of range)
    kinemvelcoef:     2.5 – 6.5   (true = 4.5,  at 50th percentile of range)
    flowexp:          0.18 – 0.35 (true = 0.24, at 37th percentile of range)
    channelroughness: 0.02 – 0.04 (true = 0.026, at 30th percentile of range)

Fixed (not swept):
    f_RS_abs: 0.020 mm^-1   (RS soil conductivity decay)

Truth exclusion:
    Any LHS sample whose four parameter values match the truth values
    within TRUTH_TOL = 1e-6 is skipped automatically. The truth run
    (SMF_20140812_60_Ks8p5x) was generated separately and must not
    appear in the inversion ensemble.

KGE ceiling:
    The self-score of the truth run against itself is KGE = 0.912 due
    to resampling asymmetry between sim (.mean()) and obs
    (.interpolate()). Interpret top runs by parameter proximity to
    truth, not by KGE approaching 1.0. A run scoring near 0.91 with
    parameters near truth is a successful recovery.

Series: 95
Output: calibration_work/03_comparisons/summary_tables/lhs_results_synth_4param_91.csv

Usage (run from the smf_demo directory):
    python run_lhs_synth_4param.py              # 50 samples, seed=42
    python run_lhs_synth_4param.py --n 100      # recommended
    python run_lhs_synth_4param.py --seed 99    # different seed
    python run_lhs_synth_4param.py --skip_existing  # resume interrupted run
"""

import argparse
import os
import time
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import build_sensitivity_run as builder
import run_sensitivity_single as runner
from pytRIBS.classes import Project, Soil, Land, Met, Model

# ------------------------------------------------------------------
# LHS PARAMETER RANGES
# True value must be interior to [lo, hi] — not at or within 10% of
# either boundary.
# ------------------------------------------------------------------
LHS_PARAMS = {
    "Ks_mult":          {"lo": 12.0,  "hi": 18.0},   # true = 15.0  (50th pct)
    "kinemvelcoef":     {"lo": 2.5,  "hi": 6.5},   # true = 4.5  (50th pct)
    "flowexp":          {"lo": 0.10, "hi": 0.35},  # true = 0.24 (37th pct)
    "channelroughness": {"lo": 0.01, "hi": 0.10},  # true = 0.026 (30th pct)
}

# Fixed — not swept
F_RS_ABS_FIXED = 0.020   # mm^-1

# Series metadata
LHS_SERIES   = "95"
LHS_CATEGORY = "95_lhs_synth_inversion"

# ------------------------------------------------------------------
# TRUTH EXCLUSION GUARD
# Samples matching these values within TRUTH_TOL are skipped.
# ------------------------------------------------------------------
TRUTH_VALUES = {
    "Ks_mult":          8.5,
    "kinemvelcoef":     4.5,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}
TRUTH_TOL = 1e-6


def is_truth_run(ks, cv, r, n):
    """Return True if all four values match the truth within TRUTH_TOL."""
    return (abs(ks - TRUTH_VALUES["Ks_mult"])          < TRUTH_TOL and
            abs(cv - TRUTH_VALUES["kinemvelcoef"])      < TRUTH_TOL and
            abs(r  - TRUTH_VALUES["flowexp"])           < TRUTH_TOL and
            abs(n  - TRUTH_VALUES["channelroughness"])  < TRUTH_TOL)


# ------------------------------------------------------------------
# LHS SAMPLE GENERATION
# ------------------------------------------------------------------
def generate_lhs_samples(n, params, seed=None):
    """
    Generate n Latin Hypercube samples across the parameter ranges.
    Each parameter range is divided into n equal intervals; one sample
    is drawn uniformly from each interval, then independently shuffled
    across parameters (ensuring full marginal coverage).

    Returns a DataFrame with one column per parameter and n rows.
    """
    rng     = np.random.default_rng(seed)
    samples = {}
    for param, bounds in params.items():
        lo, hi    = bounds["lo"], bounds["hi"]
        intervals = np.linspace(lo, hi, n + 1)
        points    = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        samples[param] = points
    return pd.DataFrame(samples)


# ------------------------------------------------------------------
# RUN ID CONSTRUCTION
# ------------------------------------------------------------------
def build_lhs_run_id(ks_mult, kinemvelcoef, flowexp, channelroughness):
    """Build a compact, human-readable run ID for a 4-parameter LHS point."""
    ks_lbl = builder.value_to_label(ks_mult)
    cv_lbl = builder.value_to_label(kinemvelcoef)
    r_lbl  = builder.value_to_label(flowexp)
    n_lbl  = builder.value_to_label(channelroughness)
    change_tested = f"Ks{ks_lbl}x_cv{cv_lbl}_r{r_lbl}_n{n_lbl}"
    run_id = f"{builder.LOCATION}_{builder.EVENT_DATE}_{LHS_SERIES}_{change_tested}"
    return run_id, change_tested


# ------------------------------------------------------------------
# SKIP CHECK (resume support)
# ------------------------------------------------------------------
def csv_already_exists(run_id, calib_dir):
    csv_path = (calib_dir / "03_comparisons" / "csv_exports"
                / f"{run_id}_compare_obs_sim.csv")
    return csv_path.exists()


def load_existing_results(out_path):
    if out_path.exists():
        try:
            df = pd.read_csv(out_path)
            print(f"  Loaded existing results: {len(df)} rows from {out_path.name}")
            return df
        except Exception as e:
            print(f"  Warning: could not load existing results ({e}). Starting fresh.")
    return pd.DataFrame()


# ------------------------------------------------------------------
# BUILD + RUN ONE LHS POINT
# ------------------------------------------------------------------
def build_and_run_lhs(ks_mult, kinemvelcoef, flowexp, channelroughness):
    """
    Build one tRIBS input file for a 4-parameter LHS point and run it.

    Patches builder.BASELINE temporarily; reverts on exit regardless of
    success or failure. f_RS_abs is pinned at F_RS_ABS_FIXED.
    OPINTRVL is set to 0.0833 hr (5-minute output) for all runs.

    Returns a metrics dict from run_sensitivity_single.run_and_score().
    """
    run_id, change_tested = build_lhs_run_id(
        ks_mult, kinemvelcoef, flowexp, channelroughness)

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir    = project_root / "calibration_work"

    run_input_dir      = calib_dir / "01_run_inputs"  / LHS_CATEGORY
    run_results_dir    = calib_dir / "02_results"     / LHS_CATEGORY / run_id
    csv_export_dir     = calib_dir / "03_comparisons" / "csv_exports"
    plot_export_dir    = calib_dir / "03_comparisons" / "hydrograph_plots"
    summary_export_dir = calib_dir / "03_comparisons" / "summary_tables"
    log_dir            = calib_dir / "06_logs"

    for folder in [run_input_dir, run_results_dir, csv_export_dir,
                   plot_export_dir, summary_export_dir, log_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    original_baseline = builder.BASELINE.copy()
    builder.BASELINE["Ks_mult"]          = ks_mult
    builder.BASELINE["kinemvelcoef"]     = kinemvelcoef
    builder.BASELINE["flowexp"]          = flowexp
    builder.BASELINE["channelroughness"] = channelroughness

    try:
        baseline    = builder.BASELINE

        proj = Project(os.getcwd(), builder.LOCATION, builder.EPSG)

        # --- Land use ---
        landuse_ras = '../smf_init_data/LandUse.asc'
        shutil.copy(landuse_ras, proj.directories['land'])
        landuse_ras = f"{proj.directories['land']}/{os.path.basename(landuse_ras)}"

        # --- Soil raster ---
        soil_ras = '../smf_init_data/ADOT_SoilTypes.asc'
        shutil.copy(soil_ras, proj.directories['soil'])
        soil_ras = f"{proj.directories['soil']}/{os.path.basename(soil_ras)}"

        # --- Soil table ---
        soil = Soil(meta=proj.meta)
        shutil.copy('../smf_init_data/SOLUS_Bedrock_m.asc', proj.directories['soil'])
        soil.bedrockfile['value'] = f"{proj.directories['soil']}/SOLUS_Bedrock_m.asc"
        shutil.copy('../smf_init_data/InitGW_95pct_mm.asc', proj.directories['soil'])
        soil.gwaterfile['value']  = f"{proj.directories['soil']}/InitGW_95pct_mm.asc"
        shutil.copy('../smf_init_data/soils.sdt', proj.directories['soil'])
        soil.soiltablename['value'] = f"{proj.directories['soil']}/soils.sdt"
        soil.soilmapname['value']   = soil_ras

        soil_table = soil.read_soil_table(textures=True)

        for soil_cls in soil_table:
            soil_cls['As'] = baseline["As_value"]
            soil_cls['Au'] = baseline["Au_value"]
            soil_cls['ks'] = 0.7
            soil_cls['Cs'] = 1.4e6
            cid = str(soil_cls['ID'])
            if cid in builder.SOIL_PARAM_LOOKUP:
                soil_params = builder.SOIL_PARAM_LOOKUP[cid]
                soil_cls['Ks']     = soil_params['Ks'] * ks_mult
                soil_cls['thetaS'] = soil_params['thetaS']
                soil_cls['thetaR'] = soil_params['thetaR']
                soil_cls['m']      = soil_params['m']
                soil_cls['PsiB']   = soil_params['PsiB']
                soil_cls['n']      = soil_params['n']
                soil_cls['f'] = F_RS_ABS_FIXED if cid == '1' else soil_params['f']
            else:
                print(f"  WARNING: Soil ID {cid} not in lookup; using fallback defaults.")
                soil_cls['Ks'] = 10.0; soil_cls['thetaS'] = 0.4; soil_cls['thetaR'] = 0.05
                soil_cls['m'] = 0.2; soil_cls['PsiB'] = -200; soil_cls['f'] = 0.001; soil_cls['n'] = 0.4

        working_soil_table    = Path("data/model/soil/soil.sdt")
        soil.write_soil_table(soil_table, str(working_soil_table), textures=True)
        run_soil_path = run_input_dir / f"soils_{run_id}.sdt"
        shutil.copy(working_soil_table, run_soil_path)
        soil.soiltablename['value'] = os.path.relpath(run_soil_path, script_dir)
        soil.optsoiltype['value']   = 0

        # --- Land use table ---
        land = Land(meta=proj.meta)
        land.landmapname['value']   = f"{proj.directories['land']}/LandUse.asc"
        land.landtablename['value'] = f"{proj.directories['land']}/land_use_params.ldt"
        landuse_list = []
        for lu_id, lp in builder.LAND_PARAM_LOOKUP.items():
            row = lp.copy(); row['ID'] = lu_id; row['a'] = -9999; row['b1'] = -9999
            landuse_list.append(row)
        land.write_landuse_table(landuse_list, land.landtablename['value'])

        # --- Met ---
        met = Met(meta=proj.meta)
        met.hydrometbasename['value'] = builder.LOCATION
        met.hydrometstations['value'] = "../smf_init_data/met/Master_Met.sdf"
        met.gaugestations['value']    = "../smf_init_data/met/Master_Precip.sdf"

        # --- Model ---
        model = Model(met=met, land=land, soil=soil, mesh=None, meta=proj.meta)
        model.parallelmode['value']  = 0
        model.optmeshinput['value']  = 1
        model.inputdatafile['value'] = "../smf_init_data/mesh/SMF_mesh"
        model.inputtime['value']     = 0
        model.optbedrock['value']    = 1
        model.optsnow['value']       = 0
        model.optlanduse['value']    = 0

        model.optpercolation['value']      = baseline["optpercolation"]
        model.channelconductivity['value'] = baseline["channelconductivity_mmhr"] / 3.6e6
        model.channelporosity['value']     = baseline["channelporosity"]

        model.kinemvelcoef['value']      = kinemvelcoef
        model.flowexp['value']           = flowexp
        model.channelroughness['value']  = channelroughness
        model.channelwidthcoeff['value'] = baseline["channelwidthcoeff"]

        model.startdate['value']   = builder.START_DATE
        model.runtime['value']     = builder.RUNTIME_HOURS
        model.rainintrvl['value']  = builder.RAIN_INTERVAL
        model.opintrvl['value']    = 0.0833   # 5-minute output (new in S92)

        input_file_abs    = run_input_dir   / f"{run_id}.in"
        log_file_abs      = log_dir         / f"{run_id}.log"
        output_prefix_abs = run_results_dir / run_id

        input_file    = os.path.relpath(input_file_abs,    script_dir)
        log_file      = os.path.relpath(log_file_abs,      script_dir)
        output_prefix = os.path.relpath(output_prefix_abs, script_dir)

        model.outfilename['value']      = output_prefix
        model.outhydrofilename['value'] = output_prefix

        model.write_node_file([1960, 1547, 3082], 'data/model/pnodes.dat')
        model.nodeoutputlist['value'] = 'data/model/pnodes.dat'
        model.write_node_file([3202], 'data/model/qnodes.dat')
        model.outletnodelist['value'] = 'data/model/qnodes.dat'

        model.write_input_file(input_file)

        print(f"  Ks={ks_mult:.3f}x  cv={kinemvelcoef:.3f}  r={flowexp:.3f}  "
              f"n={channelroughness:.4f}  "
              f"(RS Ks={builder.SOIL_PARAM_LOOKUP['1']['Ks'] * ks_mult:.2f} mm/hr)")

        run_config = {
            "location":                  builder.LOCATION,
            "event_date":                builder.EVENT_DATE,
            "run_number":                LHS_SERIES,
            "change_tested":             change_tested,
            "run_id":                    run_id,
            "run_category":              LHS_CATEGORY,
            "start_date":                builder.START_DATE,
            "runtime_hours":             builder.RUNTIME_HOURS,
            "rain_interval_hours":       builder.RAIN_INTERVAL,
            "event_start":               builder.EVENT_START,
            "event_end":                 builder.EVENT_END,
            "Ks_mult":                   ks_mult,
            "f_RS_abs":                  F_RS_ABS_FIXED,
            "As_value":                  baseline["As_value"],
            "Au_value":                  baseline["Au_value"],
            "optpercolation":            baseline["optpercolation"],
            "channelconductivity_mmhr":  baseline["channelconductivity_mmhr"],
            "channelporosity":           baseline["channelporosity"],
            "kinemvelcoef":              kinemvelcoef,
            "flowexp":                   flowexp,
            "channelroughness":          channelroughness,
            "channelwidthcoeff":         baseline["channelwidthcoeff"],
            "input_file":                input_file,
            "log_file":                  log_file,
            "output_prefix":             output_prefix,
            "csv_export_dir":            os.path.relpath(csv_export_dir,      script_dir),
            "plot_export_dir":           os.path.relpath(plot_export_dir,     script_dir),
            "summary_export_dir":        os.path.relpath(summary_export_dir,  script_dir),
            "swept_param":               "lhs_synth_4param",
            "swept_value":               ks_mult,
        }

        config_path = calib_dir / "current_run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))

    finally:
        builder.BASELINE = original_baseline

    metrics = runner.run_and_score()

    metrics["Ks_mult"]          = ks_mult
    metrics["kinemvelcoef"]     = kinemvelcoef
    metrics["flowexp"]          = flowexp
    metrics["channelroughness"] = channelroughness
    metrics["f_RS_abs"]         = F_RS_ABS_FIXED
    metrics["swept_param"]      = "lhs_synth_4param"

    return metrics


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Series 95 synthetic inversion LHS sweep — 4 free parameters.")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of LHS samples (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose compare CSV already exists")
    args = parser.parse_args()

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir   = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "lhs_results_synth_4param_95.csv"

    samples = generate_lhs_samples(args.n, LHS_PARAMS, seed=args.seed)

    print(f"\n{'='*65}")
    print(f"LHS sweep — Series {LHS_SERIES} — Synthetic Inversion — 4 free params")
    print(f"  ({args.n} samples, seed={args.seed})")
    print(f"  Ks_mult:          {LHS_PARAMS['Ks_mult']['lo']:.2f} - "
          f"{LHS_PARAMS['Ks_mult']['hi']:.2f}x        [true = {TRUTH_VALUES['Ks_mult']}]")
    print(f"  kinemvelcoef:     {LHS_PARAMS['kinemvelcoef']['lo']:.2f} - "
          f"{LHS_PARAMS['kinemvelcoef']['hi']:.2f}          [true = {TRUTH_VALUES['kinemvelcoef']}]")
    print(f"  flowexp:          {LHS_PARAMS['flowexp']['lo']:.2f} - "
          f"{LHS_PARAMS['flowexp']['hi']:.2f}          [true = {TRUTH_VALUES['flowexp']}]")
    print(f"  channelroughness: {LHS_PARAMS['channelroughness']['lo']:.3f} - "
          f"{LHS_PARAMS['channelroughness']['hi']:.3f}         [true = {TRUTH_VALUES['channelroughness']}]")
    print(f"  f_RS_abs:         {F_RS_ABS_FIXED} mm^-1    [FIXED]")
    print(f"  OPINTRVL:         0.0833 hr (5-min output)  [new in S92]")
    print(f"  KGE ceiling:      ~0.912 (resampling asymmetry; see docstring)")
    print(f"  Output: {out_path.name}")
    print(f"{'='*65}\n")

    existing_df      = load_existing_results(out_path)
    existing_run_ids = (set(existing_df["run_id"].values)
                        if not existing_df.empty else set())

    all_results = []
    if not existing_df.empty:
        all_results.extend(existing_df.to_dict("records"))

    completed   = 0
    skipped     = 0
    excluded    = 0
    failed      = 0
    sweep_start = time.time()

    for i, row in samples.iterrows():
        ks_mult          = row["Ks_mult"]
        kinemvelcoef     = row["kinemvelcoef"]
        flowexp          = row["flowexp"]
        channelroughness = row["channelroughness"]

        # --- Truth exclusion guard ---
        if is_truth_run(ks_mult, kinemvelcoef, flowexp, channelroughness):
            print(f"\n[{i+1:>3}/{args.n}]  EXCLUDED: sample matches truth values "
                  f"exactly — skipped by design.")
            excluded += 1
            continue

        run_id, _ = build_lhs_run_id(ks_mult, kinemvelcoef, flowexp, channelroughness)

        print(f"\n[{i+1:>3}/{args.n}]  Ks={ks_mult:.3f}x  cv={kinemvelcoef:.3f}  "
              f"r={flowexp:.3f}  n={channelroughness:.4f}")
        print(f"         -> {run_id}")

        if args.skip_existing and csv_already_exists(run_id, calib_dir):
            print(f"  SKIP (CSV exists): {run_id}")
            skipped += 1
            metrics_file = summary_dir / f"{run_id}_metrics_summary.csv"
            if metrics_file.exists() and run_id not in existing_run_ids:
                try:
                    df_m = pd.read_csv(metrics_file)
                    m    = df_m.iloc[0].to_dict()
                    m["Ks_mult"]          = ks_mult
                    m["kinemvelcoef"]     = kinemvelcoef
                    m["flowexp"]          = flowexp
                    m["channelroughness"] = channelroughness
                    m["f_RS_abs"]         = F_RS_ABS_FIXED
                    all_results.append(m)
                except Exception:
                    pass
            continue

        t0 = time.time()
        try:
            metrics = build_and_run_lhs(ks_mult, kinemvelcoef, flowexp, channelroughness)
            all_results = [r for r in all_results if r.get("run_id") != run_id]
            all_results.append(metrics)
            completed += 1

        except Exception as e:
            print(f"  FAILED: {run_id}")
            print(f"  Error:  {e}")
            failed += 1

        elapsed       = time.time() - t0
        total_elapsed = time.time() - sweep_start
        remaining     = args.n - completed - skipped - excluded - failed
        if completed > 0:
            avg_time = total_elapsed / completed
            eta_min  = (avg_time * remaining) / 60
            print(f"  Run time: {elapsed/60:.1f} min  |  ETA: {eta_min:.0f} min remaining")

        # Incremental save after every completed run
        if all_results:
            pd.DataFrame(all_results).to_csv(out_path, index=False)

    print(f"\n{'='*65}")
    print(f"Sweep complete:  {completed} ran,  {skipped} skipped,  "
          f"{excluded} excluded (truth),  {failed} failed")
    print(f"{'='*65}\n")

    if all_results:
        final_df = pd.DataFrame(all_results).sort_values("kge", ascending=False)
        final_df.to_csv(out_path, index=False)
        print(f"Results saved to:\n  {out_path}")
        print(f"  Total runs in file: {len(final_df)}")

        print(f"\nKGE summary (ceiling ~0.912):")
        print(f"  Min:    {final_df['kge'].min():.3f}")
        print(f"  Median: {final_df['kge'].median():.3f}")
        print(f"  Max:    {final_df['kge'].max():.3f}")

        print(f"\n  Top 10 runs by KGE:")
        cols = ["run_id", "Ks_mult", "kinemvelcoef", "flowexp",
                "channelroughness", "kge", "nse", "pbias_pct",
                "peak_timing_error_hr"]
        available = [c for c in cols if c in final_df.columns]
        print(final_df[available].head(10).to_string(index=False,
                                                      float_format="%.4f"))

        print(f"\n  Parameter-KGE correlations (Pearson r):")
        for param in ["Ks_mult", "kinemvelcoef", "flowexp", "channelroughness"]:
            if param in final_df.columns:
                r = final_df[param].corr(final_df["kge"])
                print(f"    {param:<22s}  r = {r:+.3f}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()