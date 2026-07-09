"""
run_lhs_synth_4param_series.py
===============================
Latin Hypercube Sampling sweep for the 4-parameter synthetic inversion
experiment (series 91 or 92). Scores simulated hydrographs against a
synthetic truth .qout file (not the real gauge) to test parameter
identifiability and equifinality.

Switch series by editing ACTIVE_SERIES below (one line).

Parameters swept (4 free dimensions, both series):
    Ks_mult:          7.5 - 9.5   (true = 8.5,  50th pct of range)
    kinemvelcoef:     2.5 - 6.5   (true = 4.5,  50th pct of range)
    flowexp:          0.18 - 0.35 (true = 0.24, 37th pct of range)
    channelroughness: series 91: 0.02-0.04 | series 92: 0.02-0.10
                       (widened in 92 -- n was unidentifiable in 91,
                       range too narrow to detect any signal even in
                       phase-specific metrics)

Fixed (not swept, both series):
    f_RS_abs: 0.020 mm^-1

Truth exclusion:
    Any LHS sample matching TRUTH_VALUES within TRUTH_TOL=1e-6 is skipped
    automatically. The truth run (SMF_20140812_60_Ks8p5x) must never
    appear in the inversion ensemble.

KGE ceiling:
    ~0.912 (resampling asymmetry between sim .mean() and obs .interpolate(),
    shared by both series). Interpret recovery as top runs clustering near
    the true value lines, not as KGE approaching 1.0.

Usage (run from the smf_demo directory):
    python run_lhs_synth_4param_series.py              # default n for the active series, seed=42
    python run_lhs_synth_4param_series.py --n 100      # recommended
    python run_lhs_synth_4param_series.py --seed 99    # different seed
    python run_lhs_synth_4param_series.py --skip_existing  # resume interrupted run

======================================================================
UPDATED / CONSOLIDATED -- see notes below
======================================================================
This script merges two near-identical files:
    - run_lhs_synth_4param.py     (series 91)
    - run_lhs_synth_4param_92.py  (series 92 -- widens channelroughness to
                                    0.02-0.10 and sets OPINTRVL=0.0833 hr;
                                    series 91 left OPINTRVL at pytRIBS's
                                    default of 1 hr, which caused staircase
                                    discretization in every timing metric)
A diff of the two originals showed only channelroughness's upper bound and
the OPINTRVL line differed -- truth values, run-ID format, the main loop,
and output columns were byte-identical. Companion to
plot_lhs_synth_4param_series.py, which already consolidated the matching
plot scripts for these same two series.

Series 93 (deliberately different truth, channelroughness=0.075, testing a
different hydrograph shape) stays separate here, same as it did on the plot
side -- it's a documented one-off branch, not a lineage successor, so it
isn't a third preset in this file. See run_lhs_synth_4param_93.py.

The old files (run_lhs_synth_4param.py, run_lhs_synth_4param_92.py) should
be deleted/archived once this replacement is verified against a saved
results CSV from each series.
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

# ======================================================================
# SERIES PRESETS
# ======================================================================
SERIES_PRESETS = {
    "91": dict(
        series_label="91",
        category="91_lhs_synth_inversion",
        out_csv="lhs_results_synth_4param_91.csv",
        lhs_params={
            "Ks_mult":          (7.5,  9.5),
            "kinemvelcoef":     (2.5,  6.5),
            "flowexp":          (0.18, 0.35),
            "channelroughness": (0.02, 0.04),
        },
        opintrvl=None,     # not set -- defaults to pytRIBS's 1 hr (S91 behavior;
                           # caused staircase discretization in timing metrics)
        default_n=50,
    ),
    "92": dict(
        series_label="92",
        category="92_lhs_synth_inversion",
        out_csv="lhs_results_synth_4param_92.csv",
        lhs_params={
            "Ks_mult":          (7.5,  9.5),
            "kinemvelcoef":     (2.5,  6.5),
            "flowexp":          (0.18, 0.35),
            "channelroughness": (0.02, 0.10),   # widened from series 91
        },
        opintrvl=0.0833,   # 5-minute output; fixes S91 discretization
        default_n=50,
    ),
}

# ======================================================================
# CONFIG -- edit this one line to switch series
# ======================================================================
ACTIVE_SERIES = "92"
# ======================================================================

if ACTIVE_SERIES not in SERIES_PRESETS:
    raise ValueError(f"ACTIVE_SERIES must be one of {list(SERIES_PRESETS)}, got {ACTIVE_SERIES!r}")

PRESET = SERIES_PRESETS[ACTIVE_SERIES]

F_RS_ABS_FIXED = 0.020   # mm^-1 -- fixed in both series

TRUTH_VALUES = {
    "Ks_mult":          8.5,
    "kinemvelcoef":     4.5,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}
TRUTH_TOL = 1e-6


def is_truth_run(sample: dict) -> bool:
    """Return True if all four values match the truth within TRUTH_TOL."""
    return all(abs(sample[p] - TRUTH_VALUES[p]) < TRUTH_TOL for p in TRUTH_VALUES)


# ------------------------------------------------------------------
# LHS SAMPLE GENERATION (identical across both original scripts)
# ------------------------------------------------------------------
def generate_lhs_samples(n, lhs_params, seed=None):
    """
    Generate n Latin Hypercube samples across the parameter ranges.
    Each parameter range is divided into n equal intervals; one sample is
    drawn uniformly from each interval, then independently shuffled across
    parameters (ensuring full marginal coverage).
    """
    rng     = np.random.default_rng(seed)
    samples = {}
    for param, (lo, hi) in lhs_params.items():
        intervals = np.linspace(lo, hi, n + 1)
        points    = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        samples[param] = points
    return pd.DataFrame(samples)


# ------------------------------------------------------------------
# RUN ID CONSTRUCTION
# ------------------------------------------------------------------
def build_lhs_run_id(sample: dict, preset=PRESET):
    """Build a compact, human-readable run ID for a 4-parameter LHS point."""
    ks_lbl = builder.value_to_label(sample["Ks_mult"])
    cv_lbl = builder.value_to_label(sample["kinemvelcoef"])
    r_lbl  = builder.value_to_label(sample["flowexp"])
    n_lbl  = builder.value_to_label(sample["channelroughness"])
    change_tested = f"Ks{ks_lbl}x_cv{cv_lbl}_r{r_lbl}_n{n_lbl}"
    run_id = f"{builder.LOCATION}_{builder.EVENT_DATE}_{preset['series_label']}_{change_tested}"
    return run_id, change_tested


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
def build_and_run_lhs(sample: dict, preset=PRESET):
    """
    Build one tRIBS input file for a 4-parameter LHS point and run it.

    Patches builder.BASELINE temporarily so all four swept parameters are
    applied correctly; reverts on exit regardless of success/failure.
    f_RS_abs is pinned at F_RS_ABS_FIXED. OPINTRVL is set per-preset (see
    SERIES_PRESETS docstring notes above).

    Returns a metrics dict from run_sensitivity_single.run_and_score().
    """
    ks_mult          = sample["Ks_mult"]
    kinemvelcoef     = sample["kinemvelcoef"]
    flowexp          = sample["flowexp"]
    channelroughness = sample["channelroughness"]

    run_id, change_tested = build_lhs_run_id(sample, preset)

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir    = project_root / "calibration_work"

    run_input_dir      = calib_dir / "01_run_inputs"  / preset["category"]
    run_results_dir    = calib_dir / "02_results"     / preset["category"] / run_id
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
        baseline = builder.BASELINE

        proj = Project(os.getcwd(), builder.LOCATION, builder.EPSG)

        # --- Land use ---
        landuse_ras = '../smf_init_data/LandUse.asc'
        shutil.copy(landuse_ras, proj.directories['land'])

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

        working_soil_table = Path("data/model/soil/soil.sdt")
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
        model.optsnow['value']       = 0      # never set to 1 -- see project notes
        model.optlanduse['value']    = 0

        model.optpercolation['value']      = baseline["optpercolation"]
        model.channelconductivity['value'] = baseline["channelconductivity_mmhr"] / 3.6e6
        model.channelporosity['value']     = baseline["channelporosity"]

        model.kinemvelcoef['value']      = kinemvelcoef
        model.flowexp['value']           = flowexp
        model.channelroughness['value']  = channelroughness
        model.channelwidthcoeff['value'] = baseline["channelwidthcoeff"]

        model.startdate['value']  = builder.START_DATE
        model.runtime['value']    = builder.RUNTIME_HOURS
        model.rainintrvl['value'] = builder.RAIN_INTERVAL
        if preset["opintrvl"] is not None:
            model.opintrvl['value'] = preset["opintrvl"]

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
            "run_number":                preset["series_label"],
            "change_tested":             change_tested,
            "run_id":                    run_id,
            "run_category":              preset["category"],
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
        description=f"Series {PRESET['series_label']} synthetic inversion LHS sweep -- 4 free parameters.")
    parser.add_argument("--n", type=int, default=PRESET["default_n"],
                        help=f"Number of LHS samples (default: {PRESET['default_n']})")
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
    out_path = summary_dir / PRESET["out_csv"]

    samples = generate_lhs_samples(args.n, PRESET["lhs_params"], seed=args.seed)

    lp = PRESET["lhs_params"]
    print(f"\n{'='*65}")
    print(f"LHS sweep -- Series {PRESET['series_label']} -- Synthetic Inversion -- 4 free params")
    print(f"  ({args.n} samples, seed={args.seed})")
    print(f"  Ks_mult:          {lp['Ks_mult'][0]:.2f} - {lp['Ks_mult'][1]:.2f}x        [true = {TRUTH_VALUES['Ks_mult']}]")
    print(f"  kinemvelcoef:     {lp['kinemvelcoef'][0]:.2f} - {lp['kinemvelcoef'][1]:.2f}          [true = {TRUTH_VALUES['kinemvelcoef']}]")
    print(f"  flowexp:          {lp['flowexp'][0]:.2f} - {lp['flowexp'][1]:.2f}          [true = {TRUTH_VALUES['flowexp']}]")
    print(f"  channelroughness: {lp['channelroughness'][0]:.3f} - {lp['channelroughness'][1]:.3f}        [true = {TRUTH_VALUES['channelroughness']}]")
    print(f"  f_RS_abs:         {F_RS_ABS_FIXED} mm^-1    [FIXED]")
    print(f"  OPINTRVL:         {'0.0833 hr (5-min output)' if PRESET['opintrvl'] else '1 hr (pytRIBS default -- S91 only)'}")
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
        sample = {p: row[p] for p in PRESET["lhs_params"]}

        # --- Truth exclusion guard ---
        if is_truth_run(sample):
            print(f"\n[{i+1:>3}/{args.n}]  EXCLUDED: sample matches truth values "
                  f"exactly -- skipped by design.")
            excluded += 1
            continue

        run_id, _ = build_lhs_run_id(sample)

        print(f"\n[{i+1:>3}/{args.n}]  Ks={sample['Ks_mult']:.3f}x  cv={sample['kinemvelcoef']:.3f}  "
              f"r={sample['flowexp']:.3f}  n={sample['channelroughness']:.4f}")
        print(f"         -> {run_id}")

        if args.skip_existing and csv_already_exists(run_id, calib_dir):
            print(f"  SKIP (CSV exists): {run_id}")
            skipped += 1
            metrics_file = summary_dir / f"{run_id}_metrics_summary.csv"
            if metrics_file.exists() and run_id not in existing_run_ids:
                try:
                    df_m = pd.read_csv(metrics_file)
                    m    = df_m.iloc[0].to_dict()
                    m.update(sample)
                    m["f_RS_abs"] = F_RS_ABS_FIXED
                    all_results.append(m)
                except Exception:
                    pass
            continue

        t0 = time.time()
        try:
            metrics = build_and_run_lhs(sample)
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
