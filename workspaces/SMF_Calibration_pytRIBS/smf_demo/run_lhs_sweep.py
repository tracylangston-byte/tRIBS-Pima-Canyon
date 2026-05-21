"""
run_lhs_sweep.py
================
Runs a Latin Hypercube Sampling sweep jointly across Ks_mult and f_RS_abs.
Explores the two-parameter interaction space identified as most important
from the single-parameter sensitivity sweeps.

Results are saved to a SEPARATE CSV from sensitivity_results_all.csv so
the LHS runs don't get mixed up with the single-parameter sweeps.

Usage (run from the smf_demo directory):
    python run_lhs_sweep.py                    # runs all samples
    python run_lhs_sweep.py --n 50             # run 50 samples (default: 75)
    python run_lhs_sweep.py --skip_existing    # skip runs whose CSV already exists
    python run_lhs_sweep.py --seed 42          # set random seed for reproducibility

Output:
    calibration_work/03_comparisons/summary_tables/lhs_results_Ks_f.csv

Parameter ranges (edit below before running):
    Ks_mult:   3.0 – 10.0   (viable calibration zone from Ks sweep)
    f_RS_abs:  0.010 – 0.050 (physically meaningful range from f sweep,
                               smooth response, PBIAS zero-crossing zone)

Why these ranges:
    Ks sweep showed viable zone at 3–10×; best KGE at 4–6×.
    f sweep showed smooth, physically meaningful response at 0.01–0.05;
    baseline f=0.02 is where PBIAS crosses zero at Ks=7×.
    Both ranges are intentionally conservative — staying within the
    physically plausible space where the model behaves well.
"""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

import build_sensitivity_run as builder
import run_sensitivity_single as runner

# ------------------------------------------------------------------
# LHS PARAMETER RANGES
# Edit these before running if you want a different search space.
# ------------------------------------------------------------------
LHS_PARAMS = {
    "Ks_mult":   {"lo": 3.0,   "hi": 10.0},
    "f_RS_abs":  {"lo": 0.010, "hi": 0.050},
}

# Series number for LHS runs — kept separate from single-param series 60-65
LHS_SERIES = "70"


def generate_lhs_samples(n, params, seed=None):
    """
    Generate n Latin Hypercube samples across the parameter ranges.
    Returns a DataFrame with one column per parameter and n rows.
    Each parameter range is divided into n equal intervals and one
    sample is drawn uniformly from each interval, then shuffled
    independently per parameter.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for param, bounds in params.items():
        lo, hi = bounds["lo"], bounds["hi"]
        # Divide into n equal intervals and sample one point per interval
        intervals = np.linspace(lo, hi, n + 1)
        points = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        samples[param] = points
    return pd.DataFrame(samples)


def build_lhs_run_id(ks_mult, f_rs_abs):
    """Build a run ID for a two-parameter LHS run."""
    ks_label = builder.value_to_label(ks_mult)
    f_label  = builder.value_to_label(f_rs_abs)
    change_tested = f"Ks{ks_label}x_fRS{f_label}"
    run_id = f"{builder.LOCATION}_{builder.EVENT_DATE}_{LHS_SERIES}_{change_tested}"
    return run_id, change_tested


def csv_already_exists(run_id, calib_dir):
    csv_path = (calib_dir / "03_comparisons" / "csv_exports"
                / f"{run_id}_compare_obs_sim.csv")
    return csv_path.exists()


def load_existing_lhs(out_path):
    if out_path.exists():
        try:
            df = pd.read_csv(out_path)
            print(f"  Loaded existing LHS results: {len(df)} rows")
            return df
        except Exception as e:
            print(f"  Warning: could not load existing LHS results ({e}). Starting fresh.")
    return pd.DataFrame()


def build_and_run_lhs(ks_mult, f_rs_abs):
    """
    Build and run one LHS point. Temporarily overrides PARAM_CONFIG and
    BASELINE in the builder so the run gets the right ID and parameters.
    Writes a custom current_run_config.json and calls run_and_score().
    """
    run_id, change_tested = build_lhs_run_id(ks_mult, f_rs_abs)

    notebook_dir = Path.cwd()
    project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
    calib_dir    = project_root / "calibration_work"

    run_category       = "70_lhs"
    run_input_dir      = calib_dir / "01_run_inputs"  / run_category
    run_results_dir    = calib_dir / "02_results"     / run_category / run_id
    csv_export_dir     = calib_dir / "03_comparisons" / "csv_exports"
    plot_export_dir    = calib_dir / "03_comparisons" / "hydrograph_plots"
    summary_export_dir = calib_dir / "03_comparisons" / "summary_tables"
    log_dir            = calib_dir / "06_logs"

    for folder in [run_input_dir, run_results_dir, csv_export_dir,
                   plot_export_dir, summary_export_dir, log_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # Temporarily patch PARAM_CONFIG so build_input_file uses LHS series/prefix
    original_config = builder.PARAM_CONFIG.copy()
    builder.PARAM_CONFIG["f_RS_abs"] = {
        "series": LHS_SERIES,
        "prefix": "Ks",
        "suffix": "",
        "type":   "absolute",
    }

    # Temporarily patch BASELINE so both Ks and f are set correctly
    original_baseline = builder.BASELINE.copy()
    builder.BASELINE["Ks_mult"]  = ks_mult
    builder.BASELINE["f_RS_abs"] = f_rs_abs

    try:
        # Use build_input_file but intercept to set the right run_id/paths
        # We do this by calling it directly and then fixing the config JSON
        import shutil
        import json as json_mod
        from pytRIBS.classes import Project, Soil, Land, Met, Model
        from pytRIBS.shared.inout import InOut

        name = builder.LOCATION
        proj = Project(os.getcwd(), name, builder.EPSG)

        landuse_ras = '../smf_init_data/LandUse.asc'
        shutil.copy(landuse_ras, proj.directories['land'])
        landuse_ras = f"{proj.directories['land']}/{os.path.basename(landuse_ras)}"

        soil_ras = '../smf_init_data/ADOT_SoilTypes.asc'
        shutil.copy(soil_ras, proj.directories['soil'])
        soil_ras = f"{proj.directories['soil']}/{os.path.basename(soil_ras)}"

        soil = Soil(meta=proj.meta)
        shutil.copy('../smf_init_data/SOLUS_Bedrock_m.asc', proj.directories['soil'])
        soil.bedrockfile['value'] = f"{proj.directories['soil']}/SOLUS_Bedrock_m.asc"
        shutil.copy('../smf_init_data/InitGW_95pct_mm.asc', proj.directories['soil'])
        soil.gwaterfile['value']  = f"{proj.directories['soil']}/InitGW_95pct_mm.asc"
        shutil.copy('../smf_init_data/soils.sdt', proj.directories['soil'])
        soil.soiltablename['value'] = f"{proj.directories['soil']}/soils.sdt"
        soil.soilmapname['value']   = soil_ras

        soil_table = soil.read_soil_table(textures=True)
        for cls in soil_table:
            cls['As'] = builder.BASELINE["As_value"]
            cls['Au'] = builder.BASELINE["Au_value"]
            cls['ks'] = 0.7
            cls['Cs'] = 1.4e6
            cid = str(cls['ID'])
            if cid in builder.SOIL_PARAM_LOOKUP:
                sp = builder.SOIL_PARAM_LOOKUP[cid]
                cls['Ks']     = sp['Ks'] * ks_mult
                cls['thetaS'] = sp['thetaS']
                cls['thetaR'] = sp['thetaR']
                cls['m']      = sp['m']
                cls['PsiB']   = sp['PsiB']
                cls['n']      = sp['n']
                # RS soil gets the swept f; all others use lookup baseline
                cls['f'] = f_rs_abs if cid == '1' else sp['f']

        working_soil_table    = Path("data/model/soil/soil.sdt")
        soil.write_soil_table(soil_table, str(working_soil_table), textures=True)
        run_specific_soil_abs = run_input_dir / f"soils_{run_id}.sdt"
        shutil.copy(working_soil_table, run_specific_soil_abs)
        soil.soiltablename['value'] = os.path.relpath(run_specific_soil_abs, notebook_dir)
        soil.optsoiltype['value']   = 0

        land = Land(meta=proj.meta)
        land.landmapname['value']   = f"{proj.directories['land']}/LandUse.asc"
        land.landtablename['value'] = f"{proj.directories['land']}/land_use_params.ldt"
        landuse_list = []
        for lu_id, lp in builder.LAND_PARAM_LOOKUP.items():
            row = lp.copy(); row['ID'] = lu_id; row['a'] = -9999; row['b1'] = -9999
            landuse_list.append(row)
        land.write_landuse_table(landuse_list, land.landtablename['value'])

        met = Met(meta=proj.meta)
        met.hydrometbasename['value'] = name
        met.hydrometstations['value'] = "../smf_init_data/met/Master_Met.sdf"
        met.gaugestations['value']    = "../smf_init_data/met/Master_Precip.sdf"

        model = Model(met=met, land=land, soil=soil, mesh=None, meta=proj.meta)
        model.parallelmode['value']  = 0
        model.optmeshinput['value']  = 1
        model.inputdatafile['value'] = "../smf_init_data/mesh/SMF_mesh"
        model.inputtime['value']     = 0
        model.optbedrock['value']    = 1
        model.optsnow['value']       = 0
        model.optlanduse['value']    = 0

        b = builder.BASELINE
        model.optpercolation['value']      = b["optpercolation"]
        model.channelconductivity['value'] = b["channelconductivity_mmhr"] / 3.6e6
        model.channelporosity['value']     = b["channelporosity"]
        model.kinemvelcoef['value']        = b["kinemvelcoef"]
        model.flowexp['value']             = b["flowexp"]
        model.channelroughness['value']    = b["channelroughness"]
        model.channelwidthcoeff['value']   = b["channelwidthcoeff"]
        model.startdate['value']           = builder.START_DATE
        model.runtime['value']             = builder.RUNTIME_HOURS
        model.rainintrvl['value']          = builder.RAIN_INTERVAL

        input_file_abs    = run_input_dir / f"{run_id}.in"
        log_file_abs      = log_dir / f"{run_id}.log"
        output_prefix_abs = run_results_dir / run_id
        input_file    = os.path.relpath(input_file_abs,    notebook_dir)
        log_file      = os.path.relpath(log_file_abs,      notebook_dir)
        output_prefix = os.path.relpath(output_prefix_abs, notebook_dir)

        model.outfilename['value']      = output_prefix
        model.outhydrofilename['value'] = output_prefix

        model.write_node_file([1960, 1547, 3082], 'data/model/pnodes.dat')
        model.nodeoutputlist['value'] = 'data/model/pnodes.dat'
        model.write_node_file([3202], 'data/model/qnodes.dat')
        model.outletnodelist['value'] = 'data/model/qnodes.dat'

        model.write_input_file(input_file)

        # Print compact audit
        print(f"  Ks={ks_mult:5.2f}×  f={f_rs_abs:.4f} mm⁻¹  "
              f"(RS: Ks={builder.SOIL_PARAM_LOOKUP['1']['Ks']*ks_mult:.1f} mm/hr, "
              f"1/f={1/f_rs_abs:.0f} mm)")

        # Write config JSON for run_sensitivity_single.py
        run_config = {
            "location":                  builder.LOCATION,
            "event_date":                builder.EVENT_DATE,
            "run_number":                LHS_SERIES,
            "change_tested":             change_tested,
            "run_id":                    run_id,
            "run_category":              run_category,
            "start_date":                builder.START_DATE,
            "runtime_hours":             builder.RUNTIME_HOURS,
            "rain_interval_hours":       builder.RAIN_INTERVAL,
            "event_start":               builder.EVENT_START,
            "event_end":                 builder.EVENT_END,
            "Ks_mult":                   ks_mult,
            "f_RS_abs":                  f_rs_abs,
            "As_value":                  b["As_value"],
            "Au_value":                  b["Au_value"],
            "optpercolation":            b["optpercolation"],
            "channelconductivity_mmhr":  b["channelconductivity_mmhr"],
            "channelporosity":           b["channelporosity"],
            "kinemvelcoef":              b["kinemvelcoef"],
            "flowexp":                   b["flowexp"],
            "channelroughness":          b["channelroughness"],
            "channelwidthcoeff":         b["channelwidthcoeff"],
            "input_file":                input_file,
            "log_file":                  log_file,
            "output_prefix":             output_prefix,
            "csv_export_dir":            os.path.relpath(csv_export_dir,      notebook_dir),
            "plot_export_dir":           os.path.relpath(plot_export_dir,     notebook_dir),
            "summary_export_dir":        os.path.relpath(summary_export_dir,  notebook_dir),
            "swept_param":               "lhs_Ks_f",
            "swept_value":               ks_mult,   # primary sweep value for compatibility
        }

        config_path = calib_dir / "current_run_config.json"
        config_path.write_text(json_mod.dumps(run_config, indent=2))

    finally:
        # Always restore original builder state
        builder.PARAM_CONFIG = original_config
        builder.BASELINE     = original_baseline

    # Run tRIBS and score
    metrics = runner.run_and_score()

    # Add both LHS parameter values explicitly to the metrics dict
    metrics["Ks_mult"]  = ks_mult
    metrics["f_RS_abs"] = f_rs_abs
    metrics["swept_param"] = "lhs_Ks_f"

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="LHS sweep over Ks_mult and f_RS_abs jointly.")
    parser.add_argument("--n", type=int, default=75,
                        help="Number of LHS samples (default: 75)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip runs whose compare CSV already exists")
    args = parser.parse_args()

    notebook_dir = Path.cwd()
    project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
    calib_dir    = project_root / "calibration_work"
    summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "lhs_results_Ks_f.csv"

    # Generate LHS samples
    samples = generate_lhs_samples(args.n, LHS_PARAMS, seed=args.seed)

    print(f"\n{'='*60}")
    print(f"LHS sweep: Ks_mult × f_RS_abs  ({args.n} samples, seed={args.seed})")
    print(f"  Ks_mult  range: {LHS_PARAMS['Ks_mult']['lo']:.1f} – "
          f"{LHS_PARAMS['Ks_mult']['hi']:.1f}×")
    print(f"  f_RS_abs range: {LHS_PARAMS['f_RS_abs']['lo']:.3f} – "
          f"{LHS_PARAMS['f_RS_abs']['hi']:.3f} mm⁻¹")
    print(f"  Output: {out_path.name}")
    print(f"{'='*60}\n")

    existing_df = load_existing_lhs(out_path)
    existing_run_ids = set(existing_df["run_id"].values) if not existing_df.empty else set()

    all_results = []
    # Carry forward any existing results not being re-run
    if not existing_df.empty:
        all_results.extend(existing_df.to_dict("records"))

    completed = 0
    skipped   = 0
    failed    = 0
    sweep_start = time.time()

    for i, row in samples.iterrows():
        ks_mult  = row["Ks_mult"]
        f_rs_abs = row["f_RS_abs"]

        run_id, _ = build_lhs_run_id(ks_mult, f_rs_abs)

        print(f"\n[{i+1}/{args.n}] Ks={ks_mult:.3f}×  f={f_rs_abs:.4f} mm⁻¹  →  {run_id}")

        if args.skip_existing and csv_already_exists(run_id, calib_dir):
            print(f"  SKIP (exists): {run_id}")
            skipped += 1
            # Load existing metrics if available
            metrics_file = summary_dir / f"{run_id}_metrics_summary.csv"
            if metrics_file.exists() and run_id not in existing_run_ids:
                df = pd.read_csv(metrics_file)
                m = df.iloc[0].to_dict()
                m["Ks_mult"]  = ks_mult
                m["f_RS_abs"] = f_rs_abs
                all_results.append(m)
            continue

        t0 = time.time()
        try:
            metrics = build_and_run_lhs(ks_mult, f_rs_abs)
            # Remove any old result for this run_id before appending
            all_results = [r for r in all_results if r.get("run_id") != run_id]
            all_results.append(metrics)
            completed += 1

        except Exception as e:
            print(f"  FAILED: {run_id}")
            print(f"  Error: {e}")
            failed += 1

        elapsed = time.time() - t0
        total_elapsed = time.time() - sweep_start
        remaining = args.n - completed - skipped - failed
        if completed > 0:
            avg_time = total_elapsed / completed
            eta_min  = (avg_time * remaining) / 60
            print(f"  Run time: {elapsed/60:.1f} min  |  ETA: {eta_min:.0f} min remaining")

        # Save incrementally after every run so progress isn't lost if interrupted
        if all_results:
            pd.DataFrame(all_results).to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # Final save and summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"LHS sweep complete:  {completed} ran,  {skipped} skipped,  {failed} failed")
    print(f"{'='*60}\n")

    if all_results:
        final_df = pd.DataFrame(all_results).sort_values("kge", ascending=False)
        final_df.to_csv(out_path, index=False)
        print(f"Results saved to:\n  {out_path}")
        print(f"  Total runs in file: {len(final_df)}")

        # Print top 10 by KGE
        print(f"\n  Top 10 runs by KGE:")
        cols = ["run_id", "Ks_mult", "f_RS_abs", "kge", "nse",
                "pbias_pct", "peak_timing_error_hr", "sim_peak_m3s"]
        available = [c for c in cols if c in final_df.columns]
        print(final_df[available].head(10).to_string(index=False, float_format="%.3f"))
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
