"""
run_lhs_synth_Ks_f_97log.py
=============================
Series 97log -- Joint Ks_mult x f_RS_abs LHS sweep, scored against SYNTHETIC
TRUTH, with cv/r/n pinned at the confirmed truth values throughout.

This is a log-f-stratified variant of run_lhs_synth_Ks_f_97.py. The ONLY
methodological difference is in generate_lhs_samples(): f_RS_abs (the
y-axis parameter in plot_lhs_Ks_f_97.py) is now stratified into n equal-
width bins in log10-space rather than linear space, then exponentiated
back. Ks_mult stays linearly stratified.

Why: f_RS_abs spans 0.003-0.05 (>1 order of magnitude). Under linear
stratification, the low-f region -- where the PBIAS=0 valley floor and
the Series 96 anchors (f=0.0075, f=0.011) actually live -- gets compressed
into a handful of bins, while the sparser high-f region (0.03-0.05) gets
the same bin density despite being less scientifically interesting. Log
stratification gives proportionally even coverage on both sides of the
valley floor instead.

Output is written to a SEPARATE file (lhs_results_synth_Ks_f_97log.csv)
and uses a separate run_id/category prefix (97log / 97log_lhs_synth_Ks_f)
so this sweep's runs never collide with or overwrite the original linear
Series 97 sweep -- both result sets stay on disk side by side for direct
comparison of linear- vs log-stratified sampling density.

REQUIRES exactly one *.qout file in calibration_work/synth_truth/ before
running (this activates SYNTHETIC TRUTH MODE automatically inside
run_sensitivity_single.py). The script checks this itself at startup and
will refuse to run otherwise -- see the safety check in main().

Usage (run from the smf_demo directory):
    python run_lhs_synth_Ks_f_97log.py                  # 200 samples, seed=42
    python run_lhs_synth_Ks_f_97log.py --n 400           # more samples
    python run_lhs_synth_Ks_f_97log.py --seed 7           # different seed
    python run_lhs_synth_Ks_f_97log.py --skip_existing    # resume interrupted run

Output:
    calibration_work/03_comparisons/summary_tables/lhs_results_synth_Ks_f_97log.csv

Plotting: 
    plot_lhs_Ks_f_97.py
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
# CONFIRMED SYNTHETIC TRUTH VALUES (verified via md5 checksum,
# 2026-07-06). cv/r/n are PINNED at these values for every run in this
# sweep; Ks_mult/f_RS_abs are the two swept parameters.
# ------------------------------------------------------------------
TRUTH_VALUES = {
    "Ks_mult":          8.5,
    "f_RS_abs":         0.020,
    "kinemvelcoef":     4.5,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}
TRUTH_TOL = 1e-6

PINNED_CV = TRUTH_VALUES["kinemvelcoef"]
PINNED_R  = TRUTH_VALUES["flowexp"]
PINNED_N  = TRUTH_VALUES["channelroughness"]

# ------------------------------------------------------------------
# LHS PARAMETER RANGES
#
# Ks_mult: linearly stratified, same range as Series 97 (3-11x).
#
# f_RS_abs: LOG-stratified (scale="log") -- see module docstring. Same
#   bounds as Series 97 (0.003-0.05), padded on both sides of the fitted
#   valley-floor curve so the grid captures the response-surface "walls"
#   as well as the floor. Truth (0.020) stays well interior to the bounds
#   in log-space too (log10(0.003)=-2.52, log10(0.020)=-1.70,
#   log10(0.05)=-1.30).
# ------------------------------------------------------------------
LHS_PARAMS = {
    "Ks_mult":  {"lo": 3.0,   "hi": 11.0, "scale": "linear"},  # true = 8.5
    "f_RS_abs": {"lo": 0.003, "hi": 0.05, "scale": "log"},     # true = 0.020 (y-axis, log-stratified)
}

LHS_SERIES   = "97log"
LHS_CATEGORY = "97log_lhs_synth_Ks_f"


def is_truth_run(ks_mult, f_rs_abs):
    """True only if the sampled (Ks, f) point happens to equal true values
    exactly. cv/r/n are always pinned at truth in this script, so matching
    Ks/f alone is sufficient to identify (and exclude) the truth point."""
    return (abs(ks_mult - TRUTH_VALUES["Ks_mult"])   < TRUTH_TOL and
            abs(f_rs_abs - TRUTH_VALUES["f_RS_abs"]) < TRUTH_TOL)


# ------------------------------------------------------------------
# LHS SAMPLE GENERATION -- linear stratification by default, log
# stratification when a parameter's bounds dict sets scale="log".
# ------------------------------------------------------------------
def generate_lhs_samples(n, params, seed=None):
    rng     = np.random.default_rng(seed)
    samples = {}
    for param, bounds in params.items():
        lo, hi = bounds["lo"], bounds["hi"]
        scale  = bounds.get("scale", "linear")

        if scale == "log":
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            intervals  = np.linspace(log_lo, log_hi, n + 1)
            log_points = rng.uniform(intervals[:-1], intervals[1:])
            points     = 10 ** log_points
        else:
            intervals = np.linspace(lo, hi, n + 1)
            points    = rng.uniform(intervals[:-1], intervals[1:])

        rng.shuffle(points)
        samples[param] = points
    return pd.DataFrame(samples)


def build_lhs_run_id(ks_mult, f_rs_abs):
    ks_lbl = builder.value_to_label(ks_mult)
    f_lbl  = builder.value_to_label(f_rs_abs)
    change_tested = f"synthKsflog_Ks{ks_lbl}x_f{f_lbl}"
    run_id = f"{builder.LOCATION}_{builder.EVENT_DATE}_{LHS_SERIES}_{change_tested}"
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
# BUILD + RUN ONE (Ks_mult, f_RS_abs) POINT -- cv/r/n pinned at truth
# ------------------------------------------------------------------
def build_and_run_lhs(ks_mult, f_rs_abs):
    run_id, change_tested = build_lhs_run_id(ks_mult, f_rs_abs)

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
    builder.BASELINE["f_RS_abs"]         = f_rs_abs
    builder.BASELINE["kinemvelcoef"]     = PINNED_CV
    builder.BASELINE["flowexp"]          = PINNED_R
    builder.BASELINE["channelroughness"] = PINNED_N

    try:
        baseline = builder.BASELINE

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
                # f: RS soil (ID '1') uses this sample's f_rs_abs; all
                # others use baseline soil-table f. Applied UNCONDITIONALLY
                # (not gated on a swept-param-name check) -- see the
                # build_sensitivity_run.py bug found during the Series 96
                # session, where a gated version silently discarded this
                # override whenever f_RS_abs wasn't itself the "swept"
                # parameter name.
                soil_cls['f'] = f_rs_abs if cid == '1' else soil_params['f']
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
        model.optsnow['value']       = 0      # CRITICAL: must stay 0 -- SMF.in has this wrong
        model.optlanduse['value']    = 0

        model.optpercolation['value']      = baseline["optpercolation"]
        model.channelconductivity['value'] = baseline["channelconductivity_mmhr"] / 3.6e6
        model.channelporosity['value']     = baseline["channelporosity"]

        model.kinemvelcoef['value']      = PINNED_CV
        model.flowexp['value']           = PINNED_R
        model.channelroughness['value']  = PINNED_N
        model.channelwidthcoeff['value'] = baseline["channelwidthcoeff"]

        model.startdate['value']   = builder.START_DATE
        model.runtime['value']     = builder.RUNTIME_HOURS
        model.rainintrvl['value']  = builder.RAIN_INTERVAL
        model.opintrvl['value']    = 0.0833   # 5-minute output

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

        print(f"  Ks={ks_mult:.3f}x  f={f_rs_abs:.4f}  "
              f"(cv={PINNED_CV} r={PINNED_R} n={PINNED_N} -- pinned at truth)")

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
            "f_RS_abs":                  f_rs_abs,
            "As_value":                  baseline["As_value"],
            "Au_value":                  baseline["Au_value"],
            "optpercolation":            baseline["optpercolation"],
            "channelconductivity_mmhr":  baseline["channelconductivity_mmhr"],
            "channelporosity":           baseline["channelporosity"],
            "kinemvelcoef":              PINNED_CV,
            "flowexp":                   PINNED_R,
            "channelroughness":          PINNED_N,
            "channelwidthcoeff":         baseline["channelwidthcoeff"],
            "input_file":                input_file,
            "log_file":                  log_file,
            "output_prefix":             output_prefix,
            "csv_export_dir":            os.path.relpath(csv_export_dir,      script_dir),
            "plot_export_dir":           os.path.relpath(plot_export_dir,     script_dir),
            "summary_export_dir":        os.path.relpath(summary_export_dir,  script_dir),
            "swept_param":               "lhs_synth_Ks_f_log",
            "swept_value":               ks_mult,
        }

        config_path = calib_dir / "current_run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))

    finally:
        builder.BASELINE = original_baseline

    metrics = runner.run_and_score()

    metrics["Ks_mult"]          = ks_mult
    metrics["f_RS_abs"]         = f_rs_abs
    metrics["kinemvelcoef"]     = PINNED_CV
    metrics["flowexp"]          = PINNED_R
    metrics["channelroughness"] = PINNED_N
    metrics["swept_param"]      = "lhs_synth_Ks_f_log"

    return metrics


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Series 97log -- joint Ks_mult x f_RS_abs LHS sweep "
                    "against synthetic truth, cv/r/n pinned at truth values, "
                    "f_RS_abs log-stratified.")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of LHS samples (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose compare CSV already exists")
    args = parser.parse_args()

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir   = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # SAFETY CHECK: confirm synthetic-truth mode will actually activate.
    # ------------------------------------------------------------------
    synth_dir = calib_dir / "synth_truth"
    qout_files = list(synth_dir.glob("*.qout")) if synth_dir.exists() else []
    if len(qout_files) != 1:
        raise RuntimeError(
            f"Expected exactly one *.qout file in {synth_dir} to activate "
            f"synthetic-truth mode, found {len(qout_files)}: "
            f"{[f.name for f in qout_files]}. Move any extra files into "
            f"synth_truth_archive/ before running this sweep, or results "
            f"won't be scored against the truth you expect."
        )
    print(f"Synthetic truth mode confirmed active: {qout_files[0].name}")

    samples = generate_lhs_samples(args.n, LHS_PARAMS, seed=args.seed)

    out_path = summary_dir / f"lhs_results_synth_Ks_f_{LHS_SERIES}.csv"
    existing_df      = load_existing_results(out_path)
    existing_run_ids = (set(existing_df["run_id"].values)
                        if not existing_df.empty else set())

    results = []
    if not existing_df.empty:
        results.extend(existing_df.to_dict("records"))

    print(f"\n{'='*70}")
    print(f"LHS sweep -- Series {LHS_SERIES} -- Ks_mult x f_RS_abs vs SYNTHETIC TRUTH")
    print(f"  ({args.n} samples, seed={args.seed})")
    print(f"  Ks_mult:  {LHS_PARAMS['Ks_mult']['lo']:.1f} - "
          f"{LHS_PARAMS['Ks_mult']['hi']:.1f}x        [true = {TRUTH_VALUES['Ks_mult']}]  (linear)")
    print(f"  f_RS_abs: {LHS_PARAMS['f_RS_abs']['lo']:.4f} - "
          f"{LHS_PARAMS['f_RS_abs']['hi']:.4f}   [true = {TRUTH_VALUES['f_RS_abs']}]  (LOG-stratified)")
    print(f"  PINNED:   cv={PINNED_CV}  r={PINNED_R}  n={PINNED_N}  (truth values)")
    print(f"{'='*70}\n")

    completed, skipped, excluded, failed = 0, 0, 0, 0
    sweep_start = time.time()

    for i, row in samples.iterrows():
        ks_mult  = row["Ks_mult"]
        f_rs_abs = row["f_RS_abs"]

        if is_truth_run(ks_mult, f_rs_abs):
            print(f"[{i+1:>3}/{args.n}]  EXCLUDED: matches truth exactly.")
            excluded += 1
            continue

        run_id, _ = build_lhs_run_id(ks_mult, f_rs_abs)

        print(f"\n[{i+1:>3}/{args.n}]  Ks={ks_mult:.3f}x  f={f_rs_abs:.4f}")
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
                    m["f_RS_abs"]         = f_rs_abs
                    m["kinemvelcoef"]     = PINNED_CV
                    m["flowexp"]          = PINNED_R
                    m["channelroughness"] = PINNED_N
                    results.append(m)
                except Exception:
                    pass
            continue

        t0 = time.time()
        try:
            metrics = build_and_run_lhs(ks_mult, f_rs_abs)
            results = [r for r in results if r.get("run_id") != run_id]
            results.append(metrics)
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

        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"\nSweep complete: {completed} ran, {skipped} skipped, "
          f"{excluded} excluded, {failed} failed")

    if results:
        final_df = pd.DataFrame(results).sort_values("kge", ascending=False)
        final_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path.name}  ({len(final_df)} rows)")

        # ------------------------------------------------------------------
        # Coverage check, same intent as Series 97 -- flags whether the
        # high-Ks region (>=9x) ended up sparsely sampled.
        # ------------------------------------------------------------------
        high_ks_n = final_df[final_df["Ks_mult"] >= 9.0].shape[0]
        print(f"\nCoverage check: {high_ks_n} of {len(final_df)} points have "
              f"Ks_mult >= 9.0 (the region bisection couldn't resolve).")
        if high_ks_n < 20:
            print("  NOTE: this region looks sparse -- consider a supplemental "
                  "targeted batch (e.g. rerun with LHS_PARAMS Ks_mult lo=9.0) "
                  "before drawing conclusions about the high-Ks sign reversal.")

        # ------------------------------------------------------------------
        # Coverage check for the low-f region (<0.007) -- this is the band
        # log-stratification was specifically meant to densify relative to
        # the original linear Series 97 sweep.
        # ------------------------------------------------------------------
        low_f_n = final_df[final_df["f_RS_abs"] < 0.007].shape[0]
        print(f"Coverage check: {low_f_n} of {len(final_df)} points have "
              f"f_RS_abs < 0.007 (near-anchor-A/B region).")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
