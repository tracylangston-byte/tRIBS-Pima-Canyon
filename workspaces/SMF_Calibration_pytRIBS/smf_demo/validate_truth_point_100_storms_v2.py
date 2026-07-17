"""
validate_truth_point_100_storms_v2.py
======================================
RESAMPLING-METHOD VARIANT of validate_truth_point_100_storms.py.

Same diagnostic as the original: runs tRIBS ONCE per storm magnitude
(storm080, 100_narrow, storm125) at the EXACT true parameter values
(Ks_mult=7.0x, f_RS_abs=0.012, cv=4.5, r=0.24, n=0.026), scored against
that storm's own synthetic truth. The ONLY difference from the original
script is which scoring module it calls: this version calls
run_sensitivity_single_interp.py instead of run_sensitivity_single.py,
so the simulated series is resampled to the 5-minute comparison grid by
time-interpolation rather than mean-aggregation. See the docstring in
run_sensitivity_single_interp.py for the full rationale.

Motivation (see Handoff_Series100_TruthPointAnomaly_v4.md)
------------------------------------------------------------
The mechanical/procedural investigation into the truth-point PBIAS
anomaly (v4 of the handoff doc) is closed -- every pipeline explanation
(inputs, binary, determinism, execution context, restart state, commit
provenance, full from-scratch regeneration) has been ruled out. Section
7.7 localized the entire volume residual to a ~45-minute window at and
immediately after the peak, with 15+ hours of recession/tail matching
almost perfectly. That is exactly the signature you'd expect if a
resampling step were flattening the peak -- and the original scoring
script bin-averages the sim series to 5-minute bins even though the sim
is already written on a 5-minute grid (opintrvl=0.0833 hr), so any
sub-bin timestamp misalignment gets smoothed out right where the
hydrograph is changing fastest. This script tests that specific
hypothesis directly, holding every other input identical to the original
run.

This is NOT a replacement for validate_truth_point_100_storms.py -- both
are kept, per standing practice, so the mean-aggregation and
time-interpolation results can be compared side by side rather than one
overwriting the other.

REQUIRES (same as the original):
  - calibration_work/synth_truth/*.qout             (exactly one -- baseline/100_narrow truth)
  - calibration_work/synth_truth/storm080/*.qout     (exactly one)
  - calibration_work/synth_truth/storm125/*.qout     (exactly one)
  - run_sensitivity_single_interp.py in the same directory

Usage (run from the smf_demo directory):
    python validate_truth_point_100_storms_v2.py
    python validate_truth_point_100_storms_v2.py --timeout 600

Output:
    calibration_work/03_comparisons/summary_tables/truth_point_validation_100_storms_interp.csv
    Prints the same comparison table as the original, plus a note on how
    to diff it against truth_point_validation_100_storms.csv.
"""

import argparse
import os
import sys
import signal
import subprocess
import time
import json
import shutil
import pandas as pd
from pathlib import Path

import build_sensitivity_run as builder
from pytRIBS.classes import Project, Soil, Land, Met, Model

# ------------------------------------------------------------------
# THE TRUE PARAMETER POINT -- identical across all storms; only the
# forcing/truth-file changes per storm. This is the exact point every
# sibling LHS sweep excludes from its own ensemble.
# ------------------------------------------------------------------
TRUTH_VALUES = {
    "Ks_mult":          7.0,
    "f_RS_abs":         0.012,
    "kinemvelcoef":     4.5,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}
PINNED_CV = TRUTH_VALUES["kinemvelcoef"]
PINNED_R  = TRUTH_VALUES["flowexp"]
PINNED_N  = TRUTH_VALUES["channelroughness"]

# ------------------------------------------------------------------
# STORM CONFIGS -- mirrors the GAUGE_SDF / truth_file override choices
# in the three sibling sweep scripts exactly. truth_subdir=None means
# "use the top-level synth_truth/ auto-detect" (100_narrow / baseline).
# ------------------------------------------------------------------
STORMS = [
    {"label": "storm080",   "scale": 0.80,
     "gauge_sdf": "../smf_init_data/met/Master_Precip_storm080.sdf",
     "truth_subdir": "synth_truth/storm080"},
    {"label": "100_narrow", "scale": 1.00,
     "gauge_sdf": "../smf_init_data/met/Master_Precip.sdf",
     "truth_subdir": None},
    {"label": "storm125",   "scale": 1.25,
     "gauge_sdf": "../smf_init_data/met/Master_Precip_storm125.sdf",
     "truth_subdir": "synth_truth/storm125"},
]

# Previously-recorded best-KGE sweep coordinate (shared across all 3
# storm sweeps), for direct comparison printout at the end -- from
# storm_series_summary_080_100n_125.csv.
PRIOR_BEST_KGE_POINT = {"Ks_mult": 7.695388405534351, "f_RS_abs": 0.0199220166252879}

# NOTE: distinct from the original's "100_truthcheck" -- keeps run_ids,
# input/result folders, and metrics_summary filenames from colliding
# with the mean-aggregation version's outputs.
RUN_CATEGORY = "100_truthcheck_interp"


def build_only(storm_cfg):
    """Builds the .in file + current_run_config.json for ONE run at the
    exact true (Ks_mult, f_RS_abs) point, under storm_cfg's forcing/truth.
    Identical to the original script's build_only() except RUN_CATEGORY
    (and therefore run_id / output paths) differs, keeping this variant's
    outputs separate from the mean-aggregation version's."""
    label    = storm_cfg["label"]
    ks_mult  = TRUTH_VALUES["Ks_mult"]
    f_rs_abs = TRUTH_VALUES["f_RS_abs"]
    run_id   = f"{builder.LOCATION}_{builder.EVENT_DATE}_{RUN_CATEGORY}_{label}"

    script_dir   = Path.cwd()
    project_root = (script_dir.parent if script_dir.name == "smf_demo" else script_dir)
    calib_dir    = project_root / "calibration_work"

    run_input_dir      = calib_dir / "01_run_inputs"  / RUN_CATEGORY
    run_results_dir    = calib_dir / "02_results"     / RUN_CATEGORY / run_id
    csv_export_dir     = calib_dir / "03_comparisons" / "csv_exports"
    plot_export_dir    = calib_dir / "03_comparisons" / "hydrograph_plots"
    summary_export_dir = calib_dir / "03_comparisons" / "summary_tables"
    log_dir             = calib_dir / "06_logs"

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
                soil_cls['f'] = f_rs_abs if cid == '1' else soil_params['f']
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
        met.gaugestations['value']    = storm_cfg["gauge_sdf"]

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

        print(f"  [{label}] Ks={ks_mult:.3f}x  f={f_rs_abs:.4f}  "
              f"(cv={PINNED_CV} r={PINNED_R} n={PINNED_N} -- true point, "
              f"all storms, INTERP resample)")

        run_config = {
            "location":                  builder.LOCATION,
            "event_date":                builder.EVENT_DATE,
            "run_number":                RUN_CATEGORY,
            "change_tested":             f"truthcheck_{label}_interp",
            "run_id":                    run_id,
            "run_category":              RUN_CATEGORY,
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
            "swept_param":               f"truthcheck_{RUN_CATEGORY}",
            "swept_value":               ks_mult,
            "gauge_sdf":                 storm_cfg["gauge_sdf"],
        }
        if storm_cfg["truth_subdir"] is not None:
            run_config["truth_file"] = storm_cfg["truth_subdir"]
        # else: no truth_file key -> run_sensitivity_single_interp.py's default
        # top-level synth_truth/ auto-detect (same truth as 100_narrow/baseline)

        config_path = calib_dir / "current_run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))

    finally:
        builder.BASELINE = original_baseline

    return run_id, summary_export_dir


def run_with_timeout(timeout_sec):
    """Identical pattern to the sibling sweep scripts -- own process group,
    killable on timeout, scans for tRIBS's own failure warning. Calls the
    INTERP scoring module instead of the original mean-aggregation one."""
    proc = subprocess.Popen(
        [sys.executable, "run_sensitivity_single_interp.py"],
        cwd=Path.cwd(),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    t0 = time.time()
    try:
        stdout, _ = proc.communicate(timeout=timeout_sec)
        timed_out = False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = proc.communicate()
        timed_out = True
    elapsed = time.time() - t0

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")

    tribs_warning = bool(stdout) and "WARNING: tRIBS may have failed" in stdout
    returncode    = None if timed_out else proc.returncode
    return returncode, elapsed, timed_out, tribs_warning


def check_truth_files(calib_dir):
    """Confirm all three truth files exist before running anything --
    fail fast and clearly rather than partway through the loop."""
    problems = []
    top_level_dir = calib_dir / "synth_truth"
    top_level = list(top_level_dir.glob("*.qout")) if top_level_dir.exists() else []
    if len(top_level) != 1:
        problems.append(f"synth_truth/ (baseline/100_narrow): expected 1 *.qout, "
                         f"found {len(top_level)}")
    for storm_cfg in STORMS:
        if storm_cfg["truth_subdir"] is None:
            continue
        d = calib_dir / storm_cfg["truth_subdir"]
        found = list(d.glob("*.qout")) if d.exists() else []
        if len(found) != 1:
            problems.append(f"{storm_cfg['truth_subdir']}: expected 1 *.qout, "
                             f"found {len(found)}")
    if problems:
        raise RuntimeError("Truth file check failed:\n  " + "\n  ".join(problems))
    print("Truth file check passed: all three storm truths present and unambiguous.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Single-point validation (INTERP variant): run tRIBS at "
                    "the exact true (Ks_mult=7.0x, f_RS_abs=0.012) point "
                    "under all three storm forcings and report actual (not "
                    "interpolated-contour) PBIAS/KGE/r/alpha/beta, scored "
                    "with time-interpolation resampling of the sim series "
                    "instead of mean-aggregation.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-run hard timeout in seconds (default: 300)")
    args = parser.parse_args()

    script_dir = Path.cwd()
    project_root = (script_dir.parent if script_dir.name == "smf_demo" else script_dir)
    calib_dir   = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    check_truth_files(calib_dir)

    print(f"{'='*70}")
    print(f"TRUTH-POINT VALIDATION (INTERP RESAMPLE) -- Ks_mult="
          f"{TRUTH_VALUES['Ks_mult']}, f_RS_abs={TRUTH_VALUES['f_RS_abs']}, "
          f"cv={PINNED_CV}, r={PINNED_R}, n={PINNED_N}")
    print(f"Running this EXACT point (excluded from every LHS sweep) under "
          f"all 3 storm forcings, {args.timeout}s timeout each.")
    print(f"Sim series resampled via time-interpolation, not mean-aggregation "
          f"(see run_sensitivity_single_interp.py docstring).")
    print(f"{'='*70}\n")

    results = []
    for storm_cfg in STORMS:
        label = storm_cfg["label"]
        print(f"\n--- {label} ({storm_cfg['scale']:.0%} rain) ---")
        t0 = time.time()
        try:
            run_id, summary_export_dir = build_only(storm_cfg)
            returncode, elapsed, timed_out, tribs_warning = run_with_timeout(args.timeout)

            if timed_out:
                status = "HANG"
            elif returncode != 0 or tribs_warning:
                status = "FAILED"
            else:
                status = "SUCCESS"

            if status == "SUCCESS":
                metrics_file = summary_export_dir / f"{run_id}_metrics_summary.csv"
                metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
                metrics["storm_label"] = label
                metrics["storm_scale"] = storm_cfg["scale"]
                results.append(metrics)
                print(f"  SUCCESS  ({elapsed/60:.1f} min)  "
                      f"PBIAS={metrics['pbias_pct']:+.2f}%  KGE={metrics['kge']:.4f}  "
                      f"r={metrics['kge_r']:.4f}  alpha={metrics['kge_alpha']:.4f}  "
                      f"beta={metrics['kge_beta']:.4f}")
            else:
                reason = "wall-clock timeout" if timed_out else (
                    "tRIBS reported non-zero exit" if tribs_warning
                    else f"run_sensitivity_single_interp.py exited {returncode}")
                print(f"  {status}: {run_id}  ({reason})")

        except Exception as e:
            print(f"  FAILED (build/run error) for {label}: {e}")

    if not results:
        print("\nNo successful runs -- nothing to compare.")
        return

    df = pd.DataFrame(results)
    out_path = summary_dir / "truth_point_validation_100_storms_interp.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ------------------------------------------------------------------
    # Summary comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY (INTERP RESAMPLE) -- actual metrics AT the true point, "
          "direct (not interpolated-contour)")
    print(f"{'='*70}")
    cols = ["storm_label", "pbias_pct", "kge", "kge_r", "kge_alpha", "kge_beta",
            "nse", "peak_error_pct", "volume_error_pct"]
    print(df[cols].round(4).to_string(index=False))

    print(f"\nInterpretation guide:")
    print(f"  - Compare this table against truth_point_validation_100_storms.csv")
    print(f"    (the original, mean-aggregation version). If PBIAS/peak_error_pct")
    print(f"    move substantially closer to 0% here, the truth-point anomaly (or")
    print(f"    a meaningful share of it) was a resampling-method artifact, not")
    print(f"    real KGE-formula equifinality or a tRIBS-internal behavior.")
    print(f"  - If the numbers are essentially unchanged from the original, the")
    print(f"    resample-method hypothesis is ruled out and the two remaining live")
    print(f"    hypotheses (equifinality vs. an unlogged solver behavior near the")
    print(f"    peak) stand as-is.")

    print(f"\nFor reference, the previously-recorded best-KGE sweep coordinate")
    print(f"(shared across all 3 storms' LHS sweeps) was Ks={PRIOR_BEST_KGE_POINT['Ks_mult']:.3f}, "
          f"f={PRIOR_BEST_KGE_POINT['f_RS_abs']:.4f} -- "
          f"Ks-distance {PRIOR_BEST_KGE_POINT['Ks_mult']-TRUTH_VALUES['Ks_mult']:+.3f} from truth.")


if __name__ == "__main__":
    main()
