"""
run_lhs_synth_Ks_f_100_storm080.py
====================================
Series 100_storm080 -- sibling of Series 100, scored against a SCALED-
RAINFALL truth instead of the baseline forcing. Rain intensity (R column
only, both SMF and SMPHQ gauges) is scaled to 80% of the original event
via scale_precip_forcing.py; Y/M/D/H timing is untouched. TRUTH PARAMETER
VALUES are unchanged (Ks_mult=7.0x, f_RS_abs=0.012, cv/r/n pinned at
4.5/0.24/0.026) -- only the forcing differs. LHS_PARAMS bounds are
NARROWED and n is REDUCED relative to the Series 100 baseline (Ks_mult
3.0-9.5x, f_RS_abs 0.004-0.05, n=200 by default vs. the original
3.0-11.0x/0.003-0.05/n=400) -- see the LHS_PARAMS block below for the
rationale. Because bounds/n now differ from Series 100, this sweep is
NOT sample-for-sample pointwise comparable to the Series 100 baseline;
use plot_ks_f_series_comparison.py's independent-per-series griddata
regridding for that comparison instead (same approach already used for
Series 100 vs 97log). storm080 and storm125 keep IDENTICAL bounds/n to
each other, so they remain pointwise comparable against one another.

Purpose: testing whether the Ks-f equifinality "swoosh" shifts (or changes
curvature) under a different storm magnitude. Two different storms should
trace two different degenerate manifolds in (Ks, f) space; where the
near-zero-PBIAS bands from storm080/storm100/storm125 intersect is a
stronger identifiability constraint than any one alone (per Ivanov et al.
2004's multi-event calibration guidance). This is a NULL-RESULT-TOLERANT
design: no shift/curvature-change is itself informative, not a failure.

Differences from run_lhs_synth_Ks_f_100.py:
  - met.gaugestations points at Master_Precip_storm080.sdf (via GAUGE_SDF
    below) instead of the baseline Master_Precip.sdf.
  - Truth file is resolved via the new truth_file override in
    run_sensitivity_single.py's _find_synth_truth_file(): this script
    writes "truth_file": "synth_truth/storm080" into
    current_run_config.json, which auto-detects the single *.qout file
    living in that subdirectory (baseline truth100 and storm125 truth
    live in their own separate locations, so all three can coexist
    without ambiguity or manual file-swapping between sweeps).
  - LHS_SERIES / LHS_CATEGORY are storm080-specific, so runs, CSVs, and
    result folders never collide with the baseline Series 100 or the
    storm125 sibling.
  - TRUTH_VALUES and the timeout-safe subprocess pattern are otherwise
    IDENTICAL to Series 100. LHS_PARAMS bounds and default n are NOT
    identical to Series 100 (see above) -- they ARE identical to the
    storm125 sibling.

REQUIRES exactly one *.qout file in calibration_work/synth_truth/storm080/
before running. The script checks this itself at startup and will refuse
to run otherwise.

Usage (run from the smf_demo directory):
    python run_lhs_synth_Ks_f_100_storm080.py                # 200 samples, seed=42 (full sweep)
    python run_lhs_synth_Ks_f_100_storm080.py --n 35 --seed 7  # reduced pilot -- NOTE: a
                                                                 # different --n/--seed combo
                                                                 # generates DIFFERENT sample
                                                                 # points than the full n=200/
                                                                 # seed=42 sweep (interval
                                                                 # stratification depends on n),
                                                                 # so a pilot run here is NOT
                                                                 # pointwise-comparable to the
                                                                 # storm125 sibling -- use it
                                                                 # only to smoke-test the
                                                                 # forcing/truth wiring and
                                                                 # check for hang-risk-zone
                                                                 # behavior, not for swoosh
                                                                 # analysis.
    python run_lhs_synth_Ks_f_100_storm080.py --skip_existing  # resume interrupted run
    python run_lhs_synth_Ks_f_100_storm080.py --timeout 600    # more generous per-run timeout

Output:
    calibration_work/03_comparisons/summary_tables/lhs_results_synth_Ks_f_100_storm080.csv
    calibration_work/03_comparisons/summary_tables/lhs_results_synth_Ks_f_FAILED_100_storm080.csv
"""

import argparse
import os
import sys
import signal
import subprocess
import time
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import build_sensitivity_run as builder
from pytRIBS.classes import Project, Soil, Land, Met, Model

# ------------------------------------------------------------------
# CONFIRMED SYNTHETIC TRUTH VALUES (new truth, reset this session).
# cv/r/n are PINNED at these values for every run in this sweep;
# Ks_mult/f_RS_abs are the two swept parameters.
# ------------------------------------------------------------------
TRUTH_VALUES = {
    "Ks_mult":          7.0,
    "f_RS_abs":         0.012,
    "kinemvelcoef":     4.5,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}
TRUTH_TOL = 1e-6

PINNED_CV = TRUTH_VALUES["kinemvelcoef"]
PINNED_R  = TRUTH_VALUES["flowexp"]
PINNED_N  = TRUTH_VALUES["channelroughness"]

# ------------------------------------------------------------------
# LHS PARAMETER RANGES -- NARROWED from the Series 100 baseline
# (3.0-11.0x / 0.003-0.05) to concentrate sampling in the
# identifiability-relevant region for this smaller storm-comparison
# sweep. Truth (Ks=7.0, f=0.012) stays interior on all sides:
# Ks-distance 4.0 below / 2.5 above; log10(f)-distance 0.477 below /
# 0.620 above. Ks>9.5 was confirmed uniformly poor (deep-under-predict,
# no curvature information) across the full f range in the completed
# Series 100 contour, so it's dropped here.
# ------------------------------------------------------------------
LHS_PARAMS = {
    "Ks_mult":  {"lo": 3.0,   "hi": 9.5,  "scale": "linear"},  # true = 7.0
    "f_RS_abs": {"lo": 0.004, "hi": 0.05, "scale": "log"},     # true = 0.012 (log-stratified)
}

# ------------------------------------------------------------------
# STORM-SPECIFIC CONSTANTS -- the only structural differences from the
# baseline Series 100 sweep. Forcing changes; truth parameter values do
# not (see TRUTH_VALUES above, unchanged).
# ------------------------------------------------------------------
STORM_LABEL = "storm080"
STORM_SCALE = 0.80   # informational only (already baked into GAUGE_SDF's forcing files)
GAUGE_SDF   = f"../smf_init_data/met/Master_Precip_{STORM_LABEL}.sdf"
TRUTH_SUBDIR = f"synth_truth/{STORM_LABEL}"   # relative to calib_dir; passed through
                                                # to run_sensitivity_single.py as
                                                # run_config["truth_file"]

LHS_SERIES   = f"100_{STORM_LABEL}"
LHS_CATEGORY = f"100_{STORM_LABEL}_lhs_synth_Ks_f"

# ------------------------------------------------------------------
# Documented tRIBS hang-risk zone (Ks~6.25x, f~0.011) -- used only for
# the end-of-sweep proximity coverage check, not to exclude any samples.
# ------------------------------------------------------------------
HANG_ZONE_KS_LO, HANG_ZONE_KS_HI = 5.75, 6.75
HANG_ZONE_F_LO,  HANG_ZONE_F_HI  = 0.008, 0.014


def is_truth_run(ks_mult, f_rs_abs):
    """True only if the sampled (Ks, f) point happens to equal true values
    exactly. cv/r/n are always pinned at truth in this script, so matching
    Ks/f alone is sufficient to identify (and exclude) the truth point."""
    return (abs(ks_mult - TRUTH_VALUES["Ks_mult"])   < TRUTH_TOL and
            abs(f_rs_abs - TRUTH_VALUES["f_RS_abs"]) < TRUTH_TOL)


# ------------------------------------------------------------------
# LHS SAMPLE GENERATION -- linear stratification by default, log
# stratification when a parameter's bounds dict sets scale="log".
# Identical method to run_lhs_synth_Ks_f_97log.py.
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
    change_tested = f"synthKsf_Ks{ks_lbl}x_f{f_lbl}"
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
# BUILD ONLY -- writes the .in file + current_run_config.json for one
# (Ks_mult, f_RS_abs) point. Mirrors build_and_run_lhs() from
# run_lhs_synth_Ks_f_97log.py exactly, but stops short of calling
# run_sensitivity_single.py in-process -- that happens afterward, as a
# separate, killable subprocess (see run_with_timeout), ported from
# run_lhs_nanchor_cvrn_99.py.
# ------------------------------------------------------------------
def build_only(ks_mult, f_rs_abs):
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
        met.gaugestations['value']    = GAUGE_SDF   # storm080-scaled forcing, not baseline

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
            "swept_param":               f"lhs_synth_Ks_f_{LHS_SERIES}",
            "swept_value":               ks_mult,
            "gauge_sdf":                 GAUGE_SDF,
            "truth_file":                TRUTH_SUBDIR,   # -> run_sensitivity_single.py
                                                            # resolves the single .qout in
                                                            # this subdir; see
                                                            # _find_synth_truth_file()
        }

        config_path = calib_dir / "current_run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))

    finally:
        builder.BASELINE = original_baseline

    return run_id, summary_export_dir


# ------------------------------------------------------------------
# RUN WITH TIMEOUT -- executes run_sensitivity_single.py as a separate
# subprocess in its own process group, so a hang can be killed cleanly
# (python -> shell -> tRIBS binary all die together as one unit) instead
# of blocking this script forever the way an in-process os.system() call
# would. Uses communicate(timeout=...) rather than a manual readline loop
# so a genuine hang can't leave the pipe buffer deadlocking the child.
# Ported unchanged from run_lhs_nanchor_cvrn_99.py.
#
# Also scans the captured output for tRIBS's own "WARNING: tRIBS may have
# failed" message (printed by run_sensitivity_single.py when the tRIBS
# exit code is non-zero) -- a clean returncode=0 alone isn't sufficient
# evidence of a good run, since tRIBS can crash (SIGSEGV) partway through
# and still leave run_sensitivity_single.py exiting 0 with truncated
# output.
# ------------------------------------------------------------------
def run_with_timeout(timeout_sec):
    proc = subprocess.Popen(
        [sys.executable, "run_sensitivity_single.py"],
        cwd=Path.cwd(),
        start_new_session=True,   # own process group -> killable as a unit
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
        stdout, _ = proc.communicate()   # drain pipes after kill
        timed_out = True
    elapsed = time.time() - t0

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")

    tribs_warning = bool(stdout) and "WARNING: tRIBS may have failed" in stdout
    returncode    = None if timed_out else proc.returncode
    return returncode, elapsed, timed_out, tribs_warning


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Series 100_storm080 -- joint Ks_mult x f_RS_abs LHS "
                    "sweep against the synthetic truth (Ks_mult=7.0x, "
                    "f_RS_abs=0.012) under 80%-scaled rainfall forcing, "
                    "cv/r/n pinned at truth values, f_RS_abs log-"
                    "stratified, timeout-safe execution. Sibling of "
                    "Series 100 for the multi-storm-magnitude swoosh test.")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of LHS samples (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose compare CSV already exists")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-run hard timeout in seconds (default: 300 = 5 min, "
                             "well above the <1 min baseline, well below the "
                             "13-94+ min hangs seen near the Ks6p25lo point)")
    args = parser.parse_args()

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir   = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # SAFETY CHECK: confirm the storm080 truth subdirectory has exactly
    # one .qout file (this is what TRUTH_SUBDIR points run_sensitivity_
    # single.py at via current_run_config.json -- the baseline synth_
    # truth/ top-level auto-detect is NOT used by this sibling script).
    # ------------------------------------------------------------------
    truth_dir  = calib_dir / TRUTH_SUBDIR
    qout_files = list(truth_dir.glob("*.qout")) if truth_dir.exists() else []
    if len(qout_files) != 1:
        raise RuntimeError(
            f"Expected exactly one *.qout file in {truth_dir} to activate "
            f"storm080 synthetic-truth mode, found {len(qout_files)}: "
            f"{[f.name for f in qout_files]}. This sweep scores against "
            f"the storm080-scaled-forcing truth specifically -- confirm "
            f"the storm080 truth .qout lives in this subdirectory (see "
            f"scale_precip_forcing.py / truth generation notes) before "
            f"running, or results won't be scored against the truth you "
            f"expect."
        )
    print(f"Storm080 synthetic truth mode confirmed active: {qout_files[0].name}")
    print(f"  Forcing: {GAUGE_SDF}")
    print(f"  Truth params (unchanged from baseline): Ks_mult={TRUTH_VALUES['Ks_mult']}, "
          f"f_RS_abs={TRUTH_VALUES['f_RS_abs']}, cv={PINNED_CV}, r={PINNED_R}, n={PINNED_N}")

    samples = generate_lhs_samples(args.n, LHS_PARAMS, seed=args.seed)

    out_path        = summary_dir / f"lhs_results_synth_Ks_f_{LHS_SERIES}.csv"
    failed_log_path = summary_dir / f"lhs_results_synth_Ks_f_FAILED_{LHS_SERIES}.csv"

    existing_df      = load_existing_results(out_path)
    existing_run_ids = (set(existing_df["run_id"].values)
                        if not existing_df.empty else set())

    results          = []
    failed_log_rows  = []
    if not existing_df.empty:
        results.extend(existing_df.to_dict("records"))

    print(f"\n{'='*70}")
    print(f"LHS sweep -- Series {LHS_SERIES} -- Ks_mult x f_RS_abs vs SYNTHETIC TRUTH "
          f"(storm080 forcing, {STORM_SCALE:.0%} rainfall)")
    print(f"  ({args.n} samples, seed={args.seed}, timeout={args.timeout}s)")
    print(f"  Ks_mult:  {LHS_PARAMS['Ks_mult']['lo']:.1f} - "
          f"{LHS_PARAMS['Ks_mult']['hi']:.1f}x        [true = {TRUTH_VALUES['Ks_mult']}]  (linear)")
    print(f"  f_RS_abs: {LHS_PARAMS['f_RS_abs']['lo']:.4f} - "
          f"{LHS_PARAMS['f_RS_abs']['hi']:.4f}   [true = {TRUTH_VALUES['f_RS_abs']}]  (LOG-stratified)")
    print(f"  PINNED:   cv={PINNED_CV}  r={PINNED_R}  n={PINNED_N}  (truth values)")
    print(f"  Hang-risk zone (informational only): Ks {HANG_ZONE_KS_LO}-{HANG_ZONE_KS_HI}x, "
          f"f {HANG_ZONE_F_LO}-{HANG_ZONE_F_HI}")
    print(f"{'='*70}\n")

    completed, skipped, excluded, hung, failed = 0, 0, 0, 0, 0
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
            run_id, summary_export_dir = build_only(ks_mult, f_rs_abs)
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
                metrics["Ks_mult"]          = ks_mult
                metrics["f_RS_abs"]         = f_rs_abs
                metrics["kinemvelcoef"]     = PINNED_CV
                metrics["flowexp"]          = PINNED_R
                metrics["channelroughness"] = PINNED_N
                metrics["swept_param"]      = f"lhs_synth_Ks_f_{LHS_SERIES}"
                results = [r for r in results if r.get("run_id") != run_id]
                results.append(metrics)
                completed += 1
            else:
                reason = "wall-clock timeout" if timed_out else (
                    "tRIBS reported non-zero exit" if tribs_warning
                    else f"run_sensitivity_single.py exited {returncode}")
                print(f"  {status}: {run_id}  ({reason})")
                if status == "HANG":
                    hung += 1
                else:
                    failed += 1
                failed_log_rows.append({
                    "run_id": run_id, "status": status, "reason": reason,
                    "elapsed_min": elapsed / 60,
                    "Ks_mult": ks_mult, "f_RS_abs": f_rs_abs,
                    "kinemvelcoef": PINNED_CV, "flowexp": PINNED_R,
                    "channelroughness": PINNED_N,
                })
                pd.DataFrame(failed_log_rows).to_csv(failed_log_path, index=False)

        except Exception as e:
            print(f"  FAILED (build error): {run_id}")
            print(f"  Error:  {e}")
            failed += 1
            elapsed = time.time() - t0
            failed_log_rows.append({
                "run_id": run_id, "status": "BUILD_ERROR", "reason": str(e),
                "elapsed_min": elapsed / 60,
                "Ks_mult": ks_mult, "f_RS_abs": f_rs_abs,
                "kinemvelcoef": PINNED_CV, "flowexp": PINNED_R,
                "channelroughness": PINNED_N,
            })
            pd.DataFrame(failed_log_rows).to_csv(failed_log_path, index=False)

        elapsed       = time.time() - t0
        total_elapsed = time.time() - sweep_start
        remaining     = args.n - completed - skipped - excluded - hung - failed
        if completed > 0:
            avg_time = total_elapsed / completed
            eta_min  = (avg_time * remaining) / 60
            print(f"  Run time: {elapsed/60:.1f} min  |  ETA: {eta_min:.0f} min remaining")

        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"\nSweep complete: {completed} ran, {skipped} skipped, "
          f"{excluded} excluded, {hung} hung, {failed} failed")

    if results:
        final_df = pd.DataFrame(results).sort_values("kge", ascending=False)
        final_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path.name}  ({len(final_df)} rows)")

        # ------------------------------------------------------------------
        # Coverage check 1: sampling extremes near the LHS bounds -- flags
        # thin coverage at either end of the Ks_mult range.
        # ------------------------------------------------------------------
        low_ks_n  = final_df[final_df["Ks_mult"] <= 4.0].shape[0]
        high_ks_n = final_df[final_df["Ks_mult"] >= 9.0].shape[0]
        print(f"\nCoverage check: {low_ks_n} points with Ks_mult <= 4.0, "
              f"{high_ks_n} points with Ks_mult >= 9.0 (of {len(final_df)} total).")

        # ------------------------------------------------------------------
        # Coverage check 2: low-f region (<0.007) -- the band log-
        # stratification was specifically meant to densify.
        # ------------------------------------------------------------------
        low_f_n = final_df[final_df["f_RS_abs"] < 0.007].shape[0]
        print(f"Coverage check: {low_f_n} of {len(final_df)} points have "
              f"f_RS_abs < 0.007 (near-anchor-A/B region).")

        # ------------------------------------------------------------------
        # Coverage check 3: proximity to the documented hang-risk zone
        # (Ks~6.25x, f~0.011) -- now much closer to truth than in Series
        # 97/97log. Reports both completed points and HANG/FAILED draws
        # that landed in/near this box, since this is the key new risk
        # flagged for this rebuild.
        # ------------------------------------------------------------------
        zone_completed = final_df[
            (final_df["Ks_mult"]  >= HANG_ZONE_KS_LO) & (final_df["Ks_mult"]  <= HANG_ZONE_KS_HI) &
            (final_df["f_RS_abs"] >= HANG_ZONE_F_LO)  & (final_df["f_RS_abs"] <= HANG_ZONE_F_HI)
        ].shape[0]
        print(f"Coverage check: {zone_completed} completed points fall inside the "
              f"hang-risk zone (Ks {HANG_ZONE_KS_LO}-{HANG_ZONE_KS_HI}x, "
              f"f {HANG_ZONE_F_LO}-{HANG_ZONE_F_HI}).")
        if failed_log_rows:
            failed_df = pd.DataFrame(failed_log_rows)
            zone_failed = failed_df[
                (failed_df["Ks_mult"]  >= HANG_ZONE_KS_LO) & (failed_df["Ks_mult"]  <= HANG_ZONE_KS_HI) &
                (failed_df["f_RS_abs"] >= HANG_ZONE_F_LO)  & (failed_df["f_RS_abs"] <= HANG_ZONE_F_HI)
            ].shape[0]
            print(f"  {len(failed_df)} total HANG/FAILED draw(s) this sweep "
                  f"({zone_failed} inside the hang-risk zone) -- see "
                  f"{failed_log_path.name} for the full audit trail. If the "
                  f"zone accumulates several, it may warrant the same dedicated "
                  f"probe treatment Ks6p25lo got in Series 98/99.")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
