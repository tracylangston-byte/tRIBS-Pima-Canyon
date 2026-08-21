"""
run_lhs_nanchor_cvrn_101.py
============================
Series 101 -- cv/r/n routing-parameter identifiability rebuild, superseding
Series 96/99. See Handoff_Series101_CvRnIdentifiabilityRebuild_v1.md for the
full design rationale; this docstring covers only what changed from Series 99
(run_lhs_nanchor_cvrn_99.py), which this script is ported from.

What changed from Series 99
----------------------------
1. Anchors: 7 anchors taken directly from the KGE_2012 ridge in
   ridge_width_vs_f.csv (Series 100 family, current truth, corrected
   resampling) instead of hand-hunted PBIAS=0 points. This tests only the
   single KGE_2012-optimal branch at each f -- gamma already treats the old
   high-f PBIAS=0 branch as spurious, so it isn't re-tested here.
2. Truth: Ks_mult=7.0x / f_RS_abs=0.012 (reset 7/15), reusing the existing
   calibration_work/synth_truth/SMF_20140812_60_Ks7p0x_truth100_Outlet.qout
   file directly -- cv/r/n truth values (4.5/0.24/0.026) are unchanged by
   the reset, so no new truth run is needed.
3. Sample size: all 7 anchors get n=50 (paired, shared seed, same as
   Series 99's design). The f0p012_true anchor additionally gets +100 more
   draws (a second, independently-seeded batch) to reach n=150 total --
   this is the anchor that'll be cited as the headline cv-null result, and
   the tighter CI at n=150 (~+/-0.20 vs ~+/-0.28 at n=50) matters for that.
4. Per-run disk cleanup (NEW): after a run's metrics are successfully
   extracted, the bulky raw tRIBS mesh/Voronoi output directory for that run
   is deleted immediately. Only the metrics summary row, phase-metrics row,
   and the small *_compare_obs_sim.csv are kept. This directly targets the
   failure mode that cost Series 96/97/97log/99 their raw data (permanent
   July disk cleanup) -- the goal is to never again need a re-run just to
   re-verify a number. Disable with --keep_raw for debugging.
5. Resampling / KGE formula: no changes needed here -- both fixes already
   live inside run_sensitivity_single.py (5-min time-interpolation resample,
   KGE_2009 alpha + KGE_2012 gamma computed automatically for every run).
   This script is a thin orchestration layer around that; it doesn't
   duplicate metric logic.

Everything else -- the timeout-safe subprocess pattern (separate process
group, hard wall-clock kill, "WARNING: tRIBS may have failed" scan,
HANG-vs-FAILED distinction), the shared-seed paired-anchor sampling design,
the build_only()/run_with_timeout() split, and the failed-run audit trail --
is carried over unchanged from Series 99, which is why f0p010 and
f0p012_true (both inside or near the documented hang zone,
Ks~5.75-6.75x x f 0.008-0.014) are expected to log some HANG rows rather
than stall the batch. No dedicated pre-sweep hang-zone probe this round --
Josh's source-level fix isn't available yet, and a probe only saves time,
not correctness, given the timeout-safe design is already in place.

Usage (run from the smf_demo directory):
    python run_lhs_nanchor_cvrn_101.py                   # default: n=50/anchor, n_true=150, seed=42
    python run_lhs_nanchor_cvrn_101.py --n 100            # more base samples per anchor
    python run_lhs_nanchor_cvrn_101.py --n_true 200       # more samples at the true anchor
    python run_lhs_nanchor_cvrn_101.py --seed 99          # different shared base seed
    python run_lhs_nanchor_cvrn_101.py --skip_existing    # resume an interrupted run
    python run_lhs_nanchor_cvrn_101.py --timeout 600      # more generous per-run timeout
    python run_lhs_nanchor_cvrn_101.py --keep_raw         # disable per-run disk cleanup (debugging)

Requires calibration_work/synth_truth/ to contain exactly one *.qout file --
the Ks=7.0x/f=0.012 truth (SMF_20140812_60_Ks7p0x_truth100_Outlet.qout).
run_sensitivity_single.py auto-detects synthetic mode from this file; no
truth_file override is needed since Series 101 doesn't touch the storm080/
storm125 siblings.

Output: one CSV per anchor, plus a combined CSV with all anchors stacked:
    calibration_work/03_comparisons/summary_tables/lhs_results_anchor_<label>_101.csv
    calibration_work/03_comparisons/summary_tables/lhs_results_anchor_ALL_101.csv
    calibration_work/03_comparisons/summary_tables/lhs_results_anchor_FAILED_101.csv
        (audit trail of every HANG/FAILED draw across all anchors)
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
# ANCHORS -- taken directly from the KGE_2012 ridge in
# ridge_width_vs_f.csv (Series 100 family, current truth). Ks_mult values
# are the ridge_peak_Ks column, rounded to 4 decimals (well within tRIBS's
# meaningful precision; not worth an ugly 15-digit run_id label).
#
# f0p012_true is the anchor nearest the actual synthetic truth
# (Ks_mult=7.0, f_RS_abs=0.012) and is topped up to n=150 (see --n_true).
# f0p010 and f0p012_true both sit inside or near the documented hang zone
# (Ks~5.75-6.75x x f 0.008-0.014) -- expect some HANG rows there.
# ------------------------------------------------------------------
ANCHORS = [
    {"label": "f0p006",      "f_RS_abs": 0.006, "Ks_mult": 4.3943},
    {"label": "f0p008",      "f_RS_abs": 0.008, "Ks_mult": 5.2336},
    {"label": "f0p010",      "f_RS_abs": 0.010, "Ks_mult": 6.1928},  # hang-zone adjacent
    {"label": "f0p012_true", "f_RS_abs": 0.012, "Ks_mult": 6.9322},  # true point; n=150
    {"label": "f0p015",      "f_RS_abs": 0.015, "Ks_mult": 8.0513},
    {"label": "f0p02",       "f_RS_abs": 0.020, "Ks_mult": 8.1912},
    {"label": "f0p03",       "f_RS_abs": 0.030, "Ks_mult": 7.1720},
]

TRUE_ANCHOR_LABEL = "f0p012_true"

# ------------------------------------------------------------------
# LHS PARAMETER RANGES -- unchanged from Series 96/99. Nothing in the
# Series 101 diagnosis implicates the ranges themselves.
# ------------------------------------------------------------------
LHS_PARAMS = {
    "kinemvelcoef":     {"lo": 2.5,  "hi": 6.5},   # true = 4.5
    "flowexp":          {"lo": 0.18, "hi": 0.35},  # true = 0.24
    "channelroughness": {"lo": 0.02, "hi": 0.10},  # true = 0.026
}

# Truth values -- current truth (reset 7/15). cv/r/n unchanged from Series 96/99.
TRUTH_VALUES = {
    "Ks_mult":          7.0,
    "f_RS_abs":         0.012,
    "kinemvelcoef":     4.5,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}
TRUTH_TOL = 1e-6

LHS_SERIES   = "101"
LHS_CATEGORY = "101_lhs_nanchor_cvrn"


def is_truth_run(ks_mult, f_rs_abs, cv, r, n):
    """True only if ALL five values match truth exactly."""
    return (abs(ks_mult - TRUTH_VALUES["Ks_mult"])     < TRUTH_TOL and
            abs(f_rs_abs - TRUTH_VALUES["f_RS_abs"])   < TRUTH_TOL and
            abs(cv - TRUTH_VALUES["kinemvelcoef"])     < TRUTH_TOL and
            abs(r  - TRUTH_VALUES["flowexp"])          < TRUTH_TOL and
            abs(n  - TRUTH_VALUES["channelroughness"]) < TRUTH_TOL)


# ------------------------------------------------------------------
# LHS SAMPLE GENERATION (identical method to Series 91-99)
# ------------------------------------------------------------------
def generate_lhs_samples(n, params, seed=None):
    if n <= 0:
        return pd.DataFrame({p: [] for p in params})
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
# RUN ID CONSTRUCTION -- anchor label included so identical cv/r/n triples
# reused across anchors don't collide in the filesystem.
# ------------------------------------------------------------------
def build_lhs_run_id(anchor_label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness):
    ks_lbl = builder.value_to_label(ks_mult)
    f_lbl  = builder.value_to_label(f_rs_abs)
    cv_lbl = builder.value_to_label(kinemvelcoef)
    r_lbl  = builder.value_to_label(flowexp)
    n_lbl  = builder.value_to_label(channelroughness)
    change_tested = (f"{anchor_label}_Ks{ks_lbl}x_f{f_lbl}_"
                      f"cv{cv_lbl}_r{r_lbl}_n{n_lbl}")
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
# (anchor, cv, r, n) point. Ported unchanged from run_lhs_nanchor_cvrn_99.py
# except: LHS_SERIES/LHS_CATEGORY now resolve to "101", and this now also
# returns run_results_dir so the caller can clean it up post-extraction.
# ------------------------------------------------------------------
def build_only(anchor_label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness):
    run_id, change_tested = build_lhs_run_id(
        anchor_label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness)

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir    = project_root / "calibration_work"

    run_input_dir      = calib_dir / "01_run_inputs"  / LHS_CATEGORY
    run_results_dir    = calib_dir / "02_results"     / LHS_CATEGORY / run_id
    csv_export_dir      = calib_dir / "03_comparisons" / "csv_exports"
    plot_export_dir     = calib_dir / "03_comparisons" / "hydrograph_plots"
    summary_export_dir  = calib_dir / "03_comparisons" / "summary_tables"
    log_dir              = calib_dir / "06_logs"

    for folder in [run_input_dir, run_results_dir, csv_export_dir,
                   plot_export_dir, summary_export_dir, log_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    original_baseline = builder.BASELINE.copy()
    builder.BASELINE["Ks_mult"]          = ks_mult
    builder.BASELINE["f_RS_abs"]         = f_rs_abs
    builder.BASELINE["kinemvelcoef"]     = kinemvelcoef
    builder.BASELINE["flowexp"]          = flowexp
    builder.BASELINE["channelroughness"] = channelroughness

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
                # f: RS soil uses this anchor's f_rs_abs; all others use baseline
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
        model.optsnow['value']       = 0      # MUST stay 0 -- SMF.in has this wrong
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

        print(f"  [{anchor_label}]  Ks={ks_mult:.4f}x  f={f_rs_abs:.4f}  "
              f"cv={kinemvelcoef:.3f}  r={flowexp:.3f}  n={channelroughness:.4f}")

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
            "swept_param":               f"lhs_anchor_{anchor_label}",
            "swept_value":               ks_mult,
        }

        config_path = calib_dir / "current_run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))

    finally:
        builder.BASELINE = original_baseline

    return run_id, summary_export_dir, run_results_dir


# ------------------------------------------------------------------
# RUN WITH TIMEOUT -- unchanged from Series 99. Executes
# run_sensitivity_single.py as a separate subprocess in its own process
# group so a hang can be killed cleanly, with a hard wall-clock timeout and
# a scan for tRIBS's own "WARNING: tRIBS may have failed" message (a clean
# returncode=0 isn't sufficient evidence of a good run -- see Ks6p25lo probe
# in Series 99).
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
# DISK CLEANUP -- NEW in Series 101. Deletes the bulky raw tRIBS mesh/
# Voronoi output directory for one run, once its metrics summary and
# compare_obs_sim CSV have been safely written elsewhere. Failures here are
# non-fatal (logged, not raised) since losing raw output after a successful
# extraction is not a correctness problem -- the metrics row is what's
# cited going forward.
# ------------------------------------------------------------------
def cleanup_run_output(run_results_dir):
    try:
        if run_results_dir.exists():
            shutil.rmtree(run_results_dir)
    except Exception as e:
        print(f"  WARNING: could not clean up {run_results_dir}: {e}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Series 101 -- cv/r/n LHS sweep across 7 KGE_2012-ridge anchors.")
    parser.add_argument("--n", type=int, default=50,
                        help="Base number of LHS samples per anchor, shared/paired "
                             "across all anchors via the same seed (default: 50)")
    parser.add_argument("--n_true", type=int, default=150,
                        help="Total samples at the f0p012_true anchor, i.e. the base "
                             "--n plus an additional independently-seeded top-up batch "
                             "(default: 150 = 50 base + 100 top-up)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the base (shared/paired) sample set (default: 42)")
    parser.add_argument("--true_topup_seed", type=int, default=None,
                        help="Seed for the f0p012_true top-up batch. Defaults to --seed + 1000 "
                             "to keep it independent of the base draws.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose compare CSV already exists")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-run hard timeout in seconds (default: 300 = 5 min)")
    parser.add_argument("--keep_raw", action="store_true",
                        help="Disable per-run disk cleanup of raw tRIBS mesh/Voronoi "
                             "output (kept on by default -- see module docstring point 4)")
    parser.add_argument("--anchors", type=str, default=None,
                        help="Comma-separated subset of anchor labels to run this call "
                             "(default: all 7). Anchors not listed are left untouched -- "
                             "their existing CSVs, if any, are neither read nor rewritten. "
                             "Valid labels: " + ", ".join(a["label"] for a in ANCHORS) +
                             ". E.g. --anchors f0p006,f0p008,f0p010,f0p015,f0p02,f0p03 to "
                             "proceed with everything except a hanging f0p012_true.")
    args = parser.parse_args()

    if args.n_true < args.n:
        raise ValueError(f"--n_true ({args.n_true}) must be >= --n ({args.n}).")

    if args.anchors:
        requested   = [lbl.strip() for lbl in args.anchors.split(",") if lbl.strip()]
        valid_labels = {a["label"] for a in ANCHORS}
        invalid      = [lbl for lbl in requested if lbl not in valid_labels]
        if invalid:
            raise ValueError(
                f"Unknown anchor label(s): {invalid}. Valid labels: {sorted(valid_labels)}"
            )
        anchors_to_run = [a for a in ANCHORS if a["label"] in requested]
    else:
        anchors_to_run = ANCHORS

    true_topup_seed = args.true_topup_seed if args.true_topup_seed is not None else args.seed + 1000
    n_topup = args.n_true - args.n

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir   = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Base sample set: shared/paired across every anchor.
    samples_base = generate_lhs_samples(args.n, LHS_PARAMS, seed=args.seed)
    # Top-up batch: independently seeded, used only for f0p012_true.
    samples_topup = generate_lhs_samples(n_topup, LHS_PARAMS, seed=true_topup_seed)

    print(f"\n{'='*70}")
    print(f"LHS sweep -- Series {LHS_SERIES} -- cv/r/n across "
          f"{len(anchors_to_run)}/{len(ANCHORS)} ridge anchors this call")
    print(f"  Base: {args.n} samples/anchor, seed={args.seed} (shared/paired)")
    print(f"  {TRUE_ANCHOR_LABEL}: +{n_topup} top-up samples, seed={true_topup_seed} "
          f"-> {args.n_true} total")
    print(f"  kinemvelcoef:     {LHS_PARAMS['kinemvelcoef']['lo']:.2f} - "
          f"{LHS_PARAMS['kinemvelcoef']['hi']:.2f}          [true = {TRUTH_VALUES['kinemvelcoef']}]")
    print(f"  flowexp:          {LHS_PARAMS['flowexp']['lo']:.2f} - "
          f"{LHS_PARAMS['flowexp']['hi']:.2f}          [true = {TRUTH_VALUES['flowexp']}]")
    print(f"  channelroughness: {LHS_PARAMS['channelroughness']['lo']:.3f} - "
          f"{LHS_PARAMS['channelroughness']['hi']:.3f}         [true = {TRUTH_VALUES['channelroughness']}]")
    for a in anchors_to_run:
        print(f"  Anchor '{a['label']}':  Ks_mult={a['Ks_mult']}  f_RS_abs={a['f_RS_abs']}")
    skipped_anchors = [a["label"] for a in ANCHORS if a not in anchors_to_run]
    if skipped_anchors:
        print(f"  NOT run this call (use --anchors to include): {', '.join(skipped_anchors)}")
    print(f"  Raw output cleanup: {'DISABLED (--keep_raw)' if args.keep_raw else 'enabled'}")
    print(f"{'='*70}\n")

    all_anchor_results = []
    failed_log_rows    = []
    failed_log_path    = summary_dir / f"lhs_results_anchor_FAILED_{LHS_SERIES}.csv"

    for anchor in anchors_to_run:
        label    = anchor["label"]
        ks_mult  = anchor["Ks_mult"]
        f_rs_abs = anchor["f_RS_abs"]

        samples = (pd.concat([samples_base, samples_topup], ignore_index=True)
                   if label == TRUE_ANCHOR_LABEL else samples_base)
        n_this_anchor = len(samples)

        out_path = summary_dir / f"lhs_results_anchor_{label}_{LHS_SERIES}.csv"
        existing_df      = load_existing_results(out_path)
        existing_run_ids = (set(existing_df["run_id"].values)
                            if not existing_df.empty else set())

        anchor_results = []
        if not existing_df.empty:
            anchor_results.extend(existing_df.to_dict("records"))

        completed, skipped, excluded, hung, failed = 0, 0, 0, 0, 0
        sweep_start = time.time()

        print(f"\n--- Anchor '{label}'  (Ks_mult={ks_mult}, f_RS_abs={f_rs_abs}, n={n_this_anchor}) ---")

        for i, row in samples.iterrows():
            kinemvelcoef     = row["kinemvelcoef"]
            flowexp          = row["flowexp"]
            channelroughness = row["channelroughness"]

            if is_truth_run(ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness):
                print(f"[{label} {i+1:>3}/{n_this_anchor}]  EXCLUDED: matches truth exactly.")
                excluded += 1
                continue

            run_id, _ = build_lhs_run_id(
                label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness)

            print(f"\n[{label} {i+1:>3}/{n_this_anchor}]  cv={kinemvelcoef:.3f}  "
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
                        m["anchor_label"]     = label
                        m["Ks_mult"]          = ks_mult
                        m["f_RS_abs"]         = f_rs_abs
                        m["kinemvelcoef"]     = kinemvelcoef
                        m["flowexp"]          = flowexp
                        m["channelroughness"] = channelroughness
                        anchor_results.append(m)
                    except Exception:
                        pass
                continue

            t0 = time.time()
            try:
                run_id, summary_export_dir, run_results_dir = build_only(
                    label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness)
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
                    metrics["anchor_label"]     = label
                    metrics["Ks_mult"]          = ks_mult
                    metrics["f_RS_abs"]         = f_rs_abs
                    metrics["kinemvelcoef"]     = kinemvelcoef
                    metrics["flowexp"]          = flowexp
                    metrics["channelroughness"] = channelroughness
                    metrics["swept_param"]      = f"lhs_anchor_{label}"
                    anchor_results = [r for r in anchor_results if r.get("run_id") != run_id]
                    anchor_results.append(metrics)
                    completed += 1

                    # NEW in Series 101: discard bulky raw output now that the
                    # metrics row + compare CSV are safely written elsewhere.
                    if not args.keep_raw:
                        cleanup_run_output(run_results_dir)
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
                        "anchor_label": label, "run_id": run_id, "status": status,
                        "reason": reason, "elapsed_min": elapsed / 60,
                        "Ks_mult": ks_mult, "f_RS_abs": f_rs_abs,
                        "kinemvelcoef": kinemvelcoef, "flowexp": flowexp,
                        "channelroughness": channelroughness,
                    })
                    pd.DataFrame(failed_log_rows).to_csv(failed_log_path, index=False)
                    # Raw output for HANG/FAILED runs is left in place
                    # (small in number, useful for debugging) even with
                    # cleanup enabled.

            except Exception as e:
                print(f"  FAILED (build error): {run_id}")
                print(f"  Error:  {e}")
                failed += 1
                elapsed = time.time() - t0
                failed_log_rows.append({
                    "anchor_label": label, "run_id": run_id, "status": "BUILD_ERROR",
                    "reason": str(e), "elapsed_min": elapsed / 60,
                    "Ks_mult": ks_mult, "f_RS_abs": f_rs_abs,
                    "kinemvelcoef": kinemvelcoef, "flowexp": flowexp,
                    "channelroughness": channelroughness,
                })
                pd.DataFrame(failed_log_rows).to_csv(failed_log_path, index=False)

            elapsed       = time.time() - t0
            total_elapsed = time.time() - sweep_start
            remaining     = n_this_anchor - completed - skipped - excluded - hung - failed
            if completed > 0:
                avg_time = total_elapsed / completed
                eta_min  = (avg_time * remaining) / 60
                print(f"  Run time: {elapsed/60:.1f} min  |  anchor ETA: {eta_min:.0f} min remaining")

            if anchor_results:
                pd.DataFrame(anchor_results).to_csv(out_path, index=False)

        print(f"\nAnchor '{label}' complete: {completed} ran, {skipped} skipped, "
              f"{excluded} excluded, {hung} hung, {failed} failed")

        if anchor_results:
            final_df = pd.DataFrame(anchor_results).sort_values("kge_2012", ascending=False)
            final_df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}  ({len(final_df)} rows)")
            all_anchor_results.extend(final_df.to_dict("records"))

    # ------------------------------------------------------------------
    # Combined output across all anchors.
    # ------------------------------------------------------------------
    if all_anchor_results:
        combined_path = summary_dir / f"lhs_results_anchor_ALL_{LHS_SERIES}.csv"
        combined_df = pd.DataFrame(all_anchor_results)
        combined_df.to_csv(combined_path, index=False)
        print(f"\n{'='*70}")
        print(f"All anchors complete. Combined file: {combined_path.name}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"\nKGE_2012 by anchor:")
        print(combined_df.groupby("anchor_label")["kge_2012"].describe()[["min", "50%", "max"]])
        print(f"\nParameter-KGE_2012 correlations by anchor (Pearson r):")
        for label in combined_df["anchor_label"].unique():
            sub = combined_df[combined_df["anchor_label"] == label]
            corrs = {p: sub[p].corr(sub["kge_2012"])
                     for p in ["kinemvelcoef", "flowexp", "channelroughness"]}
            print(f"  {label}: " + "  ".join(f"{p}={r:+.3f}" for p, r in corrs.items()))
        if failed_log_rows:
            print(f"\n{len(failed_log_rows)} HANG/FAILED draw(s) across all anchors -- "
                  f"see {failed_log_path.name} for the full audit trail.")
        print(f"{'='*70}\n")
        print("Next: run analyze_series101_cvrn_identifiability.py on "
              f"{combined_path.name} for the full Pearson + Spearman + PCA analysis.")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()