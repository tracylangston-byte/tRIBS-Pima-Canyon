"""
run_lhs_multiparam.py
======================
Runs a Latin Hypercube Sampling sweep across N routing/soil parameters
against the REAL GAUGE record, for series 81, 82, or 83.

Switch series by editing ACTIVE_SERIES below (one line). Everything else --
which parameters are swept, their ranges, run-ID formatting, output paths --
is driven by the SERIES_PRESETS dict.

I RECOMMEND TO MAKE SURE YOU KNOW WHAT AND WHERE THIS IS SAVING SO YOU
DON'T OVERWRITE THE PREVIOUS RUN'S SUMMARY TABLES.

Usage (run from the smf_demo directory):
    python run_lhs_multiparam.py                    # default n for the active series, seed=42
    python run_lhs_multiparam.py --n 100            # more samples
    python run_lhs_multiparam.py --seed 99          # different random seed
    python run_lhs_multiparam.py --skip_existing    # resume interrupted run

======================================================================
UPDATED / CONSOLIDATED -- see notes below
======================================================================
This script merges three near-duplicate files:
    - run_lhs_5param.py    (series 81 -- Ks 7.5-12x, cv, r, n; 4 free params)
    - run_lhs_6param.py    (series 82 -- adds psiB_mult; 5 free params)
    - run_lhs_11param.py   (series 83 -- adds channelwidthcoeff, thetaS_mult;
                             narrows Ks to 7.5-9.5x; 7 free params)
Companion to plot_lhs_multiparam.py, which already consolidated the
matching plot scripts for these same three series.

Bug found and fixed during merge: run_lhs_5param.py (81) imported
run_sensitivity_single_old from a dead path
(workspaces.SMF_Calibration_pytRIBS.smf_demo.run_sensitivity_single_old) --
that module doesn't exist anywhere in the project, so the script would
have crashed if run as-is. This consolidated version uses the current
run_sensitivity_single for all three series, same as 82 and 83 already did.

Series 80 (Ks 4-8x, the predecessor to 81) is NOT reproducible from current
files -- run_lhs_5param.py's parameter ranges were edited in place from 80
to 81 rather than kept as a separate script, so 80's exact cv/r/n ranges no
longer exist anywhere. Its CSV (lhs_results_5param_KsLo.csv), if present,
remains as historical data only; there is no ACTIVE_SERIES="80" option here.

Behavioral note (observed, not changed): none of the three original scripts
set model.opintrvl or model.spopintrvl -- unlike build_sensitivity_run.py
(0.0833 hr / 1 hr) and run_lhs_synth_Ks_f_97.py, these real-gauge multiparam
sweeps have always run at whatever pytRIBS's opintrvl default is. Preserved
as-is here; flagging in case sub-hourly phase metrics ever matter for series
81-83 the way they did for the synthetic series 91/92 (see
run_lhs_synth_4param_series.py's docstring).

thetaS_mult and psiB_mult are unified onto a shared code path with a default
of 1.0 when a series doesn't sweep them (81 sweeps neither; 82 sweeps
psiB_mult only). At mult=1.0 this reproduces the original un-multiplied
per-class baseline exactly, so 81's and 82's build behavior is unchanged.
Likewise As_value/Au_value are always read from builder.BASELINE (default
1.0) rather than a per-preset fixed constant -- 83's old AS_FIXED/AU_FIXED
were also 1.0, so this is the same value by a shorter path, not a behavior
change.

The old files (run_lhs_5param.py, run_lhs_6param.py, run_lhs_11param.py)
should be deleted/archived once this replacement is verified against a
saved results CSV from each series.
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
# SERIES PRESETS -- one entry per historical script this file replaces
# ======================================================================
SERIES_PRESETS = {
    "81": dict(
        series_label="81",
        category="81_lhs_5param_KsHigh",
        out_csv="lhs_results_5param_81.csv",
        swept_tag="lhs_5param",
        default_n=50,
        label_mode="full",          # builder.value_to_label -- full precision
        include_As_Au_in_metrics=False,
        lhs_params={
            "Ks_mult":          (7.5,   12.0),   # extended from series 80 (4-8x)
            "kinemvelcoef":     (2.0,   6.0),
            "flowexp":          (0.25,  0.35),
            "channelroughness": (0.008, 0.020),
        },
        id_spec=[
            ("Ks_mult",          "Ks",   "x"),
            ("kinemvelcoef",     "cv",   ""),
            ("flowexp",          "r",    ""),
            ("channelroughness", "n",    ""),
        ],
        description="5-parameter LHS sweep series 81: Ks 7.5-12x (high-Ks refinement).",
    ),
    "82": dict(
        series_label="82",
        category="82_lhs_6param_psiB",
        out_csv="lhs_results_6param_82.csv",
        swept_tag="lhs_6param",
        default_n=50,
        label_mode="full",
        include_As_Au_in_metrics=False,
        lhs_params={
            "Ks_mult":          (7.5,   12.0),   # same range as series 81
            "kinemvelcoef":     (2.0,   6.0),
            "flowexp":          (0.25,  0.35),
            "channelroughness": (0.008, 0.020),
            "psiB_mult":        (0.8,   1.25),   # NEW: range from series 59 single-param sweep
        },
        id_spec=[
            ("Ks_mult",          "Ks",   "x"),
            ("kinemvelcoef",     "cv",   ""),
            ("flowexp",          "r",    ""),
            ("channelroughness", "n",    ""),
            ("psiB_mult",        "psiB", "x"),
        ],
        description="6-parameter LHS sweep series 82: adds psiB_mult (0.8-1.25x) to series 81.",
    ),
    "83": dict(
        series_label="83",
        category="83_lhs_11param",
        out_csv="lhs_results_11param_83.csv",
        swept_tag="lhs_11param",
        default_n=100,
        label_mode="short",         # short_label() -- keeps run_id well under the
                                     # ~65-char threshold; longer paths have caused
                                     # tRIBS to hang and fail to write output
        include_As_Au_in_metrics=True,
        lhs_params={
            "Ks_mult":           (7.5,  9.5),    # narrowed from series 82 (7.5-12)
            "kinemvelcoef":      (2.5,  6.5),
            "flowexp":           (0.20, 0.35),
            "channelroughness":  (0.02, 0.03),   # tightened from series 82
            "channelwidthcoeff": (1.8,  2.5),    # NEW
            "thetaS_mult":       (0.93, 1.15),   # NEW; lo=0.93 keeps thetaS > theta*_s=0.37
            "psiB_mult":         (0.80, 1.25),   # carried from series 82
        },
        id_spec=[
            ("Ks_mult",           "Ks",   "x"),
            ("kinemvelcoef",      "cv",   ""),
            ("flowexp",           "r",    ""),
            ("channelroughness",  "n",    ""),
            ("channelwidthcoeff", "cw",   ""),
            ("thetaS_mult",       "thS",  "x"),
            ("psiB_mult",         "psiB", "x"),
        ],
        description=(
            "11-parameter LHS sweep series 83: Ks, cv, r, n, channelwidth, "
            "thetaS_mult, psiB_mult swept; f, As, Au fixed."
        ),
    ),
}

# ======================================================================
# CONFIG -- edit this one line to switch series
# ======================================================================
ACTIVE_SERIES = "83"
# ======================================================================

if ACTIVE_SERIES not in SERIES_PRESETS:
    raise ValueError(f"ACTIVE_SERIES must be one of {list(SERIES_PRESETS)}, got {ACTIVE_SERIES!r}")

PRESET = SERIES_PRESETS[ACTIVE_SERIES]

# Fixed across all three series -- none of 81/82/83 sweep f_RS_abs
F_RS_ABS_FIXED = 0.020   # mm^-1
AS_FIXED       = 1.0     # matches builder.BASELINE default; only 83's CSV logs it explicitly
AU_FIXED       = 1.0


def short_label(value, decimals=2):
    """
    Compact label for run IDs: round to `decimals` places, strip trailing
    zeros, replace '.' with 'p'. Used by series 83 (7 free params) to keep
    run_id length well under the filesystem/tRIBS-safe threshold -- longer
    paths have caused tRIBS to hang and fail to write output (see project
    notes). Series 81/82 use builder.value_to_label() (full precision)
    since their shorter param lists don't approach that limit.

    NOTE (found during consolidation): the original run_lhs_11param.py's
    docstring claimed this rounds to 3 decimals, but its actual function
    signature defaulted to decimals=2, and every call site used that
    default -- so every series-83 run_id already on disk is 2-decimal,
    not 3. This default is set to 2 to match that real behavior, not the
    docstring's stated intent, so --skip_existing keeps recognizing
    already-completed series 83 runs. Flagging in case 3-decimal
    precision was actually the intended design and you'd rather fix it
    going forward (would break continuity with existing run folders).
    """
    rounded = round(value, decimals)
    s = f"{rounded:.{decimals}f}".rstrip('0')
    if s.endswith('.'):
        s += '0'
    return s.replace('.', 'p')


def label_value(value, preset):
    return short_label(value) if preset["label_mode"] == "short" else builder.value_to_label(value)


# ------------------------------------------------------------------
# LHS SAMPLE GENERATION (identical across all three original scripts)
# ------------------------------------------------------------------
def generate_lhs_samples(n, lhs_params, seed=None):
    """
    Generate n Latin Hypercube samples across the parameter ranges.
    Each parameter range is divided into n equal intervals and one sample
    is drawn uniformly from each interval, then independently shuffled
    across parameters (ensuring full marginal coverage).
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for param, (lo, hi) in lhs_params.items():
        intervals = np.linspace(lo, hi, n + 1)
        points = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        samples[param] = points
    return pd.DataFrame(samples)


# ------------------------------------------------------------------
# RUN ID CONSTRUCTION
# ------------------------------------------------------------------
def build_lhs_run_id(sample: dict, preset=PRESET):
    parts = [f"{prefix}{label_value(sample[param], preset)}{suffix}"
             for param, prefix, suffix in preset["id_spec"]]
    change_tested = "_".join(parts)
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
    Build one tRIBS input file for one LHS point and run it.

    `sample` holds only the parameters this preset actually sweeps (see
    preset["lhs_params"]). Parameters a given series doesn't sweep fall
    back to their fixed/baseline value:
      - thetaS_mult, psiB_mult default to 1.0 (no-op multiplier) when not
        swept -- reproduces 81's/82's un-multiplied baseline exactly.
      - channelwidthcoeff defaults to builder.BASELINE["channelwidthcoeff"]
        (2.33) when not swept -- reproduces 81/82 exactly.
      - f_RS_abs is always pinned at F_RS_ABS_FIXED (all three series).
      - As/Au are always builder.BASELINE defaults (1.0).
    """
    ks_mult           = sample["Ks_mult"]
    kinemvelcoef      = sample["kinemvelcoef"]
    flowexp           = sample["flowexp"]
    channelroughness  = sample["channelroughness"]
    channelwidthcoeff = sample.get("channelwidthcoeff", builder.BASELINE["channelwidthcoeff"])
    thetas_mult       = sample.get("thetaS_mult", 1.0)
    psib_mult         = sample.get("psiB_mult", 1.0)

    run_id, change_tested = build_lhs_run_id(sample, preset)

    script_dir = Path.cwd()
    project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
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
    if "channelwidthcoeff" in sample:
        builder.BASELINE["channelwidthcoeff"] = channelwidthcoeff
    # thetaS_mult / psiB_mult are not stored in BASELINE -- applied directly below

    try:
        baseline = builder.BASELINE

        proj = Project(os.getcwd(), builder.LOCATION, builder.EPSG)

        # --- Land use raster ---
        landuse_ras = '../smf_init_data/LandUse.asc'
        shutil.copy(landuse_ras, proj.directories['land'])

        # --- Soil raster ---
        soil_ras = '../smf_init_data/ADOT_SoilTypes.asc'
        shutil.copy(soil_ras, proj.directories['soil'])
        soil_ras = f"{proj.directories['soil']}/{os.path.basename(soil_ras)}"

        # --- Soil class ---
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
            soil_cls['As'] = AS_FIXED
            soil_cls['Au'] = AU_FIXED
            soil_cls['ks'] = 0.7
            soil_cls['Cs'] = 1.4e6
            cid = str(soil_cls['ID'])
            if cid in builder.SOIL_PARAM_LOOKUP:
                soil_params = builder.SOIL_PARAM_LOOKUP[cid]
                soil_cls['Ks']     = soil_params['Ks'] * ks_mult
                soil_cls['thetaR'] = soil_params['thetaR']
                soil_cls['m']      = soil_params['m']
                soil_cls['n']      = soil_params['n']
                raw_thetaS = soil_params['thetaS'] * thetas_mult
                soil_cls['thetaS'] = max(raw_thetaS, soil_params['thetaR'] + 0.01)
                soil_cls['PsiB']   = soil_params['PsiB'] * psib_mult
                soil_cls['f']      = F_RS_ABS_FIXED if cid == '1' else soil_params['f']
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
        model.channelwidthcoeff['value'] = channelwidthcoeff

        model.startdate['value']  = builder.START_DATE
        model.runtime['value']    = builder.RUNTIME_HOURS
        model.rainintrvl['value'] = builder.RAIN_INTERVAL

        input_file_abs     = run_input_dir  / f"{run_id}.in"
        log_file_abs        = log_dir        / f"{run_id}.log"
        output_prefix_abs   = run_results_dir / run_id

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
              f"n={channelroughness:.4f}  cw={channelwidthcoeff:.3f}  "
              f"thS={thetas_mult:.3f}x  psiB={psib_mult:.3f}x  "
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
            "thetaS_mult":               thetas_mult,
            "psiB_mult":                 psib_mult,
            "As_value":                  AS_FIXED,
            "Au_value":                  AU_FIXED,
            "optpercolation":            baseline["optpercolation"],
            "channelconductivity_mmhr":  baseline["channelconductivity_mmhr"],
            "channelporosity":           baseline["channelporosity"],
            "kinemvelcoef":              kinemvelcoef,
            "flowexp":                   flowexp,
            "channelroughness":          channelroughness,
            "channelwidthcoeff":         channelwidthcoeff,
            "input_file":                input_file,
            "log_file":                  log_file,
            "output_prefix":             output_prefix,
            "csv_export_dir":            os.path.relpath(csv_export_dir,      script_dir),
            "plot_export_dir":           os.path.relpath(plot_export_dir,     script_dir),
            "summary_export_dir":        os.path.relpath(summary_export_dir,  script_dir),
            "swept_param":               preset["swept_tag"],
            "swept_value":               ks_mult,
        }

        config_path = calib_dir / "current_run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))

    finally:
        builder.BASELINE = original_baseline

    metrics = runner.run_and_score()

    for param in preset["lhs_params"]:
        metrics[param] = sample[param]
    metrics["f_RS_abs"] = F_RS_ABS_FIXED
    if preset["include_As_Au_in_metrics"]:
        metrics["As_value"] = AS_FIXED
        metrics["Au_value"] = AU_FIXED
    metrics["swept_param"] = preset["swept_tag"]

    return metrics


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=PRESET["description"])
    parser.add_argument("--n", type=int, default=PRESET["default_n"],
                        help=f"Number of LHS samples (default: {PRESET['default_n']})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose compare CSV already exists")
    args = parser.parse_args()

    script_dir = Path.cwd()
    project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
    calib_dir    = project_root / "calibration_work"
    summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / PRESET["out_csv"]

    samples = generate_lhs_samples(args.n, PRESET["lhs_params"], seed=args.seed)

    print(f"\n{'='*70}")
    print(f"LHS sweep -- series {PRESET['series_label']} -- "
          f"{len(PRESET['lhs_params'])} free parameters  (n={args.n}, seed={args.seed})")
    for param, (lo, hi) in PRESET["lhs_params"].items():
        print(f"  {param:<20s} {lo} - {hi}")
    print(f"  f_RS_abs:            {F_RS_ABS_FIXED} mm^-1  [FIXED, all series]")
    print(f"  Output: {out_path.name}")
    print(f"{'='*70}\n")

    existing_df      = load_existing_results(out_path)
    existing_run_ids = set(existing_df["run_id"].values) if not existing_df.empty else set()

    all_results = []
    if not existing_df.empty:
        all_results.extend(existing_df.to_dict("records"))

    completed   = 0
    skipped     = 0
    failed      = 0
    sweep_start = time.time()

    for i, row in samples.iterrows():
        sample = {param: row[param] for param in PRESET["lhs_params"]}
        run_id, _ = build_lhs_run_id(sample)

        label_str = "  ".join(f"{p}={sample[p]:.4f}" for p in PRESET["lhs_params"])
        print(f"\n[{i+1:>3}/{args.n}]  {label_str}")
        print(f"         -> {run_id}")

        if args.skip_existing and csv_already_exists(run_id, calib_dir):
            print(f"  SKIP (CSV exists): {run_id}")
            skipped += 1
            metrics_file = summary_dir / f"{run_id}_metrics_summary.csv"
            if metrics_file.exists() and run_id not in existing_run_ids:
                try:
                    df_m = pd.read_csv(metrics_file)
                    m = df_m.iloc[0].to_dict()
                    m.update(sample)
                    m["f_RS_abs"] = F_RS_ABS_FIXED
                    if PRESET["include_As_Au_in_metrics"]:
                        m["As_value"] = AS_FIXED
                        m["Au_value"] = AU_FIXED
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
        remaining     = args.n - completed - skipped - failed
        if completed > 0:
            avg_time = total_elapsed / completed
            eta_min  = (avg_time * remaining) / 60
            print(f"  Run time: {elapsed/60:.1f} min  |  ETA: {eta_min:.0f} min remaining")

        if all_results:
            pd.DataFrame(all_results).to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"Sweep complete:  {completed} ran,  {skipped} skipped,  {failed} failed")
    print(f"{'='*70}\n")

    if all_results:
        final_df = pd.DataFrame(all_results).sort_values("kge", ascending=False)
        final_df.to_csv(out_path, index=False)
        print(f"Results saved to:\n  {out_path}")
        print(f"  Total runs in file: {len(final_df)}")

        print(f"\nKGE summary:")
        print(f"  Min:    {final_df['kge'].min():.3f}")
        print(f"  Median: {final_df['kge'].median():.3f}")
        print(f"  Max:    {final_df['kge'].max():.3f}")

        print(f"\n  Top 10 runs by KGE:")
        cols = (["run_id"] + list(PRESET["lhs_params"].keys())
                + ["kge", "nse", "pbias_pct", "peak_timing_error_hr"])
        available = [c for c in cols if c in final_df.columns]
        print(final_df[available].head(10).to_string(index=False, float_format="%.4f"))

        print(f"\n  Parameter-KGE correlations (Pearson r):")
        for param in PRESET["lhs_params"]:
            if param in final_df.columns and "kge" in final_df.columns:
                r = final_df[param].corr(final_df["kge"])
                print(f"    {param:<20s}  r = {r:+.3f}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
