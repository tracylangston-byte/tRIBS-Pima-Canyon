"""
probe_anchor_Ks6p25lo_99.py
============================
Timeout-safe diagnostic probe for the Ks_mult=6.25x / f_RS_abs=0.0102 anchor
candidate, before it is trusted for inclusion in run_lhs_nanchor_cvrn_99.py.

Why this exists
----------------
This candidate is volume-matched (PBIAS=0%) and passed an individual
smoke test (one build+run at fixed/default cv-r-n) -- but it sits almost
exactly on top of the Series 98 hang point (Ks=6.226573x, f=0.010926: ~0.4%
off in Ks, ~7% off in f), where every one of several different (cv, r, n)
draws hung or ran 13-94+ min against a <1 min baseline. A single default-
param smoke test does not exercise the cv/r/n range that actually triggered
Series 98's failures, so it doesn't clear this point on its own. This probe
runs a batch of (cv, r, n) draws at this exact anchor and checks whether any
of them reproduce that hang.

What makes this different from every other run script in this project
------------------------------------------------------------------------
build_sensitivity_run.py + run_sensitivity_single.py (and every LHS script
built on them, including run_lhs_nanchor_cvrn_99.py) call tRIBS via a
blocking `os.system(...)` inside the same Python process, with no timeout --
if tRIBS hangs, the whole script hangs with it, indefinitely. That's an
acceptable risk for pre-vetted parameter regions, but it's exactly the
failure mode this probe needs to survive. Instead, this script launches
`run_sensitivity_single.py` as a SEPARATE subprocess, in its own process
group, with a hard wall-clock timeout. If a draw exceeds the timeout, the
entire process group (python -> shell -> tRIBS binary) is killed with
SIGKILL together, and the probe logs a HANG and moves to the next draw
instead of blocking forever.

IMPORTANT -- these draws are NOT reusable in the main Series 99 sweep:
this probe uses a small n and a different seed (777, vs. Series 99's 42)
purely for a go/no-go signal. Per project convention, LHS stratification
changes entirely when n changes, even with the same seed, so these draws
are not a subset of the eventual full n=50 anchor sweep. Results are
written to their own probe log / folder, kept separate from Series 99
output so there's no collision or confusion either way this turns out.

Usage (run from the smf_demo directory):
    python probe_anchor_Ks6p25lo_99.py                  # 12 draws, 5-min timeout/draw
    python probe_anchor_Ks6p25lo_99.py --n 20 --timeout 600
    python probe_anchor_Ks6p25lo_99.py --n 8 --timeout 120   # fail fast, tighter margin

Decision rule printed at the end:
    0 hangs/failures across all draws -> safe to uncomment 'Ks6p25lo' in
        run_lhs_nanchor_cvrn_99.py's ANCHORS list and rerun that script
        with --skip_existing.
    any hangs/failures                -> keep it excluded. This would be a
        second independent confirmation of a problem in the Ks~6.2x/
        f~0.011 region, worth flagging to Josh Cederstrom.

Output:
    calibration_work/03_comparisons/summary_tables/probe_log_Ks6p25lo_99probe.csv
"""

import argparse
import os
import sys
import json
import time
import signal
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

import build_sensitivity_run as builder
from pytRIBS.classes import Project, Soil, Land, Met, Model

# ------------------------------------------------------------------
# Fixed anchor under test -- do not change without renaming the script;
# this is deliberately a single-anchor diagnostic tool, not a general one.
# ------------------------------------------------------------------
ANCHOR_LABEL = "Ks6p25lo"
ANCHOR_KS    = 6.25
ANCHOR_F     = 0.0102

# ------------------------------------------------------------------
# cv/r/n ranges -- identical to run_lhs_nanchor_cvrn_99.py, so a clean
# result here is directly informative about that script's range, even
# though these particular draws aren't reused in it.
# ------------------------------------------------------------------
LHS_PARAMS = {
    "kinemvelcoef":     {"lo": 2.5,  "hi": 6.5},
    "flowexp":          {"lo": 0.18, "hi": 0.35},
    "channelroughness": {"lo": 0.02, "hi": 0.10},
}

PROBE_SERIES   = "99probe"
PROBE_CATEGORY = "99probe_Ks6p25lo"
PROBE_SEED     = 777          # deliberately different from the Series 99
                               # seed (42) -- these draws are a separate
                               # diagnostic set, not a subset.


def generate_lhs_samples(n, params, seed=None):
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
# BUILD ONLY -- writes the .in file + current_run_config.json.
# Mirrors build_and_run_lhs() in run_lhs_anchor_cvrn.py exactly, but stops
# short of calling run_sensitivity_single.py in-process -- that happens
# afterward, as a separate, killable subprocess (see run_with_timeout).
# ------------------------------------------------------------------
def build_only(kinemvelcoef, flowexp, channelroughness):
    ks_lbl = builder.value_to_label(ANCHOR_KS)
    f_lbl  = builder.value_to_label(ANCHOR_F)
    cv_lbl = builder.value_to_label(kinemvelcoef)
    r_lbl  = builder.value_to_label(flowexp)
    n_lbl  = builder.value_to_label(channelroughness)
    change_tested = (f"{ANCHOR_LABEL}_Ks{ks_lbl}x_f{f_lbl}_"
                      f"cv{cv_lbl}_r{r_lbl}_n{n_lbl}")
    run_id = f"{builder.LOCATION}_{builder.EVENT_DATE}_{PROBE_SERIES}_{change_tested}"

    script_dir   = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir    = project_root / "calibration_work"

    run_input_dir       = calib_dir / "01_run_inputs"  / PROBE_CATEGORY
    run_results_dir     = calib_dir / "02_results"     / PROBE_CATEGORY / run_id
    csv_export_dir      = calib_dir / "03_comparisons" / "csv_exports"
    plot_export_dir     = calib_dir / "03_comparisons" / "hydrograph_plots"
    summary_export_dir  = calib_dir / "03_comparisons" / "summary_tables"
    log_dir              = calib_dir / "06_logs"

    for folder in [run_input_dir, run_results_dir, csv_export_dir,
                   plot_export_dir, summary_export_dir, log_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    original_baseline = builder.BASELINE.copy()
    builder.BASELINE["Ks_mult"]          = ANCHOR_KS
    builder.BASELINE["f_RS_abs"]         = ANCHOR_F
    builder.BASELINE["kinemvelcoef"]     = kinemvelcoef
    builder.BASELINE["flowexp"]          = flowexp
    builder.BASELINE["channelroughness"] = channelroughness

    try:
        baseline = builder.BASELINE

        proj = Project(os.getcwd(), builder.LOCATION, builder.EPSG)

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

        for soil_cls in soil_table:
            soil_cls['As'] = baseline["As_value"]
            soil_cls['Au'] = baseline["Au_value"]
            soil_cls['ks'] = 0.7
            soil_cls['Cs'] = 1.4e6
            cid = str(soil_cls['ID'])
            if cid in builder.SOIL_PARAM_LOOKUP:
                soil_params = builder.SOIL_PARAM_LOOKUP[cid]
                soil_cls['Ks']     = soil_params['Ks'] * ANCHOR_KS
                soil_cls['thetaS'] = soil_params['thetaS']
                soil_cls['thetaR'] = soil_params['thetaR']
                soil_cls['m']      = soil_params['m']
                soil_cls['PsiB']   = soil_params['PsiB']
                soil_cls['n']      = soil_params['n']
                # f: RS soil uses this anchor's f; all others use baseline
                soil_cls['f'] = ANCHOR_F if cid == '1' else soil_params['f']
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

        land = Land(meta=proj.meta)
        land.landmapname['value']   = f"{proj.directories['land']}/LandUse.asc"
        land.landtablename['value'] = f"{proj.directories['land']}/land_use_params.ldt"
        landuse_list = []
        for lu_id, lp in builder.LAND_PARAM_LOOKUP.items():
            row = lp.copy(); row['ID'] = lu_id; row['a'] = -9999; row['b1'] = -9999
            landuse_list.append(row)
        land.write_landuse_table(landuse_list, land.landtablename['value'])

        met = Met(meta=proj.meta)
        met.hydrometbasename['value'] = builder.LOCATION
        met.hydrometstations['value'] = "../smf_init_data/met/Master_Met.sdf"
        met.gaugestations['value']    = "../smf_init_data/met/Master_Precip.sdf"

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

        print(f"  [{ANCHOR_LABEL}]  Ks={ANCHOR_KS:.3f}x  f={ANCHOR_F:.4f}  "
              f"cv={kinemvelcoef:.3f}  r={flowexp:.3f}  n={channelroughness:.4f}")

        run_config = {
            "location":                  builder.LOCATION,
            "event_date":                builder.EVENT_DATE,
            "run_number":                PROBE_SERIES,
            "change_tested":             change_tested,
            "run_id":                    run_id,
            "run_category":              PROBE_CATEGORY,
            "start_date":                builder.START_DATE,
            "runtime_hours":             builder.RUNTIME_HOURS,
            "rain_interval_hours":       builder.RAIN_INTERVAL,
            "event_start":               builder.EVENT_START,
            "event_end":                 builder.EVENT_END,
            "Ks_mult":                   ANCHOR_KS,
            "f_RS_abs":                  ANCHOR_F,
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
            "swept_param":               f"lhs_probe_{ANCHOR_LABEL}",
            "swept_value":               ANCHOR_KS,
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
# would.
# ------------------------------------------------------------------
def run_with_timeout(timeout_sec):
    proc = subprocess.Popen(
        [sys.executable, "run_sensitivity_single.py"],
        cwd=Path.cwd(),
        start_new_session=True,   # own process group -> killable as a unit
    )
    t0 = time.time()
    try:
        returncode = proc.wait(timeout=timeout_sec)
        elapsed = time.time() - t0
        return returncode, elapsed, False   # completed, not timed out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        elapsed = time.time() - t0
        return None, elapsed, True          # timed out / hung


def main():
    parser = argparse.ArgumentParser(
        description="Timeout-safe probe for the Ks=6.25x/f=0.0102 anchor "
                    "candidate before trusting it in run_lhs_nanchor_cvrn_99.py.")
    parser.add_argument("--n", type=int, default=12,
                        help="Number of probe draws (default: 12 -- this is a "
                             "go/no-go diagnostic, not a full sweep)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-draw hard timeout in seconds (default: 300 = "
                             "5 min, i.e. 5x the <1 min normal baseline, well "
                             "under the 13+ min Series 98 hangs took)")
    parser.add_argument("--seed", type=int, default=PROBE_SEED,
                        help=f"LHS seed for the probe draws (default: {PROBE_SEED}, "
                             "deliberately different from the Series 99 seed)")
    args = parser.parse_args()

    script_dir   = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir    = project_root / "calibration_work"
    summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    probe_log_path = summary_dir / f"probe_log_{ANCHOR_LABEL}_{PROBE_SERIES}.csv"

    samples = generate_lhs_samples(args.n, LHS_PARAMS, seed=args.seed)

    print(f"\n{'='*70}")
    print(f"PROBE -- anchor '{ANCHOR_LABEL}'  (Ks_mult={ANCHOR_KS}, f_RS_abs={ANCHOR_F})")
    print(f"  {args.n} draws, seed={args.seed}, {args.timeout}s hard timeout/draw")
    print(f"  Baseline expectation: each run <1 min. The Series 98 hang at the")
    print(f"  neighboring Ks=6.226573x/f=0.010926 point ran 13-94+ min.")
    print(f"{'='*70}\n")

    log_rows = []
    n_success, n_hang, n_failed = 0, 0, 0

    for i, row in samples.iterrows():
        cv = row["kinemvelcoef"]
        r  = row["flowexp"]
        n  = row["channelroughness"]

        print(f"[probe {i+1:>2}/{args.n}]  cv={cv:.3f}  r={r:.3f}  n={n:.4f}")

        run_id, summary_export_dir = build_only(cv, r, n)

        returncode, elapsed, timed_out = run_with_timeout(args.timeout)

        if timed_out:
            status = "HANG"
            n_hang += 1
        elif returncode == 0:
            status = "SUCCESS"
            n_success += 1
        else:
            status = "FAILED"
            n_failed += 1

        kge = np.nan
        if status == "SUCCESS":
            metrics_file = summary_export_dir / f"{run_id}_metrics_summary.csv"
            if metrics_file.exists():
                try:
                    kge = pd.read_csv(metrics_file).iloc[0]["kge"]
                except Exception:
                    pass

        flag = ("  <-- SUSPICIOUS (>3 min, watch remaining draws closely)"
                 if (not timed_out and elapsed > 180) else "")
        print(f"  {status}  elapsed={elapsed/60:.2f} min"
              + (f"  KGE={kge:.3f}" if status == "SUCCESS" else "")
              + f"{flag}\n")

        log_rows.append({
            "draw": i + 1, "run_id": run_id,
            "kinemvelcoef": cv, "flowexp": r, "channelroughness": n,
            "status": status, "elapsed_min": elapsed / 60, "kge": kge,
        })
        # Write after every draw so a killed/interrupted probe still leaves
        # a usable partial log.
        pd.DataFrame(log_rows).to_csv(probe_log_path, index=False)

    print(f"\n{'='*70}")
    print(f"PROBE COMPLETE -- {n_success} success, {n_hang} hang/timeout, {n_failed} failed")
    print(f"Log saved: {probe_log_path}")
    if n_hang == 0 and n_failed == 0:
        print(f"\nRESULT: clean across all {args.n} draws. Safe to uncomment "
              f"'{ANCHOR_LABEL}' in run_lhs_nanchor_cvrn_99.py's ANCHORS list "
              f"and rerun that script with --skip_existing.")
    else:
        print(f"\nRESULT: {n_hang} hang(s) / {n_failed} failure(s) out of "
              f"{args.n} draws. Keep '{ANCHOR_LABEL}' excluded from "
              f"run_lhs_nanchor_cvrn_99.py. This would be a second independent "
              f"confirmation of a problem in the Ks~6.2x/f~0.011 region -- "
              f"worth flagging to Josh Cederstrom.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
