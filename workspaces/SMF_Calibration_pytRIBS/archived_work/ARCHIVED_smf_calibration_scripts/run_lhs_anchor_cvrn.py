"""
run_lhs_anchor_cvrn.py
=======================
Series 96 — cv/r/n identifiability check across volume-matched Ks_mult/f_RS_abs
anchors.

Purpose
-------
You've found (tonight, by manual bisection) 2-3 (Ks_mult, f_RS_abs) pairs that
all reproduce the same synthetic-truth volume, with cv/r/n held at the truth
values (4.5, 0.24, 0.026) during that search. This script asks the follow-up
question: does the established cv/r/n identifiability picture (Ks >> r >> cv
~= n) hold up regardless of which volume-matched anchor you're sitting at, or
does the routing story change depending on which Ks/f pair you picked?

Design choice: the SAME LHS sample set (same seed) is reused across every
anchor, rather than drawing independent random samples per anchor. This makes
it a paired comparison -- for a given (cv, r, n) triple, you can directly
compare metrics across anchors, rather than confounding anchor differences
with sampling differences.

Fill in ANCHORS below with your confirmed pairs before running.

Usage (run from the smf_demo directory):
    python run_lhs_anchor_cvrn.py                  # 50 samples/anchor, seed=42
    python run_lhs_anchor_cvrn.py --n 100          # more samples per anchor
    python run_lhs_anchor_cvrn.py --seed 99        # different (shared) seed
    python run_lhs_anchor_cvrn.py --skip_existing  # resume an interrupted run

Requires calibration_work/synth_truth/ to contain the truth .qout file for
Ks_mult=8.5, f_RS_abs=0.020, kinemvelcoef=4.5, flowexp=0.24,
channelroughness=0.026 (see note in module docstring below about
regenerating it fresh rather than trusting an older archived file, since
BASELINE in build_sensitivity_run.py has since drifted from these truth
values).

Output: one CSV per anchor, plus a combined CSV with all anchors stacked:
    calibration_work/03_comparisons/summary_tables/lhs_results_anchor_<label>_96.csv
    calibration_work/03_comparisons/summary_tables/lhs_results_anchor_ALL_96.csv

Update 7.7.26: Having trouble getting this script (in run_lhs_nanchor_cvrn_98.py
and probe_old_script_new_anchor.py) to run with different Ks and f values. 
Running a test case with n (runs) = 2
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
# ANCHORS -- fill these in with tonight's confirmed volume-matched pairs.
# "label" must be filesystem-safe (used in filenames) and unique.
# ------------------------------------------------------------------
ANCHORS = [
    # {"label": "truth",   "Ks_mult": 8.5, "f_RS_abs": 0.020},  # optional: the
    #                                                            # original truth
    #                                                            # pair, for a
    #                                                            # clean reference
    #                                                            # column (S92
    #                                                            # already covers
    #                                                            # this -- only
    #                                                            # include if you
    #                                                            # want it in the
    #                                                            # SAME csv/seed
    #                                                            # as the new
    #                                                            # anchors)
    {"label": "anchorA", "Ks_mult": 5.0, "f_RS_abs": 0.0075},
    {"label": "anchorB", "Ks_mult": 6.5, "f_RS_abs": 0.011},
    # {"label": "anchorC", "Ks_mult": None, "f_RS_abs": None}, # optional 3rd --
    #                                                          # deferred: Ks=9.5
    #                                                          # and Ks=10.5 both
    #                                                          # showed anomalous
    #                                                          # (reversed-sign)
    #                                                          # f-PBIAS response
    #                                                          # and no volume
    #                                                          # match was found;
    #                                                          # both anchors here
    #                                                          # are below true
    #                                                          # Ks=8.5 as a result
]

# ------------------------------------------------------------------
# LHS PARAMETER RANGES -- identical to Series 92 for cross-series
# comparability. Only cv/r/n are free here; Ks_mult/f_RS_abs are fixed
# per-anchor above.
# ------------------------------------------------------------------
LHS_PARAMS = {
    "kinemvelcoef":     {"lo": 2.5,  "hi": 6.5},   # true = 4.5
    "flowexp":          {"lo": 0.18, "hi": 0.35},  # true = 0.24
    "channelroughness": {"lo": 0.02, "hi": 0.10},  # true = 0.026
}

# Truth values (same regardless of anchor -- only Ks_mult/f_RS_abs differ
# between anchors; the underlying synthetic truth hydrograph was generated
# with these cv/r/n values).
TRUTH_VALUES = {
    "Ks_mult":          8.5,
    "f_RS_abs":         0.020,
    "kinemvelcoef":     4.5,
    "flowexp":          0.24,
    "channelroughness": 0.026,
}
TRUTH_TOL = 1e-6

LHS_SERIES   = "96"
LHS_CATEGORY = "96_lhs_anchor_cvrn"


def is_truth_run(ks_mult, f_rs_abs, cv, r, n):
    """True only if ALL five values match truth (only possible if an anchor
    happens to equal the original truth Ks/f AND the sampled point happens
    to equal the true cv/r/n)."""
    return (abs(ks_mult - TRUTH_VALUES["Ks_mult"])          < TRUTH_TOL and
            abs(f_rs_abs - TRUTH_VALUES["f_RS_abs"])        < TRUTH_TOL and
            abs(cv - TRUTH_VALUES["kinemvelcoef"])          < TRUTH_TOL and
            abs(r  - TRUTH_VALUES["flowexp"])               < TRUTH_TOL and
            abs(n  - TRUTH_VALUES["channelroughness"])      < TRUTH_TOL)


# ------------------------------------------------------------------
# LHS SAMPLE GENERATION (identical method to Series 91-95)
# ------------------------------------------------------------------
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
# BUILD + RUN ONE (anchor, cv, r, n) POINT
# ------------------------------------------------------------------
def build_and_run_lhs(anchor_label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness):
    run_id, change_tested = build_lhs_run_id(
        anchor_label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness)

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

        print(f"  [{anchor_label}]  Ks={ks_mult:.3f}x  f={f_rs_abs:.4f}  "
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

    metrics = runner.run_and_score()

    metrics["anchor_label"]     = anchor_label
    metrics["Ks_mult"]          = ks_mult
    metrics["f_RS_abs"]         = f_rs_abs
    metrics["kinemvelcoef"]     = kinemvelcoef
    metrics["flowexp"]          = flowexp
    metrics["channelroughness"] = channelroughness
    metrics["swept_param"]      = f"lhs_anchor_{anchor_label}"

    return metrics


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Series 96 -- cv/r/n LHS sweep repeated across volume-matched anchors.")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of LHS samples per anchor (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed -- SAME seed used for every anchor by design (default: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose compare CSV already exists")
    args = parser.parse_args()

    unfilled = [a["label"] for a in ANCHORS if a["Ks_mult"] is None or a["f_RS_abs"] is None]
    if unfilled:
        raise ValueError(
            f"ANCHORS entries {unfilled} still have Ks_mult/f_RS_abs = None. "
            f"Fill in your confirmed volume-matched pairs at the top of this "
            f"script before running."
        )

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir   = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Same LHS sample set reused for every anchor -- paired design.
    samples = generate_lhs_samples(args.n, LHS_PARAMS, seed=args.seed)

    print(f"\n{'='*70}")
    print(f"LHS sweep -- Series {LHS_SERIES} -- cv/r/n across {len(ANCHORS)} anchors")
    print(f"  ({args.n} samples/anchor, seed={args.seed}, SAME sample set per anchor)")
    print(f"  kinemvelcoef:     {LHS_PARAMS['kinemvelcoef']['lo']:.2f} - "
          f"{LHS_PARAMS['kinemvelcoef']['hi']:.2f}          [true = {TRUTH_VALUES['kinemvelcoef']}]")
    print(f"  flowexp:          {LHS_PARAMS['flowexp']['lo']:.2f} - "
          f"{LHS_PARAMS['flowexp']['hi']:.2f}          [true = {TRUTH_VALUES['flowexp']}]")
    print(f"  channelroughness: {LHS_PARAMS['channelroughness']['lo']:.3f} - "
          f"{LHS_PARAMS['channelroughness']['hi']:.3f}         [true = {TRUTH_VALUES['channelroughness']}]")
    for a in ANCHORS:
        print(f"  Anchor '{a['label']}':  Ks_mult={a['Ks_mult']}  f_RS_abs={a['f_RS_abs']}")
    print(f"{'='*70}\n")

    all_anchor_results = []

    for anchor in ANCHORS:
        label    = anchor["label"]
        ks_mult  = anchor["Ks_mult"]
        f_rs_abs = anchor["f_RS_abs"]

        out_path = summary_dir / f"lhs_results_anchor_{label}_{LHS_SERIES}.csv"
        existing_df      = load_existing_results(out_path)
        existing_run_ids = (set(existing_df["run_id"].values)
                            if not existing_df.empty else set())

        anchor_results = []
        if not existing_df.empty:
            anchor_results.extend(existing_df.to_dict("records"))

        completed, skipped, excluded, failed = 0, 0, 0, 0
        sweep_start = time.time()

        print(f"\n--- Anchor '{label}'  (Ks_mult={ks_mult}, f_RS_abs={f_rs_abs}) ---")

        for i, row in samples.iterrows():
            kinemvelcoef     = row["kinemvelcoef"]
            flowexp          = row["flowexp"]
            channelroughness = row["channelroughness"]

            if is_truth_run(ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness):
                print(f"[{label} {i+1:>3}/{args.n}]  EXCLUDED: matches truth exactly.")
                excluded += 1
                continue

            run_id, _ = build_lhs_run_id(
                label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness)

            print(f"\n[{label} {i+1:>3}/{args.n}]  cv={kinemvelcoef:.3f}  "
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
                metrics = build_and_run_lhs(
                    label, ks_mult, f_rs_abs, kinemvelcoef, flowexp, channelroughness)
                anchor_results = [r for r in anchor_results if r.get("run_id") != run_id]
                anchor_results.append(metrics)
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
                print(f"  Run time: {elapsed/60:.1f} min  |  anchor ETA: {eta_min:.0f} min remaining")

            if anchor_results:
                pd.DataFrame(anchor_results).to_csv(out_path, index=False)

        print(f"\nAnchor '{label}' complete: {completed} ran, {skipped} skipped, "
              f"{excluded} excluded, {failed} failed")

        if anchor_results:
            final_df = pd.DataFrame(anchor_results).sort_values("kge", ascending=False)
            final_df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}  ({len(final_df)} rows)")
            all_anchor_results.extend(final_df.to_dict("records"))

    # ------------------------------------------------------------------
    # Combined output across all anchors -- the file you'll actually use
    # for the cross-anchor comparison in the morning.
    # ------------------------------------------------------------------
    if all_anchor_results:
        combined_path = summary_dir / f"lhs_results_anchor_ALL_{LHS_SERIES}.csv"
        combined_df = pd.DataFrame(all_anchor_results)
        combined_df.to_csv(combined_path, index=False)
        print(f"\n{'='*70}")
        print(f"All anchors complete. Combined file: {combined_path.name}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"\nKGE by anchor:")
        print(combined_df.groupby("anchor_label")["kge"].describe()[["min", "50%", "max"]])
        print(f"\nParameter-KGE correlations by anchor (Pearson r):")
        for label in combined_df["anchor_label"].unique():
            sub = combined_df[combined_df["anchor_label"] == label]
            corrs = {p: sub[p].corr(sub["kge"])
                     for p in ["kinemvelcoef", "flowexp", "channelroughness"]}
            print(f"  {label}: " + "  ".join(f"{p}={r:+.3f}" for p, r in corrs.items()))
        print(f"{'='*70}\n")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()