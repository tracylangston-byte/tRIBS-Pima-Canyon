"""
build_sensitivity_run.py
========================
Builds one tRIBS input file for a single-parameter sensitivity run.
Mirrors the logic of Make_SMF_Model.ipynb exactly.

Usage (run from the smf_demo directory):
    python build_sensitivity_run.py --param Ks_mult            --value 6.0
    python build_sensitivity_run.py --param f_RS_abs           --value 0.010
    python build_sensitivity_run.py --param f_RS_abs_Ks1       --value 0.025
    python build_sensitivity_run.py --param kinemvelcoef       --value 5.0
    python build_sensitivity_run.py --param flowexp            --value 0.5
    python build_sensitivity_run.py --param channelroughness   --value 0.08

These are for running a single build manually from the terminal — useful
if you want to test one value, inspect the soil audit output, or rebuild
a specific run without executing the full sweep. In normal use, you won't
call this script directly; run_sensitivity_sweep.py calls it automatically
for every value in the sweep list.

Naming convention (no decimal points):
    Ks_mult          -> series 60 -> SMF_20140812_60_Ks6p0x
    f_RS_abs         -> series 61 -> SMF_20140812_61_fRS0p010       (Ks_mult = 7x)
    f_RS_abs_Ks1     -> series 65 -> SMF_20140812_65_fRS0p025_Ks1  (Ks_mult = 1x)
    kinemvelcoef     -> series 62 -> SMF_20140812_62_cv5p0
    flowexp          -> series 63 -> SMF_20140812_63_r0p50
    channelroughness -> series 64 -> SMF_20140812_64_n0p080

Notes on f_RS_abs and f_RS_abs_Ks1:
    Both sweep ABSOLUTE f for the RS soil class (soil ID '1') only.
    All other soil classes (CO, CeD, EbD, Cb) are held at their baseline f.
    The difference is Ks_mult:
        f_RS_abs      uses Ks_mult = 6.1 (best calibrated Ks)
        f_RS_abs_Ks1  uses Ks_mult = 1.0 (uncalibrated baseline Ks)
    This lets you compare how the f response curve changes with Ks.

    Baseline f values by class:
        RS  (ID 1): 0.020  (1/f =  50 mm — caliche-controlled)
        CO  (ID 2): 0.002  (1/f = 500 mm)
        CeD (ID 3): 0.002  (1/f = 500 mm)
        EbD (ID 4): 0.002  (1/f = 500 mm)
        Cb  (ID 5): 0.001  (1/f = 1000 mm)
"""

import argparse
import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------
# BASELINE VALUES
# All parameters are held at these values unless being swept.
# Update previously calibrated parameters here before each new sweep.
# -----------------------------------------------------------------------
BASELINE = {
    "Ks_mult":           6.1,    # best calibrated value from Ks/f sweep
    "f_RS_abs":          0.020,  # absolute f for RS soil (1/mm), best calibrated value from Ks/f sweep
    "f_RS_abs_Ks1":      0.020,  # NA in this run
    "As_value":          1.0,
    "Au_value":          1.0,
    "optpercolation":    0,
    "channelconductivity_mmhr": 70,
    "channelporosity":   0.4,
    "kinemvelcoef":      3,      # I am not starting from the pure best-KGE Ks × cv pair (Ks ≈ 4.49, cv ≈ 2.74), 
                                 # because that region still has too much positive volume bias. 
    "flowexp":           0.3,    # best calibrated value from flowexp sweep
    "channelroughness":  0.04,   # <- this gets overridden per run
    "channelwidthcoeff": 2.33,
}

# Series number and label prefix per parameter
PARAM_CONFIG = {
    "Ks_mult":           {"series": "60", "prefix": "Ks",          "suffix": "x",    "type": "multiplier"},
    "f_RS_abs":          {"series": "61", "prefix": "fRS",         "suffix": "",     "type": "absolute"},
    "kinemvelcoef":      {"series": "62", "prefix": "cv",          "suffix": "",     "type": "absolute"},
    "flowexp":           {"series": "63", "prefix": "r",           "suffix": "",     "type": "absolute"},
    "channelroughness":  {"series": "64", "prefix": "n",           "suffix": "",     "type": "absolute"},
    "f_RS_abs_Ks1":      {"series": "65", "prefix": "fRS",         "suffix": "_Ks1", "type": "absolute"},
}

# Fixed simulation settings
LOCATION       = "SMF"
EVENT_DATE     = "20140812"
START_DATE     = "08/01/2014/00/00"
RUNTIME_HOURS  = 450
RAIN_INTERVAL  = 0.25
EVENT_START    = "2014-08-12 16:00"
EVENT_END      = "2014-08-13 12:00"
EPSG           = 26912

# Soil parameter baselines per class.
# RS f is overridden by the swept value for f_RS_abs and f_RS_abs_Ks1 runs.
# All other classes always use the f listed here.
SOIL_PARAM_LOOKUP = {
    '1': {'Ks': 3.6,  'thetaS': 0.40, 'thetaR': 0.06, 'm': 0.38, 'PsiB': -390, 'f': 0.020, 'n': 0.40},
    '2': {'Ks': 2.8,  'thetaS': 0.40, 'thetaR': 0.05, 'm': 0.25, 'PsiB': -401, 'f': 0.002, 'n': 0.40},
    '3': {'Ks': 6.6,  'thetaS': 0.40, 'thetaR': 0.05, 'm': 0.25, 'PsiB': -183, 'f': 0.002, 'n': 0.40},
    '4': {'Ks': 1.0,  'thetaS': 0.42, 'thetaR': 0.10, 'm': 0.20, 'PsiB': -450, 'f': 0.002, 'n': 0.42},
    '5': {'Ks': 17.3, 'thetaS': 0.39, 'thetaR': 0.03, 'm': 0.18, 'PsiB': -117, 'f': 0.001, 'n': 0.39},
}

# Land use parameters (fixed, not swept)
LAND_PARAM_LOOKUP = {
    '1': {'P': 0.4,  'S': 1.5,  'K': 0.12,  'b2': 4.7,   'Al': 0.18, 'h': 1,    'Kt': 0.4,  'Rs': 120,  'V': 0.15, 'LAI': 1.5, 'theta*_s': 0.37, 'theta*_t': 0.30},
    '2': {'P': 0.4,  'S': 1.5,  'K': 0.12,  'b2': 4.7,   'Al': 0.18, 'h': 1,    'Kt': 0.4,  'Rs': 120,  'V': 0.30, 'LAI': 1.5, 'theta*_s': 0.37, 'theta*_t': 0.30},
    '3': {'P': 0.99, 'S': 0.01, 'K': 0.001, 'b2': 0.001, 'Al': 0.15, 'h': 0.01, 'Kt': 0.99, 'Rs': 9999, 'V': 0.01, 'LAI': 0.01,'theta*_s': 0.37, 'theta*_t': 0.30},
}

def value_to_label(value):
    s = f"{value:.6f}".rstrip('0')
    if s.endswith('.'):
        s = s[:-1] + 'p0'   # preserve the trailing zero for integers like 1.0
    elif '.' in s:
        s = s.replace('.', 'p')
    return s

def build_run_id(param_name, value):
    cfg = PARAM_CONFIG[param_name]
    label = value_to_label(value)
    change_tested = f"{cfg['prefix']}{label}{cfg['suffix']}"
    run_id = f"{LOCATION}_{EVENT_DATE}_{cfg['series']}_{change_tested}"
    return run_id, change_tested


def get_run_category(series_str):
    """Map series number to calibration folder category."""
    n = int(series_str)
    if 60 <= n <= 69:
        return "60_sensitivity"
    return "40_multivariable"


def build_input_file(param_name, value):
    from pytRIBS.classes import Project, Soil, Land, Met, Model
    from pytRIBS.shared.inout import InOut

    # --- Run ID and paths ---
    run_id, change_tested = build_run_id(param_name, value)
    run_category = get_run_category(PARAM_CONFIG[param_name]["series"])

    notebook_dir = Path.cwd()
    project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
    calib_dir    = project_root / "calibration_work"

    run_input_dir      = calib_dir / "01_run_inputs"  / run_category
    run_results_dir    = calib_dir / "02_results"     / run_category / run_id
    csv_export_dir     = calib_dir / "03_comparisons" / "csv_exports"
    plot_export_dir    = calib_dir / "03_comparisons" / "hydrograph_plots"
    summary_export_dir = calib_dir / "03_comparisons" / "summary_tables"
    log_dir            = calib_dir / "06_logs"

    for folder in [run_input_dir, run_results_dir, csv_export_dir,
                   plot_export_dir, summary_export_dir, log_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    input_file_abs    = run_input_dir  / f"{run_id}.in"
    log_file_abs      = log_dir        / f"{run_id}.log"
    output_prefix_abs = run_results_dir / run_id

    input_file    = os.path.relpath(input_file_abs,    notebook_dir)
    log_file      = os.path.relpath(log_file_abs,      notebook_dir)
    output_prefix = os.path.relpath(output_prefix_abs, notebook_dir)

    # --- Resolve parameter values for this run ---
    params = {**BASELINE}       # start from baseline
    params[param_name] = value  # override just the swept param

    # f_RS_abs_Ks1 sweeps f but forces Ks_mult to 1.0 regardless of BASELINE
    if param_name == "f_RS_abs_Ks1":
        params["Ks_mult"] = 1.0

    Ks_mult                  = params["Ks_mult"]
    # Resolve f_RS_abs: both f_RS_abs and f_RS_abs_Ks1 sweep RS soil f
    if param_name == "f_RS_abs_Ks1":
        f_RS_abs = value
    else:
        f_RS_abs = params["f_RS_abs"]

    As_value                 = params["As_value"]
    Au_value                 = params["Au_value"]
    optpercolation           = params["optpercolation"]
    channelconductivity_mmhr = params["channelconductivity_mmhr"]
    channelporosity          = params["channelporosity"]
    kinemvelcoef             = params["kinemvelcoef"]
    flowexp                  = params["flowexp"]
    channelroughness         = params["channelroughness"]
    channelwidthcoeff        = params["channelwidthcoeff"]

    # --- pytRIBS setup ---
    name = LOCATION
    proj = Project(os.getcwd(), name, EPSG)

    # Data files
    landuse_ras = '../smf_init_data/LandUse.asc'
    shutil.copy(landuse_ras, proj.directories['land'])
    landuse_ras = f"{proj.directories['land']}/{os.path.basename(landuse_ras)}"

    soil_ras = '../smf_init_data/ADOT_SoilTypes.asc'
    shutil.copy(soil_ras, proj.directories['soil'])
    soil_ras = f"{proj.directories['soil']}/{os.path.basename(soil_ras)}"

    # Soil class
    soil = Soil(meta=proj.meta)

    shutil.copy('../smf_init_data/SOLUS_Bedrock_m.asc', proj.directories['soil'])
    soil.bedrockfile['value'] = f"{proj.directories['soil']}/SOLUS_Bedrock_m.asc"

    shutil.copy('../smf_init_data/InitGW_95pct_mm.asc', proj.directories['soil'])
    soil.gwaterfile['value'] = f"{proj.directories['soil']}/InitGW_95pct_mm.asc"

    soil_table_src = '../smf_init_data/soils.sdt'
    shutil.copy(soil_table_src, proj.directories['soil'])
    soil.soiltablename['value'] = f"{proj.directories['soil']}/soils.sdt"
    soil.soilmapname['value']   = soil_ras

    soil_table = soil.read_soil_table(textures=True)

    # f sweep params: both f_RS_abs and f_RS_abs_Ks1 override RS soil f only
    f_params = {"f_RS_abs", "f_RS_abs_Ks1"}

    for cls in soil_table:
        cls['As'] = As_value
        cls['Au'] = Au_value
        cls['ks'] = 0.7
        cls['Cs'] = 1.4e6
        cid = str(cls['ID'])
        if cid in SOIL_PARAM_LOOKUP:
            sp = SOIL_PARAM_LOOKUP[cid]
            cls['Ks']     = sp['Ks'] * Ks_mult
            cls['thetaS'] = sp['thetaS']
            cls['thetaR'] = sp['thetaR']
            cls['m']      = sp['m']
            cls['PsiB']   = sp['PsiB']
            cls['n']      = sp['n']
            # RS soil (ID '1') uses swept f for both f sweep parameter types.
            # All other classes always use their baseline f from the lookup.
            if cid == '1' and param_name in f_params:
                cls['f'] = f_RS_abs
            else:
                cls['f'] = sp['f']
        else:
            print(f"WARNING: Soil ID {cid} not in lookup; using defaults.")
            cls['Ks'] = 10.0; cls['thetaS'] = 0.4; cls['thetaR'] = 0.05
            cls['m'] = 0.2; cls['PsiB'] = -200; cls['f'] = 0.001; cls['n'] = 0.4

    working_soil_table    = Path("data/model/soil/soil.sdt")
    soil.write_soil_table(soil_table, str(working_soil_table), textures=True)

    run_specific_soil_abs = run_input_dir / f"soils_{run_id}.sdt"
    if working_soil_table.resolve() != run_specific_soil_abs.resolve():
        shutil.copy(working_soil_table, run_specific_soil_abs)

    soil.soiltablename['value'] = os.path.relpath(run_specific_soil_abs, notebook_dir)
    soil.optsoiltype['value']   = 0

    # Land class
    land = Land(meta=proj.meta)
    land.landmapname['value']   = f"{proj.directories['land']}/LandUse.asc"
    land.landtablename['value'] = f"{proj.directories['land']}/land_use_params.ldt"

    landuse_list = []
    for lu_id, lp in LAND_PARAM_LOOKUP.items():
        row = lp.copy()
        row['ID'] = lu_id
        row['a']  = -9999
        row['b1'] = -9999
        landuse_list.append(row)
    land.write_landuse_table(landuse_list, land.landtablename['value'])

    # Met class
    met = Met(meta=proj.meta)
    met.hydrometbasename['value'] = name
    met.hydrometstations['value'] = "../smf_init_data/met/Master_Met.sdf"
    met.gaugestations['value']    = "../smf_init_data/met/Master_Precip.sdf"

    # Model class
    model = Model(met=met, land=land, soil=soil, mesh=None, meta=proj.meta)

    # Mesh: pre-generated
    model.parallelmode['value']  = 0
    model.optmeshinput['value']  = 1
    model.inputdatafile['value'] = "../smf_init_data/mesh/SMF_mesh"
    model.inputtime['value']     = 0

    # Soil bedrock / snow / land
    model.optbedrock['value']  = 1
    model.optsnow['value']     = 0
    model.optlanduse['value']  = 0

    # Channel loss
    model.optpercolation['value']      = optpercolation
    model.channelconductivity['value'] = channelconductivity_mmhr / 3.6e6
    model.channelporosity['value']     = channelporosity

    # Routing
    model.kinemvelcoef['value']     = kinemvelcoef
    model.flowexp['value']          = flowexp
    model.channelroughness['value'] = channelroughness
    model.channelwidthcoeff['value']= channelwidthcoeff

    # Simulation settings
    model.startdate['value']         = START_DATE
    model.runtime['value']           = RUNTIME_HOURS
    model.outfilename['value']       = output_prefix
    model.outhydrofilename['value']  = output_prefix
    model.rainintrvl['value']        = RAIN_INTERVAL

    # Node output lists
    node_ids_pixel = [1960, 1547, 3082]
    model.write_node_file(node_ids_pixel, 'data/model/pnodes.dat')
    model.nodeoutputlist['value'] = 'data/model/pnodes.dat'

    node_ids_qout = [3202]
    model.write_node_file(node_ids_qout, 'data/model/qnodes.dat')
    model.outletnodelist['value'] = 'data/model/qnodes.dat'

    # Write input file
    model.write_input_file(input_file)

    # THIS SECTION NOT NEEDED IF f IS NOT BEING SWEPT
    # # Print soil audit
    # print(f"\n  Soil table for {run_id}:")
    # print(f"  {'ID':<4} {'Texture':<6} {'Ks (mm/hr)':<14} {'f (1/mm)':<12} {'1/f (mm)':<10} {'% of watershed'}")
    # print(f"  {'-'*4} {'-'*6} {'-'*14} {'-'*12} {'-'*10} {'-'*15}")

    # # Calculate soil class area fractions within the watershed boundary
    # soil_pct = {}
    # try:
    #     from rasterio.features import geometry_mask
    #     from shapely.geometry import mapping

    #     raw_map = InOut.read_ascii(soil_ras)
    #     data      = raw_map['data']
    #     profile   = raw_map['profile']
    #     transform = profile['transform']

    #     if data.ndim == 3:
    #         data = data[0]

    #     watershed_gdf = proj.meta.get('watershed_gdf', None)
    #     if watershed_gdf is None:
    #         import geopandas as gpd
    #         ws_candidates = list(Path(proj.directories['preprocessing']).glob("*watershed*.shp"))
    #         if ws_candidates:
    #             watershed_gdf = gpd.read_file(ws_candidates[0])

    #     if watershed_gdf is not None:
    #         inside = geometry_mask(
    #             [mapping(geom) for geom in watershed_gdf.geometry],
    #             out_shape=data.shape,
    #             transform=transform,
    #             invert=True,
    #             all_touched=False
    #         )
    #     else:
    #         inside = np.ones(data.shape, dtype=bool)

    #     nodata = profile.get('nodata', None)
    #     valid  = inside & np.isfinite(data) & (data >= 0)
    #     if nodata is not None:
    #         valid = valid & (data != nodata)

    #     soil_values = data[valid]
    #     if len(soil_values) > 0:
    #         counts = pd.Series(soil_values).value_counts().sort_index()
    #         total  = counts.sum()
    #         for sid, cnt in counts.items():
    #             soil_pct[str(int(sid))] = 100.0 * cnt / total
    # except Exception as e:
    #     print(f"  (soil % calculation skipped: {e})")

    # for cls in soil_table:
    #     cid    = str(cls['ID'])
    #     tex    = cls.get('Texture', '')
    #     ks_val = cls['Ks']
    #     f_val  = cls['f']
    #     depth  = 1.0 / f_val if f_val > 0 else float('inf')
    #     pct    = soil_pct.get(cid, float('nan'))
    #     marker = " <-- swept" if cid == '1' and param_name in f_params else ""
    #     pct_str = f"{pct:.1f}%" if not np.isnan(pct) else "n/a"
    #     print(f"  {cid:<4} {tex:<6} {ks_val:<14.3f} {f_val:<12.4f} {depth:<10.0f} {pct_str}{marker}")

    # Save run config JSON
    current_run_config = {
        "location":                  LOCATION,
        "event_date":                EVENT_DATE,
        "run_number":                PARAM_CONFIG[param_name]["series"],
        "change_tested":             change_tested,
        "run_id":                    run_id,
        "run_category":              run_category,
        "start_date":                START_DATE,
        "runtime_hours":             RUNTIME_HOURS,
        "rain_interval_hours":       RAIN_INTERVAL,
        "event_start":               EVENT_START,
        "event_end":                 EVENT_END,
        "Ks_mult":                   Ks_mult,
        "f_RS_abs":                  f_RS_abs,
        "As_value":                  As_value,
        "Au_value":                  Au_value,
        "optpercolation":            optpercolation,
        "channelconductivity_mmhr":  channelconductivity_mmhr,
        "channelporosity":           channelporosity,
        "kinemvelcoef":              kinemvelcoef,
        "flowexp":                   flowexp,
        "channelroughness":          channelroughness,
        "channelwidthcoeff":         channelwidthcoeff,
        "input_file":                input_file,
        "log_file":                  log_file,
        "output_prefix":             output_prefix,
        "csv_export_dir":            os.path.relpath(csv_export_dir,      notebook_dir),
        "plot_export_dir":           os.path.relpath(plot_export_dir,     notebook_dir),
        "summary_export_dir":        os.path.relpath(summary_export_dir,  notebook_dir),
        "swept_param":               param_name,
        "swept_value":               value,
    }

    config_path = calib_dir / "current_run_config.json"
    config_path.write_text(json.dumps(current_run_config, indent=2))

    print(f"\n  Built: {run_id}")
    print(f"    Input file: {input_file}")
    print(f"    {param_name} = {value}  (baseline = {BASELINE[param_name]})")
    print(f"    Ks_mult = {Ks_mult}")
    return run_id, input_file, log_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build one tRIBS sensitivity run input file.")
    parser.add_argument("--param", required=True,
                        choices=list(PARAM_CONFIG.keys()),
                        help="Parameter to sweep")
    parser.add_argument("--value", required=True, type=float,
                        help="Value to assign to that parameter")
    args = parser.parse_args()

    build_input_file(args.param, args.value)