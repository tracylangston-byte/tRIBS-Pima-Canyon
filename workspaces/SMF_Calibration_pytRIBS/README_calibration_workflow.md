# SMF tRIBS Calibration Workflow README

## Purpose of this folder

This folder documents a reproducible calibration workflow for running the tRIBS hydrologic model through pytRIBS for the South Mountain Fan watershed. The immediate goal is to calibrate and compare storm-event simulations for the August 12, 2014 event. The broader goal is to develop a workflow that can later be transferred to other flashy semi-arid/arid watersheds.

This README is intentionally detailed. It is written for a future user who may know nothing about the current notebook structure, the calibration files, or the reasoning behind the workflow.

---

## Big-picture concept

The workflow has three main notebooks:

1. **Make notebook**: builds or updates the tRIBS model input files.
2. **Run notebook**: runs the model and calculates single-run metrics.
3. **Compare notebook**: compares a current run to a baseline run and optionally appends the result to a master calibration log.

The intended workflow is:

```text
Edit calibration controls in Make
        ↓
Run Make to write input files + current_run_config.json
        ↓
Run Run_Model to execute tRIBS and save single-run metrics
        ↓
Run Compare_Calibration_Runs_improved to compare current run to baseline
        ↓
Optionally append one curated row to the master calibration log
        ↓
Commit meaningful notebook/code changes to GitHub
```

---

## Important project context

This work is part of a tRIBS calibration/sensitivity exercise. The current model is for the **South Mountain Fan** watershed and the main calibration event is the **August 12, 2014 storm**.

The main calibration targets are:

* peak discharge
* runoff volume
* hydrograph timing
* overall hydrograph shape
* goodness-of-fit metrics such as RMSE, NSE, KGE, percent bias, and timing error

The broader research question is how well tRIBS can represent flashy semi-arid watershed response, especially storm runoff, infiltration, channel transmission losses, and hydrograph timing.

---

## Repository orientation

In Codespaces, the terminal may open at the repository root:

```text
/workspaces/tRIBS-Pima-Canyon
```

The active SMF calibration project is inside:

```text
/workspaces/tRIBS-Pima-Canyon/workspaces/SMF_Calibration_pytRIBS
```

A common source of confusion is running terminal commands from the wrong folder. If a command says a folder does not exist, first check where the terminal is:

```bash
pwd
```

To move into the active SMF calibration project:

```bash
cd /workspaces/tRIBS-Pima-Canyon/workspaces/SMF_Calibration_pytRIBS
```

---

## Major folder structure

The active project folder contains several important folders. The most important one for calibration work is `calibration_work`.

```text
SMF_Calibration_pytRIBS/
├── calibration_work/
│   ├── _archive/
│   ├── 01_run_inputs/
│   ├── 02_results/
│   ├── 03_comparisons/
│   │   ├── csv_exports/
│   │   ├── hydrograph_plots/
│   │   └── summary_tables/
│   ├── 04_notebooks/
│   │   ├── build_model_versions/
│   │   ├── compare_notebooks/
│   │   │   └── Compare_Calibration_Runs_improved.ipynb
│   │   └── run_model_versions/
│   ├── 05_calibration_log/
│   ├── 06_logs/
│   └── current_run_config.json
├── smf_assets/
├── smf_demo/
└── smf_init_data/
```

### Folder meanings

| Folder or file                                      | Purpose                                                                                                          |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `calibration_work/01_run_inputs/`                   | Stores copies of `.in`, `.sdt`, and other model input files associated with specific runs.                       |
| `calibration_work/02_results/`                      | Stores tRIBS model output files for each run.                                                                    |
| `calibration_work/03_comparisons/csv_exports/`      | Stores exported observed/simulated hydrograph comparison CSVs.                                                   |
| `calibration_work/03_comparisons/hydrograph_plots/` | Stores comparison plots.                                                                                         |
| `calibration_work/03_comparisons/summary_tables/`   | Stores run metrics summaries, baseline-vs-current summaries, and the master calibration log.                     |
| `calibration_work/04_notebooks/`                    | Stores notebook versions used for building, running, and comparing calibration runs.                             |
| `calibration_work/06_logs/`                         | Stores terminal/model run logs.                                                                                  |
| `calibration_work/current_run_config.json`          | Stores the current run ID and calibration parameter values written by the Make notebook and read by Run/Compare. |
| `smf_init_data/`                                    | Stores initial model data such as meteorology, rasters, and observed discharge data.                             |
| `smf_demo/`                                         | Original or demo model material. Treat this as a reference/demo area unless intentionally modifying it.          |

---

## Notebook roles

## 1. `Make_SMF_Model.ipynb`

This notebook builds the model setup and writes the input files that tRIBS needs to run.

It is the place where calibration parameters should be changed.

### Main jobs of the Make notebook

* Define the run ID and calibration-control values.
* Set up model paths.
* Read/copy/link the DEM, soil raster, land-cover raster, bedrock depth, groundwater, and meteorological inputs.
* Define soil parameters and write/update the soil data table.
* Define land-use parameters.
* Define channel transmission loss parameters.
* Define routing parameters.
* Set input file keywords such as start date, runtime, rainfall interval, output paths, and output filenames.
* Write the final `.in` file used by tRIBS.
* Write `current_run_config.json`, which records the intended run settings and calibration values.

### The most important cell in Make

The top calibration control cell is the main place to make run-specific changes.

Do not hunt through the whole notebook changing parameters manually unless there is a specific reason. The goal is for most calibration work to happen in the top control cell.

---

## 2. `Run_Model.ipynb`

This notebook runs tRIBS and calculates single-run metrics.

### Main jobs of the Run notebook

* Read `current_run_config.json`.
* Identify the `.in` file for the current run.
* Execute tRIBS.
* Save the model log.
* Read the model output, especially the outlet `.qout` file.
* Read observed discharge data.
* Subset to the event window.
* Calculate hydrograph/event metrics.
* Save a single-run metrics summary CSV.
* Save event comparison CSVs for later comparison.

### Important output from Run

The Run notebook creates a single-run file like:

```text
calibration_work/03_comparisons/summary_tables/<run_id>_metrics_summary.csv
```

Example:

```text
calibration_work/03_comparisons/summary_tables/SMF_20140812_01_baseline_450hr_metrics_summary.csv
```

This file should include:

* run ID
* event window
* observed/simulated peak flow
* observed/simulated peak timing
* timing error
* observed/simulated volume
* volume error / percent bias
* RMSE
* NSE
* KGE
* calibration parameter values from `current_run_config.json`

This is the file the Compare notebook uses to pull parameter values into the master calibration log.

---

## 3. `Compare_Calibration_Runs_improved.ipynb`

This notebook compares a current run to a baseline run and optionally appends the result to the master calibration log.

### Main jobs of the Compare notebook

* Load the baseline run comparison CSV.
* Load the current run comparison CSV.
* Calculate or display metrics for each run.
* Calculate differences between baseline and current run.
* Save a baseline-vs-current comparison summary.
* Optionally append one curated row to the master calibration log.

### Important output from Compare

The Compare notebook creates comparison files with `_vs_` in the filename, such as:

```text
calibration_work/03_comparisons/summary_tables/SMF_20140812_00_baseline_clean_vs_SMF_20140812_01_baseline_450hr_metrics_summary.csv
```

This is different from the single-run metrics summary file.

| File type                           | Example                                                                                | Created by                                      | Meaning                                         |
| ----------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| Single-run metrics summary          | `SMF_20140812_01_baseline_450hr_metrics_summary.csv`                                   | Run notebook                                    | Metrics and parameter values for one run.       |
| Baseline-vs-current metrics summary | `SMF_20140812_00_baseline_clean_vs_SMF_20140812_01_baseline_450hr_metrics_summary.csv` | Compare notebook                                | Baseline, current, and delta metrics.           |
| Master calibration log              | `SMF_20140812_calibration_log.csv`                                                     | Compare notebook, only when append is turned on | Curated record of important runs and decisions. |

---

## Run naming convention

Run IDs should follow this pattern:

```text
location_YYYYMMDD_2digitrun#_changetested
```

Example:

```text
SMF_20140812_00_baseline_clean
SMF_20140812_01_baseline_450hr
SMF_20140812_10_Ks_half
SMF_20140812_20_channelK_200mmhr
SMF_20140812_30_routing_cv_high
```

Recommended run number groups:

| Run numbers | Category                                                                  |
| ----------- | ------------------------------------------------------------------------- |
| `00–09`     | baseline/setup tests                                                      |
| `10–19`     | soil hydraulic tests, such as `Ks_mult`, `f_mult`, `As_value`, `Au_value` |
| `20–29`     | channel transmission loss tests                                           |
| `30–39`     | routing/timing tests                                                      |

The goal is to make the run ID informative enough that someone can tell what was changed without opening every notebook.

---

## Current calibration controls

The calibration workflow is designed so these parameters can be controlled from the top calibration cell in `Make_SMF_Model.ipynb`.

### Soil hydraulic controls

```python
Ks_mult = 1.0
f_mult = 1.0
As_value = 1.0
Au_value = 1.0
```

| Parameter  | Meaning                                                                  | Notes                                                                                                                                                       |
| ---------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ks_mult`  | Multiplier on saturated hydraulic conductivity values in the soil table. | Main soil infiltration/runoff-volume control. Values greater than 1 increase saturated conductivity; values less than 1 decrease it.                        |
| `f_mult`   | Multiplier on the hydraulic decay parameter `f`.                         | Controls how quickly saturated hydraulic conductivity decreases with depth. Strongly interacts with `Ks_mult`.                                              |
| `As_value` | Saturated anisotropy ratio.                                              | Advanced soil control. Represents directional difference in saturated hydraulic conductivity. Keep fixed unless intentionally testing anisotropy.           |
| `Au_value` | Unsaturated anisotropy ratio.                                            | Advanced soil control. Represents directional difference in unsaturated/vadose-zone hydraulic behavior. Keep fixed unless intentionally testing anisotropy. |

### Channel transmission loss controls

```python
optpercolation = 0
channelconductivity_mmhr = 70
channelporosity = 0.4
```

| Parameter                  | Meaning                                                       | Notes                                                                                             |
| -------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `optpercolation`           | Turns channel transmission loss/percolation option on or off. | Usually `0` means off and `1` means on. Confirm in notebook/model settings before major tests.    |
| `channelconductivity_mmhr` | User-friendly channel conductivity in mm/hr.                  | Converted in the notebook to model units. Main channel-loss control.                              |
| `channelporosity`          | Channel porosity.                                             | Secondary channel-loss/storage-related parameter. Likely less dominant than channel conductivity. |

### Routing/timing controls

```python
kinemvelcoef = 3
flowexp = 0.3
channelroughness = 0.04
channelwidthcoeff = 2.33
```

| Parameter           | Meaning                                                       | Notes                                                                                                   |
| ------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `kinemvelcoef`      | Hillslope velocity coefficient, often associated with `c_v`.  | Controls speed of hillslope/overland routing. Higher values generally make hydrographs faster/narrower. |
| `flowexp`           | Flow exponent, often associated with `r`.                     | Controls nonlinearity of hillslope routing. Interacts with `kinemvelcoef`.                              |
| `channelroughness`  | Manning-style channel roughness, often associated with `n_e`. | Higher roughness generally slows/damps channel routing.                                                 |
| `channelwidthcoeff` | Channel width-area coefficient.                               | Controls channel geometry/routing. Leave fixed unless intentionally testing channel geometry.           |

---

## How `Ks_mult` and `f_mult` work

`Ks_mult` and `f_mult` are multipliers, not absolute replacements.

The soil assignment logic follows the idea:

```python
cls['Ks'] = params['Ks'] * Ks_mult
cls['f'] = params['f'] * f_mult
```

This means that each soil class keeps its relative baseline value, but all soil classes are scaled together.

Example:

```python
Ks_mult = 0.5
```

means every baseline soil `Ks` value is cut in half.

```python
Ks_mult = 2.0
```

means every baseline soil `Ks` value is doubled.

This is useful for sensitivity analysis because it preserves differences among soil classes while testing the overall effect of higher or lower conductivity.

---

## How `As_value` and `Au_value` work

The current workflow exposes two anisotropy fields:

```python
As_value = 1.0
Au_value = 1.0
```

These are written directly into the soil table fields:

```python
cls['As'] = As_value
cls['Au'] = Au_value
```

They are not currently implemented as multipliers. They are direct values.

A value of `1.0` means no anisotropy: water is treated as moving equally well in the relevant directions.

Values greater than `1.0` represent stronger lateral/downslope tendency relative to vertical/normal movement. These parameters can affect lateral subsurface exchange, interflow, baseflow, and convergence toward channels.

For the first round of sensitivity analysis, these should probably be kept fixed unless specifically testing anisotropy.

---

## How channel conductivity units work

The top cell uses:

```python
channelconductivity_mmhr = 70
```

This is intentionally user-friendly because mm/hr is easier to think about hydrologically.

The notebook converts it to model units, likely using:

```python
channelconductivity_mps = channelconductivity_mmhr / 3.6e6
```

Reason:

```text
1 mm/hr = 0.001 m / 3600 s = 1 / 3,600,000 m/s
```

So do not manually enter m/s in the top control cell unless the code is changed.

---

## Current run configuration file

The file:

```text
calibration_work/current_run_config.json
```

is written by the Make notebook and read by the Run notebook.

It records values such as:

```json
{
  "run_id": "SMF_20140812_01_baseline_450hr",
  "event_start": "2014-08-12 16:00",
  "event_end": "2014-08-13 12:00",
  "Ks_mult": 1.0,
  "f_mult": 1.0,
  "As_value": 1.0,
  "Au_value": 1.0,
  "optpercolation": 0,
  "channelconductivity_mmhr": 70,
  "channelporosity": 0.4,
  "kinemvelcoef": 3,
  "flowexp": 0.3,
  "channelroughness": 0.04,
  "channelwidthcoeff": 2.33
}
```

The exact contents may differ, but the purpose is the same: it creates a traceable record of the intended run setup.

---

## Calibration log system

The calibration log is a curated master table of important runs.

The file is:

```text
calibration_work/03_comparisons/summary_tables/SMF_20140812_calibration_log.csv
```

It is not automatically updated every time the model runs. This is intentional.

The Compare notebook has a cell with:

```python
append_to_calibration_log = False
```

When this is `False`, the cell only previews the existing log.

When this is changed to `True`, the cell appends one new row to the master log.

### Usual safe workflow

```python
append_to_calibration_log = False
```

Keep it false while checking outputs.

When ready to officially log a run:

1. Update the manual metadata fields.
2. Change `append_to_calibration_log` to `True`.
3. Run the cell once.
4. Change it back to `False`.
5. Save the notebook.

### Why this matters

If the cell is rerun multiple times with:

```python
append_to_calibration_log = True
```

it will append duplicate rows.

---

## What gets stored in the master calibration log

Each row should include:

* run ID
* baseline run ID
* event start/end
* manual interpretation fields:

  * change category
  * knob changed
  * knob value
  * notes
  * decision
* calibration parameter values:

  * `Ks_mult`
  * `f_mult`
  * `As_value`
  * `Au_value`
  * `optpercolation`
  * `channelconductivity_mmhr`
  * `channelporosity`
  * `kinemvelcoef`
  * `flowexp`
  * `channelroughness`
  * `channelwidthcoeff`
* performance metrics:

  * observed peak flow
  * simulated peak flow
  * peak timing error
  * observed volume
  * simulated volume
  * volume error / percent bias
  * RMSE
  * NSE
  * KGE

The updated log cell pulls parameter values from the single-run metrics summary created by the Run notebook.

---

## Important distinction: single-run summary vs comparison summary

This is a common point of confusion.

### Single-run summary

Created by Run notebook:

```text
<run_id>_metrics_summary.csv
```

Example:

```text
SMF_20140812_01_baseline_450hr_metrics_summary.csv
```

This contains metrics and parameter values for one run.

### Baseline-vs-current summary

Created by Compare notebook:

```text
<baseline_run_id>_vs_<current_run_id>_metrics_summary.csv
```

Example:

```text
SMF_20140812_00_baseline_clean_vs_SMF_20140812_01_baseline_450hr_metrics_summary.csv
```

This contains baseline/current/delta comparison results.

### Master calibration log

Created/updated by Compare notebook only when append is turned on:

```text
SMF_20140812_calibration_log.csv
```

This is the curated record of runs worth remembering.

---

## Basic run workflow

## Step 1 — Start from the correct folder

In the terminal:

```bash
cd /workspaces/tRIBS-Pima-Canyon/workspaces/SMF_Calibration_pytRIBS
```

Check:

```bash
pwd
```

---

## Step 2 — Edit the top calibration control cell in Make

Open `Make_SMF_Model.ipynb`.

Change only the needed controls, such as:

```python
run_id = "SMF_20140812_10_Ks_half"
Ks_mult = 0.5
f_mult = 1.0
As_value = 1.0
Au_value = 1.0
```

For a structural/control test where nothing should change hydrologically:

```python
Ks_mult = 1.0
f_mult = 1.0
As_value = 1.0
Au_value = 1.0
```

---

## Step 3 — Run Make

Run the necessary Make notebook cells to:

* apply soil parameters
* apply channel loss parameters
* apply routing parameters
* write the input file
* write `current_run_config.json`

Check that the notebook prints expected values for the controls.

---

## Step 4 — Run the model

Open `Run_Model.ipynb`.

Run the notebook or the needed cells.

Confirm that it saves a metrics summary like:

```text
calibration_work/03_comparisons/summary_tables/<run_id>_metrics_summary.csv
```

---

## Step 5 — Compare to baseline

Open:

```text
calibration_work/04_notebooks/compare_notebooks/Compare_Calibration_Runs_improved.ipynb
```

Set:

```python
baseline_run_id = "SMF_20140812_00_baseline_clean"
current_run_id = "your_current_run_id"
```

Run the comparison cells.

Inspect:

* hydrograph plot
* metrics table
* peak timing error
* volume error / percent bias
* RMSE
* NSE
* KGE

---

## Step 6 — Append to calibration log only if useful

Only log runs that are meaningful or that document an important setup decision.

In the log cell, update:

```python
change_category = "soil Ks"
knob_changed = "Ks_mult"
knob_value = "0.5"
notes = "Reduced Ks to test whether lower infiltration increases runoff volume and peak."
decision = "keep testing / compare with high Ks"
```

Then set:

```python
append_to_calibration_log = True
```

Run the cell once.

Immediately set back to:

```python
append_to_calibration_log = False
```

---

## Step 7 — Check Git status

Before committing:

```bash
git status
```

Look carefully at what changed.

Expected notebook/code changes may include:

```text
modified: Make_SMF_Model.ipynb
modified: Run_Model.ipynb
modified: Compare_Calibration_Runs_improved.ipynb
modified: calibration_work/current_run_config.json
```

Generated outputs may also appear depending on what is tracked.

---

## What to delete and what not to delete

## Safe to delete

Only delete clearly accidental empty folders, such as:

```text
calibration_work/04_notebooks/03_comparisons/
```

This folder was likely created accidentally by a relative-path issue if a notebook was run from inside `04_notebooks`.

Before deleting, check whether it contains files:

```bash
find calibration_work/04_notebooks/03_comparisons -type f -print
```

If nothing prints:

```bash
rm -r calibration_work/04_notebooks/03_comparisons
```

## Do not delete without a specific reason

```text
calibration_work/03_comparisons/summary_tables/
calibration_work/03_comparisons/csv_exports/
calibration_work/02_results/
calibration_work/01_run_inputs/
calibration_work/current_run_config.json
calibration_work/06_logs/
```

These folders contain evidence of model runs, comparison outputs, input files, and logs.

---

## Common troubleshooting

## Problem: terminal says folder does not exist

Likely cause: terminal is in the repo root, not the SMF project folder.

Check:

```bash
pwd
```

Move to the project folder:

```bash
cd /workspaces/tRIBS-Pima-Canyon/workspaces/SMF_Calibration_pytRIBS
```

---

## Problem: cannot find `metrics_summary.csv`

Search for it:

```bash
find calibration_work -name "*metrics_summary.csv" -print
```

If nothing appears, the Run notebook probably has not successfully saved the metrics summary yet.

If only `_vs_` files appear, those are Compare outputs. Look for the single-run summary without `_vs_`.

---

## Problem: calibration log has duplicate rows

Likely cause: the log append cell was run more than once with:

```python
append_to_calibration_log = True
```

Fix manually by opening the CSV or with Python/pandas, then remember to set it back to `False` after appending.

---

## Problem: an accidental `03_comparisons` appears inside `04_notebooks`

This probably happened because a relative output path was evaluated from the wrong working directory.

The correct comparison folder is:

```text
calibration_work/03_comparisons/
```

The accidental one is:

```text
calibration_work/04_notebooks/03_comparisons/
```

Delete the accidental one only if empty.

---

## Problem: parameter values are blank in calibration log

The Compare notebook pulls parameter values from:

```text
calibration_work/03_comparisons/summary_tables/<current_run_id>_metrics_summary.csv
```

If that file does not exist, or if it was created before the parameter columns were added, those values may be blank.

Solution:

1. Rerun the metrics-saving cell in `Run_Model.ipynb`.
2. Confirm the single-run summary includes the parameter columns.
3. Rerun the Compare log append cell only once.

---

## Calibration interpretation guide

## Soil parameters

### `Ks_mult`

Main question:

```text
Does changing saturated hydraulic conductivity improve runoff volume and peak flow?
```

Expected general behavior:

* Lower `Ks_mult` usually means less infiltration capacity and more surface runoff.
* Higher `Ks_mult` usually means more infiltration and potentially less immediate runoff.
* Effects may interact with soil moisture, bedrock depth, `f`, and anisotropy.

### `f_mult`

Main question:

```text
How does the vertical decay of hydraulic conductivity affect peak, volume, and recession behavior?
```

Expected general behavior:

* Higher `f` means conductivity decreases faster with depth.
* This can produce faster/sharper hydrograph response and more infiltration-excess behavior.
* Lower `f` means deeper soils remain more conductive, which can affect infiltration, subsurface flow, and recession.

### `As_value` and `Au_value`

Main question:

```text
How does directional conductivity affect lateral subsurface movement and convergence toward channels?
```

Expected general behavior:

* `1.0` means no anisotropy.
* Larger values can increase lateral/downslope subsurface movement.
* These parameters are advanced and interact with `Ks` and `f`.

Recommended first-round decision:

```text
Keep As_value = 1.0 and Au_value = 1.0 unless specifically testing anisotropy.
```

---

## Channel transmission loss parameters

### `optpercolation`

Main question:

```text
Are channel losses being represented?
```

Expected behavior:

* Off: flow is routed without the channel transmission loss option.
* On: some water can be lost from the channel, depending on conductivity/porosity settings.

### `channelconductivity_mmhr`

Main question:

```text
How much does channel-bed conductivity reduce outlet flow volume and peak?
```

Expected behavior:

* Higher channel conductivity should increase channel losses when percolation is on.
* This should generally reduce simulated outlet volume and possibly peak flow.

### `channelporosity`

Main question:

```text
How sensitive are channel losses to channel storage/porosity assumptions?
```

Expected behavior:

* Likely less dominant than channel conductivity.
* Keep fixed unless doing a focused channel-loss test.

---

## Routing parameters

### `kinemvelcoef`

Main question:

```text
Does changing hillslope routing speed improve peak timing and hydrograph width?
```

Expected behavior:

* Higher values generally route water faster.
* This can make peaks earlier, higher, and narrower.

### `flowexp`

Main question:

```text
Does changing the nonlinear flow exponent improve hydrograph timing/shape?
```

Expected behavior:

* Interacts strongly with `kinemvelcoef`.
* It may be harder to interpret alone.

### `channelroughness`

Main question:

```text
Does channel roughness improve timing and damping of the hydrograph?
```

Expected behavior:

* Higher roughness generally slows channel flow and may dampen the hydrograph.

### `channelwidthcoeff`

Main question:

```text
Does channel geometry affect routing enough to matter for the outlet hydrograph?
```

Expected behavior:

* Alters channel geometry assumptions.
* Keep fixed unless intentionally testing channel routing geometry.

---

## Recommended first-round sensitivity choices

For a limited first single-parameter sensitivity exercise, avoid testing everything.

Reasonable candidates:

1. `Ks_mult`
2. `f_mult`
3. `channelconductivity_mmhr` with `optpercolation` on
4. one routing parameter, probably `kinemvelcoef` or `channelroughness`

Keep advanced/interdependent parameters fixed unless needed:

```python
As_value = 1.0
Au_value = 1.0
channelporosity = 0.4
channelwidthcoeff = 2.33
```

---

## Suggested baseline/control test after structural changes

After adding or changing the calibration-control workflow, run one boring setup test where the hydrologic parameters are unchanged.

Example:

```python
run_id = "SMF_20140812_02_control_cell_check"
Ks_mult = 1.0
f_mult = 1.0
As_value = 1.0
Au_value = 1.0
optpercolation = 0
channelconductivity_mmhr = 70
channelporosity = 0.4
kinemvelcoef = 3
flowexp = 0.3
channelroughness = 0.04
channelwidthcoeff = 2.33
```

Purpose:

```text
Confirm that the refactored control cell and anisotropy additions do not accidentally change the baseline behavior.
```

---

## Git workflow reminders

Codespaces does not permanently save work to GitHub unless changes are committed and pushed.

Basic workflow:

```bash
git status
git add .
git commit -m "Add calibration workflow controls and logging"
git push
```

Before `git add .`, inspect whether any huge/unwanted files are included.

If there are generated outputs that should not be committed, add them to `.gitignore` or restore them before committing.

---

## Suggested commit message for current update

For the current work adding anisotropy controls and improving logging:

```bash
git commit -m "Add calibration controls and parameter logging"
```

Or, more specific:

```bash
git commit -m "Add soil anisotropy controls to SMF calibration workflow"
```

---

## Source documents and conceptual references

This workflow was informed by:

* ASU tRIBS Exercise 4 slides: basic pytRIBS/tRIBS workflow, Make and Run notebooks, and calibration targets.
* ASU tRIBS Exercise 5 slides: continuous simulation setup, input-file changes, and extended simulation workflow.
* Ivanov et al. (2004): tRIBS calibration methodology and parameter group importance.
* Hüner thesis (2025): recent tRIBS sensitivity/calibration example, including soil, routing, and anisotropy parameters.
* Schreiner-McGraw and Vivoni (2017): arid/semi-arid channel and percolation context.
* UofA/SRP draft proposal: broader project motivation, transferable workflow, and semi-arid watershed application.

---

## Quick memory jogger

If you forget everything else, remember this:

```text
Make = choose parameters and write input files
Run = run tRIBS and save single-run metrics
Compare = compare to baseline and optionally log the run
```

The most important files are:

```text
calibration_work/current_run_config.json
calibration_work/03_comparisons/summary_tables/<run_id>_metrics_summary.csv
calibration_work/03_comparisons/summary_tables/<baseline>_vs_<current>_metrics_summary.csv
calibration_work/03_comparisons/summary_tables/SMF_20140812_calibration_log.csv
```

The most important rule is:

```text
Change calibration values in the top control cell, not scattered throughout the notebook.
```

The second most important rule is:

```text
Only append to the calibration log once per run, and set append_to_calibration_log back to False afterward.
```
