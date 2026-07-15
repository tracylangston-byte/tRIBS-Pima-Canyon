# SMF Calibration — Scripts Reference

All scripts run from the `smf_demo/` directory. Listed in the order they'd
run in a typical pipeline; one-off/special-case scripts and general
utilities are called out separately at the end.

---

## 0. Foundational — imported by every run script below

**`build_sensitivity_run.py`**
Builds one tRIBS input file (`.in`) for a single parameter/value combination, using a shared `BASELINE` dict and per-soil-class parameter lookup table. Every LHS and OAT run script imports this and temporarily patches `BASELINE` to inject its own swept values.

**`run_sensitivity_single.py`**
Executes tRIBS on whatever `.in`/config the builder just wrote, then scores the resulting hydrograph against either the real gauge or a synthetic truth `.qout` (auto-detected), returning a metrics dict (KGE, NSE, PBIAS, timing metrics, etc.). Called in-process by every run script except the Series 99 family, which calls it as a killable subprocess instead.

---

## 1. Single-parameter sensitivity (OAT) — Series 59–69

**`run_sensitivity_sweep.py`**
Runs a one-at-a-time sweep for any single parameter (or all of them) across a hardcoded set of test values. This is the starting point for identifying viable ranges before any parameter goes into a multi-parameter LHS.

**`plot_sensitivity.py`**
Generates the 5 standard sensitivity figures for any parameter in `sensitivity_results_all.csv` — pass `--param <name>`. Works for any OAT parameter without modification.

---

## 2. Multi-parameter LHS vs. real gauge — Series 80–83

**`run_lhs_multiparam.py`**
Runs an N-parameter LHS sweep against the real SMF gauge record. Switch series (81/82/83) by editing `ACTIVE_SERIES`; each preset controls its own parameter set, ranges, and output CSV.

**`plot_lhs_multiparam.py`**
Generates the full diagnostic figure set (hydrograph envelopes, correlation bars, parallel coordinates, pairwise scatter, per-series deep-dive figures) for whichever series `run_lhs_multiparam.py` produced. Same `ACTIVE_SERIES` switch.

---

## 3. Synthetic inversion — 4-parameter, Series 91–92

**`run_lhs_synth_4param_series.py`**
Runs a 4-parameter (Ks, cv, r, n) LHS sweep scored against a synthetic truth hydrograph rather than the real gauge, to test parameter identifiability. Switch between series 91/92 via `ACTIVE_SERIES`.

**`plot_lhs_synth_4param_series.py`**
Standard 9-figure diagnostic set for the synthetic 4-param sweep, with true-value reference lines added throughout to assess recovery. Same `ACTIVE_SERIES` switch.

**`plot_lhs_synth_permetric.py`**
Deeper per-metric sensitivity pass on the Series 92 results — top-vs-bottom parameter distribution comparisons and a console summary of every parameter–metric correlation above threshold. Companion to, not a replacement for, the standard 9-figure set above.

---

## 4. Ks × f synthetic sweep — Series 97 / 97log

**`run_lhs_synth_Ks_f_97.py`**
Joint Ks_mult × f_RS_abs LHS sweep against synthetic truth, with cv/r/n pinned at their confirmed true values — maps the volume-bias (PBIAS) response surface densely rather than by manual bisection.

**`run_lhs_synth_Ks_f_97log.py`**
Same sweep as above, but with log-scale stratification on the f_RS_abs axis so the low-f region (where the PBIAS=0 valley sits) gets denser sample coverage. Writes to a separate CSV/folder from 97, so the two never collide.

**`plot_lhs_Ks_f_97.py`**
Contour plots (KGE, PBIAS, NSE, and KGE sub-components) over the Series 97 Ks×f surface, each drawn with both linear and log10 f-axis panels side by side, with colorbars clipped so the high-Ks failure band doesn't wash out the readable mid-Ks structure.

**`plot_lhs_Ks_f_97log.py`**
Same figure set as above, pointed at the log-stratified 97log results instead.

---

## 5. cv/r/n identifiability across anchors — Series 96 / 99

**`run_lhs_anchor_cvrn.py`**
Runs a cv/r/n LHS sweep at one or more fixed, volume-matched Ks/f "anchor" points (anchorA, anchorB), testing whether routing-parameter identifiability holds away from the true Ks/f pair.

**`probe_anchor_Ks6p25lo_99.py`**
Standalone diagnostic that pre-vets a single risky anchor candidate before it's trusted in the main Series 99 sweep. Runs each draw as a subprocess with a hard timeout so a hang can be killed and logged instead of stalling the script indefinitely.

**`run_lhs_nanchor_cvrn_99.py`**
The current N-anchor cv/r/n sweep: 8 active volume-matched Ks/f anchors spanning 4.25×–8.25×, using the same shared LHS sample set at every anchor for a paired comparison. Runs each draw as a killable subprocess with a wall-clock timeout, and screens out any run where tRIBS silently crashed but still exited 0.

**`plot_pearson_nanchor_99.py`**
3×3 grid of metric panels, one Pearson-r trend line per routing parameter (cv, r, n), plotted across anchor Ks position — the main output for the cv/r/n identifiability question. Also produces an anchorA-vs-anchorB delta table. Works opportunistically on however many anchor CSVs currently exist.

**`plot_pearson_comparison.py`**
General-purpose two-series Pearson r heatmap comparison (side-by-side + difference panel) for any two LHS result CSVs — currently configured for Series 96's anchorA vs. anchorB. Reusable for other series pairs by editing the `SERIES` config block.

---

## 6. Cross-series diagnostic tools

**`plot_recession_residual_pca.py`**
Investigates what drives scatter in the flowexp(r)-vs-recession-rate relationship across synthetic inversion series results: residual-vs-parameter panels, a Ks-confound check, within-r-bin partial regression, and a PCA on phase-specific metrics. Not tied to one series — points at whichever results CSV you configure.

---

## 7. One-off / special-case scripts

**`run_lhs_synth_4param_93.py`**
Series 93 — identical to Series 92 except the synthetic truth's channelroughness is set to 0.075 instead of 0.026, testing a heavier-roughness hydrograph shape. A deliberately different truth condition, not a lineage successor to 91/92; has no dedicated plotting script.

**`plot_doublepeak_r_hypothesis.py`**
One specific hypothesis test on the Series 92 ensemble: whether the spurious double-peaked hydrograph seen in some runs is produced specifically by low-flowexp(r) runs. Splits the ensemble into r terciles and reports double-peak frequency per tercile.

**`plot_truth_hydrograph_comparison.py`**
Plots the real SMF gauge against three different synthetic truth hydrographs (Series 93, 94, 95) on one figure for a visual sanity check of how each truth condition differs from observed.

**`plot_lhs_Ks_pair.py`**
Contour plots for a real-gauge Ks × f or Ks × cv two-parameter sweep (switch via `ACTIVE_PAIR`). Both pairings read from older, pre-consolidation CSVs (`lhs_results_Ks_f.csv`, `lhs_results_Ks_cv.csv`); the run-side scripts that produced those CSVs are no longer part of the active toolset, so this script is only useful against already-collected historical data, not for regenerating new results.

**`plot_best_run_diagnostic.py`**
Single combined 2×3 diagnostic figure (hydrograph, KGE components, residuals, volume comparison) for the best run in `lhs_results_5param_KsHi.csv`. That filename doesn't match any CSV the current run scripts produce — worth checking before relying on this one.

---

## 8. Utilities — not part of the calibration run pipeline

**`summarize_basin_classes.py`**
Reads the soil-type and land-use rasters and reports the percentage of the basin in each class. Standalone basin-characterization tool, useful for write-ups; doesn't touch tRIBS or any calibration output.

**`parameter_key.py`**
Not runnable on its own — shared metadata module (`PARAM_KEY`, `METRIC_KEY`) with display names, symbols, units, and short tags for every calibration parameter and metric. Every plotting script imports from here rather than defining its own labels.

---

## Notes on this inventory

- `run_lhs_multiparam.py` and `run_lhs_synth_4param_series.py` each replace what used to be several near-duplicate scripts; only their current, consolidated behavior is described above.
- Series 98 (an earlier N-anchor attempt) is not listed — it's fully replaced by Series 99 and no longer exists in the project.
- This list was compiled via project knowledge search rather than a directory listing, after finding that several files (`run_lhs_nanchor_cvrn_99.py`, `plot_lhs_Ks_f_97.py`, `plot_lhs_Ks_f_97log.py`, `plot_lhs_synth_permetric.py`, `plot_pearson_comparison.py`) don't show up through direct file-browsing tools even though they're indexed and searchable. If something you know exists isn't in this doc, it's worth flagging.
