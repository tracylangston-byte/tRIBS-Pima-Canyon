"""
plot_lhs_Ks_f_100.py
======================
Generates contour plots from the Series 100 synthetic Ks_mult x f_RS_abs LHS
sweep (lhs_results_synth_Ks_f_100_RESCORED.csv), scored against the NEW
noise-free synthetic truth hydrograph (cv/r/n pinned at confirmed truth
values: cv=4.5, r=0.24, n=0.026; Ks_mult=7.0x, f_RS_abs=0.012).

UPDATED (Handoff_KGE2012Transition_v2.md): the composite metric and
variability term now use the Kling et al. (2012) formulation throughout --
KGE_2012 (column `kge_2012`) replaces KGE_2009 (column `kge`) as the
plotted/ranked composite, and gamma (column `kge_gamma`, the CV-based
variability ratio) replaces alpha (column `kge_alpha`) as the plotted
variability component. r and beta are unchanged between formulations, so
those figures are untouched. The retired 2009 columns (`kge`, `kge_alpha`)
are still pulled into the top-20 audit table for traceability, but no
longer drive any figure. See the handoff doc for why: alpha and beta were
~99% collinear in this dataset (alpha was re-reporting volume bias, not
independent shape information); gamma decouples that (r~0.37).

UPDATED (2026-07-30, this version): folds in two things that were
previously separate, one-off analyses, per Tracy's preference for complete
rewrites over accumulating separate patch scripts once enough has changed:

  1. The ridge-width-vs-f measurement, formerly the standalone script
     `measure_ridge_width_vs_f.py`, now FIGURE 9 + `ridge_width_vs_f.csv`.
     That script is fully superseded by this one and can be deleted/
     archived -- nothing here depends on it still existing.
  2. FIGURE 10 (new): top row is a KGE'-equivalent composite computed from
     gamma and beta ONLY, with r left out entirely (`1 - sqrt((gamma-1)^2 +
     (beta-1)^2)`) -- since kge_r sits close to 1.0 almost everywhere in
     this sweep (see fig3), this tests directly whether r is contributing
     anything to the ridge's shape, rather than just assuming it from the
     flat r contour. Bottom row is a literal overlap map of where beta and
     gamma are EACH individually near-ideal (|value-1| <= 0.02, the same
     tolerance already used elsewhere in this project for |PBIAS| <= 2%),
     shaded separately, with their intersection shaded a third way -- the
     direct visual for why the composite optimum sits where it does: at
     the overlap of "volume is right" (broad, follows the diagonal ridge)
     and "shape is right" (narrower, does NOT follow the diagonal ridge --
     see fig4's closed-island shape and the Series100 phase-conclusions
     doc, Section 2.2a, for why that's expected rather than a bug).

  A NOTE ON A DELIBERATE DEPARTURE FROM THIS SCRIPT'S ORIGINAL DESIGN:
  the paragraph below (previously) said the script was "fully data-driven
  ... so no truth values are hardcoded in the plotting logic itself." That
  is still true for figures 1-8. It is NOT true for figures 9 and 10:
  both need TRUE_KS/TRUE_F (defined in CONFIG below) to mark the true
  parameter point and to center the f-slice cross-sections on the actual
  true f. This is a deliberate scope difference, not an oversight --
  figures 9 and 10 only make sense in a synthetic-truth-inversion context
  where the true answer is known and the question is whether the sweep
  recovers it, whereas figures 1-8 are written to remain meaningful even
  if this script were ever pointed at a real (non-synthetic, truth-unknown)
  sweep. If that ever happens, figures 9 and 10 should be skipped or
  reworked, not silently fed a placeholder true value.

This is a direct adaptation of plot_lhs_Ks_f_97log.py -- same methodology,
same normalized-coordinate interpolation fix, same original 8-figure output
set, now extended with figures 9-10 above. The script is otherwise fully
data-driven for figures 1-8: the best-run, axis ranges, and interpolation
grids are all computed from whatever's in the CSV.

KGE_CLIP/NSE_CLIP/PBIAS_CLIP colorbar ranges are CARRIED OVER UNCHANGED
from Series 97log, where they were tuned to that series' old-truth Ks>=9x
anomaly (KGE bottoming near -0.4, PBIAS reaching -72%/+93%). The new
truth's surface may not have the same extremes in the same place -- check
the printed kge_vals.min()/max() and pbias_vals.min()/max() after the
first run and tighten/loosen these three constants if the panels look
washed out or over-clipped. (KGE_2012 tracks KGE_2009 closely -- r=0.995
on this dataset -- so the carried-over clip range is still a reasonable
starting point.)

Usage (run from the smf_demo directory):
    python plot_lhs_Ks_f_100.py

Produces 11 image files + 2 CSVs saved to:
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_f_100/

Figure list:
    fig1_kge_contour.png          -- KGE_2012, linear|log side by side, clipped colorbar
    fig2_pbias_contour.png        -- PBIAS, linear|log side by side, clipped colorbar
    fig3_r_contour.png            -- r (correlation), linear|log side by side
    fig4_gamma_contour.png        -- gamma (CV-based variability), linear|log side by side
    fig5_beta_contour.png         -- beta (volume bias), linear|log side by side
    fig6_nse_contour.png          -- NSE, linear|log side by side, clipped colorbar
    fig7_all_metrics_panel_linear.png  -- all six metrics, 2x3, linear f axis
    fig7_all_metrics_panel_log.png     -- all six metrics, 2x3, log10 f axis
    fig8_raw_scatter_diagnostic.png    -- raw (no-interpolation) KGE_2012/PBIAS vs Ks, binned by f
    fig9_ridge_width_vs_f.png          -- (A) KGE' vs Ks cross-sections at fixed
                                           f-slices incl. true f, shaded >=0.8 band;
                                           (B) that band's width (Ks units) vs f,
                                           dense curve, true f marked
    fig10_beta_gamma_intersection.png  -- (top) KGE' with r excluded, from gamma+beta
                                           only, linear|log; (bottom) literal overlap
                                           of beta-feasible and gamma-feasible regions,
                                           linear|log
    top20_pbias_synth_100.csv     -- 20 runs with smallest |PBIAS| (includes retired kge/kge_alpha for audit)
    ridge_width_vs_f.csv          -- checkpoint table at 7 reference f-slices (see fig9)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.interpolate import griddata

# ======================================================================
# CONFIG
# ======================================================================
RESULTS_CSV   = "lhs_results_synth_Ks_f_100_RESCORED.csv"
OUTPUT_SUBDIR = "lhs_Ks_f_100"
SECOND_COL    = "f_RS_abs"
SECOND_LABEL  = "Hydraulic conductivity decay f (RS soil, mm\u207b\u00b9)"
EVENT_LABEL   = "SMF Aug 12, 2014 (synthetic truth, Series 100 -- Ks=7.0x/f=0.012)"
PAIR_LABEL    = "Ks x f"

# Clipped colorbar ranges -- carried over unchanged from Series 97log (see
# module docstring). Confirm against this run's printed KGE/PBIAS ranges
# before trusting the colorbar isn't washing out or over-clipping the
# mid-Ks valley structure.
KGE_CLIP   = (-0.3, 1.0)     # extend='min'
NSE_CLIP   = (-0.3, 1.0)     # extend='min'
PBIAS_CLIP = (-50.0, 50.0)   # extend='both'

TOP_N_TABLE = 20

# -- True parameters: ONLY used by figures 9-10 (see the docstring's note
# on why this departs from the rest of the script being data-driven).
TRUE_KS = 7.0
TRUE_F  = 0.012

# -- Figure 9 (ridge width vs f) config -- carried over from the retired
# measure_ridge_width_vs_f.py unchanged.
RIDGE_KGE_THRESHOLD = 0.8   # matches the threshold the v6 handoff originally used
N_GRID_KS_FINE       = 400   # finer than the N_GRID=200 used for the 2D contour
                              # grids below -- fig9 only needs 1D cross-sections,
                              # not a full 2D contourf render, so the extra
                              # resolution is cheap
N_F_CURVE            = 60    # dense log-spaced f values for fig9 panel B's curve
REFERENCE_F_SLICES = [0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030]  # checkpoint table
CROSS_SECTION_SLICES = [0.006, 0.009, TRUE_F, 0.015, 0.020, 0.030]     # fig9 panel A

# -- Figure 10 (beta/gamma intersection) config.
# Same +/-0.02 tolerance applied to BOTH beta and gamma, deliberately --
# it's the same magnitude as the |PBIAS| <= 2% convention already used
# elsewhere in this project (PBIAS_pct = 100*(beta-1), so |beta-1| <= 0.02
# IS |PBIAS| <= 2%), applied symmetrically to gamma so the two feasibility
# regions are being judged by the same yardstick rather than two
# differently-chosen cutoffs.
BETA_GAMMA_FEASIBLE_TOL = 0.02

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / OUTPUT_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(
        f"LHS results not found: {results_path}\n"
        "Run run_lhs_synth_Ks_f_100.py first."
    )

df = pd.read_csv(results_path)
print(f"Loaded {len(df)} LHS runs from {results_path.name}")

required_cols = ["Ks_mult", SECOND_COL, "kge_2012", "nse", "pbias_pct",
                  "kge_r", "kge_gamma", "kge_beta"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in results CSV: {missing_cols}")

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"  {len(df)} runs after dropping NaN rows")

# -----------------------------------------------------------------------
# TOP-20 BY |PBIAS| TABLE
# -- kge/kge_alpha (the retired 2009 columns) are included here purely for
#    audit-trail traceability if present; no figure below uses them.
# -----------------------------------------------------------------------
table_cols = [c for c in ["run_id", "Ks_mult", "f_RS_abs", "kge_2012", "pbias_pct",
                           "nse", "kge_gamma", "kge_beta", "kge", "kge_alpha"]
              if c in df.columns]
top20 = df.reindex(df["pbias_pct"].abs().sort_values().index)[table_cols].head(TOP_N_TABLE)
top20_path = plot_dir / "top20_pbias_synth_100.csv"
top20.to_csv(top20_path, index=False)
print(f"\nTop {TOP_N_TABLE} runs by |PBIAS| -> {top20_path.name}")
print(top20.to_string(index=False))

# -----------------------------------------------------------------------
# EXTRACT 1D SCATTERED DATA ARRAYS
# -----------------------------------------------------------------------
ks_pts     = df["Ks_mult"].values
second_pts = df[SECOND_COL].values
kge_vals   = df["kge_2012"].values   # primary composite is now KGE_2012
nse_vals   = df["nse"].values
pbias_vals = df["pbias_pct"].values
r_vals     = df["kge_r"].values
gamma_vals = df["kge_gamma"].values  # replaces alpha as the plotted variability term
beta_vals  = df["kge_beta"].values

# -- Figure 10 panel A: KGE' with r excluded, computed directly from the
# already-extracted gamma/beta arrays -- no new interpolation, pure numpy.
kge_no_r_vals = 1 - np.sqrt((gamma_vals - 1) ** 2 + (beta_vals - 1) ** 2)
df["kge_no_r"] = kge_no_r_vals

best_idx    = np.argmax(kge_vals)
best_ks     = ks_pts[best_idx]
best_second = second_pts[best_idx]
best_kge    = kge_vals[best_idx]

print(f"\n  Best run (by KGE_2012): Ks={best_ks:.2f}x  f={best_second:.4g}  "
      f"KGE_2012={best_kge:.3f}  PBIAS={pbias_vals[best_idx]:+.1f}%")
print(f"  KGE_2012 range: {kge_vals.min():.3f} to {kge_vals.max():.3f}")
print(f"  PBIAS range:    {pbias_vals.min():.1f}% to {pbias_vals.max():.1f}%")

# -----------------------------------------------------------------------
# BUILD TWO INTERPOLATION GRIDS -- one linear-spaced in f, one log-spaced
# -----------------------------------------------------------------------
N_GRID = 200

ks_grid_1d = np.linspace(ks_pts.min(), ks_pts.max(), N_GRID)

f_lin_1d = np.linspace(second_pts.min(), second_pts.max(), N_GRID)
f_log_1d = np.logspace(np.log10(second_pts.min()), np.log10(second_pts.max()), N_GRID)

KS_GRID_LIN, F_GRID_LIN = np.meshgrid(ks_grid_1d, f_lin_1d)
KS_GRID_LOG, F_GRID_LOG = np.meshgrid(ks_grid_1d, f_log_1d)

# ------------------------------------------------------------------
# NORMALIZED COORDINATES FOR TRIANGULATION
#
# griddata's cubic method builds a Delaunay triangulation directly on
# whatever coordinates it's given. Ks_mult spans ~3-11 while f_RS_abs
# spans ~0.003-0.05 -- a >100x difference in absolute scale. Euclidean-
# distance-based triangulation on such mismatched scales produces
# severely elongated, ill-conditioned triangles (essentially collapsing
# f-axis proximity), and cubic (Clough-Tocher) interpolation is known to
# oscillate badly on ill-conditioned triangulations -- confirmed as the
# cause of the dense vertical banding in the contour panels, since the
# same data sliced and plotted raw (fig8_raw_scatter_diagnostic.png) is
# smooth and single-peaked in every f-bin.
#
# Fix: triangulate in min-max-normalized (Ks, log10(f)) space instead,
# so both axes span comparable [0, 1] ranges before Delaunay runs.
# log10(f), not raw f, because the raw-scatter diagnostic confirmed the
# surface is smooth in log-f slices -- matching how the LHS sampling
# itself was stratified. Only the VALUES fed to griddata change here;
# the actual (x, y) grid positions used for plotting (KS_GRID_LIN/LOG,
# F_GRID_LIN/LOG) are untouched.
#
# NOTE: this same points_norm / normalize_ks / normalize_logf machinery
# is reused below by figures 9 and 10 -- it is defined exactly once, here,
# for the whole script.
# ------------------------------------------------------------------
def normalize_ks(ks):
    return (ks - ks_pts.min()) / (ks_pts.max() - ks_pts.min())


def normalize_logf(f):
    log_lo, log_hi = np.log10(second_pts.min()), np.log10(second_pts.max())
    return (np.log10(f) - log_lo) / (log_hi - log_lo)


points_norm = np.column_stack([normalize_ks(ks_pts), normalize_logf(second_pts)])

KS_GRID_LIN_NORM = normalize_ks(KS_GRID_LIN)
F_GRID_LIN_NORM  = normalize_logf(F_GRID_LIN)
KS_GRID_LOG_NORM = normalize_ks(KS_GRID_LOG)
F_GRID_LOG_NORM  = normalize_logf(F_GRID_LOG)


def interp_surface(values, grid_ks_norm, grid_f_norm):
    return griddata(points_norm, values, (grid_ks_norm, grid_f_norm), method='cubic')


SURFACES_LIN = {
    "kge":   interp_surface(kge_vals,   KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
    "nse":   interp_surface(nse_vals,   KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
    "pbias": interp_surface(pbias_vals, KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
    "r":     interp_surface(r_vals,     KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
    "gamma": interp_surface(gamma_vals, KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
    "beta":  interp_surface(beta_vals,  KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
}
SURFACES_LOG = {
    "kge":   interp_surface(kge_vals,   KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "nse":   interp_surface(nse_vals,   KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "pbias": interp_surface(pbias_vals, KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "r":     interp_surface(r_vals,     KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "gamma": interp_surface(gamma_vals, KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "beta":  interp_surface(beta_vals,  KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
}

# -- Figure 10 panel A surface: derived directly from the gamma/beta
# surfaces just built above -- pure numpy, no new griddata calls.
SURFACES_LIN["kge_no_r"] = 1 - np.sqrt((SURFACES_LIN["gamma"] - 1) ** 2 +
                                        (SURFACES_LIN["beta"] - 1) ** 2)
SURFACES_LOG["kge_no_r"] = 1 - np.sqrt((SURFACES_LOG["gamma"] - 1) ** 2 +
                                        (SURFACES_LOG["beta"] - 1) ** 2)

gamma_dev = max(abs(gamma_vals.min() - 1), abs(gamma_vals.max() - 1))
beta_dev  = max(abs(beta_vals.min() - 1), abs(beta_vals.max() - 1))

XLABEL = "Ks multiplier"
YLABEL = SECOND_LABEL

SCATTER_KW = dict(s=16, edgecolors='white', linewidths=0.4, alpha=0.7, zorder=5)
BEST_KW    = dict(s=110, marker='*', color='white', edgecolors='black',
                  linewidths=1.2, zorder=10)
TRUE_KW    = dict(s=140, marker='X', color='white', edgecolors='black',
                  linewidths=1.5, zorder=11)


def save_fig(fig, filename):
    path = plot_dir / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path.name}")
    plt.close(fig)


def draw_metric_panel(ax, key, cmap, norm, cbar_label, log_scale,
                       extend='neither', show_pbias_zero=True):
    """Draw one metric contour panel (linear or log f-axis) on ax."""
    surf = SURFACES_LOG[key] if log_scale else SURFACES_LIN[key]
    ks_g = KS_GRID_LOG if log_scale else KS_GRID_LIN
    f_g  = F_GRID_LOG if log_scale else F_GRID_LIN

    cf = ax.contourf(ks_g, f_g, surf, levels=20, cmap=cmap, norm=norm, extend=extend)

    try:
        cl = ax.contour(ks_g, f_g, surf, levels=8, colors='black',
                         linewidths=0.5, alpha=0.4)
        ax.clabel(cl, fmt="%.2f", fontsize=6, colors='black')
    except Exception:
        pass

    if show_pbias_zero:
        pbias_surf = SURFACES_LOG["pbias"] if log_scale else SURFACES_LIN["pbias"]
        try:
            cs = ax.contour(ks_g, f_g, pbias_surf, levels=[0], colors=['white'],
                             linewidths=2.0, linestyles='--', zorder=6)
        except Exception:
            pass

    ax.scatter(ks_pts, second_pts, c=df[COL_MAP[key]].values, cmap=cmap, norm=norm,
               **SCATTER_KW)
    ax.scatter([best_ks], [best_second], **BEST_KW)

    if log_scale:
        ax.set_yscale('log')
        ax.set_title("log\u2081\u2080 f axis", fontsize=10)
    else:
        ax.set_title("linear f axis", fontsize=10)

    ax.set_xlabel(XLABEL, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10)
    return cf


COL_MAP = {"kge": "kge_2012", "nse": "nse", "pbias": "pbias_pct",
           "r": "kge_r", "gamma": "kge_gamma", "beta": "kge_beta",
           "kge_no_r": "kge_no_r"}

# -----------------------------------------------------------------------
# FIGURE 1: KGE_2012 -- linear | log, clipped colorbar
# -----------------------------------------------------------------------
kge_norm = mcolors.Normalize(vmin=KGE_CLIP[0], vmax=KGE_CLIP[1])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "kge", 'RdYlGn', kge_norm, "KGE'", log_scale=False, extend='min')
cf = draw_metric_panel(axes[1], "kge", 'RdYlGn', kge_norm, "KGE'", log_scale=True, extend='min')
fig.colorbar(cf, ax=axes, label=f"KGE' (2012), clipped at {KGE_CLIP[0]}, true min {kge_vals.min():.2f}",
             shrink=0.85)
fig.suptitle(f"KGE' (2012 formulation) -- {PAIR_LABEL} joint sensitivity\n{EVENT_LABEL}  |  "
             f"White dashed = PBIAS zero-crossing  |  colorbar clipped below {KGE_CLIP[0]}",
             fontsize=12)
save_fig(fig, "fig1_kge_contour.png")

# -----------------------------------------------------------------------
# FIGURE 2: PBIAS -- linear | log, clipped colorbar
# -----------------------------------------------------------------------
pbias_norm = mcolors.TwoSlopeNorm(vmin=PBIAS_CLIP[0], vcenter=0, vmax=PBIAS_CLIP[1])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "pbias", 'RdBu_r', pbias_norm, 'PBIAS (%)', log_scale=False, extend='both')
cf = draw_metric_panel(axes[1], "pbias", 'RdBu_r', pbias_norm, 'PBIAS (%)', log_scale=True, extend='both')
fig.colorbar(cf, ax=axes, label=f'PBIAS (%), clipped to \u00b1{PBIAS_CLIP[1]:.0f}% '
                                 f'(true range {pbias_vals.min():.0f}% to {pbias_vals.max():.0f}%)',
             shrink=0.85)
fig.suptitle(f"PBIAS (%) -- volume bias in {PAIR_LABEL} space\n{EVENT_LABEL}  |  "
             f"Blue = under-predict, Red = over-predict  |  colorbar clipped to \u00b1{PBIAS_CLIP[1]:.0f}%",
             fontsize=12)
save_fig(fig, "fig2_pbias_contour.png")

# -----------------------------------------------------------------------
# FIGURE 3: r (correlation) -- linear | log
# -----------------------------------------------------------------------
r_norm = mcolors.Normalize(0, 1)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "r", 'Blues', r_norm, 'r', log_scale=False)
cf = draw_metric_panel(axes[1], "r", 'Blues', r_norm, 'r', log_scale=True)
fig.colorbar(cf, ax=axes, label='r (correlation coefficient)', shrink=0.85)
fig.suptitle(f"r (KGE correlation component) -- timing and shape\n{EVENT_LABEL}  |  "
             f"Nearly flat ({r_vals.min():.2f}-{r_vals.max():.2f}) = Ks/f barely affect timing "
             f"(cv/r/n pinned)", fontsize=12)
save_fig(fig, "fig3_r_contour.png")

# -----------------------------------------------------------------------
# FIGURE 4: gamma (CV-based variability ratio, Kling et al. 2012) -- linear | log
# -----------------------------------------------------------------------
gamma_norm = mcolors.TwoSlopeNorm(vmin=1 - gamma_dev, vcenter=1, vmax=1 + gamma_dev)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "gamma", 'RdYlGn', gamma_norm, 'gamma', log_scale=False,
                   show_pbias_zero=False)
cf = draw_metric_panel(axes[1], "gamma", 'RdYlGn', gamma_norm, 'gamma', log_scale=True,
                        show_pbias_zero=False)
fig.colorbar(cf, ax=axes, label='gamma (CV-based variability ratio)  perfect = 1.0', shrink=0.85)
fig.suptitle(f"gamma (2012 variability ratio) -- variability match in {PAIR_LABEL} space\n"
             f"{EVENT_LABEL}  |  >1 = too flashy, <1 = too damped  |  "
             f"replaces alpha (retired, see Handoff_KGE2012Transition_v2.md)", fontsize=12)
save_fig(fig, "fig4_gamma_contour.png")

# -----------------------------------------------------------------------
# FIGURE 5: beta (volume bias) -- linear | log
# -----------------------------------------------------------------------
beta_norm = mcolors.TwoSlopeNorm(vmin=1 - beta_dev, vcenter=1, vmax=1 + beta_dev)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "beta", 'RdYlGn', beta_norm, 'beta', log_scale=False,
                   show_pbias_zero=False)
cf = draw_metric_panel(axes[1], "beta", 'RdYlGn', beta_norm, 'beta', log_scale=True,
                        show_pbias_zero=False)
fig.colorbar(cf, ax=axes, label='beta (bias ratio)  perfect = 1.0', shrink=0.85)
fig.suptitle(f"beta (volume bias ratio) -- mean flow match in {PAIR_LABEL} space\n"
             f"{EVENT_LABEL}  |  >1 = over-predict, <1 = under-predict", fontsize=12)
save_fig(fig, "fig5_beta_contour.png")

# -----------------------------------------------------------------------
# FIGURE 6: NSE -- linear | log, clipped colorbar
# -----------------------------------------------------------------------
nse_norm = mcolors.Normalize(vmin=NSE_CLIP[0], vmax=NSE_CLIP[1])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "nse", 'RdYlGn', nse_norm, 'NSE', log_scale=False, extend='min')
cf = draw_metric_panel(axes[1], "nse", 'RdYlGn', nse_norm, 'NSE', log_scale=True, extend='min')
fig.colorbar(cf, ax=axes, label=f'NSE (clipped at {NSE_CLIP[0]}, true min {nse_vals.min():.2f})',
             shrink=0.85)
fig.suptitle(f"NSE -- {PAIR_LABEL} joint sensitivity\n{EVENT_LABEL}  |  "
             f"colorbar clipped below {NSE_CLIP[0]}", fontsize=12)
save_fig(fig, "fig6_nse_contour.png")

# -----------------------------------------------------------------------
# FIGURE 7: all six metrics, 2x3 panel -- one version per axis scale
# -----------------------------------------------------------------------
panel_configs = [
    ("kge",   "KGE' (2012)",        'RdYlGn', kge_norm,   "KGE'",             True,  'min'),
    ("nse",   "NSE",                'RdYlGn', nse_norm,   'NSE',              True,  'min'),
    ("pbias", "PBIAS (%)",          'RdBu_r', pbias_norm, 'PBIAS (%)',        True,  'both'),
    ("r",     "r (correlation)",    'Blues',  r_norm,     'r',                False, 'neither'),
    ("gamma", "gamma (2012 variability)", 'RdYlGn', gamma_norm, 'gamma (perfect=1)', False, 'neither'),
    ("beta",  "beta (volume bias)", 'RdYlGn', beta_norm,  'beta (perfect=1)',  False, 'neither'),
]

for log_scale, suffix, axis_label in [(False, "linear", "linear f axis"),
                                       (True, "log", "log\u2081\u2080 f axis")]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Joint {PAIR_LABEL} sensitivity -- all metrics ({axis_label})\n"
        f"{EVENT_LABEL}  |  White dashed = PBIAS zero-crossing  |  star = best KGE run  |  "
        f"KGE/NSE/PBIAS colorbars clipped to reduce washout",
        fontsize=13
    )
    for ax, (key, title, cmap, norm, cbar_label, do_pbias_zero, extend) in zip(
            axes.flat, panel_configs):
        cf = draw_metric_panel(ax, key, cmap, norm, cbar_label, log_scale=log_scale,
                                extend=extend, show_pbias_zero=do_pbias_zero)
        fig.colorbar(cf, ax=ax, label=cbar_label, shrink=0.85)
        ax.set_title(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f"fig7_all_metrics_panel_{suffix}.png")

# -----------------------------------------------------------------------
# FIGURE 8: RAW-SCATTER DIAGNOSTIC -- NO INTERPOLATION
#
# Purpose: the contourf/griddata panels above can show dense vertical
# banding or a zigzagging PBIAS=0 contour line if the interpolation
# misbehaves. This figure sidesteps griddata entirely -- it bins runs
# into a few narrow f_RS_abs slices and plots raw (kge_2012, gamma, pbias) vs
# Ks_mult directly, so any fine-scale "banding" that shows up here is real
# point-to-point run noise, and anything that DOESN'T show up here (i.e.
# only appears in the interpolated panels) is a cubic-interpolation/
# contour artifact.
#
# Three rows: KGE_2012, gamma (2012 variability), PBIAS. The PBIAS row's
# y-axis is algebraically identical to kge_beta (PBIAS_pct = 100*(kge_beta
# - 1), confirmed exactly to floating-point precision) -- labeled
# explicitly below so the redundancy is visible rather than silently
# duplicated. gamma is DELIBERATELY decoupled from PBIAS/beta by
# construction (Kling et al. 2012's whole point) -- r~0.37 on this
# dataset, vs. r~0.99 for the retired alpha term it replaces. Expect the
# gamma row to NOT track the PBIAS row's shape -- that's the fix working,
# not a bug. See Handoff_KGE2012Transition_v2.md.
# -----------------------------------------------------------------------
N_F_BINS = 4

# log-spaced bin centers across the observed f range, with a tolerance
# window scaled to local density (log-stratified sampling means points
# are much denser at low f, sparser at high f)
f_bin_centers = np.logspace(np.log10(second_pts.min()), np.log10(second_pts.max()), N_F_BINS)

fig, axes = plt.subplots(3, N_F_BINS, figsize=(4.5 * N_F_BINS, 11), sharex=True)

for j, f_center in enumerate(f_bin_centers):
    # +/-15% multiplicative window around the log-spaced center
    lo_bound, hi_bound = f_center / 1.15, f_center * 1.15
    mask = (second_pts >= lo_bound) & (second_pts <= hi_bound)
    n_pts = mask.sum()

    ks_bin    = ks_pts[mask]
    kge_bin   = kge_vals[mask]
    gamma_bin = gamma_vals[mask]
    pbias_bin = pbias_vals[mask]

    order = np.argsort(ks_bin)

    ax_kge = axes[0, j]
    ax_kge.scatter(ks_bin, kge_bin, s=22, color='steelblue', edgecolors='white', linewidths=0.5)
    ax_kge.plot(ks_bin[order], kge_bin[order], color='steelblue', alpha=0.3, linewidth=1.0)
    ax_kge.set_title(f"f \u2248 {f_center:.4f}  (\u00b115%, n={n_pts})", fontsize=10)
    ax_kge.set_ylabel("KGE' 2012 (raw)" if j == 0 else "")
    ax_kge.axhline(best_kge, color='gray', linestyle=':', linewidth=0.8)

    ax_gamma = axes[1, j]
    ax_gamma.scatter(ks_bin, gamma_bin, s=22, color='seagreen', edgecolors='white', linewidths=0.5)
    ax_gamma.plot(ks_bin[order], gamma_bin[order], color='seagreen', alpha=0.3, linewidth=1.0)
    ax_gamma.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax_gamma.set_ylabel("gamma (raw, perfect=1)" if j == 0 else "")

    ax_pbias = axes[2, j]
    ax_pbias.scatter(ks_bin, pbias_bin, s=22, color='firebrick', edgecolors='white', linewidths=0.5)
    ax_pbias.plot(ks_bin[order], pbias_bin[order], color='firebrick', alpha=0.3, linewidth=1.0)
    ax_pbias.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax_pbias.set_ylabel("PBIAS % (raw)\n(= kge_beta)" if j == 0 else "")
    ax_pbias.set_xlabel("Ks multiplier")

fig.suptitle(
    f"Raw-scatter diagnostic (no interpolation) -- {PAIR_LABEL}\n"
    f"{EVENT_LABEL}  |  Points binned into narrow f-slices, connecting line is raw "
    f"point order only (not a fit)  |  Dotted gray = best sampled KGE', dashed black = "
    f"gamma=1 / PBIAS=0  |  PBIAS (%) = 100\u00d7(kge_beta \u2212 1), shown here in PBIAS units",
    fontsize=12
)
fig.tight_layout()
save_fig(fig, "fig8_raw_scatter_diagnostic.png")

# -----------------------------------------------------------------------
# FIGURE 9: RIDGE WIDTH VS f (folded in from measure_ridge_width_vs_f.py)
#
# Reuses points_norm / normalize_ks / normalize_logf / kge_vals already
# built above -- no re-import, no duplicate interpolation setup. Only new
# thing needed is a finer 1D Ks grid for clean cross-section reads.
# -----------------------------------------------------------------------
ks_grid_1d_fine = np.linspace(ks_pts.min(), ks_pts.max(), N_GRID_KS_FINE)


def kge_cross_section(f_slice):
    """Interpolated KGE_2012 vs Ks at a single fixed f value (1D slice
    through the same normalized-coordinate cubic surface used for the 2D
    contours above)."""
    ks_norm = normalize_ks(ks_grid_1d_fine)
    f_norm  = np.full_like(ks_grid_1d_fine, normalize_logf(f_slice))
    return griddata(points_norm, kge_vals, (ks_norm, f_norm), method='cubic')


def ridge_stats(f_slice, threshold=RIDGE_KGE_THRESHOLD):
    """Ridge peak Ks, peak KGE_2012, and >=threshold band width (Ks units)
    at one f-slice. Returns (nan, nan, nan) if the slice falls outside the
    convex hull of sampled points (griddata returns all-NaN)."""
    kge_line = kge_cross_section(f_slice)
    valid = ~np.isnan(kge_line)
    if valid.sum() == 0:
        return np.nan, np.nan, np.nan
    ks_valid, kge_valid = ks_grid_1d_fine[valid], kge_line[valid]
    peak_idx = np.argmax(kge_valid)
    peak_ks, peak_kge = ks_valid[peak_idx], kge_valid[peak_idx]
    above = ks_valid[kge_valid >= threshold]
    width = (above.max() - above.min()) if len(above) > 0 else 0.0
    return peak_ks, peak_kge, width


# -- checkpoint table
ridge_rows = []
for f_slice in REFERENCE_F_SLICES:
    peak_ks, peak_kge, width = ridge_stats(f_slice)
    ridge_rows.append({
        "f_slice": f_slice,
        "ridge_peak_Ks": peak_ks,
        "peak_KGE_2012": peak_kge,
        f"KGE_2012_ge_{RIDGE_KGE_THRESHOLD}_band_width_Ks_units": width,
    })

ridge_table = pd.DataFrame(ridge_rows)
ridge_table_path = plot_dir / "ridge_width_vs_f.csv"
ridge_table.to_csv(ridge_table_path, index=False)
print(f"\nRidge width checkpoint table -> {ridge_table_path.name}")
print(ridge_table.to_string(index=False))

# -- dense curve for panel B
f_curve = np.logspace(np.log10(second_pts.min()), np.log10(second_pts.max()), N_F_CURVE)
curve_rows = [ridge_stats(f) for f in f_curve]
peak_ks_curve, peak_kge_curve, width_curve = map(np.array, zip(*curve_rows))

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(CROSS_SECTION_SLICES)))
for f_slice, color in zip(CROSS_SECTION_SLICES, colors):
    kge_line = kge_cross_section(f_slice)
    is_true_f = np.isclose(f_slice, TRUE_F)
    label = f"f={f_slice:g}" + ("  (true f)" if is_true_f else "")
    ax_a.plot(ks_grid_1d_fine, kge_line, color=color,
              linewidth=2.4 if is_true_f else 1.3, label=label)
    above = kge_line >= RIDGE_KGE_THRESHOLD
    if np.any(above):
        ax_a.fill_between(ks_grid_1d_fine, RIDGE_KGE_THRESHOLD, kge_line, where=above,
                           color=color, alpha=0.15)

ax_a.axhline(RIDGE_KGE_THRESHOLD, color='gray', linestyle=':', linewidth=1.0,
             label=f"KGE'={RIDGE_KGE_THRESHOLD} threshold")
ax_a.axvline(TRUE_KS, color='black', linestyle='--', linewidth=1.0, alpha=0.6,
             label="true Ks")
ax_a.set_xlabel("Ks multiplier")
ax_a.set_ylabel("KGE' (2012), interpolated cross-section")
ax_a.set_title("(A) KGE' vs Ks at fixed f-slices\nshaded = counted in panel B's width")
ax_a.legend(fontsize=8, loc='lower center')
ax_a.set_ylim(0, 1.02)

ax_b.plot(f_curve, width_curve, color='steelblue', linewidth=2.0)
ax_b.axvline(TRUE_F, color='black', linestyle='--', linewidth=1.0, alpha=0.6,
             label="true f")
ax_b.set_xscale('log')
ax_b.set_xlabel("f_RS_abs (log)")
ax_b.set_ylabel(f"KGE' \u2265 {RIDGE_KGE_THRESHOLD} band width (Ks units)")
ax_b.set_title("(B) Ridge width vs f\nwidest near true f, collapses above f~0.015")
ax_b.legend(fontsize=9)

fig.suptitle(f"Equifinality ridge width vs f -- {EVENT_LABEL}", fontsize=12)
fig.tight_layout()
save_fig(fig, "fig9_ridge_width_vs_f.png")

# -----------------------------------------------------------------------
# FIGURE 10: KGE' WITHOUT r (top), AND BETA/GAMMA LITERAL INTERSECTION (bottom)
#
# Top row: reuses draw_metric_panel exactly like figures 1-6, on the
# "kge_no_r" surface built earlier from gamma+beta only (r excluded). Uses
# the SAME kge_norm/colorbar range as fig1 so the two are directly
# comparable side by side -- if r were doing anything interesting, this
# panel would look visibly different from fig1. If it looks nearly
# identical, that's confirmation r is contributing ~nothing, not a
# failed test.
#
# Bottom row: NOT a continuous metric -- a literal overlap of two binary
# feasibility masks (|beta-1| <= tol, |gamma-1| <= tol), shaded separately
# with their intersection shaded a third way. This is the direct picture
# of why the composite optimum sits where it does: at the overlap of
# "volume is right" (broad, follows the diagonal ridge) and "shape is
# right" (narrower, does not follow the diagonal ridge -- see fig4).
# -----------------------------------------------------------------------
def draw_intersection_panel(ax, log_scale, tol=BETA_GAMMA_FEASIBLE_TOL):
    surfaces = SURFACES_LOG if log_scale else SURFACES_LIN
    ks_g = KS_GRID_LOG if log_scale else KS_GRID_LIN
    f_g  = F_GRID_LOG if log_scale else F_GRID_LIN

    beta_mask  = (np.abs(surfaces["beta"]  - 1) <= tol).astype(float)
    gamma_mask = (np.abs(surfaces["gamma"] - 1) <= tol).astype(float)
    joint_mask = ((beta_mask == 1) & (gamma_mask == 1)).astype(float)

    ax.contourf(ks_g, f_g, beta_mask,  levels=[0.5, 1.5], colors=['tab:blue'],   alpha=0.30)
    ax.contourf(ks_g, f_g, gamma_mask, levels=[0.5, 1.5], colors=['tab:orange'], alpha=0.30)
    ax.contourf(ks_g, f_g, joint_mask, levels=[0.5, 1.5], colors=['black'],      alpha=0.55)

    ax.scatter(ks_pts, second_pts, s=8, color='gray', alpha=0.3, zorder=4)
    ax.scatter([best_ks], [best_second], **BEST_KW)
    ax.scatter([TRUE_KS], [TRUE_F], **TRUE_KW)

    if log_scale:
        ax.set_yscale('log')
        ax.set_title("log\u2081\u2080 f axis", fontsize=10)
    else:
        ax.set_title("linear f axis", fontsize=10)

    ax.set_xlabel(XLABEL, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10)


from matplotlib.gridspec import GridSpec

# Third column is narrower and reserved ONLY for the top-row colorbar and
# the bottom-row legend -- giving both their own dedicated axes (via `cax=`
# and a turned-off legend-holder axes) instead of letting fig.colorbar's
# `ax=` auto-shrink and fig.legend's `loc='lower center'` fight with
# tight_layout for space after the fact. That fight was the original cause
# of the colorbar landing on top of the top-right panel and the legend
# being pinned below the whole figure instead of beside it.
fig = plt.figure(figsize=(17, 12))
gs = GridSpec(2, 3, width_ratios=[1, 1, 0.4], wspace=0.32, hspace=0.32, figure=fig)

ax_00    = fig.add_subplot(gs[0, 0])
ax_01    = fig.add_subplot(gs[0, 1])
ax_cbar  = fig.add_subplot(gs[0, 2])   # dedicated colorbar axes, top row only

ax_10     = fig.add_subplot(gs[1, 0])
ax_11     = fig.add_subplot(gs[1, 1])
ax_legend = fig.add_subplot(gs[1, 2])  # dedicated legend-holder axes, bottom row only
ax_legend.axis('off')

kge_no_r_norm = kge_norm  # same clipped range as fig1, for direct comparability
draw_metric_panel(ax_00, "kge_no_r", 'RdYlGn', kge_no_r_norm, "KGE' (no r)",
                   log_scale=False, extend='min')
cf_top = draw_metric_panel(ax_01, "kge_no_r", 'RdYlGn', kge_no_r_norm, "KGE' (no r)",
                            log_scale=True, extend='min')
fig.colorbar(cf_top, cax=ax_cbar, label="KGE' (\u03b3+\u03b2 only, r excluded)")
ax_00.set_title("linear f axis -- KGE' (no r)", fontsize=10)
ax_01.set_title("log\u2081\u2080 f axis -- KGE' (no r)", fontsize=10)

draw_intersection_panel(ax_10, log_scale=False)
draw_intersection_panel(ax_11, log_scale=True)
ax_10.set_title("linear f axis -- beta/gamma overlap", fontsize=10)
ax_11.set_title("log\u2081\u2080 f axis -- beta/gamma overlap", fontsize=10)

legend_handles = [
    mpatches.Patch(color='tab:blue', alpha=0.30,
                    label=f"beta feasible\n(|\u03b2\u22121| \u2264 {BETA_GAMMA_FEASIBLE_TOL})"),
    mpatches.Patch(color='tab:orange', alpha=0.30,
                    label=f"gamma feasible\n(|\u03b3\u22121| \u2264 {BETA_GAMMA_FEASIBLE_TOL})"),
    mpatches.Patch(color='black', alpha=0.55, label="both (joint)"),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=12, linestyle='None', label="best KGE' run"),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=11, linestyle='None', label="true parameters"),
]
ax_legend.legend(handles=legend_handles, loc='center', fontsize=9, frameon=False,
                  handletextpad=0.8, labelspacing=1.3)

fig.suptitle(
    f"KGE' without r, and the literal beta/gamma intersection -- {PAIR_LABEL}\n"
    f"{EVENT_LABEL}  |  Top: does removing r change the ridge? (compare to fig1)  |  "
    f"Bottom: where are volume-match and shape-match BOTH satisfied?",
    fontsize=13
)
save_fig(fig, "fig10_beta_gamma_intersection.png")

print(f"\nAll figures + tables saved to:\n  {plot_dir}")
