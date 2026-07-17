"""
plot_lhs_Ks_f_100.py
======================
Generates contour plots from the Series 100 synthetic Ks_mult x f_RS_abs LHS
sweep (lhs_results_synth_Ks_f_100.csv), scored against the NEW noise-free
synthetic truth hydrograph (cv/r/n pinned at confirmed truth values:
cv=4.5, r=0.24, n=0.026; Ks_mult=7.0x, f_RS_abs=0.012).

This is a direct adaptation of plot_lhs_Ks_f_97log.py -- same methodology,
same normalized-coordinate interpolation fix, same 8-figure output set.
Only the config block changed (results file, output subdir, event label,
top-20 table filename) to point at the Series 100 truth reset. The script
is otherwise fully data-driven: the best-KGE run, axis ranges, and
interpolation grids are all computed from whatever's in the CSV, so no
truth values are hardcoded in the plotting logic itself.

KGE_CLIP/NSE_CLIP/PBIAS_CLIP colorbar ranges are CARRIED OVER UNCHANGED
from Series 97log, where they were tuned to that series' old-truth Ks>=9x
anomaly (KGE bottoming near -0.4, PBIAS reaching -72%/+93%). The new
truth's surface may not have the same extremes in the same place -- check
the printed kge_vals.min()/max() and pbias_vals.min()/max() after the
first run and tighten/loosen these three constants if the panels look
washed out or over-clipped.

Usage (run from the smf_demo directory):
    python plot_lhs_Ks_f_100.py

Produces 8 figures + 1 CSV saved to:
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_f_100/

Figure list:
    fig1_kge_contour.png          -- KGE, linear|log side by side, clipped colorbar
    fig2_pbias_contour.png        -- PBIAS, linear|log side by side, clipped colorbar
    fig3_r_contour.png            -- r (correlation), linear|log side by side
    fig4_alpha_contour.png        -- alpha (flashiness), linear|log side by side
    fig5_beta_contour.png         -- beta (volume bias), linear|log side by side
    fig6_nse_contour.png          -- NSE, linear|log side by side, clipped colorbar
    fig7_all_metrics_panel_linear.png  -- all six metrics, 2x3, linear f axis
    fig7_all_metrics_panel_log.png     -- all six metrics, 2x3, log10 f axis
    fig8_raw_scatter_diagnostic.png    -- raw (no-interpolation) KGE/PBIAS vs Ks, binned by f
    top20_pbias_synth_100.csv     -- 20 runs with smallest |PBIAS|
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.interpolate import griddata

# ======================================================================
# CONFIG
# ======================================================================
RESULTS_CSV   = "lhs_results_synth_Ks_f_100.csv"
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

required_cols = ["Ks_mult", SECOND_COL, "kge", "nse", "pbias_pct",
                  "kge_r", "kge_alpha", "kge_beta"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in results CSV: {missing_cols}")

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"  {len(df)} runs after dropping NaN rows")

# -----------------------------------------------------------------------
# TOP-20 BY |PBIAS| TABLE
# -----------------------------------------------------------------------
table_cols = [c for c in ["run_id", "Ks_mult", "f_RS_abs", "kge", "pbias_pct",
                           "nse", "kge_alpha", "kge_beta"] if c in df.columns]
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
kge_vals   = df["kge"].values
nse_vals   = df["nse"].values
pbias_vals = df["pbias_pct"].values
r_vals     = df["kge_r"].values
alpha_vals = df["kge_alpha"].values
beta_vals  = df["kge_beta"].values

best_idx    = np.argmax(kge_vals)
best_ks     = ks_pts[best_idx]
best_second = second_pts[best_idx]
best_kge    = kge_vals[best_idx]

print(f"\n  Best run: Ks={best_ks:.2f}x  f={best_second:.4g}  KGE={best_kge:.3f}  "
      f"PBIAS={pbias_vals[best_idx]:+.1f}%")
print(f"  KGE range:   {kge_vals.min():.3f} to {kge_vals.max():.3f}")
print(f"  PBIAS range: {pbias_vals.min():.1f}% to {pbias_vals.max():.1f}%")

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
    "alpha": interp_surface(alpha_vals, KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
    "beta":  interp_surface(beta_vals,  KS_GRID_LIN_NORM, F_GRID_LIN_NORM),
}
SURFACES_LOG = {
    "kge":   interp_surface(kge_vals,   KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "nse":   interp_surface(nse_vals,   KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "pbias": interp_surface(pbias_vals, KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "r":     interp_surface(r_vals,     KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "alpha": interp_surface(alpha_vals, KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
    "beta":  interp_surface(beta_vals,  KS_GRID_LOG_NORM, F_GRID_LOG_NORM),
}

alpha_dev = max(abs(alpha_vals.min() - 1), abs(alpha_vals.max() - 1))
beta_dev  = max(abs(beta_vals.min() - 1), abs(beta_vals.max() - 1))

XLABEL = "Ks multiplier"
YLABEL = SECOND_LABEL

SCATTER_KW = dict(s=16, edgecolors='white', linewidths=0.4, alpha=0.7, zorder=5)
BEST_KW    = dict(s=110, marker='*', color='white', edgecolors='black',
                  linewidths=1.2, zorder=10)


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


COL_MAP = {"kge": "kge", "nse": "nse", "pbias": "pbias_pct",
           "r": "kge_r", "alpha": "kge_alpha", "beta": "kge_beta"}

# -----------------------------------------------------------------------
# FIGURE 1: KGE -- linear | log, clipped colorbar
# -----------------------------------------------------------------------
kge_norm = mcolors.Normalize(vmin=KGE_CLIP[0], vmax=KGE_CLIP[1])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "kge", 'RdYlGn', kge_norm, 'KGE', log_scale=False, extend='min')
cf = draw_metric_panel(axes[1], "kge", 'RdYlGn', kge_norm, 'KGE', log_scale=True, extend='min')
fig.colorbar(cf, ax=axes, label=f'KGE (clipped at {KGE_CLIP[0]}, true min {kge_vals.min():.2f})',
             shrink=0.85)
fig.suptitle(f"KGE -- {PAIR_LABEL} joint sensitivity\n{EVENT_LABEL}  |  "
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
# FIGURE 4: alpha (flashiness) -- linear | log
# -----------------------------------------------------------------------
alpha_norm = mcolors.TwoSlopeNorm(vmin=1 - alpha_dev, vcenter=1, vmax=1 + alpha_dev)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_metric_panel(axes[0], "alpha", 'RdYlGn', alpha_norm, 'alpha', log_scale=False,
                   show_pbias_zero=False)
cf = draw_metric_panel(axes[1], "alpha", 'RdYlGn', alpha_norm, 'alpha', log_scale=True,
                        show_pbias_zero=False)
fig.colorbar(cf, ax=axes, label='alpha (variability ratio)  perfect = 1.0', shrink=0.85)
fig.suptitle(f"alpha (flashiness ratio) -- variability match in {PAIR_LABEL} space\n"
             f"{EVENT_LABEL}  |  >1 = too flashy, <1 = too damped", fontsize=12)
save_fig(fig, "fig4_alpha_contour.png")

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
    ("kge",   "KGE",                'RdYlGn', kge_norm,   'KGE',              True,  'min'),
    ("nse",   "NSE",                'RdYlGn', nse_norm,   'NSE',              True,  'min'),
    ("pbias", "PBIAS (%)",          'RdBu_r', pbias_norm, 'PBIAS (%)',        True,  'both'),
    ("r",     "r (correlation)",    'Blues',  r_norm,     'r',                False, 'neither'),
    ("alpha", "alpha (flashiness)", 'RdYlGn', alpha_norm, 'alpha (perfect=1)', False, 'neither'),
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
# into a few narrow f_RS_abs slices and plots raw (kge, alpha, pbias) vs
# Ks_mult directly, so any fine-scale "banding" that shows up here is real
# point-to-point run noise, and anything that DOESN'T show up here (i.e.
# only appears in the interpolated panels) is a cubic-interpolation/
# contour artifact.
#
# Three rows: KGE, alpha (flashiness), PBIAS. The PBIAS row's y-axis is
# algebraically identical to kge_beta (PBIAS_pct = 100*(kge_beta - 1),
# confirmed exactly to floating-point precision) -- labeled explicitly
# below so the redundancy is visible rather than silently duplicated.
# alpha is NOT redundant with PBIAS/beta on its own axis definition, but
# note the two are highly correlated in this Ks x f design (r~0.99 across
# the full dataset) -- expect the alpha row to look similar in shape to
# the PBIAS row, just on a different scale.
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
    alpha_bin = alpha_vals[mask]
    pbias_bin = pbias_vals[mask]

    order = np.argsort(ks_bin)

    ax_kge = axes[0, j]
    ax_kge.scatter(ks_bin, kge_bin, s=22, color='steelblue', edgecolors='white', linewidths=0.5)
    ax_kge.plot(ks_bin[order], kge_bin[order], color='steelblue', alpha=0.3, linewidth=1.0)
    ax_kge.set_title(f"f \u2248 {f_center:.4f}  (\u00b115%, n={n_pts})", fontsize=10)
    ax_kge.set_ylabel("KGE (raw)" if j == 0 else "")
    ax_kge.axhline(best_kge, color='gray', linestyle=':', linewidth=0.8)

    ax_alpha = axes[1, j]
    ax_alpha.scatter(ks_bin, alpha_bin, s=22, color='seagreen', edgecolors='white', linewidths=0.5)
    ax_alpha.plot(ks_bin[order], alpha_bin[order], color='seagreen', alpha=0.3, linewidth=1.0)
    ax_alpha.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax_alpha.set_ylabel("alpha (raw, perfect=1)" if j == 0 else "")

    ax_pbias = axes[2, j]
    ax_pbias.scatter(ks_bin, pbias_bin, s=22, color='firebrick', edgecolors='white', linewidths=0.5)
    ax_pbias.plot(ks_bin[order], pbias_bin[order], color='firebrick', alpha=0.3, linewidth=1.0)
    ax_pbias.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax_pbias.set_ylabel("PBIAS % (raw)\n(= kge_beta)" if j == 0 else "")
    ax_pbias.set_xlabel("Ks multiplier")

fig.suptitle(
    f"Raw-scatter diagnostic (no interpolation) -- {PAIR_LABEL}\n"
    f"{EVENT_LABEL}  |  Points binned into narrow f-slices, connecting line is raw "
    f"point order only (not a fit)  |  Dotted gray = best sampled KGE, dashed black = "
    f"alpha=1 / PBIAS=0  |  PBIAS (%) = 100\u00d7(kge_beta \u2212 1), shown here in PBIAS units",
    fontsize=12
)
fig.tight_layout()
save_fig(fig, "fig8_raw_scatter_diagnostic.png")

print(f"\nAll figures + top-20 table saved to:\n  {plot_dir}")
