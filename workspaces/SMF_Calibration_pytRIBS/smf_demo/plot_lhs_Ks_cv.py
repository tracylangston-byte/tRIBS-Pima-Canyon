"""
plot_lhs_Ks_cv.py
=================
Generates contour plots from the joint Ks_mult × kinemvelcoef (cv) LHS sweep.
Run after run_lhs_sweep.py completes.

Usage (run from the smf_demo directory):
    python plot_lhs_Ks_cv.py

Produces 7 figures saved to:
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_cv/

Figure list:
    fig1_kge_contour.png          — KGE contour + PBIAS zero-crossing overlay
    fig2_pbias_contour.png        — PBIAS contour with zero-crossing line
    fig3_r_contour.png            — Correlation coefficient r (timing/shape)
    fig4_alpha_contour.png        — alpha variability ratio (flashiness)
    fig5_beta_contour.png         — beta bias ratio (volume)
    fig6_nse_contour.png          — NSE
    fig7_all_metrics_panel.png    — All six metrics in one 2x3 panel figure

Naming conventions used in this script
---------------------------------------
  Scattered data (1D arrays from DataFrame, one value per LHS run):
      ks_pts   — Ks multiplier values
      cv_pts   — hillslope velocity coefficient (kinemvelcoef) values
      kge_vals, nse_vals, pbias_vals, r_vals, alpha_vals, beta_vals

  Interpolation grid (1D axis vectors):
      ks_grid_1d  — evenly spaced Ks axis
      cv_grid_1d  — evenly spaced cv axis

  Meshgrid arrays (2D, used in contourf/contour calls):
      KS_GRID  — 2D Ks meshgrid
      CV_GRID  — 2D cv meshgrid

  Interpolated metric surfaces (2D, same shape as KS_GRID / CV_GRID):
      KGE_SURF, NSE_SURF, PBIAS_SURF, R_SURF, ALPHA_SURF, BETA_SURF

  "kinemvelcoef" appears only when referencing the DataFrame column by name.
  Axis labels use the full name "Hillslope velocity coef. (cv)".
  Annotations and legends use the short form "cv".
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.interpolate import griddata

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / "lhs_Ks_cv"
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------
results_path = summary_dir / "lhs_results_Ks_cv.csv"
if not results_path.exists():
    raise FileNotFoundError(
        f"LHS results not found: {results_path}\n"
        "Run run_lhs_sweep.py first."
    )

df = pd.read_csv(results_path)
print(f"Loaded {len(df)} LHS runs from {results_path.name}")

# Verify required columns exist
required_cols = ["Ks_mult", "kinemvelcoef", "kge", "nse", "pbias_pct",
                 "kge_r", "kge_alpha", "kge_beta"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in results CSV: {missing_cols}")

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"  {len(df)} runs after dropping NaN rows")

# -----------------------------------------------------------------------
# EXTRACT 1D SCATTERED DATA ARRAYS
# All named with _pts or _vals suffix to distinguish from grid arrays.
# -----------------------------------------------------------------------
ks_pts    = df["Ks_mult"].values
cv_pts    = df["kinemvelcoef"].values   # "kinemvelcoef" = column name only
kge_vals  = df["kge"].values
nse_vals  = df["nse"].values
pbias_vals = df["pbias_pct"].values
r_vals    = df["kge_r"].values
alpha_vals = df["kge_alpha"].values
beta_vals  = df["kge_beta"].values

# Best run by KGE
best_idx = np.argmax(kge_vals)
best_ks  = ks_pts[best_idx]
best_cv  = cv_pts[best_idx]
best_kge = kge_vals[best_idx]

print(f"\n  Best run: Ks={best_ks:.2f}x  cv={best_cv:.3f}  KGE={best_kge:.3f}")
print(f"  KGE range:   {kge_vals.min():.3f} to {kge_vals.max():.3f}")
print(f"  PBIAS range: {pbias_vals.min():.1f}% to {pbias_vals.max():.1f}%")

# -----------------------------------------------------------------------
# BUILD INTERPOLATION GRID
# ks_grid_1d / cv_grid_1d are 1D axis vectors.
# KS_GRID / CV_GRID are the 2D meshgrid arrays used in contour calls.
# points is the (N, 2) array of scattered sample locations.
# -----------------------------------------------------------------------
N_GRID = 200   # grid resolution per axis — increase for smoother contours

ks_grid_1d = np.linspace(ks_pts.min(), ks_pts.max(), N_GRID)
cv_grid_1d = np.linspace(cv_pts.min(), cv_pts.max(), N_GRID)
KS_GRID, CV_GRID = np.meshgrid(ks_grid_1d, cv_grid_1d)

# Scattered sample locations — built from 1D arrays before meshgrid
points = np.column_stack([ks_pts, cv_pts])


def interp_surface(values):
    """Interpolate scattered metric values onto the regular KS_GRID/CV_GRID."""
    return griddata(points, values, (KS_GRID, CV_GRID), method='cubic')


KGE_SURF   = interp_surface(kge_vals)
NSE_SURF   = interp_surface(nse_vals)
PBIAS_SURF = interp_surface(pbias_vals)
R_SURF     = interp_surface(r_vals)
ALPHA_SURF = interp_surface(alpha_vals)
BETA_SURF  = interp_surface(beta_vals)

# -----------------------------------------------------------------------
# SHARED STYLE CONSTANTS AND HELPERS
# -----------------------------------------------------------------------
XLABEL      = "Ks multiplier"
YLABEL      = "Hillslope velocity coef. (cv)"
EVENT_LABEL = "SMF Aug 12, 2014"

SCATTER_KW = dict(s=18, edgecolors='white', linewidths=0.4, alpha=0.7, zorder=5)
BEST_KW    = dict(s=120, marker='*', color='white', edgecolors='black',
                  linewidths=1.2, zorder=10)


def label_axes(ax):
    ax.set_xlabel(XLABEL, fontsize=11)
    ax.set_ylabel(YLABEL, fontsize=11)


def plot_best_star(ax):
    """Mark the best-KGE run with a star."""
    ax.scatter([best_ks], [best_cv], **BEST_KW,
               label=f"Best: Ks={best_ks:.2f}x, cv={best_cv:.2f}, KGE={best_kge:.3f}")


def plot_pbias_zeroline(ax):
    """Overlay the PBIAS=0 contour as a white dashed line."""
    try:
        cs = ax.contour(KS_GRID, CV_GRID, PBIAS_SURF, levels=[0],
                        colors=['white'], linewidths=2.0, linestyles='--', zorder=6)
        ax.clabel(cs, fmt="PBIAS=0", fontsize=8, colors='white')
    except Exception:
        pass  # Skip gracefully if zero not within range


def save_fig(fig, filename):
    path = plot_dir / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path.name}")
    plt.close(fig)


# -----------------------------------------------------------------------
# FIGURE 1: KGE contour + PBIAS zero-crossing overlay
# Headline figure — shows where calibration optimum lies
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

cf = ax.contourf(KS_GRID, CV_GRID, KGE_SURF, levels=20, cmap='RdYlGn',
                 vmin=max(-1, kge_vals.min()), vmax=1)
fig.colorbar(cf, ax=ax, label='KGE')

cl = ax.contour(KS_GRID, CV_GRID, KGE_SURF, levels=10,
                colors='black', linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%.2f", fontsize=7, colors='black')

ax.scatter(ks_pts, cv_pts, c=kge_vals, cmap='RdYlGn',
           vmin=max(-1, kge_vals.min()), vmax=1, **SCATTER_KW)

plot_pbias_zeroline(ax)
plot_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
label_axes(ax)
ax.set_title(
    f"KGE — Ks x cv joint sensitivity\n"
    f"{EVENT_LABEL}  |  White dashed = PBIAS zero-crossing",
    fontsize=12
)
fig.tight_layout()
save_fig(fig, "fig1_kge_contour.png")


# -----------------------------------------------------------------------
# FIGURE 2: PBIAS contour
# Diverging colormap centered on zero — shows volume bias across parameter space
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

pbias_abs_max = max(abs(pbias_vals.min()), abs(pbias_vals.max()))
pbias_norm = mcolors.TwoSlopeNorm(
    vmin=-pbias_abs_max, vcenter=0, vmax=pbias_abs_max
)

cf = ax.contourf(KS_GRID, CV_GRID, PBIAS_SURF, levels=20,
                 cmap='RdBu_r', norm=pbias_norm)
fig.colorbar(cf, ax=ax, label='PBIAS (%)')

cl = ax.contour(KS_GRID, CV_GRID, PBIAS_SURF, levels=10,
                colors='black', linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%+.0f%%", fontsize=7, colors='black')

ax.scatter(ks_pts, cv_pts, c=pbias_vals, cmap='RdBu_r', norm=pbias_norm,
           **SCATTER_KW)

plot_pbias_zeroline(ax)
plot_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
label_axes(ax)
ax.set_title(
    f"PBIAS (%) — volume bias in Ks x cv space\n"
    f"{EVENT_LABEL}  |  White dashed = zero bias  |  "
    f"Blue = under-predict, Red = over-predict",
    fontsize=11
)
fig.tight_layout()
save_fig(fig, "fig2_pbias_contour.png")


# -----------------------------------------------------------------------
# FIGURE 3: r (correlation) contour
# Expected to be nearly flat — confirms timing is controlled by forcing
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

cf = ax.contourf(KS_GRID, CV_GRID, R_SURF, levels=20, cmap='Blues',
                 vmin=0, vmax=1)
fig.colorbar(cf, ax=ax, label='r (correlation coefficient)')

cl = ax.contour(KS_GRID, CV_GRID, R_SURF, levels=10,
                colors='black', linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%.2f", fontsize=7, colors='black')

ax.scatter(ks_pts, cv_pts, c=r_vals, cmap='Blues', vmin=0, vmax=1,
           **SCATTER_KW)

plot_pbias_zeroline(ax)
plot_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
label_axes(ax)
ax.set_title(
    f"r (KGE correlation component) — timing and shape\n"
    f"{EVENT_LABEL}  |  Flat surface = timing controlled by forcing, not parameters",
    fontsize=11
)
fig.tight_layout()
save_fig(fig, "fig3_r_contour.png")


# -----------------------------------------------------------------------
# FIGURE 4: alpha (variability ratio / flashiness) contour
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

alpha_dev = max(abs(alpha_vals.min() - 1), abs(alpha_vals.max() - 1))
alpha_norm = mcolors.TwoSlopeNorm(
    vmin=1 - alpha_dev, vcenter=1, vmax=1 + alpha_dev
)

cf = ax.contourf(KS_GRID, CV_GRID, ALPHA_SURF, levels=20,
                 cmap='RdYlGn', norm=alpha_norm)
fig.colorbar(cf, ax=ax, label='alpha (variability ratio)  perfect = 1.0')

# Mark the alpha=1 line explicitly
try:
    cl1 = ax.contour(KS_GRID, CV_GRID, ALPHA_SURF, levels=[1.0],
                     colors='white', linewidths=2.0, linestyles='--', zorder=6)
    ax.clabel(cl1, fmt="alpha=1", fontsize=8, colors='white')
except Exception:
    pass

ax.scatter(ks_pts, cv_pts, c=alpha_vals, cmap='RdYlGn', norm=alpha_norm,
           **SCATTER_KW)

plot_best_star(ax)
ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
label_axes(ax)
ax.set_title(
    f"alpha (flashiness ratio) — variability match in Ks x cv space\n"
    f"{EVENT_LABEL}  |  White dashed = alpha=1 (perfect)  |  "
    f">1 = too flashy, <1 = too damped",
    fontsize=11
)
fig.tight_layout()
save_fig(fig, "fig4_alpha_contour.png")


# -----------------------------------------------------------------------
# FIGURE 5: beta (bias ratio / mean flow) contour
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

beta_dev = max(abs(beta_vals.min() - 1), abs(beta_vals.max() - 1))
beta_norm = mcolors.TwoSlopeNorm(
    vmin=1 - beta_dev, vcenter=1, vmax=1 + beta_dev
)

cf = ax.contourf(KS_GRID, CV_GRID, BETA_SURF, levels=20,
                 cmap='RdYlGn', norm=beta_norm)
fig.colorbar(cf, ax=ax, label='beta (bias ratio)  perfect = 1.0')

try:
    cl1 = ax.contour(KS_GRID, CV_GRID, BETA_SURF, levels=[1.0],
                     colors='white', linewidths=2.0, linestyles='--', zorder=6)
    ax.clabel(cl1, fmt="beta=1", fontsize=8, colors='white')
except Exception:
    pass

ax.scatter(ks_pts, cv_pts, c=beta_vals, cmap='RdYlGn', norm=beta_norm,
           **SCATTER_KW)

plot_best_star(ax)
ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
label_axes(ax)
ax.set_title(
    f"beta (volume bias ratio) — mean flow match in Ks x cv space\n"
    f"{EVENT_LABEL}  |  White dashed = beta=1 (perfect)  |  "
    f">1 = over-predict, <1 = under-predict",
    fontsize=11
)
fig.tight_layout()
save_fig(fig, "fig5_beta_contour.png")


# -----------------------------------------------------------------------
# FIGURE 6: NSE contour
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

cf = ax.contourf(KS_GRID, CV_GRID, NSE_SURF, levels=20, cmap='RdYlGn',
                 vmin=-1, vmax=1)
fig.colorbar(cf, ax=ax, label='NSE')

cl = ax.contour(KS_GRID, CV_GRID, NSE_SURF, levels=10,
                colors='black', linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%.2f", fontsize=7, colors='black')

# NSE=0 line: model performs no better than mean
try:
    cs0 = ax.contour(KS_GRID, CV_GRID, NSE_SURF, levels=[0],
                     colors=['gray'], linewidths=1.5, linestyles=':', zorder=6)
    ax.clabel(cs0, fmt="NSE=0", fontsize=8, colors='gray')
except Exception:
    pass

ax.scatter(ks_pts, cv_pts, c=nse_vals, cmap='RdYlGn', vmin=-1, vmax=1,
           **SCATTER_KW)

plot_pbias_zeroline(ax)
plot_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
label_axes(ax)
ax.set_title(
    f"NSE — Ks x cv joint sensitivity\n"
    f"{EVENT_LABEL}",
    fontsize=12
)
fig.tight_layout()
save_fig(fig, "fig6_nse_contour.png")


# -----------------------------------------------------------------------
# FIGURE 7: All six metrics in one 2x3 panel
# Summary figure — most useful for presentations and reports
# -----------------------------------------------------------------------

# Map each surface to its DataFrame column name for scatter coloring
SURF_TO_COL = {
    id(KGE_SURF):   "kge",
    id(NSE_SURF):   "nse",
    id(PBIAS_SURF): "pbias_pct",
    id(R_SURF):     "kge_r",
    id(ALPHA_SURF): "kge_alpha",
    id(BETA_SURF):  "kge_beta",
}

# Each entry: (surface, title, cmap, norm, colorbar_label, overlay_pbias_zero)
panel_configs = [
    (KGE_SURF,   "KGE",
     'RdYlGn', mcolors.Normalize(max(-1, kge_vals.min()), 1),
     'KGE', True),

    (NSE_SURF,   "NSE",
     'RdYlGn', mcolors.Normalize(-1, 1),
     'NSE', True),

    (PBIAS_SURF, "PBIAS (%)",
     'RdBu_r', mcolors.TwoSlopeNorm(vmin=-pbias_abs_max, vcenter=0, vmax=pbias_abs_max),
     'PBIAS (%)', True),

    (R_SURF,     "r (correlation)",
     'Blues', mcolors.Normalize(0, 1),
     'r', False),

    (ALPHA_SURF, "alpha (flashiness)",
     'RdYlGn', mcolors.TwoSlopeNorm(vmin=1 - alpha_dev, vcenter=1, vmax=1 + alpha_dev),
     'alpha (perfect=1)', False),

    (BETA_SURF,  "beta (volume bias)",
     'RdYlGn', mcolors.TwoSlopeNorm(vmin=1 - beta_dev, vcenter=1, vmax=1 + beta_dev),
     'beta (perfect=1)', False),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Joint Ks x cv sensitivity — all metrics\n"
    f"{EVENT_LABEL}  |  White dashed = PBIAS zero-crossing  |  star = best KGE run",
    fontsize=13
)

for ax, (surf, title, cmap, norm, cbar_label, do_pbias_zero) in zip(
        axes.flat, panel_configs):

    cf = ax.contourf(KS_GRID, CV_GRID, surf, levels=20, cmap=cmap, norm=norm)
    fig.colorbar(cf, ax=ax, label=cbar_label, shrink=0.85)

    # Thin reference contour lines
    try:
        ax.contour(KS_GRID, CV_GRID, surf, levels=8,
                   colors='black', linewidths=0.4, alpha=0.3)
    except Exception:
        pass

    # Scatter the LHS sample points colored by the same metric
    col = SURF_TO_COL[id(surf)]
    ax.scatter(ks_pts, cv_pts, c=df[col].values, cmap=cmap, norm=norm,
               **SCATTER_KW)

    # PBIAS zero-crossing overlay on relevant panels
    if do_pbias_zero:
        try:
            ax.contour(KS_GRID, CV_GRID, PBIAS_SURF, levels=[0],
                       colors=['white'], linewidths=1.8, linestyles='--', zorder=6)
        except Exception:
            pass

    # Best run star on every panel
    ax.scatter([best_ks], [best_cv], **BEST_KW)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(XLABEL, fontsize=9)
    ax.set_ylabel(YLABEL, fontsize=9)

fig.tight_layout()
save_fig(fig, "fig7_all_metrics_panel.png")

print(f"\nAll figures saved to:\n  {plot_dir}")
