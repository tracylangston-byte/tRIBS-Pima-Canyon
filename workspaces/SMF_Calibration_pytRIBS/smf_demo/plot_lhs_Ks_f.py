"""
plot_lhs_Ks_f.py
================
Generates contour plots from the joint Ks_mult × f_RS_abs LHS sweep.
Run after run_lhs_sweep.py completes.

Usage (run from the smf_demo directory):
    python plot_lhs_Ks_f.py

Produces 7 figures saved to:
    calibration_work/03_comparisons/sensitivity_plots/lhs_Ks_f/

Figure list:
    fig1_kge_contour.png          — KGE contour + PBIAS zero-crossing overlay
    fig2_pbias_contour.png        — PBIAS contour with zero-crossing line
    fig3_r_contour.png            — Correlation coefficient r (timing/shape)
    fig4_alpha_contour.png        — α variability ratio (flashiness)
    fig5_beta_contour.png         — β bias ratio (volume)
    fig6_nse_contour.png          — NSE (for Luke comparison)
    fig7_all_metrics_panel.png    — All six metrics in one 2×3 panel figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.interpolate import griddata

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / "lhs_Ks_f"
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------
results_path = summary_dir / "lhs_results_Ks_f.csv"
if not results_path.exists():
    raise FileNotFoundError(f"LHS results not found: {results_path}\n"
                            "Run run_lhs_sweep.py first.")

df = pd.read_csv(results_path)
print(f"Loaded {len(df)} LHS runs from {results_path.name}")

# Verify required columns exist
required = ["Ks_mult", "f_RS_abs", "kge", "nse", "pbias_pct",
            "kge_r", "kge_alpha", "kge_beta"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results CSV: {missing}")

# Drop any rows with NaN in key columns
df = df.dropna(subset=required).reset_index(drop=True)
print(f"  {len(df)} runs after dropping NaN rows")

# Extract arrays
ks   = df["Ks_mult"].values
f    = df["f_RS_abs"].values
kge  = df["kge"].values
nse  = df["nse"].values
pbias = df["pbias_pct"].values
r    = df["kge_r"].values
alpha = df["kge_alpha"].values
beta  = df["kge_beta"].values

# Best run by KGE
best_idx = np.argmax(kge)
best_ks  = ks[best_idx]
best_f   = f[best_idx]
best_kge = kge[best_idx]

print(f"\n  Best run: Ks={best_ks:.2f}×  f={best_f:.4f} mm⁻¹  KGE={best_kge:.3f}")
print(f"  KGE range: {kge.min():.3f} – {kge.max():.3f}")
print(f"  PBIAS range: {pbias.min():.1f}% – {pbias.max():.1f}%")

# -----------------------------------------------------------------------
# INTERPOLATION GRID
# We use scipy griddata to interpolate the scattered LHS points onto a
# regular grid for smooth contour plotting.
# -----------------------------------------------------------------------
n_grid = 200  # grid resolution — increase for smoother contours
ks_grid  = np.linspace(ks.min(),  ks.max(),  n_grid)
f_grid   = np.linspace(f.min(),   f.max(),   n_grid)
KS, F    = np.meshgrid(ks_grid, f_grid)
points   = np.column_stack([ks, f])

def interp(values):
    """Interpolate scattered values onto the regular grid."""
    return griddata(points, values, (KS, F), method='cubic')

KGE_grid   = interp(kge)
NSE_grid   = interp(nse)
PBIAS_grid = interp(pbias)
R_grid     = interp(r)
ALPHA_grid = interp(alpha)
BETA_grid  = interp(beta)

# -----------------------------------------------------------------------
# SHARED STYLE
# -----------------------------------------------------------------------
SCATTER_KW  = dict(s=18, edgecolors='white', linewidths=0.4, alpha=0.7, zorder=5)
BEST_KW     = dict(s=120, marker='*', color='white', edgecolors='black',
                   linewidths=1.2, zorder=10)
XLABEL      = "Ks multiplier"
YLABEL      = "RS soil f (1/mm)"
EVENT_LABEL = "SMF Aug 12, 2014"

def add_axes_labels(ax):
    ax.set_xlabel(XLABEL, fontsize=11)
    ax.set_ylabel(YLABEL, fontsize=11)

def add_best_star(ax):
    ax.scatter([best_ks], [best_f], **BEST_KW, label=f"Best KGE={best_kge:.3f}")

def add_pbias_zeroline(ax):
    """Overlay the PBIAS zero-crossing contour line."""
    try:
        cs = ax.contour(KS, F, PBIAS_grid, levels=[0],
                        colors=['white'], linewidths=2.0, linestyles='--', zorder=6)
        ax.clabel(cs, fmt="PBIAS=0", fontsize=8, colors='white')
    except Exception:
        pass  # Skip if contour can't be drawn (e.g. zero not in range)

def save(fig, name):
    path = plot_dir / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path.name}")
    plt.close(fig)


# -----------------------------------------------------------------------
# FIGURE 1: KGE contour + PBIAS zero-crossing overlay
# This is the headline figure — Luke comparison + PBIAS equifinality line
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Contour fill
cf = ax.contourf(KS, F, KGE_grid, levels=20, cmap='RdYlGn',
                 vmin=max(-1, kge.min()), vmax=1)
cbar = fig.colorbar(cf, ax=ax, label='KGE')

# Contour lines for reference
cl = ax.contour(KS, F, KGE_grid, levels=10, colors='black',
                linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%.2f", fontsize=7, colors='black')

# Scatter points colored by KGE
sc = ax.scatter(ks, f, c=kge, cmap='RdYlGn',
                vmin=max(-1, kge.min()), vmax=1, **SCATTER_KW)

# PBIAS zero-crossing line
add_pbias_zeroline(ax)

# Best run star
add_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
add_axes_labels(ax)
ax.set_title(f"KGE — Ks × f joint sensitivity\n"
             f"{EVENT_LABEL}  |  White dashed line = PBIAS zero-crossing", fontsize=12)
fig.tight_layout()
save(fig, "fig1_kge_contour.png")


# -----------------------------------------------------------------------
# FIGURE 2: PBIAS contour
# Diverging colormap centered on zero — shows equifinality band clearly
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

pbias_abs_max = max(abs(pbias.min()), abs(pbias.max()))
cf = ax.contourf(KS, F, PBIAS_grid, levels=20,
                 cmap='RdBu_r',
                 norm=mcolors.TwoSlopeNorm(vmin=-pbias_abs_max, vcenter=0,
                                           vmax=pbias_abs_max))
cbar = fig.colorbar(cf, ax=ax, label='PBIAS (%)')

cl = ax.contour(KS, F, PBIAS_grid, levels=10, colors='black',
                linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%+.0f%%", fontsize=7, colors='black')

sc = ax.scatter(ks, f, c=pbias, cmap='RdBu_r',
                norm=mcolors.TwoSlopeNorm(vmin=-pbias_abs_max, vcenter=0,
                                          vmax=pbias_abs_max),
                **SCATTER_KW)

# Zero-crossing line — this is the key feature
add_pbias_zeroline(ax)
add_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
add_axes_labels(ax)
ax.set_title(f"PBIAS (%) — volume bias in Ks × f space\n"
             f"{EVENT_LABEL}  |  White dashed = zero bias line  |  "
             f"Blue = under-predict, Red = over-predict", fontsize=11)
fig.tight_layout()
save(fig, "fig2_pbias_contour.png")


# -----------------------------------------------------------------------
# FIGURE 3: r (correlation) contour
# Expect this to be essentially flat — proves timing is structural
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

cf = ax.contourf(KS, F, R_grid, levels=20, cmap='Blues', vmin=0, vmax=1)
cbar = fig.colorbar(cf, ax=ax, label='r (correlation coefficient)')

cl = ax.contour(KS, F, R_grid, levels=10, colors='black',
                linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%.2f", fontsize=7, colors='black')

sc = ax.scatter(ks, f, c=r, cmap='Blues', vmin=0, vmax=1, **SCATTER_KW)

add_pbias_zeroline(ax)
add_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
add_axes_labels(ax)
ax.set_title(f"r (KGE correlation component) — timing and shape\n"
             f"{EVENT_LABEL}  |  Flat surface = timing controlled by forcing, not parameters", fontsize=11)
fig.tight_layout()
save(fig, "fig3_r_contour.png")


# -----------------------------------------------------------------------
# FIGURE 4: α (variability ratio / flashiness) contour
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

alpha_abs_max = max(abs(alpha.min() - 1), abs(alpha.max() - 1))
cf = ax.contourf(KS, F, ALPHA_grid, levels=20, cmap='RdYlGn',
                 norm=mcolors.TwoSlopeNorm(vmin=1 - alpha_abs_max, vcenter=1,
                                           vmax=1 + alpha_abs_max))
cbar = fig.colorbar(cf, ax=ax, label='α (variability ratio)  perfect = 1.0')

cl = ax.contour(KS, F, ALPHA_grid, levels=[1.0], colors='white',
                linewidths=2.0, linestyles='--', zorder=6)
ax.clabel(cl, fmt="α=1", fontsize=8, colors='white')

sc = ax.scatter(ks, f, c=alpha, cmap='RdYlGn',
                norm=mcolors.TwoSlopeNorm(vmin=1 - alpha_abs_max, vcenter=1,
                                          vmax=1 + alpha_abs_max),
                **SCATTER_KW)

add_best_star(ax)
ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
add_axes_labels(ax)
ax.set_title(f"α (flashiness ratio) — variability match in Ks × f space\n"
             f"{EVENT_LABEL}  |  White dashed = α=1 (perfect)  |  "
             f">1 = too flashy, <1 = too damped", fontsize=11)
fig.tight_layout()
save(fig, "fig4_alpha_contour.png")


# -----------------------------------------------------------------------
# FIGURE 5: β (bias ratio / volume) contour
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

beta_abs_max = max(abs(beta.min() - 1), abs(beta.max() - 1))
cf = ax.contourf(KS, F, BETA_grid, levels=20, cmap='RdYlGn',
                 norm=mcolors.TwoSlopeNorm(vmin=1 - beta_abs_max, vcenter=1,
                                           vmax=1 + beta_abs_max))
cbar = fig.colorbar(cf, ax=ax, label='β (bias ratio)  perfect = 1.0')

cl = ax.contour(KS, F, BETA_grid, levels=[1.0], colors='white',
                linewidths=2.0, linestyles='--', zorder=6)
ax.clabel(cl, fmt="β=1", fontsize=8, colors='white')

sc = ax.scatter(ks, f, c=beta, cmap='RdYlGn',
                norm=mcolors.TwoSlopeNorm(vmin=1 - beta_abs_max, vcenter=1,
                                          vmax=1 + beta_abs_max),
                **SCATTER_KW)

add_best_star(ax)
ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
add_axes_labels(ax)
ax.set_title(f"β (volume bias ratio) — mean flow match in Ks × f space\n"
             f"{EVENT_LABEL}  |  White dashed = β=1 (perfect)  |  "
             f">1 = over-predict, <1 = under-predict", fontsize=11)
fig.tight_layout()
save(fig, "fig5_beta_contour.png")


# -----------------------------------------------------------------------
# FIGURE 6: NSE contour (for Luke comparison)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

cf = ax.contourf(KS, F, NSE_grid, levels=20, cmap='RdYlGn', vmin=-1, vmax=1)
cbar = fig.colorbar(cf, ax=ax, label='NSE')

cl = ax.contour(KS, F, NSE_grid, levels=10, colors='black',
                linewidths=0.5, alpha=0.4)
ax.clabel(cl, fmt="%.2f", fontsize=7, colors='black')

# NSE = 0 line (model no better than mean)
try:
    cs0 = ax.contour(KS, F, NSE_grid, levels=[0],
                     colors=['gray'], linewidths=1.5, linestyles=':', zorder=6)
    ax.clabel(cs0, fmt="NSE=0", fontsize=8, colors='gray')
except Exception:
    pass

sc = ax.scatter(ks, f, c=nse, cmap='RdYlGn', vmin=-1, vmax=1, **SCATTER_KW)

add_pbias_zeroline(ax)
add_best_star(ax)

ax.legend(fontsize=9, loc='upper right', facecolor='white', framealpha=0.8)
add_axes_labels(ax)
ax.set_title(f"NSE — Ks × f joint sensitivity\n"
             f"{EVENT_LABEL}  |  For comparison with Luke (2026) LHS results", fontsize=12)
fig.tight_layout()
save(fig, "fig6_nse_contour.png")


# -----------------------------------------------------------------------
# FIGURE 7: All six metrics in one 2×3 panel
# The summary figure — most useful for presentations
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f"Joint Ks × f sensitivity — all metrics\n"
             f"{EVENT_LABEL}  |  White dashed = PBIAS zero-crossing  |  "
             f"★ = best KGE run", fontsize=13)

panel_configs = [
    (axes[0,0], KGE_grid,   "KGE",             'RdYlGn', mcolors.Normalize(-1, 1),                      'KGE',          True),
    (axes[0,1], NSE_grid,   "NSE",             'RdYlGn', mcolors.Normalize(-1, 1),                      'NSE',          True),
    (axes[0,2], PBIAS_grid, "PBIAS (%)",       'RdBu_r', mcolors.Normalize(-pbias_abs_max,
                                                                             pbias_abs_max),              'PBIAS (%)',    True),
    (axes[1,0], R_grid,     "r (correlation)", 'Blues',  mcolors.Normalize(0, 1),                        'r',            False),
    (axes[1,1], ALPHA_grid, "α (flashiness)",  'RdYlGn', mcolors.Normalize(alpha.min(), alpha.max()),    'α (perfect=1)',False),
    (axes[1,2], BETA_grid,  "β (volume bias)", 'RdYlGn', mcolors.Normalize(beta.min(),  beta.max()),     'β (perfect=1)',False),
]

for ax, grid, title, cmap, norm, cbar_label, do_pbias in panel_configs:
    cf = ax.contourf(KS, F, grid, levels=20, cmap=cmap, norm=norm)
    fig.colorbar(cf, ax=ax, label=cbar_label, shrink=0.85)

    # Thin contour lines
    try:
        cl = ax.contour(KS, F, grid, levels=8, colors='black',
                        linewidths=0.4, alpha=0.3)
    except Exception:
        pass

    # Map grid to DataFrame column for scatter coloring
    grid_to_col = {
        id(KGE_grid):   "kge",
        id(NSE_grid):   "nse",
        id(PBIAS_grid): "pbias_pct",
        id(R_grid):     "kge_r",
        id(ALPHA_grid): "kge_alpha",
        id(BETA_grid):  "kge_beta",
    }
    scatter_col = grid_to_col.get(id(grid), "kge")
    sc = ax.scatter(ks, f, c=df[scatter_col].values,
                    cmap=cmap, norm=norm, **SCATTER_KW)

    if do_pbias:
        try:
            cs = ax.contour(KS, F, PBIAS_grid, levels=[0],
                            colors=['white'], linewidths=1.8, linestyles='--', zorder=6)
        except Exception:
            pass

    ax.scatter([best_ks], [best_f], **BEST_KW)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(XLABEL, fontsize=9)
    ax.set_ylabel(YLABEL, fontsize=9)

fig.tight_layout()
save(fig, "fig7_all_metrics_panel.png")

print(f"\nAll figures saved to:\n  {plot_dir}")
