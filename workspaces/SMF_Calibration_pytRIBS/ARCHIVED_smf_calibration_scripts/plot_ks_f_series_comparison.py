"""
plot_ks_f_series_comparison.py
================================
Direct side-by-side comparison of two Ks_mult x f_RS_abs synthetic LHS
sweeps (e.g. a truth-reset before/after pair like Series 100 vs Series
97log). Companion to plot_pearson_comparison.py, but built for the Ks x f
contour-surface sweep design (plot_lhs_Ks_f_*.py family) rather than the
cv/r/n anchor correlation-matrix design -- kept as its own script rather
than folded into plot_pearson_comparison.py since the two compare
fundamentally different data shapes (interpolated 2D contour surfaces vs.
a parameter x metric Pearson matrix) and share no plotting code.

Both series are interpolated independently (same normalized-coordinate
cubic-interpolation fix as plot_lhs_Ks_f_100.py / plot_lhs_Ks_f_97log.py)
so the two panels use identical methodology and can be compared fairly.

To adapt for a different pair of Ks x f series, edit the SERIES block
only.

Usage (run from the smf_demo directory):
    python plot_ks_f_series_comparison.py

Produces, saved to:
    calibration_work/03_comparisons/sensitivity_plots/Comparisons/
        fig_compare_kge_alpha_{A_label}_vs_{B_label}.png
        fig_compare_swoosh_overlay_{A_label}_vs_{B_label}.png
        series_comparison_summary_{A_label}_vs_{B_label}.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# =======================================================================
# CONFIG -- edit this block to change which series are compared
# =======================================================================
SERIES = {
    "A": {
        "label": "Series 100",
        "csv":   "lhs_results_synth_Ks_f_100.csv",
        "truth_ks": 7.0,
        "truth_f":  0.012,
        "color": "#2a9d8f",
    },
    "B": {
        "label": "Series 97log",
        "csv":   "lhs_results_synth_Ks_f_97log.csv",
        "truth_ks": 8.5,
        "truth_f":  0.020,
        "color": "#e76f51",
    },
}

N_GRID = 200
KGE_CLIP = (-0.3, 1.0)          # colorbar clip, extend='min' -- matches
                                 # plot_lhs_Ks_f_100.py / _97log.py convention
COMPARISON_SUBDIR = "Comparisons"

# =======================================================================
# PATHS
# =======================================================================
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / COMPARISON_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)

tag = f"{SERIES['A']['label'].replace(' ', '')}_vs_{SERIES['B']['label'].replace(' ', '')}"

# =======================================================================
# LOAD + INTERPOLATE EACH SERIES INDEPENDENTLY
# =======================================================================
REQUIRED = ["Ks_mult", "f_RS_abs", "kge", "pbias_pct", "kge_alpha", "kge_beta", "kge_r", "nse"]

data = {}
for key, cfg in SERIES.items():
    path = summary_dir / cfg["csv"]
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)
    print(f"  Loaded {len(df)} rows from {path.name} ({cfg['label']})")

    ks_pts = df["Ks_mult"].values
    f_pts  = df["f_RS_abs"].values
    kge_vals   = df["kge"].values
    alpha_vals = df["kge_alpha"].values
    pbias_vals = df["pbias_pct"].values

    best_idx = np.argmax(kge_vals)

    ks_grid_1d = np.linspace(ks_pts.min(), ks_pts.max(), N_GRID)
    f_log_1d   = np.logspace(np.log10(f_pts.min()), np.log10(f_pts.max()), N_GRID)
    KS_GRID, F_GRID = np.meshgrid(ks_grid_1d, f_log_1d)

    def norm_ks(ks, ks_pts=ks_pts):
        return (ks - ks_pts.min()) / (ks_pts.max() - ks_pts.min())

    def norm_logf(f, f_pts=f_pts):
        lo, hi = np.log10(f_pts.min()), np.log10(f_pts.max())
        return (np.log10(f) - lo) / (hi - lo)

    points_norm = np.column_stack([norm_ks(ks_pts), norm_logf(f_pts)])
    KS_NORM = norm_ks(KS_GRID)
    F_NORM  = norm_logf(F_GRID)

    kge_surf   = griddata(points_norm, kge_vals,   (KS_NORM, F_NORM), method="cubic")
    alpha_surf = griddata(points_norm, alpha_vals, (KS_NORM, F_NORM), method="cubic")
    pbias_surf = griddata(points_norm, pbias_vals, (KS_NORM, F_NORM), method="cubic")

    r_ks, _ = pearsonr(ks_pts, kge_vals)
    r_f, _  = pearsonr(f_pts, kge_vals)

    good90 = df[df.kge >= 0.9]
    good80 = df[df.kge >= 0.8]
    top20  = df.nlargest(20, "kge")
    close_alpha = df[(df.kge_alpha >= 0.9) & (df.kge_alpha <= 1.1)]

    valid = ~np.isnan(kge_surf)
    area_frac_90 = np.sum((kge_surf >= 0.9) & valid) / np.sum(valid)
    area_frac_80 = np.sum((kge_surf >= 0.8) & valid) / np.sum(valid)

    data[key] = dict(
        df=df, ks_pts=ks_pts, f_pts=f_pts, kge_vals=kge_vals, alpha_vals=alpha_vals,
        pbias_vals=pbias_vals, best_idx=best_idx, KS_GRID=KS_GRID, F_GRID=F_GRID,
        kge_surf=kge_surf, alpha_surf=alpha_surf, pbias_surf=pbias_surf,
        r_ks=r_ks, r_f=r_f, good90=good90, good80=good80, top20=top20,
        close_alpha=close_alpha, area_frac_90=area_frac_90, area_frac_80=area_frac_80,
    )
    print(f"    best KGE={kge_vals[best_idx]:.4f}  "
          f"KGE>=0.9 area frac={area_frac_90:.3f}  KGE>=0.8 area frac={area_frac_80:.3f}")

# =======================================================================
# FIGURE 1: KGE (top row) + alpha (bottom row), Series A | Series B
# =======================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

kge_norm = mcolors.Normalize(vmin=KGE_CLIP[0], vmax=KGE_CLIP[1])
alpha_all = np.concatenate([data["A"]["alpha_vals"], data["B"]["alpha_vals"]])
alpha_dev = max(abs(alpha_all.min() - 1), abs(alpha_all.max() - 1))
alpha_norm = mcolors.TwoSlopeNorm(vmin=1 - alpha_dev, vcenter=1, vmax=1 + alpha_dev)

for col, key in enumerate(["A", "B"]):
    d = data[key]
    cfg = SERIES[key]

    ax = axes[0, col]
    cf = ax.contourf(d["KS_GRID"], d["F_GRID"], d["kge_surf"], levels=20,
                      cmap="RdYlGn", norm=kge_norm, extend="min")
    ax.contour(d["KS_GRID"], d["F_GRID"], d["pbias_surf"], levels=[0],
               colors="white", linewidths=2.0, linestyles="--")
    ax.scatter(d["ks_pts"], d["f_pts"], c=d["kge_vals"], cmap="RdYlGn", norm=kge_norm,
               s=14, edgecolors="white", linewidths=0.3, alpha=0.7)
    ax.scatter([d["ks_pts"][d["best_idx"]]], [d["f_pts"][d["best_idx"]]],
               s=140, marker="*", color="white", edgecolors="black", linewidths=1.3, zorder=10)
    ax.scatter([cfg["truth_ks"]], [cfg["truth_f"]], s=90, marker="X", color="cyan",
               edgecolors="black", linewidths=1.0, zorder=10)
    ax.set_yscale("log")
    ax.set_xlabel("Ks multiplier")
    ax.set_ylabel("f_RS_abs (log)")
    ax.set_title(f"KGE -- {cfg['label']}\n"
                 f"KGE\u22650.9 area frac={d['area_frac_90']:.2f}, n(KGE\u22650.9)={len(d['good90'])}",
                 fontsize=10)
    if col == 1:
        fig.colorbar(cf, ax=[axes[0, 0], axes[0, 1]], label="KGE (clipped)", shrink=0.85)

    ax = axes[1, col]
    cf2 = ax.contourf(d["KS_GRID"], d["F_GRID"], d["alpha_surf"], levels=20,
                       cmap="RdYlGn", norm=alpha_norm)
    ax.scatter(d["ks_pts"], d["f_pts"], c=d["alpha_vals"], cmap="RdYlGn", norm=alpha_norm,
               s=14, edgecolors="white", linewidths=0.3, alpha=0.7)
    ax.scatter([d["ks_pts"][d["best_idx"]]], [d["f_pts"][d["best_idx"]]],
               s=140, marker="*", color="white", edgecolors="black", linewidths=1.3, zorder=10)
    ax.scatter([cfg["truth_ks"]], [cfg["truth_f"]], s=90, marker="X", color="cyan",
               edgecolors="black", linewidths=1.0, zorder=10)
    ax.set_yscale("log")
    ax.set_xlabel("Ks multiplier")
    ax.set_ylabel("f_RS_abs (log)")
    top20 = d["top20"]
    ax.set_title(f"alpha (flashiness) -- {cfg['label']}\n"
                 f"top-20 KGE runs: alpha={top20.kge_alpha.mean():.3f}\u00b1{top20.kge_alpha.std():.3f}",
                 fontsize=10)
    if col == 1:
        fig.colorbar(cf2, ax=[axes[1, 0], axes[1, 1]], label="alpha (perfect=1)", shrink=0.85)

fig.suptitle(f"{SERIES['A']['label']} vs {SERIES['B']['label']} -- KGE and alpha (flashiness) in Ks\u00d7f space\n"
             "White dashed = PBIAS zero-crossing (equifinal valley)  |  star = best-KGE run  |  X = true parameter value",
             fontsize=13)
fig1_path = plot_dir / f"fig_compare_kge_alpha_{tag}.png"
fig.savefig(fig1_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig1_path.name}")

# =======================================================================
# FIGURE 2: PBIAS=0 swoosh overlay -- both series on one axis
# =======================================================================
fig, ax = plt.subplots(figsize=(9, 7))
for key in ["A", "B"]:
    d = data[key]
    cfg = SERIES[key]
    ax.contour(d["KS_GRID"], d["F_GRID"], d["pbias_surf"], levels=[0],
               colors=[cfg["color"]], linewidths=2.5)
    ax.plot([], [], color=cfg["color"], linewidth=2.5, label=f"{cfg['label']} (PBIAS=0)")
    ax.scatter([cfg["truth_ks"]], [cfg["truth_f"]], s=140, marker="X", color=cfg["color"],
               edgecolors="black", linewidths=1.3, zorder=10)

ax.set_yscale("log")
ax.set_xlabel("Ks multiplier")
ax.set_ylabel("f_RS_abs (log)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title(f"Equifinal valley (\"swoosh\") location: {SERIES['A']['label']} vs {SERIES['B']['label']}\n"
             "X markers = true parameter values used for each series' synthetic truth",
             fontsize=12)
fig.tight_layout()
fig2_path = plot_dir / f"fig_compare_swoosh_overlay_{tag}.png"
fig.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig2_path.name}")

# =======================================================================
# SUMMARY CSV
# =======================================================================
rows = []
for key in ["A", "B"]:
    d = data[key]
    df = d["df"]
    best = df.loc[df.kge.idxmax()]
    top20 = d["top20"]
    rows.append({
        "series": SERIES[key]["label"],
        "n_runs": len(df),
        "truth_Ks_mult": SERIES[key]["truth_ks"],
        "truth_f_RS_abs": SERIES[key]["truth_f"],
        "best_KGE": best.kge,
        "best_run_Ks_mult": best.Ks_mult,
        "best_run_f_RS_abs": best.f_RS_abs,
        "best_run_PBIAS_pct": best.pbias_pct,
        "best_run_alpha": best.kge_alpha,
        "best_run_beta": best.kge_beta,
        "Ks_distance_from_truth": abs(best.Ks_mult - SERIES[key]["truth_ks"]),
        "pearson_r_Ks_vs_KGE": d["r_ks"],
        "pearson_r_f_vs_KGE": d["r_f"],
        "n_KGE_ge_0.9": len(d["good90"]),
        "n_KGE_ge_0.8": len(d["good80"]),
        "pct_KGE_ge_0.9": 100 * len(d["good90"]) / len(df),
        "pct_KGE_ge_0.8": 100 * len(d["good80"]) / len(df),
        "grid_area_frac_KGE_ge_0.9": d["area_frac_90"],
        "grid_area_frac_KGE_ge_0.8": d["area_frac_80"],
        "top20_alpha_mean": top20.kge_alpha.mean(),
        "top20_alpha_std": top20.kge_alpha.std(),
        "top20_beta_mean": top20.kge_beta.mean(),
        "top20_beta_std": top20.kge_beta.std(),
        "top20_KGE_mean": top20.kge.mean(),
        "top20_KGE_min": top20.kge.min(),
        "n_alpha_close_to_1": len(d["close_alpha"]),
        "pct_alpha_close_to_1": 100 * len(d["close_alpha"]) / len(df),
        "mean_KGE_when_alpha_close_to_1": d["close_alpha"].kge.mean(),
        "KGE_min_overall": df.kge.min(),
        "NSE_min_overall": df.nse.min(),
        "PBIAS_min_overall": df.pbias_pct.min(),
        "PBIAS_max_overall": df.pbias_pct.max(),
    })
summary_df = pd.DataFrame(rows)
csv_path = plot_dir / f"series_comparison_summary_{tag}.csv"
summary_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path.name}")
print("\n" + summary_df.T.to_string())
