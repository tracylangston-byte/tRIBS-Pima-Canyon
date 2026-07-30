"""
plot_storm_magnitude_comparison_100.py
========================================
Pointwise comparison of the storm-magnitude sibling sweeps to Series 100:
storm080 (0.8x rain), 100_narrow (1.0x rain, matched bounds/n), and
storm125 (1.25x rain). Tests whether the Ks-f equifinality "swoosh"
shifts position and/or changes curvature with storm magnitude, per the
Handoff: Multi-Storm-Magnitude Test of the Ks-f Equifinality "Swoosh".

UPDATED (Handoff_KGE2012Transition_v2.md): reads the `_RESCORED.csv`
family (resampling-fix applied) and uses the Kling et al. (2012)
formulation throughout -- KGE_2012 (`kge_2012`) replaces KGE_2009 (`kge`)
as the composite, and gamma (`kge_gamma`) replaces alpha (`kge_alpha`) as
the plotted variability component. r and beta are unchanged between
formulations. Figure A's bottom row (previously alpha) now shows gamma.

UNLIKE plot_ks_f_series_comparison.py (which independently interpolates
two series with potentially different bounds/n and compares only the
resulting surfaces), this script assumes all available series share
IDENTICAL LHS bounds, n, and seed -- and exploits that to do a genuine
pointwise comparison (same (Ks,f) coordinates, different PBIAS/KGE) in
addition to the surface-level view. This is only valid if storm080,
100_narrow, and storm125 were all built with matching
LHS_PARAMS bounds (Ks_mult 3.0-9.5, f_RS_abs 0.004-0.05) and n=200,
seed=42, per the storm080/storm125 handoff.

IMPORTANT -- one input does not exist yet:
    lhs_results_synth_Ks_f_100_narrow.csv (the "100_narrow" sweep) has
    not been run as of writing this script. This script is written to
    degrade gracefully: it will run a 2-way comparison on whichever
    CSVs it finds (e.g. storm080 vs storm125 alone) and automatically
    upgrade to the full 3-way analysis (including joint-feasibility
    narrowing) once all three are present. If the 100_narrow sweep ends
    up using a different filename/label than assumed below, edit the
    SERIES config block.

Usage (run from the smf_demo directory):
    python plot_storm_magnitude_comparison_100.py

Produces, saved to:
    calibration_work/03_comparisons/summary_tables/
        storm_series_summary_{tag}.csv
        storm_pairwise_delta_summary_{tag}.csv
    calibration_work/03_comparisons/sensitivity_plots/Comparisons/
        fig_storm_kge_gamma_{tag}.png
        fig_storm_swoosh_overlay_{tag}.png
        fig_storm_delta_pbias_{tag}.png
        fig_storm_joint_feasibility_{tag}.png   (only if all 3 present)
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# =======================================================================
# CONFIG -- edit this block if filenames/labels/bounds differ
# =======================================================================
SERIES = {
    "080":  {"label": "storm080 (0.8x rain)",
             "csv": "lhs_results_synth_Ks_f_100_storm080_RESCORED.csv",
             "storm_scale": 0.80, "color": "#4c72b0"},
    "100n": {"label": "100_narrow (1.0x rain, matched bounds)",
             "csv": "lhs_results_synth_Ks_f_100_narrow_RESCORED.csv",
             "storm_scale": 1.00, "color": "#55a868"},
    "125":  {"label": "storm125 (1.25x rain)",
             "csv": "lhs_results_synth_Ks_f_100_storm125_RESCORED.csv",
             "storm_scale": 1.25, "color": "#c44e52"},
}

TRUTH_KS = 7.0      # identical across all three -- only forcing differs
TRUTH_F = 0.012
PBIAS_TOL = 2.0      # |PBIAS| <= this (%) counts as "feasible" / near-zero
N_GRID = 200
KGE_CLIP = (-0.3, 1.0)
COMPARISON_SUBDIR = "Comparisons"
# kge_2012/kge_gamma (Kling et al. 2012) are now required; kge/kge_alpha
# (2009) are no longer required for any figure but are still read for the
# summary-CSV audit column if present.
REQUIRED = ["Ks_mult", "f_RS_abs", "pbias_pct", "kge_2012", "kge_gamma", "kge_beta", "kge_r", "nse"]
KS_KEY_DECIMALS = 6
F_KEY_DECIMALS = 8
MIN_CONTOUR_PTS_FOR_CURVATURE_FIT = 8

# =======================================================================
# PATHS
# =======================================================================
script_dir = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir = project_root / "calibration_work"
summary_dir = calib_dir / "03_comparisons" / "summary_tables"
plot_dir = calib_dir / "03_comparisons" / "sensitivity_plots" / COMPARISON_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)

# =======================================================================
# LOAD WHICHEVER SERIES EXIST
# =======================================================================
available = {}
for key, cfg in SERIES.items():
    path = summary_dir / cfg["csv"]
    if not path.exists():
        print(f"  [skip] {cfg['label']}: {path.name} not found yet")
        continue
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        print(f"  [skip] {cfg['label']}: missing columns {missing_cols}")
        continue
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)
    df["_ks_key"] = df["Ks_mult"].round(KS_KEY_DECIMALS)
    df["_f_key"] = df["f_RS_abs"].round(F_KEY_DECIMALS)
    available[key] = df
    print(f"  Loaded {len(df)} rows: {cfg['label']} ({path.name})")

if len(available) < 2:
    sys.exit(
        f"\nOnly {len(available)} series found in {summary_dir} -- need at "
        f"least 2 (storm080 + storm125) for any comparison. Check filenames "
        f"in the SERIES config block against what's actually in "
        f"03_comparisons/summary_tables/."
    )

tag = "_".join(available.keys())
print(f"\nRunning comparison across: {list(available.keys())}  (tag={tag})")
if len(available) < 3:
    print("  NOTE: fewer than 3 series present -- joint-feasibility narrowing "
          "and the full 3-way summary will be skipped. Re-run once all three "
          "sweeps are complete.")

# =======================================================================
# SHARED GRID (intersection of each series' actual sampled range) --
# all series get interpolated onto THIS SAME grid so PBIAS/KGE surfaces
# are directly comparable/combinable (needed for joint-feasibility).
# =======================================================================
ks_lo = max(df["Ks_mult"].min() for df in available.values())
ks_hi = min(df["Ks_mult"].max() for df in available.values())
f_lo = max(df["f_RS_abs"].min() for df in available.values())
f_hi = min(df["f_RS_abs"].max() for df in available.values())
print(f"  Shared grid domain: Ks [{ks_lo:.3f}, {ks_hi:.3f}]  "
      f"f [{f_lo:.5f}, {f_hi:.5f}]")
if ks_hi <= ks_lo or f_hi <= f_lo:
    sys.exit("Series ranges don't overlap -- check that all sweeps used "
              "matching LHS_PARAMS bounds.")

ks_grid_1d = np.linspace(ks_lo, ks_hi, N_GRID)
f_log_1d = np.logspace(np.log10(f_lo), np.log10(f_hi), N_GRID)
KS_GRID, F_GRID = np.meshgrid(ks_grid_1d, f_log_1d)


def norm_ks(ks):
    return (ks - ks_lo) / (ks_hi - ks_lo)


def norm_logf(f):
    return (np.log10(f) - np.log10(f_lo)) / (np.log10(f_hi) - np.log10(f_lo))


KS_NORM = norm_ks(KS_GRID)
F_NORM = norm_logf(F_GRID)


def interpolate_surfaces(df):
    points_norm = np.column_stack([norm_ks(df["Ks_mult"].values),
                                    norm_logf(df["f_RS_abs"].values)])
    surfs = {}
    for col in ("pbias_pct", "kge_2012", "kge_gamma"):
        surfs[col] = griddata(points_norm, df[col].values, (KS_NORM, F_NORM), method="cubic")
    return surfs


def fit_swoosh_curvature(pbias_surf):
    """Extract the PBIAS=0 contour on the shared grid and fit
    Ks = a*log10(f)^2 + b*log10(f) + c to its longest branch. Returns
    (coeffs, r2, n_pts_used, n_segments) or (None, ...) if extraction fails.
    """
    fig_tmp, ax_tmp = plt.subplots()
    try:
        cs = ax_tmp.contour(KS_GRID, F_GRID, pbias_surf, levels=[0])
        segs = cs.allsegs[0]
    finally:
        plt.close(fig_tmp)
    if not segs:
        return None, np.nan, 0, 0
    seg = max(segs, key=len)
    if len(seg) < MIN_CONTOUR_PTS_FOR_CURVATURE_FIT:
        return None, np.nan, len(seg), len(segs)
    ks_vals = seg[:, 0]
    logf_vals = np.log10(seg[:, 1])
    coeffs = np.polyfit(logf_vals, ks_vals, 2)
    pred = np.polyval(coeffs, logf_vals)
    ss_res = np.sum((ks_vals - pred) ** 2)
    ss_tot = np.sum((ks_vals - ks_vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coeffs, r2, len(seg), len(segs)


data = {}
for key, df in available.items():
    surfs = interpolate_surfaces(df)
    curv_coeffs, curv_r2, curv_npts, n_segs = fit_swoosh_curvature(surfs["pbias_pct"])
    best_idx = df["kge_2012"].values.argmax()
    valid = ~np.isnan(surfs["pbias_pct"])
    feasible_mask = (np.abs(surfs["pbias_pct"]) <= PBIAS_TOL) & valid
    area_frac_feasible = feasible_mask.sum() / valid.sum() if valid.sum() else np.nan
    r_ks, _ = pearsonr(df["Ks_mult"], df["kge_2012"])
    r_f, _ = pearsonr(df["f_RS_abs"], df["kge_2012"])
    data[key] = dict(
        df=df, surfs=surfs, best_idx=best_idx,
        curv_coeffs=curv_coeffs, curv_r2=curv_r2, curv_npts=curv_npts, n_segs=n_segs,
        feasible_mask=feasible_mask, area_frac_feasible=area_frac_feasible,
        r_ks=r_ks, r_f=r_f,
    )
    curv_str = f"quad_coef={curv_coeffs[0]:.3f} (R2={curv_r2:.3f}, n={curv_npts}pts)" if curv_coeffs is not None else "curvature fit skipped (short/absent contour)"
    print(f"  {SERIES[key]['label']}: best KGE_2012={df['kge_2012'].values[best_idx]:.4f}  "
          f"area_frac |PBIAS|<={PBIAS_TOL}%={area_frac_feasible:.3f}  "
          f"swoosh branches={n_segs}  {curv_str}")

# =======================================================================
# POINTWISE MERGES (rounded Ks_mult/f_RS_abs as join key)
# =======================================================================
keyed = available  # already has _ks_key/_f_key from load loop

pairwise_merged = {}
for a, b in combinations(available.keys(), 2):
    cols_common = ["_ks_key", "_f_key", "pbias_pct", "kge_2012", "kge_gamma"]
    m = pd.merge(
        keyed[a][["_ks_key", "_f_key", "Ks_mult", "f_RS_abs"] + cols_common[2:]],
        keyed[b][cols_common],
        on=["_ks_key", "_f_key"], suffixes=(f"_{a}", f"_{b}"),
    )
    pairwise_merged[(a, b)] = m
    n_possible = min(len(keyed[a]), len(keyed[b]))
    pct = 100 * len(m) / n_possible if n_possible else np.nan
    print(f"  Pointwise match {a} vs {b}: {len(m)}/{n_possible} coordinates ({pct:.0f}%)")
    if pct < 80:
        print(f"    WARNING: <80% pointwise match for {a} vs {b} -- these sweeps may "
              f"NOT share identical seed/bounds/n draws. Pointwise delta analysis "
              f"below is only as trustworthy as this match rate; if it's low, treat "
              f"the surface-level (interpolated) comparison as primary instead.")

m3 = None
if len(available) == 3:
    keys3 = list(available.keys())
    m3 = keyed[keys3[0]][["_ks_key", "_f_key", "Ks_mult", "f_RS_abs"]].copy()
    for k in keys3:
        sub = keyed[k][["_ks_key", "_f_key", "pbias_pct", "kge_2012"]].rename(
            columns={"pbias_pct": f"pbias_{k}", "kge_2012": f"kge_{k}"})
        m3 = pd.merge(m3, sub, on=["_ks_key", "_f_key"], how="inner")
    print(f"  3-way pointwise match: {len(m3)} coordinates present in all three sweeps")

# =======================================================================
# PAIRWISE DELTA-PBIAS REGRESSION
# ΔPBIAS(Ks, logf) = c0 + c1*Ks + c2*log10(f) + c3*Ks*log10(f)
# c3 (interaction term) is the operational test for curvature change --
# a pure shift shows up in c1/c2, a curvature/shape change shows up in c3.
# =======================================================================
delta_results = {}
for (a, b), m in pairwise_merged.items():
    if len(m) < MIN_CONTOUR_PTS_FOR_CURVATURE_FIT:
        print(f"  [skip regression] {a} vs {b}: too few matched points ({len(m)})")
        continue
    delta = (m[f"pbias_pct_{b}"] - m[f"pbias_pct_{a}"]).values
    ks = m["Ks_mult"].values
    logf = np.log10(m["f_RS_abs"].values)
    X = np.column_stack([np.ones_like(ks), ks, logf, ks * logf])
    coeffs, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((delta - pred) ** 2)
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r_ks, _ = pearsonr(ks, delta)
    r_logf, _ = pearsonr(logf, delta)
    delta_results[(a, b)] = dict(
        n=len(m), intercept=coeffs[0], coef_Ks=coeffs[1], coef_logf=coeffs[2],
        coef_Ks_logf_interaction=coeffs[3], r2=r2,
        pearson_r_delta_vs_Ks=r_ks, pearson_r_delta_vs_logf=r_logf,
        mean_delta_pbias=delta.mean(), std_delta_pbias=delta.std(),
    )
    print(f"  Delta-PBIAS[{b} - {a}]: mean={delta.mean():+.2f}%  "
          f"interaction_coef={coeffs[3]:+.3f}  R2={r2:.3f}  (n={len(m)})")

# =======================================================================
# JOINT FEASIBILITY (3-way only)
# =======================================================================
joint_stats = None
if len(available) == 3:
    valid_all = np.ones_like(KS_GRID, dtype=bool)
    for key in available:
        valid_all &= ~np.isnan(data[key]["surfs"]["pbias_pct"])
    joint_feasible = np.ones_like(KS_GRID, dtype=bool)
    for key in available:
        joint_feasible &= data[key]["feasible_mask"]
    area_frac_joint = joint_feasible[valid_all].sum() / valid_all.sum() if valid_all.sum() else np.nan
    single_fracs = {key: data[key]["area_frac_feasible"] for key in available}
    joint_stats = dict(area_frac_joint_feasible=area_frac_joint, **{
        f"area_frac_feasible_{key}": v for key, v in single_fracs.items()
    })
    narrowing_ratio = area_frac_joint / min(single_fracs.values()) if min(single_fracs.values()) else np.nan
    print(f"\n  Joint feasibility (|PBIAS|<={PBIAS_TOL}% in ALL three storms): "
          f"{area_frac_joint:.3f} of grid area")
    print(f"  Narrowest single-storm feasible fraction: {min(single_fracs.values()):.3f}")
    print(f"  Narrowing ratio (joint / narrowest single): {narrowing_ratio:.3f}  "
          f"(<1.0 means multi-event calibration shrinks the equifinal region)")

# =======================================================================
# FIGURE A: KGE_2012 (top row) + gamma (bottom row), one column per series
# =======================================================================
n_cols = len(available)
fig, axes = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 11), squeeze=False)

kge_norm = mcolors.Normalize(vmin=KGE_CLIP[0], vmax=KGE_CLIP[1])
gamma_all = np.concatenate([data[k]["df"]["kge_gamma"].values for k in available])
gamma_dev = max(abs(gamma_all.min() - 1), abs(gamma_all.max() - 1))
gamma_norm = mcolors.TwoSlopeNorm(vmin=1 - gamma_dev, vcenter=1, vmax=1 + gamma_dev)

for col, key in enumerate(available):
    d = data[key]
    df = d["df"]
    cfg = SERIES[key]

    ax = axes[0, col]
    cf = ax.contourf(KS_GRID, F_GRID, d["surfs"]["kge_2012"], levels=20,
                      cmap="RdYlGn", norm=kge_norm, extend="min")
    ax.contour(KS_GRID, F_GRID, d["surfs"]["pbias_pct"], levels=[0],
               colors="white", linewidths=2.0, linestyles="--")
    ax.scatter(df["Ks_mult"], df["f_RS_abs"], c=df["kge_2012"], cmap="RdYlGn", norm=kge_norm,
               s=14, edgecolors="white", linewidths=0.3, alpha=0.7)
    ax.scatter([df["Ks_mult"].values[d["best_idx"]]], [df["f_RS_abs"].values[d["best_idx"]]],
               s=140, marker="*", color="white", edgecolors="black", linewidths=1.3, zorder=10)
    ax.scatter([TRUTH_KS], [TRUTH_F], s=90, marker="X", color="cyan",
               edgecolors="black", linewidths=1.0, zorder=10)
    ax.set_yscale("log")
    ax.set_xlabel("Ks multiplier")
    ax.set_ylabel("f_RS_abs (log)")
    ax.set_title(f"KGE' (2012) -- {cfg['label']}\n"
                 f"|PBIAS|<={PBIAS_TOL}% area frac={d['area_frac_feasible']:.2f}",
                 fontsize=10)
    if col == n_cols - 1:
        fig.colorbar(cf, ax=list(axes[0, :]), label="KGE' (clipped)", shrink=0.85)

    ax = axes[1, col]
    cf2 = ax.contourf(KS_GRID, F_GRID, d["surfs"]["kge_gamma"], levels=20,
                       cmap="RdYlGn", norm=gamma_norm)
    ax.scatter(df["Ks_mult"], df["f_RS_abs"], c=df["kge_gamma"], cmap="RdYlGn", norm=gamma_norm,
               s=14, edgecolors="white", linewidths=0.3, alpha=0.7)
    ax.scatter([df["Ks_mult"].values[d["best_idx"]]], [df["f_RS_abs"].values[d["best_idx"]]],
               s=140, marker="*", color="white", edgecolors="black", linewidths=1.3, zorder=10)
    ax.scatter([TRUTH_KS], [TRUTH_F], s=90, marker="X", color="cyan",
               edgecolors="black", linewidths=1.0, zorder=10)
    ax.set_yscale("log")
    ax.set_xlabel("Ks multiplier")
    ax.set_ylabel("f_RS_abs (log)")
    top20 = df.nlargest(20, "kge_2012")
    ax.set_title(f"gamma (2012 variability) -- {cfg['label']}\n"
                 f"top-20 KGE' runs: gamma={top20.kge_gamma.mean():.3f}\u00b1{top20.kge_gamma.std():.3f}",
                 fontsize=10)
    if col == n_cols - 1:
        fig.colorbar(cf2, ax=list(axes[1, :]), label="gamma (perfect=1)", shrink=0.85)

fig.suptitle(f"Storm-magnitude comparison ({', '.join(SERIES[k]['label'] for k in available)})\n"
             "White dashed = PBIAS zero-crossing (equifinal valley)  |  star = best-KGE' run  |  X = true parameter value (shared across all storms)",
             fontsize=13)
figA_path = plot_dir / f"fig_storm_kge_gamma_{tag}.png"
fig.savefig(figA_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {figA_path.name}")

# =======================================================================
# FIGURE B: PBIAS=0 swoosh overlay -- all available series on one axis
# =======================================================================
fig, ax = plt.subplots(figsize=(9, 7))
for key in available:
    d = data[key]
    cfg = SERIES[key]
    ax.contour(KS_GRID, F_GRID, d["surfs"]["pbias_pct"], levels=[0],
               colors=[cfg["color"]], linewidths=2.5)
    ax.plot([], [], color=cfg["color"], linewidth=2.5, label=f"{cfg['label']} (PBIAS=0)")
ax.scatter([TRUTH_KS], [TRUTH_F], s=140, marker="X", color="black",
           edgecolors="white", linewidths=1.3, zorder=10, label="true parameters (shared)")
ax.set_yscale("log")
ax.set_xlabel("Ks multiplier")
ax.set_ylabel("f_RS_abs (log)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Equifinal valley (\"swoosh\") location by storm magnitude\n"
             "Position shift -> forcing-magnitude bias; shape/curvature change -> f governs depth-decay of Ks",
             fontsize=11)
fig.tight_layout()
figB_path = plot_dir / f"fig_storm_swoosh_overlay_{tag}.png"
fig.savefig(figB_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {figB_path.name}")

# =======================================================================
# FIGURE C: pointwise Delta-PBIAS scatter, one panel per pair
# =======================================================================
pairs = list(pairwise_merged.keys())
if pairs:
    fig, axes_c = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6), squeeze=False)
    axes_c = axes_c[0]
    for i, (a, b) in enumerate(pairs):
        m = pairwise_merged[(a, b)]
        delta = m[f"pbias_pct_{b}"] - m[f"pbias_pct_{a}"]
        dev = max(abs(delta.min()), abs(delta.max())) or 1.0
        norm_d = mcolors.TwoSlopeNorm(vmin=-dev, vcenter=0, vmax=dev)
        ax = axes_c[i]
        sc = ax.scatter(m["Ks_mult"], m["f_RS_abs"], c=delta, cmap="RdBu_r", norm=norm_d,
                         s=30, edgecolors="black", linewidths=0.3)
        ax.scatter([TRUTH_KS], [TRUTH_F], s=140, marker="X", color="black",
                   edgecolors="white", linewidths=1.3, zorder=10)
        ax.set_yscale("log")
        ax.set_xlabel("Ks multiplier")
        ax.set_ylabel("f_RS_abs (log)")
        dr = delta_results.get((a, b))
        subtitle = (f"n={dr['n']}  R2(Ks,logf,interact)={dr['r2']:.2f}  "
                    f"interaction_coef={dr['coef_Ks_logf_interaction']:+.3f}") if dr else "regression skipped"
        ax.set_title(f"\u0394PBIAS = PBIAS[{SERIES[b]['label']}] \u2212 PBIAS[{SERIES[a]['label']}]\n{subtitle}",
                     fontsize=10)
        fig.colorbar(sc, ax=ax, label="\u0394PBIAS (%)", shrink=0.85)
    fig.suptitle("Pointwise PBIAS divergence between storm magnitudes (same Ks,f draws)", fontsize=12)
    fig.tight_layout()
    figC_path = plot_dir / f"fig_storm_delta_pbias_{tag}.png"
    fig.savefig(figC_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {figC_path.name}")

# =======================================================================
# FIGURE D: joint feasibility overlay (3-way only)
# =======================================================================
if len(available) == 3:
    fig, ax = plt.subplots(figsize=(8, 7))
    legend_handles = []
    for key in available:
        cfg = SERIES[key]
        mask = data[key]["feasible_mask"].astype(float)
        ax.contourf(KS_GRID, F_GRID, mask, levels=[0.5, 1.5], colors=[cfg["color"]], alpha=0.25)
        ax.contour(KS_GRID, F_GRID, mask, levels=[0.5], colors=[cfg["color"]], linewidths=1.2)
        legend_handles.append(plt.Line2D([0], [0], color=cfg["color"], lw=6, alpha=0.5,
                                          label=f"{cfg['label']} feasible (frac={data[key]['area_frac_feasible']:.2f})"))
    joint_mask = np.ones_like(KS_GRID, dtype=bool)
    for key in available:
        joint_mask &= data[key]["feasible_mask"]
    ax.contourf(KS_GRID, F_GRID, joint_mask.astype(float), levels=[0.5, 1.5],
                colors=["black"], alpha=0.55, hatches=["//"])
    ax.contour(KS_GRID, F_GRID, joint_mask.astype(float), levels=[0.5], colors="black", linewidths=2.0)
    legend_handles.append(plt.Line2D([0], [0], color="black", lw=6, alpha=0.6,
                                      label=f"joint (all 3) feasible (frac={joint_stats['area_frac_joint_feasible']:.2f})"))
    ax.scatter([TRUTH_KS], [TRUTH_F], s=160, marker="X", color="white",
               edgecolors="black", linewidths=1.5, zorder=10)
    ax.set_yscale("log")
    ax.set_xlabel("Ks multiplier")
    ax.set_ylabel("f_RS_abs (log)")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_title(f"Joint feasibility narrowing under multi-event calibration\n"
                 f"(|PBIAS| \u2264 {PBIAS_TOL}% threshold)", fontsize=11)
    fig.tight_layout()
    figD_path = plot_dir / f"fig_storm_joint_feasibility_{tag}.png"
    fig.savefig(figD_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {figD_path.name}")

# =======================================================================
# SUMMARY CSVs
# =======================================================================
rows = []
for key in available:
    d = data[key]
    df = d["df"]
    best = df.iloc[d["best_idx"]]
    curv = d["curv_coeffs"]
    row = {
        "series": SERIES[key]["label"],
        "storm_scale": SERIES[key]["storm_scale"],
        "n_runs": len(df),
        "truth_Ks_mult": TRUTH_KS,
        "truth_f_RS_abs": TRUTH_F,
        "best_KGE_2012": best["kge_2012"],
        "best_KGE_2009": best.get("kge", np.nan),  # retired formula, audit only
        "best_run_Ks_mult": best["Ks_mult"],
        "best_run_f_RS_abs": best["f_RS_abs"],
        "best_run_PBIAS_pct": best["pbias_pct"],
        "Ks_distance_from_truth": abs(best["Ks_mult"] - TRUTH_KS),
        "pearson_r_Ks_vs_KGE": d["r_ks"],
        "pearson_r_f_vs_KGE": d["r_f"],
        "area_frac_PBIAS_feasible": d["area_frac_feasible"],
        "n_swoosh_branches": d["n_segs"],
        "swoosh_quad_coef_Ks_vs_log10f": curv[0] if curv is not None else np.nan,
        "swoosh_curvature_fit_r2": d["curv_r2"],
        "swoosh_curvature_fit_npts": d["curv_npts"],
    }
    if joint_stats:
        row["area_frac_joint_feasible_all3"] = joint_stats["area_frac_joint_feasible"]
    rows.append(row)
summary_df = pd.DataFrame(rows)
csv1_path = summary_dir / f"storm_series_summary_{tag}.csv"
summary_df.to_csv(csv1_path, index=False)
print(f"\n  Saved: {csv1_path.name}")

pair_rows = []
for (a, b), dr in delta_results.items():
    pair_rows.append({"pair": f"{a}_vs_{b}", "series_A": SERIES[a]["label"], "series_B": SERIES[b]["label"], **dr})
if pair_rows:
    pair_df = pd.DataFrame(pair_rows)
    csv2_path = summary_dir / f"storm_pairwise_delta_summary_{tag}.csv"
    pair_df.to_csv(csv2_path, index=False)
    print(f"  Saved: {csv2_path.name}")
    print("\n" + pair_df.T.to_string())

print("\n" + summary_df.T.to_string())
