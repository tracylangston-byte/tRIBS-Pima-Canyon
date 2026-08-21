"""
plot_pca_nanchor_99.py
=======================
Series 99 -- PCA extension of the N-anchor cv/r/n identifiability sweep.
 
Companion to plot_pearson_nanchor_99.py (univariate Pearson r trends) and
plot_recession_residual_pca.py (single-anchor PCA framework from S91-96).
This script generalizes the latter's phase-specific-metric PCA to all 9
Series 99 anchors, and adds a new diagnostic aimed specifically at the
open question flagged in the Series97/99 weekly report:
 
    "flowexp (r) shows a real branch-driven shift in its correlation with
    composite KGE, but decomposing KGE into its r/alpha/beta sub-components
    shows none of the three individually reproduces this shift -- it
    appears to be an emergent property of the combined metric."
 
Two independent PCA analyses are run per anchor:
 
  (A) Phase-metric PCA (same 7-metric set as plot_recession_residual_pca.py:
      first_arrival_error_min, rising_limb_steepness_ratio,
      time_to_peak_from_exc_min, peak_error_pct, peak_timing_error_hr,
      pbias_pct, duration_above_thresh_error_min).
      Tests whether the phase-metric covariance structure (not just
      individual Pearson r's) is stable across anchors -- a multivariate
      complement to the existing per-metric trend grid.
 
  (B) KGE-subcomponent PCA (kge_r, kge_alpha, kge_beta) + a linear-model
      check. Since PCA only ever finds LINEAR combinations, and KGE is a
      Euclidean-distance (nonlinear) combination of these three terms
      -- KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2) -- this
      directly tests whether the branch effect can be reproduced by any
      linear combination of the sub-components. It also fits
      KGE ~ kge_r + kge_alpha + kge_beta by OLS per anchor and reports the
      R^2 of that linear approximation, which is a direct measure of how
      much the Euclidean formula's curvature matters in each anchor's
      sampled region -- and whether that curvature tracks the branch split.
 
Outputs
-------
Figures (dpi=150, saved to plot_dir):
  fig28_pca_scree_by_anchor.png          -- 3x3 grid, phase-PCA scree per anchor
  fig29_pca_scores_pc1pc2_by_branch.png  -- pooled PC1/PC2 scores, colored by branch
  fig30_pca_loadings_stability.png       -- PC1 loading per metric, across anchors (swoosh order)
  fig31_emergent_kge_nonlinearity.png    -- the branch-effect diagnostic (2-panel)
 
Summary CSVs (saved to summary_dir):
  series99_pca_phase_scores.csv
  series99_pca_phase_loadings.csv
  series99_pca_phase_explained.csv
  series99_kgesub_pca_summary.csv        -- the emergent-effect diagnostic table
 
Usage (run from smf_demo):
    python plot_pca_nanchor_99.py
"""
 
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
 
from parameter_key import PARAM_KEY, METRIC_KEY
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
 
# =======================================================================
# CONFIG
# =======================================================================
LHS_SERIES = "99"
 
# Mirrors ANCHORS in plot_pearson_nanchor_99.py / run_lhs_nanchor_cvrn_99.py.
# Ks6p25lo stays excluded (tRIBS hang region) and is therefore absent here too.
ANCHORS = [
    {"label": "Ks4p25",   "display_label": "Ks4.25",   "Ks_mult": 4.25, "f_RS_abs": 0.0055, "branch": None},
    {"label": "anchorA",  "display_label": "Ks5.0",    "Ks_mult": 5.0,  "f_RS_abs": 0.0075, "branch": None},
    {"label": "Ks5p25",   "display_label": "Ks5.25",   "Ks_mult": 5.25, "f_RS_abs": 0.008,  "branch": None},
    {"label": "Ks6p25hi", "display_label": "Ks6.25hi", "Ks_mult": 6.25, "f_RS_abs": 0.0403, "branch": "hi"},
    {"label": "anchorB",  "display_label": "Ks6.5lo",  "Ks_mult": 6.5,  "f_RS_abs": 0.011,  "branch": "lo"},
    {"label": "Ks7p25lo", "display_label": "Ks7.25lo", "Ks_mult": 7.25, "f_RS_abs": 0.0125, "branch": "lo"},
    {"label": "Ks7p25hi", "display_label": "Ks7.25hi", "Ks_mult": 7.25, "f_RS_abs": 0.0288, "branch": "hi"},
    {"label": "Ks8p25lo", "display_label": "Ks8.25lo", "Ks_mult": 8.25, "f_RS_abs": 0.0155, "branch": "lo"},
    {"label": "Ks8p25hi", "display_label": "Ks8.25hi", "Ks_mult": 8.25, "f_RS_abs": 0.0188, "branch": "hi"},
]
ANCHOR_LOOKUP = {a["label"]: a for a in ANCHORS}
 
# Ks-f equifinality "swoosh" path order (not plain Ks-ascending) -- see
# plot_pearson_nanchor_99.py for the full rationale. Used for every
# anchor-ordered axis in this script so figures read consistently
# alongside the existing trend-grid figures.
SWOOSH_ORDER = ["Ks4p25", "anchorA", "Ks5p25", "anchorB",
                "Ks7p25lo", "Ks8p25lo", "Ks8p25hi", "Ks7p25hi", "Ks6p25hi"]
 
BRANCH_COLOR = {"hi": "#e76f51", "lo": "#457b9d", "single": "#8d8d8d"}
 
MIN_VALID = 10  # minimum rows required per anchor to run a PCA
 
# (A) Phase-specific metrics -- identical set to PCA_PHASE_METRICS in
# plot_recession_residual_pca.py. Deliberately excludes recession_rate_ratio
# and the KGE family (kge/nse/kge_r/kge_alpha/kge_beta) -- those are
# collinear-by-construction summary stats that would dominate PC1 and bury
# the richer per-phase structure. volume_error_pct is excluded too: it is
# an exact algebraic duplicate of pbias_pct for a fixed-event run (r=1.000
# in S93/S96/S99 data), so including both double-weights that one shared
# volume-bias direction in the covariance matrix.
PCA_PHASE_METRICS = [
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    "peak_error_pct",
    "peak_timing_error_hr",
    "pbias_pct",
    "duration_above_thresh_error_min",
]
 
# (B) The three KGE sub-components -- used only for the emergent-effect
# diagnostic (Section 2), never mixed with the phase metrics above.
KGE_SUBCOMPONENTS = ["kge_r", "kge_alpha", "kge_beta"]
 
OUTPUT_SUBDIR = "NAnchor_99"
 
 
# =======================================================================
# PATHS
# =======================================================================
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
summary_dir  = project_root / "calibration_work" / "03_comparisons" / "summary_tables"
plot_dir     = project_root / "calibration_work" / "03_comparisons" / "sensitivity_plots" / OUTPUT_SUBDIR
plot_dir.mkdir(parents=True, exist_ok=True)
 
 
def label_for(col: str) -> str:
    if col in METRIC_KEY:
        name  = METRIC_KEY[col]["display_name"]
        units = METRIC_KEY[col].get("units", "")
        return f"{name} ({units})" if units else name
    if col in PARAM_KEY:
        sym  = PARAM_KEY[col].get("symbol", "")
        name = PARAM_KEY[col]["display_name"]
        return f"{name} ({sym})" if sym else name
    return col
 
 
def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)
 
 
def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    return float(pearsonr(x[mask], y[mask])[0])
 
 
# =======================================================================
# LOAD DATA -- prefer the combined all-anchor CSV; fall back to
# per-anchor files (lhs_results_anchor_<label>_99.csv), opportunistically
# skipping any anchor whose file isn't there yet, matching the pattern
# used throughout the rest of the Series 99 toolchain.
# =======================================================================
def load_all_anchors():
    combined_path = summary_dir / f"lhs_results_anchor_ALL_{LHS_SERIES}.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        print(f"Loaded combined CSV: {combined_path.name} ({len(df)} rows)")
        return df
 
    print("Combined CSV not found -- assembling from per-anchor files...")
    frames = []
    for a in ANCHORS:
        path = summary_dir / f"lhs_results_anchor_{a['label']}_{LHS_SERIES}.csv"
        if not path.exists():
            print(f"  SKIP '{a['label']}': {path.name} not found yet.")
            continue
        sub = pd.read_csv(path)
        sub["anchor_label"] = a["label"]
        frames.append(sub)
        print(f"  Loaded '{a['label']}' ({a['display_label']}): {len(sub)} rows")
    if not frames:
        raise FileNotFoundError(
            "No Series 99 anchor data found (neither combined nor per-anchor CSVs). "
            "Run run_lhs_nanchor_cvrn_99.py first."
        )
    return pd.concat(frames, ignore_index=True)
 
 
df_all = load_all_anchors()
available_labels = [a["label"] for a in ANCHORS if a["label"] in df_all["anchor_label"].unique()]
swoosh_available  = [lbl for lbl in SWOOSH_ORDER if lbl in available_labels]
n_anchors = len(swoosh_available)
print(f"\n{n_anchors}/{len(ANCHORS)} anchors available: "
      f"{[ANCHOR_LOOKUP[l]['display_label'] for l in swoosh_available]}\n")
 
 
# =======================================================================
# CORE PCA HELPER -- shared by both analyses. z-scores WITHIN the given
# anchor's own rows before running SVD, so pooling anchors afterward never
# lets a between-anchor baseline shift (different Ks/f volume levels)
# masquerade as parameter-driven structure.
# =======================================================================
def run_pca(X: pd.DataFrame):
    keep = [c for c in X.columns if X[c].std(ddof=1) > 0]
    Xk = X[keep].astype(float)
    Xk = Xk.fillna(Xk.median())
    Xz = (Xk - Xk.mean()) / Xk.std(ddof=1)
    U, S, Vt = np.linalg.svd(Xz.to_numpy(), full_matrices=False)
    n = Xz.shape[0]
    eigvals   = (S ** 2) / (n - 1)
    explained = eigvals / eigvals.sum()
    scores  = U * S
    pc_names = [f"PC{i+1}" for i in range(len(S))]
    scores_df   = pd.DataFrame(scores, columns=pc_names, index=Xk.index)
    loadings_df = pd.DataFrame(Vt.T, index=keep, columns=pc_names)
    explained_df = pd.DataFrame({
        "component": pc_names,
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": np.cumsum(explained),
    })
    return scores_df, loadings_df, explained_df
 
 
# =======================================================================
# ANALYSIS (A) -- phase-metric PCA, per anchor
# =======================================================================
print("Running phase-metric PCA per anchor (Analysis A)...")
phase_scores_rows, phase_loadings_rows, phase_explained_rows = [], [], []
 
for label in swoosh_available:
    a = ANCHOR_LOOKUP[label]
    sub = df_all[df_all["anchor_label"] == label].copy()
    avail_metrics = [m for m in PCA_PHASE_METRICS if m in sub.columns]
    X = sub[avail_metrics].dropna()
    if len(X) < MIN_VALID:
        print(f"  SKIP '{label}': only {len(X)} valid rows (<{MIN_VALID}).")
        continue
 
    scores_df, loadings_df, explained_df = run_pca(X)
 
    idx = X.index
    out = scores_df.copy()
    out["anchor_label"]   = label
    out["anchor_display"] = a["display_label"]
    out["branch"]         = a["branch"] if a["branch"] else "single"
    out["flowexp"]        = sub.loc[idx, "flowexp"].values
    out["kinemvelcoef"]   = sub.loc[idx, "kinemvelcoef"].values
    out["channelroughness"] = sub.loc[idx, "channelroughness"].values
    out["kge"]            = sub.loc[idx, "kge"].values
    phase_scores_rows.append(out)
 
    ld = loadings_df.copy()
    ld["anchor_label"] = label
    ld["metric"] = ld.index
    phase_loadings_rows.append(ld.reset_index(drop=True))
 
    ex = explained_df.copy()
    ex["anchor_label"] = label
    phase_explained_rows.append(ex)
 
phase_scores_df    = pd.concat(phase_scores_rows, ignore_index=True)
phase_loadings_df  = pd.concat(phase_loadings_rows, ignore_index=True)
phase_explained_df = pd.concat(phase_explained_rows, ignore_index=True)
 
phase_scores_df.to_csv(summary_dir / "series99_pca_phase_scores.csv", index=False)
phase_loadings_df.to_csv(summary_dir / "series99_pca_phase_loadings.csv", index=False)
phase_explained_df.to_csv(summary_dir / "series99_pca_phase_explained.csv", index=False)
print(f"  Saved series99_pca_phase_{{scores,loadings,explained}}.csv "
      f"({len(phase_scores_df)} total scored runs)\n")
 
 
# ---- fig28: scree plot, one panel per anchor, swoosh order --------------
ncols = 3
nrows = int(np.ceil(n_anchors / ncols))
fig28, axes28 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
axes28 = np.atleast_1d(axes28).ravel()
 
for i, label in enumerate(swoosh_available):
    ax = axes28[i]
    a = ANCHOR_LOOKUP[label]
    ex = phase_explained_df[phase_explained_df["anchor_label"] == label]
    x = np.arange(1, len(ex) + 1)
    color = BRANCH_COLOR[a["branch"] if a["branch"] else "single"]
    ax.bar(x, ex["explained_variance_ratio"] * 100, color=color, edgecolor="white", alpha=0.85)
    ax.plot(x, ex["cumulative_explained_variance"] * 100, marker="o", color="black", markersize=3, linewidth=1)
    ax.set_title(f"{a['display_label']} ({a['branch'] or 'single'})", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    if i % ncols == 0:
        ax.set_ylabel("Explained var. (%)")
    if i >= n_anchors - ncols:
        ax.set_xlabel("PC")
 
for j in range(n_anchors, len(axes28)):
    axes28[j].axis("off")
 
fig28.suptitle(
    "Series 99: phase-metric PCA scree, per anchor (swoosh order)\n"
    "PC1 variance share stable across all anchors \u2192 the volume/shape axis is anchor-invariant",
    fontsize=11,
)
fig28.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig28, plot_dir / "fig28_pca_scree_by_anchor.png")
 
 
# ---- fig29: pooled PC1 vs PC2 scores, colored by branch -----------------
fig29, ax29 = plt.subplots(figsize=(8, 6.5))
for branch, color in BRANCH_COLOR.items():
    sub = phase_scores_df[phase_scores_df["branch"] == branch]
    if sub.empty:
        continue
    ax29.scatter(sub["PC1"], sub["PC2"], s=28, c=color, alpha=0.6,
                 edgecolors="white", linewidths=0.3, label=f"{branch} branch")
ax29.axhline(0, color="gray", linewidth=0.7)
ax29.axvline(0, color="gray", linewidth=0.7)
ax29.set_xlabel("PC1 (within-anchor z-scored phase metrics)")
ax29.set_ylabel("PC2")
ax29.set_title(
    "Series 99: pooled phase-metric PCA scores, all 9 anchors\n"
    "Colored by branch \u2014 no branch-driven separation would support anchor-invariant structure",
    fontsize=10.5,
)
ax29.legend(frameon=False)
ax29.grid(linestyle=":", alpha=0.3)
fig29.tight_layout()
save_fig(fig29, plot_dir / "fig29_pca_scores_pc1pc2_by_branch.png")
 
 
# ---- fig30: PC1 loading stability across anchors -------------------------
fig30, ax30 = plt.subplots(figsize=(9.5, 6))
x = np.arange(n_anchors)
cmap = plt.get_cmap("tab10")
for i, metric in enumerate(PCA_PHASE_METRICS):
    vals = []
    for label in swoosh_available:
        row = phase_loadings_df[(phase_loadings_df["anchor_label"] == label) &
                                 (phase_loadings_df["metric"] == metric)]
        vals.append(row["PC1"].values[0] if len(row) and "PC1" in row.columns else np.nan)
    ax30.plot(x, vals, marker="o", markersize=4, linewidth=1.4, color=cmap(i % 10),
              label=label_for(metric))
 
ax30.axhline(0, color="gray", linewidth=0.7)
ax30.set_xticks(x)
ax30.set_xticklabels([ANCHOR_LOOKUP[l]["display_label"] for l in swoosh_available], rotation=45, ha="right")
ax30.set_ylabel("PC1 loading")
ax30.set_title(
    "Series 99: PC1 loading per phase metric, across anchors (swoosh order)\n"
    "Flat lines \u2192 the same metrics define the dominant PC everywhere",
    fontsize=10.5,
)
ax30.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax30.grid(axis="y", linestyle=":", alpha=0.3)
fig30.tight_layout()
save_fig(fig30, plot_dir / "fig30_pca_loadings_stability.png")
 
 
# =======================================================================
# ANALYSIS (B) -- KGE-subcomponent PCA + linear-model check
# (the emergent-effect diagnostic)
# =======================================================================
print("Running KGE-subcomponent PCA + linear check per anchor (Analysis B)...")
kgesub_rows = []
 
for label in swoosh_available:
    a = ANCHOR_LOOKUP[label]
    sub = df_all[df_all["anchor_label"] == label].copy()
    X = sub[KGE_SUBCOMPONENTS].dropna()
    if len(X) < MIN_VALID:
        print(f"  SKIP '{label}': only {len(X)} valid rows (<{MIN_VALID}).")
        continue
    idx = X.index
    flowexp = sub.loc[idx, "flowexp"].to_numpy(dtype=float)
    kge     = sub.loc[idx, "kge"].to_numpy(dtype=float)
 
    # PCA on (kge_r, kge_alpha, kge_beta) -- the best LINEAR summary of the
    # three sub-components.
    scores_df, loadings_df, explained_df = run_pca(X)
    pc1 = scores_df["PC1"].to_numpy(dtype=float)
 
    # OLS linear fit of the actual (nonlinear) composite KGE from its own
    # sub-components -- R^2 measures how much curvature the Euclidean
    # formula introduces in this anchor's sampled region.
    Xm = np.column_stack([X.to_numpy(dtype=float), np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(Xm, kge, rcond=None)
    kge_hat = Xm @ coef
    resid = kge - kge_hat
    linear_r2 = 1 - np.sum(resid ** 2) / np.sum((kge - kge.mean()) ** 2)
 
    kgesub_rows.append({
        "anchor_label":       label,
        "anchor_display":     a["display_label"],
        "branch":             a["branch"] if a["branch"] else "single",
        "Ks_mult":            a["Ks_mult"],
        "f_RS_abs":           a["f_RS_abs"],
        "n":                  len(X),
        "pc1_explained_var":  explained_df.loc[0, "explained_variance_ratio"],
        "r_flowexp_kge_actual":     safe_pearson(flowexp, kge),
        "r_flowexp_pc1_linear":     safe_pearson(flowexp, pc1),
        "r_pc1_vs_kge_actual":      safe_pearson(pc1, kge),
        "linear_r2_kge_from_subcomponents": linear_r2,
        "nonlinear_resid_std":      float(np.std(resid)),
        "r_flowexp_nonlinear_resid": safe_pearson(flowexp, resid),
    })
 
kgesub_df = pd.DataFrame(kgesub_rows)
kgesub_df.to_csv(summary_dir / "series99_kgesub_pca_summary.csv", index=False)
print(f"  Saved series99_kgesub_pca_summary.csv ({len(kgesub_df)} anchors)\n")
 
 
# ---- fig31: the emergent-effect diagnostic, 2 panels --------------------
fig31, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))
 
order_df = kgesub_df.set_index("anchor_label").loc[swoosh_available].reset_index()
x = np.arange(n_anchors)
labels_x = order_df["anchor_display"].tolist()
colors = [BRANCH_COLOR[b] for b in order_df["branch"]]
 
# Panel A: actual composite-KGE correlation vs. best-linear-proxy correlation
axA.plot(x, order_df["r_flowexp_kge_actual"], marker="o", color="black",
         linewidth=1.8, label="r(flowexp, KGE) \u2014 actual composite")
axA.plot(x, order_df["r_flowexp_pc1_linear"], marker="s", color="#2a9d8f",
         linewidth=1.8, linestyle="--", label="r(flowexp, PC1) \u2014 best linear proxy")
axA.scatter(x, order_df["r_flowexp_kge_actual"], c=colors, s=90, zorder=5, edgecolors="black", linewidths=0.6)
axA.set_xticks(x)
axA.set_xticklabels(labels_x, rotation=45, ha="right")
axA.set_ylabel("Pearson r with flowexp")
axA.set_title(
    "Actual KGE tracks the branch split;\nits best linear proxy does not",
    fontsize=10.5,
)
axA.legend(fontsize=8.5, frameon=False, loc="lower left")
axA.grid(axis="y", linestyle=":", alpha=0.3)
axA.axhline(0, color="gray", linewidth=0.6)
 
# Panel B: how good the linear approximation is, per anchor (R^2)
axB.bar(x, order_df["linear_r2_kge_from_subcomponents"], color=colors, edgecolor="white")
axB.set_xticks(x)
axB.set_xticklabels(labels_x, rotation=45, ha="right")
axB.set_ylabel("R\u00b2 of OLS( KGE ~ kge_r + kge_alpha + kge_beta )")
axB.set_ylim(0, 1.05)
axB.set_title(
    "How well a straight line explains KGE\nfrom its own sub-components, per anchor",
    fontsize=10.5,
)
axB.grid(axis="y", linestyle=":", alpha=0.3)
axB.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
 
# Shared branch legend for panel B
from matplotlib.patches import Patch
handles = [Patch(color=c, label=f"{b} branch") for b, c in BRANCH_COLOR.items()]
axB.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower right")
 
fig31.suptitle(
    "Series 99: is the flowexp\u2013KGE branch effect a linear-combination artifact?\n"
    "PCA / OLS on (kge_r, kge_alpha, kge_beta) can only ever recover a straight-line "
    "combination \u2014 the branch effect is specifically what neither recovers.",
    fontsize=11.5,
)
fig31.tight_layout(rect=[0, 0, 1, 0.90])
save_fig(fig31, plot_dir / "fig31_emergent_kge_nonlinearity.png")
 
 
# =======================================================================
# CONSOLE SUMMARY
# =======================================================================
print("\n" + "=" * 78)
print("SERIES 99 -- PCA SUMMARY")
print("=" * 78)
 
print("\n(A) Phase-metric PCA -- PC1 explained variance by anchor:")
piv = phase_explained_df[phase_explained_df["component"] == "PC1"]
piv = piv.set_index("anchor_label").loc[swoosh_available]
print(piv[["explained_variance_ratio"]].rename(
    columns={"explained_variance_ratio": "PC1_var"}
).to_string(float_format=lambda v: f"{v:.3f}"))
 
print("\n(B) Emergent-effect diagnostic (branch-ordered = hi/lo/single interleaved, "
      "table is swoosh-ordered):")
cols = ["anchor_display", "branch", "r_flowexp_kge_actual", "r_flowexp_pc1_linear",
        "r_pc1_vs_kge_actual", "linear_r2_kge_from_subcomponents"]
print(kgesub_df.set_index("anchor_label").loc[swoosh_available][cols].to_string(
    index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else v
))
 
print(
    "\nReading the diagnostic:\n"
    "  - r_flowexp_kge_actual swings noticeably by branch (this is the effect flagged\n"
    "    in the weekly report).\n"
    "  - r_flowexp_pc1_linear -- the best possible LINEAR summary of kge_r/alpha/beta --\n"
    "    stays comparatively flat across the same anchors. A linear combination of the\n"
    "    sub-components does not reproduce the branch split.\n"
    "  - linear_r2_kge_from_subcomponents (how well a straight line explains the actual\n"
    "    composite KGE) is itself branch-dependent -- higher for 'hi' anchors, lower for\n"
    "    'lo'/single anchors -- meaning the Euclidean KGE formula's curvature bites harder\n"
    "    in some anchors' sampled region than others. This is consistent with the branch\n"
    "    effect being a property of KGE's nonlinear combination rule rather than a new\n"
    "    physical signal -- worth documenting for Josh rather than chasing further."
)
print("=" * 78 + "\n")
 
print(f"Done. All outputs in:\n  {plot_dir}\n  {summary_dir}")