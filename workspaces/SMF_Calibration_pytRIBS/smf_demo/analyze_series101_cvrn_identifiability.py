"""
analyze_series101_cvrn_identifiability.py
===========================================
Series 101 -- Pearson + Spearman correlation and PCA analysis for the cv/r/n
routing-parameter identifiability rebuild. Replaces plot_pearson_nanchor_99.py
and plot_pca_nanchor_99.py (both alpha-based, tied to deleted Series 96/99
output -- see Handoff_Series101_CvRnIdentifiabilityRebuild_v1.md §3).

What this checks
-----------------
The established (pre-101) identifiability hierarchy is: Ks/f own volume,
flowexp (r) owns shape, channelroughness (n) is timing-only-via-timing-
metrics, kinemvelcoef (cv) is equifinal (composite-KGE-blind). This script
re-tests that hierarchy against Series 101 data:
  - KGE_2012 (gamma) as primary, KGE_2009 (alpha) also reported for
    continuity with the pre-transition literature.
  - Both Pearson r (linear) and Spearman rho (monotonic) for every
    parameter x metric pair -- Spearman matters here because flowexp/r is
    documented as non-monotone (r=0.3 narrowest/highest-peak, r=0.4 most
    attenuated in Huner 2025), which a pure Pearson pass can understate or
    miss entirely.
  - The full phase-specific metric set (first arrival, rising-limb
    steepness, time-to-peak-from-exceedance, duration-above-threshold,
    recession rate) alongside composite KGE, since composite score alone is
    already known not to discriminate cv or n.
  - PCA on the standardized (cv, r, n) parameter space, restricted to each
    anchor's top-KGE_2012 runs, to test whether cv's null KGE correlation
    reflects genuine equifinality (a degenerate/low-variance direction in
    parameter space among near-equally-good runs) rather than merely weak
    sensitivity.

Usage (run from the smf_demo directory, after run_lhs_nanchor_cvrn_101.py
has produced the combined results file):
    python analyze_series101_cvrn_identifiability.py
    python analyze_series101_cvrn_identifiability.py --top_frac 0.10   # tighter PCA subset
    python analyze_series101_cvrn_identifiability.py --no_plots        # skip PNG output

Input:
    calibration_work/03_comparisons/summary_tables/lhs_results_anchor_ALL_101.csv

Output (all to calibration_work/03_comparisons/summary_tables/):
    series101_correlation_summary.csv   -- tidy: anchor x parameter x metric
                                            Pearson r/p, Spearman rho/p, n
    series101_pca_summary.csv           -- tidy: anchor x component
                                            explained variance ratio + loadings
Output (all to calibration_work/03_comparisons/hydrograph_plots/, unless
--no_plots):
    fig_series101_correlation_heatmap.png
    fig_series101_pca_loadings.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------
# Parameters and metrics tested. Metric column names match exactly what
# run_sensitivity_single.py writes into *_metrics_summary.csv (and
# therefore into lhs_results_anchor_ALL_101.csv).
# ------------------------------------------------------------------
PARAMS = ["kinemvelcoef", "flowexp", "channelroughness"]
PARAM_LABELS = {"kinemvelcoef": "cv", "flowexp": "r", "channelroughness": "n"}

METRICS = [
    "kge_2012",
    "kge",                              # KGE_2009, retained for continuity
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    "duration_above_thresh_error_min",
    "recession_rate_ratio",
]


def load_combined_results(summary_dir):
    combined_path = summary_dir / "lhs_results_anchor_ALL_101.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"{combined_path} not found. Run run_lhs_nanchor_cvrn_101.py first."
        )
    df = pd.read_csv(combined_path)
    print(f"Loaded {len(df)} rows from {combined_path.name} "
          f"across {df['anchor_label'].nunique()} anchors.")
    return df


# ------------------------------------------------------------------
# CORRELATIONS -- Pearson + Spearman for every (anchor, parameter, metric)
# triple, plus an ALL_POOLED group that ignores anchor identity (useful as
# a sanity check but not the headline result -- pooling across anchors with
# different Ks/f can itself introduce spurious correlation via the
# KGE-formula/anchor interaction documented in the KGE_2012 transition, so
# treat ALL_POOLED as descriptive only).
# ------------------------------------------------------------------
def compute_correlations(df):
    rows = []
    groups = list(df.groupby("anchor_label"))
    groups.append(("ALL_POOLED", df))

    for anchor_label, sub in groups:
        for param in PARAMS:
            for metric in METRICS:
                if param not in sub.columns or metric not in sub.columns:
                    continue
                pair = sub[[param, metric]].dropna()
                n = len(pair)
                if n < 3:
                    pearson_r, pearson_p = np.nan, np.nan
                    spearman_rho, spearman_p = np.nan, np.nan
                else:
                    pearson_r, pearson_p = stats.pearsonr(pair[param], pair[metric])
                    spearman_rho, spearman_p = stats.spearmanr(pair[param], pair[metric])
                rows.append({
                    "anchor_label": anchor_label,
                    "parameter": param,
                    "parameter_symbol": PARAM_LABELS[param],
                    "metric": metric,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "n": n,
                })
    return pd.DataFrame(rows)


def print_correlation_summary(corr_df):
    print(f"\n{'='*70}")
    print("Parameter x KGE_2012 correlations by anchor (Pearson r / Spearman rho)")
    print(f"{'='*70}")
    kge_rows = corr_df[corr_df["metric"] == "kge_2012"]
    for anchor_label in kge_rows["anchor_label"].unique():
        sub = kge_rows[kge_rows["anchor_label"] == anchor_label]
        parts = []
        for _, r in sub.iterrows():
            parts.append(f"{r['parameter_symbol']}: r={r['pearson_r']:+.3f} "
                         f"rho={r['spearman_rho']:+.3f}")
        print(f"  {anchor_label:>14}:  " + "   ".join(parts))

    print(f"\n{'='*70}")
    print("Pearson/Spearman divergence flags (|rho - r| > 0.15)")
    print("-- catches monotonic-but-nonlinear relationships Pearson understates,")
    print("   most relevant for flowexp's documented non-monotone behavior.")
    print(f"{'='*70}")
    diverge = corr_df.dropna(subset=["pearson_r", "spearman_rho"]).copy()
    diverge["abs_diff"] = (diverge["spearman_rho"] - diverge["pearson_r"]).abs()
    flagged = diverge[diverge["abs_diff"] > 0.15].sort_values("abs_diff", ascending=False)
    if flagged.empty:
        print("  None flagged.")
    else:
        for _, r in flagged.iterrows():
            print(f"  {r['anchor_label']:>14} | {r['parameter_symbol']:>2} vs {r['metric']:<32} "
                  f"r={r['pearson_r']:+.3f}  rho={r['spearman_rho']:+.3f}  "
                  f"(diff={r['abs_diff']:.3f})")

    print(f"\n{'='*70}")
    print("cv (kinemvelcoef) null-result check across all metrics/anchors")
    print("-- if this hierarchy holds, cv should show |r|,|rho| < ~0.15 nearly")
    print("   everywhere, consistent with genuine equifinality rather than an")
    print("   underpowered null.")
    print(f"{'='*70}")
    cv_rows = corr_df[(corr_df["parameter"] == "kinemvelcoef") &
                       (corr_df["anchor_label"] != "ALL_POOLED")]
    cv_rows = cv_rows.dropna(subset=["pearson_r", "spearman_rho"])
    if cv_rows.empty:
        print("  No cv correlation rows available.")
    else:
        max_abs_r   = cv_rows["pearson_r"].abs().max()
        max_abs_rho = cv_rows["spearman_rho"].abs().max()
        print(f"  max |Pearson r|  across all anchors/metrics: {max_abs_r:.3f}")
        print(f"  max |Spearman rho| across all anchors/metrics: {max_abs_rho:.3f}")
        worst = cv_rows.loc[cv_rows["pearson_r"].abs().idxmax()]
        print(f"  strongest single cv signal: {worst['anchor_label']} / "
              f"{worst['metric']}  (r={worst['pearson_r']:+.3f})")


# ------------------------------------------------------------------
# PCA -- standardized (cv, r, n) space, restricted to each anchor's top
# KGE_2012 runs (default top_frac=0.20). Implemented via SVD on the
# standardized data matrix rather than pulling in sklearn, since numpy/
# pandas/scipy are already the project's standard toolkit.
#
# A degenerate (low-explained-variance) direction among the top-KGE runs
# means those near-equally-good runs are NOT clustering tightly around the
# true (cv, r, n) point in that direction -- i.e. genuine equifinality
# along that axis, not just a parameter the metric happens to be
# insensitive to at a single point.
# ------------------------------------------------------------------
def run_pca(df, top_frac=0.20, min_n=8):
    rows = []
    for anchor_label, sub in df.groupby("anchor_label"):
        sub = sub.dropna(subset=PARAMS + ["kge_2012"])
        n_keep = max(min_n, int(np.ceil(len(sub) * top_frac)))
        top = sub.sort_values("kge_2012", ascending=False).head(n_keep)
        if len(top) < min_n:
            print(f"  Skipping PCA for '{anchor_label}': only {len(top)} runs "
                  f"available (< min_n={min_n}).")
            continue

        X = top[PARAMS].to_numpy(dtype=float)
        mu, sigma = X.mean(axis=0), X.std(axis=0, ddof=1)
        sigma_safe = np.where(sigma == 0, 1.0, sigma)
        Xs = (X - mu) / sigma_safe

        # SVD-based PCA: Xs = U S Vt; components are rows of Vt, explained
        # variance ratio from squared singular values.
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        explained_var = (S ** 2) / (len(top) - 1)
        explained_ratio = explained_var / explained_var.sum()

        for comp_idx in range(Vt.shape[0]):
            rows.append({
                "anchor_label": anchor_label,
                "n_runs_used": len(top),
                "top_frac_requested": top_frac,
                "component": f"PC{comp_idx + 1}",
                "explained_variance_ratio": explained_ratio[comp_idx],
                "loading_cv": Vt[comp_idx, PARAMS.index("kinemvelcoef")],
                "loading_r":  Vt[comp_idx, PARAMS.index("flowexp")],
                "loading_n":  Vt[comp_idx, PARAMS.index("channelroughness")],
            })
    return pd.DataFrame(rows)


def print_pca_summary(pca_df):
    print(f"\n{'='*70}")
    print("PCA on standardized (cv, r, n), top-KGE_2012 runs per anchor")
    print("-- a low explained-variance-ratio component with a large |loading_cv|")
    print("   indicates cv varies close to freely among near-equally-good runs:")
    print("   equifinality, not just insensitivity at a single point.")
    print(f"{'='*70}")
    if pca_df.empty:
        print("  No PCA results (too few runs per anchor -- check --top_frac / min_n).")
        return
    for anchor_label in pca_df["anchor_label"].unique():
        sub = pca_df[pca_df["anchor_label"] == anchor_label]
        n_used = sub["n_runs_used"].iloc[0]
        print(f"\n  Anchor '{anchor_label}' (n={n_used} top runs):")
        for _, r in sub.iterrows():
            print(f"    {r['component']}: explained_var={r['explained_variance_ratio']:.1%}  "
                  f"loadings [cv={r['loading_cv']:+.3f}  r={r['loading_r']:+.3f}  "
                  f"n={r['loading_n']:+.3f}]")


# ------------------------------------------------------------------
# PLOTS (optional)
# ------------------------------------------------------------------
def make_plots(corr_df, pca_df, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Correlation heatmap (Pearson r), anchors x (parameter, metric) ---
    kge_df = corr_df[(corr_df["anchor_label"] != "ALL_POOLED")]
    pivot = kge_df.pivot_table(index="anchor_label", columns=["parameter_symbol", "metric"],
                                values="pearson_r")
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * pivot.shape[1]), max(4, 0.5 * pivot.shape[0])))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{p}: {m}" for p, m in pivot.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Series 101: Pearson r, parameter x metric, by anchor")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    out1 = plot_dir / "fig_series101_correlation_heatmap.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out1.name}")

    # --- PCA loadings bar chart, one panel per anchor ---
    if not pca_df.empty:
        anchors = list(pca_df["anchor_label"].unique())
        fig, axes = plt.subplots(1, len(anchors), figsize=(3.2 * len(anchors), 4), sharey=True)
        if len(anchors) == 1:
            axes = [axes]
        for ax, anchor_label in zip(axes, anchors):
            sub = pca_df[pca_df["anchor_label"] == anchor_label]
            x = np.arange(len(sub))
            width = 0.25
            ax.bar(x - width, sub["loading_cv"], width, label="cv")
            ax.bar(x,          sub["loading_r"],  width, label="r")
            ax.bar(x + width,  sub["loading_n"],  width, label="n")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["component"], fontsize=8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(anchor_label, fontsize=9)
        axes[0].set_ylabel("PCA loading")
        axes[-1].legend(fontsize=8)
        fig.suptitle("Series 101: PCA loadings on standardized (cv, r, n), top-KGE_2012 runs")
        fig.tight_layout()
        out2 = plot_dir / "fig_series101_pca_loadings.png"
        fig.savefig(out2, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out2.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Series 101 -- Pearson/Spearman correlation + PCA identifiability analysis.")
    parser.add_argument("--top_frac", type=float, default=0.20,
                        help="Fraction of each anchor's runs (by KGE_2012) used for PCA (default: 0.20)")
    parser.add_argument("--min_n", type=int, default=8,
                        help="Minimum runs required per anchor for PCA (default: 8)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip PNG output (correlation/PCA CSVs are always written)")
    args = parser.parse_args()

    script_dir = Path.cwd()
    project_root = (script_dir.parent
                    if script_dir.name == "smf_demo" else script_dir)
    calib_dir   = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"
    plot_dir    = calib_dir / "03_comparisons" / "hydrograph_plots"

    df = load_combined_results(summary_dir)

    print("\nComputing Pearson + Spearman correlations...")
    corr_df = compute_correlations(df)
    corr_path = summary_dir / "series101_correlation_summary.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Saved: {corr_path.name}  ({len(corr_df)} rows)")
    print_correlation_summary(corr_df)

    print(f"\nRunning PCA (top_frac={args.top_frac}, min_n={args.min_n})...")
    pca_df = run_pca(df, top_frac=args.top_frac, min_n=args.min_n)
    pca_path = summary_dir / "series101_pca_summary.csv"
    pca_df.to_csv(pca_path, index=False)
    print(f"  Saved: {pca_path.name}  ({len(pca_df)} rows)")
    print_pca_summary(pca_df)

    if not args.no_plots:
        print("\nGenerating plots...")
        make_plots(corr_df, pca_df, plot_dir)

    print(f"\n{'='*70}")
    print("Done. Cross-reference series101_correlation_summary.csv and")
    print("series101_pca_summary.csv against the Series 96/99 qualitative")
    print("hierarchy (Ks/f own volume, r owns shape, n is timing-only, cv is")
    print("equifinal) before citing any of this in the paper -- that hierarchy")
    print("is exactly what Series 101 exists to re-verify, not assume.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
