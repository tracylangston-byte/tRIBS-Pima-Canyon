"""
plot_series_residual_pca.py
===========================
Written by CHATGPT. Residual analysis + PCA for synthetic inversion series results.

Purpose
-------
1. Characterize scatter around the flowexp (r) vs recession_rate_ratio relationship.
2. Ask what explains the remaining recession residual scatter after the main r trend is removed.
3. Run PCA on the hydrograph metric matrix.

Designed to match the existing smf_demo file/path logic:
    script_dir   = Path.cwd()
    project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
    calib_dir    = project_root / "calibration_work"

It can read either:
    A) a combined LHS table, e.g. lhs_results_synth_4param_92.csv
    B) per-run summary files, e.g.
       SMF_20140812_92_Ks9p348923x_cv6p220846_r0p285581_n0p03255_metrics_summary.csv

Usage, from smf_demo:
    python plot_series_residual_pca.py --series 92
    python plot_series_residual_pca.py --series 93

Optional:
    python plot_series_residual_pca.py --series 92 --poly_degree 2
    python plot_series_residual_pca.py --series 92 --results_csv lhs_results_synth_4param_92.csv

Outputs
-------
Figures:
    ../calibration_work/03_comparisons/sensitivity_plots/SeriesXX_SynthInversion/residual_pca/

Tables:
    ../calibration_work/03_comparisons/summary_tables/seriesXX_*.csv

Notes
-----
- This script does NOT rerun tRIBS.
- PCA is computed with NumPy SVD, so scikit-learn is not required.
- Spearman rho is used if scipy is installed; otherwise only Pearson r is used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from parameter_key import PARAM_KEY, METRIC_KEY
except Exception:
    PARAM_KEY = {}
    METRIC_KEY = {}


# ======================================================================
# CONFIG
# ======================================================================
DEFAULT_LOCATION = "SMF"
DEFAULT_EVENT_DATE = "20140812"
KGE_CEILING = 0.912

PARAM_KEYS = [
    "Ks_mult",
    "kinemvelcoef",
    "flowexp",
    "channelroughness",
]

# Known truth values for Series 91/92 synthetic inversion.
# For other series, reference lines are only drawn if values are provided by args.
DEFAULT_TRUE_VALUES_BY_SERIES = {
    "91": {"Ks_mult": 8.50, "kinemvelcoef": 4.50, "flowexp": 0.24, "channelroughness": 0.026},
    "92": {"Ks_mult": 8.50, "kinemvelcoef": 4.50, "flowexp": 0.24, "channelroughness": 0.026},
}

PCA_METRIC_CANDIDATES = [
    # Pre-peak
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    # Peak
    "peak_error_pct",
    "peak_error_m3s",
    "peak_timing_error_hr",
    # Volume
    "pbias_pct",
    "volume_error_pct",
    "duration_above_thresh_error_min",
    # Recession
    "recession_rate_ratio",
    # Summary / KGE family
    "kge",
    "nse",
    "rmse_m3s",
    "kge_r",
    "kge_alpha",
    "kge_beta",
]

RESIDUAL_DRIVER_CANDIDATES = [
    # Parameters other than flowexp, because flowexp is the fitted trend axis
    "Ks_mult",
    "kinemvelcoef",
    "channelroughness",
    # Other metrics
    "kge",
    "nse",
    "rmse_m3s",
    "pbias_pct",
    "volume_error_pct",
    "kge_r",
    "kge_alpha",
    "kge_beta",
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    "peak_error_pct",
    "peak_timing_error_hr",
    "duration_above_thresh_error_min",
]


# ======================================================================
# SMALL HELPERS
# ======================================================================
def label_for(col: str) -> str:
    """Human-readable label for a parameter or metric column."""
    if col in PARAM_KEY:
        sym = PARAM_KEY[col].get("symbol", "")
        name = PARAM_KEY[col].get("display_name", col)
        return f"{name} ({sym})" if sym else name
    if col in METRIC_KEY:
        name = METRIC_KEY[col].get("display_name", col)
        units = METRIC_KEY[col].get("units", "")
        return f"{name} ({units})" if units else name
    fallback = {
        "Ks_mult": "Ks multiplier",
        "kinemvelcoef": "Hillslope velocity coefficient (cv)",
        "flowexp": "Hillslope velocity exponent (r)",
        "channelroughness": "Channel Manning's n",
        "recession_rate_ratio": "Recession rate ratio",
        "pbias_pct": "PBIAS (%)",
        "volume_error_pct": "Volume error (%)",
        "peak_error_pct": "Peak error (%)",
        "peak_error_m3s": "Peak error (m³/s)",
        "peak_timing_error_hr": "Peak timing error (hr)",
        "duration_above_thresh_error_min": "Duration above threshold error (min)",
        "kge": "KGE",
        "nse": "NSE",
        "rmse_m3s": "RMSE (m³/s)",
        "kge_r": "KGE r",
        "kge_alpha": "KGE alpha",
        "kge_beta": "KGE beta",
        "recession_residual": "Recession residual",
    }
    return fallback.get(col, col)


def safe_float_from_label(text: str) -> float:
    """Convert compact run-id labels like '9p348923' to 9.348923."""
    return float(text.replace("p", "."))


def parse_run_metadata_from_filename(path: Path, location: str, event_date: str, series: str) -> dict:
    """Parse run_id and 4-parameter values from existing per-run metrics filenames."""
    stem = path.stem
    if stem.endswith("_metrics_summary"):
        run_id = stem[: -len("_metrics_summary")]
    else:
        run_id = stem

    out = {"run_id": run_id, "source_file": path.name}

    # Expected example:
    # SMF_20140812_92_Ks9p348923x_cv6p220846_r0p285581_n0p03255_metrics_summary.csv
    pattern = (
        rf"^{re.escape(location)}_{re.escape(event_date)}_{re.escape(str(series))}_"
        r"Ks(?P<Ks>[0-9p]+)x_"
        r"cv(?P<cv>[0-9p]+)_"
        r"r(?P<r>[0-9p]+)_"
        r"n(?P<n>[0-9p]+)"
    )
    match = re.search(pattern, run_id)
    if match:
        out["Ks_mult"] = safe_float_from_label(match.group("Ks"))
        out["kinemvelcoef"] = safe_float_from_label(match.group("cv"))
        out["flowexp"] = safe_float_from_label(match.group("r"))
        out["channelroughness"] = safe_float_from_label(match.group("n"))
    return out


def normalize_metrics_summary_csv(path: Path) -> pd.DataFrame:
    """
    Read a per-run metrics summary CSV robustly.

    Supports two common forms:
      1. Wide one-row table: columns are metric names.
      2. Long two-column table: metric/value or Metric/Value.
    """
    raw = pd.read_csv(path)
    if raw.empty:
        return raw

    lower_map = {str(c).strip().lower(): c for c in raw.columns}

    # Long format: metric,value or Metric,Value
    metric_col = None
    value_col = None
    for candidate in ["metric", "metrics", "name", "stat", "statistic"]:
        if candidate in lower_map:
            metric_col = lower_map[candidate]
            break
    for candidate in ["value", "values", "val"]:
        if candidate in lower_map:
            value_col = lower_map[candidate]
            break

    if metric_col is not None and value_col is not None:
        wide = raw[[metric_col, value_col]].dropna(subset=[metric_col]).copy()
        wide[metric_col] = wide[metric_col].astype(str)
        wide = wide.drop_duplicates(subset=[metric_col], keep="first")
        out = wide.set_index(metric_col)[value_col].to_frame().T.reset_index(drop=True)
        return out

    # Otherwise assume wide format. If there are multiple rows, each row is retained;
    # downstream code can still handle it, but normal tRIBS summaries should be one row.
    return raw.copy()


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-id columns to numeric where possible."""
    out = df.copy()
    id_like = {"run_id", "source_file", "change_tested", "series", "location", "event_date"}
    for col in out.columns:
        if col in id_like:
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().sum() > 0:
            out[col] = converted
    return out


def safe_pearson(x, y) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def safe_spearman(x, y) -> float:
    if not HAS_SCIPY:
        return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan
    rho, _ = spearmanr(xv, yv)
    return float(rho)


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    means = out.mean(axis=0)
    stds = out.std(axis=0, ddof=1).replace(0, np.nan)
    return (out - means) / stds


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


# ======================================================================
# LOADING LOGIC
# ======================================================================
def candidate_combined_csv_names(series: str) -> list[str]:
    return [
        f"lhs_results_synth_4param_{series}.csv",
        f"lhs_results_synth_4param_{series}_doublepeak.csv",
        f"lhs_results_{series}.csv",
    ]


def load_combined_or_per_run_results(
    summary_dir: Path,
    location: str,
    event_date: str,
    series: str,
    results_csv: str | None,
) -> tuple[pd.DataFrame, str]:
    """
    Prefer a combined series table if present. If not, combine all matching
    per-run *_metrics_summary.csv files for the requested series.
    """
    if results_csv:
        path = summary_dir / results_csv
        if not path.exists():
            raise FileNotFoundError(f"Requested --results_csv not found: {path}")
        df = pd.read_csv(path)
        source = f"combined CSV: {path.name}"
    else:
        chosen = None
        for name in candidate_combined_csv_names(series):
            p = summary_dir / name
            if p.exists():
                chosen = p
                break

        if chosen is not None:
            df = pd.read_csv(chosen)
            source = f"combined CSV: {chosen.name}"
        else:
            pattern = f"{location}_{event_date}_{series}_*_metrics_summary.csv"
            files = sorted(summary_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No combined CSV and no per-run metric summaries found.\n"
                    f"Looked for combined names: {candidate_combined_csv_names(series)}\n"
                    f"Looked for per-run pattern: {summary_dir / pattern}\n"
                    f"Tip: from smf_demo, your summary path should normally be ../calibration_work/03_comparisons/summary_tables"
                )

            rows = []
            for path in files:
                meta = parse_run_metadata_from_filename(path, location, event_date, series)
                one = normalize_metrics_summary_csv(path)
                if one.empty:
                    print(f"  Warning: skipped empty file {path.name}")
                    continue

                # Most files should be one row; preserve all rows if multiple exist.
                for _, row in one.iterrows():
                    record = meta.copy()
                    record.update(row.to_dict())
                    rows.append(record)

            df = pd.DataFrame(rows)
            source = f"{len(files)} per-run metric summary files matching {pattern}"

    df = coerce_numeric_columns(df)

    # If the combined CSV lacks run_id but has enough filename/source info, make a placeholder.
    if "run_id" not in df.columns:
        if "source_file" in df.columns:
            df["run_id"] = df["source_file"].astype(str).str.replace("_metrics_summary.csv", "", regex=False)
        else:
            df["run_id"] = [f"series{series}_row{i:04d}" for i in range(len(df))]

    # If parameters are missing from a combined CSV but run_id follows the existing naming convention, parse them.
    missing_params = [p for p in PARAM_KEYS if p not in df.columns]
    if missing_params:
        parsed_rows = []
        for run_id in df["run_id"].astype(str):
            fake_path = Path(f"{run_id}_metrics_summary.csv")
            parsed_rows.append(parse_run_metadata_from_filename(fake_path, location, event_date, series))
        parsed_df = pd.DataFrame(parsed_rows)
        for p in missing_params:
            if p in parsed_df.columns:
                df[p] = parsed_df[p].values

    # Keep only exact requested series if run_id is available. This prevents accidental *_vs_* files.
    prefix = f"{location}_{event_date}_{series}_"
    if "run_id" in df.columns:
        series_mask = df["run_id"].astype(str).str.startswith(prefix)
        if series_mask.any():
            df = df.loc[series_mask].copy()

    return df.reset_index(drop=True), source


# ======================================================================
# CORE ANALYSIS
# ======================================================================
def fit_recession_residuals(df: pd.DataFrame, poly_degree: int) -> tuple[pd.DataFrame, np.ndarray]:
    required = ["flowexp", "recession_rate_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for residual analysis: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"If recession_rate_ratio is missing, rerun scoring with the phase-specific metrics enabled."
        )

    out = df.copy()
    x = out["flowexp"].to_numpy(dtype=float)
    y = out["recession_rate_ratio"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < poly_degree + 2:
        raise ValueError(
            f"Not enough finite flowexp/recession_rate_ratio rows for degree {poly_degree} fit. "
            f"Finite rows: {mask.sum()}"
        )

    coeff = np.polyfit(x[mask], y[mask], deg=poly_degree)
    yhat = np.full(len(out), np.nan)
    yhat[mask] = np.polyval(coeff, x[mask])

    out["recession_fit_from_flowexp"] = yhat
    out["recession_residual"] = y - yhat
    out["abs_recession_residual"] = np.abs(out["recession_residual"])
    return out, coeff


def residual_driver_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = df["recession_residual"].to_numpy(dtype=float)
    y_abs = df["abs_recession_residual"].to_numpy(dtype=float)

    for col in RESIDUAL_DRIVER_CANDIDATES:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        rows.append({
            "driver": col,
            "driver_label": label_for(col),
            "pearson_with_signed_residual": safe_pearson(x, y),
            "spearman_with_signed_residual": safe_spearman(x, y),
            "pearson_with_abs_residual": safe_pearson(x, y_abs),
            "spearman_with_abs_residual": safe_spearman(x, y_abs),
            "n_finite": int((np.isfinite(x) & np.isfinite(y)).sum()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["rank_score"] = out[["pearson_with_abs_residual", "spearman_with_abs_residual"]].abs().max(axis=1, skipna=True)
    return out.sort_values("rank_score", ascending=False).reset_index(drop=True)


def run_metric_pca(df: pd.DataFrame):
    available = [c for c in PCA_METRIC_CANDIDATES if c in df.columns]
    if not available:
        raise ValueError("No PCA metric columns were found in the loaded results.")

    Xraw = df[available].copy().astype(float)
    keep_cols = []
    for c in available:
        finite = np.isfinite(Xraw[c].to_numpy(dtype=float))
        if finite.sum() < 10:
            continue
        if Xraw.loc[finite, c].std(ddof=1) == 0:
            continue
        keep_cols.append(c)

    Xraw = Xraw[keep_cols]
    if Xraw.shape[1] < 2:
        raise ValueError(f"Need at least 2 usable metric columns for PCA; found {list(Xraw.columns)}")

    Xfilled = Xraw.copy()
    for c in Xfilled.columns:
        Xfilled[c] = Xfilled[c].fillna(Xfilled[c].median())

    Xz = zscore_frame(Xfilled)
    if Xz.isna().any().any():
        bad = Xz.columns[Xz.isna().any()].tolist()
        raise ValueError(f"NaN remained after z-scoring these columns: {bad}")

    X = Xz.to_numpy(dtype=float)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    n_samples = X.shape[0]
    eigenvalues = (S ** 2) / (n_samples - 1)
    explained = eigenvalues / eigenvalues.sum()

    pc_names = [f"PC{i+1}" for i in range(len(S))]
    scores = U * S
    scores_df = pd.DataFrame(scores, columns=pc_names)
    scores_df.insert(0, "run_id", df["run_id"].values)
    for col in PARAM_KEYS + ["kge", "recession_residual", "abs_recession_residual"]:
        if col in df.columns:
            scores_df[col] = df[col].values

    loadings_df = pd.DataFrame(Vt.T, index=Xraw.columns, columns=pc_names)
    loadings_df.insert(0, "metric_label", [label_for(c) for c in Xraw.columns])
    loadings_df.index.name = "metric"

    explained_df = pd.DataFrame({
        "component": pc_names,
        "eigenvalue": eigenvalues,
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": np.cumsum(explained),
    })
    return scores_df, loadings_df, explained_df, list(Xraw.columns)


# ======================================================================
# PLOTS
# ======================================================================
def plot_r_recession_fit(df: pd.DataFrame, coeff, plot_dir: Path, series: str, true_values: dict):
    fig, ax = plt.subplots(figsize=(8.5, 6))

    color_col = "kge" if "kge" in df.columns else "Ks_mult"
    cvals = df[color_col].to_numpy(dtype=float)
    cmap = "plasma" if color_col == "kge" else "viridis"

    sc = ax.scatter(df["flowexp"], df["recession_rate_ratio"], c=cvals, cmap=cmap,
                    s=45, edgecolors="white", linewidths=0.4, alpha=0.9)
    fig.colorbar(sc, ax=ax, label=label_for(color_col))

    xfit = np.linspace(df["flowexp"].min(), df["flowexp"].max(), 300)
    yfit = np.polyval(coeff, xfit)
    ax.plot(xfit, yfit, color="black", linewidth=2.0, label="Fitted r-recession trend")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Perfect recession ratio = 1")
    if "flowexp" in true_values:
        ax.axvline(true_values["flowexp"], color="red", linestyle="--", linewidth=1.2, label="Truth/reference r")

    ax.set_xlabel("Hillslope velocity exponent r (flowexp)")
    ax.set_ylabel("Recession rate ratio")
    ax.set_title(f"Series {series}: r vs recession behavior\nResiduals are vertical distance from the fitted trend")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig19_r_recession_residual_fit.png")


def plot_residual_vs_parameters(df: pd.DataFrame, plot_dir: Path, true_values: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    cvals = df["kge"].to_numpy(dtype=float) if "kge" in df.columns else df["flowexp"].to_numpy(dtype=float)

    for ax, p in zip(axes, PARAM_KEYS):
        if p not in df.columns:
            ax.axis("off")
            continue
        sc = ax.scatter(df[p], df["recession_residual"], c=cvals, cmap="plasma",
                        s=42, edgecolors="white", linewidths=0.35, alpha=0.9)
        ax.axhline(0, color="black", linewidth=1.0)
        if p in true_values:
            ax.axvline(true_values[p], color="red", linestyle="--", linewidth=1.0)
        r = safe_pearson(df[p].to_numpy(dtype=float), df["recession_residual"].to_numpy(dtype=float))
        ax.set_title(f"Residual vs {label_for(p)}\nPearson r = {r:+.2f}", fontsize=10)
        ax.set_xlabel(label_for(p))
        ax.set_ylabel("Recession residual")
        ax.grid(linestyle=":", alpha=0.35)

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.85)
    cbar.set_label("KGE" if "kge" in df.columns else "flowexp")
    fig.suptitle("What explains scatter after removing the main flowexp–recession trend?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, plot_dir / "fig20_recession_residual_vs_parameters.png")


def plot_residual_driver_bars(driver_df: pd.DataFrame, plot_dir: Path):
    if driver_df.empty:
        return
    top = driver_df.head(12).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    vals = top["pearson_with_abs_residual"].to_numpy(dtype=float)
    colors = np.where(vals >= 0, "#2a9d8f", "#e76f51")
    ax.barh(top["driver_label"], vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with |recession residual|")
    ax.set_title("Top candidate drivers of scatter around the r–recession relationship\nPositive = larger driver value associated with larger residual magnitude")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig21_residual_driver_correlations.png")


def plot_pca_explained(explained_df: pd.DataFrame, plot_dir: Path, series: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(explained_df) + 1)
    ax.bar(x, explained_df["explained_variance_ratio"] * 100, edgecolor="white")
    ax.plot(x, explained_df["cumulative_explained_variance"] * 100, marker="o", color="black")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"PCA on Series {series} hydrograph metric matrix")
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig22_pca_explained_variance.png")


def plot_pca_scores(scores_df: pd.DataFrame, explained_df: pd.DataFrame, plot_dir: Path, series: str):
    if "PC1" not in scores_df.columns or "PC2" not in scores_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    color_col = "recession_residual" if "recession_residual" in scores_df.columns else "flowexp"
    cvals = scores_df[color_col].to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(cvals)) if color_col == "recession_residual" else None
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if color_col == "recession_residual" and vmax and vmax > 0 else None
    cmap = "RdBu_r" if color_col == "recession_residual" else "viridis"
    sc = ax.scatter(scores_df["PC1"], scores_df["PC2"], c=cvals, cmap=cmap, norm=norm,
                    s=48, edgecolors="white", linewidths=0.35, alpha=0.9)
    fig.colorbar(sc, ax=ax, label=label_for(color_col))
    pc1_pct = explained_df.loc[0, "explained_variance_ratio"] * 100
    pc2_pct = explained_df.loc[1, "explained_variance_ratio"] * 100
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title(f"PCA scores: each point is one Series {series} run")
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig23_pca_scores_pc1_pc2.png")


def plot_pca_loadings(loadings_df: pd.DataFrame, explained_df: pd.DataFrame, plot_dir: Path, n_pcs=4):
    pc_cols = [c for c in loadings_df.columns if c.startswith("PC")][:n_pcs]
    if not pc_cols:
        return
    data = loadings_df[pc_cols].to_numpy(dtype=float)
    labels = loadings_df["metric_label"].tolist()
    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.42 * len(labels))))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Loading")
    ax.set_xticks(np.arange(len(pc_cols)))
    pc_labels = []
    for c in pc_cols:
        idx = int(c.replace("PC", "")) - 1
        pct = explained_df.loc[idx, "explained_variance_ratio"] * 100
        pc_labels.append(f"{c}\n({pct:.1f}%)")
    ax.set_xticklabels(pc_labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("PCA loadings: which metrics define each component?")
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig24_pca_loadings_heatmap.png")


def print_console_summary(driver_df, explained_df, loadings_df):
    print("\n" + "=" * 72)
    print("RESIDUAL ANALYSIS SUMMARY")
    print("=" * 72)
    if driver_df.empty:
        print("No residual driver correlations could be computed.")
    else:
        print("Top drivers of |recession residual|:")
        cols = ["driver", "pearson_with_abs_residual", "spearman_with_abs_residual", "n_finite"]
        print(driver_df[cols].head(8).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\n" + "=" * 72)
    print("PCA SUMMARY")
    print("=" * 72)
    print(explained_df.head(6).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for pc in ["PC1", "PC2", "PC3"]:
        if pc not in loadings_df.columns:
            continue
        tmp = loadings_df[["metric_label", pc]].copy()
        tmp["abs_loading"] = tmp[pc].abs()
        tmp = tmp.sort_values("abs_loading", ascending=False).head(5)
        print(f"\nLargest loadings on {pc}:")
        print(tmp[["metric_label", pc]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("=" * 72 + "\n")


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="92", help="Series number, e.g. 92 or 93.")
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--event_date", default=DEFAULT_EVENT_DATE)
    parser.add_argument("--results_csv", default=None, help="Optional combined results CSV to use instead of auto-detection.")
    parser.add_argument("--poly_degree", type=int, default=2, help="Polynomial degree for recession_rate_ratio ~ flowexp fit.")
    parser.add_argument("--true_ks", type=float, default=None)
    parser.add_argument("--true_cv", type=float, default=None)
    parser.add_argument("--true_r", type=float, default=None)
    parser.add_argument("--true_n", type=float, default=None)
    args = parser.parse_args()

    if args.poly_degree < 1:
        raise ValueError("--poly_degree must be 1 or greater.")

    series = str(args.series)
    script_dir = Path.cwd()
    project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
    calib_dir = project_root / "calibration_work"
    summary_dir = calib_dir / "03_comparisons" / "summary_tables"

    series_label = f"Series{series}_SynthInversion"
    plot_dir = calib_dir / "03_comparisons" / "sensitivity_plots" / series_label / "residual_pca"
    plot_dir.mkdir(parents=True, exist_ok=True)

    true_values = DEFAULT_TRUE_VALUES_BY_SERIES.get(series, {}).copy()
    if args.true_ks is not None:
        true_values["Ks_mult"] = args.true_ks
    if args.true_cv is not None:
        true_values["kinemvelcoef"] = args.true_cv
    if args.true_r is not None:
        true_values["flowexp"] = args.true_r
    if args.true_n is not None:
        true_values["channelroughness"] = args.true_n

    df, source = load_combined_or_per_run_results(
        summary_dir=summary_dir,
        location=args.location,
        event_date=args.event_date,
        series=series,
        results_csv=args.results_csv,
    )
    print(f"Loaded {len(df)} rows from {source}")
    print(f"Summary dir: {summary_dir}")

    missing_params = [c for c in PARAM_KEYS if c not in df.columns]
    if missing_params:
        raise ValueError(
            f"Missing expected parameter columns after loading/parsing: {missing_params}\n"
            f"Expected filenames like: {args.location}_{args.event_date}_{series}_Ks9p0x_cv4p0_r0p25_n0p03_metrics_summary.csv"
        )

    df = df.dropna(subset=["flowexp", "recession_rate_ratio"]).reset_index(drop=True)
    print(f"  {len(df)} rows after dropping missing flowexp/recession_rate_ratio")

    df, coeff = fit_recession_residuals(df, poly_degree=args.poly_degree)

    prefix = f"series{series}"
    augmented_path = summary_dir / f"{prefix}_residual_pca_augmented.csv"
    driver_path = summary_dir / f"{prefix}_residual_driver_correlations.csv"
    scores_path = summary_dir / f"{prefix}_pca_scores.csv"
    loadings_path = summary_dir / f"{prefix}_pca_loadings.csv"
    explained_path = summary_dir / f"{prefix}_pca_explained_variance.csv"

    df.to_csv(augmented_path, index=False)
    print(f"Saved augmented residual/PCA input table: {augmented_path.name}")

    driver_df = residual_driver_table(df)
    driver_df.to_csv(driver_path, index=False)
    print(f"Saved residual driver table: {driver_path.name}")

    scores_df, loadings_df, explained_df, metric_cols = run_metric_pca(df)
    scores_df.to_csv(scores_path, index=False)
    loadings_df.to_csv(loadings_path)
    explained_df.to_csv(explained_path, index=False)
    print(f"Saved PCA scores: {scores_path.name}")
    print(f"Saved PCA loadings: {loadings_path.name}")
    print(f"Saved PCA explained variance: {explained_path.name}")
    print(f"PCA metrics used ({len(metric_cols)}): {metric_cols}")

    print("\nMaking figures...")
    plot_r_recession_fit(df, coeff, plot_dir, series, true_values)
    plot_residual_vs_parameters(df, plot_dir, true_values)
    plot_residual_driver_bars(driver_df, plot_dir)
    plot_pca_explained(explained_df, plot_dir, series)
    plot_pca_scores(scores_df, explained_df, plot_dir, series)
    plot_pca_loadings(loadings_df, explained_df, plot_dir)

    print_console_summary(driver_df, explained_df, loadings_df)
    print(f"All residual/PCA figures saved to:\n  {plot_dir}")


if __name__ == "__main__":
    main()
