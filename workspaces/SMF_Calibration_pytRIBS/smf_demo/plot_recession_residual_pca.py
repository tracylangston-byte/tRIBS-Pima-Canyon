"""
plot_recession_residual_pca.py
==============================
Residual analysis + PCA for synthetic inversion series results.
Supersedes plot_series_residual_pca.py.

Science question
----------------
What controls the scatter around the flowexp (r) vs recession_rate_ratio
relationship?

Analysis structure
------------------
Section 1 — Core r–recession relationship
  fig19  r vs recession_rate_ratio: degree-1 fit, KGE colormap, truth lines

Section 2 — Direct residual vs parameter panels
  fig20  2×2: recession_residual vs each of the 4 swept parameters (Ks, cv, r, n)

Section 3 — Ks confound diagnostic
  fig21  Ks_mult vs recession_residual, colored by r
         Tests whether Ks errors explain residual scatter independently of r

Section 4 — Mechanistic chain test (Ks → PBIAS → recession residual)
  fig22  PBIAS vs recession_residual, colored by Ks_mult

Section 5 — Within-r-bin partial regression
  fig23  3×3 grid: rows = Ks / cv / n, cols = low / mid / high r tercile
         Within each r stratum, does any secondary parameter still correlate
         with the residual?  Confirms or rules out residual confounding.

Section 6 — Residual driver bar chart
  fig24  Two-panel horizontal bars: signed residual (left) and |residual| (right)
         Signed reveals systematic bias direction; absolute reveals scatter inflation.

Section 7 — PCA on phase-specific metrics
  fig25  Scree plot
  fig26  PC1 vs PC2 scores, 2×2 sub-panels colored by recession_residual /
         Ks_mult / flowexp / channelroughness
  fig27  Biplot: scores + loading vectors overlaid, colored by recession_residual

Section 8 — Parameter projection heatmap
  fig28  Heatmap of Pearson r between each swept parameter and each PC score.
         Rows = parameters (Ks, cv, r, n); columns = PC1 … PCk.
         Mirrors the metric-loadings heatmap style; answers "which parameter
         drives each PC?" in the same visual idiom.
         Note: these are parameter–PC-score correlations, not eigenvector
         loadings — bounded [−1, +1] by construction.

Methodology notes
-----------------
- Poly degree defaults to 1 (linear). The r–recession relationship is near-monotone
  in S91/92 (Pearson r ≈ −0.92); a degree-2 fit overfits at the tails and inflates
  residuals there artifactually.

- PCA input: phase-specific metrics ONLY (pre-peak, peak, volume).
  Excluded intentionally:
    • recession_rate_ratio — including it would cause one PC to align with the
      residual axis by construction, making the PCA appear to "find" recession
      structure when it's just recovering recession_rate_ratio itself.
    • kge / nse / rmse / kge_r / kge_alpha / kge_beta — these six summary stats
      are collinear by construction and would absorb PC1, burying the richer
      per-phase structure.

- Colorbar bug fix: fig20 and fig23 use an explicit ScalarMappable rather than
  the last scatter object, so the colorbar is always correct regardless of which
  panels are populated.

Paths (consistent with existing scripts, run from smf_demo):
    script_dir   = Path.cwd()
    project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
    calib_dir    = project_root / "calibration_work"
    plot_dir     = calib_dir / 03_comparisons / sensitivity_plots / SeriesXX_SynthInversion/
    summary_dir  = calib_dir / 03_comparisons / summary_tables/

Usage:
    python plot_recession_residual_pca.py --series 92
    python plot_recession_residual_pca.py --series 93
    python plot_recession_residual_pca.py --series 92 --poly_degree 2
    python plot_recession_residual_pca.py --series 92 --results_csv lhs_results_synth_4param_92.csv
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
DEFAULT_LOCATION   = "SMF"
DEFAULT_EVENT_DATE = "20140812"
KGE_CEILING        = 0.912   # truth self-score ceiling (S91/S92 resampling artefact)

PARAM_KEYS = ["Ks_mult", "kinemvelcoef", "flowexp", "channelroughness"]

# The three non-r parameters tested in partial regression (fig23)
SECONDARY_PARAMS = ["Ks_mult", "kinemvelcoef", "channelroughness"]

DEFAULT_TRUE_VALUES_BY_SERIES = {
    "91": {"Ks_mult": 8.50, "kinemvelcoef": 4.50, "flowexp": 0.24, "channelroughness": 0.026},
    "92": {"Ks_mult": 8.50, "kinemvelcoef": 4.50, "flowexp": 0.24, "channelroughness": 0.026},
}

# Phase-specific metrics for PCA — see methodology note in module docstring
PCA_PHASE_METRICS = [
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    "peak_error_pct",
    "peak_timing_error_hr",
    "pbias_pct",
    "volume_error_pct",
    "duration_above_thresh_error_min",
]

# Candidates tested for correlation with the recession residual.
# flowexp is deliberately excluded — it is the axis we fitted out.
RESIDUAL_DRIVER_CANDIDATES = [
    "Ks_mult",
    "kinemvelcoef",
    "channelroughness",
    "pbias_pct",
    "volume_error_pct",
    "kge_beta",
    "kge_alpha",
    "kge_r",
    "peak_error_pct",
    "peak_timing_error_hr",
    "first_arrival_error_min",
    "rising_limb_steepness_ratio",
    "time_to_peak_from_exc_min",
    "duration_above_thresh_error_min",
    "kge",
    "nse",
]


# ======================================================================
# HELPERS
# ======================================================================
def label_for(col: str) -> str:
    """Return a human-readable axis label for a parameter or metric column."""
    if col in PARAM_KEY:
        sym  = PARAM_KEY[col].get("symbol", "")
        name = PARAM_KEY[col].get("display_name", col)
        return f"{name} ({sym})" if sym else name
    if col in METRIC_KEY:
        name  = METRIC_KEY[col].get("display_name", col)
        units = METRIC_KEY[col].get("units", "")
        return f"{name} ({units})" if units else name
    fallback = {
        "Ks_mult":                         "Ks multiplier",
        "kinemvelcoef":                    "Hillslope velocity coeff (cv)",
        "flowexp":                         "Hillslope velocity exponent (r)",
        "channelroughness":                "Channel Manning's n",
        "recession_rate_ratio":            "Recession rate ratio",
        "recession_residual":              "Recession residual",
        "abs_recession_residual":          "|Recession residual|",
        "pbias_pct":                       "PBIAS (%)",
        "volume_error_pct":                "Volume error (%)",
        "peak_error_pct":                  "Peak error (%)",
        "peak_error_m3s":                  "Peak error (m\u00b3/s)",
        "peak_timing_error_hr":            "Peak timing error (hr)",
        "duration_above_thresh_error_min": "Duration above thresh error (min)",
        "first_arrival_error_min":         "First arrival error (min)",
        "rising_limb_steepness_ratio":     "Rising limb steepness ratio",
        "time_to_peak_from_exc_min":       "Time to peak from exc (min)",
        "kge":                             "KGE",
        "nse":                             "NSE",
        "rmse_m3s":                        "RMSE (m\u00b3/s)",
        "kge_r":                           "KGE r",
        "kge_alpha":                       "KGE \u03b1",
        "kge_beta":                        "KGE \u03b2",
    }
    return fallback.get(col, col)


def safe_float_from_label(text: str) -> float:
    """Convert compact run-id labels like '9p348923' to 9.348923."""
    return float(text.replace("p", "."))


def parse_run_metadata_from_filename(
    path: Path, location: str, event_date: str, series: str
) -> dict:
    """Parse run_id and 4-parameter values from a per-run metrics filename."""
    stem = path.stem
    run_id = stem[: -len("_metrics_summary")] if stem.endswith("_metrics_summary") else stem
    out = {"run_id": run_id, "source_file": path.name}
    pattern = (
        rf"^{re.escape(location)}_{re.escape(event_date)}_{re.escape(str(series))}_"
        r"Ks(?P<Ks>[0-9p]+)x_cv(?P<cv>[0-9p]+)_r(?P<r>[0-9p]+)_n(?P<n>[0-9p]+)"
    )
    m = re.search(pattern, run_id)
    if m:
        out["Ks_mult"]          = safe_float_from_label(m.group("Ks"))
        out["kinemvelcoef"]     = safe_float_from_label(m.group("cv"))
        out["flowexp"]          = safe_float_from_label(m.group("r"))
        out["channelroughness"] = safe_float_from_label(m.group("n"))
    return out


def normalize_metrics_summary_csv(path: Path) -> pd.DataFrame:
    """Read a per-run metrics summary CSV; handles both wide and long formats."""
    raw = pd.read_csv(path)
    if raw.empty:
        return raw
    lower_map = {str(c).strip().lower(): c for c in raw.columns}
    metric_col = next(
        (lower_map[k] for k in ["metric", "metrics", "name", "stat", "statistic"]
         if k in lower_map), None
    )
    value_col = next(
        (lower_map[k] for k in ["value", "values", "val"] if k in lower_map), None
    )
    if metric_col is not None and value_col is not None:
        wide = raw[[metric_col, value_col]].dropna(subset=[metric_col]).copy()
        wide[metric_col] = wide[metric_col].astype(str)
        wide = wide.drop_duplicates(subset=[metric_col], keep="first")
        return wide.set_index(metric_col)[value_col].to_frame().T.reset_index(drop=True)
    return raw.copy()


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-identifier columns to numeric where possible."""
    id_like = {"run_id", "source_file", "change_tested", "series", "location", "event_date"}
    out = df.copy()
    for col in out.columns:
        if col in id_like:
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().sum() > 0:
            out[col] = converted
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv, yv = x[mask], y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if not HAS_SCIPY:
        return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv, yv = x[mask], y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan
    rho, _ = spearmanr(xv, yv)
    return float(rho)


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    stds = out.std(axis=0, ddof=1).replace(0, np.nan)
    return (out - out.mean(axis=0)) / stds


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


# ======================================================================
# LOADING
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
    Prefer a combined series CSV if present.
    Falls back to assembling from per-run *_metrics_summary.csv files.
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
                    f"Candidate combined names: {candidate_combined_csv_names(series)}\n"
                    f"Per-run pattern: {summary_dir / pattern}\n"
                    f"Tip: run from smf_demo so paths resolve to "
                    f"../calibration_work/03_comparisons/summary_tables"
                )
            rows = []
            for p in files:
                meta = parse_run_metadata_from_filename(p, location, event_date, series)
                one = normalize_metrics_summary_csv(p)
                if one.empty:
                    print(f"  Warning: skipped empty file {p.name}")
                    continue
                for _, row in one.iterrows():
                    rec = meta.copy()
                    rec.update(row.to_dict())
                    rows.append(rec)
            df = pd.DataFrame(rows)
            source = f"{len(files)} per-run metric summary files matching {pattern}"

    df = coerce_numeric_columns(df)

    if "run_id" not in df.columns:
        if "source_file" in df.columns:
            df["run_id"] = (
                df["source_file"].astype(str)
                .str.replace("_metrics_summary.csv", "", regex=False)
            )
        else:
            df["run_id"] = [f"series{series}_row{i:04d}" for i in range(len(df))]

    # Recover parameter columns from run_id if they are missing in the combined CSV
    missing_params = [p for p in PARAM_KEYS if p not in df.columns]
    if missing_params:
        parsed_rows = [
            parse_run_metadata_from_filename(
                Path(f"{rid}_metrics_summary.csv"), location, event_date, series
            )
            for rid in df["run_id"].astype(str)
        ]
        parsed_df = pd.DataFrame(parsed_rows)
        for p in missing_params:
            if p in parsed_df.columns:
                df[p] = parsed_df[p].values

    # Drop rows from other series that may have been pulled in by a wildcard CSV
    prefix = f"{location}_{event_date}_{series}_"
    if "run_id" in df.columns:
        mask = df["run_id"].astype(str).str.startswith(prefix)
        if mask.any():
            df = df.loc[mask].copy()

    return df.reset_index(drop=True), source


# ======================================================================
# CORE ANALYSIS
# ======================================================================
def fit_recession_residuals(
    df: pd.DataFrame, poly_degree: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Fit polynomial(flowexp, recession_rate_ratio) and add residual columns.

    Adds to df:
        recession_fit_from_flowexp  — fitted y-values
        recession_residual          — observed minus fitted
        abs_recession_residual      — |recession_residual|
    """
    x = df["flowexp"].to_numpy(dtype=float)
    y = df["recession_rate_ratio"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < poly_degree + 2:
        raise ValueError(
            f"Insufficient finite rows for degree-{poly_degree} fit "
            f"(have {mask.sum()}, need ≥ {poly_degree + 2})."
        )
    coeff = np.polyfit(x[mask], y[mask], deg=poly_degree)
    yhat = np.full(len(df), np.nan)
    yhat[mask] = np.polyval(coeff, x[mask])

    out = df.copy()
    out["recession_fit_from_flowexp"] = yhat
    out["recession_residual"]         = y - yhat
    out["abs_recession_residual"]     = np.abs(out["recession_residual"])
    return out, coeff


def residual_driver_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r and Spearman ρ between each candidate driver and
    (a) the signed recession residual and (b) |recession residual|.

    Returns a DataFrame sorted by the highest absolute correlation with
    |residual|, which ranks drivers by their ability to explain scatter.
    """
    y_signed = df["recession_residual"].to_numpy(dtype=float)
    y_abs    = df["abs_recession_residual"].to_numpy(dtype=float)
    rows = []
    for col in RESIDUAL_DRIVER_CANDIDATES:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        rows.append({
            "driver":          col,
            "driver_label":    label_for(col),
            "pearson_signed":  safe_pearson(x, y_signed),
            "spearman_signed": safe_spearman(x, y_signed),
            "pearson_abs":     safe_pearson(x, y_abs),
            "spearman_abs":    safe_spearman(x, y_abs),
            "n_finite":        int((np.isfinite(x) & np.isfinite(y_signed)).sum()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_score"] = (
        out[["pearson_abs", "spearman_abs"]].abs().max(axis=1, skipna=True)
    )
    return out.sort_values("rank_score", ascending=False).reset_index(drop=True)


def run_phase_pca(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    PCA on phase-specific metrics only.

    Excludes recession_rate_ratio (would create circularity when scores are
    colored by recession_residual) and all summary stats (KGE family — they
    are collinear and would dominate PC1 without adding interpretive value).

    Returns:
        scores_df    — shape (n_runs, n_pcs), with run_id and param columns appended
        loadings_df  — shape (n_metrics, n_pcs), rows indexed by metric name
        explained_df — eigenvalues, explained variance ratio, cumulative
        metric_cols  — list of metric column names that passed the quality filter
    """
    available = [c for c in PCA_PHASE_METRICS if c in df.columns]
    if not available:
        raise ValueError(
            "No phase-specific PCA metrics found in data. "
            f"Looking for: {PCA_PHASE_METRICS}"
        )

    Xraw = df[available].copy().astype(float)
    keep = [
        c for c in available
        if np.isfinite(Xraw[c].to_numpy()).sum() >= 10
        and Xraw[c].std(ddof=1) > 0
    ]
    if len(keep) < 2:
        raise ValueError(f"Need ≥2 usable metrics for PCA; kept only {keep}")

    Xraw = Xraw[keep]
    Xfilled = Xraw.copy()
    for c in Xfilled.columns:
        Xfilled[c] = Xfilled[c].fillna(Xfilled[c].median())

    Xz = zscore_frame(Xfilled)
    if Xz.isna().any().any():
        bad = Xz.columns[Xz.isna().any()].tolist()
        raise ValueError(f"NaN survived z-scoring in: {bad}")

    X = Xz.to_numpy(dtype=float)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = X.shape[0]
    eigenvalues = (S ** 2) / (n - 1)
    explained   = eigenvalues / eigenvalues.sum()
    pc_names    = [f"PC{i + 1}" for i in range(len(S))]

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
        "component":                      pc_names,
        "eigenvalue":                     eigenvalues,
        "explained_variance_ratio":       explained,
        "cumulative_explained_variance":  np.cumsum(explained),
    })
    return scores_df, loadings_df, explained_df, keep


# ======================================================================
# PLOT HELPERS
# ======================================================================
def _add_truth_vline(ax, param: str, true_values: dict, **kwargs) -> None:
    if param in true_values:
        defaults = dict(color="red", linestyle="--", linewidth=1.1,
                        label=f"Truth = {true_values[param]}")
        defaults.update(kwargs)
        ax.axvline(true_values[param], **defaults)


def _pearson_text(ax, x: np.ndarray, y: np.ndarray, loc: str = "bl") -> None:
    r = safe_pearson(x, y)
    if np.isnan(r):
        return
    pos = {"bl": (0.04, 0.05), "br": (0.96, 0.05), "tl": (0.04, 0.93)}[loc]
    ha  = "right" if loc == "br" else "left"
    ax.text(*pos, f"Pearson r = {r:+.3f}", transform=ax.transAxes,
            fontsize=9, ha=ha, va="bottom",
            bbox=dict(fc="white", alpha=0.85, edgecolor="none", pad=1.5))


def _linear_trend(ax, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    coeff = np.polyfit(x[mask], y[mask], 1)
    xf = np.linspace(x[mask].min(), x[mask].max(), 200)
    defaults = dict(color="black", linewidth=1.8, linestyle="-", label="Linear trend", zorder=5)
    defaults.update(kwargs)
    ax.plot(xf, np.polyval(coeff, xf), **defaults)


# ======================================================================
# SECTION 1 — Fig 19: r vs recession_rate_ratio
# ======================================================================
def plot_r_recession_fit(
    df: pd.DataFrame, coeff: np.ndarray,
    plot_dir: Path, series: str, true_values: dict, poly_degree: int
) -> None:
    """fig19 — the primary r–recession scatter with fitted trend."""
    fig, ax = plt.subplots(figsize=(8.5, 6))

    cvals = df["kge"].to_numpy(dtype=float) if "kge" in df.columns else df["Ks_mult"].to_numpy(dtype=float)
    sc = ax.scatter(
        df["flowexp"], df["recession_rate_ratio"],
        c=cvals, cmap="plasma", s=45,
        edgecolors="white", linewidths=0.4, alpha=0.9
    )
    fig.colorbar(sc, ax=ax, label=label_for("kge") if "kge" in df.columns else label_for("Ks_mult"))

    xfit = np.linspace(df["flowexp"].min(), df["flowexp"].max(), 300)
    yfit = np.polyval(coeff, xfit)
    degree_label = "linear" if poly_degree == 1 else f"deg-{poly_degree} poly"
    ax.plot(xfit, yfit, color="black", linewidth=2.0,
            label=f"Fitted trend ({degree_label})")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2,
               label="Perfect recession ratio = 1")
    _add_truth_vline(ax, "flowexp", true_values)

    # Global Pearson r annotation
    _pearson_text(
        ax,
        df["flowexp"].to_numpy(dtype=float),
        df["recession_rate_ratio"].to_numpy(dtype=float),
    )

    ax.set_xlabel("Hillslope velocity exponent r (flowexp)")
    ax.set_ylabel("Recession rate ratio")
    ax.set_title(
        f"Series {series}: r vs recession rate ratio\n"
        f"Residuals = vertical distance from the fitted trend"
    )
    ax.legend(fontsize=9, facecolor="white", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig19_r_recession_fit.png")


# ======================================================================
# SECTION 2 — Fig 20: residual vs all 4 parameters
# ======================================================================
def plot_residual_vs_parameters(
    df: pd.DataFrame, plot_dir: Path, true_values: dict
) -> None:
    """fig20 — 2×2: recession_residual vs each swept parameter, colored by KGE."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    kge_vals = (
        df["kge"].to_numpy(dtype=float) if "kge" in df.columns
        else df["flowexp"].to_numpy(dtype=float)
    )
    # Explicit ScalarMappable so the colorbar is correct regardless of
    # which axes are populated (avoids the "last sc" fragility bug).
    norm = plt.Normalize(vmin=np.nanmin(kge_vals), vmax=np.nanmax(kge_vals))
    cmap = plt.cm.plasma

    for ax, p in zip(axes, PARAM_KEYS):
        if p not in df.columns:
            ax.axis("off")
            continue
        ax.scatter(
            df[p], df["recession_residual"],
            c=kge_vals, cmap=cmap, norm=norm,
            s=42, edgecolors="white", linewidths=0.35, alpha=0.9
        )
        ax.axhline(0, color="black", linewidth=1.0)
        _add_truth_vline(ax, p, true_values)
        _pearson_text(
            ax,
            df[p].to_numpy(dtype=float),
            df["recession_residual"].to_numpy(dtype=float),
        )
        ax.set_title(f"Residual vs {label_for(p)}", fontsize=10)
        ax.set_xlabel(label_for(p))
        ax.set_ylabel("Recession residual")
        ax.grid(linestyle=":", alpha=0.35)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.85)
    cbar.set_label("KGE" if "kge" in df.columns else "flowexp")

    fig.suptitle(
        "Recession residual vs each parameter\n"
        "What survives after removing the main r–recession trend?",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, plot_dir / "fig20_recession_residual_vs_parameters.png")


# ======================================================================
# SECTION 3 — Fig 21: Ks_mult vs recession_residual colored by r
# ======================================================================
def plot_ks_vs_residual(
    df: pd.DataFrame, plot_dir: Path, true_values: dict
) -> None:
    """
    fig21 — Central Ks confound diagnostic.
    Does Ks error explain recession residual independently of r?
    """
    if "Ks_mult" not in df.columns:
        print("  Skipping fig21: Ks_mult column not found.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))

    cvals = df["flowexp"].to_numpy(dtype=float)
    sc = ax.scatter(
        df["Ks_mult"], df["recession_residual"],
        c=cvals, cmap="viridis", s=48,
        edgecolors="white", linewidths=0.4, alpha=0.9
    )
    fig.colorbar(sc, ax=ax, label="flowexp (r)")

    _linear_trend(ax, df["Ks_mult"].to_numpy(dtype=float),
                  df["recession_residual"].to_numpy(dtype=float))
    ax.axhline(0, color="gray", linewidth=1.0, linestyle=":", zorder=1)
    _add_truth_vline(ax, "Ks_mult", true_values)

    _pearson_text(
        ax,
        df["Ks_mult"].to_numpy(dtype=float),
        df["recession_residual"].to_numpy(dtype=float),
    )

    ax.set_xlabel("Ks multiplier")
    ax.set_ylabel("Recession residual")
    ax.set_title(
        f"Ks_mult vs recession residual, colored by r\n"
        f"If Pearson r ≠ 0, Ks errors confound the r–recession relationship"
    )
    ax.legend(fontsize=9, facecolor="white", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig21_Ks_vs_recession_residual.png")


# ======================================================================
# SECTION 4 — Fig 22: PBIAS vs recession_residual (mechanistic chain)
# ======================================================================
def plot_pbias_vs_residual(
    df: pd.DataFrame, plot_dir: Path, true_values: dict
) -> None:
    """
    fig22 — Mechanistic chain test.
    Ks error → volume error (PBIAS) → recession residual.
    If PBIAS correlates with the residual, the chain is closed.
    """
    if "pbias_pct" not in df.columns:
        print("  Skipping fig22: pbias_pct column not found.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))

    cvals = (
        df["Ks_mult"].to_numpy(dtype=float) if "Ks_mult" in df.columns
        else df["flowexp"].to_numpy(dtype=float)
    )
    sc = ax.scatter(
        df["pbias_pct"], df["recession_residual"],
        c=cvals, cmap="viridis", s=48,
        edgecolors="white", linewidths=0.4, alpha=0.9
    )
    cbar_label = label_for("Ks_mult") if "Ks_mult" in df.columns else label_for("flowexp")
    fig.colorbar(sc, ax=ax, label=cbar_label)

    _linear_trend(ax, df["pbias_pct"].to_numpy(dtype=float),
                  df["recession_residual"].to_numpy(dtype=float))
    ax.axhline(0, color="gray", linewidth=1.0, linestyle=":", label="Zero residual")
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="-.", label="Zero PBIAS", alpha=0.6)

    _pearson_text(
        ax,
        df["pbias_pct"].to_numpy(dtype=float),
        df["recession_residual"].to_numpy(dtype=float),
    )

    ax.set_xlabel("PBIAS (%)")
    ax.set_ylabel("Recession residual")
    ax.set_title(
        "PBIAS vs recession residual, colored by Ks multiplier\n"
        "Mechanistic chain test: Ks error \u2192 PBIAS \u2192 recession residual"
    )
    ax.legend(fontsize=9, facecolor="white", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig22_pbias_vs_recession_residual.png")


# ======================================================================
# SECTION 5 — Fig 23: Within-r-bin partial regression
# ======================================================================
def plot_within_rbin_partial_regression(
    df: pd.DataFrame, plot_dir: Path, true_values: dict
) -> None:
    """
    fig23 — 3×3 partial regression grid.
    Rows: Ks_mult / kinemvelcoef / channelroughness
    Cols: low / mid / high flowexp tercile

    Within each r stratum, does the secondary parameter still correlate
    with the recession residual?
      - Consistent correlation across bins → genuine secondary driver
      - Correlation vanishes within bins  → spurious (driven by LHS joint dist.)
    """
    secondary = [p for p in SECONDARY_PARAMS if p in df.columns]
    if not secondary or "flowexp" not in df.columns:
        print("  Skipping fig23: required columns not present.")
        return

    df_work = df.copy()
    try:
        df_work["r_bin"] = pd.qcut(
            df_work["flowexp"], q=3, labels=["Low r", "Mid r", "High r"]
        )
    except Exception as e:
        print(f"  Skipping fig23: could not form r terciles ({e})")
        return

    n_rows = len(secondary)
    bin_labels = ["Low r", "Mid r", "High r"]

    fig, axes = plt.subplots(
        n_rows, 3, figsize=(13, 4.2 * n_rows), sharey=False
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    kge_all = df_work["kge"].to_numpy(dtype=float) if "kge" in df_work.columns else None
    norm = plt.Normalize(
        vmin=np.nanmin(kge_all), vmax=np.nanmax(kge_all)
    ) if kge_all is not None else plt.Normalize(0, 1)
    cmap = plt.cm.plasma

    for row_i, param in enumerate(secondary):
        for col_i, bin_label in enumerate(bin_labels):
            ax = axes[row_i, col_i]
            sub = df_work[df_work["r_bin"] == bin_label]

            xv = sub[param].to_numpy(dtype=float)
            yr = sub["recession_residual"].to_numpy(dtype=float)
            ck = (
                sub["kge"].to_numpy(dtype=float) if "kge" in sub.columns
                else np.full(len(sub), 0.5)
            )

            ax.scatter(xv, yr, c=ck, cmap=cmap, norm=norm,
                       s=38, edgecolors="white", linewidths=0.3, alpha=0.9)
            ax.axhline(0, color="black", linewidth=0.8)
            _add_truth_vline(ax, param, true_values, linewidth=0.9)

            r_val = safe_pearson(xv, yr)
            n_bin = int((np.isfinite(xv) & np.isfinite(yr)).sum())
            ax.set_title(
                f"{bin_label}  (n={n_bin})\nPearson r = {r_val:+.3f}",
                fontsize=9.5
            )
            ax.set_xlabel(label_for(param) if row_i == n_rows - 1 else "")
            ax.grid(linestyle=":", alpha=0.3)

        # Row label on left axis
        axes[row_i, 0].set_ylabel(
            f"{label_for(param)}\n\nRecession residual", fontsize=9
        )

    # Shared colorbar using explicit ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="KGE", shrink=0.55, pad=0.02)

    fig.suptitle(
        "Within-r-bin partial regression\n"
        "Does each secondary parameter explain residual variance within r strata?",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.94, 0.95])
    save_fig(fig, plot_dir / "fig23_within_rbin_partial_regression.png")


# ======================================================================
# SECTION 6 — Fig 24: Signed + absolute driver correlation bars
# ======================================================================
def plot_driver_bars(driver_df: pd.DataFrame, plot_dir: Path) -> None:
    """
    fig24 — Two-panel horizontal bar chart.
    Left  (signed): reveals systematic bias direction
                    (+) = higher driver value → residual too large (model recedes too slow)
    Right (absolute): reveals scatter inflation
                    (+) = higher driver value → larger residual magnitude
    Both panels sorted by |abs| correlation so each driver occupies the same row.
    """
    if driver_df.empty:
        print("  Skipping fig24: empty driver table.")
        return

    top = driver_df.head(12).copy().iloc[::-1]  # reversed: highest at top
    labels = top["driver_label"].tolist()

    fig, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(14, 6.5))

    def _bar_panel(ax, vals, title, xlabel):
        colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
        ax.barh(labels, vals, color=colors, edgecolor="white", height=0.65)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1.05, 1.05)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    _bar_panel(
        ax_s,
        top["pearson_signed"].to_numpy(dtype=float),
        "Systematic bias direction\n(+) \u2192 higher driver = residual skews positive",
        "Pearson r with signed recession residual",
    )
    _bar_panel(
        ax_a,
        top["pearson_abs"].to_numpy(dtype=float),
        "Scatter inflation\n(+) \u2192 higher driver = larger residual magnitude",
        "Pearson r with |recession residual|",
    )
    # Remove duplicate y-tick labels on right panel
    ax_a.set_yticklabels([])

    fig.suptitle(
        "Top drivers of recession residual (sorted by |abs| Pearson r)\n"
        "Same row order in both panels for direct comparison",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig24_residual_driver_bars.png")


# ======================================================================
# SECTION 7 — PCA: Figs 25–27
# ======================================================================
def plot_pca_scree(
    explained_df: pd.DataFrame, plot_dir: Path, series: str
) -> None:
    """fig25 — Scree plot for phase-specific PCA."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(explained_df) + 1)
    ax.bar(x, explained_df["explained_variance_ratio"] * 100,
           edgecolor="white", color="#457b9d", label="Individual")
    ax.plot(x, explained_df["cumulative_explained_variance"] * 100,
            marker="o", color="black", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(
        f"Series {series}: PCA scree — phase-specific metrics only\n"
        f"(Excludes recession_rate_ratio and KGE summary stats)"
    )
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig25_pca_scree.png")


def plot_pca_scores_multipanel(
    scores_df: pd.DataFrame, explained_df: pd.DataFrame,
    plot_dir: Path, series: str
) -> None:
    """
    fig26 — PC1 vs PC2 scores, 2×2 sub-panels each colored by a different
    variable.  Same point positions; different color mappings reveal which
    PCs align with which parameters or the recession residual.
    """
    if "PC1" not in scores_df.columns or "PC2" not in scores_df.columns:
        return

    pc1_pct = explained_df.loc[0, "explained_variance_ratio"] * 100
    pc2_pct = explained_df.loc[1, "explained_variance_ratio"] * 100

    color_specs = [
        ("recession_residual", "RdBu_r",  True,  "Recession residual"),
        ("Ks_mult",            "viridis",  False, "Ks multiplier"),
        ("flowexp",            "plasma",   False, "flowexp (r)"),
        ("channelroughness",   "cividis",  False, "Channel Manning's n"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for ax, (col, cmap_name, diverge, cbar_label) in zip(axes, color_specs):
        if col not in scores_df.columns:
            ax.axis("off")
            continue
        cvals = scores_df[col].to_numpy(dtype=float)
        if diverge:
            finite = cvals[np.isfinite(cvals)]
            vmax = float(np.nanpercentile(np.abs(finite), 95)) if len(finite) else 1.0
            vmax = vmax if vmax > 0 else 1.0
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            norm = plt.Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))

        sc = ax.scatter(
            scores_df["PC1"], scores_df["PC2"],
            c=cvals, cmap=cmap_name, norm=norm,
            s=42, edgecolors="white", linewidths=0.35, alpha=0.9
        )
        fig.colorbar(sc, ax=ax, label=cbar_label)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.axvline(0, color="gray", linewidth=0.7)
        ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)")
        ax.set_title(f"Colored by: {cbar_label}", fontsize=10)
        ax.grid(linestyle=":", alpha=0.3)

    fig.suptitle(
        f"Series {series}: PCA scores (PC1 vs PC2) — 4 color mappings\n"
        f"Reveals which PCs align with volume (Ks), timing (r), or recession residual",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, plot_dir / "fig26_pca_scores_multipanel.png")


def plot_pca_biplot(
    scores_df: pd.DataFrame, loadings_df: pd.DataFrame,
    explained_df: pd.DataFrame, plot_dir: Path, series: str
) -> None:
    """
    fig27 — Biplot: PC scores + metric loading vectors overlaid.
    Scores colored by recession_residual.  Arrow tips show which metrics
    drive each PC direction; overlap with residual coloring reveals
    mechanistic links.
    """
    if "PC1" not in scores_df.columns or "PC2" not in scores_df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    color_col = (
        "recession_residual" if "recession_residual" in scores_df.columns else "kge"
    )
    cvals = scores_df[color_col].to_numpy(dtype=float)

    if color_col == "recession_residual":
        finite = cvals[np.isfinite(cvals)]
        vmax = float(np.nanpercentile(np.abs(finite), 95)) if len(finite) else 1.0
        vmax = vmax if vmax > 0 else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = "RdBu_r"
    else:
        norm = plt.Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))
        cmap = "plasma"

    sc = ax.scatter(
        scores_df["PC1"], scores_df["PC2"],
        c=cvals, cmap=cmap, norm=norm,
        s=38, edgecolors="white", linewidths=0.35, alpha=0.85, zorder=2
    )
    fig.colorbar(sc, ax=ax, label=label_for(color_col), shrink=0.82)

    # Scale loading vectors to fit comfortably inside the score cloud
    pc1_range = float(scores_df["PC1"].abs().max())
    pc2_range = float(scores_df["PC2"].abs().max())
    load_scale = 0.40 * max(pc1_range, pc2_range)

    pc1_loads  = loadings_df["PC1"].to_numpy(dtype=float)
    pc2_loads  = loadings_df["PC2"].to_numpy(dtype=float)
    load_labels = loadings_df["metric_label"].tolist()
    arrow_color = "#e63946"

    for lx, ly, lbl in zip(pc1_loads, pc2_loads, load_labels):
        ax.annotate(
            "",
            xy=(lx * load_scale, ly * load_scale),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>", color=arrow_color, lw=1.5, mutation_scale=12
            ),
            zorder=5,
        )
        ax.text(
            lx * load_scale * 1.14, ly * load_scale * 1.14, lbl,
            fontsize=7.5, color=arrow_color, ha="center", va="center",
            zorder=6,
            bbox=dict(fc="white", alpha=0.65, edgecolor="none", pad=1.2),
        )

    pc1_pct = explained_df.loc[0, "explained_variance_ratio"] * 100
    pc2_pct = explained_df.loc[1, "explained_variance_ratio"] * 100
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance)")
    ax.axhline(0, color="gray", linewidth=0.7, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.7, zorder=1)
    ax.set_title(
        f"Series {series}: PCA biplot — phase-specific metrics\n"
        f"Scores colored by {label_for(color_col)}; arrows = metric loading vectors",
        fontsize=11,
    )
    ax.grid(linestyle=":", alpha=0.25, zorder=0)
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig27_pca_biplot.png")


# ======================================================================
# SECTION 8 — Fig 28: parameter projection heatmap
# ======================================================================
def plot_pca_param_projection_heatmap(
    scores_df: pd.DataFrame, explained_df: pd.DataFrame,
    plot_dir: Path, series: str,
) -> None:
    """
    fig28 — Heatmap of Pearson r between each swept parameter and each PC score.

    Rows = swept parameters (Ks, cv, r, n).
    Columns = PC1 … PCk (components explaining ≥2 % variance, up to 6).
    Color = diverging RdBu_r, vmin=−1, vmax=+1 — same scheme as the
            metric-loadings heatmap in plot_series_residual_pca.py.

    These are parameter–PC-score *correlations*, not eigenvector loadings.
    The distinction matters: a near-unity value for (Ks, PC2) means Ks
    almost perfectly explains the variance PC2 captures in metric space,
    but it does not mean Ks has a unit coefficient in the eigenvector.
    """
    # ---- Determine which PCs to show --------------------------------
    all_pcs = [c for c in explained_df["component"] if c in scores_df.columns]
    if not all_pcs:
        return
    evr = explained_df.set_index("component")["explained_variance_ratio"]
    pc_names = [p for p in all_pcs if evr.get(p, 0) >= 0.02][:6]
    if not pc_names:                        # fallback: first 4
        pc_names = all_pcs[:4]

    # ---- Parameters present in scores_df ---------------------------
    params = [p for p in PARAM_KEYS if p in scores_df.columns]
    if not params:
        return

    # ---- Correlation matrix (n_params × n_pcs) ---------------------
    corr = np.full((len(params), len(pc_names)), np.nan)
    for i, param in enumerate(params):
        x = scores_df[param].to_numpy(dtype=float)
        for j, pc in enumerate(pc_names):
            y = scores_df[pc].to_numpy(dtype=float)
            corr[i, j] = safe_pearson(x, y)

    param_labels = [label_for(p) for p in params]
    pc_labels = [
        f"{pc}\n({evr[pc] * 100:.1f}%)"
        for pc in pc_names
    ]

    # ---- Figure layout ---------------------------------------------
    fig_w = max(6.0, 1.4 * len(pc_names) + 2.5)
    fig_h = max(3.5, 0.75 * len(params) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=10)

    ax.set_xticks(range(len(pc_names)))
    ax.set_xticklabels(pc_labels, fontsize=10)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(param_labels, fontsize=11)

    # Cell annotations
    for i in range(len(params)):
        for j in range(len(pc_names)):
            val = corr[i, j]
            if np.isfinite(val):
                txt_color = "white" if abs(val) > 0.65 else "black"
                ax.text(j, i, f"{val:+.2f}",
                        ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight="bold")

    # White grid lines between cells
    for xp in np.arange(-0.5, len(pc_names), 1):
        ax.axvline(xp, color="white", linewidth=0.9)
    for yp in np.arange(-0.5, len(params), 1):
        ax.axhline(yp, color="white", linewidth=0.9)

    ax.set_title(
        f"Series {series}: which parameters drive each PCA component?\n"
        f"Pearson r (parameter vs PC score) — phase-specific metric PCA",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, plot_dir / "fig28_pca_param_projection.png")


# ======================================================================
# CONSOLE SUMMARY
# ======================================================================
def print_console_summary(
    driver_df: pd.DataFrame,
    explained_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    series: str,
) -> None:
    print("\n" + "=" * 72)
    print(f"SERIES {series} — RECESSION RESIDUAL & PCA SUMMARY")
    print("=" * 72)

    if not driver_df.empty:
        print(
            "\nTop drivers of recession residual (sorted by |abs| Pearson r):\n"
            "  'signed' = direction of bias  |  'abs' = scatter inflation"
        )
        cols = [
            "driver", "pearson_signed", "spearman_signed",
            "pearson_abs", "spearman_abs", "n_finite",
        ]
        avail = [c for c in cols if c in driver_df.columns]
        print(
            driver_df[avail].head(10).to_string(
                index=False, float_format=lambda v: f"{v:+.3f}"
            )
        )

    print("\n" + "=" * 72)
    print("PCA EXPLAINED VARIANCE (phase-specific metrics, no summary stats)")
    print("=" * 72)
    print(
        explained_df.head(6).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )

    for pc in ["PC1", "PC2", "PC3"]:
        if pc not in loadings_df.columns:
            continue
        tmp = loadings_df[["metric_label", pc]].copy()
        tmp["abs"] = tmp[pc].abs()
        tmp = tmp.sort_values("abs", ascending=False).head(5)
        print(f"\nLargest loadings on {pc}:")
        print(
            tmp[["metric_label", pc]].to_string(
                index=False, float_format=lambda v: f"{v:+.3f}"
            )
        )
    print("=" * 72 + "\n")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recession residual analysis + PCA for synthetic inversion series."
    )
    parser.add_argument("--series",      default="92",
                        help="Series number, e.g. 92 or 93.")
    parser.add_argument("--location",    default=DEFAULT_LOCATION)
    parser.add_argument("--event_date",  default=DEFAULT_EVENT_DATE)
    parser.add_argument("--results_csv", default=None,
                        help="Override auto-detection with a specific combined CSV.")
    parser.add_argument("--poly_degree", type=int, default=1,
                        help="Polynomial degree for r–recession fit. "
                             "Default 1 (linear) — the relationship is near-monotone "
                             "in S91/92. Use 2 only to test sensitivity.")
    parser.add_argument("--true_ks",    type=float, default=None)
    parser.add_argument("--true_cv",    type=float, default=None)
    parser.add_argument("--true_r",     type=float, default=None)
    parser.add_argument("--true_n",     type=float, default=None)
    args = parser.parse_args()

    if args.poly_degree < 1:
        raise ValueError("--poly_degree must be ≥ 1.")

    series       = str(args.series)
    script_dir   = Path.cwd()
    project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
    calib_dir    = project_root / "calibration_work"
    summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
    series_label = f"Series{series}_SynthInversion"
    plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / series_label
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Override truth values from CLI if provided
    true_values = DEFAULT_TRUE_VALUES_BY_SERIES.get(series, {}).copy()
    if args.true_ks is not None: true_values["Ks_mult"]          = args.true_ks
    if args.true_cv is not None: true_values["kinemvelcoef"]     = args.true_cv
    if args.true_r  is not None: true_values["flowexp"]          = args.true_r
    if args.true_n  is not None: true_values["channelroughness"] = args.true_n

    # ---- Load -------------------------------------------------------
    df, source = load_combined_or_per_run_results(
        summary_dir  = summary_dir,
        location     = args.location,
        event_date   = args.event_date,
        series       = series,
        results_csv  = args.results_csv,
    )
    print(f"Loaded {len(df)} rows from {source}")

    missing_params = [c for c in PARAM_KEYS if c not in df.columns]
    if missing_params:
        raise ValueError(
            f"Missing parameter columns after loading: {missing_params}\n"
            f"Expected filenames like: "
            f"{args.location}_{args.event_date}_{series}_"
            f"Ks9p0x_cv4p0_r0p25_n0p03_metrics_summary.csv"
        )

    if "recession_rate_ratio" not in df.columns:
        raise ValueError(
            "recession_rate_ratio column not found. "
            "Rerun scoring with phase-specific metrics enabled."
        )

    df = df.dropna(subset=["flowexp", "recession_rate_ratio"]).reset_index(drop=True)
    print(f"  {len(df)} rows after dropping missing flowexp / recession_rate_ratio")

    # ---- Residual fit -----------------------------------------------
    df, coeff = fit_recession_residuals(df, poly_degree=args.poly_degree)
    prefix = f"series{series}"
    out_path = summary_dir / f"{prefix}_residual_augmented.csv"
    df.to_csv(out_path, index=False)
    print(f"Augmented table saved: {out_path.name}")

    # ---- Driver table -----------------------------------------------
    driver_df = residual_driver_table(df)
    driver_path = summary_dir / f"{prefix}_residual_drivers.csv"
    driver_df.to_csv(driver_path, index=False)
    print(f"Driver table saved ({len(driver_df)} candidates): {driver_path.name}")

    # ---- PCA --------------------------------------------------------
    scores_df, loadings_df, explained_df, metric_cols = run_phase_pca(df)
    scores_df.to_csv(summary_dir / f"{prefix}_pca_scores.csv", index=False)
    loadings_df.to_csv(summary_dir / f"{prefix}_pca_loadings.csv")
    explained_df.to_csv(summary_dir / f"{prefix}_pca_explained.csv", index=False)
    print(f"PCA computed on {len(metric_cols)} phase metrics: {metric_cols}")

    # ---- Figures ----------------------------------------------------
    print(f"\nGenerating figures → {plot_dir}\n")
    plot_r_recession_fit(df, coeff, plot_dir, series, true_values, args.poly_degree)
    plot_residual_vs_parameters(df, plot_dir, true_values)
    plot_ks_vs_residual(df, plot_dir, true_values)
    plot_pbias_vs_residual(df, plot_dir, true_values)
    plot_within_rbin_partial_regression(df, plot_dir, true_values)
    plot_driver_bars(driver_df, plot_dir)
    plot_pca_scree(explained_df, plot_dir, series)
    plot_pca_scores_multipanel(scores_df, explained_df, plot_dir, series)
    plot_pca_biplot(scores_df, loadings_df, explained_df, plot_dir, series)
    plot_pca_param_projection_heatmap(scores_df, explained_df, plot_dir, series)

    print_console_summary(driver_df, explained_df, loadings_df, series)
    print(f"Done. All outputs in:\n  {plot_dir}\n  {summary_dir}")


if __name__ == "__main__":
    main()