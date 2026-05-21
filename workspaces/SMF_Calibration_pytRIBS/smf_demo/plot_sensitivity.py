"""
plot_sensitivity.py
===================
Generates all sensitivity analysis figures for any single-parameter sweep.
Works for any parameter in sensitivity_results_all.csv.

Usage (run from the smf_demo directory):
    python plot_sensitivity.py --param Ks_mult
    python plot_sensitivity.py --param f_RS_abs
    python plot_sensitivity.py --param f_RS_abs_Ks1
    python plot_sensitivity.py --param kinemvelcoef
    python plot_sensitivity.py --param flowexp
    python plot_sensitivity.py --param channelroughness

Produces 5 figures saved to:
    calibration_work/03_comparisons/sensitivity_plots/{param_name}/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path

# -----------------------------------------------------------------------
# PARAMETER METADATA
# Add entries here when new parameters are added to the sweep.
# -----------------------------------------------------------------------
PARAM_META = {
    "Ks_mult": {
        "xlabel":       "Ks multiplier",
        "log_x":        True,
        "tick_fmt":     lambda v, _: f"{v:g}x",
        "colorbar_lbl": "Ks multiplier",
        "color_low":    "Purple = low Ks (too much runoff)",
        "color_high":   "Yellow = high Ks (too little runoff)",
        "baseline":     1.0,
        "sweet_lo":     6.0,
        "sweet_hi":     8.0,
        "sweet_label":  "Luke's sweet spot (6–8×)",
        "best_fmt":     lambda v, kge: f"Best run (Ks={v:g}×, KGE={kge:.3f})",
    },
    "f_RS_abs": {
        "xlabel":       "RS soil f (1/mm)",
        "log_x":        True,
        "tick_fmt":     lambda v, _: f"{v:g}",
        "colorbar_lbl": "f (1/mm)",
        "color_low":    "Purple = low f (deep conductivity, possible subsurface flow)",
        "color_high":   "Yellow = high f (shallow collapse, infiltration-excess only)",
        "baseline":     0.020,
        "sweet_lo":     None,
        "sweet_hi":     None,
        "sweet_label":  None,
        "best_fmt":     lambda v, kge: f"Best run (f={v:g} mm⁻¹, KGE={kge:.3f})",
    },
    "f_RS_abs_Ks1": {
        "xlabel":       "RS soil f (1/mm)  [Ks = 1× baseline]",
        "log_x":        True,
        "tick_fmt":     lambda v, _: f"{v:g}",
        "colorbar_lbl": "f (1/mm)",
        "color_low":    "Purple = low f (deep conductivity, possible subsurface flow)",
        "color_high":   "Yellow = high f (shallow collapse, infiltration-excess only)",
        "baseline":     0.020,
        "sweet_lo":     None,
        "sweet_hi":     None,
        "sweet_label":  None,
        "best_fmt":     lambda v, kge: f"Best run (f={v:g} mm⁻¹, KGE={kge:.3f})",
    },
    "kinemvelcoef": {
        "xlabel":       "Hillslope velocity coefficient (cv)",
        "log_x":        True,
        "tick_fmt":     lambda v, _: f"{v:g}",
        "colorbar_lbl": "cv",
        "color_low":    "Purple = low cv (slow hillslope routing)",
        "color_high":   "Yellow = high cv (fast hillslope routing)",
        "baseline":     3.0,
        "sweet_lo":     None,
        "sweet_hi":     None,
        "sweet_label":  None,
        "best_fmt":     lambda v, kge: f"Best run (cv={v:g}, KGE={kge:.3f})",
    },
    "flowexp": {
        "xlabel":       "Hillslope velocity exponent (r)",
        "log_x":        False,
        "tick_fmt":     lambda v, _: f"{v:g}",
        "colorbar_lbl": "r",
        "color_low":    "Purple = low r",
        "color_high":   "Yellow = high r",
        "baseline":     0.3,
        "sweet_lo":     None,
        "sweet_hi":     None,
        "sweet_label":  None,
        "best_fmt":     lambda v, kge: f"Best run (r={v:g}, KGE={kge:.3f})",
    },
    "channelroughness": {
        "xlabel":       "Manning's channel roughness (n)",
        "log_x":        False,
        "tick_fmt":     lambda v, _: f"{v:g}",
        "colorbar_lbl": "Manning's n",
        "color_low":    "Purple = low n (fast channel flow)",
        "color_high":   "Yellow = high n (slow channel flow)",
        "baseline":     0.04,
        "sweet_lo":     None,
        "sweet_hi":     None,
        "sweet_label":  None,
        "best_fmt":     lambda v, kge: f"Best run (n={v:g}, KGE={kge:.3f})",
    },
}

OBS_COLOR   = "black"
GRID_KW     = dict(linestyle=":", alpha=0.5, color="gray")
BASELINE_KW = dict(color="steelblue", linestyle="--", linewidth=1.2, alpha=0.8)
SWEET_KW    = dict(color="forestgreen", alpha=0.12)


def styled_xaxis(ax, vals, meta):
    if meta["log_x"]:
        ax.set_xscale("log")
        ax.set_xlabel(f"{meta['xlabel']} (log scale)", fontsize=11)
    else:
        ax.set_xlabel(meta["xlabel"], fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(meta["tick_fmt"]))
    ax.axvline(meta["baseline"], label=f"Baseline ({meta['baseline']:g})", **BASELINE_KW)
    if meta["sweet_lo"] is not None:
        ax.axvspan(meta["sweet_lo"], meta["sweet_hi"],
                   label=meta["sweet_label"], **SWEET_KW)
    ax.grid(**GRID_KW)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path.name}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot sensitivity figures for one swept parameter.")
    parser.add_argument("--param", required=True,
                        choices=list(PARAM_META.keys()),
                        help="Parameter to plot")
    args = parser.parse_args()
    param_name = args.param
    meta = PARAM_META[param_name]

    # -----------------------------------------------------------------------
    # PATHS
    # -----------------------------------------------------------------------
    notebook_dir = Path.cwd()
    project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
    calib_dir    = project_root / "calibration_work"
    summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
    csv_dir      = calib_dir / "03_comparisons" / "csv_exports"
    plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / param_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # LOAD AND FILTER RESULTS
    # -----------------------------------------------------------------------
    results_path = summary_dir / "sensitivity_results_all.csv"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run the sweep first.")
        return

    df = pd.read_csv(results_path)
    param_df = df[df["swept_param"] == param_name].copy()

    if param_df.empty:
        print(f"ERROR: No rows found for swept_param == '{param_name}'")
        print(f"  Parameters available: {df['swept_param'].unique().tolist()}")
        return

    param_df = param_df.sort_values("swept_value").reset_index(drop=True)
    print(f"\nLoaded {len(param_df)} rows for {param_name}")

    vals    = param_df["swept_value"].values
    kge     = param_df["kge"].values
    nse     = param_df["nse"].values
    rmse    = param_df["rmse_m3s"].values
    pbias   = param_df["pbias_pct"].values
    kge_r   = param_df["kge_r"].values
    kge_a   = param_df["kge_alpha"].values
    kge_b   = param_df["kge_beta"].values
    sim_pk  = param_df["sim_peak_m3s"].values
    obs_pk  = param_df["obs_peak_m3s"].values[0]
    obs_vol = param_df["obs_volume_m3"].values[0]
    sim_vol = param_df["sim_volume_m3"].values
    timing  = param_df["peak_timing_error_hr"].values
    run_ids = param_df["run_id"].values

    # -----------------------------------------------------------------------
    # FIGURE 1: Sensitivity curve
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(vals, kge,  color="#1f77b4", linewidth=2, marker="o", markersize=5, label="KGE")
    ax.plot(vals, nse,  color="#ff7f0e", linewidth=2, marker="s", markersize=5, label="NSE")
    ax.plot(vals, rmse / rmse.max(),
            color="#d62728", linewidth=2, marker="^", markersize=5,
            label=f"RMSE (normalized, max={rmse.max():.1f} m³/s)")
    ax.plot(vals, pbias / 100,
            color="#9467bd", linewidth=2, marker="D", markersize=5,
            label="PBIAS / 100")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(1, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    styled_xaxis(ax, vals, meta)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title(f"Sensitivity of goodness-of-fit metrics to {param_name}\nSMF Aug 12, 2014 event",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, plot_dir / "fig1_sensitivity_curve_all_metrics.png")

    # -----------------------------------------------------------------------
    # FIGURE 2 & 2b: Hydrograph overlays
    # -----------------------------------------------------------------------
    hydrographs = {}
    for run_id, val in zip(run_ids, vals):
        csv_path = csv_dir / f"{run_id}_compare_obs_sim.csv"
        if csv_path.exists():
            tmp = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            hydrographs[val] = tmp

    if hydrographs:
        cmap = plt.get_cmap("plasma")
        if meta["log_x"] and vals.min() > 0:
            log_lo = np.log10(vals.min())
            log_hi = np.log10(vals.max())
            norm_fn    = lambda v: (np.log10(v) - log_lo) / (log_hi - log_lo) if log_hi > log_lo else 0.5
            color_norm = mcolors.LogNorm(vmin=vals.min(), vmax=vals.max())
        else:
            lo, hi = vals.min(), vals.max()
            norm_fn    = lambda v: (v - lo) / (hi - lo) if hi > lo else 0.5
            color_norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())

        best_val  = vals[np.argmax(kge)]
        first_hdf = list(hydrographs.values())[0]

        for fig_suffix, zoom in [("", False), ("b", True)]:
            fig2, ax2 = plt.subplots(figsize=(11, 6))
            ax2.set_facecolor("#e8e8e8")
            fig2.patch.set_facecolor("white")

            for val, hdf in sorted(hydrographs.items()):
                color   = cmap(norm_fn(val))
                is_best = val == best_val
                ax2.plot(hdf.index, hdf["Simulated"],
                         color=color,
                         linewidth=2.5 if is_best else 1.4,
                         alpha=1.0   if is_best else 0.75,
                         zorder=5    if is_best else 2)

            ax2.plot(first_hdf.index, first_hdf["Observed"],
                     color="white", linewidth=4.5, zorder=9)
            ax2.plot(first_hdf.index, first_hdf["Observed"],
                     color="black", linewidth=2.5, label="Observed", zorder=10)
            ax2.plot([], [], color=cmap(norm_fn(best_val)), linewidth=2.5,
                     label=meta["best_fmt"](best_val, kge.max()))

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
            sm.set_array([])
            cbar = fig2.colorbar(sm, ax=ax2, pad=0.01)
            cbar.set_label(meta["colorbar_lbl"], fontsize=10)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(meta["tick_fmt"]))

            ax2.grid(linestyle=":", alpha=0.6, color="#aaaaaa")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax2.set_xlabel("Time (Aug 12–13, 2014)", fontsize=11)
            ax2.set_ylabel("Discharge (m³/s)", fontsize=11)

            zoom_label = " — event window only" if zoom else ""
            ax2.set_title(
                f"Simulated hydrographs across {param_name} range{zoom_label}\n"
                f"{meta['color_low']}  |  {meta['color_high']}",
                fontsize=11)

            if zoom:
                ax2.set_xlim(pd.Timestamp("2014-08-12 17:30"),
                             pd.Timestamp("2014-08-12 21:00"))
                ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))

            ax2.legend(fontsize=10, facecolor="white", framealpha=0.9)
            fig2.tight_layout()
            fname = f"fig2{fig_suffix}_hydrograph_overlay{'_zoomed' if zoom else ''}.png"
            save(fig2, plot_dir / fname)
    else:
        print("No comparison CSVs found — skipping Figure 2.")

    # -----------------------------------------------------------------------
    # FIGURE 3: Peak and volume
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(vals, sim_pk, color="#1f77b4", linewidth=2, marker="o", markersize=6,
             label="Simulated peak")
    ax1.axhline(obs_pk, color=OBS_COLOR, linewidth=2, linestyle="--",
                label=f"Observed peak ({obs_pk:.1f} m³/s)")
    ax1.axvline(meta["baseline"], **BASELINE_KW)
    if meta["sweet_lo"]:
        ax1.axvspan(meta["sweet_lo"], meta["sweet_hi"], **SWEET_KW)
    if meta["log_x"]:
        ax1.set_xscale("log")
    ax1.set_ylabel("Peak discharge (m³/s)", fontsize=11)
    ax1.set_title(f"Peak discharge and event volume vs {param_name}\nSMF Aug 12, 2014 event",
                  fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(**GRID_KW)

    ax2.plot(vals, sim_vol / 1000, color="#ff7f0e", linewidth=2, marker="s", markersize=6,
             label="Simulated volume")
    ax2.axhline(obs_vol / 1000, color=OBS_COLOR, linewidth=2, linestyle="--",
                label=f"Observed volume ({obs_vol/1000:.1f} × 10³ m³)")
    ax2.axvline(meta["baseline"], **BASELINE_KW)
    if meta["sweet_lo"]:
        ax2.axvspan(meta["sweet_lo"], meta["sweet_hi"], **SWEET_KW)
    ax2.set_ylabel("Event volume (× 10³ m³)", fontsize=11)
    ax2.set_xlabel(meta["xlabel"] + (" (log scale)" if meta["log_x"] else ""), fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(**GRID_KW)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(meta["tick_fmt"]))
    fig.tight_layout()
    save(fig, plot_dir / "fig3_peak_and_volume.png")

    # -----------------------------------------------------------------------
    # FIGURE 4: Timing error
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(vals, timing * 60, color="#2ca02c", linewidth=2, marker="o", markersize=6)
    ax.axhline(0,   color=OBS_COLOR, linewidth=1.2, linestyle="--", label="Perfect timing")
    ax.axhline( 20, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
                label="±20 min reference")
    ax.axhline(-20, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    styled_xaxis(ax, vals, meta)
    ax.set_ylabel("Peak timing error (minutes)\nPositive = model peaks late", fontsize=11)
    ax.set_title(
        f"Peak timing error vs {param_name}\n"
        "Expected to be flat if timing is controlled by forcing, not this parameter",
        fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, plot_dir / "fig4_peak_timing_error.png")

    # -----------------------------------------------------------------------
    # FIGURE 5: KGE decomposition
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(vals, kge_r, color="#1f77b4", linewidth=2, marker="o", markersize=5,
            label="r  (correlation — timing and shape)")
    ax.plot(vals, kge_a, color="#ff7f0e", linewidth=2, marker="s", markersize=5,
            label="α  (variability ratio — flashiness)")
    ax.plot(vals, kge_b, color="#d62728", linewidth=2, marker="^", markersize=5,
            label="β  (bias ratio — volume)")
    ax.plot(vals, kge,   color="black",   linewidth=2.5, marker="D", markersize=5,
            linestyle="--", label="KGE (combined)")
    ax.axhline(1, color="black", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Perfect value (1.0 for all components)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
    styled_xaxis(ax, vals, meta)
    ax.set_ylabel("Component value\n(perfect = 1.0 for all)", fontsize=11)
    ax.set_title(f"KGE decomposition vs {param_name}\nWhich component drives KGE changes?",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, plot_dir / "fig5_kge_decomposition.png")

    print(f"\nAll figures saved to:\n  {plot_dir}")


if __name__ == "__main__":
    main()