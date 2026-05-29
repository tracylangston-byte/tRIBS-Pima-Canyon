"""
plot_best_run_diagnostic.py
============================
Produces a single combined 2x3 diagnostic figure for the best run from
the series 81 (KsHi) LHS sweep. The best run is identified automatically
from lhs_results_5param_KsHi.csv.

Usage (run from the smf_demo directory):
    python plot_best_run_diagnostic.py

Output
------
    calibration_work/03_comparisons/sensitivity_plots/best_run_diagnostic/
        fig_best_run_diagnostic.png   — 2x3 combined panel figure

Panel layout
------------
    [Row 1, Col 1] Hydrograph — event window (17:30-21:00), obs vs sim
    [Row 1, Col 2] KGE component bar chart — r, alpha, beta vs ideal (1.0)
    [Row 2, Col 1] Hydrograph — full recession (full event window)
    [Row 2, Col 2] Metrics summary table — KGE, NSE, PBIAS, RMSE, peak Q, timing error
    [Row 3, Col 1] Residuals — (sim - obs) through event window
    [Row 3, Col 2] Volume comparison — cumulative obs vs sim through event

The figure is designed to be dropped directly into a presentation or report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path

# =======================================================================
# CONFIG — change RESULTS_CSV to switch series
# =======================================================================
RESULTS_CSV  = "lhs_results_11param_83.csv"
SERIES_TITLE = "Series 83  |  11 param run"
# =======================================================================

EVENT_LABEL      = "SMF Aug 12, 2014"
EVENT_CROP_START = "2014-08-12 17:30"
EVENT_CROP_END   = "2014-08-12 21:00"

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "smf_demo" else notebook_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"
csv_dir      = calib_dir / "03_comparisons" / "csv_exports"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / "best_run_diagnostic"
plot_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD RESULTS AND FIND BEST RUN
# -----------------------------------------------------------------------
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(
        f"LHS results not found: {results_path}\n"
        "Run run_lhs_5param.py (series 81) first."
    )

df = pd.read_csv(results_path)
df = df.dropna(subset=["kge"]).reset_index(drop=True)

best_idx = int(np.argmax(df["kge"].values))
best     = df.iloc[best_idx]
run_id   = best["run_id"]

print(f"Best run: {run_id}")
print(f"  KGE={best['kge']:.3f}  NSE={best['nse']:.3f}  "
      f"PBIAS={best['pbias_pct']:+.1f}%  RMSE={best['rmse_m3s']:.2f} m3/s")
print(f"  Ks={best['Ks_mult']:.2f}x  cv={best['kinemvelcoef']:.2f}  "
      f"r={best['flowexp']:.3f}  n={best['channelroughness']:.4f}")

# -----------------------------------------------------------------------
# LOAD HYDROGRAPH CSV FOR BEST RUN
# -----------------------------------------------------------------------
hydro_path = csv_dir / f"{run_id}_compare_obs_sim.csv"
if not hydro_path.exists():
    raise FileNotFoundError(
        f"Hydrograph CSV not found: {hydro_path}\n"
        "Check that the run completed successfully."
    )

hydro = pd.read_csv(hydro_path, index_col=0, parse_dates=True)
obs_full = hydro["Observed"]
sim_full = hydro["Simulated"]

# Cropped event window
obs_crop = obs_full.loc[EVENT_CROP_START:EVENT_CROP_END]
sim_crop = sim_full.loc[EVENT_CROP_START:EVENT_CROP_END]

# Residuals
residuals = sim_crop - obs_crop

# Cumulative volumes (m3, using 5-min = 300s timestep)
dt = 300
obs_cumvol = (obs_crop * dt).cumsum() / 1000   # convert to 1000 m3 for readability
sim_cumvol = (sim_crop * dt).cumsum() / 1000

# -----------------------------------------------------------------------
# PARAMETER STRING FOR TITLES
# -----------------------------------------------------------------------
param_str = (f"Ks={best['Ks_mult']:.2f}x  "
             f"cv={best['kinemvelcoef']:.2f}  "
             f"r={best['flowexp']:.3f}  "
             f"n={best['channelroughness']:.4f}  "
             f"f=0.020 mm$^{{-1}}$")

# -----------------------------------------------------------------------
# BUILD FIGURE
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    f"Best-run diagnostic  —  {SERIES_TITLE}  |  {EVENT_LABEL}\n"
    f"{param_str}",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])   # Hydrograph cropped
ax2 = fig.add_subplot(gs[0, 1])   # KGE components
ax3 = fig.add_subplot(gs[1, 0])   # Hydrograph full recession
ax4 = fig.add_subplot(gs[1, 1])   # Metrics table
ax5 = fig.add_subplot(gs[2, 0])   # Residuals
ax6 = fig.add_subplot(gs[2, 1])   # Cumulative volume

date_fmt = plt.matplotlib.dates.DateFormatter("%H:%M")

# -----------------------------------------------------------------------
# PANEL 1: Hydrograph — event window
# -----------------------------------------------------------------------
ax1.plot(obs_crop.index, obs_crop.values, color="black",
         linewidth=2.2, label="Observed", zorder=4)
ax1.plot(sim_crop.index, sim_crop.values, color="#e63946",
         linewidth=1.8, linestyle="--", label=f"Simulated (KGE={best['kge']:.3f})", zorder=3)

ax1.fill_between(obs_crop.index, obs_crop.values, sim_crop.values,
                 where=(sim_crop.values >= obs_crop.values),
                 alpha=0.15, color="#e63946", label="Over-predict")
ax1.fill_between(obs_crop.index, obs_crop.values, sim_crop.values,
                 where=(sim_crop.values < obs_crop.values),
                 alpha=0.15, color="#457b9d", label="Under-predict")

ax1.set_xlabel("Time", fontsize=10)
ax1.set_ylabel("Discharge (m\u00b3/s)", fontsize=10)
ax1.set_title("Hydrograph — event window", fontsize=11)
ax1.legend(fontsize=8.5, loc="upper right", facecolor="white", framealpha=0.9)
ax1.xaxis.set_major_formatter(date_fmt)
ax1.grid(alpha=0.25)
ax1.xaxis.set_tick_params(rotation=0)

# -----------------------------------------------------------------------
# PANEL 2: KGE component bar chart
# -----------------------------------------------------------------------
components  = ["r\n(timing corr.)", "\u03b1\n(variability)", "\u03b2\n(bias)"]
values      = [best["kge_r"], best["kge_alpha"], best["kge_beta"]]
ideal       = [1.0, 1.0, 1.0]
bar_colors  = ["#2a9d8f", "#e9c46a", "#e76f51"]

x = np.arange(len(components))
width = 0.35

bars_sim  = ax2.bar(x - width/2, values, width, color=bar_colors,
                    alpha=0.85, label="Best run", zorder=3)
bars_ideal = ax2.bar(x + width/2, ideal, width, color="lightgray",
                     alpha=0.7, edgecolor="gray", label="Ideal (1.0)", zorder=2)

# Annotate values on bars
for bar, val in zip(bars_sim, values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
ax2.set_xticks(x)
ax2.set_xticklabels(components, fontsize=10)
ax2.set_ylabel("Component value", fontsize=10)
ax2.set_ylim(0, 1.35)
ax2.set_title(f"KGE components  (KGE={best['kge']:.3f})", fontsize=11)
ax2.legend(fontsize=8.5, loc="upper right", facecolor="white", framealpha=0.9)
ax2.grid(axis="y", alpha=0.25)

# -----------------------------------------------------------------------
# PANEL 3: Hydrograph — full recession
# -----------------------------------------------------------------------
ax3.plot(obs_full.index, obs_full.values, color="black",
         linewidth=2.0, label="Observed", zorder=4)
ax3.plot(sim_full.index, sim_full.values, color="#e63946",
         linewidth=1.6, linestyle="--", label="Simulated", zorder=3)

ax3.set_xlabel("Time", fontsize=10)
ax3.set_ylabel("Discharge (m\u00b3/s)", fontsize=10)
ax3.set_title("Hydrograph — full event window", fontsize=11)
ax3.legend(fontsize=8.5, loc="upper right", facecolor="white", framealpha=0.9)
ax3.xaxis.set_major_formatter(date_fmt)
ax3.grid(alpha=0.25)
ax3.xaxis.set_tick_params(rotation=0)

# -----------------------------------------------------------------------
# PANEL 4: Metrics summary table
# -----------------------------------------------------------------------
ax4.axis("off")

# Build table data
peak_timing_hr = best.get("peak_timing_error_hr", np.nan)
peak_timing_min = peak_timing_hr * 60 if not np.isnan(peak_timing_hr) else np.nan

table_data = [
    ["KGE",                    f"{best['kge']:.3f}"],
    ["NSE",                    f"{best['nse']:.3f}"],
    ["PBIAS",                  f"{best['pbias_pct']:+.1f}%"],
    ["RMSE",                   f"{best['rmse_m3s']:.2f} m\u00b3/s"],
    ["Obs peak",               f"{best['obs_peak_m3s']:.1f} m\u00b3/s"],
    ["Sim peak",               f"{best['sim_peak_m3s']:.1f} m\u00b3/s"],
    ["Peak timing error",      f"{peak_timing_min:+.0f} min" if not np.isnan(peak_timing_min) else "n/a"],
    ["Obs volume",             f"{best['obs_volume_m3']/1000:.1f} \u00d7 10\u00b3 m\u00b3"],
    ["Sim volume",             f"{best['sim_volume_m3']/1000:.1f} \u00d7 10\u00b3 m\u00b3"],
    ["KGE r",                  f"{best['kge_r']:.3f}"],
    ["\u03b1 (variability)",   f"{best['kge_alpha']:.3f}"],
    ["\u03b2 (bias)",          f"{best['kge_beta']:.3f}"],
]

tbl = ax4.table(
    cellText=table_data,
    colLabels=["Metric", "Value"],
    cellLoc="center",
    loc="center",
    bbox=[0.05, 0.0, 0.9, 1.0]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

# Header styling
for j in range(2):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Row shading — highlight KGE row
highlight_rows = {1}   # row 1 = KGE (1-indexed after header)
for i in range(1, len(table_data) + 1):
    bg = "#eaf4ea" if i in highlight_rows else ("#f7f7f7" if i % 2 == 0 else "white")
    for j in range(2):
        tbl[i, j].set_facecolor(bg)
        if i in highlight_rows:
            tbl[i, j].set_text_props(fontweight="bold")

ax4.set_title("Performance metrics", fontsize=11, pad=8)

# -----------------------------------------------------------------------
# PANEL 5: Residuals
# -----------------------------------------------------------------------
ax5.bar(residuals.index, residuals.values,
        width=pd.Timedelta(minutes=4),
        color=np.where(residuals.values >= 0, "#e63946", "#457b9d"),
        alpha=0.75, zorder=3)
ax5.axhline(0, color="black", linewidth=0.9, zorder=4)

ax5.set_xlabel("Time", fontsize=10)
ax5.set_ylabel("Residual (m\u00b3/s)", fontsize=10)
ax5.set_title("Residuals: Simulated \u2212 Observed", fontsize=11)
ax5.xaxis.set_major_formatter(date_fmt)
ax5.grid(axis="y", alpha=0.25)

# Annotate mean residual
mean_resid = residuals.mean()
ax5.axhline(mean_resid, color="black", linewidth=1.2, linestyle="--", alpha=0.6,
            label=f"Mean residual = {mean_resid:+.2f} m\u00b3/s")
ax5.legend(fontsize=8.5, loc="upper right", facecolor="white", framealpha=0.9)
ax5.xaxis.set_tick_params(rotation=0)

# -----------------------------------------------------------------------
# PANEL 6: Cumulative volume
# -----------------------------------------------------------------------
ax6.plot(obs_cumvol.index, obs_cumvol.values, color="black",
         linewidth=2.0, label="Observed", zorder=4)
ax6.plot(sim_cumvol.index, sim_cumvol.values, color="#e63946",
         linewidth=1.8, linestyle="--", label="Simulated", zorder=3)

ax6.fill_between(obs_cumvol.index, obs_cumvol.values, sim_cumvol.values,
                 alpha=0.15,
                 color="#e63946" if sim_cumvol.iloc[-1] > obs_cumvol.iloc[-1] else "#457b9d")

# Annotate final volume difference
final_obs = obs_cumvol.iloc[-1]
final_sim = sim_cumvol.iloc[-1]
ax6.annotate(
    f"Final vol. diff: {((final_sim - final_obs) / final_obs * 100):+.1f}%\n"
    f"Obs: {final_obs:.1f}  Sim: {final_sim:.1f}  (\u00d710\u00b3 m\u00b3)",
    xy=(obs_cumvol.index[-1], max(final_obs, final_sim)),
    xytext=(-10, -30), textcoords="offset points",
    fontsize=8.5, ha="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
)

ax6.set_xlabel("Time", fontsize=10)
ax6.set_ylabel("Cumulative volume (\u00d710\u00b3 m\u00b3)", fontsize=10)
ax6.set_title("Cumulative volume — event window", fontsize=11)
ax6.legend(fontsize=8.5, loc="upper left", facecolor="white", framealpha=0.9)
ax6.xaxis.set_major_formatter(date_fmt)
ax6.grid(alpha=0.25)
ax6.xaxis.set_tick_params(rotation=0)

# -----------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------
out_path = plot_dir / "fig_best_run_diagnostic.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close(fig)

print(f"\nParameter set:")
print(f"  Ks_mult:          {best['Ks_mult']:.3f}x")
print(f"  kinemvelcoef:     {best['kinemvelcoef']:.3f}")
print(f"  flowexp:          {best['flowexp']:.3f}")
print(f"  channelroughness: {best['channelroughness']:.4f}")
print(f"  f_RS_abs:         0.020 mm^-1 (fixed)")
