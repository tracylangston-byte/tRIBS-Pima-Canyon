"""
plot_series83_double_peak.py
=============================
Pulls out a single, enlarged, annotated version of the "Hydrograph --
event window" panel from plot_best_run_diagnostic.py's Series 83
best-run diagnostic -- built specifically to make the double-peak /
SMPHQ forcing-timing artifact the standalone headline of slide 8,
rather than one panel among six.

Reads the exact same two inputs plot_best_run_diagnostic.py already
uses -- no new data, no re-run required:
    lhs_results_11param_83.csv       (finds the best run by KGE)
    {run_id}_compare_obs_sim.csv     (the actual obs/sim hydrograph)

IMPORTANT (per Tracy, 7/9/26 docstring note in the live Codespace copy
of plot_best_run_diagnostic.py): no current run script in the active
pipeline regenerates {run_id}_compare_obs_sim.csv. It's a surviving
file from the original Series 83 run (pre-Series-90 era), not
something that can be rebuilt if it's ever lost. Worth backing this
specific file up somewhere safe (or uploading a copy) independent of
this plotting task.

Usage (run from smf_demo/):
    python plot_series83_double_peak.py
    python plot_series83_double_peak.py --no-chart-title
    python plot_series83_double_peak.py --no-annotation

Output:
    calibration_work/03_comparisons/sensitivity_plots/best_run_diagnostic/
        fig_series83_double_peak.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RESULTS_CSV = "lhs_results_11param_83.csv"
EVENT_CROP_START = "2014-08-12 17:30"
EVENT_CROP_END = "2014-08-12 21:00"

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--no-chart-title", action="store_true",
                     help="Suppress the in-chart title, if the slide already has its own headline.")
parser.add_argument("--no-annotation", action="store_true",
                     help="Drop the callout arrow pointing at the early sub-peak.")
parser.add_argument("--outdir", default=None, help="Output directory override.")
args = parser.parse_args()

script_dir = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir = project_root / "calibration_work"
summary_dir = calib_dir / "03_comparisons" / "summary_tables"
csv_dir = calib_dir / "03_comparisons" / "csv_exports"
plot_dir = Path(args.outdir) if args.outdir else calib_dir / "03_comparisons" / "sensitivity_plots" / "best_run_diagnostic"
plot_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# LOAD RESULTS, FIND BEST RUN -- identical logic to
# plot_best_run_diagnostic.py, so the same run_id gets selected.
# ---------------------------------------------------------------
results_path = summary_dir / RESULTS_CSV
if not results_path.exists():
    raise FileNotFoundError(f"LHS results not found: {results_path}")

df = pd.read_csv(results_path).dropna(subset=["kge"]).reset_index(drop=True)
best_idx = int(np.argmax(df["kge"].values))
best = df.iloc[best_idx]
run_id = best["run_id"]

print(f"Best run: {run_id}  (KGE={best['kge']:.3f})")

# ---------------------------------------------------------------
# LOAD HYDROGRAPH CSV -- see docstring note above re: this file's fragility.
# ---------------------------------------------------------------
hydro_path = csv_dir / f"{run_id}_compare_obs_sim.csv"
if not hydro_path.exists():
    raise FileNotFoundError(
        f"Hydrograph CSV not found: {hydro_path}\n"
        f"No current run script regenerates this file -- if it's gone, "
        f"check for a backup before troubleshooting further."
    )

hydro = pd.read_csv(hydro_path, index_col=0, parse_dates=True)
obs_crop = hydro["Observed"].loc[EVENT_CROP_START:EVENT_CROP_END]
sim_crop = hydro["Simulated"].loc[EVENT_CROP_START:EVENT_CROP_END]

# ---------------------------------------------------------------
# PLOT -- single enlarged panel, presentation-legible fonts/lines
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(obs_crop.index, obs_crop.values, color="#232320", linewidth=3,
        label="Observed", zorder=4)
ax.plot(sim_crop.index, sim_crop.values, color="#D85A30", linewidth=2.5,
        linestyle="--", label=f"Simulated (KGE={best['kge']:.3f})", zorder=3)

ax.fill_between(obs_crop.index, obs_crop.values, sim_crop.values,
                where=(sim_crop.values >= obs_crop.values),
                alpha=0.15, color="#D85A30", label="Over-predict")
ax.fill_between(obs_crop.index, obs_crop.values, sim_crop.values,
                where=(sim_crop.values < obs_crop.values),
                alpha=0.15, color="#185FA5", label="Under-predict")

ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Discharge (m3/s)", fontsize=16)
ax.tick_params(labelsize=14)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.legend(fontsize=13, loc="upper right", framealpha=0.9)
ax.grid(alpha=0.25)

if not args.no_chart_title:
    ax.set_title("Calibration best run — observed vs. simulated (Aug 12, 2014)", fontsize=17)

if not args.no_annotation:
    # Looks for the simulated curve's local peak in the early-artifact
    # window. Widened generously since the exact timing wasn't confirmed
    # from the image alone -- check where this lands and narrow/shift
    # the search window below if it grabs the wrong bump.
    bump_window = sim_crop.loc["2014-08-12 18:00":"2014-08-12 18:40"]
    if not bump_window.empty:
        bump_t = bump_window.idxmax()
        bump_q = bump_window.max()

        # Find genuinely clear space in the middle of the plot, rather than
        # guessing a fixed offset: first point well after the bump where
        # BOTH curves have flattened out near baseline -- tightened past
        # the over-predict shading, not just "low-ish."
        after_bump = obs_crop.index > (bump_t + pd.Timedelta(minutes=30))
        both_low = (obs_crop < 3) & (sim_crop < 5)
        clear_times = obs_crop.index[after_bump & both_low]

        mid_y = (obs_crop.max() + obs_crop.min()) / 2  # roughly mid-height of the plot

        if len(clear_times) > 0:
            label_t = clear_times[0] + pd.Timedelta(minutes=8)
        else:
            label_t = bump_t + pd.Timedelta(minutes=75)  # fallback if nothing found

        ax.annotate("Early sub-peak\n(Dual input timing)",
                    xy=(bump_t, bump_q), xytext=(label_t, mid_y), textcoords="data",
                    fontsize=16, fontweight="bold", color="#378DBD", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#378DBD", lw=1.8))
    else:
        print("Annotation window found no data -- check EVENT_CROP bounds "
              "and the bump-search window against your actual data.")

out_path = plot_dir / "fig_series83_double_peak.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")