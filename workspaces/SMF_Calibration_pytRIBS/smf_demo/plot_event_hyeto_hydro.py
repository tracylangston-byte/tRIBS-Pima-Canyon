"""
plot_event_hyeto_hydro.py
==========================
Plots the OBSERVED rainfall hyetograph and OBSERVED streamflow hydrograph
for the August 12, 2014 SMF storm event -- the real, historical event that
synthetic truth (truth100) is built from.

This is deliberately a clean teaching figure (for AHS slide 7a: timing,
shape, volume), NOT the truth-vs-simulated validation diagnostic that
plot_truth_hydrograph_comparison.py produces -- that script reads a
*_compare_obs_sim.csv export and compares two model runs against each
other. This script instead reads the two RAW data sources directly:

  Rainfall:   ../smf_init_data/met/precip_SMF_1.mdf
              (Y M D H R, mm/hr, no minute column -- sub-hourly rows
              share the same H and are assumed evenly spaced within it)
              [optionally also precip_SMPHQ_2.mdf, via --show-smphq]
  Streamflow: ../smf_init_data/met/SMF_Observations_1993-2025.xlsx
              (sheet 'Discharge', skiprows=6, columns Date/Time/cfs)

Usage (run from smf_demo/):
    python plot_event_hyeto_hydro.py
    python plot_event_hyeto_hydro.py --start "2014-08-11" --end "2014-08-13"

    # NEW: auto-zoom onto just the storm (rise -> peak -> recession),
    # instead of guessing --start/--end by eye:
    python plot_event_hyeto_hydro.py --auto-zoom
    python plot_event_hyeto_hydro.py --auto-zoom --buffer-hours 2
    python plot_event_hyeto_hydro.py --auto-zoom --trim-hours 1   # shave 1h off each end further

    python plot_event_hyeto_hydro.py --show-smphq   # overlay SMPHQ gauge too

    # Also save a simplified, room-legible version for slide 6
    # (no title/totals/tick numbers, big bold panel labels):
    python plot_event_hyeto_hydro.py --auto-zoom --big-picture

Output:
    calibration_work/03_comparisons/sensitivity_plots/
        event_hyeto_hydro_20140812.png                (always)
        event_hyeto_hydro_20140812_bigpicture.png      (only with --big-picture)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--start", default="2014-08-11",
                     help="Outer window start to load (YYYY-MM-DD[ HH:MM]). Default 2014-08-11. "
                          "With --auto-zoom this is just the search range, not the final plot window.")
parser.add_argument("--end", default="2014-08-13",
                     help="Outer window end to load. Default 2014-08-13.")
parser.add_argument("--auto-zoom", action="store_true",
                     help="After loading --start/--end, automatically narrow the PLOTTED "
                          "window to bracket just the storm: first meaningful rise above "
                          "baseline, through the peak, to where flow recedes back near "
                          "baseline -- plus --buffer-hours on each side.")
parser.add_argument("--buffer-hours", type=float, default=3.0,
                     help="Hours of buffer before rise onset / after recession, used only "
                          "with --auto-zoom. Default 3.")
parser.add_argument("--show-smphq", action="store_true",
                     help="Overlay the SMPHQ gauge on the hyetograph too "
                          "(shows the timing/intensity offset from SMF -- useful "
                          "for slide 4/8's double-peak artifact story, not slide 7a).")
parser.add_argument("--big-picture", action="store_true",
                     help="Also save a second, simplified 'what is a hyetograph/hydrograph' "
                          "version for slide 6: no title/totals, no tick numbers or date "
                          "labels, no gridlines, thicker lines, large bold panel labels. "
                          "Saved as event_hyeto_hydro_20140812_bigpicture.png alongside the "
                          "normal detailed output.")
parser.add_argument("--trim-hours", type=float, default=0.0,
                     help="Shave this many hours off the start AND end of whatever window is "
                          "about to be plotted (applied after --auto-zoom, if used). Totals "
                          "are computed AFTER trimming, so they'll shift if rain/flow existed "
                          "in the trimmed hours -- check the printed totals after running.")
parser.add_argument("--no-chart-title", action="store_true",
                     help="Suppress the in-chart title (rainfall/runoff totals). Use this when "
                          "the slide already has its own big headline (e.g. assertion-evidence "
                          "style) and the in-chart title would just compete with it for space "
                          "and attention.")
parser.add_argument("--outdir", default=None, help="Output directory override.")
args = parser.parse_args()

start = pd.Timestamp(args.start)
end = pd.Timestamp(args.end)

script_dir = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
met_dir = project_root / "smf_init_data" / "met"
calib_dir = project_root / "calibration_work"
plot_dir = Path(args.outdir) if args.outdir else calib_dir / "03_comparisons" / "sensitivity_plots"
plot_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# LOAD RAINFALL
# ---------------------------------------------------------------
def load_mdf_rainfall(path):
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, names=["Y", "M", "D", "H", "R"])
    df["block"] = list(zip(df["Y"], df["M"], df["D"], df["H"]))
    minute_offsets = []
    for _, grp in df.groupby("block", sort=False):
        n = len(grp)
        step = 60.0 / n
        minute_offsets.extend([i * step for i in range(n)])
    df["minute"] = minute_offsets
    df["datetime"] = pd.to_datetime(
        dict(year=df["Y"], month=df["M"], day=df["D"], hour=df["H"])
    ) + pd.to_timedelta(df["minute"], unit="m")
    return df.set_index("datetime").sort_index()["R"]  # mm/hr


rain_path = met_dir / "precip_SMF_1.mdf"
if not rain_path.exists():
    raise SystemExit(f"ERROR: {rain_path} not found -- check met_dir / filename.")

rain_smf = load_mdf_rainfall(rain_path).loc[start:end]

if args.show_smphq:
    rain_smphq = load_mdf_rainfall(met_dir / "precip_SMPHQ_2.mdf").loc[start:end]

# ---------------------------------------------------------------
# LOAD OBSERVED STREAMFLOW
# ---------------------------------------------------------------
obs_path = met_dir / "SMF_Observations_1993-2025.xlsx"
if not obs_path.exists():
    raise SystemExit(f"ERROR: {obs_path} not found -- check met_dir / filename.")

obs = pd.read_excel(obs_path, sheet_name="Discharge", skiprows=6)
obs["datetime"] = pd.to_datetime(obs["Date"].astype(str) + " " + obs["Time"].astype(str))
obs = obs.set_index("datetime").sort_index()
obs["Q_cms"] = obs["cfs"] * 0.0283168
q = obs["Q_cms"].loc[start:end]

if q.empty:
    raise SystemExit(
        f"No observed streamflow found between {start} and {end} -- "
        f"widen --start/--end and try again."
    )

peak_q, peak_t = q.max(), q.idxmax()

# ---------------------------------------------------------------
# AUTO-ZOOM: find rise onset and recession-back-to-baseline, crop
# rain + flow to that window (+ buffer) before plotting.
# ---------------------------------------------------------------
if args.auto_zoom:
    n_baseline_pts = max(1, len(q) // 20)
    baseflow = q.iloc[:n_baseline_pts].median()
    threshold = baseflow + 0.05 * (peak_q - baseflow)

    above = q[q > threshold]
    if above.empty:
        print("auto-zoom: no clear rise detected above baseline -- keeping full --start/--end window.")
    else:
        rise_t = above.index[0]
        after_peak = q.loc[peak_t:]
        recede = after_peak[after_peak <= threshold]
        recede_t = recede.index[0] if not recede.empty else q.index[-1]

        zoom_start = rise_t - pd.Timedelta(hours=args.buffer_hours)
        zoom_end = recede_t + pd.Timedelta(hours=args.buffer_hours)

        rain_smf = rain_smf.loc[zoom_start:zoom_end]
        if args.show_smphq:
            rain_smphq = rain_smphq.loc[zoom_start:zoom_end]
        q = q.loc[zoom_start:zoom_end]

        print(f"auto-zoom: rise detected {rise_t}, recession-to-baseline {recede_t}")
        print(f"auto-zoom: plotted window {zoom_start} to {zoom_end} "
              f"(+/- {args.buffer_hours}h buffer)")

# ---------------------------------------------------------------
# TRIM: shave a fixed amount off each end of whatever window is
# about to be plotted (applied after auto-zoom, or after manual
# --start/--end if --auto-zoom wasn't used).
# ---------------------------------------------------------------
if args.trim_hours > 0:
    trim_start = q.index[0] + pd.Timedelta(hours=args.trim_hours)
    trim_end = q.index[-1] - pd.Timedelta(hours=args.trim_hours)
    if trim_start >= trim_end:
        print(f"trim-hours={args.trim_hours} would remove the whole window -- skipping trim.")
    else:
        rain_smf = rain_smf.loc[trim_start:trim_end]
        if args.show_smphq:
            rain_smphq = rain_smphq.loc[trim_start:trim_end]
        q = q.loc[trim_start:trim_end]
        print(f"trim: window now {trim_start} to {trim_end}")

# ---------------------------------------------------------------
# TOTALS (computed on whatever window ends up plotted)
# ---------------------------------------------------------------
dt_rain_min = rain_smf.index.to_series().diff().median().total_seconds() / 60.0
dt_q_min = q.index.to_series().diff().median().total_seconds() / 60.0
total_rain_in = rain_smf.sum() * (dt_rain_min / 60) / 25.4
total_vol_af = q.sum() * (dt_q_min * 60) / 1233.48

# ---------------------------------------------------------------
# PLOT -- hyetograph (top, inverted) + hydrograph (bottom), shared x-axis
# ---------------------------------------------------------------
fig, (ax_rain, ax_flow) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True,
    gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08},
)

bar_width = (dt_rain_min / 1440) * 0.9  # in days, matplotlib date units
ax_rain.bar(rain_smf.index, rain_smf.values, width=bar_width, color="#185FA5", label="SMF gauge")
if args.show_smphq:
    ax_rain.bar(rain_smphq.index, rain_smphq.values, width=bar_width, color="#7FB2DE",
                alpha=0.7, label="SMPHQ gauge")
    ax_rain.legend(fontsize=12, loc="upper right")
ax_rain.invert_yaxis()
ax_rain.set_ylabel("Rainfall\n(mm/hr)", fontsize=15)
ax_rain.tick_params(labelsize=13)
if not args.no_chart_title:
    ax_rain.set_title(
        f"August 12, 2014 SMF event — observed rainfall and streamflow\n"
        f"SMF gauge rainfall: {total_rain_in:.2f} in    |    runoff volume: {total_vol_af:.1f} ac-ft",
        fontsize=15)

ax_flow.plot(q.index, q.values, color="#0818BC", linewidth=2.5)
ax_flow.axvline(peak_t, color="#BA7517", linestyle=":", linewidth=1.5)
ax_flow.annotate(f"Peak: {peak_q:.2f} m3/s\n@ {peak_t.strftime('%m-%d %H:%M')}",
                  xy=(peak_t, peak_q), xytext=(10, -25), textcoords="offset points",
                  fontsize=13, color="#BA7517")
ax_flow.set_ylabel("Discharge (m3/s)", fontsize=15)
ax_flow.set_xlabel("Date / time", fontsize=15)
ax_flow.tick_params(labelsize=13)
ax_flow.grid(alpha=0.3)
ax_flow.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()

out_path = plot_dir / "event_hyeto_hydro_20140812.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {out_path}")
print(f"Peak discharge: {peak_q:.3f} m3/s @ {peak_t}")
print(f"Total SMF rainfall in window: {total_rain_in:.2f} in")
print(f"Total runoff volume in window: {total_vol_af:.1f} ac-ft")

# ---------------------------------------------------------------
# BIG-PICTURE VERSION (slide 6): same data, radically simplified --
# no title/totals, no tick numbers, no gridlines, thick lines, big
# bold panel labels naming the term and what it represents.
# ---------------------------------------------------------------
if args.big_picture:
    fig2, (bax_rain, bax_flow) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
    )

    bax_rain.bar(rain_smf.index, rain_smf.values, width=bar_width * 1.4, color="#185FA5")
    bax_rain.invert_yaxis()
    bax_rain.set_yticks([])
    bax_rain.set_xticks([])
    for spine in ["top", "right", "left"]:
        bax_rain.spines[spine].set_visible(False)
    bax_rain.text(0.02, 0.90, "HYETOGRAPH", transform=bax_rain.transAxes,
                  fontsize=24, fontweight="bold", color="#185FA5",
                  va="top", ha="left")
    bax_rain.text(0.02, 0.55, "rainfall over time", transform=bax_rain.transAxes,
                  fontsize=15, color="#185FA5", va="top", ha="left")

    bax_flow.plot(q.index, q.values, color="#0818BC", linewidth=5)
    bax_flow.set_yticks([])
    bax_flow.set_xticks([])
    for spine in ["top", "right", "left"]:
        bax_flow.spines[spine].set_visible(False)
    bax_flow.text(0.02, 0.92, "HYDROGRAPH", transform=bax_flow.transAxes,
                  fontsize=24, fontweight="bold", color="#0818BC",
                  va="top", ha="left")
    bax_flow.text(0.02, 0.78, "streamflow over time", transform=bax_flow.transAxes,
                  fontsize=15, color="#0818BC", va="top", ha="left")
    bax_flow.set_xlabel("Time  →", fontsize=18)

    out_path_bp = plot_dir / "event_hyeto_hydro_20140812_bigpicture.png"
    fig2.savefig(out_path_bp, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved (big-picture): {out_path_bp}")