"""
plot_truth_hydrograph_comparison.py
=====================================
Plots the real SMF gauge (observed) alongside the synthetic truth
hydrographs for Series 93, 94, and 95 on a single figure.

Series truth files (all in calibration_work/synth_truth_archive/):
  S93 — synth_truth_Ks8p5_cv4p5_r0p24_n0p075.qout  (n = 0.075, heavy roughness)
  S94 — SMF_20140812_63_r0p15_Outlet.qout            (r = 0.15,  low recession exponent)
  S95 — SMF_20140812_60_Ks15p0x_Outlet.qout          (Ks = 15×,  high conductivity)

Notes:
  - S94/S95 are from the 60s single-param sweep (OPINTRVL = 1 hr); they are
    interpolated to 5-min after resampling so the event shape is preserved.
  - S93 was run at OPINTRVL = 0.0833 hr (5-min); no interpolation needed.
  - "Observed" = real SMF gauge from SMF_Observations_1993-2025.xlsx.

Usage (run from smf_demo/):
    python plot_truth_hydrograph_comparison.py

Output:
    calibration_work/03_comparisons/sensitivity_plots/truth_comparison/
        truth_hydrograph_comparison_S93_S94_S95.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
archive_dir  = calib_dir / "synth_truth_archive"
plot_dir     = calib_dir / "03_comparisons" / "sensitivity_plots" / "truth_comparison"
plot_dir.mkdir(parents=True, exist_ok=True)

OBS_XLSX    = project_root / "smf_init_data" / "met" / "SMF_Observations_1993-2025.xlsx"
QOUT_ORIGIN = pd.Timestamp("2014-08-01")   # fractional-hour Time column origin

EVENT_START = "2014-08-12 17:30"
EVENT_END   = "2014-08-12 21:00"

# -----------------------------------------------------------------------
# CONFIG
# Add or remove series here; no other edits needed.
# Keys: label, file, color, lw, ls
# -----------------------------------------------------------------------
SERIES = [
    {
        "label": "S93  |  n = 0.075",
        "file":  archive_dir / "synth_truth_Ks8p5_cv4p5_r0p24_n0p075.qout",
        "color": "#1976D2",   # blue
        "lw":    2.0,
        "ls":    "-",
    },
    {
        "label": "S94  |  r = 0.15",
        "file":  archive_dir / "SMF_20140812_63_r0p15_Outlet.qout",
        "color": "#E65100",   # orange
        "lw":    2.0,
        "ls":    "-",
    },
    {
        "label": "S95  |  Ks = 15×",
        "file":  archive_dir / "SMF_20140812_60_Ks15p0x_Outlet.qout",
        "color": "#6A1B9A",   # purple
        "lw":    2.0,
        "ls":    "-",
    },
    {
        "label": "truth100  |  Ks=7.0x, f=0.012",
        "file":  calib_dir / "02_results" / "60_sensitivity" / "SMF_20140812_60_Ks7p0x_truth100"
                 / "SMF_20140812_60_Ks7p0x_truth100_Outlet.qout",
        "color": "#2E7D32",   # green
        "lw":    2.5,
        "ls":    "-",
    },
]


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def load_qout(path: Path) -> pd.Series:
    """
    Read a tRIBS *_Outlet.qout file; return 5-min discharge Series.

    Handles both:
      - 5-min output (OPINTRVL = 0.0833 hr): resample().mean() is clean
      - 1-hr output  (OPINTRVL = 1 hr):       gaps filled by time-interpolation
    """
    df = pd.read_csv(
        path, sep=r'\s+', skiprows=1,
        names=["Time_hr", "Qstrm_m3s", "Hlev_m"]
    )
    df["datetime"] = pd.to_datetime(
        df["Time_hr"] * 3600, unit="s", origin=QOUT_ORIGIN
    )
    df.set_index("datetime", inplace=True)

    # Resample to 5-min; interpolate fills NaNs introduced for hourly-output files
    s = (
        df["Qstrm_m3s"]
        .resample("5min")
        .mean()
        .interpolate(method="time")
    )
    return s.loc[EVENT_START:EVENT_END]


def load_gauge() -> pd.Series:
    """Read real SMF gauge; return m³/s Series at native resolution, cropped."""
    obs = pd.read_excel(OBS_XLSX, sheet_name="Discharge", skiprows=6)
    obs["datetime"] = pd.to_datetime(
        obs["Date"].astype(str) + " " + obs["Time"].astype(str)
    )
    obs.set_index("datetime", inplace=True)
    s = (obs["cfs"] * 0.0283168).resample("5min").mean()
    return s.loc[EVENT_START:EVENT_END]


def peak_label(series: pd.Series) -> str:
    """Return a compact peak annotation string."""
    pk_q = series.max()
    pk_t = series.idxmax()
    return f"peak {pk_q:.2f} m³/s @ {pk_t.strftime('%H:%M')}"


# -----------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------
print("Loading gauge data...")
obs = load_gauge()
print(f"  Observed:  {peak_label(obs)}")

loaded = []   # (label, series, color, lw, ls)
for cfg in SERIES:
    p = cfg["file"]
    if not p.exists():
        print(f"  WARNING: file not found — {p.name}  (skipping)")
        continue
    s = load_qout(p)
    print(f"  {cfg['label']:30s}  {peak_label(s)}")
    loaded.append((cfg["label"], s, cfg["color"], cfg["lw"], cfg["ls"]))

if not loaded:
    raise RuntimeError("No truth series were loaded — check archive_dir and filenames.")

# -----------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

# Observed — always black, thickest line
ax.plot(
    obs.index, obs.values,
    color="black", linewidth=2.8, zorder=10,
    label=f"Observed (real gauge)  —  {peak_label(obs)}"
)

# Synthetic truths
for label, series, color, lw, ls in loaded:
    ax.plot(
        series.index, series.values,
        color=color, linewidth=lw, linestyle=ls,
        label=f"{label}  —  {peak_label(series)}"
    )

# Formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Time (Aug 12, 2014)", fontsize=12)
ax.set_ylabel("Discharge (m³/s)", fontsize=12)
ax.set_title(
    "SMF  |  Aug 12, 2014  |  Observed vs. New Truth Candidate (Ks=7.0x, f=0.012)",
    fontsize=12
)
ax.legend(fontsize=9.5, framealpha=0.88, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()

# -----------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------
out_path = plot_dir / "truth_hydrograph_comparison_truth100.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close(fig)
