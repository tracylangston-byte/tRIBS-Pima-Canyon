"""
compare_truthcheck_reruns.py
==============================
Determinism check for the Section 7 truth-point validation anomaly (see
"Handoff: Series 100 Storm-Magnitude Investigation + Truth-Point
Validation Anomaly").

Compares two *_compare_obs_sim.csv exports from
validate_truth_point_100_storms.py, run TWICE IN A ROW at the identical
(Ks_mult=7.0x, f_RS_abs=0.012, cv=4.5, r=0.24, n=0.026) point under
identical forcing, to test whether tRIBS is deterministic under these run
conditions.

  - If RUN1 and RUN2 are bit-for-bit identical: non-determinism is ruled
    out. The Section 7 anomaly must then come from an execution-context
    difference between the truth-generation pipeline (almost certainly
    build_sensitivity_run.py run directly, per Section 2's naming
    evidence) and the validate_truth_point_100_storms.py / subprocess-
    wrapper pipeline -- next step would be comparing how each actually
    launches tRIBS (in-process vs. subprocess, env/threading/cwd).

  - If RUN1 and RUN2 differ at all: non-determinism is confirmed as (at
    least part of) the cause, independent of any pipeline comparison --
    worth escalating to Josh directly given tRIBS' other documented
    stability issues (Ks~6.25x/f~0.011 hang zone, silent SIGSEGV), since
    a model already known to be fragile under rapid transitions is a
    reasonable candidate for subtle non-determinism under the same
    conditions even when it doesn't crash outright.

Because both CSVs read "Observed" from the SAME synthetic-truth .qout
file (unchanged between reruns), the Observed columns are expected to
match exactly regardless of the outcome -- this script checks that too,
as a basic sanity check that both files really are comparable runs.

Usage:
    python compare_truthcheck_reruns.py
    python compare_truthcheck_reruns.py --label storm080
    python compare_truthcheck_reruns.py --run1 /path/to/RUN1.csv --run2 /path/to/RUN2.csv

    Defaults assume the standard convention from this investigation:
        RUN1 = SMF_20140812_100_truthcheck_<label>_compare_obs_sim_RUN1.csv
               (your saved copy of the first rerun)
        RUN2 = SMF_20140812_100_truthcheck_<label>_compare_obs_sim.csv
               (the current file, i.e. whatever the most recent rerun wrote)
    for --label (default 100_narrow), unless --run1/--run2 are given explicitly.

Output:
    Prints: whether Observed matches between the two files (sanity check),
    whether Simulated matches exactly, and if not, where and by how much
    (max/mean abs diff, RMSE, number of differing rows, largest single
    discrepancy and its timestamp). Recomputes PBIAS/KGE independently
    for each run's Simulated-vs-Observed pair and compares those too.

    If Simulated differs at all between RUN1 and RUN2, saves a diagnostic
    plot (RUN1 vs RUN2 overlay + their residual over time) to:
        calibration_work/03_comparisons/sensitivity_plots/truth_comparison/
            truthcheck_determinism_check_<label>.png
    so the divergence can be compared visually to where the Observed-vs-
    Simulated mismatch was concentrated in the original diagnostic.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--label", default="100_narrow",
                     help="Validation run label used to build default RUN1/RUN2 "
                          "paths (100_narrow, storm080, storm125). Default: 100_narrow")
parser.add_argument("--run1", default=None, help="Explicit path to the RUN1 csv.")
parser.add_argument("--run2", default=None, help="Explicit path to the RUN2 csv.")
parser.add_argument("--outdir", default=None,
                     help="Directory to save the diagnostic figure, if divergence is found.")
parser.add_argument("--tol", type=float, default=0.0,
                     help="Absolute-difference tolerance (m3/s) below which a value is "
                          "still counted as 'matching' -- default 0.0 (exact match required).")
args = parser.parse_args()

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
csv_dir      = calib_dir / "03_comparisons" / "csv_exports"

run1_path = Path(args.run1) if args.run1 else \
    csv_dir / f"SMF_20140812_100_truthcheck_{args.label}_compare_obs_sim_RUN1.csv"
run2_path = Path(args.run2) if args.run2 else \
    csv_dir / f"SMF_20140812_100_truthcheck_{args.label}_compare_obs_sim.csv"

if args.outdir is not None:
    plot_dir = Path(args.outdir)
else:
    plot_dir = calib_dir / "03_comparisons" / "sensitivity_plots" / "truth_comparison"

for p, tag in [(run1_path, "RUN1"), (run2_path, "RUN2")]:
    if not p.exists():
        sys.exit(f"ERROR: {tag} file not found: {p}")

# -----------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------
run1 = pd.read_csv(run1_path, index_col=0, parse_dates=True)
run2 = pd.read_csv(run2_path, index_col=0, parse_dates=True)

print(f"RUN1: {run1_path.name}  ({len(run1)} rows)")
print(f"RUN2: {run2_path.name}  ({len(run2)} rows)\n")

if len(run1) != len(run2) or not run1.index.equals(run2.index):
    print("WARNING: RUN1 and RUN2 do not share an identical timestamp index. "
          "Aligning on the intersection before comparing -- this alone would "
          "be a notable finding (non-identical event windows between reruns).")
    common = run1.index.intersection(run2.index)
    run1, run2 = run1.loc[common], run2.loc[common]
    print(f"  Aligned on {len(common)} common timestamps.\n")

# -----------------------------------------------------------------------
# METRICS (for independent PBIAS/KGE comparison) -- same formulas as
# run_sensitivity_single.py
# -----------------------------------------------------------------------
def compute_metrics(obs, sim):
    r     = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    pbias = 100 * (np.sum(sim - obs) / np.sum(obs))
    rmse  = np.sqrt(np.mean((sim - obs) ** 2))
    # Kling et al. (2012) modified KGE, added alongside 2009 per
    # Handoff_KGE2012Transition_v1.md. gamma = alpha/beta exactly.
    gamma    = alpha / beta
    kge_2012 = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
    return dict(r=r, alpha=alpha, beta=beta, kge=kge, pbias=pbias, rmse=rmse,
                gamma=gamma, kge_2012=kge_2012)

m1 = compute_metrics(run1["Observed"], run1["Simulated"])
m2 = compute_metrics(run2["Observed"], run2["Simulated"])

print("=== Independently recomputed metrics, each run vs. its own Observed column ===")
print(f"  RUN1:  PBIAS={m1['pbias']:+.4f}%  KGE={m1['kge']:.4f}  KGE_2012={m1['kge_2012']:.4f}  (r={m1['r']:.4f}, alpha={m1['alpha']:.4f}, gamma={m1['gamma']:.4f}, beta={m1['beta']:.4f})")
print(f"  RUN2:  PBIAS={m2['pbias']:+.4f}%  KGE={m2['kge']:.4f}  KGE_2012={m2['kge_2012']:.4f}  (r={m2['r']:.4f}, alpha={m2['alpha']:.4f}, gamma={m2['gamma']:.4f}, beta={m2['beta']:.4f})")
print(f"  Delta: dPBIAS={m2['pbias']-m1['pbias']:+.6f} pct pts   dKGE={m2['kge']-m1['kge']:+.6f}   dKGE_2012={m2['kge_2012']-m1['kge_2012']:+.6f}\n")

# -----------------------------------------------------------------------
# SANITY CHECK: Observed columns should be identical (same truth .qout
# read both times) -- if not, something more fundamental than run-to-run
# tRIBS determinism is going on.
# -----------------------------------------------------------------------
obs_diff = (run1["Observed"] - run2["Observed"]).abs()
if (obs_diff <= args.tol).all():
    print(f"Observed columns MATCH between RUN1 and RUN2 (max abs diff = {obs_diff.max():.2e}). Sanity check passed.\n")
else:
    print(f"WARNING: Observed columns DIFFER between RUN1 and RUN2 "
          f"(max abs diff = {obs_diff.max():.6f} m3/s at {obs_diff.idxmax()}). "
          f"This is unexpected -- the truth .qout file should be identical and unread-modified "
          f"between reruns; investigate this before trusting the Simulated comparison below.\n")

# -----------------------------------------------------------------------
# THE ACTUAL DETERMINISM CHECK: does Simulated match between reruns?
# -----------------------------------------------------------------------
sim_diff     = (run1["Simulated"] - run2["Simulated"])
sim_diff_abs = sim_diff.abs()
n_differ     = int((sim_diff_abs > args.tol).sum())
n_total      = len(sim_diff_abs)

print("=== DETERMINISM CHECK: Simulated (RUN1 vs RUN2) ===")
if n_differ == 0:
    print(f"  IDENTICAL. All {n_total} timesteps match within tolerance ({args.tol} m3/s).")
    print("  --> Non-determinism is RULED OUT. The Section 7 anomaly must come from an")
    print("      execution-context difference between the truth-generation pipeline and")
    print("      the validate_truth_point_100_storms.py / subprocess-wrapper pipeline.")
else:
    worst_t = sim_diff_abs.idxmax()
    print(f"  DIFFER at {n_differ}/{n_total} timesteps ({100*n_differ/n_total:.1f}%).")
    print(f"  Max abs diff:   {sim_diff_abs.max():.6f} m3/s   at {worst_t}")
    print(f"  Mean abs diff (over differing rows): {sim_diff_abs[sim_diff_abs > args.tol].mean():.6f} m3/s")
    print(f"  RMSE(RUN1, RUN2):  {np.sqrt(np.mean(sim_diff ** 2)):.6f} m3/s")
    print(f"  RUN1 value at worst timestep: {run1['Simulated'].loc[worst_t]:.4f}   "
          f"RUN2 value: {run2['Simulated'].loc[worst_t]:.4f}")
    print("  --> Non-determinism CONFIRMED. This is independent of any pipeline comparison")
    print("      and worth taking directly to Josh.")
print()

# -----------------------------------------------------------------------
# PLOT (only if a divergence was found -- nothing useful to show for a
# bit-for-bit match)
# -----------------------------------------------------------------------
if n_differ > 0:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    ax = axes[0]
    ax.plot(run1.index, run1["Observed"].values, color="black", linewidth=1.6,
            label="Observed (truth, both runs)")
    ax.plot(run1.index, run1["Simulated"].values, color="#1976D2", linewidth=1.6,
            linestyle="-", label="Simulated — RUN1")
    ax.plot(run2.index, run2["Simulated"].values, color="#E65100", linewidth=1.6,
            linestyle="--", label="Simulated — RUN2")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(f"{args.label}: determinism check — two reruns at the identical true point")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    ax = axes[1]
    ax.axhline(0, color="gray", linewidth=1)
    ax.plot(sim_diff.index, sim_diff.values, color="#6A1B9A", linewidth=1.3)
    ax.fill_between(sim_diff.index, sim_diff.values, 0, color="#6A1B9A", alpha=0.25)
    ax.set_ylabel("RUN1 − RUN2\nSimulated (m3/s)")
    ax.set_title("Where the two reruns diverge")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.tight_layout()
    out_path = plot_dir / f"truthcheck_determinism_check_{args.label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
