"""
rebuild_100_narrow_config.py
==============================
Re-points current_run_config.json (and rewrites the matching .in file) at
the 100_narrow truth-point validation run, WITHOUT running tRIBS.

Context: validate_truth_point_100_storms.py's main() loops over all three
storms in sequence (storm080, 100_narrow, storm125) and calls build_only()
for each in turn -- each call overwrites current_run_config.json with that
storm's config. After a full run, current_run_config.json is left pointing
at whichever storm ran LAST (storm125), not 100_narrow.

Run_Model.ipynb (Cell 5) reads current_run_config.json directly, so before
running the notebook to test the execution-context hypothesis (Section 7
of the "Handoff: Series 100 Storm-Magnitude Investigation + Truth-Point
Validation Anomaly"), it needs to be pointed at 100_narrow specifically.
This script does exactly that -- it imports build_only() and STORMS from
validate_truth_point_100_storms.py and calls build_only() for the
100_narrow entry alone. It rewrites the .in file (content will be
identical to what's already on disk -- the true point, forcing, and
bounds haven't changed) and current_run_config.json. It does NOT execute
tRIBS -- that happens interactively when you run Run_Model.ipynb's Cell 8.

Usage (run from smf_demo/):
    python rebuild_100_narrow_config.py

After running this:
    1. Open Run_Model.ipynb and run Cells 3-8 (imports through "Run The
       Model") to execute tRIBS for the 100_narrow point via the
       notebook's own execution path (a Jupyter kernel process calling
       os.system() directly -- no subprocess wrapper, no detached
       session).
    2. Run score_notebook_run_against_truth.py to score that output
       against the 100_narrow synthetic truth and compare it directly to
       the wrapped-script result (PBIAS=-6.92%, KGE=0.9211).

Note: this reuses the exact same run_id as validate_truth_point_100_storms.py
used for 100_narrow (SMF_20140812_100_truthcheck_100_narrow), so the
notebook run will overwrite the raw .qout/.pixel files from that earlier
wrapped-script run in 02_results/100_truthcheck/. That's expected and fine
for this test -- the wrapped-script's comparison CSV and metrics summary
are already saved separately (plus the RUN1 copy from the determinism
check) and are not touched by this.
"""

from validate_truth_point_100_storms import build_only, STORMS

target = next(s for s in STORMS if s["label"] == "100_narrow")
run_id, summary_export_dir = build_only(target)

print(f"\ncurrent_run_config.json now points at: {run_id}")
print("Ready -- open Run_Model.ipynb and run Cells 3-8 to execute tRIBS for this point.")
