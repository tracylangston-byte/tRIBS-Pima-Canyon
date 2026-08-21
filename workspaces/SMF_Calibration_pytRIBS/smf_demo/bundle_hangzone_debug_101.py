"""
bundle_hangzone_debug_101.py
=============================
Packages tRIBS input files, parameter files, console logs, and raw output
for HANG/FAILED draws from Series 101, for sending to Josh to investigate
the hang-zone behavior.

Read-only with respect to calibration_work/ -- only reads existing files and
writes a single new zip archive. Safe to run concurrently with an active
run_lhs_nanchor_cvrn_101.py sweep: it only touches run_ids already recorded
in lhs_results_anchor_FAILED_101.csv (plus, optionally, orphaned run_ids
found via --find_orphans), which are a disjoint filename set from whatever
the live sweep is currently building/running (different anchor label ->
different run_id string -> different files on disk).

v2 changes from the first version:
  - results_dir reporting is now three-way instead of a single MISSING
    label: NO_DIR (directory never created), EMPTY (directory exists but
    tRIBS wrote nothing into it before being killed), or OK (has files).
    The old single MISSING label conflated the first two, which matters
    here: build_only() creates run_results_dir via mkdir() before tRIBS
    ever runs, so EMPTY is the expected state for a run killed early, and
    is itself informative (hang happens before any output write) rather
    than a script problem.
  - --find_orphans: scans 01_run_inputs/ for <anchor>*.in files whose
    run_id never made it into lhs_results_anchor_FAILED_101.csv (or the
    per-anchor SUCCESS csv, if present). This catches runs that were
    hanging when the *outer* sweep process was killed directly (rather
    than killed by its own 5-minute per-run timeout) -- that outer kill
    happens before the HANG row would have been logged, so the run is
    otherwise invisible to anything reading only the FAILED csv.

Usage (run from the smf_demo directory):
    python bundle_hangzone_debug_101.py                                    # all HANG/FAILED rows in the CSV
    python bundle_hangzone_debug_101.py --anchor f0p012_true               # just this anchor
    python bundle_hangzone_debug_101.py --anchor f0p012_true --find_orphans # also look for unlogged hung runs
    python bundle_hangzone_debug_101.py --status HANG                      # only HANG, not FAILED/BUILD_ERROR
    python bundle_hangzone_debug_101.py --dry_run                          # list what would be included, don't zip

Output: series101_hangzone_bundle_<timestamp>.zip in the current directory,
containing, per run_id:
    <run_id>/<run_id>.in                  -- tRIBS input file
    <run_id>/soils_<run_id>.sdt           -- soil table
    <run_id>/<run_id>.log                 -- tRIBS console output up to the kill
    <run_id>/results_dir/...              -- whatever raw mesh/Voronoi output
                                              tRIBS wrote before being killed
                                              (often empty -- see above)
plus manifest.txt (what was found / empty / missing, and which run_ids came
from the CSV vs. the orphan scan) and a copy of lhs_results_anchor_FAILED_101.csv
itself, for full context.
"""

import argparse
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

LHS_CATEGORY = "101_lhs_nanchor_cvrn"
LOCATION     = "SMF"
EVENT_DATE   = "20140812"
LHS_SERIES   = "101"


def resolve_calib_dir():
    script_dir = Path.cwd()
    project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
    calib_dir = project_root / "calibration_work"
    if not calib_dir.exists():
        sys.exit(f"ERROR: could not find calibration_work/ from {script_dir} "
                  f"(looked at {calib_dir}). Run this from smf_demo/.")
    return calib_dir


def gather_run_files(calib_dir, run_id):
    """Return dict of {label: Path} for every file/dir associated with one
    run_id, whether or not it currently exists on disk."""
    run_input_dir   = calib_dir / "01_run_inputs" / LHS_CATEGORY
    run_results_dir = calib_dir / "02_results"    / LHS_CATEGORY / run_id
    log_dir         = calib_dir / "06_logs"

    return {
        "input_file":  run_input_dir / f"{run_id}.in",
        "soil_table":  run_input_dir / f"soils_{run_id}.sdt",
        "console_log": log_dir / f"{run_id}.log",
        "results_dir": run_results_dir,
    }


def classify_results_dir(path):
    """Three-way state instead of a single MISSING label -- see module
    docstring for why the distinction matters here."""
    if not path.exists():
        return "NO_DIR", 0
    files = [f for f in path.rglob("*") if f.is_file()]
    if not files:
        return "EMPTY_DIR", 0
    return "OK", len(files)


def find_orphan_run_ids(calib_dir, anchor_label, known_run_ids):
    """Find <anchor>*.in files whose run_id isn't in known_run_ids -- i.e.
    runs that were built and (likely) launched but whose HANG/FAILED/SUCCESS
    status never got logged anywhere, e.g. because the outer sweep process
    was killed directly, before its own per-run timeout could fire and log
    the row."""
    run_input_dir = calib_dir / "01_run_inputs" / LHS_CATEGORY
    prefix = f"{LOCATION}_{EVENT_DATE}_{LHS_SERIES}_{anchor_label}_"
    orphans = []
    if not run_input_dir.exists():
        return orphans
    for f in sorted(run_input_dir.glob(f"{prefix}*.in")):
        run_id = f.stem
        if run_id not in known_run_ids:
            orphans.append(run_id)
    return orphans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", default=None,
                     help="Only bundle rows with this anchor_label (default: all anchors)")
    ap.add_argument("--status", default=None,
                     help="Only bundle rows with this status, e.g. HANG (default: all statuses)")
    ap.add_argument("--find_orphans", action="store_true",
                     help="Also scan for .in files under this anchor whose run_id never made "
                          "it into the FAILED csv (e.g. because the outer process was killed "
                          "before its own timeout could log the row). Requires --anchor.")
    ap.add_argument("--dry_run", action="store_true",
                     help="List what would be included; don't write a zip")
    args = ap.parse_args()

    if args.find_orphans and not args.anchor:
        sys.exit("ERROR: --find_orphans requires --anchor (orphan scanning matches "
                  "filenames by anchor-label prefix).")

    calib_dir  = resolve_calib_dir()
    failed_csv = calib_dir / "03_comparisons" / "summary_tables" / "lhs_results_anchor_FAILED_101.csv"
    if not failed_csv.exists():
        sys.exit(f"ERROR: {failed_csv} not found.")

    df = pd.read_csv(failed_csv)
    if args.anchor:
        df = df[df["anchor_label"] == args.anchor]
    if args.status:
        df = df[df["status"] == args.status]

    run_entries = [(row["run_id"], row["status"], row["anchor_label"], "FAILED_csv")
                   for _, row in df.iterrows()]

    if args.find_orphans:
        known_ids = set(r[0] for r in run_entries)
        success_csv = (calib_dir / "03_comparisons" / "summary_tables"
                        / f"lhs_results_anchor_{args.anchor}_{LHS_SERIES}.csv")
        if success_csv.exists():
            try:
                sdf = pd.read_csv(success_csv)
                if "run_id" in sdf.columns:
                    known_ids |= set(sdf["run_id"])
            except Exception as e:
                print(f"  Warning: could not read {success_csv.name} ({e}); "
                      f"proceeding without it for orphan-exclusion.")

        orphans = find_orphan_run_ids(calib_dir, args.anchor, known_ids)
        if orphans:
            print(f"Found {len(orphans)} orphaned run(s) -- built, but no status ever "
                  f"logged (likely killed before their own timeout fired):")
            for o in orphans:
                print(f"  {o}")
            print()
        else:
            print("No orphaned runs found -- every .in file for this anchor is "
                  "accounted for in the FAILED csv (or SUCCESS csv, if present).\n")
        for run_id in orphans:
            run_entries.append((run_id, "ORPHAN_UNLOGGED", args.anchor, "orphan_scan"))

    if not run_entries:
        sys.exit("No matching runs found -- check --anchor/--status filters.")

    print(f"Bundling {len(run_entries)} run(s) "
          f"(anchor={args.anchor or 'ALL'}, status={args.status or 'ALL'}, "
          f"find_orphans={args.find_orphans})\n")

    found, not_found = [], []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path  = Path.cwd() / f"series101_hangzone_bundle_{timestamp}.zip"

    zf = None if args.dry_run else zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

    manifest_lines = [
        "Series 101 HANG/FAILED debug bundle",
        f"Generated: {datetime.now().isoformat()}",
        f"Source: {failed_csv}" + (" + orphan scan" if args.find_orphans else ""),
        f"Filter: anchor={args.anchor or 'ALL'}  status={args.status or 'ALL'}",
        f"Rows: {len(run_entries)}",
        "",
    ]

    for run_id, status, anchor, source in run_entries:
        manifest_lines.append(f"--- {run_id}  (anchor={anchor}, status={status}, source={source}) ---")
        paths = gather_run_files(calib_dir, run_id)

        for label, path in paths.items():
            if label == "results_dir":
                state, n_files = classify_results_dir(path)
                if state == "OK":
                    manifest_lines.append(f"  [OK]      {label}: {path}  ({n_files} files)")
                    found.append((run_id, label, path))
                    if zf is not None:
                        for f in path.rglob("*"):
                            if f.is_file():
                                arcname = f"{run_id}/results_dir/{f.relative_to(path)}"
                                zf.write(f, arcname)
                elif state == "EMPTY_DIR":
                    manifest_lines.append(f"  [EMPTY]   {label}: {path}  "
                                           f"(dir exists, 0 files -- killed before any output write)")
                    not_found.append((run_id, label, path, "empty"))
                else:
                    manifest_lines.append(f"  [NO_DIR]  {label}: {path}  (directory never created)")
                    not_found.append((run_id, label, path, "no_dir"))
            else:
                if path.exists():
                    manifest_lines.append(f"  [OK]      {label}: {path}")
                    found.append((run_id, label, path))
                    if zf is not None:
                        zf.write(path, f"{run_id}/{path.name}")
                else:
                    manifest_lines.append(f"  [MISSING] {label}: {path}")
                    not_found.append((run_id, label, path, "missing"))
        manifest_lines.append("")

    manifest_text = "\n".join(manifest_lines)
    print(manifest_text)

    if zf is not None:
        zf.writestr("manifest.txt", manifest_text)
        zf.write(failed_csv, "lhs_results_anchor_FAILED_101.csv")
        zf.close()
        print(f"\nWrote {zip_path}  ({len(found)} item(s) found, {len(not_found)} empty/missing)")
    else:
        print(f"\n[DRY RUN] Would write to {zip_path}  "
              f"({len(found)} item(s) found, {len(not_found)} empty/missing)")


if __name__ == "__main__":
    main()