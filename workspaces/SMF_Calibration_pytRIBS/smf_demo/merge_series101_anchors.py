"""
merge_series101_anchors.py
=============================
Fixes lhs_results_anchor_ALL_101.csv, which turned out not to be a real
merge -- it's byte-identical to lhs_results_anchor_f0p03_101.csv (same
size, same timestamp: 59952 bytes, Aug 23 22:42 for both), meaning
whatever step was supposed to concatenate all six anchors into it
instead just left it holding the most-recently-processed anchor. The
other five anchors' data was never lost -- it's sitting untouched in
their own correctly-named per-anchor files. This script concatenates
the six real anchor files directly, without touching or depending on
the broken ALL file at all.

Explicit file list (not a glob) so this can't accidentally pull in
lhs_results_anchor_FAILED_101.csv, the Series 96/99 anchor files, or
anything else matching "anchor" in that directory -- there are a lot of
similarly-named files in that folder.

Usage (run from smf_demo/):
    python merge_series101_anchors.py

Output:
    calibration_work/03_comparisons/summary_tables/
        lhs_results_anchor_ALL_101_MERGED.csv
    (deliberately NOT overwriting the broken ALL_101.csv -- decide for
    yourself whether to replace it once you've spot-checked this output)
"""

from pathlib import Path

import pandas as pd

# ======================================================================
# CONFIG
# ======================================================================
ANCHOR_FILES = [
    "lhs_results_anchor_f0p006_101.csv",
    "lhs_results_anchor_f0p008_101.csv",
    "lhs_results_anchor_f0p010_101.csv",
    "lhs_results_anchor_f0p015_101.csv",
    "lhs_results_anchor_f0p02_101.csv",
    "lhs_results_anchor_f0p03_101.csv",
]
EXPECTED_ANCHOR_LABELS = {"f0p006", "f0p008", "f0p010", "f0p015", "f0p02", "f0p03"}
OUTPUT_NAME = "lhs_results_anchor_ALL_101_MERGED.csv"

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
script_dir   = Path.cwd()
project_root = script_dir.parent if script_dir.name == "smf_demo" else script_dir
calib_dir    = project_root / "calibration_work"
summary_dir  = calib_dir / "03_comparisons" / "summary_tables"

# -----------------------------------------------------------------------
# LOAD + CONCAT
# -----------------------------------------------------------------------
frames = []
for fname in ANCHOR_FILES:
    path = summary_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Expected anchor file not found: {path}")
    df = pd.read_csv(path)
    if "anchor_label" not in df.columns:
        raise ValueError(f"{fname} has no 'anchor_label' column -- can't verify "
                          f"which anchor this data belongs to. Check the file.")
    labels_found = set(df["anchor_label"].unique())
    if len(labels_found) != 1:
        print(f"  WARNING: {fname} contains multiple anchor_label values: "
              f"{labels_found} -- expected exactly one.")
    frames.append(df)
    print(f"  Loaded {len(df)} rows from {fname}  (anchor_label: {labels_found})")

merged = pd.concat(frames, ignore_index=True)

# -----------------------------------------------------------------------
# VALIDATE
# -----------------------------------------------------------------------
found_labels = set(merged["anchor_label"].unique())
print(f"\nMerged: {len(merged)} total rows across {len(found_labels)} anchors: "
      f"{sorted(found_labels)}")

if found_labels != EXPECTED_ANCHOR_LABELS:
    missing = EXPECTED_ANCHOR_LABELS - found_labels
    extra = found_labels - EXPECTED_ANCHOR_LABELS
    print(f"  WARNING: anchor labels don't match expected set.")
    if missing:
        print(f"    Missing: {missing}")
    if extra:
        print(f"    Unexpected: {extra}")
else:
    print("  All six expected anchors present, no unexpected labels. Looks right.")

counts = merged["anchor_label"].value_counts()
print("\nRows per anchor:")
for label, n in counts.items():
    flag = "" if n == 100 else "  <-- expected 100 (50 matched + 50 independent), check this one"
    print(f"  {label}: {n}{flag}")

# -----------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------
out_path = summary_dir / OUTPUT_NAME
merged.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\nThis is a NEW file -- lhs_results_anchor_ALL_101.csv (the broken one) "
      "was not touched. Point analyze_series101_kge_components.py's "
      "COMBINED_RESULTS_CSV at this new filename, or spot-check this output "
      "and rename it over the broken one yourself.")
