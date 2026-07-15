"""
scale_precip_forcing.py
========================
Generates a storm-magnitude-scaled copy of the SMF/SMPHQ rain gauge
forcing (both MDF time series + the Master_Precip SDF), for testing
whether the Series 97 Ks-f equifinality "swoosh" shifts with storm size.

Only the R (rainfall, mm/hr) column is scaled by a constant factor.
Y/M/D/H are copied through unchanged, so timing/duration/spatial pattern
(including the known SMF/SMPHQ offset that drives the double-peak
artifact) are preserved exactly -- storm magnitude is the only variable
being changed. Station metadata in the SDF (lat/long, RecordLength,
elevation) is copied through unchanged; only FilePath is updated.

Usage (run from the smf_demo directory):
    python scale_precip_forcing.py --scale 0.80 --label storm080
    python scale_precip_forcing.py --scale 1.25 --label storm125

Output (written into ../smf_init_data/met/, alongside the originals):
    precip_SMF_1_<label>.mdf
    precip_SMPHQ_2_<label>.mdf
    Master_Precip_<label>.sdf

Then point build_sensitivity_run.py at the result with:
    --gauge_sdf ../smf_init_data/met/Master_Precip_<label>.sdf
"""

import argparse
from pathlib import Path

MET_DIR = Path("../smf_init_data/met")

# Station metadata copied verbatim from Master_Precip.sdf -- only the
# FilePath field changes for the scaled version. If Master_Precip.sdf is
# ever regenerated with different stations/metadata, update this to match.
STATIONS = [
    {"id": 1, "mdf": "precip_SMF_1.mdf",   "lat": 3686807, "long": 394483,
     "reclen": 1917, "nparams": 5, "elev": 389},
    {"id": 2, "mdf": "precip_SMPHQ_2.mdf", "lat": 3690196, "long": 398949,
     "reclen": 1917, "nparams": 5, "elev": 431},
]


def scale_mdf(src_path, dst_path, factor):
    """Scale the R column of a rain gauge MDF by `factor`; Y/M/D/H untouched."""
    with open(src_path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"{src_path} is empty")

    header = lines[0]
    if not header.strip().upper().startswith("Y"):
        raise ValueError(
            f"Unexpected header in {src_path}: {header!r} "
            f"(expected something like 'Y M D H R')"
        )

    out_lines = [header]
    n_rows = 0
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Unexpected column count in {src_path}: {line!r}")
        y, m, d, h, r = parts
        r_scaled = float(r) * factor
        out_lines.append(f"{y} {m} {d} {h} {r_scaled:.4f}\n")
        n_rows += 1

    with open(dst_path, "w") as f:
        f.writelines(out_lines)

    return n_rows


def write_sdf(dst_path, station_filenames):
    """Write a Master_Precip-style SDF pointing at the scaled MDF files."""
    lines = [f"{len(STATIONS)} 7\n"]
    for st, fname in zip(STATIONS, station_filenames):
        rel_path = f"../smf_init_data/met/{fname}"
        lines.append(
            f"{st['id']} {rel_path} {st['lat']} {st['long']} "
            f"{st['reclen']} {st['nparams']} {st['elev']}\n"
        )
    with open(dst_path, "w") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a storm-magnitude-scaled rain gauge forcing set."
    )
    parser.add_argument("--scale", required=True, type=float,
                         help="Multiplier on the R (rainfall) column, e.g. 0.80 or 1.25")
    parser.add_argument("--label", required=True, type=str,
                         help="Short label appended to filenames, e.g. 'storm080'")
    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError("--scale must be positive")

    print(f"Scaling precip forcing by x{args.scale} -> label '{args.label}'")

    station_out_files = []
    for st in STATIONS:
        src = MET_DIR / st["mdf"]
        if not src.exists():
            raise FileNotFoundError(f"Expected source MDF not found: {src}")
        stem = Path(st["mdf"]).stem  # e.g. "precip_SMF_1"
        dst_name = f"{stem}_{args.label}.mdf"
        dst = MET_DIR / dst_name
        n_rows = scale_mdf(src, dst, args.scale)
        station_out_files.append(dst_name)
        print(f"  Wrote {dst}  ({n_rows} rows, x{args.scale})")

    sdf_dst = MET_DIR / f"Master_Precip_{args.label}.sdf"
    write_sdf(sdf_dst, station_out_files)
    print(f"  Wrote {sdf_dst}")

    print(f"\nNext: point build_sensitivity_run.py at this forcing with")
    print(f"  --gauge_sdf ../smf_init_data/met/Master_Precip_{args.label}.sdf")


if __name__ == "__main__":
    main()
