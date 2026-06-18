"""
summarize_basin_classes.py
==========================
Reads ADOT_SoilTypes.asc and LandUse.asc from smf_init_data/ and reports
the percentage of valid (non-NODATA) raster cells belonging to each class.

Run from the smf_demo/ directory:
    python summarize_basin_classes.py

Output: printed table + optional CSV export to calibration_work/summary_tables/
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to smf_demo/)
# ---------------------------------------------------------------------------
SOIL_RAS  = Path('../smf_init_data/ADOT_SoilTypes.asc')
LAND_RAS  = Path('../smf_init_data/LandUse.asc')
OUTPUT_DIR = Path('../calibration_work/summary_tables')

# ---------------------------------------------------------------------------
# Class name lookups
# ---------------------------------------------------------------------------
SOIL_NAMES = {
    1: 'RS  (Rocky shallow, caliche-controlled)',
    2: 'CO  (Colluvial/outwash)',
    3: 'CeD (Ajo-Ebon gravelly sandy loam)',
    4: 'EbD (Ebon gravelly loam)',
    5: 'Cb  (Carrizo coarse sand)',
}

LAND_NAMES = {
    1: 'Desert scrub, sparse  (V=0.15)',
    2: 'Desert scrub, moderate (V=0.30)',
    3: 'Bare / impervious      (V=0.01)',
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def load_asc(path):
    """Read an ESRI ASCII raster, return header dict and data array.
    Header always includes 'nodata_value' parsed from the file itself."""
    header = {}
    with open(path) as f:
        for _ in range(6):
            key, val = f.readline().split()
            header[key.lower()] = val
    data = np.loadtxt(path, skiprows=6)
    return header, data


def class_summary(data, header, names_lookup, label):
    nodata = float(header.get('nodata_value', -9999))
    valid = data[data != nodata].astype(int)
    total = valid.size
    ids, counts = np.unique(valid, return_counts=True)

    rows = []
    for cid, count in zip(ids, counts):
        rows.append({
            'ID': int(cid),
            'Name': names_lookup.get(int(cid), f'Unknown ID {cid}'),
            'Cell count': int(count),
            'Pct of basin (%)': round(count / total * 100, 2),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Total valid cells: {total:,}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Soil
    hdr_s, soil_data = load_asc(SOIL_RAS)
    print(f"\nSoil raster  : {SOIL_RAS}")
    print(f"  nrows={hdr_s['nrows']}, ncols={hdr_s['ncols']}, cellsize={hdr_s['cellsize']} m, nodata={hdr_s.get('nodata_value','?')}")
    df_soil = class_summary(soil_data, hdr_s, SOIL_NAMES, 'SOIL CLASSES (ADOT_SoilTypes.asc)')

    # Land use
    hdr_l, land_data = load_asc(LAND_RAS)
    print(f"\nLand use raster: {LAND_RAS}")
    print(f"  nrows={hdr_l['nrows']}, ncols={hdr_l['ncols']}, cellsize={hdr_l['cellsize']} m, nodata={hdr_l.get('nodata_value','?')}")
    df_land = class_summary(land_data, hdr_l, LAND_NAMES, 'LAND USE CLASSES (LandUse.asc)')

    # Optional CSV export
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    soil_out = OUTPUT_DIR / 'basin_soil_class_coverage.csv'
    land_out = OUTPUT_DIR / 'basin_landuse_class_coverage.csv'
    df_soil.to_csv(soil_out, index=False)
    df_land.to_csv(land_out, index=False)
    print(f"\nCSVs written to:")
    print(f"  {soil_out}")
    print(f"  {land_out}")


if __name__ == '__main__':
    main()
