#!/usr/bin/env python3
"""
bin_to_csv.py  —  Convert an aerovoxel sparse voxel .bin file to CSV.

Automatically reads profile.json from the same directory as the .bin file
to resolve grid center and other parameters, matching the behaviour of
ray_voxel exactly. Falls back to hardcoded defaults if no profile is found.

Binary format (little-endian):
    int32   N           grid resolution (N x N x N)
    float32 voxel_size  side length of one voxel in world units
    int32   count       number of active (non-zero) voxel entries
    [int32 flat_index, float32 value] x count

Flat index encodes (ix, iy, iz) as:
    flat_index = ix * N*N + iy * N + iz

Usage:
    python bin_to_csv.py input.bin [output.csv]

Arguments:
    input.bin   Path to the sparse voxel binary file.
    output.csv  Output CSV path. Defaults to <input_stem>.csv.

Output columns:
    ix, iy, iz      Voxel grid indices (0-based)
    x, y, z         World-space voxel centre coordinates
    value           Accumulated motion-signal energy
"""

import argparse
import csv
import json
import math
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Profile loading — mirrors the logic in ray_voxel.cpp load_profile()
# ---------------------------------------------------------------------------

DEFAULTS = {
    "center":     [0.0, 0.0, 500.0],
    "center_auto": False,
}


def load_profile(bin_path: Path) -> dict:
    """
    Look for profile.json in the same directory as the .bin file.
    Returns a dict with resolved parameters, falling back to defaults
    for any key that is absent.

    Mirrors the field resolution in ray_voxel.cpp's load_profile().
    """
    profile_path = bin_path.parent / "profile.json"
    params = dict(DEFAULTS)

    if not profile_path.exists():
        print(f"  [Profile] No profile.json found at {profile_path} — using defaults.")
        return params

    try:
        with open(profile_path) as f:
            j = json.load(f)
    except Exception as e:
        print(f"  [Profile] Failed to parse {profile_path}: {e} — using defaults.")
        return params

    print(f"  [Profile] Loaded {profile_path}")

    grid = j.get("grid", {})
    center = grid.get("center", "auto")

    if center == "auto":
        # Signal to caller that we need the metadata to compute the centroid.
        params["center_auto"] = True
    elif isinstance(center, list) and len(center) == 3:
        params["center"] = [float(v) for v in center]
        params["center_auto"] = False
    else:
        print(f"  [Profile] Unrecognised center value '{center}' — using default.")

    return params


def resolve_auto_center(bin_path: Path) -> list:
    """
    When center=auto, compute the mean camera position from metadata.json,
    matching what ray_voxel.cpp does at runtime.
    """
    metadata_path = bin_path.parent / "metadata.json"
    if not metadata_path.exists():
        print(f"  [Profile] center=auto but no metadata.json found at {metadata_path}.")
        print(f"  [Profile] Falling back to default center {DEFAULTS['center']}.")
        return DEFAULTS["center"]

    with open(metadata_path) as f:
        frames = json.load(f)

    positions = [fr["camera_position"] for fr in frames if "camera_position" in fr]
    if not positions:
        print("  [Profile] center=auto but no camera_position entries in metadata.")
        print(f"  [Profile] Falling back to default center {DEFAULTS['center']}.")
        return DEFAULTS["center"]

    n   = len(positions)
    cx  = sum(p[0] for p in positions) / n
    cy  = sum(p[1] for p in positions) / n
    cz  = sum(p[2] for p in positions) / n
    center = [cx, cy, cz]
    print(f"  [Profile] Auto-centred grid at {[round(v, 2) for v in center]}")
    return center


# ---------------------------------------------------------------------------
# Binary parsing
# ---------------------------------------------------------------------------

def read_bin(path: Path):
    """
    Parse the sparse binary voxel file.

    Returns:
        N          (int)
        voxel_size (float)
        indices    (tuple of int)
        values     (list of float)
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 12:
        sys.exit(f"ERROR: File too small to be a valid .bin ({len(data)} bytes)")

    N          = struct.unpack_from("<i", data, 0)[0]
    voxel_size = struct.unpack_from("<f", data, 4)[0]
    count      = struct.unpack_from("<i", data, 8)[0]

    expected = 12 + count * 8
    if len(data) < expected:
        sys.exit(
            f"ERROR: File is {len(data)} bytes but header says {count} entries "
            f"(need {expected} bytes)"
        )

    entries = struct.unpack_from(f"<{count * 2}i", data, 12)
    indices = entries[0::2]
    values  = [struct.unpack("<f", struct.pack("<i", v))[0] for v in entries[1::2]]

    return N, voxel_size, indices, values


# ---------------------------------------------------------------------------
# Decoding and CSV writing
# ---------------------------------------------------------------------------

def decode_and_write(N, voxel_size, indices, values, center, out_path: Path):
    """
    Decode flat indices to (ix, iy, iz), compute world-space coordinates,
    and write one CSV row per active voxel sorted by value descending.
    """
    cx, cy, cz = center
    half = 0.5 * N * voxel_size
    gx   = cx - half
    gy   = cy - half
    gz   = cz - half

    rows = []
    n2   = N * N
    for flat, val in zip(indices, values):
        ix = flat // n2
        iy = (flat // N) % N
        iz = flat % N
        wx = gx + (ix + 0.5) * voxel_size
        wy = gy + (iy + 0.5) * voxel_size
        wz = gz + (iz + 0.5) * voxel_size
        rows.append((ix, iy, iz, wx, wy, wz, val))

    rows.sort(key=lambda r: r[6], reverse=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ix", "iy", "iz", "x", "y", "z", "value"])
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Convert aerovoxel sparse .bin voxel grid to CSV."
    )
    p.add_argument("input",  help="Path to the .bin file")
    p.add_argument("output", nargs="?", help="Output CSV path (default: <input>.csv)")
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")

    if not in_path.exists():
        sys.exit(f"ERROR: Input file not found: {in_path}")

    print(f"Reading {in_path} ...")
    N, voxel_size, indices, values = read_bin(in_path)

    params = load_profile(in_path)
    if params["center_auto"]:
        center = resolve_auto_center(in_path)
    else:
        center = params["center"]

    half = 0.5 * N * voxel_size
    print(f"  Grid:       {N} x {N} x {N}")
    print(f"  Voxel size: {voxel_size}")
    print(f"  Entries:    {len(indices):,}")
    print(f"  Center:     {[round(v, 2) for v in center]}")
    print(f"  X range:    [{center[0]-half:.1f}, {center[0]+half:.1f}]")
    print(f"  Y range:    [{center[1]-half:.1f}, {center[1]+half:.1f}]")
    print(f"  Z range:    [{center[2]-half:.1f}, {center[2]+half:.1f}]")
    print(f"Writing {out_path} ...")

    count = decode_and_write(N, voxel_size, indices, values, center, out_path)
    print(f"Done. {count:,} rows written.")


if __name__ == "__main__":
    main()