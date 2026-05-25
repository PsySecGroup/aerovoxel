#!/usr/bin/env python3
"""
bin_to_csv.py  —  Convert an aerovoxel sparse voxel .bin file to CSV.

Binary format (little-endian):
    int32   N           grid resolution (N x N x N)
    float32 voxel_size  side length of one voxel in world units
    int32   count       number of active (non-zero) voxel entries
    [int32 flat_index, float32 value] x count

Flat index encodes (ix, iy, iz) as:
    flat_index = ix * N*N + iy * N + iz

World-space voxel centre:
    x = grid_min_x + (ix + 0.5) * voxel_size
    y = grid_min_y + (iy + 0.5) * voxel_size
    z = grid_min_z + (iz + 0.5) * voxel_size

where grid_min = grid_center - 0.5 * N * voxel_size  (default center: 0, 0, 500)

Usage:
    python bin_to_csv.py input.bin [output.csv] [--center X Y Z]

Arguments:
    input.bin       Path to the sparse voxel binary file.
    output.csv      Output CSV path. Defaults to <input_stem>.csv.
    --center X Y Z  World-space grid centre. Default: 0 0 500
                    (must match the value used in ray_voxel.cpp)

Output columns:
    ix, iy, iz      Voxel grid indices (0-based)
    x, y, z         World-space voxel centre coordinates
    value           Accumulated motion-signal energy
"""

import argparse
import csv
import struct
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert aerovoxel sparse .bin voxel grid to CSV."
    )
    p.add_argument("input",  help="Path to the .bin file")
    p.add_argument("output", nargs="?", help="Output CSV path (default: <input>.csv)")
    p.add_argument(
        "--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 500.0],
        help="World-space grid centre (default: 0 0 500)"
    )
    return p.parse_args()


def read_bin(path):
    """
    Parse the sparse binary voxel file.

    Returns:
        N          (int)   grid resolution
        voxel_size (float) voxel side length in world units
        indices    (list of int) flat voxel indices
        values     (list of float) accumulated values
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

    # Unpack all (index, value) pairs in one shot using struct.iter_unpack.
    entries = struct.unpack_from(f"<{count * 2}i", data, 12)
    # Re-interpret alternating int32s as (index_int, value_float).
    # Value bits are stored as raw int32 and must be reinterpreted as float.
    indices = entries[0::2]
    values  = [struct.unpack("<f", struct.pack("<i", v))[0] for v in entries[1::2]]

    return N, voxel_size, indices, values


def decode_and_write(N, voxel_size, indices, values, center, out_path):
    """
    Decode flat indices to (ix, iy, iz), compute world-space coordinates,
    and write one CSV row per active voxel sorted by value descending.
    """
    cx, cy, cz = center
    half = 0.5 * N * voxel_size
    gx  = cx - half
    gy  = cy - half
    gz  = cz - half

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

    # Sort by value descending so the most significant voxels are at the top.
    rows.sort(key=lambda r: r[6], reverse=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ix", "iy", "iz", "x", "y", "z", "value"])
        writer.writerows(rows)

    return len(rows)


def main():
    args = parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")

    if not in_path.exists():
        sys.exit(f"ERROR: Input file not found: {in_path}")

    print(f"Reading {in_path} ...")
    N, voxel_size, indices, values = read_bin(in_path)

    print(f"  Grid:       {N} x {N} x {N}")
    print(f"  Voxel size: {voxel_size}")
    print(f"  Entries:    {len(indices):,}")
    print(f"  Center:     {args.center}")
    print(f"Writing {out_path} ...")

    count = decode_and_write(N, voxel_size, indices, values, args.center, out_path)
    print(f"Done. {count:,} rows written.")


if __name__ == "__main__":
    main()