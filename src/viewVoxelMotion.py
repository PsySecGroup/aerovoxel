"""
viewVoxelMotion.py

Loads a sparse voxel grid (.bin), reads grid_center from the companion
profile.json, and displays the result in world-space coordinates with:
  - Correct X/Y/Z mapping (fixes original X↔Z swap)
  - Labelled XYZ axis lines from the world origin
  - Wireframe box showing the full grid extent
  - Camera position markers (from metadata.json if present)
  - show_bounds() grid with world-space tick labels
  - Corner axes widget

Usage:
    python src/viewVoxelMotion.py examples/afternoon/voxel_grid.bin
"""

import os
import sys
import re
import json
import math
import numpy as np
import pyvista as pv
from pathlib import Path


# ---------------------------------------------------------------------------
# Bin loading
# ---------------------------------------------------------------------------

def load_voxel_grid(filename):
    """
    Sparse binary format:
      int32  N           (resolution; grid is N×N×N)
      float32 voxel_size
      int32  count
      count × (int32 flat_index, float32 value)

    flat_index = ix*N*N + iy*N + iz   where ix→X, iy→Y, iz→Z
    """
    with open(filename, "rb") as f:
        N          = np.frombuffer(f.read(4), dtype=np.int32)[0]
        voxel_size = np.frombuffer(f.read(4), dtype=np.float32)[0]
        count      = np.frombuffer(f.read(4), dtype=np.int32)[0]
        raw        = np.frombuffer(f.read(count * 8), dtype=np.int32)

    indices = raw[0::2]
    values  = raw[1::2].view(np.float32)

    voxel_grid = np.zeros((N, N, N), dtype=np.float32)
    ix = indices // (N * N)
    iy = (indices // N) % N
    iz = indices % N
    voxel_grid[ix, iy, iz] = values

    print(f"Loaded voxel grid: {voxel_grid.shape}  voxel_size={voxel_size}")
    print(f"Max voxel value:   {voxel_grid.max():.3f}  active={count}")
    return voxel_grid, voxel_size


# ---------------------------------------------------------------------------
# Profile + metadata loading
# ---------------------------------------------------------------------------

def load_profile(bin_path):
    """
    Read profile.json from the same directory as the bin file.
    Returns (grid_center_or_None, profile_dict).
    grid_center is None when set to 'auto' in the profile.
    """
    profile_path = Path(bin_path).parent / 'profile.json'
    if not profile_path.exists():
        print(f"[Warn] No profile.json found at {profile_path} — grid_center will be auto-computed.")
        return None, {}
    with open(profile_path) as f:
        prof = json.load(f)
    center_val = prof.get('grid', {}).get('center', 'auto')
    if isinstance(center_val, list) and len(center_val) == 3:
        gc = np.array(center_val, dtype=np.float32)
        print(f"[Profile] grid_center = {gc.tolist()}")
        return gc, prof
    print("[Profile] grid_center = auto (will compute from voxel centroid)")
    return None, prof


def load_cameras(bin_path):
    """
    Read metadata.json from the same directory as the bin file.
    Returns a list of dicts with 'index' and 'position' (world-space).
    De-duplicated by camera_index — one entry per physical camera.
    """
    meta_path = Path(bin_path).parent / 'metadata.json'
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    seen, cameras = set(), []
    for entry in meta:
        idx = entry['camera_index']
        if idx not in seen:
            seen.add(idx)
            cameras.append({
                'index':    idx,
                'position': np.array(entry['camera_position'], dtype=np.float32),
                'yaw':      entry.get('yaw',   0),
                'pitch':    entry.get('pitch', 0),
            })
    print(f"[Metadata] {len(cameras)} camera(s) loaded")
    return cameras


# ---------------------------------------------------------------------------
# Voxel extraction
# ---------------------------------------------------------------------------

def extract_voxels(voxel_grid, voxel_size, grid_center, percentile=99.5):
    """
    Extract voxels above the given percentile threshold.

    Array layout matches the C++ writer:
      voxel_grid[ix, iy, iz]  →  world (X, Y, Z)
      dim 0 = ix = X
      dim 1 = iy = Y
      dim 2 = iz = Z

    World position:
      x_world = grid_min.x + (ix + 0.5) * voxel_size
      y_world = grid_min.y + (iy + 0.5) * voxel_size
      z_world = grid_min.z + (iz + 0.5) * voxel_size
    """
    N         = voxel_grid.shape[0]
    half_side = N * voxel_size * 0.5
    grid_min  = grid_center - half_side

    thresh = np.percentile(voxel_grid.ravel(), percentile)
    print(f"[Extract] percentile={percentile}  threshold={thresh:.4f}")

    coords = np.argwhere(voxel_grid > thresh)
    if coords.size == 0:
        print(f"No voxels above threshold {thresh:.4f}. Nothing to display.")
        return None, None

    intensities = voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Correct mapping: dim 0 → X, dim 1 → Y, dim 2 → Z
    x_world = grid_min[0] + (coords[:, 0] + 0.5) * voxel_size
    y_world = grid_min[1] + (coords[:, 1] + 0.5) * voxel_size
    z_world = grid_min[2] + (coords[:, 2] + 0.5) * voxel_size

    points = np.column_stack((x_world, y_world, z_world))
    print(f"[Extract] {len(points)} points  "
          f"X=[{x_world.min():.0f},{x_world.max():.0f}]  "
          f"Y=[{y_world.min():.0f},{y_world.max():.0f}]  "
          f"Z=[{z_world.min():.0f},{z_world.max():.0f}]")
    return points, intensities


# ---------------------------------------------------------------------------
# Reference geometry
# ---------------------------------------------------------------------------

def add_reference_geometry(plotter, grid_center, vox_size, N, cameras):
    """
    Add visual reference elements to the plotter (all in world space):
      - Wireframe box showing grid extent
      - World origin sphere + label
      - X (red), Y (green), Z (blue) axis lines from origin
      - Camera position spheres + coordinate labels
      - show_bounds() grid with world-space tick labels
      - Corner axes widget
    """
    half = N * vox_size / 2.0
    gx, gy, gz = grid_center

    # ── Grid extent wireframe ──────────────────────────────────────────────
    box = pv.Box(bounds=[gx-half, gx+half,
                          gy-half, gy+half,
                          gz-half, gz+half])
    plotter.add_mesh(box, style='wireframe', color='lightgray',
                     opacity=0.5, line_width=1)

    # ── World origin ───────────────────────────────────────────────────────
    plotter.add_mesh(pv.Sphere(radius=vox_size * 3, center=[0, 0, 0]),
                     color='black', opacity=0.8)
    plotter.add_point_labels(
        [[0, 0, 0]], ['origin\n(0,0,0)'],
        font_size=13, text_color='black',
        fill_shape=True, shape_opacity=0.6, margin=4,
    )

    # ── Axis lines from origin ─────────────────────────────────────────────
    axis_len = half * 0.4
    for end, color, label in [
        ([axis_len, 0,        0       ], '#cc0000', 'X+'),
        ([0,        axis_len, 0       ], '#007700', 'Y+ (fwd)'),
        ([0,        0,        axis_len], '#0000cc', 'Z+ (up)'),
    ]:
        plotter.add_mesh(pv.Line([0, 0, 0], end),
                         color=color, line_width=5)
        plotter.add_point_labels(
            [end], [label],
            font_size=15, text_color=color, bold=True,
            fill_shape=False, margin=3,
        )

    # ── Camera markers ─────────────────────────────────────────────────────
    for cam in cameras:
        pos = cam['position']
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        plotter.add_mesh(pv.Sphere(radius=vox_size * 5, center=[px, py, pz]),
                         color='cyan', opacity=0.9)
        label = f"cam{cam['index']}\n({px:.0f}, {py:.0f}, {pz:.0f})"
        plotter.add_point_labels(
            np.array([[px, py, pz]], dtype=np.float32), [label],
            font_size=13, text_color='#003399',
            fill_shape=True, shape_opacity=0.7, margin=4,
        )

    # ── Bounds grid + tick labels ──────────────────────────────────────────
    try:
        plotter.show_bounds(
            grid=True,
            location='outer',
            ticks='outside',
            n_xlabels=5, n_ylabels=5, n_zlabels=5,
            xtitle='X', ytitle='Y (fwd)', ztitle='Z (up)',
            font_size=11,
            bold=False,
            color='gray',
        )
    except Exception as e:
        print(f"[Warn] show_bounds failed: {e}")

    # ── Corner axes widget ─────────────────────────────────────────────────
    plotter.add_axes(line_width=4, labels_off=False)


# ---------------------------------------------------------------------------
# Screenshot index
# ---------------------------------------------------------------------------

def get_next_image_index(folder, prefix="voxel_", suffix=".png"):
    if not os.path.exists(folder):
        return 1
    pattern = re.compile(rf"^{prefix}(\d+){suffix}$")
    max_index = 0
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            max_index = max(max_index, int(m.group(1)))
    return max_index + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bin_path = sys.argv[1] if len(sys.argv) > 1 else "voxel_grid.bin"

    # ── Load data ──────────────────────────────────────────────────────────
    voxel_grid, vox_size = load_voxel_grid(bin_path)
    grid_center, prof    = load_profile(bin_path)
    cameras              = load_cameras(bin_path)

    N = voxel_grid.shape[0]

    # Auto-compute grid_center if not set in profile
    if grid_center is None:
        half = N * vox_size / 2.0
        grid_center = np.array([half, half, half], dtype=np.float32)
        print(f"[Auto] grid_center = {grid_center.tolist()}")

    # ── Extract voxels ─────────────────────────────────────────────────────
    percentile_to_show = 99.9
    points, intensities = extract_voxels(
        voxel_grid, vox_size, grid_center,
        percentile=percentile_to_show,
    )
    if points is None:
        return

    # ── Initial camera viewpoint ───────────────────────────────────────────
    # Look at grid_center from behind (−Y), elevated (+Z), offset to side (−X).
    # Y is the forward/depth axis, Z is up — matches the generator convention.
    half   = N * vox_size / 2.0
    dist   = half * 2.8
    gx, gy, gz = float(grid_center[0]), float(grid_center[1]), float(grid_center[2])
    cam_pos = [gx - dist*0.5, gy - dist*1.1, gz + dist*0.7]
    focal   = [gx, gy, gz]
    view_up = [0, 0, 1]   # Z is up

    # ── Off-screen screenshot ──────────────────────────────────────────────
    screenshot_folder = os.path.join(os.path.dirname(bin_path), "screenshots")
    os.makedirs(screenshot_folder, exist_ok=True)
    next_idx  = get_next_image_index(screenshot_folder)
    out_path  = os.path.join(screenshot_folder, f"voxel_{next_idx:04d}.png")

    plotter_ss = pv.Plotter(off_screen=True, window_size=[3840, 2160])
    plotter_ss.set_background("white")
    plotter_ss.enable_terrain_style()

    cloud = pv.PolyData(points)
    cloud["intensity"] = intensities
    plotter_ss.add_points(cloud, scalars="intensity", cmap="hot",
                          point_size=4.0, render_points_as_spheres=True, opacity=0.15)
    plotter_ss.add_scalar_bar(title="intensity", n_labels=5)
    add_reference_geometry(plotter_ss, grid_center, vox_size, N, cameras)
    plotter_ss.camera_position = [cam_pos, focal, view_up]
    plotter_ss.show(auto_close=False, screenshot=out_path)
    plotter_ss.close()
    print(f"[Screenshot] Saved to {out_path}")

    # ── Interactive window ─────────────────────────────────────────────────
    plotter = pv.Plotter(off_screen=False, window_size=[1920, 1080])
    plotter.set_background("white")
    plotter.enable_terrain_style()

    cloud2 = pv.PolyData(points)
    cloud2["intensity"] = intensities
    plotter.add_points(cloud2, scalars="intensity", cmap="hot",
                       point_size=4.0, render_points_as_spheres=True, opacity=0.08)
    plotter.add_scalar_bar(title="intensity", n_labels=5)
    add_reference_geometry(plotter, grid_center, vox_size, N, cameras)
    plotter.camera_position = [cam_pos, focal, view_up]

    print("[Done] Close the window to exit.")
    plotter.show(auto_close=False)


if __name__ == "__main__":
    main()