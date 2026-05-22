#!/usr/bin/env python3
"""
scene_generator.py

Generates synthetic multi-camera frames from a YAML scene config and writes
a metadata.json compatible with ray_voxel.

Usage:
    python src/scene_generator.py scenes/sunny_day.yaml

Output:
    output/<scene_name>/image_CAM_frame_FRAME.png
    output/<scene_name>/metadata.json
"""

import sys
import json
import math
import yaml
import numpy as np
import pyvista as pv
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def yaw_pitch_to_direction(yaw_deg, pitch_deg):
    """Convert yaw/pitch angles to a unit direction vector."""
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    dx = math.cos(pitch) * math.sin(yaw)
    dy = math.cos(pitch) * math.cos(yaw)
    dz = math.sin(pitch)
    return np.array([dx, dy, dz])


def object_position(obj, t):
    """Linear interpolation of object position at normalized time t (0-1)."""
    start = np.array(obj['start'], dtype=float)
    end   = np.array(obj['end'],   dtype=float)
    return start + (end - start) * t


def sky_color(time_of_day):
    """
    Map time_of_day (0-1) to an RGB tuple (0-1 range) using keyframe interpolation.

    Keyframes:
        0.00 = midnight      ( 10,  10,  30)
        0.20 = pre-dawn      ( 20,  20,  60)
        0.28 = dawn          (255, 120,  50)
        0.35 = morning       (180, 210, 240)
        0.50 = noon          (100, 180, 255)
        0.70 = afternoon     ( 90, 170, 245)
        0.78 = golden hour   (255, 160,  60)
        0.85 = dusk          (200,  80,  40)
        0.92 = twilight      ( 40,  30,  80)
        1.00 = midnight      ( 10,  10,  30)
    """
    keyframes = [
        (0.00, np.array([ 10,  10,  30])),
        (0.20, np.array([ 20,  20,  60])),
        (0.28, np.array([255, 120,  50])),
        (0.35, np.array([180, 210, 240])),
        (0.50, np.array([100, 180, 255])),
        (0.70, np.array([ 90, 170, 245])),
        (0.78, np.array([255, 160,  60])),
        (0.85, np.array([200,  80,  40])),
        (0.92, np.array([ 40,  30,  80])),
        (1.00, np.array([ 10,  10,  30])),
    ]

    t = time_of_day
    for i in range(len(keyframes) - 1):
        t0, c0 = keyframes[i]
        t1, c1 = keyframes[i + 1]
        if t <= t1:
            alpha = (t - t0) / (t1 - t0)
            rgb = c0 * (1 - alpha) + c1 * alpha
            return tuple(rgb / 255.0)

    return tuple(keyframes[-1][1] / 255.0)


def sun_position(time_of_day):
    """Arc the sun across the sky. Returns (x, y, z) for the light source."""
    angle = math.pi * (time_of_day * 2 - 0.5)
    x = 50000 * math.cos(angle)
    z = 50000 * math.sin(angle)
    return (x, 0, z)


def sun_intensity(time_of_day):
    """Bright at noon, dim at dawn/dusk, off at night."""
    return max(0.0, math.sin(math.pi * time_of_day)) * 1.5


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_scene(plotter, cfg, frame_objects):
    """
    Populate a plotter with environment and objects for a given frame.
    frame_objects: list of (position, obj_config) tuples.
    """
    env = cfg['environment']
    tod = env.get('time_of_day', 0.5)

    plotter.set_background(sky_color(tod))

    plotter.remove_all_lights()
    sun_pos   = sun_position(tod)
    intensity = sun_intensity(tod)
    if intensity > 0:
        sun = pv.Light(position=sun_pos, focal_point=(0, 0, 0),
        light_type='scene light')
        sun.intensity = intensity
        plotter.add_light(sun)

    # Ambient fill so shadowed areas aren't pitch black
    ambient = pv.Light(light_type='headlight')
    ambient.intensity = 0.3
    plotter.add_light(ambient)

    for pos, obj in frame_objects:
        radius = obj.get('radius', 8)
        color  = obj.get('color', [1.0, 1.0, 1.0])
        mesh   = pv.Sphere(radius=radius, center=pos.tolist(),
            theta_resolution=16, phi_resolution=16)
        plotter.add_mesh(mesh, color=color, smooth_shading=True)


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def apply_camera(plotter, cam):
    pos       = cam['position']
    yaw       = cam.get('yaw',   0)
    pitch     = cam.get('pitch', 0)
    fov       = cam.get('fov',   60)
    direction = yaw_pitch_to_direction(yaw, pitch)
    focal     = [pos[i] + direction[i] * 1000 for i in range(3)]

    plotter.camera.position    = pos
    plotter.camera.focal_point = focal
    plotter.camera.view_angle  = fov

    if abs(pitch) > 85:
        plotter.camera.up = (0, 1, 0)
    else:
        plotter.camera.up = (0, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path, output_dir=None):
    cfg      = load_config(config_path)
    scene    = cfg['scene']
    cameras  = cfg['cameras']
    objects  = cfg['objects']

    fps      = scene['fps']
    duration = scene['duration']
    n_frames = int(fps * duration)

    out_dir  = Path(output_dir) if output_dir else Path('output') / scene['name']
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    total    = n_frames * len(cameras)
    done     = 0
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    for frame_idx in range(n_frames):
        t = frame_idx / max(n_frames - 1, 1)

        frame_objects = [(object_position(obj, t), obj) for obj in objects]

        for cam_idx, cam in enumerate(cameras):
            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.enable_anti_aliasing('ssaa')

            build_scene(plotter, cfg, frame_objects)
            apply_camera(plotter, cam)

            img_name = f"camera_{cam_idx:03d}_frame_{frame_idx:03d}.png"
            img_path = out_dir / "frames" / img_name

            plotter.show(auto_close=False)
            plotter.screenshot(str(img_path))
            plotter.close()

            metadata.append({
                "camera_index":    cam_idx,
                "frame_index":     frame_idx,
                "camera_position": cam['position'],
                "yaw":             cam.get('yaw',   0),
                "pitch":           cam.get('pitch', 0),
                "roll":            cam.get('roll',  0),
                "fov_degrees":     cam.get('fov',   60),
                "image_file":      img_name,
                "rendered":        True
            })

            done += 1
            print(f"[{done}/{total}] cam {cam_idx:03d} frame {frame_idx:03d} -> {img_name}")

    meta_path = out_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Done] {done} frames -> {out_dir}")
    print(f"[Done] metadata -> {meta_path}")
    print(f"\nTo build voxel grid:")
    print(f"  ./ray_voxel {meta_path} {out_dir} voxel_grid.bin")

def generate(example_dir):
    main(
        config_path = Path(example_dir).resolve() / 'scene.yml',
        output_dir  = Path(example_dir).resolve(),
    )


if __name__ == '__main__':
    config = sys.argv[1] if len(sys.argv) > 1 else 'scenes/sunny_day.yaml'
    main(config)