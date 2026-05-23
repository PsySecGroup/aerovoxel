#!/usr/bin/env python3
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
    angle = math.pi * (time_of_day * 2 - 0.5)
    x = 50000 * math.cos(angle)
    z = 50000 * math.sin(angle)
    return (x, 0, z)


def sun_intensity(time_of_day):
    return max(0.0, math.sin(math.pi * time_of_day)) * 1.5


# ---------------------------------------------------------------------------
# Fog rendering
# ---------------------------------------------------------------------------

def render_fog(plotter, fog_density, objects, seed=99):
    """
    Render fog as a field of scattered semi-transparent spheres filling the
    object zone. A box would show its interior faces as walls when the camera
    is inside — scattered spheres give true volumetric opacity instead.
    fog_density (0-1) controls both sphere count and opacity.
    """
    if fog_density <= 0:
        return

    # Derive bounds from object positions with generous padding
    positions = [np.array(obj['start']) for obj in objects] + \
                [np.array(obj['end'])   for obj in objects]
    positions = np.array(positions)

    padding = 600
    lo = positions.min(axis=0) - padding
    hi = positions.max(axis=0) + padding

    rng      = np.random.default_rng(seed)
    n_puffs  = int(fog_density * 120)          # more spheres = denser fog
    opacity  = fog_density * 0.08              # each sphere very faint; they accumulate
    radius   = (hi - lo).mean() * 0.12        # puff size relative to scene volume

    centers = rng.uniform(lo, hi, size=(n_puffs, 3))

    for center in centers:
        puff = pv.Sphere(radius=radius, center=center.tolist(),
                         theta_resolution=6, phi_resolution=6)
        plotter.add_mesh(puff, color='white', opacity=opacity,
                         smooth_shading=False)


# ---------------------------------------------------------------------------
# Cloud rendering
# ---------------------------------------------------------------------------

def render_clouds(plotter, clouds, t):
    """
    Render each cloud as a cluster of overlapping semi-transparent spheres.
    Each cloud drifts slowly from its start to end position over the scene duration.
    A fixed seed per cloud ensures the cluster shape is stable across frames.
    """
    for cloud in clouds:
        center  = object_position(cloud, t)
        radius  = cloud.get('radius', 200)
        height  = cloud.get('height', 60)
        opacity = cloud.get('opacity', 0.35)
        n_puffs = cloud.get('puffs', 12)
        seed    = cloud.get('seed', 0)

        rng = np.random.default_rng(seed)

        offsets    = rng.uniform(-1, 1, size=(n_puffs, 3))
        offsets[:, 0] *= radius
        offsets[:, 1] *= radius
        offsets[:, 2] *= height

        puff_radii = rng.uniform(radius * 0.3, radius * 0.6, size=n_puffs)

        for offset, puff_radius in zip(offsets, puff_radii):
            puff_center = (center + offset).tolist()
            puff = pv.Sphere(radius=puff_radius, center=puff_center,
                             theta_resolution=10, phi_resolution=10)
            plotter.add_mesh(puff, color='white', opacity=opacity,
                             smooth_shading=True)



# ---------------------------------------------------------------------------
# Rain rendering
# ---------------------------------------------------------------------------

def render_rain(plotter, rain_density, objects, t, seed=77):
    """
    Render rain as falling vertical streaks (thin cylinders).
    X/Y positions are fixed per drop (same column every frame).
    Z position advances with t and wraps so drops fall continuously.
    rain_density (0-1) controls drop count and opacity.
    """
    if rain_density <= 0:
        return

    positions = [np.array(obj['start']) for obj in objects] + \
                [np.array(obj['end'])   for obj in objects]
    positions = np.array(positions)

    padding = 400
    lo = positions.min(axis=0) - padding
    hi = positions.max(axis=0) + padding

    rng       = np.random.default_rng(seed)
    n_drops   = int(rain_density * 300)
    opacity   = 0.3 + rain_density * 0.3
    length    = (hi[2] - lo[2]) * 0.06   # streak length ~6% of z range
    radius    = length * 0.04            # thin relative to length
    fall_span = hi[2] - lo[2]

    # Fixed x/y per drop, random z phase so they start spread out
    xy      = rng.uniform([lo[0], lo[1]], [hi[0], hi[1]], size=(n_drops, 2))
    z_phase = rng.uniform(0, 1, size=n_drops)

    for i in range(n_drops):
        # subtract t so drops fall downward (-z), speed=6 = 6 full passes per scene
        z = lo[2] + ((z_phase[i] - t * 6) % 1.0) * fall_span
        center = [xy[i, 0], xy[i, 1], z]
        streak = pv.Cylinder(center=center, direction=(0, 0, 1),
                             radius=radius, height=length,
                             resolution=4, capping=False)
        plotter.add_mesh(streak, color='lightblue', opacity=opacity,
                         smooth_shading=False)


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_scene(plotter, cfg, frame_objects, t):
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

    ambient = pv.Light(light_type='headlight')
    ambient.intensity = 0.3
    plotter.add_light(ambient)

    # Fog
    fog_density = env.get('fog_density', 0.0)
    if fog_density > 0:
        render_fog(plotter, fog_density, cfg['objects'])

    # Rain
    rain_density = env.get('rain_density', 0.0)
    if rain_density > 0:
        render_rain(plotter, rain_density, cfg['objects'], t)

    # Clouds
    clouds = cfg.get('clouds', [])
    if clouds:
        render_clouds(plotter, clouds, t)

    # Objects
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
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    metadata = []
    total    = n_frames * len(cameras)
    done     = 0

    for frame_idx in range(n_frames):
        t = frame_idx / max(n_frames - 1, 1)

        frame_objects = [(object_position(obj, t), obj) for obj in objects]

        for cam_idx, cam in enumerate(cameras):
            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.enable_anti_aliasing('ssaa')

            build_scene(plotter, cfg, frame_objects, t)
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