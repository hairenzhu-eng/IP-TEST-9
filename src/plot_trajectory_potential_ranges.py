"""Plot one robot trajectory and the ranges of real/predicted/continuous APF fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from plot_apf_snapshots import (
    OWN_EAST_COLUMNS,
    OWN_NORTH_COLUMNS,
    obstacle_dimensions,
    read_trajectory_samples_csv,
    track_length_axis,
)

ROOT = Path(__file__).resolve().parents[1]


def point(value):
    try:
        value = np.asarray(value, dtype=float).reshape(-1)[:2]
    except (TypeError, ValueError):
        return None
    return value if len(value) == 2 and np.isfinite(value).all() else None


def primary_csv(run_dir):
    files = [p for p in run_dir.glob("log_*.csv") if "pseudo" not in p.name]
    if not files:
        raise FileNotFoundError(f"No primary CSV found in {run_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def webots_name(run_dir):
    for path in sorted(run_dir.glob("obstacle_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = payload.get("run_context", {}).get("webots_environment")
        if name and str(name).upper() not in {"WEBOTS_UNKNOWN", "UNKNOWN"}:
            return Path(str(name)).name
    return None


def latest_runs_by_webots(logs_dir):
    latest = {}
    for run_dir in logs_dir.glob("run_*"):
        if not run_dir.is_dir() or not list(run_dir.glob("obstacle_*.json")):
            continue
        name = webots_name(run_dir)
        if name is None:
            continue
        if name not in latest or run_dir.stat().st_mtime > latest[name].stat().st_mtime:
            latest[name] = run_dir
    return latest


def add_range(ax, centre, width, height, axis, label, color, linestyle="-"):
    angle = np.degrees(np.arctan2(axis[0], axis[1]))
    ax.add_patch(Ellipse(
        (centre[1], centre[0]), width=width, height=height, angle=angle,
        fill=False, edgecolor=color, linestyle=linestyle, linewidth=1.0,
        label=label,
    ))


def field_geometry(field, settings, scale_key="risk_pc_scale"):
    centre = point(field.get("position_ne", field.get("centre_ne")))
    if centre is None:
        return None
    pc1 = float(field.get("pc1_m", settings.get("minimum_pc1_m", 0.36)))
    pc2 = float(field.get("pc2_m", settings.get("minimum_pc2_m", 0.20)))
    half_along = field.get("field_half_along_m")
    half_lateral = field.get("field_half_lateral_m")
    if not (isinstance(half_along, (int, float)) and isinstance(half_lateral, (int, float))):
        own_radius = float(settings.get("own_equivalent_radius_m", 0.3))
        scale = float(settings.get(scale_key, settings.get("avoidance_pc_scale", 2.5)))
        half_along = 0.5 * scale * pc1 + own_radius
        half_lateral = 0.5 * scale * pc2 + own_radius
    heading = field.get("heading_rad")
    if not isinstance(heading, (int, float)):
        heading = 0.0
    return centre, 2.0 * float(half_along), 2.0 * float(half_lateral), np.array([
        np.cos(float(heading)), np.sin(float(heading))
    ])


def plot(run_dir, output, target_snapshot=None):
    csv_path = primary_csv(run_dir)
    trajectory_times, trajectory = read_trajectory_samples_csv(csv_path, OWN_NORTH_COLUMNS, OWN_EAST_COLUMNS)
    snapshots = []
    for path in sorted(run_dir.glob("obstacle_*.json")):
        try:
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if len(trajectory) < 2 or not snapshots:
        raise ValueError("Trajectory or obstacle snapshots are missing")

    continuous_snapshots = snapshots
    bounds_points = [*trajectory]
    for item in snapshots:
        for group in ("clusters", "tracks", "virtual_obstacles", "webots_obstacle_truth"):
            for field in item.get(group, []):
                candidate = point(field.get("position_ne", field.get("centre_ne")))
                if candidate is not None:
                    bounds_points.append(candidate)
        fields = item.get("potential_fields", {})
        for group in ("actual", "predicted", "continuous"):
            for field in fields.get(group, []):
                candidate = point(field.get("position_ne"))
                if candidate is not None:
                    bounds_points.append(candidate)
    bounds_points = np.asarray(bounds_points, dtype=float)
    north_min, east_min = np.min(bounds_points, axis=0)
    north_max, east_max = np.max(bounds_points, axis=0)
    span = max(north_max - north_min, east_max - east_min, 1.0)
    margin = 0.10 * span
    plot_limits = (east_min - margin, east_max + margin, north_min - margin, north_max + margin)
    if target_snapshot is not None:
        end_time = float(target_snapshot.get("t", float("inf")))
        trajectory = trajectory[trajectory_times <= end_time]
        if len(trajectory) < 2:
            trajectory = trajectory[:2]
        snapshots = [target_snapshot]
        continuous_snapshots = [item for item in continuous_snapshots
                                if float(item.get("t", 0.0)) <= end_time]
    settings = snapshots[-1].get("apf_settings", {})
    fig, ax = plt.subplots(figsize=(18, 14), dpi=180)
    ax.plot(trajectory[:, 1], trajectory[:, 0], color="black", linewidth=2.2, label="Robot trajectory", zorder=8)
    ax.scatter(trajectory[0, 1], trajectory[0, 0], color="black", marker="o", s=45, label="Start", zorder=9)
    ax.scatter(trajectory[-1, 1], trajectory[-1, 0], color="black", marker="x", s=65, label="End", zorder=9)

    current = snapshots[-1]
    fields = current.get("potential_fields", {})
    actual_fields = fields.get("actual", [])
    if actual_fields:
        for field in actual_fields:
            geometry = field_geometry(field, settings)
            if geometry is not None:
                add_range(ax, *geometry, "Real potential field", "#d62728")
    else:
        # Compatibility with logs created before potential_fields was added.
        clusters = current.get("clusters", [])
        if clusters:
            geometry = field_geometry(clusters[0], settings)
            if geometry is not None:
                add_range(ax, *geometry, "Real potential field", "#d62728")

    # Prediction and continuous fields exist only when the current log has a valid TCPA.
    apf = snapshots[-1].get("apf", {})
    tcpa = apf.get("tcpa_s")
    tcpa_valid = isinstance(tcpa, (int, float)) and np.isfinite(tcpa) and tcpa > 0.0
    if tcpa_valid:
        predicted_fields = fields.get("predicted", [])
        if predicted_fields:
            geometry = field_geometry(predicted_fields[0], settings, "avoidance_pc_scale")
            if geometry is not None:
                add_range(ax, *geometry, "Predicted potential field", "#1f77b4", "--")

        # Continuous field: all bridge ellipses generated at the current time.
        for field in fields.get("continuous", []):
            geometry = field_geometry(field, settings, "avoidance_pc_scale")
            if geometry is not None:
                add_range(ax, *geometry, "Continuous potential field", "#2ca02c", ":")

    frame_text = ""
    if target_snapshot is not None:
        frame_text = f" at t={float(target_snapshot.get('t', 0.0)):.1f} s"
    ax.set_title(f"Robot trajectory with real, predicted and continuous potential-field ranges{frame_text}")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(plot_limits[0], plot_limits[1])
    ax.set_ylim(plot_limits[2], plot_limits[3])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "generated_figures" / "trajectory_potential_ranges")
    args = parser.parse_args()
    runs = {webots_name(args.run_dir): args.run_dir} if args.run_dir else latest_runs_by_webots(args.logs_dir)
    if not runs:
        raise FileNotFoundError(f"No usable Webots logs found in {args.logs_dir}")
    for name, run_dir in sorted(runs.items()):
        snapshot_paths = sorted(run_dir.glob("obstacle_*.json"))
        snapshots = []
        for path in snapshot_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload.get("t"), (int, float)):
                snapshots.append(payload)
        if not snapshots:
            continue
        times = np.asarray([float(item["t"]) for item in snapshots])
        targets = np.arange(0.0, float(times[-1]) + 1e-9, 5.0)
        frame_dir = args.output_dir / Path(name).stem
        for target in targets:
            snapshot = snapshots[int(np.argmin(np.abs(times - target)))]
            output = frame_dir / f"apf_snapshot_{float(snapshot['t']):07.1f}s.png"
            plot(run_dir, output, snapshot)
            print(output)


if __name__ == "__main__":
    main()
