#!/usr/bin/env python3
"""Aggregate captured ASSERT/YIELD interaction replays in a robot-frame BEV grid.

The script consumes ``interaction_event_cases.json`` and its referenced NPZ
replays.  Labels are read verbatim from the replay manifest: this module does
not run the interaction detector or classify events.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
import numpy as np


ASSERT_LABEL = "assert"
YIELD_LABEL = "yield"
SUPPORTED_LABELS = frozenset((ASSERT_LABEL, YIELD_LABEL))
DEFAULT_CSV_NAME = "crossing_behavior_grid.csv"
DEFAULT_NPZ_NAME = "crossing_behavior_grid.npz"
DEFAULT_PLOT_NAME = "crossing_behavior_heatmap.png"


@dataclass(frozen=True)
class GridResult:
    """Counts and probabilities on a ``(y, x)`` grid."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    assert_count: np.ndarray
    yield_count: np.ndarray
    total_count: np.ndarray
    p_assert: np.ndarray
    p_assert_masked: np.ndarray
    min_samples: int
    event_count_assert: int
    event_count_yield: int
    sample_count_before_radius_filter: int
    sample_count_after_radius_filter: int


def _positive_finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def _nonnegative_finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a robot-frame BEV heatmap of P(ASSERT) from already captured "
            "ASSERT/YIELD interaction replay cases."
        )
    )
    parser.add_argument(
        "input", nargs="?",
        type=Path,
        help=(
            "Evaluation run directory, interaction_events replay directory, or "
            "interaction_event_cases.json manifest."
        ),
    )
    parser.add_argument("--database", type=Path, help="ResearchAgent SQLite catalog for structured telemetry.")
    parser.add_argument("--telemetry-root", type=Path, help="Lightsail telemetry storage root.")
    parser.add_argument("--detector-run", help="Existing crossing detector run whose stored labels should be used.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <evaluation run>/crossing_behavior_analysis).",
    )
    parser.add_argument("--r-min", type=_nonnegative_finite, default=1.0, help="Minimum relative distance in m.")
    parser.add_argument("--r-max", type=_positive_finite, default=4.0, help="Maximum relative distance in m.")
    parser.add_argument(
        "--grid-resolution",
        type=_positive_finite,
        default=0.25,
        help="BEV cell size in m (default: 0.25).",
    )
    parser.add_argument(
        "--min-samples",
        type=_positive_integer,
        default=5,
        help="Mask cells containing fewer samples than this (default: 5).",
    )
    parser.add_argument(
        "--x-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional lateral BEV limits in m (default: [-r_max, r_max]).",
    )
    parser.add_argument(
        "--y-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional longitudinal BEV limits in m (default: [-r_max, r_max]).",
    )
    parser.add_argument("--dpi", type=_positive_integer, default=180, help="Heatmap image DPI (default: 180).")
    return parser.parse_args(argv)


def resolve_manifest(input_path: Path) -> tuple[Path, Path]:
    """Return ``(manifest_path, evaluation_run_dir)`` for supported inputs."""
    path = input_path.expanduser().resolve()
    candidates: list[Path]
    if path.is_file():
        candidates = [path]
    else:
        candidates = [
            path / "interaction_event_cases.json",
            path / "interaction_events" / "interaction_event_cases.json",
            path / "episode_cases" / "interaction_events" / "interaction_event_cases.json",
        ]
    manifest = next((candidate for candidate in candidates if candidate.is_file()), None)
    if manifest is None:
        checked = "\n  ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find interaction_event_cases.json. Checked:\n  {checked}")

    replay_dir = manifest.parent
    if replay_dir.name == "interaction_events" and replay_dir.parent.name == "episode_cases":
        run_dir = replay_dir.parent.parent
    else:
        run_dir = replay_dir
    return manifest, run_dir


def load_captured_cases(manifest_path: Path) -> list[dict[str, Any]]:
    """Load only captured ASSERT/YIELD cases, preserving their stored labels."""
    with manifest_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Invalid interaction replay manifest: {manifest_path}")

    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(payload["cases"]):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Manifest case {index} is not an object.")
        label = str(raw_case.get("canonical_label", "")).strip().lower()
        if label not in SUPPORTED_LABELS:
            continue
        missing = [
            key
            for key in ("pedestrian_id", "start_time_s", "end_time_s", "replay_file")
            if raw_case.get(key) is None
        ]
        if missing:
            raise ValueError(f"Captured {label.upper()} case {index} is missing: {', '.join(missing)}")
        case = dict(raw_case)
        case["canonical_label"] = label
        cases.append(case)
    if not cases:
        raise ValueError(f"No captured ASSERT/YIELD replay cases found in {manifest_path}")
    return cases


def load_telemetry_cases(database: Path, telemetry_root: Path, detector_run: str) -> list[dict[str, Any]]:
    """Load stored ASSERT/YIELD labels and their Parquet windows without detecting again."""
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError("PyArrow is required for structured telemetry input.") from error
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT e.*, d.relative_root FROM detected_events e
               JOIN telemetry_datasets d USING(dataset_id)
               WHERE e.detector_run_id = ? AND lower(e.label) IN ('assert','yield')
               ORDER BY e.episode_id, e.start_step""", (detector_run,),
        ).fetchall()
    finally:
        connection.close()
    cases: list[dict[str, Any]] = []
    datasets: dict[Path, tuple[Any, Any]] = {}
    for row in rows:
        root = (telemetry_root.expanduser().resolve() / row["relative_root"]).resolve()
        if not root.is_relative_to(telemetry_root.expanduser().resolve()):
            raise ValueError("Telemetry catalog contains an invalid relative root.")
        if root not in datasets:
            datasets[root] = (
                ds.dataset(root / "frames", format="parquet"),
                ds.dataset(root / "agents", format="parquet"),
            )
        frames_dataset, agents_dataset = datasets[root]
        condition = (
            (ds.field("episode_id") == row["episode_id"])
            & (ds.field("step") >= row["start_step"])
            & (ds.field("step") < row["end_step_exclusive"])
        )
        frames = frames_dataset.to_table(filter=condition).to_pylist()
        agents = agents_dataset.to_table(
            filter=condition & (ds.field("agent_id") == row["agent_id"])
        ).to_pylist()
        frames_by_step = {int(frame["step"]): frame for frame in frames}
        robot, yaw, pedestrian = [], [], []
        for agent in sorted(agents, key=lambda item: item["step"]):
            frame = frames_by_step.get(int(agent["step"]))
            if frame is None:
                continue
            robot.append((frame["robot_x_world"], frame["robot_y_world"]))
            yaw.append(frame["robot_yaw"])
            pedestrian.append((agent["agent_x_world"], agent["agent_y_world"]))
        if not robot:
            continue
        samples = pedestrian_in_robot_bev(np.asarray(robot), np.asarray(yaw), np.asarray(pedestrian))
        cases.append({"case_id": row["event_id"], "canonical_label": row["label"].lower(), "_samples": samples})
    if not cases:
        raise ValueError(f"Detector run {detector_run} contains no stored ASSERT/YIELD events.")
    return cases


def make_edges(lower: float, upper: float, resolution: float) -> np.ndarray:
    """Create increasing bin edges, keeping a possibly shorter final cell."""
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
        raise ValueError("Grid limits must be finite and satisfy MIN < MAX.")
    count = int(math.floor((upper - lower) / resolution))
    edges = lower + np.arange(count + 1, dtype=np.float64) * resolution
    tolerance = resolution * 1.0e-9
    if edges[-1] < upper - tolerance:
        edges = np.append(edges, upper)
    else:
        edges[-1] = upper
    return edges


def pedestrian_in_robot_bev(
    robot_position_xy: np.ndarray,
    robot_yaw: np.ndarray,
    pedestrian_position_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world positions to BEV coordinates with +X right and +Y forward."""
    robot_position = np.asarray(robot_position_xy, dtype=np.float64)
    yaw = np.asarray(robot_yaw, dtype=np.float64)
    pedestrian_position = np.asarray(pedestrian_position_xy, dtype=np.float64)
    if robot_position.ndim != 2 or robot_position.shape[1] != 2:
        raise ValueError("robot_position_xy must have shape (samples, 2).")
    if pedestrian_position.shape != robot_position.shape or yaw.shape != (len(robot_position),):
        raise ValueError("Robot yaw and pedestrian positions must have the same sample count as robot positions.")

    delta = pedestrian_position - robot_position
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    # Simulator body +X is forward and +Y is left. Rotate into body axes,
    # then map body forward to plot +Y and body right to plot +X.
    y_relative = cosine * delta[:, 0] + sine * delta[:, 1]
    x_relative = sine * delta[:, 0] - cosine * delta[:, 1]
    return x_relative, y_relative


def _case_samples(manifest_dir: Path, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "_samples" in case:
        return case["_samples"]
    replay_path = (manifest_dir / str(case["replay_file"])).resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"Replay for {case.get('case_id', '<unknown>')} does not exist: {replay_path}")

    required = {
        "time_s",
        "robot_position_xy",
        "robot_yaw",
        "pedestrian_position_xy",
        "pedestrian_active_mask",
    }
    with np.load(replay_path, allow_pickle=False) as replay:
        missing = required.difference(replay.files)
        if missing:
            raise ValueError(f"Replay {replay_path} is missing arrays: {', '.join(sorted(missing))}")
        time_s = np.asarray(replay["time_s"], dtype=np.float64)
        robot_position = np.asarray(replay["robot_position_xy"], dtype=np.float64)
        robot_yaw = np.asarray(replay["robot_yaw"], dtype=np.float64)
        pedestrians = np.asarray(replay["pedestrian_position_xy"], dtype=np.float64)
        active = np.asarray(replay["pedestrian_active_mask"], dtype=bool)

    if time_s.ndim != 1:
        raise ValueError(f"time_s in {replay_path} must be one-dimensional.")
    frame_count = len(time_s)
    if robot_position.shape != (frame_count, 2) or robot_yaw.shape != (frame_count,):
        raise ValueError(f"Robot arrays in {replay_path} have inconsistent shapes.")
    if pedestrians.ndim != 3 or pedestrians.shape[0] != frame_count or pedestrians.shape[2] != 2:
        raise ValueError(f"pedestrian_position_xy in {replay_path} must have shape (samples, pedestrians, 2).")
    if active.shape != pedestrians.shape[:2]:
        raise ValueError(f"pedestrian_active_mask in {replay_path} does not match pedestrian positions.")

    pedestrian_id = int(case["pedestrian_id"])
    if pedestrian_id < 0 or pedestrian_id >= pedestrians.shape[1]:
        raise ValueError(f"pedestrian_id {pedestrian_id} is outside replay {replay_path}.")
    start_time = float(case["start_time_s"])
    end_time = float(case["end_time_s"])
    if not (math.isfinite(start_time) and math.isfinite(end_time)) or start_time > end_time:
        raise ValueError(f"Invalid event time interval for {case.get('case_id', '<unknown>')}.")

    # The archive includes padding. Keep only frames belonging to the captured
    # event, with a small tolerance for float32 timestamps.
    step_dt = float(case.get("step_dt_s") or 0.0)
    tolerance = max(abs(step_dt) * 1.0e-4, 1.0e-6)
    event_mask = (
        (time_s >= start_time - tolerance)
        & (time_s <= end_time + tolerance)
        & active[:, pedestrian_id]
    )
    finite_mask = (
        np.isfinite(time_s)
        & np.isfinite(robot_yaw)
        & np.all(np.isfinite(robot_position), axis=1)
        & np.all(np.isfinite(pedestrians[:, pedestrian_id]), axis=1)
    )
    mask = event_mask & finite_mask
    return pedestrian_in_robot_bev(
        robot_position[mask],
        robot_yaw[mask],
        pedestrians[mask, pedestrian_id],
    )


def aggregate_cases(
    manifest_path: Path,
    cases: Iterable[dict[str, Any]],
    *,
    r_min: float,
    r_max: float,
    resolution: float,
    min_samples: int,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> GridResult:
    """Aggregate target-pedestrian samples from labelled event intervals."""
    if not (math.isfinite(r_min) and math.isfinite(r_max)) or r_min < 0.0 or r_min >= r_max:
        raise ValueError("Radii must satisfy 0 <= r_min < r_max.")
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("Grid resolution must be finite and greater than zero.")
    if min_samples < 1:
        raise ValueError("min_samples must be at least one.")

    x_limits = x_limits or (-r_max, r_max)
    y_limits = y_limits or (-r_max, r_max)
    x_edges = make_edges(float(x_limits[0]), float(x_limits[1]), resolution)
    y_edges = make_edges(float(y_limits[0]), float(y_limits[1]), resolution)
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    counts = {label: np.zeros(shape, dtype=np.int64) for label in SUPPORTED_LABELS}
    event_counts = {label: 0 for label in SUPPORTED_LABELS}
    before_radius = 0
    after_radius = 0

    for case in cases:
        label = str(case["canonical_label"]).lower()
        if label not in SUPPORTED_LABELS:
            continue
        x_relative, y_relative = _case_samples(manifest_path.parent, case)
        event_counts[label] += 1
        before_radius += len(x_relative)
        radius = np.hypot(x_relative, y_relative)
        keep = (radius >= r_min) & (radius <= r_max)
        x_kept = x_relative[keep]
        y_kept = y_relative[keep]
        after_radius += len(x_kept)
        histogram, _, _ = np.histogram2d(y_kept, x_kept, bins=(y_edges, x_edges))
        counts[label] += histogram.astype(np.int64)

    assert_count = counts[ASSERT_LABEL]
    yield_count = counts[YIELD_LABEL]
    total_count = assert_count + yield_count
    p_assert = np.full(shape, np.nan, dtype=np.float64)
    np.divide(assert_count, total_count, out=p_assert, where=total_count > 0)
    p_assert_masked = p_assert.copy()
    p_assert_masked[total_count < min_samples] = np.nan
    return GridResult(
        x_edges=x_edges,
        y_edges=y_edges,
        assert_count=assert_count,
        yield_count=yield_count,
        total_count=total_count,
        p_assert=p_assert,
        p_assert_masked=p_assert_masked,
        min_samples=min_samples,
        event_count_assert=event_counts[ASSERT_LABEL],
        event_count_yield=event_counts[YIELD_LABEL],
        sample_count_before_radius_filter=before_radius,
        sample_count_after_radius_filter=after_radius,
    )


def save_csv(path: Path, result: GridResult) -> None:
    """Write one row per BEV cell, including masked cells."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "x_index",
                "y_index",
                "x_min_m",
                "x_max_m",
                "x_center_m",
                "y_min_m",
                "y_max_m",
                "y_center_m",
                "assert_count",
                "yield_count",
                "total_count",
                "p_assert",
                "masked",
            )
        )
        for y_index in range(result.total_count.shape[0]):
            for x_index in range(result.total_count.shape[1]):
                probability = result.p_assert[y_index, x_index]
                writer.writerow(
                    (
                        x_index,
                        y_index,
                        result.x_edges[x_index],
                        result.x_edges[x_index + 1],
                        (result.x_edges[x_index] + result.x_edges[x_index + 1]) / 2.0,
                        result.y_edges[y_index],
                        result.y_edges[y_index + 1],
                        (result.y_edges[y_index] + result.y_edges[y_index + 1]) / 2.0,
                        result.assert_count[y_index, x_index],
                        result.yield_count[y_index, x_index],
                        result.total_count[y_index, x_index],
                        "" if np.isnan(probability) else probability,
                        int(result.total_count[y_index, x_index] < result.min_samples),
                    )
                )


def save_npz(
    path: Path,
    result: GridResult,
    *,
    manifest_path: Path,
    r_min: float,
    r_max: float,
    resolution: float,
) -> None:
    """Write machine-readable grid arrays and analysis configuration."""
    np.savez_compressed(
        path,
        x_edges=result.x_edges,
        y_edges=result.y_edges,
        x_centers=(result.x_edges[:-1] + result.x_edges[1:]) / 2.0,
        y_centers=(result.y_edges[:-1] + result.y_edges[1:]) / 2.0,
        assert_count=result.assert_count,
        yield_count=result.yield_count,
        total_count=result.total_count,
        p_assert=result.p_assert,
        p_assert_masked=result.p_assert_masked,
        r_min=np.asarray(r_min),
        r_max=np.asarray(r_max),
        requested_grid_resolution=np.asarray(resolution),
        min_samples=np.asarray(result.min_samples),
        event_count_assert=np.asarray(result.event_count_assert),
        event_count_yield=np.asarray(result.event_count_yield),
        sample_count_before_radius_filter=np.asarray(result.sample_count_before_radius_filter),
        sample_count_after_radius_filter=np.asarray(result.sample_count_after_radius_filter),
        source_manifest=np.asarray(str(manifest_path)),
        coordinate_convention=np.asarray("+X right, +Y forward, robot at origin"),
    )


def save_heatmap(
    path: Path,
    result: GridResult,
    *,
    r_min: float,
    r_max: float,
    dpi: int,
) -> None:
    """Plot masked P(ASSERT), with the robot at the origin facing +Y."""
    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    axis.set_facecolor("#dedede")
    probability = np.ma.masked_invalid(result.p_assert_masked)
    mesh = axis.pcolormesh(
        result.x_edges,
        result.y_edges,
        probability,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("P(ASSERT)")

    axis.add_patch(Circle((0.0, 0.0), r_max, fill=False, color="black", linewidth=0.8, linestyle="--"))
    if r_min > 0.0:
        axis.add_patch(Circle((0.0, 0.0), r_min, fill=False, color="black", linewidth=0.8, linestyle=":"))
    robot_width = min(0.42, r_max * 0.14)
    robot_length = min(0.65, r_max * 0.22)
    robot = Polygon(
        [
            (-robot_width / 2.0, -robot_length / 2.0),
            (robot_width / 2.0, -robot_length / 2.0),
            (0.0, robot_length / 2.0),
        ],
        closed=True,
        facecolor="#202020",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    axis.add_patch(robot)
    arrow_length = min(0.6, r_max * 0.15)
    axis.annotate(
        "",
        xy=(0.0, robot_length / 2.0 + arrow_length),
        xytext=(0.0, robot_length / 2.0),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.9},
    )
    axis.text(
        0.0,
        robot_length / 2.0 + arrow_length + min(0.10, r_max * 0.025),
        "forward",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axis.axhline(0.0, color="black", linewidth=0.35, alpha=0.45)
    axis.axvline(0.0, color="black", linewidth=0.35, alpha=0.45)
    axis.set_xlim(result.x_edges[0], result.x_edges[-1])
    axis.set_ylim(result.y_edges[0], result.y_edges[-1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Relative lateral position x (m; +right)")
    axis.set_ylabel("Relative longitudinal position y (m; +forward)")
    axis.set_title(
        "Captured crossing behavior: P(ASSERT)\n"
        f"{result.event_count_assert} ASSERT events, {result.event_count_yield} YIELD events; "
        f"cells with n < {result.min_samples} masked"
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.r_min >= args.r_max:
        raise SystemExit("--r-min must be smaller than --r-max.")
    telemetry_options = (args.database, args.telemetry_root, args.detector_run)
    if any(telemetry_options):
        if not all(telemetry_options) or args.input is not None:
            raise SystemExit("Use either legacy INPUT or all of --database, --telemetry-root, and --detector-run.")
        manifest_path = args.database.expanduser().resolve()
        run_dir = Path.cwd()
        cases = load_telemetry_cases(manifest_path, args.telemetry_root, args.detector_run)
    else:
        if args.input is None:
            raise SystemExit("Legacy input path is required unless structured telemetry options are supplied.")
        manifest_path, run_dir = resolve_manifest(args.input)
        cases = load_captured_cases(manifest_path)
    x_limits = tuple(args.x_limits) if args.x_limits is not None else None
    y_limits = tuple(args.y_limits) if args.y_limits is not None else None
    result = aggregate_cases(
        manifest_path,
        cases,
        r_min=args.r_min,
        r_max=args.r_max,
        resolution=args.grid_resolution,
        min_samples=args.min_samples,
        x_limits=x_limits,
        y_limits=y_limits,
    )

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "crossing_behavior_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / DEFAULT_CSV_NAME
    npz_path = output_dir / DEFAULT_NPZ_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    save_csv(csv_path, result)
    save_npz(
        npz_path,
        result,
        manifest_path=manifest_path,
        r_min=args.r_min,
        r_max=args.r_max,
        resolution=args.grid_resolution,
    )
    save_heatmap(plot_path, result, r_min=args.r_min, r_max=args.r_max, dpi=args.dpi)

    populated = int(np.count_nonzero(result.total_count))
    visible = int(np.count_nonzero(result.total_count >= result.min_samples))
    print(f"Loaded {result.event_count_assert} ASSERT and {result.event_count_yield} YIELD captured events.")
    print(
        f"Kept {result.sample_count_after_radius_filter}/{result.sample_count_before_radius_filter} "
        f"active event samples in [{args.r_min:g}, {args.r_max:g}] m."
    )
    print(f"Populated cells: {populated}; unmasked cells: {visible}.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
