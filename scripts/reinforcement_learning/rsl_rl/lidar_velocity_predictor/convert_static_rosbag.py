"""Convert static-scene CBF ROS bags to schema-v3 predictor HDF5 episodes.

Run with a ROS 2 environment that can deserialize go2_dds_ros2_bridge_msgs.
All reflected returns are labeled static; use only bags of genuinely static scenes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from src.projection import CONTRACT_VERSION, MAX_DISTANCE_M, project_history

WORLD_TOPIC = "/cbf/world_scan"
SNAPSHOT_TOPIC = "/cbf/snapshot"


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def storage_id_for(path: Path) -> str:
    if path.suffix == ".mcap" or (path.is_dir() and list(path.glob("*.mcap"))):
        return "mcap"
    return "sqlite3"


def bag_paths(path: Path) -> list[Path]:
    """Accept one ROS bag or a directory containing several bag directories."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() or (path / "metadata.yaml").is_file():
        return [path]
    bags = sorted(child for child in path.iterdir() if child.is_dir() and (child / "metadata.yaml").is_file())
    if not bags:
        raise ValueError(f"No ROS bag directories with metadata.yaml found under {path}")
    return bags


def output_paths(bags: list[Path], output: Path) -> list[Path]:
    if len(bags) == 1 and output.suffix == ".hdf5":
        return [output]
    if output.suffix == ".hdf5":
        raise ValueError("--output must be a directory when --bag contains multiple runs")
    return [output / f"{bag.name}.hdf5" for bag in bags]


def bag_messages(path: Path, storage_id: str):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError("Source ROS 2 and the bridge workspace before running this converter.") from error
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    message_types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    missing = {WORLD_TOPIC, SNAPSHOT_TOPIC} - message_types.keys()
    if missing:
        raise ValueError(f"Bag is missing required topics: {sorted(missing)}")
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic in (WORLD_TOPIC, SNAPSHOT_TOPIC):
            yield topic, deserialize_message(serialized, message_types[topic])


def scan_key(scan):
    return scan.session_id, int(scan.sequence)


def select_history(snapshot, scans: dict, frames: int):
    """Use the snapshot's embedded capture and consecutive older world scans."""
    newest = snapshot.scan
    if newest.session_id != snapshot.session_id:
        return []
    history = [newest]
    previous_time = stamp_ns(newest.header.stamp)
    for age in range(1, frames):
        sequence = (int(newest.sequence) - age) & 0xFFFFFFFF
        prior = scans.get((newest.session_id, sequence))
        if prior is None:
            break
        capture_time = stamp_ns(prior.header.stamp)
        if capture_time >= previous_time:
            break
        history.append(prior)
        previous_time = capture_time
    return history


def scan_arrays(history, frames: int):
    hits = np.zeros((frames, 256, 2), dtype=np.float64)
    states = np.zeros((frames, 256), dtype=np.uint8)
    for age, scan in enumerate(history):
        hits[age] = np.asarray(scan.capture_points_xy_m, dtype=np.float64).reshape(256, 2)
        states[age] = np.asarray(scan.ray_states, dtype=np.uint8)
    if np.any(states > 2) or not np.isfinite(hits[states > 0]).all():
        raise ValueError("invalid capture ray states or positions")
    return hits, states


def sample_from_snapshot(snapshot, history, frames: int):
    evaluation_time = stamp_ns(snapshot.header.stamp)
    capture_time = stamp_ns(snapshot.scan.header.stamp)
    if not snapshot.valid or capture_time > evaluation_time:
        return None
    xy = np.asarray(snapshot.robot_xy_m, dtype=np.float64)
    yaw = float(snapshot.robot_yaw)
    if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(yaw):
        return None
    hits, states = scan_arrays(history, frames)
    measured, winners = project_history(hits, states, xy, yaw)
    reflection = winners >= 0
    history_reflection = (measured[1] > 0.5) & (measured[0] < 1.0)
    return {
        "lidar_measured": measured,
        "point_velocity_b": np.zeros((128, 2), dtype=np.float32),
        "reflection_mask": reflection,
        "dynamic_mask": np.zeros(128, dtype=bool),
        "range_m": (measured[0, 0] * MAX_DISTANCE_M).astype(np.float32),
        "history_range_m": (measured[0] * MAX_DISTANCE_M).astype(np.float32),
        "history_reflection_mask": history_reflection,
        "history_dynamic_mask": np.zeros((frames, 128), dtype=bool),
        "capture_index": np.int64(snapshot.scan.sequence),
        "capture_time_s": np.float64(capture_time * 1e-9),
        "evaluation_time_s": np.float64(evaluation_time * 1e-9),
        "evaluation_xy": xy,
        "evaluation_yaw": np.float64(yaw),
        "ray_coverage": np.float32(np.count_nonzero(states[0]) / 256.0),
        "reflection_coverage": np.float32(np.mean(reflection)),
    }


def convert(messages, output: Path, frames: int = 8) -> dict:
    """Write one episode per session from an iterable of (topic, ROS message)."""
    if frames not in (4, 8):
        raise ValueError("frames must be 4 or 8")
    scans = {}
    snapshots = []
    for topic, message in messages:
        if topic == WORLD_TOPIC:
            scans.setdefault(scan_key(message), message)
        elif topic == SNAPSHOT_TOPIC and message.valid:
            snapshots.append(message)
    if not snapshots:
        raise ValueError("Bag contains no valid /cbf/snapshot messages")
    snapshots.sort(key=lambda item: (stamp_ns(item.header.stamp), item.session_id))
    episodes = defaultdict(lambda: defaultdict(list))
    prior_capture = {}
    skipped_history = skipped_invalid = 0
    for snapshot in snapshots:
        history = select_history(snapshot, scans, frames)
        if len(history) < frames:
            skipped_history += 1
            continue
        try:
            sample = sample_from_snapshot(snapshot, history, frames)
        except ValueError:
            sample = None
        if sample is None:
            skipped_invalid += 1
            continue
        session = snapshot.session_id
        capture = (int(snapshot.scan.sequence), stamp_ns(snapshot.scan.header.stamp))
        sample["first_after_capture"] = np.bool_(prior_capture.get(session) != capture)
        prior_capture[session] = capture
        fields = episodes[session]
        for key, value in sample.items():
            fields[key].append(value)
    if not episodes:
        raise ValueError("No snapshots have complete, valid predictor history")
    metadata = {
        "schema_version": 3, "velocity_frame": "evaluation_yaw_xy",
        "projection_contract": CONTRACT_VERSION, "num_frames": frames,
        "source": "real_static_rosbag", "data_source": "real_sensor",
        "label_assumption": "all_reflected_returns_static",
        "input_source": "measured_lidar", "sample_timing": "each_valid_cbf_snapshot",
        "audit_history_format": 1, "yaw_drift_std_rad_per_scan": 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            data = handle.create_group("data")
            data.attrs["metadata"] = json.dumps(metadata)
            for index, (session, fields) in enumerate(episodes.items()):
                group = data.create_group(f"episode_{index:07d}")
                group.attrs["session_id"] = session
                group.attrs["terrain_name"] = "real_static"
                group.attrs["terrain_level"] = -1
                group.attrs["replica_index"] = -1
                group.attrs["scenario_mode"] = -1
                for key, values in fields.items():
                    group.create_dataset(key, data=np.stack(values), compression="gzip", compression_opts=4)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    sample_count = sum(len(fields["lidar_measured"]) for fields in episodes.values())
    return {"output": str(output), "episodes": len(episodes),
            "samples": sample_count,
            "skipped_incomplete_history": skipped_history, "skipped_invalid": skipped_invalid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="One ROS bag or a parent directory of bag runs")
    parser.add_argument("--output", type=Path, required=True, help="A .hdf5 file for one bag, or an output directory")
    parser.add_argument("--num_frames", type=int, choices=(4, 8), default=8)
    parser.add_argument("--storage_id", choices=("sqlite3", "mcap"), default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip previously converted runs in a bag directory")
    args = parser.parse_args()
    bags = bag_paths(args.bag)
    targets = output_paths(bags, args.output)
    # Check every destination before converting the first run.
    for target in targets:
        if target.exists() and not args.skip_existing:
            raise FileExistsError(f"{target} already exists; use --skip_existing to convert only new runs")
    results = []
    for bag, target in zip(bags, targets, strict=True):
        if target.exists():
            results.append({"bag": str(bag), "output": str(target), "status": "skipped_existing"})
            continue
        result = convert(bag_messages(bag, args.storage_id or storage_id_for(bag)),
                         target, args.num_frames)
        results.append({"bag": str(bag), **result})
    print(json.dumps({"runs": results}, indent=2))


if __name__ == "__main__":
    main()
