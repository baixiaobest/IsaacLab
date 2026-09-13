"""Structured Parquet telemetry writer for the dynamic-crowd evaluator.

The simulation thread only snapshots completed rings and submits them to a
bounded queue. A worker converts batches to immutable Parquet shards and
uploads each closed shard before evaluation can report completion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_TARGET_SHARD_BYTES = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _PendingEpisode:
    environment_id: int
    episode_id: str
    episode_number: int
    seed: int
    profile: Any
    outcome: str
    success: bool
    collision: bool
    timeout: bool
    base_contact: bool
    goal_region_collision: bool
    frames: dict[str, np.ndarray]


class TelemetryUploadClient:
    def __init__(self):
        self.api = os.environ.get("RESEARCH_AGENT_TELEMETRY_API", "").rstrip("/")
        self.session = os.environ.get("RESEARCH_AGENT_TELEMETRY_UPLOAD_SESSION", "")
        self.token = os.environ.get("RESEARCH_AGENT_TELEMETRY_UPLOAD_TOKEN", "")
        if not self.api or not self.session or not self.token:
            raise RuntimeError("Structured telemetry upload configuration is incomplete.")

    def _request(self, method: str, suffix: str, body: bytes | None = None, headers: dict[str, str] | None = None):
        merged = {"Authorization": f"Bearer {self.token}", **(headers or {})}
        request = Request(f"{self.api}/telemetry/uploads/{self.session}{suffix}", data=body, method=method, headers=merged)
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                with urlopen(request, timeout=120) as response:  # noqa: S310 - operator-configured HTTPS origin
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as error:
                last_error = error
                if isinstance(error, HTTPError) and error.code < 500:
                    break
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"Telemetry upload failed after retries: {last_error}")

    def upload(self, path: Path, relative_path: str, sha256: str, row_count: int) -> None:
        endpoint = urlsplit(self.api)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise RuntimeError("Telemetry ingestion URL is invalid.")
        target = f"{endpoint.path}/telemetry/uploads/{self.session}/files/{relative_path}"
        last_error: Exception | None = None
        for attempt in range(6):
            connection_type = HTTPSConnection if endpoint.scheme == "https" else HTTPConnection
            connection = connection_type(endpoint.hostname, endpoint.port, timeout=120)
            try:
                connection.putrequest("PUT", target)
                connection.putheader("Authorization", f"Bearer {self.token}")
                connection.putheader("Content-Type", "application/octet-stream")
                connection.putheader("Content-Length", str(path.stat().st_size))
                connection.putheader("X-Content-SHA256", sha256)
                connection.putheader("X-Parquet-Rows", str(row_count))
                connection.endheaders()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        connection.send(chunk)
                response = connection.getresponse()
                payload = response.read().decode("utf-8", errors="replace")
                if 200 <= response.status < 300:
                    return
                error = RuntimeError(f"HTTP {response.status}: {payload[:500]}")
                if response.status < 500:
                    raise error
                last_error = error
            except (OSError, TimeoutError) as error:
                last_error = error
            finally:
                connection.close()
            time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"Telemetry upload failed after retries: {last_error}")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "")

    def finalize(self, manifest: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"manifest": manifest}, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return self._request("POST", "/finalize", body, {"Content-Type": "application/json"})


class ParquetTelemetryRecorder:
    """Export every completed replay ring to normalized Parquet tables."""

    def __init__(
        self,
        output_dir: Path,
        dataset_id: str,
        replay_recorder: Any,
        profiles: list[Any],
        env_profile_indices: list[int],
        step_dt_s: float,
        *,
        source_commit: str | None = None,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    ):
        self.output_dir = Path(output_dir)
        self.dataset_id = dataset_id
        self.replay_recorder = replay_recorder
        self.profiles = profiles
        self.env_profile_indices = list(env_profile_indices)
        self.step_dt_s = float(step_dt_s)
        self.source_commit = source_commit
        self.target_shard_bytes = max(1, int(target_shard_bytes))
        self._pending: dict[int, _PendingEpisode] = {}
        self._queue: queue.Queue[_PendingEpisode | None] = queue.Queue(maxsize=32)
        self._worker_error: BaseException | None = None
        self._uploader = TelemetryUploadClient()
        status = self._uploader.status()
        if status.get("status") != "uploading":
            raise RuntimeError("Structured telemetry upload session is not open.")
        self._files = [
            {"path": item["relative_path"], "format": item["format"], "byte_size": item["byte_size"],
             "sha256": item["sha256"], "row_count": item.get("row_count")}
            for item in status.get("files", [])
        ]
        shard_numbers = []
        for item in self._files:
            stem = Path(item["path"]).stem
            if stem.startswith("part-") and stem[5:].isdigit():
                shard_numbers.append(int(stem[5:]))
        self._shard_index = max(shard_numbers, default=-1) + 1
        # The server derives this value from checksum-verified episode shards,
        # allowing resume even when a replacement pod has no prior scratch
        # directory mounted.
        maximum_episode = status.get("max_episode_number")
        self._next_episode_number = int(maximum_episode) + 1 if maximum_episode is not None else 0
        self._episode_ids = [uuid4().hex for _ in env_profile_indices]
        self._episode_numbers = list(
            range(self._next_episode_number, self._next_episode_number + len(env_profile_indices))
        )
        self._next_episode_number += len(env_profile_indices)
        self._thread = threading.Thread(target=self._worker, name="parquet-telemetry", daemon=True)
        self._thread.start()

    @staticmethod
    def _ids(value: Any) -> set[int]:
        """Normalize terminal environment IDs from tensors and Python containers.

        Isaac Lab's reset hooks commonly provide tensors, but our evaluator's
        derived contact helpers deliberately return Python sets.  ``torch``
        cannot construct a tensor directly from a set, so normalize ordinary
        containers before using the tensor fast path.
        """
        if isinstance(value, (set, frozenset)):
            return {int(item) for item in value}
        try:
            import torch

            return set(torch.as_tensor(value).reshape(-1).detach().cpu().tolist())
        except (ImportError, RuntimeError, TypeError, ValueError):
            return {int(item) for item in value}

    def stage_terminal(
        self,
        env: Any,
        env_ids: Any,
        *,
        seed: int,
        success_ids: Any,
        collision_ids: Any,
        timeout_ids: Any,
        base_contact_ids: Any,
        goal_region_collision_ids: Any,
    ) -> None:
        success = self._ids(success_ids)
        collision = self._ids(collision_ids)
        timeout = self._ids(timeout_ids)
        base_contact = self._ids(base_contact_ids)
        goal_collision = self._ids(goal_region_collision_ids)
        for env_id in sorted(self._ids(env_ids)):
            frames = self.replay_recorder._ordered_frames(env_id)  # shared recorder owns the GPU ring
            terminal = self.replay_recorder._terminal_frame(env, env_id)
            frames = {name: np.concatenate([values, terminal[name]], axis=0) for name, values in frames.items()}
            is_collision = env_id in collision
            outcome = (
                "collision" if is_collision else "success" if env_id in success else
                "timeout" if env_id in timeout else "base_contact" if env_id in base_contact else "terminated"
            )
            self._pending[env_id] = _PendingEpisode(
                environment_id=env_id,
                episode_id=self._episode_ids[env_id],
                episode_number=self._episode_numbers[env_id],
                seed=seed,
                profile=self.profiles[self.env_profile_indices[env_id]],
                outcome=outcome,
                success=env_id in success,
                collision=is_collision,
                timeout=env_id in timeout,
                base_contact=env_id in base_contact,
                goal_region_collision=env_id in goal_collision,
                frames=frames,
            )
            self._episode_ids[env_id] = uuid4().hex
            self._episode_numbers[env_id] = self._next_episode_number
            self._next_episode_number += 1

    def resolve_terminal(self, env_ids: Any, accepted_ids: set[int]) -> None:
        self._raise_worker_error()
        for env_id in sorted(self._ids(env_ids)):
            episode = self._pending.pop(env_id)
            episode.frames["accepted_for_metrics"] = np.asarray(env_id in accepted_ids)
            self._enqueue(episode)  # bounded backpressure: data is never dropped

    def attach_terminal_rewards(self, rewards: Any, env_ids: Any) -> None:
        values = np.asarray(rewards.detach().cpu() if hasattr(rewards, "detach") else rewards).reshape(-1)
        for env_id in self._ids(env_ids):
            pending = self._pending.get(env_id)
            if pending is not None:
                pending.frames["reward"][-1] = float(values[env_id])

    def _worker(self) -> None:
        batch: list[_PendingEpisode] = []
        estimated_bytes = 0
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        if batch:
                            self._write_shard(batch)
                        return
                    batch.append(item)
                    estimated_bytes += sum(value.nbytes for value in item.frames.values())
                    if estimated_bytes >= self.target_shard_bytes:
                        self._write_shard(batch)
                        batch = []
                        estimated_bytes = 0
                finally:
                    self._queue.task_done()
        except BaseException as error:  # noqa: BLE001 - re-raised on simulation thread
            self._worker_error = error

    def _write_shard(self, episodes: list[_PendingEpisode]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schemas = {
            "episodes": pa.schema([
                ("dataset_id", pa.string()), ("episode_id", pa.string()), ("episode_number", pa.int64()),
                ("profile_episode_number", pa.int64()), ("environment_id", pa.int32()), ("scenario", pa.string()),
                ("pedestrian_count", pa.int32()), ("seed", pa.int64()), ("outcome", pa.string()),
                ("accepted_for_metrics", pa.bool_()), ("success", pa.bool_()), ("collision", pa.bool_()),
                ("timeout", pa.bool_()), ("base_contact", pa.bool_()), ("goal_region_collision", pa.bool_()),
                ("start_step", pa.int64()), ("end_step_exclusive", pa.int64()), ("step_dt_s", pa.float64()),
            ]),
            "frames": pa.schema([
                ("dataset_id", pa.string()), ("episode_id", pa.string()), ("episode_number", pa.int64()),
                ("step", pa.int64()), ("time_s", pa.float64()), ("robot_x_world", pa.float64()),
                ("robot_y_world", pa.float64()), ("robot_yaw", pa.float64()), ("robot_radius_m", pa.float64()),
                ("robot_vx_world", pa.float64()), ("robot_vy_world", pa.float64()),
                ("robot_vx_body", pa.float64()), ("robot_vy_body", pa.float64()),
                ("action_vx_body", pa.float64()), ("action_vy_body", pa.float64()),
                ("action_yaw_rate", pa.float64()), ("command_vx_body", pa.float64()),
                ("command_vy_body", pa.float64()), ("command_yaw_rate", pa.float64()),
                ("goal_x_world", pa.float64()), ("goal_y_world", pa.float64()), ("reward", pa.float64()),
                ("crowd_flow_direction", pa.float64()), ("corridor_origin_x_world", pa.float64()),
                ("corridor_origin_y_world", pa.float64()), ("corridor_length_m", pa.float64()),
                ("done", pa.bool_()), ("terminal_success", pa.bool_()), ("terminal_collision", pa.bool_()),
                ("terminal_timeout", pa.bool_()), ("terminal_base_contact", pa.bool_()),
                ("cbf_command_vx_body", pa.float64()), ("cbf_command_vy_body", pa.float64()),
                ("cbf_command_yaw_rate", pa.float64()), ("cbf_nominal_ax_body", pa.float64()),
                ("cbf_nominal_ay_body", pa.float64()), ("cbf_filtered_ax_world", pa.float64()),
                ("cbf_filtered_ay_world", pa.float64()),
            ]),
            "agents": pa.schema([
                ("dataset_id", pa.string()), ("episode_id", pa.string()), ("episode_number", pa.int64()),
                ("step", pa.int64()), ("time_s", pa.float64()), ("agent_id", pa.int32()),
                ("agent_x_world", pa.float64()), ("agent_y_world", pa.float64()),
                ("agent_vx_world", pa.float64()), ("agent_vy_world", pa.float64()),
                ("agent_radius_m", pa.float64()), ("active", pa.bool_()),
            ]),
            "contacts": pa.schema([
                ("dataset_id", pa.string()), ("episode_id", pa.string()), ("episode_number", pa.int64()),
                ("step", pa.int64()), ("time_s", pa.float64()), ("contact_type", pa.string()),
                ("robot_id", pa.string()), ("agent_id", pa.int32()),
            ]),
        }

        shard = self._shard_index
        episodes = sorted(episodes, key=lambda item: item.episode_number)
        episode_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        agent_rows: list[dict[str, Any]] = []
        contact_rows: list[dict[str, Any]] = []
        for episode in episodes:
            frames = episode.frames
            frame_count = len(frames["time_s"])
            episode_rows.append({
                "dataset_id": self.dataset_id, "episode_id": episode.episode_id,
                "episode_number": episode.episode_number, "profile_episode_number": None,
                "environment_id": episode.environment_id, "scenario": episode.profile.scenario,
                "pedestrian_count": episode.profile.pedestrian_count, "seed": episode.seed,
                "outcome": episode.outcome,
                "accepted_for_metrics": bool(frames.pop("accepted_for_metrics")),
                "success": episode.success, "collision": episode.collision, "timeout": episode.timeout,
                "base_contact": episode.base_contact, "goal_region_collision": episode.goal_region_collision,
                "start_step": 0, "end_step_exclusive": frame_count, "step_dt_s": self.step_dt_s,
            })
            for contact_type, present in (
                ("base_contact", episode.base_contact),
                ("collision", episode.collision),
                ("goal_region_collision", episode.goal_region_collision),
            ):
                if present:
                    contact_rows.append({
                        "dataset_id": self.dataset_id, "episode_id": episode.episode_id,
                        "episode_number": episode.episode_number, "step": frame_count - 1,
                        "time_s": float(frames["time_s"][-1]), "contact_type": contact_type,
                        "robot_id": "robot", "agent_id": None,
                    })
            for step in range(frame_count):
                yaw = float(frames["robot_yaw"][step])
                velocity_x = float(frames["robot_velocity_xy_world"][step, 0])
                velocity_y = float(frames["robot_velocity_xy_world"][step, 1])
                row = {
                    "dataset_id": self.dataset_id, "episode_id": episode.episode_id,
                    "episode_number": episode.episode_number, "step": step, "time_s": float(frames["time_s"][step]),
                    "robot_x_world": float(frames["robot_position_xy"][step, 0]),
                    "robot_y_world": float(frames["robot_position_xy"][step, 1]),
                    "robot_yaw": yaw,
                    "robot_radius_m": float(getattr(self.replay_recorder, "robot_radius_m", 0.3)),
                    "robot_vx_world": velocity_x, "robot_vy_world": velocity_y,
                    "robot_vx_body": math.cos(yaw) * velocity_x + math.sin(yaw) * velocity_y,
                    "robot_vy_body": -math.sin(yaw) * velocity_x + math.cos(yaw) * velocity_y,
                    "action_vx_body": float(frames["navigation_policy_velocity_body"][step, 0]),
                    "action_vy_body": float(frames["navigation_policy_velocity_body"][step, 1]),
                    "action_yaw_rate": float(frames["navigation_policy_velocity_body"][step, 2]),
                    "command_vx_body": float(frames["robot_command_velocity_body"][step, 0]),
                    "command_vy_body": float(frames["robot_command_velocity_body"][step, 1]),
                    "command_yaw_rate": float(frames["robot_command_velocity_body"][step, 2]),
                    "goal_x_world": float(frames["goal_position_xy"][step, 0]),
                    "goal_y_world": float(frames["goal_position_xy"][step, 1]),
                    "crowd_flow_direction": float(frames["crowd_flow_direction"][step]),
                    "corridor_origin_x_world": float(frames["corridor_origin_xy"][step, 0]),
                    "corridor_origin_y_world": float(frames["corridor_origin_xy"][step, 1]),
                    "corridor_length_m": float(frames["corridor_length"][step]),
                    "reward": float(frames.get("reward", np.full(frame_count, np.nan))[step]),
                    "done": step == frame_count - 1,
                    "terminal_success": bool(episode.success and step == frame_count - 1),
                    "terminal_collision": bool(episode.collision and step == frame_count - 1),
                    "terminal_timeout": bool(episode.timeout and step == frame_count - 1),
                    "terminal_base_contact": bool(episode.base_contact and step == frame_count - 1),
                }
                for name, columns in (
                    ("cbf_filtered_command_velocity_body", ("cbf_command_vx_body", "cbf_command_vy_body", "cbf_command_yaw_rate")),
                    ("cbf_nominal_acceleration_body", ("cbf_nominal_ax_body", "cbf_nominal_ay_body")),
                    ("cbf_filtered_acceleration_xy_world", ("cbf_filtered_ax_world", "cbf_filtered_ay_world")),
                ):
                    values = frames.get(name)
                    for index, column in enumerate(columns):
                        row[column] = float(values[step, index]) if values is not None else None
                frame_rows.append(row)
                active = frames["pedestrian_active_mask"][step]
                positions = frames["pedestrian_position_xy"][step]
                velocities = frames["pedestrian_velocity_xy_world"][step]
                for agent_id in np.flatnonzero(active):
                    agent_rows.append({
                        "dataset_id": self.dataset_id, "episode_id": episode.episode_id,
                        "episode_number": episode.episode_number, "step": step, "time_s": float(frames["time_s"][step]),
                        "agent_id": int(agent_id), "agent_x_world": float(positions[agent_id, 0]),
                        "agent_y_world": float(positions[agent_id, 1]),
                        "agent_vx_world": float(velocities[agent_id, 0]),
                        "agent_vy_world": float(velocities[agent_id, 1]),
                        "agent_radius_m": float(frames["pedestrian_radius"][step, agent_id]),
                        "active": True,
                    })
        for table_name, rows in (
            ("episodes", episode_rows), ("frames", frame_rows), ("agents", agent_rows), ("contacts", contact_rows)
        ):
            if not rows and table_name == "contacts":
                continue
            directory = self.output_dir / table_name
            directory.mkdir(parents=True, exist_ok=True)
            relative = f"{table_name}/part-{shard:06d}.parquet"
            path = self.output_dir / relative
            temporary = path.with_suffix(".parquet.part")
            table = pa.Table.from_pylist(rows, schema=schemas[table_name])
            pq.write_table(table, temporary, compression="zstd", row_group_size=64_000)
            os.replace(temporary, path)
            checksum = _sha256(path)
            entry = {"path": relative, "format": "parquet", "byte_size": path.stat().st_size,
                     "sha256": checksum, "row_count": table.num_rows}
            self._uploader.upload(path, relative, checksum, table.num_rows)
            self._files.append(entry)
        self._shard_index += 1

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Telemetry writer failed") from self._worker_error

    def _enqueue(self, item: _PendingEpisode | None) -> None:
        """Apply bounded backpressure while still surfacing a dead worker."""
        while True:
            self._raise_worker_error()
            try:
                self._queue.put(item, timeout=0.25)
                return
            except queue.Full:
                continue

    def close(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError(
                "Telemetry finalization refused because completed episodes have not been assigned metric acceptance."
            )
        self._enqueue(None)
        self._thread.join()
        self._raise_worker_error()
        row_counts: dict[str, int] = {}
        for item in self._files:
            table_name = item["path"].split("/", 1)[0]
            row_counts[table_name] = row_counts.get(table_name, 0) + int(item["row_count"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "experiment_id": os.environ.get("RESEARCH_EXPERIMENT_ID"),
            "evaluation_attempt_id": os.environ.get("RESEARCH_AGENT_EVALUATION_ATTEMPT_ID"),
            "source_commit": self.source_commit,
            "coordinate_conventions": {"world": "IsaacLab world XY", "yaw_radians": True,
                                       "body": {"x": "forward", "y": "left"},
                                       "ego": {"x": "lateral-right", "y": "forward"}},
            "units": {"position": "metres", "linear_velocity": "metres/second",
                      "linear_acceleration": "metres/second^2", "angle": "radians",
                      "angular_velocity": "radians/second", "reward": "environment scalar"},
            "time": {"unit": "seconds", "step_dt_s": self.step_dt_s, "intervals": "half-open"},
            "row_counts": row_counts,
            "files": sorted(self._files, key=lambda item: item["path"]),
            "assets": [],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        return self._uploader.finalize(manifest)
