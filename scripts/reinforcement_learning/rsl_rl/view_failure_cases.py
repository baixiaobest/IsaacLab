"""Local web viewer for dynamic-crowd episode replays and evaluation summaries.

Run after evaluation, for example::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/view_failure_cases.py \
        logs/rsl_rl/<experiment>/evaluations/dynamic_crowd

The server binds to localhost and opens a browser-based viewer with a selector
for timestamped evaluation runs. It uses only the Python standard library plus
NumPy, both of which are already available in the Isaac Lab runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from evaluation import SCENARIO_LABELS, SCENARIO_ORDER


INDEX_FILENAME = "failure_cases.json"
TAGS_FILENAME = "failure_case_tags.json"
WEB_APP_FILENAME = "failure_case_viewer.html"
RESULTS_FILENAME = "dynamic_crowd_results.json"
INTERACTION_RESULTS_FILENAME = "interaction_events.json"
INTERACTION_REPLAY_DIRNAME = "interaction_events"
INTERACTION_INDEX_FILENAME = "interaction_event_cases.json"
INTERACTION_PRESETS_FILENAME = "interaction_event_presets.json"
REPLAY_DIRNAME = "episode_cases"
LEGACY_REPLAY_DIRNAME = "failure_cases"


def _json_safe(value: Any) -> Any:
    """Keep the localhost API valid even when reading legacy non-standard JSON artifacts."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_atomically(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        json.dump(_json_safe(payload), file, indent=2, allow_nan=False)
        file.write("\n")
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


@dataclass(frozen=True)
class EvaluationRun:
    """Filesystem locations for one evaluation run shown by the web viewer."""

    run_id: str
    replay_dir: Path
    evaluation_dir: Path


def parse_tags(value: str) -> list[str]:
    """Return distinct, trimmed tags while preserving the user's spelling and order."""
    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in value.split(","):
        tag = raw_tag.strip()
        key = tag.casefold()
        if tag and key not in seen:
            tags.append(tag)
            seen.add(key)
    return tags


def load_case_index(replay_dir: str | Path) -> dict[str, Any]:
    path = Path(replay_dir) / INDEX_FILENAME
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    # Version 2 added richer case metadata (goal-region classification and
    # complete-success replay fields) without changing the per-case fields the
    # local viewer reads.  Accept both so local diagnostic runs and ordinary
    # evaluator artifacts are viewable through the same endpoint.
    if payload.get("schema_version") not in (1, 2) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Unsupported failure-case index: {path}")
    return payload


def load_evaluation_results(evaluation_dir: str | Path) -> dict[str, Any]:
    """Load the dynamic-crowd results exported by ``evaluate.py``."""
    path = Path(evaluation_dir) / RESULTS_FILENAME
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload.get("results"), list) or not isinstance(payload.get("aggregates"), list):
        raise ValueError(f"Unsupported dynamic-crowd results: {path}")
    return payload


def load_interaction_results(evaluation_dir: str | Path) -> dict[str, Any]:
    path = Path(evaluation_dir) / INTERACTION_RESULTS_FILENAME
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("events"), list):
        raise ValueError(f"Unsupported interaction-event results: {path}")
    return payload


def load_interaction_index(replay_dir: str | Path) -> dict[str, Any]:
    path = Path(replay_dir) / INTERACTION_REPLAY_DIRNAME / INTERACTION_INDEX_FILENAME
    if not path.is_file():
        return {"schema_version": 1, "cases": []}
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Unsupported interaction-event replay index: {path}")
    return payload


def load_interaction_presets(evaluation_dir: str | Path) -> dict[str, dict[str, float]]:
    path = Path(evaluation_dir) / INTERACTION_PRESETS_FILENAME
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    presets = payload.get("presets", {})
    if payload.get("schema_version") != 1 or not isinstance(presets, dict):
        raise ValueError(f"Unsupported interaction-event presets: {path}")
    return presets


def save_interaction_presets(evaluation_dir: str | Path, presets: dict[str, dict[str, float]]) -> None:
    _write_json_atomically(Path(evaluation_dir) / INTERACTION_PRESETS_FILENAME, {"schema_version": 1, "presets": presets})


def discover_evaluation_runs(artifact_dir: str | Path, evaluation_dir: str | Path | None = None) -> list[EvaluationRun]:
    """Discover timestamped evaluation runs while retaining legacy single-run paths."""
    artifact_path = Path(artifact_dir).resolve()
    explicit_evaluation_dir = Path(evaluation_dir).resolve() if evaluation_dir else None

    def timestamped_runs(root: Path) -> list[EvaluationRun]:
        if not root.is_dir():
            return []
        return [
            EvaluationRun(
                child.name,
                child / REPLAY_DIRNAME if (child / REPLAY_DIRNAME).is_dir() else child / LEGACY_REPLAY_DIRNAME,
                child,
            )
            for child in sorted(root.iterdir(), reverse=True)
            if child.is_dir() and (child / RESULTS_FILENAME).is_file()
        ]

    if explicit_evaluation_dir is not None:
        return [EvaluationRun(explicit_evaluation_dir.name, artifact_path, explicit_evaluation_dir)]
    # Before episode replays included successes, artifacts lived in
    # ``dynamic_crowd/failure_cases``. Keep that command working by treating either
    # replay-directory name as an alias for its timestamped-run parent.
    if artifact_path.name in (REPLAY_DIRNAME, LEGACY_REPLAY_DIRNAME):
        sibling_runs = timestamped_runs(artifact_path.parent)
        if sibling_runs:
            return sibling_runs
    if (artifact_path / INDEX_FILENAME).is_file():
        run_dir = artifact_path.parent
        siblings = timestamped_runs(run_dir.parent)
        if any(run.evaluation_dir == run_dir for run in siblings):
            return siblings
        return [EvaluationRun(run_dir.name, artifact_path, run_dir)]
    if (artifact_path / RESULTS_FILENAME).is_file():
        siblings = timestamped_runs(artifact_path.parent)
        if siblings:
            return siblings
        replay_dir = artifact_path / REPLAY_DIRNAME
        if not replay_dir.is_dir():
            replay_dir = artifact_path / LEGACY_REPLAY_DIRNAME
        return [EvaluationRun(artifact_path.name, replay_dir, artifact_path)]
    runs = timestamped_runs(artifact_path)
    if runs:
        return runs
    raise ValueError(
        f"No evaluation runs found in {artifact_path}. Expected timestamped directories containing {RESULTS_FILENAME}."
    )


def load_case_tags(replay_dir: str | Path) -> dict[str, list[str]]:
    path = Path(replay_dir) / TAGS_FILENAME
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("tags"), dict):
        raise ValueError(f"Unsupported failure-case tags: {path}")
    return {str(case_id): parse_tags(",".join(map(str, tags))) for case_id, tags in payload["tags"].items()}


def save_case_tags(replay_dir: str | Path, tags_by_case: dict[str, list[str]]) -> None:
    """Persist case tags atomically so interrupted edits do not corrupt the tag filter."""
    path = Path(replay_dir) / TAGS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "tags": tags_by_case}
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def case_tags(case: dict[str, Any], tags_by_case: dict[str, list[str]]) -> list[str]:
    """Return immutable automatic tags followed by user tags, without duplicates."""
    automatic_tags = case.get("automatic_tags", [])
    if not isinstance(automatic_tags, list):
        automatic_tags = []
    return parse_tags(",".join([*(str(tag) for tag in automatic_tags), *tags_by_case.get(case["case_id"], [])]))


def available_tags(cases: list[dict[str, Any]], tags_by_case: dict[str, list[str]]) -> list[str]:
    """Return every automatic and user-defined tag, ordered case-insensitively."""
    tags = {tag for case in cases for tag in case_tags(case, tags_by_case)}
    return sorted(tags, key=str.casefold)


def filter_cases(
    cases: list[dict[str, Any]],
    tags_by_case: dict[str, list[str]],
    scenario: str | None = None,
    pedestrian_count: int | None = None,
    tag_filter: str = "",
    exclude_tag: str | None = None,
) -> list[dict[str, Any]]:
    """Filter cases; comma-separated terms include any tag and ``exclude_tag`` removes matches."""
    wanted_tags = {tag.casefold() for tag in parse_tags(tag_filter)}
    excluded_tag = exclude_tag.casefold() if exclude_tag else None
    filtered = []
    for case in cases:
        if scenario is not None and case["scenario"] != scenario:
            continue
        if pedestrian_count is not None and case["pedestrian_count"] != pedestrian_count:
            continue
        tags = {tag.casefold() for tag in case_tags(case, tags_by_case)}
        if wanted_tags and not (wanted_tags & tags):
            continue
        if excluded_tag is not None and excluded_tag in tags:
            continue
        filtered.append(case)
    return filtered


def body_velocity_to_world(command_xy: np.ndarray, yaw: float | np.ndarray) -> np.ndarray:
    """Rotate body-frame linear velocity vectors into world XY for bird's-eye rendering."""
    command_xy = np.asarray(command_xy)
    yaw = np.asarray(yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    world_x = cos_yaw * command_xy[..., 0] - sin_yaw * command_xy[..., 1]
    world_y = sin_yaw * command_xy[..., 0] + cos_yaw * command_xy[..., 1]
    return np.stack([world_x, world_y], axis=-1)


def robot_triangle_vertices(
    position_xy: np.ndarray, yaw: float, length: float = 0.60, half_width: float = 0.225
) -> np.ndarray:
    """Return an acute, yaw-aligned triangular robot footprint in world XY."""
    position_xy = np.asarray(position_xy, dtype=float)
    local_vertices = np.array([[length * 0.58, 0.0], [-length * 0.42, half_width], [-length * 0.42, -half_width]])
    return body_velocity_to_world(local_vertices, yaw) + position_xy


class FailureCaseWebServer(ThreadingHTTPServer):
    """Serve replay data and tag edits to the local browser application."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        replay_dir: Path,
        view_radius: float,
        evaluation_dir: Path | None = None,
    ):
        self.runs = discover_evaluation_runs(replay_dir, evaluation_dir)
        self.runs_by_id = {run.run_id: run for run in self.runs}
        self.default_run_id = self.runs[0].run_id
        self.view_radius = view_radius
        self.web_app = (Path(__file__).with_name(WEB_APP_FILENAME)).read_bytes()
        super().__init__(address, FailureCaseRequestHandler)

    def _run(self, run_id: str | None) -> EvaluationRun:
        selected_run_id = run_id or self.default_run_id
        try:
            return self.runs_by_id[selected_run_id]
        except KeyError as error:
            raise KeyError(f"Unknown evaluation run: {selected_run_id}") from error

    def index_payload(self, run_id: str | None = None) -> dict[str, Any]:
        """Return one selected run plus the available timestamped run folders."""
        run = self._run(run_id)
        index = load_case_index(run.replay_dir) if (run.replay_dir / INDEX_FILENAME).is_file() else {
            "schema_version": 1,
            "cases": [],
        }
        tags_by_case = load_case_tags(run.replay_dir)
        evaluation_results: dict[str, Any] | None = None
        evaluation_error: str | None = None
        interaction_results: dict[str, Any] | None = None
        interaction_error: str | None = None
        try:
            evaluation_results = load_evaluation_results(run.evaluation_dir)
        except (OSError, ValueError) as error:
            evaluation_error = str(error)
        try:
            interaction_results = load_interaction_results(run.evaluation_dir)
        except (OSError, ValueError) as error:
            interaction_error = str(error)
        return {
            "index": index,
            "tags_by_case": tags_by_case,
            "scenario_labels": SCENARIO_LABELS,
            "scenario_order": SCENARIO_ORDER,
            "view_radius": self.view_radius,
            "evaluation": evaluation_results,
            "evaluation_error": evaluation_error,
            "interaction": interaction_results,
            "interaction_error": interaction_error,
            "interaction_index": load_interaction_index(run.replay_dir),
            "interaction_presets": load_interaction_presets(run.evaluation_dir),
            "selected_run_id": run.run_id,
            "runs": [{"id": item.run_id, "label": item.run_id} for item in self.runs],
        }

    def replay_payload(self, case_id: str, run_id: str | None = None) -> dict[str, Any]:
        run = self._run(run_id)
        index = load_case_index(run.replay_dir)
        case = next((item for item in index["cases"] if item.get("case_id") == case_id), None)
        if case is None:
            raise KeyError(f"Unknown case ID: {case_id}")
        replay_path = (run.replay_dir / case["replay_file"]).resolve()
        if not replay_path.is_relative_to(run.replay_dir) or not replay_path.is_file():
            raise ValueError(f"Invalid replay path for case {case_id}")
        with np.load(replay_path, allow_pickle=False) as replay:
            return {name: replay[name].tolist() for name in replay.files}

    def interaction_replay_payload(self, case_id: str, run_id: str | None = None) -> dict[str, Any]:
        run = self._run(run_id)
        index = load_interaction_index(run.replay_dir)
        case = next((item for item in index["cases"] if item.get("case_id") == case_id), None)
        if case is None:
            raise KeyError(f"Unknown interaction case ID: {case_id}")
        root = (run.replay_dir / INTERACTION_REPLAY_DIRNAME).resolve()
        replay_path = (root / case["replay_file"]).resolve()
        if not replay_path.is_relative_to(root) or not replay_path.is_file():
            raise ValueError(f"Invalid interaction replay path for case {case_id}")
        with np.load(replay_path, allow_pickle=False) as replay:
            return {name: replay[name].tolist() for name in replay.files}

    def save_interaction_preset(self, name: str, yield_ratio: float, assert_ratio: float,
                                run_id: str | None = None) -> dict[str, dict[str, float]]:
        if not name.strip() or not 0.0 <= yield_ratio < assert_ratio <= 2.0:
            raise ValueError("Preset requires a name and 0 <= yield ratio < assert ratio <= 2.")
        run = self._run(run_id)
        presets = load_interaction_presets(run.evaluation_dir)
        presets[name.strip()] = {"yield_speed_ratio": float(yield_ratio), "assert_speed_ratio": float(assert_ratio)}
        save_interaction_presets(run.evaluation_dir, presets)
        return presets

    def update_tags(self, case_id: str, value: str, run_id: str | None = None) -> dict[str, list[str]]:
        run = self._run(run_id)
        index = load_case_index(run.replay_dir)
        if not any(item.get("case_id") == case_id for item in index["cases"]):
            raise KeyError(f"Unknown case ID: {case_id}")
        tags_by_case = load_case_tags(run.replay_dir)
        tags_by_case[case_id] = parse_tags(value)
        save_case_tags(run.replay_dir, tags_by_case)
        return tags_by_case


class FailureCaseRequestHandler(BaseHTTPRequestHandler):
    """JSON API and static app handler.  It is intentionally localhost-only by default."""

    server: FailureCaseWebServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: HTTPStatus, content_type: str, payload: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # A browser may cancel an in-flight replay request when changing
            # occurrences or closing the tab.  There is no response left to send.
            return

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        self._send(
            status,
            "application/json; charset=utf-8",
            json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8"),
        )

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        run_id = parse_qs(parsed_url.query).get("run", [None])[0]
        try:
            if path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", self.server.web_app)
            elif path == "/api/index":
                self._send_json(HTTPStatus.OK, self.server.index_payload(run_id))
            elif path.startswith("/api/case/"):
                self._send_json(
                    HTTPStatus.OK, self.server.replay_payload(unquote(path.removeprefix("/api/case/")), run_id)
                )
            elif path.startswith("/api/interaction-case/"):
                self._send_json(
                    HTTPStatus.OK,
                    self.server.interaction_replay_payload(unquote(path.removeprefix("/api/interaction-case/")), run_id),
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except (KeyError, ValueError) as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
        except Exception as error:  # pragma: no cover - defensive response for malformed artifacts
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        run_id = parse_qs(parsed_url.query).get("run", [None])[0]
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if path.startswith("/api/tags/"):
                if not isinstance(request.get("tags"), str):
                    raise ValueError("'tags' must be a comma-separated string")
                tags = self.server.update_tags(unquote(path.removeprefix("/api/tags/")), request["tags"], run_id)
                self._send_json(HTTPStatus.OK, {"tags_by_case": tags})
            elif path == "/api/interaction-presets":
                presets = self.server.save_interaction_preset(
                    str(request.get("name", "")), float(request.get("yield_speed_ratio")),
                    float(request.get("assert_speed_ratio")), run_id,
                )
                self._send_json(HTTPStatus.OK, {"presets": presets})
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:  # pragma: no cover - disk failures are environment dependent
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the dynamic-crowd episode replay and evaluation web viewer.")
    parser.add_argument(
        "replay_dir",
        type=Path,
        help="Evaluation root containing timestamped runs, or a legacy replay-artifact directory.",
    )
    parser.add_argument(
        "--evaluation_dir",
        type=Path,
        default=None,
        help="Directory containing dynamic_crowd_results.json (defaults to the parent of replay_dir).",
    )
    parser.add_argument("--view_radius", type=float, default=5.0, help="Robot-centered bird's-eye plot radius in metres.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: localhost only).")
    parser.add_argument("--port", type=int, default=0, help="Port to bind; use 0 to choose a free port (default).")
    parser.add_argument("--no_browser", action="store_true", help="Start the server without opening a browser window.")
    args = parser.parse_args()

    with FailureCaseWebServer(
        (args.host, args.port), args.replay_dir, args.view_radius, evaluation_dir=args.evaluation_dir
    ) as server:
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}/"
        print(f"Failure-case viewer: {url}")
        print("Press Ctrl+C to stop the local viewer server.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nFailure-case viewer stopped.")


if __name__ == "__main__":
    main()
