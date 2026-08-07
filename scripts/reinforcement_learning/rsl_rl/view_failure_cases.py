"""Interactive desktop viewer for dynamic-crowd collision replay artifacts.

Run after evaluation, for example::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/view_failure_cases.py \
        logs/rsl_rl/<experiment>/evaluations/dynamic_crowd/failure_cases
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import SCENARIO_LABELS, SCENARIO_ORDER


INDEX_FILENAME = "failure_cases.json"
TAGS_FILENAME = "failure_case_tags.json"


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
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Unsupported failure-case index: {path}")
    return payload


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


def filter_cases(
    cases: list[dict[str, Any]],
    tags_by_case: dict[str, list[str]],
    scenario: str | None = None,
    pedestrian_count: int | None = None,
    tag_filter: str = "",
) -> list[dict[str, Any]]:
    """Filter cases; comma-separated tag terms match any tag case-insensitively."""
    wanted_tags = {tag.casefold() for tag in parse_tags(tag_filter)}
    filtered = []
    for case in cases:
        if scenario is not None and case["scenario"] != scenario:
            continue
        if pedestrian_count is not None and case["pedestrian_count"] != pedestrian_count:
            continue
        case_tags = {tag.casefold() for tag in tags_by_case.get(case["case_id"], [])}
        if wanted_tags and not (wanted_tags & case_tags):
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
    # The rear corners sit behind the centre.  This keeps all three interior
    # angles acute while giving the front vertex a clear heading direction.
    local_vertices = np.array(
        [
            [length * 0.58, 0.0],
            [-length * 0.42, half_width],
            [-length * 0.42, -half_width],
        ]
    )
    return body_velocity_to_world(local_vertices, yaw) + position_xy


class FailureCaseViewer:
    """Matplotlib controls and plot state for a directory of collision replays."""

    def __init__(self, replay_dir: str | Path, view_radius: float = 5.0):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

        self.replay_dir = Path(replay_dir)
        self.index = load_case_index(self.replay_dir)
        self.cases = self.index["cases"]
        self.tags_by_case = load_case_tags(self.replay_dir)
        self.view_radius = float(view_radius)
        self.selected_scenario: str | None = None
        self.selected_pedestrian_count: int | None = None
        self.tag_filter = ""
        self.matches: list[dict[str, Any]] = []
        self.case: dict[str, Any] | None = None
        self.frames: dict[str, np.ndarray] | None = None

        self.figure, self.axis = plt.subplots(figsize=(11, 8))
        self._polygon_type = Polygon
        self.figure.subplots_adjust(left=0.08, right=0.72, bottom=0.28, top=0.92)
        self.figure.canvas.manager.set_window_title("Dynamic-crowd collision analysis")

        self.time_slider = Slider(self.figure.add_axes((0.08, 0.17, 0.64, 0.03)), "Time", 0, 1, valinit=0, valstep=1)
        self.previous_occurrence_button = Button(self.figure.add_axes((0.08, 0.09, 0.05, 0.05)), "◀")
        self.occurrence_slider = Slider(
            self.figure.add_axes((0.15, 0.10, 0.50, 0.03)), "Occurrence", 1, 1, valinit=1, valstep=1
        )
        self.next_occurrence_button = Button(self.figure.add_axes((0.67, 0.09, 0.05, 0.05)), "▶")
        self.scenario_buttons = RadioButtons(
            self.figure.add_axes((0.76, 0.65, 0.20, 0.20)),
            ["All", *(SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER)],
            active=0,
        )
        self.count_box = TextBox(
            self.figure.add_axes((0.76, 0.58, 0.20, 0.04)), "Crowd count\n(blank = all)", initial=""
        )
        self.tag_filter_box = TextBox(
            self.figure.add_axes((0.76, 0.48, 0.20, 0.04)), "Tag filter\n(any comma tag)", initial=""
        )
        self.tag_editor_box = TextBox(self.figure.add_axes((0.76, 0.35, 0.20, 0.04)), "Tags for case", initial="")
        self.save_tags_button = Button(self.figure.add_axes((0.76, 0.29, 0.20, 0.04)), "Save tags")
        self.toggles = CheckButtons(
            self.figure.add_axes((0.76, 0.08, 0.20, 0.17)),
            ["Pedestrian velocity", "Robot actual velocity", "Robot command", "Trails"],
            [True, True, True, True],
        )
        self.status_text = self.figure.text(0.76, 0.88, "", va="top", wrap=True)

        self.time_slider.on_changed(lambda _: self.draw())
        self.occurrence_slider.on_changed(self._select_occurrence)
        self.previous_occurrence_button.on_clicked(lambda _: self._cycle_occurrence(-1))
        self.next_occurrence_button.on_clicked(lambda _: self._cycle_occurrence(1))
        self.scenario_buttons.on_clicked(self._set_scenario)
        self.count_box.on_submit(self._set_count)
        self.tag_filter_box.on_submit(self._set_tag_filter)
        self.save_tags_button.on_clicked(self._save_tags)
        self.toggles.on_clicked(lambda _: self.draw())
        self._update_matches()

    def _set_scenario(self, label: str) -> None:
        lookup = {SCENARIO_LABELS[scenario]: scenario for scenario in SCENARIO_ORDER}
        self.selected_scenario = lookup.get(label)
        self._update_matches()

    def _set_count(self, value: str) -> None:
        value = value.strip()
        try:
            self.selected_pedestrian_count = int(value) if value else None
        except ValueError:
            self.selected_pedestrian_count = None
            self.count_box.set_val("")
        self._update_matches()

    def _set_tag_filter(self, value: str) -> None:
        self.tag_filter = value
        self._update_matches()

    def _update_matches(self, selected_case_id: str | None = None) -> None:
        self.matches = filter_cases(
            self.cases,
            self.tags_by_case,
            scenario=self.selected_scenario,
            pedestrian_count=self.selected_pedestrian_count,
            tag_filter=self.tag_filter,
        )
        maximum = max(1, len(self.matches))
        self.occurrence_slider.valmax = maximum
        self.occurrence_slider.ax.set_xlim(1, maximum)
        selected_occurrence = 1
        if selected_case_id is not None:
            for index, case in enumerate(self.matches, start=1):
                if case["case_id"] == selected_case_id:
                    selected_occurrence = index
                    break
        self.occurrence_slider.set_val(selected_occurrence)
        self._load_selected_case()

    def _select_occurrence(self, _: float) -> None:
        self._load_selected_case()

    def _cycle_occurrence(self, direction: int) -> None:
        """Select the previous or next filtered case, wrapping at either end."""
        if not self.matches:
            return
        current = int(round(self.occurrence_slider.val)) - 1
        self.occurrence_slider.set_val((current + direction) % len(self.matches) + 1)

    def _load_selected_case(self) -> None:
        if not self.matches:
            self.case = None
            self.frames = None
            self.draw()
            return
        index = min(max(int(round(self.occurrence_slider.val)) - 1, 0), len(self.matches) - 1)
        self.case = self.matches[index]
        with np.load(self.replay_dir / self.case["replay_file"], allow_pickle=False) as replay:
            self.frames = {name: replay[name] for name in replay.files}
        frame_count = len(self.frames["time_s"])
        self.time_slider.valmax = max(0, frame_count - 1)
        self.time_slider.ax.set_xlim(0, max(1, frame_count - 1))
        self.time_slider.set_val(frame_count - 1)
        self.tag_editor_box.set_val(", ".join(self.tags_by_case.get(self.case["case_id"], [])))
        self.draw()

    def _save_tags(self, _event: Any) -> None:
        if self.case is None:
            return
        selected_case_id = self.case["case_id"]
        self.tags_by_case[selected_case_id] = parse_tags(self.tag_editor_box.text)
        save_case_tags(self.replay_dir, self.tags_by_case)
        self._update_matches(selected_case_id=selected_case_id)

    def _toggle_enabled(self, label: str) -> bool:
        labels = [text.get_text() for text in self.toggles.labels]
        return self.toggles.get_status()[labels.index(label)]

    def draw(self) -> None:
        self.axis.clear()
        if self.case is None or self.frames is None:
            self.axis.set_axis_off()
            self.status_text.set_text("No collision cases match the selected filters.")
            self.figure.canvas.draw_idle()
            return

        self.axis.set_axis_on()
        frame_index = min(int(round(self.time_slider.val)), len(self.frames["time_s"]) - 1)
        robot_position = self.frames["robot_position_xy"][frame_index]
        robot_yaw = float(self.frames["robot_yaw"][frame_index])
        pedestrian_position = self.frames["pedestrian_position_xy"][frame_index]
        pedestrian_velocity = self.frames["pedestrian_velocity_xy_world"][frame_index]
        active = self.frames["pedestrian_active_mask"][frame_index]
        collider_ids = set(self.case["colliding_agent_ids"])

        robot_shape = self._polygon_type(
            robot_triangle_vertices(robot_position, robot_yaw),
            closed=True,
            facecolor="tab:blue",
            edgecolor="navy",
            linewidth=1.5,
            label="Robot (heading)",
            zorder=5,
        )
        self.axis.add_patch(robot_shape)
        self.axis.scatter(
            *self.frames["goal_position_xy"][frame_index], marker="*", s=145, color="tab:green", label="Goal", zorder=3
        )
        self.axis.scatter(
            pedestrian_position[active, 0],
            pedestrian_position[active, 1],
            s=42,
            color="tab:orange",
            label="Pedestrian",
            zorder=3,
        )
        for pedestrian_id in collider_ids:
            if active[pedestrian_id]:
                self.axis.scatter(
                    pedestrian_position[pedestrian_id, 0], pedestrian_position[pedestrian_id, 1],
                    s=140, facecolors="none", edgecolors="red", linewidths=2.5, zorder=6, label="Collider"
                )

        if self._toggle_enabled("Pedestrian velocity") and np.any(active):
            self.axis.quiver(
                pedestrian_position[active, 0],
                pedestrian_position[active, 1],
                pedestrian_velocity[active, 0],
                pedestrian_velocity[active, 1],
                color="tab:orange",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.004,
                zorder=2,
            )
        robot_velocity = self.frames["robot_velocity_xy_world"][frame_index]
        if self._toggle_enabled("Robot actual velocity"):
            self.axis.quiver(
                robot_position[0], robot_position[1], robot_velocity[0], robot_velocity[1],
                color="tab:blue",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.007,
                zorder=6,
                label="Actual velocity",
            )
        command = self.frames["robot_command_velocity_body"][frame_index]
        command_world = body_velocity_to_world(command[:2], robot_yaw)
        if self._toggle_enabled("Robot command"):
            self.axis.quiver(
                robot_position[0], robot_position[1], command_world[0], command_world[1],
                color="tab:purple",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.007,
                zorder=6,
                label="Command velocity",
            )
        if self._toggle_enabled("Trails"):
            steps = max(1, int(round(1.0 / self.case["step_dt_s"])))
            start = max(0, frame_index - steps)
            robot_trail = self.frames["robot_position_xy"][start : frame_index + 1]
            self.axis.plot(robot_trail[:, 0], robot_trail[:, 1], color="tab:blue", alpha=0.55, linewidth=2)
            trail_positions = self.frames["pedestrian_position_xy"][start : frame_index + 1]
            trail_active = self.frames["pedestrian_active_mask"][start : frame_index + 1]
            for pedestrian_id in np.flatnonzero(np.any(trail_active, axis=0)):
                positions = trail_positions[:, pedestrian_id]
                mask = trail_active[:, pedestrian_id]
                self.axis.plot(positions[mask, 0], positions[mask, 1], color="tab:orange", alpha=0.30, linewidth=1)

        self.axis.set_aspect("equal", adjustable="box")
        self.axis.set_xlim(robot_position[0] - self.view_radius, robot_position[0] + self.view_radius)
        self.axis.set_ylim(robot_position[1] - self.view_radius, robot_position[1] + self.view_radius)
        self.axis.set_xlabel("World X (m)")
        self.axis.set_ylabel("World Y (m)")
        relative_time = float(self.frames["time_s"][frame_index] - self.frames["time_s"][-1])
        self.axis.set_title(
            f"{SCENARIO_LABELS[self.case['scenario']]} | {self.case['case_id']} | "
            f"{relative_time:+.2f} s to collision"
        )
        handles, labels = self.axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.axis.legend(unique.values(), unique.keys(), loc="upper left")
        self.axis.grid(True, alpha=0.25)
        self.status_text.set_text(
            f"{len(self.matches)} matching case(s)\n"
            f"crowd: {self.case['pedestrian_count']}\n"
            f"collider slots: {self.case['colliding_agent_ids']}\n"
            f"body command: ({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f} rad/s)"
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dynamic-crowd pedestrian collision replays.")
    parser.add_argument("replay_dir", type=Path, help="Directory containing failure_cases.json and cases/.")
    parser.add_argument(
        "--view_radius", type=float, default=5.0, help="Robot-centered bird's-eye plot radius in metres."
    )
    args = parser.parse_args()
    FailureCaseViewer(args.replay_dir, args.view_radius).show()


if __name__ == "__main__":
    main()
