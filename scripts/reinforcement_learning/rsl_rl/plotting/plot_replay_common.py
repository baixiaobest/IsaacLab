#!/usr/bin/env python3
"""Shared top-view drawing primitives for saved robot and pedestrian replays."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon

ROBOT_CMAP = LinearSegmentedColormap.from_list(
    "robot_speed", ["#B2182B", "#FDDC6C", "#1B9E77"]
)

PEDESTRIAN_COLOR = "#9AA8B7"

HIGHLIGHT_COLOR = "#2B6CB0"

NAVIGATION_VECTOR_COLOR = "#6A1B9A"

CBF_VECTOR_COLOR = "#00796B"

RESPAWN_JUMP_M = 0.75

EVALUATION_MEAN_PEDESTRIAN_RADIUS_M = 0.25793

GOAL_COLOR = "#6B7280"

def _nearest_index(time_s: np.ndarray, requested_time_s: float) -> int:
    return int(np.abs(time_s - requested_time_s).argmin())

def _line_segments(points: np.ndarray) -> np.ndarray:
    """Return consecutive XY line segments suitable for LineCollection."""
    return np.stack((points[:-1], points[1:]), axis=1)

def _continuous_past_track(positions: np.ndarray, active: np.ndarray, frame: int) -> np.ndarray:
    """Keep only an agent's latest continuous history through ``frame``.

    Pedestrians respawn on the far side of the corridor after leaving the
    scene. The resulting position jump is not a physical trajectory, so the
    track is cut at the last inactive gap or displacement larger than
    ``RESPAWN_JUMP_M``.
    """
    history = positions[: frame + 1].astype(float).copy()
    visible = active[: frame + 1]
    history[~visible] = np.nan
    if not visible[-1]:
        return history * np.nan
    discontinuities = (~visible[:-1]) | (~visible[1:])
    displacements = np.linalg.norm(np.diff(positions[: frame + 1], axis=0), axis=1)
    discontinuities |= displacements > RESPAWN_JUMP_M
    start = int(np.flatnonzero(discontinuities)[-1] + 2) if discontinuities.any() else 0
    history[:start] = np.nan
    return history

def _plot_pedestrian_trajectories(
    ax: plt.Axes,
    positions: np.ndarray,
    active: np.ndarray,
    *,
    frame: int,
    highlight_pedestrian: int | None = None,
    highlighted_pedestrians: dict[int, str] | None = None,
    alpha: float = 0.38,
    linewidth: float = 1.0,
) -> None:
    highlighted_pedestrians = dict(highlighted_pedestrians or {})
    if highlight_pedestrian is not None:
        highlighted_pedestrians.setdefault(highlight_pedestrian, HIGHLIGHT_COLOR)
    for pedestrian_id in range(positions.shape[1]):
        track = _continuous_past_track(positions[:, pedestrian_id], active[:, pedestrian_id], frame)
        color = highlighted_pedestrians.get(pedestrian_id, PEDESTRIAN_COLOR)
        highlighted = pedestrian_id in highlighted_pedestrians
        width = 2.4 if highlighted else linewidth
        opacity = 0.92 if highlighted else alpha
        ax.plot(track[:, 0], track[:, 1], color=color, linewidth=width, alpha=opacity, zorder=1)

def _plot_robot_speed_trajectory(
    ax: plt.Axes,
    position_xy: np.ndarray,
    speed_mps: np.ndarray,
    norm: Normalize,
    *,
    linewidth: float = 4.0,
    alpha: float = 1.0,
) -> LineCollection:
    collection = LineCollection(
        _line_segments(position_xy),
        array=speed_mps[:-1],
        cmap=ROBOT_CMAP,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
        capstyle="round",
        zorder=4,
    )
    ax.add_collection(collection)
    return collection

def _robot_marker(ax: plt.Axes, position: np.ndarray, yaw: float) -> None:
    """Draw a heading-aligned acute dart with a concave tail notch."""
    # Local +x is the robot heading.  The tail notch is deliberately inset so
    # the silhouette remains directional even when command arrows overlap it.
    local_vertices = np.array(((0.34, 0.0), (-0.20, 0.22), (-0.07, 0.0), (-0.20, -0.22)))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    vertices = local_vertices @ rotation.T + position
    ax.add_patch(Polygon(
        vertices, closed=True, facecolor="white", edgecolor="#202124", linewidth=1.8,
        joinstyle="round", zorder=8,
    ))

def _navigation_goal_marker(ax: plt.Axes, position: np.ndarray, yaw: float) -> None:
    """Draw the position-and-heading navigation target as a dashed gray dart."""
    local_vertices = np.array(((0.36, 0.0), (-0.21, 0.23), (-0.075, 0.0), (-0.21, -0.23)))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    vertices = local_vertices @ rotation.T + position
    ax.add_patch(Polygon(
        vertices, closed=True, facecolor="none", edgecolor=GOAL_COLOR, linewidth=1.9,
        linestyle=(0, (3.0, 2.0)), joinstyle="round", zorder=7,
    ))

def _body_velocity_to_world(velocity_body: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a planar body-frame velocity into the plotted world frame."""
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array((
        cosine * velocity_body[0] - sine * velocity_body[1],
        sine * velocity_body[0] + cosine * velocity_body[1],
    ))

def _velocity_arrow(ax: plt.Axes, origin: np.ndarray, velocity_world: np.ndarray, color: str) -> None:
    """Draw a velocity vector in metres-per-second with a readable plot scale."""
    scale = 0.72
    ax.arrow(
        origin[0], origin[1], scale * velocity_world[0], scale * velocity_world[1],
        width=0.028, head_width=0.16, head_length=0.13, color=color,
        length_includes_head=True, zorder=9,
    )

