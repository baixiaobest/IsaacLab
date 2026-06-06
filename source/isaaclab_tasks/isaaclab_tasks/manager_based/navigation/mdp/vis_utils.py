"""Shared viewport visualization helpers for occupancy-grid-based environments."""

from __future__ import annotations

import numpy as np


def acquire_debug_draw():
    """Return the Isaac Sim debug-draw interface, or None in headless mode."""
    try:
        from isaacsim.util.debug_draw import _debug_draw
        return _debug_draw.acquire_debug_draw_interface()
    except Exception:
        try:
            import omni.debugdraw
            return omni.debugdraw.get_debug_draw_interface()
        except Exception:
            return None


def draw_occupancy_grid_points(
    draw,
    grid_2d: np.ndarray,
    center_xy: tuple[float, float],
    grid_resolution: float = 0.1,
    z: float = 0.1,
    show_free: bool = True,
):
    """Draw an occupancy grid as coloured points in the Isaac Sim viewport.

    Occupied cells are drawn as orange points (size 8); free cells are drawn
    as dim grey points (size 4) when *show_free* is True.  Call
    ``draw.clear_points()`` before this function if you want to replace the
    previous frame rather than accumulate.

    Args:
        draw:             Debug-draw interface from :func:`acquire_debug_draw`.
        grid_2d:          2-D numpy array (H, W) with values 0 (free) / 1 (occupied).
        center_xy:        World-frame (x, y) of the grid centre (robot / sensor position).
        grid_resolution:  Metres per cell.
        z:                Height at which points are drawn.
        show_free:        Whether to also visualise free cells.
    """
    if draw is None:
        return

    grid_h, grid_w = grid_2d.shape
    cx_world, cy_world = center_xy
    half_x = grid_h * grid_resolution / 2.0
    half_y = grid_w * grid_resolution / 2.0

    # Cell-centre world coordinates — vectorised, no Python loops.
    # Row index → X axis, column index → Y axis.
    col_idx = np.arange(grid_w)
    row_idx = np.arange(grid_h)
    cols, rows = np.meshgrid(col_idx, row_idx, indexing="xy")
    px = cx_world + (rows + 0.5) * grid_resolution - half_x
    py = cy_world + (cols + 0.5) * grid_resolution - half_y

    occ_mask = grid_2d >= 0.5
    points, colors, sizes = [], [], []

    # Occupied — orange
    ox, oy = px[occ_mask].ravel(), py[occ_mask].ravel()
    if len(ox):
        z_arr = np.full(len(ox), z)
        points += list(zip(ox.tolist(), oy.tolist(), z_arr.tolist()))
        colors += [(1.0, 0.4, 0.0, 0.85)] * len(ox)
        sizes  += [8.0] * len(ox)

    # Free — dim grey
    if show_free:
        fx, fy = px[~occ_mask].ravel(), py[~occ_mask].ravel()
        if len(fx):
            z_arr = np.full(len(fx), z)
            points += list(zip(fx.tolist(), fy.tolist(), z_arr.tolist()))
            colors += [(0.4, 0.4, 0.4, 0.3)] * len(fx)
            sizes  += [4.0] * len(fx)

    if points:
        draw.draw_points(points, colors, sizes)
