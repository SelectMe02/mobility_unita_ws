"""Obstacle points on open roads (GPS route), separate from the tunnel FTG.

The tunnel filter fits one road plane and keeps every return when that fit
fails, which on slopes and bends turns the road surface ahead into
obstacles. Here the ground is local: the lowest return within a small
neighbourhood of grid cells. A return counts as an obstacle when it rises
min_height_m to max_height_m above that local ground, so kerbs, slopes and
road camber drop out while cars, pedestrians and boxes stay. No ROS.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoadPointConfig:
    lidar_to_rear_x_m: float = 1.4
    cell_m: float = 1.5               # ground grid; neighbourhood is 3x3 cells
    x_range_m: tuple = (-10., 60.)    # LiDAR frame
    half_width_m: float = 12.
    min_height_m: float = .3          # above local ground: kerbs are lower
    max_height_m: float = 2.5         # above local ground: signs and canopies are higher
    max_ground_below_lidar_m: float = 2.5
    front_extent_m: float = 3.845     # Ego body in rear-axle coordinates
    rear_extent_m: float = .79
    half_width_body_m: float = .946
    body_margin_m: float = .15


def obstacle_points(points, config=RoadPointConfig()):
    """Return the LiDAR-frame returns that stand on the road as obstacles."""
    cfg = config
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    p = p[(p[:, 0] >= cfg.x_range_m[0]) & (p[:, 0] < cfg.x_range_m[1])
          & (np.abs(p[:, 1]) < cfg.half_width_m)
          & (p[:, 2] > -cfg.max_ground_below_lidar_m)]
    rear_x = p[:, 0] + cfg.lidar_to_rear_x_m
    body = ((rear_x > -cfg.rear_extent_m - cfg.body_margin_m)
            & (rear_x < cfg.front_extent_m + cfg.body_margin_m)
            & (np.abs(p[:, 1]) < cfg.half_width_body_m + cfg.body_margin_m))
    p = p[~body]
    if not len(p):
        return p
    nx = int(np.ceil((cfg.x_range_m[1] - cfg.x_range_m[0]) / cfg.cell_m))
    ny = int(np.ceil(2. * cfg.half_width_m / cfg.cell_m))
    ix = np.clip(((p[:, 0] - cfg.x_range_m[0]) / cfg.cell_m).astype(int), 0, nx - 1)
    iy = np.clip(((p[:, 1] + cfg.half_width_m) / cfg.cell_m).astype(int), 0, ny - 1)
    lowest = np.full((nx + 2, ny + 2), np.inf)
    np.minimum.at(lowest, (ix + 1, iy + 1), p[:, 2])
    # Lowest return of each cell and its eight neighbours: ground under a car
    # is hidden, but the road beside it is not.
    ground = lowest[1:-1, 1:-1].copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            ground = np.minimum(ground, lowest[1 + dx:nx + 1 + dx, 1 + dy:ny + 1 + dy])
    height = p[:, 2] - ground[ix, iy]
    return p[(height >= cfg.min_height_m) & (height <= cfg.max_height_m)]


@dataclass(frozen=True)
class BoundaryConfig:
    s_bin_m: float = 1.
    d_bin_m: float = .5
    min_run_m: float = 6.         # longer than a car side, tolerates partial rail views
    max_gap_m: float = 1.        # bridge missing rail returns within a run
    lane_keep_m: float = 1.25     # never drop anything this close to the route
    side_window_bins: int = 2     # thinness check this many bins either side
    max_side_fill: float = .3     # share of the run occupied beside the line


def boundary_mask(s, d, config=BoundaryConfig()):
    """True for returns on guardrails, fences and walls along the road.

    In route coordinates (s along the route, d to its left) a roadside
    barrier is a thin line at nearly constant d, also on bends. A line of
    occupied cells at least min_run_m long, with little beside it, is a
    boundary; a car, pedestrian or box is shorter or thick.
    """
    cfg = config
    s = np.asarray(s, float)
    d = np.asarray(d, float)
    mask = np.zeros(len(s), bool)
    if not len(s):
        return mask
    si = np.floor((s - s.min()) / cfg.s_bin_m).astype(int)
    di = np.floor(d / cfg.d_bin_m).astype(int)
    d0 = di.min() - cfg.side_window_bins
    grid = np.zeros((si.max() + 1, di.max() - d0 + cfg.side_window_bins + 1), bool)
    grid[si, di - d0] = True
    run = int(np.ceil(cfg.min_run_m / cfg.s_bin_m))
    gap = int(np.floor(cfg.max_gap_m / cfg.s_bin_m))
    drop = np.zeros_like(grid)
    for col in range(cfg.side_window_bins, grid.shape[1] - cfg.side_window_bins):
        centre = (col + d0 + .5) * cfg.d_bin_m
        if abs(centre) < cfg.lane_keep_m:
            continue
        occupied = grid[:, col].copy()
        if gap:
            hits = np.flatnonzero(occupied)
            for left, right in zip(hits[:-1], hits[1:]):
                if right - left - 1 <= gap:
                    occupied[left:right + 1] = True
        edges = np.flatnonzero(np.diff(np.r_[0, occupied.astype(np.int8), 0]))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start < run:
                continue
            beside = np.r_[grid[start:end, col - cfg.side_window_bins:col],
                           grid[start:end, col + 1:col + 1 + cfg.side_window_bins]]
            if beside.mean() <= cfg.max_side_fill:
                drop[start:end, col] = True
    mask[:] = drop[si, di - d0]
    return mask
