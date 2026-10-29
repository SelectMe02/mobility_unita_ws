"""Conservative, measured visibility evidence for a lost pedestrian footprint."""
import math

import numpy as np


def rectangle_observed_clear(raw_points, corners_lidar):
    """Require returns beyond the entire footprint in every half-degree bin.

    Missing rays, nearer surfaces (including occluders), and a rectangle
    outside the forward view are unknown/occupied, never clear. The vertical
    slice observes pedestrian-body height, not road rays passing under feet.
    This intentionally cannot release holds in open sky with no backstop.
    """
    corners = np.asarray(corners_lidar, float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all() or np.min(corners[:, 0]) <= .5:
        return False
    angles = np.arctan2(corners[:, 1], corners[:, 0])
    if np.max(np.abs(angles)) >= math.radians(75.):
        return False
    step = math.radians(.5)
    first, last = int(math.floor(angles.min()/step)), int(math.ceil(angles.max()/step))
    points = np.asarray(raw_points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 0] > 0.) & (points[:, 2] >= -.8) & (points[:, 2] <= .5)]
    if len(points) == 0:
        return False
    bins = np.floor(np.arctan2(points[:, 1], points[:, 0])/step).astype(int)
    selected = (bins >= first) & (bins <= last)
    depth = np.full(last-first+1, np.inf)
    np.minimum.at(depth, bins[selected]-first, np.hypot(points[selected, 0], points[selected, 1]))
    far = float(np.max(np.linalg.norm(corners, axis=1))) + .3
    return bool(np.all(np.isfinite(depth)) and np.all(depth > far))


def pedestrian_clearance_observer(raw_points, pose, route, config, lidar_to_rear_x_m):
    """Build an observer tied to one validated scan and its capture pose."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)

    def observed(hold, offset_at):
        along = hold['s']
        centre = offset_at(along)
        half_width = config.vehicle_half_width_m + config.side_margin_m + .5
        # Inspect both the driven corridor and the last pedestrian footprint.
        low = min(centre-half_width, hold['d']-.6)
        high = max(centre+half_width, hold['d']+.6)
        half_length = max(.6, hold['l']) + 1.
        corners = []
        for ds, d in ((-half_length, low), (-half_length, high),
                      (half_length, high), (half_length, low)):
            (px, py), _ = route.pose(along+ds, d)
            dx, dy = px-x, py-y
            corners.append((c*dx+s*dy-lidar_to_rear_x_m, -s*dx+c*dy))
        return rectangle_observed_clear(raw_points, corners)
    return observed


def static_clearance_observer(raw_points, pose, route, lidar_to_rear_x_m):
    """Observed visibility of a remembered S4 footprint, not track absence."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)

    def observed(item):
        corners = []
        for ds, dd in ((-1., -1.), (-1., 1.), (1., 1.), (1., -1.)):
            (px, py), _ = route.pose(item['s']+ds*(item['l']+.3),
                                    item['d']+dd*(item['r']+.3))
            dx, dy = px-x, py-y
            corners.append((c*dx+s*dy-lidar_to_rear_x_m, -s*dx+c*dy))
        return rectangle_observed_clear(raw_points, corners)
    return observed
