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


def corridor_observed_clear(raw_points, corners_lidar):
    """Positive ray evidence through the corridor's body-height slice.

    A far road return may be valid evidence: the ray must cross the entire
    inspected distance at the same body heights as the legacy observer.
    A nearby road return below that slice is not evidence of empty space.
    No-return rays and angular holes still fail the original coverage test.
    """
    corners = np.asarray(corners_lidar, float)
    points = np.asarray(raw_points, float).reshape(-1, 3)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return False
    points = points[np.isfinite(points).all(axis=1)]
    if np.min(corners[:, 0]) <= .5:
        return False
    angles = np.arctan2(corners[:, 1], corners[:, 0])
    if np.max(np.abs(angles)) >= math.radians(75.):
        return False
    step = math.radians(.5)
    radius = np.hypot(points[:, 0], points[:, 1])
    bins = np.floor(np.arctan2(points[:, 1], points[:, 0])/step).astype(int)

    def intersections(angle):
        ray = np.array([math.cos(angle), math.sin(angle)])
        distances = []
        for p, q in zip(corners, np.roll(corners, -1, axis=0)):
            edge = q-p
            cross = ray[0]*edge[1]-ray[1]*edge[0]
            if abs(cross) < 1e-9:
                continue
            t = (p[0]*edge[1]-p[1]*edge[0])/cross
            u = (p[0]*ray[1]-p[1]*ray[0])/cross
            if t > 0. and -1e-8 <= u <= 1.+1e-8:
                distances.append(t)
        return distances

    for index in range(int(math.floor(angles.min()/step)),
                       int(math.floor(angles.max()/step))+1):
        low, high = max(index*step, angles.min()), min((index+1)*step, angles.max())
        probes = [low, high, (low+high)/2.] + list(angles[(angles >= low) & (angles <= high)])
        distances = [v for angle in probes for v in intersections(angle)]
        if not distances:
            return False
        near, far = min(distances), max(distances)+.3
        selected = (bins == index) & (points[:, 0] > 0.)
        r, z = radius[selected], points[selected, 2]
        # Actual nearer body-height returns mean occlusion or occupancy.
        if np.any((r <= far) & (z >= -.8) & (z <= .5)):
            return False
        denominator = np.maximum(r, .001)
        z_near, z_far = z*near/denominator, z*far/denominator
        if not np.any((r > far) & (z_near >= -.8) & (z_near <= .5)
                      & (z_far >= -.8) & (z_far <= .5)):
            return False
    return True


def pedestrian_clearance_observer(raw_points, pose, route, config, lidar_to_rear_x_m):
    """Build an observer tied to one validated scan and its capture pose."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)

    def observed(hold, offset_at):
        along = hold['s']
        half_width = config.vehicle_half_width_m + config.side_margin_m + .5
        # Release belongs to the observed driving corridor, not the old ID's
        # roadside footprint. Current crossing/occupancy vetoes live in the
        # planner; unknown rays here still fail closed.
        half_length = max(.6, hold['l']) + 1.
        # Include the approach from Ego's front bumper, not only the old
        # crossing location. Restrict projection to this hazard's route window.
        lo, hi = route.window(along-50., along+10.)
        ego_s, _, _ = route.project(x, y, lo, hi)
        start = min(-half_length, ego_s+config.front_extent_m-along)
        # Follow the actual route/avoidance offsets through the whole corridor
        # rather than certifying a straight chord through a curved road.
        stations = np.linspace(along+start, along+half_length,
            max(2, int(math.ceil((half_length-start)/half_length))+1))
        for first, last in zip(stations, stations[1:]):
            corners = []
            for station, side in ((first, -1.), (first, 1.), (last, 1.), (last, -1.)):
                (px, py), _ = route.pose(station, offset_at(station)+side*half_width)
                dx, dy = px-x, py-y
                corners.append((c*dx+s*dy-lidar_to_rear_x_m, -s*dx+c*dy))
            if not corridor_observed_clear(raw_points, corners):
                return False
        return True

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
