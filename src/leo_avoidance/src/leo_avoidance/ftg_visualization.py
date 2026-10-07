"""Diagnostic clustering of the exact point set used by blackout FTG."""
import math
import numpy as np
from scipy.spatial import cKDTree


def relative_lidar_xy(pose, output_pose, lidar_to_rear_x_m=1.4):
    """Place a map-frame pose in the current output pose's LiDAR view."""
    if (pose is None or output_pose is None or len(pose) < 2
            or len(output_pose) < 3):
        return None
    values = tuple(pose[:2]) + tuple(output_pose[:3])
    if not all(math.isfinite(float(value)) for value in values):
        return None
    dx = float(pose[0]) - float(output_pose[0])
    dy = float(pose[1]) - float(output_pose[1])
    yaw = float(output_pose[2])
    return (math.cos(yaw) * dx + math.sin(yaw) * dy - lidar_to_rear_x_m,
            -math.sin(yaw) * dx + math.cos(yaw) * dy)


def line_candidate_points(group, size, yaw, bin_m=.5):
    """Return a polyline for a long narrow candidate, else None."""
    if size[0] < 3. or size[0] < 4. * size[1] or len(group) < 8:
        return None
    direction = np.array([math.cos(yaw), math.sin(yaw)])
    along = group[:, :2] @ direction
    edges = np.arange(along.min(), along.max() + bin_m, bin_m)
    if len(edges) < 3:
        return None
    line = []
    for left, right in zip(edges[:-1], edges[1:]):
        patch = group[(along >= left) & (along < right)]
        if len(patch):
            line.append(np.median(patch, axis=0))
    return np.asarray(line) if len(line) >= 3 else None


def cluster_boxes(points, voxel_m=.25, link_m=.75, max_boxes=100,
                  oriented=False):
    """Return (minimum XYZ, maximum XYZ, point count, distance) per group.

    Voxelization only affects grouping; each box bounds the original points.
    With oriented=True, append fitted center, size, yaw, and points for RViz.
    There is no minimum cluster size because FTG also reacts to single hits.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('cluster points must contain XYZ')
    if len(points) == 0:
        return []
    if not np.isfinite(points).all() or voxel_m <= 0 or link_m <= 0 or max_boxes < 1:
        raise ValueError('invalid clustering input')
    cells, inverse = np.unique(np.floor(points / voxel_m).astype(np.int32),
                               axis=0, return_inverse=True)
    centers = (cells + .5) * voxel_m
    tree = cKDTree(centers)
    seen = np.zeros(len(cells), dtype=bool)
    boxes = []
    for start in range(len(cells)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in tree.query_ball_point(centers[index], link_m):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        group = points[np.isin(inverse, component)]
        box = (group.min(axis=0), group.max(axis=0), len(group),
               float(np.min(np.linalg.norm(group, axis=1))))
        if oriented:
            xy_center = np.mean(group[:, :2], axis=0)
            centered = group[:, :2] - xy_center
            if len(group) > 1 and np.ptp(group[:, :2], axis=0).max() > 1e-6:
                covariance = centered.T @ centered
                axis = np.linalg.eigh(covariance)[1][:, -1]
                yaw = math.atan2(axis[1], axis[0])
            else:
                yaw = 0.
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[c, -s], [s, c]])
            local = centered @ rotation
            low, high = local.min(axis=0), local.max(axis=0)
            fitted_xy = xy_center + ((low + high) / 2.) @ rotation.T
            center = np.array([fitted_xy[0], fitted_xy[1],
                               (box[0][2] + box[1][2]) / 2.])
            size = np.maximum(np.r_[high - low, box[1][2] - box[0][2]], .15)
            box += (center, size, yaw, group)
        boxes.append(box)
    boxes.sort(key=lambda item: item[3])
    return boxes[:max_boxes]
