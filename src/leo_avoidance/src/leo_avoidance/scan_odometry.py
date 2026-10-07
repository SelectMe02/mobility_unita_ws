"""Planar iterative closest point LiDAR odometry with an odometry prior.

Each scan is in LiDAR coordinates (+x forward, +y left). The estimated
transform maps the current scan into the previous scan. No GPS data enters
scan matching; GPS is used only to anchor the resulting relative trajectory.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


def wrap(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compose(a, b):
    ax, ay, yaw = a
    bx, by, byaw = b
    c, s = math.cos(yaw), math.sin(yaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, wrap(yaw + byaw))


def inverse(pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (-c * x - s * y, s * x - c * y, wrap(-yaw))


def between(a, b):
    return compose(inverse(a), b)


def transform(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + [x, y]


def rigid_fit(source, target):
    """Least-squares 2D transform mapping paired source points to targets."""
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    matrix = (source - src_mean).T @ (target - dst_mean)
    u, _, vt = np.linalg.svd(matrix)
    rotation = vt.T @ np.diag([1., np.linalg.det(vt.T @ u.T)]) @ u.T
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    translation = dst_mean - rotation @ src_mean
    return float(translation[0]), float(translation[1]), yaw


@dataclass(frozen=True)
class Match:
    delta_lidar: tuple
    pairs: int
    rmse_m: float
    valid: bool


class ScanOdometry:
    def __init__(self, lidar_to_rear_x_m=1.4, min_z_m=-0.8,
                 max_z_m=0.8, max_range_m=18., voxel_m=.25,
                 max_pair_distance_m=.65, min_pairs=35,
                 max_rmse_m=.25, max_correction_m=.5,
                 max_correction_rad=.12):
        self.extrinsic = (float(lidar_to_rear_x_m), 0., 0.)
        self.min_z_m, self.max_z_m = float(min_z_m), float(max_z_m)
        self.max_range_m, self.voxel_m = float(max_range_m), float(voxel_m)
        self.max_pair_distance_m = float(max_pair_distance_m)
        self.min_pairs, self.max_rmse_m = int(min_pairs), float(max_rmse_m)
        self.max_correction_m = float(max_correction_m)
        self.max_correction_rad = float(max_correction_rad)
        values = (self.extrinsic[0], self.min_z_m, self.max_z_m,
                  self.max_range_m, self.voxel_m, self.max_pair_distance_m,
                  self.max_rmse_m, self.max_correction_m, self.max_correction_rad)
        if (not all(math.isfinite(v) for v in values)
                or self.min_z_m >= self.max_z_m or self.max_range_m <= 0
                or self.voxel_m <= 0 or self.max_pair_distance_m <= 0
                or self.min_pairs < 8 or self.max_rmse_m <= 0
                or self.max_correction_m <= 0 or self.max_correction_rad <= 0):
            raise ValueError('invalid LiDAR odometry parameters')

    def prepare(self, xyz):
        points = np.asarray(xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('cloud must contain XYZ points')
        points = points[np.isfinite(points).all(axis=1)]
        points = points[(points[:, 2] >= self.min_z_m)
                        & (points[:, 2] <= self.max_z_m)]
        points = points[(np.hypot(points[:, 0], points[:, 1]) <= self.max_range_m)
                        & (np.hypot(points[:, 0], points[:, 1]) >= 1.)]
        if len(points) < self.min_pairs:
            raise ValueError('too few obstacle-height LiDAR returns')
        xy = points[:, :2]
        # Deterministic first point per planar voxel, bounded for 10 Hz scans.
        cells = np.floor(xy / self.voxel_m).astype(np.int32)
        _, indices = np.unique(cells, axis=0, return_index=True)
        xy = xy[np.sort(indices)]
        if len(xy) > 1200:
            xy = xy[np.linspace(0, len(xy) - 1, 1200).astype(int)]
        if len(xy) < self.min_pairs:
            raise ValueError('too few independent LiDAR features')
        return xy

    def lidar_delta_from_rear(self, rear_delta):
        return compose(compose(inverse(self.extrinsic), rear_delta), self.extrinsic)

    def rear_delta_from_lidar(self, lidar_delta):
        return compose(compose(self.extrinsic, lidar_delta), inverse(self.extrinsic))

    def match(self, previous_xy, current_xy, odom_rear_delta):
        previous_xy = np.asarray(previous_xy, dtype=float)
        current_xy = np.asarray(current_xy, dtype=float)
        if len(previous_xy) < self.min_pairs or len(current_xy) < self.min_pairs:
            return Match((0., 0., 0.), 0, math.inf, False)
        prior = self.lidar_delta_from_rear(odom_rear_delta)
        estimate = prior
        tree = cKDTree(previous_xy)
        pairs = 0
        for _ in range(10):
            moved = transform(current_xy, estimate)
            distance, index = tree.query(moved, k=1)
            mask = distance <= self.max_pair_distance_m
            pairs = int(mask.sum())
            if pairs < self.min_pairs:
                return Match(estimate, pairs, math.inf, False)
            correction = rigid_fit(moved[mask], previous_xy[index[mask]])
            estimate = compose(correction, estimate)
            if math.hypot(correction[0], correction[1]) < .002 and abs(correction[2]) < .001:
                break
        moved = transform(current_xy, estimate)
        distance, _ = tree.query(moved, k=1)
        inliers = distance <= self.max_pair_distance_m
        pairs = int(inliers.sum())
        rmse = float(np.sqrt(np.mean(distance[inliers] ** 2))) if pairs else math.inf
        deviation = between(prior, estimate)
        valid = (pairs >= self.min_pairs and rmse <= self.max_rmse_m
                 and math.hypot(deviation[0], deviation[1]) <= self.max_correction_m
                 and abs(deviation[2]) <= self.max_correction_rad)
        return Match(estimate, pairs, rmse, valid)
