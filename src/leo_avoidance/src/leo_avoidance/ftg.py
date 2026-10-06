"""PointCloud2-adapted Follow The Gap, based on roboracer_unita_ws FTG.

Coordinates are metres in the LiDAR frame: +x forward, +y left, +z up.
The reference's disparity bubble and metric gap width are retained. This
version stops when no gap fits the MORAI vehicle, instead of choosing a beam
through an undersized opening.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    lidar_to_rear_x_m: float = 1.4
    vehicle_half_width_m: float = 0.946
    side_margin_m: float = 0.25
    front_bumper_from_rear_m: float = 3.845
    max_range_m: float = 15.0
    front_fov_deg: float = 130.0
    scan_bins: int = 261
    disparity_m: float = 0.5
    max_target_bearing_rad: float = 0.35
    max_speed_mps: float = 1.5
    no_return_speed_mps: float = 2.0
    min_no_return_evidence_points: int = 10
    reaction_s: float = 0.4
    brake_mps2: float = 2.0

    def validate(self):
        values = tuple(value for value in vars(self).values() if isinstance(value, (float, int)))
        if not all(math.isfinite(value) for value in values):
            raise ValueError('non-finite avoidance parameter')
        if (self.vehicle_half_width_m <= 0 or self.side_margin_m < 0
                or self.front_bumper_from_rear_m <= self.lidar_to_rear_x_m
                or not 0 < self.front_fov_deg <= 180 or self.scan_bins < 5
                or self.max_range_m <= 0 or self.disparity_m <= 0
                or self.max_target_bearing_rad <= 0 or self.max_speed_mps <= 0
                or self.no_return_speed_mps <= 0
                or self.min_no_return_evidence_points < 1
                or self.reaction_s < 0 or self.brake_mps2 <= 0):
            raise ValueError('invalid avoidance geometry or limits')


@dataclass(frozen=True)
class Plan:
    speed_mps: float
    target_bearing_rad: float
    reason: str
    front_clearance_m: float
    gap_width_m: float
    obstacle_points: int = 0
    front_points: int = 0
    nearest_front_x_m: float = 0.0


class FollowTheGap:
    def __init__(self, config=None):
        self.config = config or Config()
        self.config.validate()
        self.half_fov_rad = math.radians(self.config.front_fov_deg / 2.0)
        self.angles = np.linspace(-self.half_fov_rad, self.half_fov_rad,
                                  self.config.scan_bins)
        self.angle_step_rad = self.angles[1] - self.angles[0]
        self.previous_bearing_rad = 0.0

    @staticmethod
    def _runs(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
        return list(zip(edges[::2], edges[1::2] - 1))

    @staticmethod
    def _remove_ground(points):
        """Discard a measured road plane while keeping returns at any height.

        Accept a plane only when low returns cover a broad part of the scan.
        Sparse low boxes cannot define their own ground; uncertain scans keep
        all returns and retain the existing stop behavior.
        """
        sample = points[(points[:, 0] >= 1.) & (points[:, 0] <= 12.)
                        & (np.abs(points[:, 1]) <= 5.)
                        & (points[:, 2] >= -2.5) & (points[:, 2] <= -.3)]
        if len(sample) < 80:
            return points
        low_limit = np.percentile(sample[:, 2], 40.)
        low = sample[sample[:, 2] <= low_limit]
        bins = np.floor(low[:, 2] / .1).astype(np.int32)
        unique, counts = np.unique(bins, return_counts=True)
        center_z = (unique[np.argmax(counts)] + .5) * .1
        seed = sample[np.abs(sample[:, 2] - center_z) <= .18]
        if (len(seed) < 60 or np.ptp(seed[:, 0]) < 4.
                or np.ptp(seed[:, 1]) < 2.):
            return points
        for _ in range(2):
            design = np.column_stack((seed[:, :2], np.ones(len(seed))))
            plane = np.linalg.lstsq(design, seed[:, 2], rcond=None)[0]
            if (abs(plane[0]) > .12 or abs(plane[1]) > .12
                    or not -2.5 <= plane[2] <= -.3):
                return points
            residual = sample[:, 2] - (sample[:, :2] @ plane[:2] + plane[2])
            seed = sample[np.abs(residual) <= .1]
            if (len(seed) < 60 or np.ptp(seed[:, 0]) < 4.
                    or np.ptp(seed[:, 1]) < 2.):
                return points
        residual = points[:, 2] - (points[:, :2] @ plane[:2] + plane[2])
        return points[np.abs(residual) > .1]

    def plan(self, points_lidar_m, ego_speed_mps, preferred_bearing_rad=0.):
        """Return a bounded low-speed plan or a full-stop plan.

        The cloud must be fresh before calling this function. Empty or
        non-finite input is treated as missing evidence, not free road.
        """
        cfg = self.config
        obstacle_points = 0
        front_points = 0
        nearest_front_x_m = 0.0
        stop = lambda reason, clearance=0.0: Plan(
            0.0, 0.0, reason, clearance, 0.0,
            obstacle_points, front_points, nearest_front_x_m)
        points = np.asarray(points_lidar_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            return stop('invalid cloud shape')
        if not math.isfinite(ego_speed_mps) or ego_speed_mps < 0:
            return stop('invalid Ego speed')
        if not math.isfinite(preferred_bearing_rad):
            return stop('invalid route bearing')
        preferred_bearing_rad = max(-self.half_fov_rad,
                                    min(self.half_fov_rad, preferred_bearing_rad))
        if len(points) == 0:
            return stop('empty cloud')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return stop('no finite points')
        points = points[points[:, 0] > 0]
        points = self._remove_ground(points)
        obstacle_points = len(points)
        if len(points) == 0:
            return stop('no forward points')

        radii = np.hypot(points[:, 0], points[:, 1])
        bearings = np.arctan2(points[:, 1], points[:, 0])
        valid = (radii <= cfg.max_range_m) & (np.abs(bearings) <= self.half_fov_rad)
        if not valid.any():
            # A healthy scan can contain side-wall returns but no hits in
            # the forward sector as the car leaves the tunnel. The caller
            # separately requires valid LiDAR odometry. Crawl toward the
            # route until GPS returns; sparse scans still command a stop.
            if len(points) < cfg.min_no_return_evidence_points:
                return stop('no forward returns')
            bearing_rad = max(-cfg.max_target_bearing_rad,
                              min(cfg.max_target_bearing_rad, preferred_bearing_rad))
            self.previous_bearing_rad = bearing_rad
            return Plan(min(cfg.max_speed_mps, cfg.no_return_speed_mps),
                        bearing_rad, 'no forward returns; cautious travel',
                        0.0, 0.0, obstacle_points)
        points, radii, bearings = points[valid], radii[valid], bearings[valid]
        indices = np.rint((bearings + self.half_fov_rad) / self.angle_step_rad).astype(int)
        indices = np.clip(indices, 0, cfg.scan_bins - 1)
        ranges = np.full(cfg.scan_bins, cfg.max_range_m)
        np.minimum.at(ranges, indices, radii)

        # Extend each close return by the vehicle half width and margin.
        # Also extend across a range discontinuity, as in the reference FTG.
        bubble_m = cfg.vehicle_half_width_m + cfg.side_margin_m
        blocked = np.zeros(cfg.scan_bins, dtype=bool)
        hits = np.flatnonzero(ranges < cfg.max_range_m)
        for index in hits:
            beams = int(math.ceil(math.atan2(bubble_m, max(0.1, ranges[index]))
                                  / self.angle_step_rad))
            blocked[max(0, index - beams):min(cfg.scan_bins, index + beams + 1)] = True
        for index in np.flatnonzero(np.abs(np.diff(ranges)) > cfg.disparity_m):
            near_index = index if ranges[index] < ranges[index + 1] else index + 1
            beams = int(math.ceil(math.atan2(bubble_m, max(0.1, ranges[near_index]))
                                  / self.angle_step_rad))
            blocked[max(0, index + 1 - beams):min(cfg.scan_bins, index + 1 + beams)] = True

        front_bumper_from_lidar_m = cfg.front_bumper_from_rear_m - cfg.lidar_to_rear_x_m
        forward = points[(np.abs(points[:, 1]) <= bubble_m)
                         & (points[:, 0] > 0)]
        front_points = len(forward)
        closest_x_m = float(np.min(forward[:, 0])) if len(forward) else cfg.max_range_m
        nearest_front_x_m = closest_x_m if len(forward) else 0.0
        front_clearance_m = max(0.0, closest_x_m - front_bumper_from_lidar_m)
        stopping_distance_m = (ego_speed_mps * cfg.reaction_s
                               + ego_speed_mps ** 2 / (2 * cfg.brake_mps2))
        if len(forward) and front_clearance_m <= stopping_distance_m + cfg.side_margin_m:
            return stop('front obstacle inside stopping distance', front_clearance_m)

        candidates = []
        steer_left = int(np.searchsorted(self.angles, -cfg.max_target_bearing_rad, side='left'))
        steer_right = int(np.searchsorted(self.angles, cfg.max_target_bearing_rad, side='right')) - 1
        for left, right in self._runs(~blocked):
            left = max(left, steer_left)
            right = min(right, steer_right)
            if left > right:
                continue
            # A finite range cap prevents empty sky from overwhelming the score.
            depth_m = min(6.0, float(np.min(ranges[left:right + 1])))
            width_m = 2.0 * depth_m * math.sin(min(math.pi,
                (right - left + 1) * self.angle_step_rad) / 2.0)
            # Blocked bins already include the vehicle half width and side
            # margin around every return. Requiring that width once more
            # rejects physically passable gaps beside tunnel traffic.
            # Aim at the closest safe bearing to the mapped route. The
            # obstacle bubble already excludes unsafe edge bearings; aiming
            # at a gap's midpoint can steer away from a curved route.
            preferred_index = int(round((preferred_bearing_rad + self.half_fov_rad)
                                        / self.angle_step_rad))
            middle = max(left, min(right, preferred_index))
            angle_rad = float(self.angles[middle])
            score = (-abs(angle_rad - preferred_bearing_rad)
                     + .02 * width_m
                     - .1 * abs(angle_rad - self.previous_bearing_rad))
            candidates.append((score, angle_rad, width_m))
        if not candidates:
            return stop('no gap fits vehicle width', front_clearance_m)
        _, angle_rad, gap_width_m = max(candidates, key=lambda item: item[0])
        bearing_rad = angle_rad
        self.previous_bearing_rad = bearing_rad
        steering_fraction = abs(bearing_rad) / cfg.max_target_bearing_rad
        # Continuous steering cap avoids abrupt 18 -> 14.4 -> 8.1 km/h
        # target jumps as a LiDAR bearing crosses a bucket boundary.
        speed_fraction = float(np.interp(steering_fraction,
                                        [0., .1, .35, .7, 1.],
                                        [1., .8, .45, .3, .3]))
        target_speed_mps = cfg.max_speed_mps * speed_fraction
        return Plan(target_speed_mps, bearing_rad, 'gap fits', front_clearance_m,
                    gap_width_m, obstacle_points, front_points, nearest_front_x_m)
