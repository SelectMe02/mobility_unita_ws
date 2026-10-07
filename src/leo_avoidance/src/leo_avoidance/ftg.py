"""PointCloud2-adapted Follow The Gap, based on roboracer_unita_ws FTG.

Coordinates are metres in the LiDAR frame: +x forward, +y left, +z up.
The reference's disparity bubble and metric gap width are retained. This
version stops when no gap fits the MORAI vehicle, instead of choosing a beam
through an undersized opening.
"""
from dataclasses import dataclass
import math

import numpy as np

from leo_avoidance.ftg_visualization import cluster_boxes


@dataclass(frozen=True)
class Config:
    lidar_to_rear_x_m: float = 1.4
    vehicle_half_width_m: float = 0.946
    side_margin_m: float = 0.4
    front_bumper_from_rear_m: float = 3.845
    max_range_m: float = 15.0
    front_fov_deg: float = 270.0
    scan_bins: int = 541
    disparity_m: float = 0.5
    max_target_bearing_rad: float = math.radians(35.)
    post_pass_target_bearing_rad: float = math.radians(50.)
    post_pass_clear_front_m: float = 15.0
    post_pass_planning_depth_m: float = 20.0
    max_speed_mps: float = 1.5
    no_return_speed_mps: float = 2.0
    min_no_return_evidence_points: int = 10
    reaction_s: float = 0.4
    brake_mps2: float = 2.0
    wall_slow_clearance_m: float = 0.5
    wall_fast_clearance_m: float = 1.4
    expected_ground_z_m: float = -1.23
    ground_height_tolerance_m: float = 0.25
    collision_min_z_m: float = -1.13
    collision_max_z_m: float = 0.8
    right_wall_target_clearance_m: float = 2.9
    right_wall_clearance_tolerance_m: float = 0.1
    wall_point_tolerance_m: float = 0.35
    wall_corridor_horizon_m: float = 6.0
    right_wall_min_clearance_m: float = 0.9
    avoidance_trigger_range_m: float = 20.0
    rear_detection_half_angle_deg: float = 15.0
    rear_detection_max_range_m: float = 12.0
    rear_detection_max_lateral_m: float = 2.8
    rear_detection_min_points: int = 3
    rear_detection_confirm_scans: int = 2
    right_wall_offset_gain_rad_per_m: float = 0.25
    right_wall_heading_gain: float = 1.0
    right_wall_max_correction_rad: float = 0.22
    right_wall_route_disagreement_rad: float = math.radians(15.)
    right_wall_drift_weight: float = 0.9
    unwalled_route_bearing_limit_rad: float = math.radians(7.)

    def validate(self):
        values = tuple(value for value in vars(self).values() if isinstance(value, (float, int)))
        if not all(math.isfinite(value) for value in values):
            raise ValueError('non-finite avoidance parameter')
        if (self.vehicle_half_width_m <= 0 or self.side_margin_m < 0
                or self.front_bumper_from_rear_m <= self.lidar_to_rear_x_m
                or not 0 < self.front_fov_deg <= 360 or self.scan_bins < 5
                or self.max_range_m <= 0 or self.disparity_m <= 0
                or self.max_target_bearing_rad <= 0 or self.max_speed_mps <= 0
                or self.post_pass_target_bearing_rad < self.max_target_bearing_rad
                or self.post_pass_target_bearing_rad > math.radians(self.front_fov_deg / 2.)
                or self.post_pass_clear_front_m <= 0
                or self.post_pass_planning_depth_m <= 0
                or self.no_return_speed_mps <= 0
                or self.min_no_return_evidence_points < 1
                or self.reaction_s < 0 or self.brake_mps2 <= 0
                or self.wall_slow_clearance_m < 0
                or self.wall_fast_clearance_m <= self.wall_slow_clearance_m
                or self.expected_ground_z_m >= 0
                or self.ground_height_tolerance_m <= 0
                or self.collision_min_z_m >= self.collision_max_z_m
                or self.right_wall_target_clearance_m < self.right_wall_min_clearance_m
                or self.right_wall_clearance_tolerance_m < 0
                or self.wall_point_tolerance_m <= 0
                or self.wall_corridor_horizon_m <= 0
                or self.right_wall_target_clearance_m - self.right_wall_clearance_tolerance_m
                <= self.right_wall_min_clearance_m
                or self.right_wall_min_clearance_m <= 0
                or self.avoidance_trigger_range_m <= 0
                or not 0 < self.rear_detection_half_angle_deg < 90
                or self.rear_detection_max_range_m <= 0
                or self.rear_detection_max_lateral_m <= 0
                or self.rear_detection_min_points < 1
                or self.rear_detection_confirm_scans < 1
                or self.right_wall_offset_gain_rad_per_m < 0
                or self.right_wall_heading_gain < 0
                or self.right_wall_max_correction_rad < 0
                or self.right_wall_route_disagreement_rad <= 0
                or not 0 <= self.right_wall_drift_weight <= 1
                or self.unwalled_route_bearing_limit_rad <= 0):
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
    right_wall_clearance_m: float = None
    right_wall_target_clearance_m: float = None
    right_wall_heading_rad: float = None
    wall_follow_weight: float = 0.0
    route_rejoin_remaining_m: float = None
    avoidance_phase: str = 'WALL_FOLLOW'
    rear_obstacle_points: int = 0
    rear_confirm_scans: int = 0
    rear_scan_points: int = 0
    candidate_bearings_rad: tuple = ()


class FollowTheGap:
    def __init__(self, config=None):
        self.config = config or Config()
        self.config.validate()
        self.half_fov_rad = math.radians(self.config.front_fov_deg / 2.0)
        self.angles = np.linspace(-self.half_fov_rad, self.half_fov_rad,
                                  self.config.scan_bins)
        self.angle_step_rad = self.angles[1] - self.angles[0]
        self.previous_bearing_rad = 0.0
        self.right_wall_target_clearance_m = None
        self.right_wall_filtered_clearance_m = None
        self.avoidance_active = False
        self.rear_confirm_scans = 0
        self.passing_side = 0

    def reset_wall_target(self):
        self.right_wall_target_clearance_m = None
        self.right_wall_filtered_clearance_m = None
        self.avoidance_active = False
        self.rear_confirm_scans = 0
        self.passing_side = 0

    def _rear_obstacle_points(self, points):
        """Close returns within 15 degrees of straight behind the LiDAR."""
        radii = np.hypot(points[:, 0], points[:, 1])
        bearings = np.abs(np.arctan2(points[:, 1], points[:, 0]))
        return points[(points[:, 0] < 0.)
                      & (math.pi - bearings <= math.radians(
                          self.config.rear_detection_half_angle_deg))
                      & (radii <= self.config.rear_detection_max_range_m)
                      & (np.abs(points[:, 1]) <=
                         self.config.rear_detection_max_lateral_m)]

    def _update_avoidance_phase(self, wall_follow, closest_front_x_m,
                                rear_obstacle_points):
        if wall_follow:
            if closest_front_x_m is not None \
                    and closest_front_x_m <= self.config.avoidance_trigger_range_m:
                self.avoidance_active = True
                self.rear_confirm_scans = 0
            elif self.avoidance_active:
                if (self.passing_side != 0 and rear_obstacle_points >=
                        self.config.rear_detection_min_points):
                    self.rear_confirm_scans += 1
                else:
                    self.rear_confirm_scans = 0
                if self.rear_confirm_scans >= self.config.rear_detection_confirm_scans:
                    self.avoidance_active = False
                    self.rear_confirm_scans = 0
                    self.passing_side = 0
        return 'PASSING' if self.avoidance_active else (
            'WALL_FOLLOW' if wall_follow else 'ROUTE')

    def _wall_fit(self, points, sign):
        """Fit one persistent outside boundary, not a short obstacle patch."""
        side = points[(points[:, 0] >= 1.) & (points[:, 0] <= 20.)
                      & (points[:, 1] * sign > self.config.vehicle_half_width_m + .3)]
        samples = []
        for start in range(1, 20):
            strip = side[(side[:, 0] >= start) & (side[:, 0] < start + 1.)]
            if len(strip) >= 5:
                # Use the road-facing wall surface for vehicle clearance.
                # The outermost surface can be metres behind wall panels.
                samples.append((float(np.median(strip[:, 0])),
                                self._wall_strip_surface(strip, sign)))
        if len(samples) < 5 or samples[-1][0] - samples[0][0] < 4.:
            return None
        x, y = np.asarray(samples).T
        slope, offset = np.polyfit(x, y, 1)
        curvature = 0.
        residual = np.abs(y - (slope * x + offset))
        if len(samples) >= 8 and x[-1] - x[0] >= 7.:
            curve, tangent, intercept = np.polyfit(x, y, 2)
            curve_residual = np.abs(y - (curve * x * x + tangent * x + intercept))
            if np.percentile(curve_residual, 80.) < .55 * np.percentile(residual, 80.):
                curvature, slope, offset = curve, tangent, intercept
                residual = curve_residual
        distance = sign * offset / math.hypot(1., slope)
        end_slope = slope + 2. * curvature * x[-1]
        if (max(abs(slope), abs(end_slope)) > .6
                or np.percentile(residual, 80.) > .4
                or not 1.3 < distance < 8.):
            return None
        return (max(0., distance - self.config.vehicle_half_width_m),
                math.atan(slope), float(slope), float(offset),
                float(curvature), float(x[0]), float(x[-1]))

    def _right_wall_geometry(self, points):
        fit = self._wall_fit(points, -1.)
        return fit[:2] if fit is not None else None

    @staticmethod
    def _wall_strip_surface(strip, sign):
        """Road-facing edge of one wall panel, excluding separate inner objects."""
        outward = sign * strip[:, 1]
        outer = np.percentile(outward, 90.)
        wall_band = outward[outward >= outer - .9]
        return sign * float(np.percentile(wall_band, 10.))

    def _wall_profile(self, points, sign):
        """Trace a long outer side surface one metre at a time.

        A short car or box cannot qualify as a tunnel boundary.  This also
        works when wall panels are not well described by one polynomial.
        """
        side = points[(points[:, 0] >= 1.)
                      & (points[:, 0] <= self.config.max_range_m)
                      & (points[:, 1] * sign >
                         self.config.vehicle_half_width_m + .3)]
        samples = []
        for start in range(1, int(self.config.max_range_m) + 1):
            strip = side[(side[:, 0] >= start) & (side[:, 0] < start + 1.)]
            if len(strip) >= 5:
                samples.append((float(np.median(strip[:, 0])),
                                self._wall_strip_surface(strip, sign)))
        runs, current = [], []
        for sample in samples:
            if current and (sample[0] - current[-1][0] > 1.8
                            or abs(sample[1] - current[-1][1]) > 1.2):
                runs.append(current)
                current = []
            current.append(sample)
        if current:
            runs.append(current)
        eligible = [run for run in runs if len(run) >= 7
                    and run[-1][0] - run[0][0] >= 6.
                    and run[0][0] <= 5.]
        return np.asarray(max(eligible, key=len), dtype=float) if eligible else None

    def _wall_fit_from_profile(self, profile, sign):
        """Use the measured near segment to keep a wall corridor on fit failure."""
        near = profile[profile[:, 0] <= profile[0, 0] + 7.]
        if len(near) < 5:
            return None
        slope, offset = np.polyfit(near[:, 0], near[:, 1], 1)
        distance = sign * offset / math.hypot(1., slope)
        if abs(slope) > .6 or not 1.3 < distance < 8.:
            return None
        return (max(0., distance - self.config.vehicle_half_width_m),
                math.atan(slope), float(slope), float(offset), 0.,
                float(near[0, 0]), float(near[-1, 0]))

    def _linear_side_boundaries(self, points, walls):
        """Find long connected wall fragments missed by the strip profile."""
        keep = np.ones(len(points), dtype=bool)
        walls = list(walls)
        for box in cluster_boxes(points, oriented=True):
            _, _, count, _, _, size, _, group = box
            if (count < 80 or size[0] < 6. or size[0] < 4. * size[1]
                    or np.ptp(group[:, 0]) < 5.5):
                continue
            sign = 1. if np.median(group[:, 1]) > 0. else -1.
            if (np.min(sign * group[:, 1]) <
                    self.config.vehicle_half_width_m + .5):
                continue
            index = 1 if sign > 0 else 0
            if walls[index] is None and np.min(group[:, 0]) > 8.:
                continue
            slope, offset = np.polyfit(group[:, 0], group[:, 1], 1)
            distance = sign * offset / math.hypot(1., slope)
            residual = np.abs(group[:, 1] - (slope * group[:, 0] + offset))
            if (abs(slope) > .6 or not 1.3 < distance < 8.
                    or np.percentile(residual, 80.) > .55):
                continue
            current = walls[index]
            if current is not None:
                midpoint = float(np.median(group[:, 0]))
                _, _, old_slope, old_offset, curvature, _, _ = current
                old_y = curvature * midpoint ** 2 + old_slope * midpoint + old_offset
                if abs(old_y - (slope * midpoint + offset)) > 1.2:
                    continue
            else:
                walls[index] = (max(0., distance - self.config.vehicle_half_width_m),
                                math.atan(slope), float(slope), float(offset), 0.,
                                float(np.min(group[:, 0])),
                                float(np.max(group[:, 0])))
            # Remove only this connected surface; nearby isolated boxes stay.
            from scipy.spatial import cKDTree
            near = cKDTree(group).query(points, k=1)[0] < 1e-6
            keep &= ~near
        return points[keep], tuple(walls)

    def _track_obstacles(self, points):
        """Separate line-like tunnel boundaries from interior returns."""
        profiles = (self._wall_profile(points, -1.),
                    self._wall_profile(points, 1.))
        walls = tuple(self._wall_fit(points, sign)
                      or (self._wall_fit_from_profile(profile, sign)
                          if profile is not None else None)
                      for sign, profile in zip((-1., 1.), profiles))
        keep = np.ones(len(points), dtype=bool)
        for sign, wall, profile in zip((-1., 1.), walls, profiles):
            if profile is not None and wall is not None:
                near_profile = ((points[:, 0] >= profile[0, 0] - 1.)
                                & (points[:, 0] <= profile[-1, 0] + 1.))
                predicted_y = np.interp(points[:, 0], profile[:, 0],
                                        profile[:, 1])
                # Wall panels have depth: discard returns behind the measured
                # road-facing surface as well as returns on that surface.
                keep &= ~((points[:, 1] * sign > 0.) & near_profile
                          & (sign * (points[:, 1] - predicted_y)
                             >= -self.config.wall_point_tolerance_m))
            elif wall is not None:
                _, _, slope, offset, curvature, first_x, last_x = wall
                predicted_y = (curvature * points[:, 0] ** 2
                               + slope * points[:, 0] + offset)
                keep &= ~((points[:, 1] * sign > 0.)
                          & (points[:, 0] >= first_x - 1.)
                          & (points[:, 0] <= last_x + 1.)
                          & (np.abs(points[:, 1] - predicted_y)
                             <= self.config.wall_point_tolerance_m))
        return self._linear_side_boundaries(points[keep], walls)

    @staticmethod
    def _runs(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
        return list(zip(edges[::2], edges[1::2] - 1))

    def _remove_ground(self, points):
        """Discard a measured road plane while keeping returns at any height.

        Accept a plane only when low returns cover a broad part of the scan.
        Sparse low boxes cannot define their own ground; uncertain scans keep
        all returns and retain the existing stop behavior.
        """
        sample = points[(points[:, 0] >= 1.) & (points[:, 0] <= 12.)
                        & (np.abs(points[:, 1]) <= 5.)
                        & (points[:, 2] >= -2.5) & (points[:, 2] <= -.3)]
        if len(sample) < 60:
            return points
        low_limit = np.percentile(sample[:, 2], 40.)
        low = sample[sample[:, 2] <= low_limit]
        bins = np.floor(low[:, 2] / .1).astype(np.int32)
        unique, counts = np.unique(bins, return_counts=True)
        center_z = (unique[np.argmax(counts)] + .5) * .1
        seed = sample[np.abs(sample[:, 2] - center_z) <= .18]
        if (len(seed) < 30 or np.ptp(seed[:, 0]) < 2.
                or np.ptp(seed[:, 1]) < 2.):
            return points
        for _ in range(2):
            design = np.column_stack((seed[:, :2], np.ones(len(seed))))
            plane = np.linalg.lstsq(design, seed[:, 2], rcond=None)[0]
            if (abs(plane[0]) > .12 or abs(plane[1]) > .12
                    or abs(plane[2] - self.config.expected_ground_z_m)
                    > self.config.ground_height_tolerance_m):
                return points
            residual = sample[:, 2] - (sample[:, :2] @ plane[:2] + plane[2])
            seed = sample[np.abs(residual) <= .1]
            if (len(seed) < 30 or np.ptp(seed[:, 0]) < 2.
                    or np.ptp(seed[:, 1]) < 2.):
                return points
        residual = points[:, 2] - (points[:, :2] @ plane[:2] + plane[2])
        return points[np.abs(residual) > .1]

    @staticmethod
    def _safe_speed_for_clearance(clearance_m, reaction_s, brake_mps2):
        """Maximum speed that can stop within the observed clearance."""
        clearance_m = max(0., clearance_m)
        return (math.sqrt((brake_mps2 * reaction_s) ** 2
                          + 2. * brake_mps2 * clearance_m)
                - brake_mps2 * reaction_s)

    def _in_avoidance_sector(self, points):
        radii = np.hypot(points[:, 0], points[:, 1])
        bearings = np.arctan2(points[:, 1], points[:, 0])
        return points[(radii <= self.config.max_range_m)
                      & (np.abs(bearings) <= self.half_fov_rad)]

    def _in_collision_height(self, points):
        # Tunnel ceilings and the road surface are visible to LiDAR but do
        # not occupy the Ego vehicle's vertical collision envelope.
        return points[(points[:, 2] >= self.config.collision_min_z_m)
                      & (points[:, 2] <= self.config.collision_max_z_m)]

    def obstacle_candidates(self, points_lidar_m):
        """Exactly the cloud returns FTG considers as obstacle hits."""
        points = np.asarray(points_lidar_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('cloud must contain XYZ points')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return points
        candidates = self._in_avoidance_sector(
            self._in_collision_height(self._remove_ground(points)))
        return self._track_obstacles(candidates)[0]

    def rear_obstacle_candidates(self, points_lidar_m):
        """Filtered returns that can confirm completion of an active pass."""
        points = np.asarray(points_lidar_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('cloud must contain XYZ points')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return points
        return self._rear_obstacle_points(
            self._in_collision_height(self._remove_ground(points)))

    def plan(self, points_lidar_m, ego_speed_mps, preferred_bearing_rad=0.,
             wall_follow=False, wall_follow_weight=1.,
             route_rejoin_remaining_m=None):
        """Return a bounded low-speed plan or a full-stop plan.

        The cloud must be fresh before calling this function. Empty or
        non-finite input is treated as missing evidence, not free road.
        """
        cfg = self.config
        obstacle_points = 0
        front_points = 0
        nearest_front_x_m = 0.0
        wall_clearance = None
        wall_heading = None
        rear_obstacle_points = 0
        rear_scan_points = 0
        gap_options = []
        avoidance_phase = 'WALL_FOLLOW' if wall_follow else 'ROUTE'
        if not wall_follow:
            self.reset_wall_target()
        wall_follow_weight = float(wall_follow_weight) if wall_follow else 0.
        if not math.isfinite(wall_follow_weight) or not 0. <= wall_follow_weight <= 1.:
            return Plan(0., 0., 'invalid wall follow weight', 0., 0.)
        stop = lambda reason, clearance=0.0: Plan(
            0.0, 0.0, reason, clearance, 0.0,
            obstacle_points, front_points, nearest_front_x_m,
            wall_clearance, self.right_wall_target_clearance_m, wall_heading,
            wall_follow_weight, route_rejoin_remaining_m,
            avoidance_phase, rear_obstacle_points, self.rear_confirm_scans,
            rear_scan_points,
            tuple(float(item[1]) for item in gap_options))
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
        points = self._remove_ground(points)
        points = self._in_collision_height(points)
        if len(points) == 0:
            return stop('no points at collision height')
        rear_scan_points = int(np.count_nonzero(
            (points[:, 0] < 0.)
            & (np.hypot(points[:, 0], points[:, 1]) <=
               cfg.rear_detection_max_range_m)))
        rear_obstacle_points = len(self._rear_obstacle_points(points))

        candidates = self._in_avoidance_sector(points)
        if not len(candidates):
            avoidance_phase = self._update_avoidance_phase(
                wall_follow, None, rear_obstacle_points)
            if avoidance_phase == 'PASSING':
                wall_follow_weight = 0.
            # A healthy scan can contain side-wall returns but no hits in
            # the forward sector as the car leaves the tunnel. The caller
            # separately requires valid LiDAR odometry. Crawl toward the
            # route until GPS returns; sparse scans still command a stop.
            forward_evidence = int(np.count_nonzero(points[:, 0] > 0.))
            if forward_evidence < cfg.min_no_return_evidence_points:
                return stop('no forward returns')
            bearing_rad = (float(np.clip(preferred_bearing_rad,
                           -cfg.unwalled_route_bearing_limit_rad,
                           cfg.unwalled_route_bearing_limit_rad))
                           if wall_follow else
                           float(np.clip(preferred_bearing_rad,
                           -cfg.max_target_bearing_rad,
                           cfg.max_target_bearing_rad)))
            self.previous_bearing_rad = bearing_rad
            return Plan(min(cfg.max_speed_mps, cfg.no_return_speed_mps),
                        bearing_rad, 'no forward returns; cautious travel',
                        0.0, 0.0, obstacle_points,
                        rear_obstacle_points=rear_obstacle_points,
                        rear_scan_points=rear_scan_points,
                        rear_confirm_scans=self.rear_confirm_scans,
                        avoidance_phase=avoidance_phase)
        points = candidates
        obstacles, wall_fits = self._track_obstacles(points)
        obstacle_points = len(obstacles)
        radii = np.hypot(obstacles[:, 0], obstacles[:, 1])
        bearings = np.arctan2(obstacles[:, 1], obstacles[:, 0])
        forward_radii = np.hypot(points[points[:, 0] > 0., 0],
                                 points[points[:, 0] > 0., 1])
        if len(forward_radii) == 0:
            return stop('no forward returns')
        wall = wall_fits[0][:2] if wall_fits[0] is not None else None
        if wall is not None:
            wall_clearance, wall_heading = wall
            if wall_follow:
                if self.right_wall_target_clearance_m is None:
                    self.right_wall_target_clearance_m = (
                        cfg.right_wall_target_clearance_m)
                previous = self.right_wall_filtered_clearance_m
                self.right_wall_filtered_clearance_m = (
                    wall_clearance if previous is None
                    else .65 * previous + .35 * wall_clearance)
        elif wall_follow:
            # When the wall disappears at the portal, do not let an
            # uncorrected LiDAR map pose demand a full 35-degree turn.
            preferred_bearing_rad = float(np.clip(
                preferred_bearing_rad, -cfg.unwalled_route_bearing_limit_rad,
                cfg.unwalled_route_bearing_limit_rad))
        bubble_m = cfg.vehicle_half_width_m + cfg.side_margin_m
        front_bumper_from_lidar_m = cfg.front_bumper_from_rear_m - cfg.lidar_to_rear_x_m
        forward = obstacles[(np.abs(obstacles[:, 1]) <= bubble_m)
                            & (obstacles[:, 0] > 0)]
        front_points = len(forward)
        closest_x_m = float(np.min(forward[:, 0])) if len(forward) else cfg.max_range_m
        nearest_front_x_m = closest_x_m if len(forward) else 0.0
        front_clearance_m = max(0.0, closest_x_m - front_bumper_from_lidar_m)
        avoidance_phase = self._update_avoidance_phase(
            wall_follow, closest_x_m if len(forward) else None,
            rear_obstacle_points)
        post_pass_search = (avoidance_phase == 'PASSING'
                            and (not len(forward)
                                 or (self.passing_side != 0
                                     and closest_x_m > cfg.post_pass_clear_front_m)))
        steering_limit_rad = (cfg.post_pass_target_bearing_rad
                              if post_pass_search else cfg.max_target_bearing_rad)
        if avoidance_phase == 'PASSING':
            wall_follow_weight = 0.
        stopping_distance_m = (ego_speed_mps * cfg.reaction_s
                               + ego_speed_mps ** 2 / (2 * cfg.brake_mps2))
        if len(forward) and front_clearance_m <= cfg.side_margin_m + .5:
            return stop('front obstacle inside stopping distance', front_clearance_m)

        # A ray to the far end of a tunnel eventually intersects its wall,
        # even when a curved path around a nearer obstacle is wide enough.
        # Pick the immediate gap near the first obstacle; retain the full
        # scan for wall following, front clearance, and speed limiting.
        planning_depth_m = min(cfg.max_range_m, closest_x_m + .5)
        if post_pass_search:
            planning_depth_m = min(planning_depth_m,
                                   cfg.post_pass_planning_depth_m)
        local = radii <= planning_depth_m
        indices = np.rint((bearings[local] + self.half_fov_rad)
                          / self.angle_step_rad).astype(int)
        indices = np.clip(indices, 0, cfg.scan_bins - 1)
        ranges = np.full(cfg.scan_bins, cfg.max_range_m)
        np.minimum.at(ranges, indices, radii[local])

        # Extend each close return by the vehicle half width and margin.
        # Also extend across a range discontinuity, as in the reference FTG.
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

        # Fitted walls define the drivable corridor. Keep every candidate
        # target at least a vehicle half-width plus margin inside each wall.
        horizon_m = min(13., max(cfg.wall_corridor_horizon_m,
                                 min(closest_x_m, planning_depth_m)))
        target_x = horizon_m * np.cos(self.angles)
        target_y = horizon_m * np.sin(self.angles)
        for sign, fit in zip((-1., 1.), wall_fits):
            if fit is None:
                continue
            _, _, slope, offset, curvature, _, _ = fit
            wall_y = curvature * target_x ** 2 + slope * target_x + offset
            if sign < 0:
                blocked |= target_y < wall_y + bubble_m
            else:
                blocked |= target_y > wall_y - bubble_m

        # Near tunnel walls should slow the car and gently bias its next
        # target away from the closer side, even when the map pose drifts.
        sides = points[(points[:, 0] >= 1.) & (points[:, 0] <= 8.)
                       & (np.abs(points[:, 1]) >= cfg.vehicle_half_width_m)]
        def side_clearance(sign):
            side = sides[sides[:, 1] * sign > 0]
            return (float(np.percentile(np.abs(side[:, 1]), 10.))
                    - cfg.vehicle_half_width_m if len(side) >= 5 else math.inf)
        left_clearance = side_clearance(1.)
        right_clearance = side_clearance(-1.)
        wall_bias = (.5 * max(0., .8 - right_clearance)
                     - .5 * max(0., .8 - left_clearance))
        if wall_follow and wall is not None:
            wall_error_m = (self.right_wall_target_clearance_m
                            - self.right_wall_filtered_clearance_m)
            wall_error_m = math.copysign(
                max(0., abs(wall_error_m) - cfg.right_wall_clearance_tolerance_m),
                wall_error_m)
            wall_bearing = float(np.clip(
                cfg.right_wall_offset_gain_rad_per_m * wall_error_m
                + cfg.right_wall_heading_gain * wall_heading,
                -cfg.right_wall_max_correction_rad,
                cfg.right_wall_max_correction_rad))
            # A large difference between the mapped route and the measured
            # boundary is evidence of map-pose drift, not an exit cue.
            if (avoidance_phase != 'PASSING'
                    and abs(math.atan2(math.sin(preferred_bearing_rad - wall_bearing),
                                       math.cos(preferred_bearing_rad - wall_bearing)))
                    > cfg.right_wall_route_disagreement_rad):
                wall_follow_weight = max(wall_follow_weight,
                                         cfg.right_wall_drift_weight)
            # A drifted map bearing must not cancel the measured wall
            # clearance correction inside the tunnel. Hand control back to
            # the saved route as the exit waypoint approaches.
            preferred_bearing_rad = ((1. - wall_follow_weight) * preferred_bearing_rad
                                     + wall_follow_weight * wall_bearing)
        if avoidance_phase == 'PASSING':
            # Hold the selected passing side until the object's far face is
            # behind our rear; otherwise the route can pull us across it.
            if self.passing_side > 0:
                preferred_bearing_rad = max(0., preferred_bearing_rad)
            elif self.passing_side < 0:
                preferred_bearing_rad = min(0., preferred_bearing_rad)
        preferred_bearing_rad = max(-steering_limit_rad,
                                    min(steering_limit_rad,
                                        preferred_bearing_rad + wall_bias))
        if avoidance_phase == 'PASSING' and self.passing_side:
            preferred_bearing_rad = (max(0., preferred_bearing_rad)
                                     if self.passing_side > 0 else
                                     min(0., preferred_bearing_rad))

        steer_left = int(np.searchsorted(self.angles, -steering_limit_rad, side='left'))
        steer_right = int(np.searchsorted(self.angles, steering_limit_rad, side='right')) - 1
        # Once a side is selected, never swap across the obstacle while it
        # is still in front of (or alongside) the vehicle. If that side has
        # no gap, the existing no-gap stop is safer than an abrupt reversal.
        if avoidance_phase == 'PASSING' and self.passing_side:
            center = int(np.searchsorted(self.angles, 0.))
            if self.passing_side > 0:
                steer_left = max(steer_left, center)
            else:
                steer_right = min(steer_right, center)
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
            gap_options.append((score, angle_rad, width_m))
        if not gap_options:
            # At a curved tunnel portal, distant wall returns can fill the
            # entire angular grid even though the near corridor is clear.
            # Creep only with a measured side wall and enough distance to
            # stop from the *current* speed. A near blocked road still stops.
            if (wall_follow and self.passing_side == 0
                    and wall is not None
                    and wall_clearance >= cfg.right_wall_min_clearance_m
                    and front_clearance_m > max(
                        10., stopping_distance_m + cfg.side_margin_m + 1.)):
                # A distant return alone is not a committed pass. Release
                # the provisional PASSING state so wall guidance resumes.
                self.avoidance_active = False
                self.rear_confirm_scans = 0
                avoidance_phase = 'WALL_FOLLOW'
                wall_follow_weight = 1.
                bearing_rad = float(np.clip(
                    wall_bearing, -cfg.unwalled_route_bearing_limit_rad,
                    cfg.unwalled_route_bearing_limit_rad))
                self.previous_bearing_rad = bearing_rad
                crawl_speed = min(cfg.no_return_speed_mps,
                                  self._safe_speed_for_clearance(
                                      front_clearance_m - cfg.side_margin_m,
                                      cfg.reaction_s, cfg.brake_mps2))
                return Plan(crawl_speed, bearing_rad,
                            'distant returns; wall-guided crawl',
                            front_clearance_m, 0., obstacle_points,
                            front_points, nearest_front_x_m,
                            wall_clearance, self.right_wall_target_clearance_m,
                            wall_heading, wall_follow_weight,
                            route_rejoin_remaining_m, avoidance_phase,
                            rear_obstacle_points, self.rear_confirm_scans,
                            rear_scan_points)
            return stop('no gap fits vehicle width', front_clearance_m)
        _, angle_rad, gap_width_m = max(gap_options, key=lambda item: item[0])
        bearing_rad = angle_rad
        self.previous_bearing_rad = bearing_rad
        if avoidance_phase == 'PASSING' and len(forward) and self.passing_side == 0:
            if abs(bearing_rad) >= math.radians(3.):
                self.passing_side = 1 if bearing_rad > 0. else -1
        steering_fraction = abs(bearing_rad) / steering_limit_rad
        # Continuous steering cap avoids abrupt 18 -> 14.4 -> 8.1 km/h
        # target jumps as a LiDAR bearing crosses a bucket boundary.
        speed_fraction = float(np.interp(steering_fraction,
                                        [0., .1, .35, .7, 1.],
                                        [1., .8, .45, .3, .3]))
        target_speed_mps = cfg.max_speed_mps * speed_fraction
        if post_pass_search:
            target_speed_mps = min(target_speed_mps, cfg.no_return_speed_mps)
        near_wall_clearance = min(left_clearance, right_clearance)
        if math.isfinite(near_wall_clearance):
            wall_fraction = float(np.clip(
                (near_wall_clearance - cfg.wall_slow_clearance_m) /
                (cfg.wall_fast_clearance_m - cfg.wall_slow_clearance_m), 0., 1.))
            target_speed_mps = min(target_speed_mps,
                                   cfg.no_return_speed_mps + wall_fraction *
                                   (cfg.max_speed_mps - cfg.no_return_speed_mps))
        if wall_follow and wall_clearance is not None \
                and wall_clearance < cfg.right_wall_min_clearance_m:
            target_speed_mps = min(target_speed_mps, 2.)
        # Rear-quarter returns are included in the avoidance scan but cannot
        # certify clear distance for forward acceleration.
        observed_range_m = float(np.percentile(forward_radii, 95.))
        safe_clearance_m = min(front_clearance_m, observed_range_m -
                               front_bumper_from_lidar_m) - cfg.side_margin_m
        target_speed_mps = min(target_speed_mps,
                               self._safe_speed_for_clearance(
                                   safe_clearance_m, cfg.reaction_s,
                                   cfg.brake_mps2))
        if len(forward) and front_clearance_m <= stopping_distance_m + cfg.side_margin_m:
            # Brake while preserving a safe steering target. A zero-speed
            # plan suppresses the target publisher and loses that option.
            target_speed_mps = min(target_speed_mps, 2.)
        if wall_follow:
            # Slow for alignment with the original mapped route at the exit.
            target_speed_mps = min(target_speed_mps,
                                   5. + wall_follow_weight *
                                   max(0., cfg.max_speed_mps - 5.))
        return Plan(target_speed_mps, bearing_rad, 'gap fits', front_clearance_m,
                    gap_width_m, obstacle_points, front_points, nearest_front_x_m,
                    wall_clearance, self.right_wall_target_clearance_m, wall_heading,
                    wall_follow_weight, route_rejoin_remaining_m,
                    avoidance_phase, rear_obstacle_points,
                    self.rear_confirm_scans, rear_scan_points,
                    tuple(float(item[1]) for item in gap_options))
