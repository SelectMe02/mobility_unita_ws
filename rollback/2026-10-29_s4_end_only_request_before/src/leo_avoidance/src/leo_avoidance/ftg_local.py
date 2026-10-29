"""PointCloud2-adapted Follow The Gap, based on roboracer_unita_ws FTG.

Coordinates are metres in the LiDAR frame: +x forward, +y left, +z up.
The reference's disparity bubble and metric gap width are retained. This
version stops when no gap fits the MORAI vehicle, instead of choosing a beam
through an undersized opening.
"""
from dataclasses import dataclass, replace
import math
import time

import numpy as np

from leo_avoidance.ftg_visualization import cluster_boxes
from leo_avoidance.obstacle_tracker import ObstacleTracker


@dataclass(frozen=True)
class Config:
    mapped_corridor_observation: bool = False
    lidar_to_rear_x_m: float = 1.4
    vehicle_half_width_m: float = 0.946
    side_margin_m: float = 0.4
    bubble_margin_m: float = 0.25
    gap_target_inset_m: float = 0.0
    bearing_smoothing_tau_s: float = 0.0
    wall_target_clear_front_m: float = 20.0
    wall_target_clear_rear_m: float = 4.0
    wall_target_clear_scans: int = 3
    front_bumper_from_rear_m: float = 3.845
    ego_self_filter_inset_m: float = 0.15
    max_range_m: float = 15.0
    local_planning_range_m: float = 9.0
    front_fov_deg: float = 270.0
    scan_bins: int = 541
    disparity_m: float = 0.5
    max_target_bearing_rad: float = math.radians(35.)
    post_pass_target_bearing_rad: float = math.radians(50.)
    post_pass_clear_front_m: float = 15.0
    post_pass_planning_depth_m: float = 20.0
    passing_gap_width_weight: float = 0.35
    min_active_pass_bearing_rad: float = math.radians(10.)
    pass_lateral_shift_m: float = 0.7
    passing_wall_follow_grace_scans: int = 2
    pass_handoff_crawl_min_clearance_m: float = 7.0
    target_distance_from_lidar_m: float = 4.0
    swept_path_horizon_m: float = 4.0
    swept_side_margin_m: float = 0.4
    max_speed_mps: float = 1.5
    no_return_speed_mps: float = 2.0
    # Below right_wall_min_clearance_m during wall following.
    wall_min_clearance_speed_mps: float = 2.0
    # Obstacle inside the stopping-distance clamp, steering target kept.
    near_obstacle_speed_mps: float = 2.0
    # Wall-following floor while the route bearing is not yet trusted.
    wall_follow_min_speed_mps: float = 5.0
    min_no_return_evidence_points: int = 10
    obstacle_speed_hold_s: float = 1.0
    target_accel_mps2: float = 0.8
    reaction_s: float = 0.4
    brake_mps2: float = 2.0
    emergency_brake_mps2: float = 6.0
    wall_slow_clearance_m: float = 0.5
    wall_fast_clearance_m: float = 1.4
    expected_ground_z_m: float = -1.23
    ground_height_tolerance_m: float = 0.25
    collision_min_z_m: float = -1.13
    collision_max_z_m: float = 0.8
    right_wall_target_clearance_m: float = 2.9
    right_wall_clearance_tolerance_m: float = 0.1
    wall_point_tolerance_m: float = 0.35
    exit_wall_trigger_inset_m: float = 0.8
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
    return_heading_tolerance_rad: float = math.radians(5.)
    return_wall_clearance_tolerance_m: float = 0.35
    right_wall_max_correction_rad: float = 0.22
    right_wall_route_disagreement_rad: float = math.radians(15.)
    right_wall_drift_weight: float = 0.9
    unwalled_route_bearing_limit_rad: float = math.radians(7.)
    track_gate_m: float = 1.5
    track_history_s: float = 0.6
    track_min_span_s: float = 0.2
    track_lost_s: float = 0.5
    dynamic_min_speed_mps: float = 0.8
    dynamic_release_speed_mps: float = 0.4
    dynamic_confirm_scans: int = 2
    crossing_min_lateral_speed_mps: float = 0.5
    crossing_horizon_s: float = 3.0
    crossing_lookahead_m: float = 20.0
    crossing_margin_m: float = 0.5
    crossing_confirm_scans: int = 3
    track_max_size_jump_m: float = 0.5

    def validate(self):
        values = tuple(value for value in vars(self).values() if isinstance(value, (float, int)))
        if not all(math.isfinite(value) for value in values):
            raise ValueError('non-finite avoidance parameter')
        if (self.vehicle_half_width_m <= 0 or self.side_margin_m < 0
                or self.gap_target_inset_m < 0 or self.bearing_smoothing_tau_s < 0
                or not 0 <= self.bubble_margin_m <= self.side_margin_m
                or self.wall_target_clear_front_m <= 0
                or self.wall_target_clear_rear_m <= 0
                or self.wall_target_clear_scans < 1
                or self.front_bumper_from_rear_m <= self.lidar_to_rear_x_m
                or self.ego_self_filter_inset_m < 0
                or self.ego_self_filter_inset_m >= self.vehicle_half_width_m
                or 2 * self.ego_self_filter_inset_m >= self.front_bumper_from_rear_m
                or not 0 < self.front_fov_deg <= 360 or self.scan_bins < 5
                or self.max_range_m <= 0 or self.local_planning_range_m <= 0
                or self.disparity_m <= 0
                or self.max_target_bearing_rad <= 0 or self.max_speed_mps <= 0
                or self.post_pass_target_bearing_rad < self.max_target_bearing_rad
                or self.post_pass_target_bearing_rad > math.radians(self.front_fov_deg / 2.)
                or self.post_pass_clear_front_m <= 0
                or self.post_pass_planning_depth_m <= 0
                or self.passing_gap_width_weight < 0
                or not 0 <= self.min_active_pass_bearing_rad <= self.max_target_bearing_rad
                or self.pass_lateral_shift_m <= 0
                or self.passing_wall_follow_grace_scans < 0
                or self.pass_handoff_crawl_min_clearance_m <= 0
                or self.target_distance_from_lidar_m <= 0
                or self.swept_path_horizon_m <= 0
                or not 0 <= self.swept_side_margin_m <= self.side_margin_m
                or self.no_return_speed_mps <= 0
                or self.wall_min_clearance_speed_mps <= 0
                or self.near_obstacle_speed_mps <= 0
                or self.wall_follow_min_speed_mps <= 0
                or self.min_no_return_evidence_points < 1
                or self.obstacle_speed_hold_s < 0 or self.target_accel_mps2 <= 0
                or self.reaction_s < 0 or self.brake_mps2 <= 0
                or self.emergency_brake_mps2 < self.brake_mps2
                or self.wall_slow_clearance_m < 0
                or self.wall_fast_clearance_m <= self.wall_slow_clearance_m
                or self.expected_ground_z_m >= 0
                or self.ground_height_tolerance_m <= 0
                or self.collision_min_z_m >= self.collision_max_z_m
                or self.right_wall_target_clearance_m < self.right_wall_min_clearance_m
                or self.right_wall_clearance_tolerance_m < 0
                or self.wall_point_tolerance_m <= 0
                or self.exit_wall_trigger_inset_m < self.wall_point_tolerance_m
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
                or self.return_heading_tolerance_rad <= 0
                or self.return_wall_clearance_tolerance_m < 0
                or self.right_wall_max_correction_rad < 0
                or self.right_wall_route_disagreement_rad <= 0
                or not 0 <= self.right_wall_drift_weight <= 1
                or self.unwalled_route_bearing_limit_rad <= 0
                or self.track_gate_m <= 0 or self.track_lost_s <= 0
                or not 0 < self.track_min_span_s <= self.track_history_s
                or not 0 < self.dynamic_release_speed_mps
                < self.dynamic_min_speed_mps
                or self.dynamic_confirm_scans < 1
                or self.crossing_min_lateral_speed_mps <= 0
                or self.crossing_horizon_s <= 0
                or self.crossing_lookahead_m <= 0
                or self.crossing_margin_m < 0
                or self.crossing_confirm_scans < 1
                or self.track_max_size_jump_m <= 0):
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
    swept_rejection: dict = None
    # Zero speed because the scan shows an obstacle, not a sensor/plan fault.
    obstacle_stop: bool = False
    # (id, x, y, vx, vy, dynamic, lateral_scans) per cluster, LiDAR frame.
    tracked_obstacles: tuple = ()
    crossing_threats: int = 0


class FollowTheGap:
    def __init__(self, config=None, clock=None):
        self._clock = clock or time.monotonic
        self._last_speed_time = None
        self._last_speed_target = None
        self._held_obstacle_cap = math.inf
        self._obstacle_cap_until = -math.inf
        # (first_x, last_x) of the fitted right wall in the LiDAR frame for
        # the latest scan, or None. Read-only evidence for the tunnel portal.
        self.right_wall_span_m = None
        self._current_obstacle_cap = math.inf
        self.config = config or Config()
        self.config.validate()
        self.half_fov_rad = math.radians(self.config.front_fov_deg / 2.0)
        self.angles = np.linspace(-self.half_fov_rad, self.half_fov_rad,
                                  self.config.scan_bins)
        self.angle_step_rad = self.angles[1] - self.angles[0]
        self.previous_bearing_rad = 0.0
        self._previous_bearing_yaw = None
        self.right_wall_target_clearance_m = None
        self.right_wall_filtered_clearance_m = None
        self.avoidance_active = False
        self.rear_confirm_scans = 0
        self.passing_side = 0
        self.rear_confirmation_armed = True
        self.rear_seen_during_pass = False
        self.rear_disappeared_scans = 0
        self.clear_corridor_scans = 0
        self.provisional_clear_scans = 0
        self.returning_to_heading = False
        self.pass_start_side_wall_clearance_m = None
        self.pass_start_opposite_wall_clearance_m = None
        self.wall_follow_drop_scans = 0
        self.wall_target_clear_count = 0
        self.wall_target_allowed = False
        cfg = self.config
        self.tracker = ObstacleTracker(
            cfg.lidar_to_rear_x_m, gate_m=cfg.track_gate_m,
            history_s=cfg.track_history_s, min_span_s=cfg.track_min_span_s,
            dynamic_speed_mps=cfg.dynamic_min_speed_mps,
            static_speed_mps=cfg.dynamic_release_speed_mps,
            confirm_scans=cfg.dynamic_confirm_scans, lost_s=cfg.track_lost_s,
            region_x_m=(-cfg.lidar_to_rear_x_m - 2.,
                        max(cfg.crossing_lookahead_m, 5.)),
            lateral_speed_mps=cfg.crossing_min_lateral_speed_mps,
            max_size_jump_m=cfg.track_max_size_jump_m)
        self._ego_pose = None
        self._scan_stamp = None
        self.debug_capture = False
        self.debug_snapshot = None

    def _capture_yaw(self):
        pose = self._ego_pose
        if (pose is None or len(pose) != 3
                or not all(math.isfinite(value) for value in pose)):
            return None
        return pose[2]

    def _remember_bearing(self, bearing_rad):
        self.previous_bearing_rad = bearing_rad
        # Stops retain the last selected target, so its frame must also stay
        # at the capture that produced that target, not at the stopped scan.
        self._previous_bearing_yaw = self._capture_yaw()

    def _bearing_in_current_frame(self):
        yaw = self._capture_yaw()
        if yaw is None or self._previous_bearing_yaw is None:
            return self.previous_bearing_rad
        angle = self.previous_bearing_rad + self._previous_bearing_yaw - yaw
        return math.atan2(math.sin(angle), math.cos(angle))

    def reset_wall_target(self):
        self.right_wall_target_clearance_m = None
        self.right_wall_filtered_clearance_m = None
        self.avoidance_active = False
        self.rear_confirm_scans = 0
        self.passing_side = 0
        self.rear_confirmation_armed = True
        self.rear_seen_during_pass = False
        self.rear_disappeared_scans = 0
        self.clear_corridor_scans = 0
        self.provisional_clear_scans = 0
        self.returning_to_heading = False
        self.pass_start_side_wall_clearance_m = None
        self.pass_start_opposite_wall_clearance_m = None
        self.wall_follow_drop_scans = 0
        self.wall_target_clear_count = 0
        self.wall_target_allowed = False

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

    def _swept_body_clear(self, points, bearing_rad):
        """Check the entire car, including rear corners, while beginning a turn."""
        return self._swept_body_collision(points, bearing_rad) is None

    def _swept_body_collision(self, points, bearing_rad, horizon_m=None):
        """Return the first colliding scan point and car pose for diagnostics."""
        cfg = self.config
        horizon_m = cfg.swept_path_horizon_m if horizon_m is None else horizon_m
        target_x = (cfg.lidar_to_rear_x_m
                    + cfg.target_distance_from_lidar_m * math.cos(bearing_rad))
        target_y = cfg.target_distance_from_lidar_m * math.sin(bearing_rad)
        curvature = 2. * target_y / (target_x * target_x + target_y * target_y)
        half_width = cfg.vehicle_half_width_m + cfg.swept_side_margin_m
        nearby = points[(points[:, 0] >= -cfg.lidar_to_rear_x_m - 1.)
                        & (points[:, 0] <= horizon_m
                           + cfg.front_bumper_from_rear_m)
                        & (np.abs(points[:, 1]) <= half_width
                           + horizon_m)]
        for travel in np.arange(0., horizon_m + .125, .25):
            yaw = curvature * travel
            if abs(curvature) < 1e-6:
                rear_x = travel - cfg.lidar_to_rear_x_m
                rear_y = 0.
            else:
                rear_x = math.sin(yaw) / curvature - cfg.lidar_to_rear_x_m
                rear_y = (1. - math.cos(yaw)) / curvature
            dx = nearby[:, 0] - rear_x
            dy = nearby[:, 1] - rear_y
            along = math.cos(yaw) * dx + math.sin(yaw) * dy
            across = -math.sin(yaw) * dx + math.cos(yaw) * dy
            colliding = np.flatnonzero(
                (along >= 0.) & (along <= cfg.front_bumper_from_rear_m)
                & (np.abs(across) <= half_width))
            if len(colliding):
                index = int(colliding[0])
                return (nearby[index], float(travel), float(along[index]),
                        float(across[index]))
        return None

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

    def _inside_walls(self, points, walls, inset_m=None):
        """Mask of returns on the road side of both fitted walls.

        The wall filter only removes returns along the measured wall span.
        Past its end, e.g. objects beyond the tunnel exit, extend the wall
        along its end tangent so outside returns do not count as in-lane.
        """
        inside = np.ones(len(points), dtype=bool)
        inset_m = self.config.wall_point_tolerance_m if inset_m is None else inset_m
        for sign, wall in zip((-1., 1.), walls):
            if wall is None:
                continue
            _, _, slope, offset, curvature, first_x, last_x = wall
            x = points[:, 0]
            anchor = np.clip(x, first_x, last_x)
            wall_y = (curvature * anchor ** 2 + slope * anchor + offset
                      + (slope + 2. * curvature * anchor) * (x - anchor))
            # Only extend forward; nearer than the measured span stays in-lane.
            inside &= ((x < first_x)
                       | (sign * (points[:, 1] - wall_y)
                          < -inset_m - 1e-9))
        return inside

    @staticmethod
    def _runs(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
        return list(zip(edges[::2], edges[1::2] - 1))

    @staticmethod
    def _validated_ground_reference(ground_reference_z_m):
        if ground_reference_z_m is None:
            return None
        try:
            reference = float(ground_reference_z_m)
        except (TypeError, ValueError):
            raise ValueError('invalid ground reference')
        if not math.isfinite(reference) or reference >= 0.:
            raise ValueError('invalid ground reference')
        return reference

    def _ground_plane(self, points, reference_z_m):
        """Fit the existing broadly supported road model at its calibrated height."""
        sample = points[(points[:, 0] >= 1.) & (points[:, 0] <= 12.)
                        & (np.abs(points[:, 1]) <= 5.)
                        & (points[:, 2] >= -2.5) & (points[:, 2] <= -.3)]
        if len(sample) < 60:
            return None
        low_limit = np.percentile(sample[:, 2], 40.)
        low = sample[sample[:, 2] <= low_limit]
        bins = np.floor(low[:, 2] / .1).astype(np.int32)
        unique, counts = np.unique(bins, return_counts=True)
        center_z = (unique[np.argmax(counts)] + .5) * .1
        seed = sample[np.abs(sample[:, 2] - center_z) <= .18]
        if (len(seed) < 30 or np.ptp(seed[:, 0]) < 2.
                or np.ptp(seed[:, 1]) < 2.):
            return None
        for _ in range(2):
            design = np.column_stack((seed[:, :2], np.ones(len(seed))))
            plane = np.linalg.lstsq(design, seed[:, 2], rcond=None)[0]
            if (abs(plane[0]) > .12 or abs(plane[1]) > .12
                    or abs(plane[2] - reference_z_m)
                    > self.config.ground_height_tolerance_m):
                return None
            residual = sample[:, 2] - (sample[:, :2] @ plane[:2] + plane[2])
            seed = sample[np.abs(residual) <= .1]
            if (len(seed) < 30 or np.ptp(seed[:, 0]) < 2.
                    or np.ptp(seed[:, 1]) < 2.):
                return None
        return plane

    def _ground_filter(self, points, ground_reference_z_m=None):
        reference = self._validated_ground_reference(ground_reference_z_m)
        plane = self._ground_plane(points, self.config.expected_ground_z_m
                                   if reference is None else reference)
        if plane is None:
            return points, None
        residual = points[:, 2] - (points[:, :2] @ plane[:2] + plane[2])
        return points[np.abs(residual) > .1], plane

    def _remove_ground(self, points, ground_reference_z_m=None):
        """Discard measured road; keep the original ten-centimetre tolerance.

        A per-call reference changes only the road-height anchor. Uncertain
        fits keep all points for the subsequent collision-height check.
        """
        return self._ground_filter(points, ground_reference_z_m)[0]

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

    def _in_collision_height(self, points, ground_reference_z_m=None, ground_plane=None):
        # Tunnel ceilings and the road surface are visible to LiDAR but do
        # not occupy the Ego vehicle's vertical collision envelope.
        reference = self._validated_ground_reference(ground_reference_z_m)
        lower = self.config.collision_min_z_m
        if reference is not None:
            road_height = (reference if ground_plane is None else
                           points[:, :2] @ ground_plane[:2] + ground_plane[2])
            lower = road_height + (self.config.collision_min_z_m
                                   - self.config.expected_ground_z_m)
        return points[(points[:, 2] >= lower)
                      & (points[:, 2] <= self.config.collision_max_z_m)]

    def _remove_ego_returns(self, points):
        """Discard LiDAR reflections strictly inside the physical Ego body."""
        cfg = self.config
        inset = cfg.ego_self_filter_inset_m
        inside = ((points[:, 0] > -cfg.lidar_to_rear_x_m + inset)
                  & (points[:, 0] < cfg.front_bumper_from_rear_m
                     - cfg.lidar_to_rear_x_m - inset)
                  & (np.abs(points[:, 1]) < cfg.vehicle_half_width_m - inset))
        return points[~inside]

    def obstacle_candidates(self, points_lidar_m, ground_reference_z_m=None):
        """Exactly the cloud returns FTG considers as obstacle hits."""
        points = np.asarray(points_lidar_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('cloud must contain XYZ points')
        reference = self._validated_ground_reference(ground_reference_z_m)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return points
        points, plane = self._ground_filter(points, reference)
        candidates = self._in_avoidance_sector(
            self._remove_ego_returns(
                self._in_collision_height(points, reference, plane)))
        return self._track_obstacles(candidates)[0]

    def rear_obstacle_candidates(self, points_lidar_m, ground_reference_z_m=None):
        """Filtered returns that can confirm completion of an active pass."""
        points = np.asarray(points_lidar_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('cloud must contain XYZ points')
        reference = self._validated_ground_reference(ground_reference_z_m)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return points
        points, plane = self._ground_filter(points, reference)
        return self._rear_obstacle_points(
            self._remove_ego_returns(
                self._in_collision_height(points, reference, plane)))

    def _body_intrusion_threats(self, tracked, ego_speed_mps):
        """Predict simultaneous body occupancy, including intrusion from beside us.

        Tracker velocities are world motion expressed in the current LiDAR
        axes: subtract Ego speed for longitudinal relative motion. Unconfirmed
        motion only qualifies within 1.2 seconds and with a reliable signed
        lateral observation. Unknown velocity is not evidence of approach.
        """
        cfg = self.config
        front = cfg.front_bumper_from_rear_m - cfg.lidar_to_rear_x_m
        # Include a one-metre rear-axle buffer for the rear body, rather than
        # stopping the occupancy envelope at the axle itself.
        rear = -cfg.lidar_to_rear_x_m - 1.
        threats = []
        for track in tracked:
            if (not track.velocity_known or track.lateral_scans == 0
                    or abs(track.vy_mps) < cfg.crossing_min_lateral_speed_mps
                    or track.y_m * track.vy_mps >= 0.):
                continue
            confirmed = (track.dynamic and
                         abs(track.lateral_scans) >= cfg.crossing_confirm_scans)
            horizon = cfg.crossing_horizon_s if confirmed else min(1.2, cfg.crossing_horizon_s)
            half_x = track.size_x_m / 2.
            half_y = (cfg.vehicle_half_width_m + cfg.crossing_margin_m
                      + track.size_y_m / 2.)
            vx = track.vx_mps - ego_speed_mps
            # A fully passed obstacle that cannot catch us must not brake us.
            if track.x_m + half_x < rear and vx <= 0.:
                continue
            entry, leave = 0., horizon
            for position, velocity, low, high in (
                    (track.x_m, vx, rear - half_x, front + half_x),
                    (track.y_m, track.vy_mps, -half_y, half_y)):
                if abs(velocity) < 1e-6:
                    if not low <= position <= high:
                        leave = -1.
                        break
                else:
                    times = sorted(((low-position)/velocity, (high-position)/velocity))
                    entry, leave = max(entry, times[0]), min(leave, times[1])
            if entry > leave:
                continue
            # Imminent side intrusion needs an immediate stop request, even
            # when the object's longitudinal position is behind the bumper.
            clearance = (0. if entry <= 1.2 else
                         ego_speed_mps * max(0., entry - cfg.reaction_s))
            threats.append((clearance, math.atan2(track.y_m, track.x_m),
                            math.copysign(1., track.vy_mps)))
        return threats

    def _crossing_threats(self, tracked):
        """Moving obstacles predicted inside the Ego lane within the horizon.

        Returns (front clearance at lane entry, current bearing, lateral
        direction) per threat. The lane is held still: the response is to
        yield, so Ego motion toward the crossing point is not credited.
        """
        cfg = self.config
        front_bumper_from_lidar_m = cfg.front_bumper_from_rear_m - cfg.lidar_to_rear_x_m
        threats = []
        for track in tracked:
            # One noisy lateral estimate must not stop the car: require the
            # same crossing direction over several consecutive scans.
            if (not track.dynamic or not track.velocity_known
                    or abs(track.lateral_scans) < cfg.crossing_confirm_scans):
                continue
            half_lane_m = (cfg.vehicle_half_width_m + cfg.crossing_margin_m
                           + track.size_y_m / 2.)
            entry_s = (-math.copysign(half_lane_m, track.vy_mps)
                       - track.y_m) / track.vy_mps
            exit_s = (math.copysign(half_lane_m, track.vy_mps)
                      - track.y_m) / track.vy_mps
            if exit_s < 0. or entry_s > cfg.crossing_horizon_s:
                continue
            entry_s = max(0., entry_s)
            entry_x_m = track.x_m + track.vx_mps * entry_s
            near_x_m = entry_x_m - track.size_x_m / 2.
            # Yield only to crossings ahead of the front bumper. Stopping
            # cannot avoid an object entering beside the car; it would only
            # brake hard next to an obstacle that is being passed.
            if (near_x_m < front_bumper_from_lidar_m
                    or near_x_m > cfg.crossing_lookahead_m):
                continue
            threats.append((near_x_m - front_bumper_from_lidar_m,
                            math.atan2(track.y_m, track.x_m),
                            math.copysign(1., track.vy_mps)))
        return threats

    def plan(self, points_lidar_m, ego_speed_mps, preferred_bearing_rad=0.,
             wall_follow=False, wall_follow_weight=1., route_rejoin_remaining_m=None,
             ego_pose=None, stamp=None, route_trusted=False, exit_heading_error_rad=None,
             ground_reference_z_m=None):
        """Plan one scan. ego_pose is the odometry rear-axle pose (x, y, yaw)
        at the scan stamp; without it obstacles are not tracked over time.

        route_trusted means preferred_bearing_rad comes from a map pose that
        an independent observation has corrected (the MORAI exit anchor), so
        the guards against a drifted LiDAR map pose are not applied.

        exit_heading_error_rad is a waypoint direction minus the captured
        heading sensor yaw, supplied only within the configured exit window.
        It replaces the wall heading gradually via wall_follow_weight while
        retaining the measured wall-spacing correction and all collision checks.

        ground_reference_z_m is an optional per-scan road-height calibration.
        When present, the lower collision envelope follows the measured plane
        (or this reference when no plane is supported), keeping its old height
        above road. It is never saved for a subsequent scan.
        """
        now = self._clock()
        # Optional observation of this exact pass through the filters. The
        # ROS adapter enables it only while debug consumers are connected.
        self.debug_snapshot = None
        self._ego_pose = ego_pose
        self._scan_stamp = now if stamp is None else stamp
        self._bearing_dt = (None if self._last_speed_time is None else
                            max(0., min(.25, now - self._last_speed_time)))
        self._current_obstacle_cap = math.inf
        plan = self._plan_geometry(points_lidar_m, ego_speed_mps,
                                   preferred_bearing_rad, wall_follow,
                                   wall_follow_weight, route_rejoin_remaining_m,
                                   route_trusted, exit_heading_error_rad,
                                   ground_reference_z_m)
        cap = self._current_obstacle_cap
        if math.isfinite(cap):
            cap = min(cap, plan.speed_mps)
            if now >= self._obstacle_cap_until:
                self._held_obstacle_cap = cap
                self._obstacle_cap_until = now + self.config.obstacle_speed_hold_s
            elif cap <= self._held_obstacle_cap:
                self._held_obstacle_cap = cap
                self._obstacle_cap_until = now + self.config.obstacle_speed_hold_s
        target = plan.speed_mps
        if now < self._obstacle_cap_until:
            target = min(target, self._held_obstacle_cap)
        if self._last_speed_target is not None:
            # Bound acceleration even after a scan gap; never delay braking.
            dt = max(0., min(.25, now - self._last_speed_time))
            target = min(target, self._last_speed_target + self.config.target_accel_mps2 * dt)
        self._last_speed_time = now
        self._last_speed_target = target
        # A geometric go held at zero by a recent obstacle cap is still an
        # obstacle stop, not a planning fault.
        obstacle_stop = plan.obstacle_stop or (
            target <= 0. < plan.speed_mps and now < self._obstacle_cap_until
            and self._held_obstacle_cap <= 0.)
        return replace(plan, speed_mps=target, obstacle_stop=obstacle_stop)

    def _plan_geometry(self, points_lidar_m, ego_speed_mps, preferred_bearing_rad=0.,
             wall_follow=False, wall_follow_weight=1.,
             route_rejoin_remaining_m=None, route_trusted=False,
             exit_heading_error_rad=None, ground_reference_z_m=None):
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
        swept_rejection = None
        swept_rejected_angles = 0
        avoidance_phase = 'WALL_FOLLOW' if wall_follow else 'ROUTE'
        if not wall_follow:
            self.reset_wall_target()
        wall_follow_weight = float(wall_follow_weight) if wall_follow else 0.
        if not math.isfinite(wall_follow_weight) or not 0. <= wall_follow_weight <= 1.:
            return Plan(0., 0., 'invalid wall follow weight', 0., 0.)
        self.wall_target_clear_count = 0
        self.wall_target_allowed = False
        self.right_wall_span_m = None
        tracks_summary = ()
        threats = []
        stop = lambda reason, clearance=0.0, obstacle=False: Plan(
            0.0, 0.0, reason, clearance, 0.0,
            obstacle_points, front_points, nearest_front_x_m,
            wall_clearance, self.right_wall_target_clearance_m, wall_heading,
            wall_follow_weight, route_rejoin_remaining_m,
            avoidance_phase, rear_obstacle_points, self.rear_confirm_scans,
            rear_scan_points,
            tuple(float(item[1]) for item in gap_options),
            swept_rejection, obstacle, tracks_summary, len(threats))
        points = np.asarray(points_lidar_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            return stop('invalid cloud shape')
        if not math.isfinite(ego_speed_mps) or ego_speed_mps < 0:
            return stop('invalid Ego speed')
        if not math.isfinite(preferred_bearing_rad):
            return stop('invalid route bearing')
        try:
            reference = self._validated_ground_reference(ground_reference_z_m)
        except ValueError:
            return stop('invalid ground reference')
        if exit_heading_error_rad is not None and not math.isfinite(exit_heading_error_rad):
            return stop('invalid exit heading')
        exit_rejoin = wall_follow and exit_heading_error_rad is not None
        direction_trusted = route_trusted or exit_rejoin
        if exit_rejoin:
            preferred_bearing_rad = exit_heading_error_rad
        preferred_bearing_rad = max(-self.half_fov_rad,
                                    min(self.half_fov_rad, preferred_bearing_rad))
        if len(points) == 0:
            return stop('empty cloud')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return stop('no finite points')
        before_ground = len(points)
        points, plane = self._ground_filter(points, reference)
        after_ground = len(points)
        points = self._in_collision_height(points, reference, plane)
        after_height = len(points)
        points = self._remove_ego_returns(points)
        if getattr(self, 'debug_capture', False):
            self.debug_snapshot = dict(
                points=points, ground_removed_points=before_ground-after_ground,
                height_removed_points=after_ground-after_height,
                ego_removed_points=after_height-len(points),
                ground_reference_z_m=(self.config.expected_ground_z_m
                                      if reference is None else reference),
                ground_reference_override=reference is not None,
                ground_plane=None if plane is None else plane.tolist(),
                collision_height_basis=('fixed_lidar_z' if reference is None else
                                        'measured_ground_plane' if plane is not None else
                                        'ground_reference'),
                collision_min_height_above_ground_m=(self.config.collision_min_z_m
                                                     - self.config.expected_ground_z_m),
                collision_min_z_at_origin_m=(self.config.collision_min_z_m
                    if reference is None else (reference if plane is None else plane[2])
                    + self.config.collision_min_z_m - self.config.expected_ground_z_m),
                collision_max_z_m=self.config.collision_max_z_m)
        if len(points) == 0:
            return stop('no points at collision height')
        rear_scan_points = int(np.count_nonzero(
            (points[:, 0] < 0.)
            & (np.hypot(points[:, 0], points[:, 1]) <=
               cfg.rear_detection_max_range_m)))
        rear_points = self._rear_obstacle_points(points)
        rear_obstacle_points = len(rear_points)
        # Rear returns remain collision evidence, never a phase transition trigger.
        body_scan_points = points

        candidates = self._in_avoidance_sector(points)
        if self.debug_snapshot is not None:
            self.debug_snapshot['forward_sector_points'] = len(candidates)
        if not len(candidates):
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
                           if wall_follow and not direction_trusted else
                           float(np.clip(preferred_bearing_rad,
                           -cfg.max_target_bearing_rad,
                           cfg.max_target_bearing_rad)))
            self._remember_bearing(bearing_rad)
            return Plan(min(cfg.max_speed_mps, cfg.no_return_speed_mps),
                        bearing_rad, 'no forward returns; cautious travel',
                        0.0, 0.0, obstacle_points,
                        rear_obstacle_points=rear_obstacle_points,
                        rear_scan_points=rear_scan_points,
                        rear_confirm_scans=self.rear_confirm_scans,
                        avoidance_phase=avoidance_phase)
        points = candidates
        obstacles, wall_fits = self._track_obstacles(points)
        if wall_fits[0] is not None:
            self.right_wall_span_m = (wall_fits[0][5], wall_fits[0][6])
        if self.debug_snapshot is not None:
            self.debug_snapshot['tracker_input_points'] = len(obstacles)
            self.debug_snapshot['wall_evidence_points'] = len(points)-len(obstacles)
        obstacle_points = len(obstacles)
        tracked = self.tracker.update(obstacles, self._ego_pose,
                                      self._scan_stamp)
        if self.debug_snapshot is not None:
            self.debug_snapshot['tracked'] = tracked
        tracks_summary = tuple(
            (t.track_id, round(t.x_m, 2), round(t.y_m, 2), round(t.vx_mps, 2),
             round(t.vy_mps, 2), t.dynamic, t.lateral_scans) for t in tracked)
        crossing_tracks = ([t for t in tracked if t.y_m * t.vy_mps < 0.]
                           if wall_follow else tracked)
        threats = self._crossing_threats(crossing_tracks)
        if wall_follow:
            # Tunnel-only supplement: preserve ordinary route behaviour and
            # the existing ahead-of-bumper crossing yield calculation.
            threats.extend(self._body_intrusion_threats(tracked, ego_speed_mps))
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
        elif wall_follow and not direction_trusted:
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
        # Re-evaluate every scan. The 300-degree obstacle cloud constrains
        # candidates; only returns beside or ahead of the car inside the
        # clear zone keep the pass active. Blocks already left behind or
        # beyond wall_target_clear_front_m must not prevent the return to the
        # saved right-wall spacing. No obstacle identity or side latch.
        in_clear_zone = ((np.abs(bearings) <= math.radians(135.))
                         & (obstacles[:, 0] >= -cfg.wall_target_clear_rear_m)
                         & (obstacles[:, 0] <= cfg.wall_target_clear_front_m)
                         & self._inside_walls(obstacles, wall_fits,
                             cfg.exit_wall_trigger_inset_m if exit_rejoin else None))
        occupied = bool(np.any(in_clear_zone))
        self.passing_side = 0
        self.rear_confirm_scans = 0
        self.returning_to_heading = False
        self.avoidance_active = bool(wall_follow and occupied)
        self.wall_target_allowed = bool(wall_follow and not occupied
                                       and wall is not None)
        avoidance_phase = ('PASSING' if self.avoidance_active else
                           'WALL_FOLLOW' if wall_follow else 'ROUTE')
        steering_limit_rad = cfg.max_target_bearing_rad
        if self.avoidance_active:
            wall_follow_weight = 0.
            # Match tunnel heading while selecting any collision-free gap.
            # An anchored route bearing already follows the road, including
            # the bend at the tunnel exit where the wall does not.
            if wall_heading is not None and not direction_trusted:
                preferred_bearing_rad = wall_heading
        stopping_distance_m = (ego_speed_mps * cfg.reaction_s
                               + ego_speed_mps ** 2 / (2 * cfg.brake_mps2))
        # The obstacle speed caps below use brake_mps2, so a car following
        # them sits exactly on the planning stopping distance. Stop outright
        # only when even emergency braking would not stop in time; otherwise
        # the caps brake smoothly and keep the steering target.
        emergency_stopping_m = (ego_speed_mps * cfg.reaction_s + ego_speed_mps ** 2
                                / (2 * cfg.emergency_brake_mps2))
        if len(forward) and front_clearance_m <= cfg.side_margin_m + .5:
            self._current_obstacle_cap = 0.
            return stop('front obstacle inside stopping distance', front_clearance_m,
                        obstacle=True)
        if (len(forward) and closest_x_m > cfg.local_planning_range_m
                and front_clearance_m <= emergency_stopping_m + cfg.side_margin_m):
            self._current_obstacle_cap = 0.
            return stop('distant obstacle inside stopping distance',
                        front_clearance_m, obstacle=True)
        if threats:
            # Yield to a moving obstacle predicted to cross the Ego lane:
            # be able to stop short of where it enters, instead of steering
            # into the side it is moving toward.
            crossing_clearance_m = min(item[0] for item in threats)
            if crossing_clearance_m <= cfg.side_margin_m + .5:
                self._current_obstacle_cap = 0.
                return stop('yielding to crossing obstacle',
                            max(0., crossing_clearance_m), obstacle=True)
            self._current_obstacle_cap = min(
                self._current_obstacle_cap, self._safe_speed_for_clearance(
                    crossing_clearance_m - cfg.side_margin_m,
                    cfg.reaction_s, cfg.brake_mps2))

        # A ray to the far end of a tunnel eventually intersects its wall,
        # even when a curved path around a nearer obstacle is wide enough.
        # Pick the immediate gap near the first obstacle; retain the full
        # scan for wall following, front clearance, and speed limiting.
        planning_depth_m = min(cfg.max_range_m, cfg.local_planning_range_m,
                               closest_x_m + .5)
        # Include side/rear-quarter returns within the 300-degree local
        # window; the full cloud separately protects the swept body.
        local = radii <= planning_depth_m
        indices = np.rint((bearings[local] + self.half_fov_rad)
                          / self.angle_step_rad).astype(int)
        indices = np.clip(indices, 0, cfg.scan_bins - 1)
        ranges = np.full(cfg.scan_bins, cfg.max_range_m)
        np.minimum.at(ranges, indices, radii[local])

        candidate_bubble_m = cfg.vehicle_half_width_m + cfg.bubble_margin_m
        # Candidate inflation is separate from physical swept-body clearance.
        # Also extend across a range discontinuity, as in the reference FTG.
        blocked = np.zeros(cfg.scan_bins, dtype=bool)
        hits = np.flatnonzero(ranges < cfg.max_range_m)
        for index in hits:
            beams = int(math.ceil(math.atan2(candidate_bubble_m, max(0.1, ranges[index]))
                                  / self.angle_step_rad))
            blocked[max(0, index - beams):min(cfg.scan_bins, index + beams + 1)] = True
        for index in np.flatnonzero(np.abs(np.diff(ranges)) > cfg.disparity_m):
            near_index = index if ranges[index] < ranges[index + 1] else index + 1
            beams = int(math.ceil(math.atan2(candidate_bubble_m, max(0.1, ranges[near_index]))
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
        for _, bearing_obstacle_rad, direction in threats:
            # Never pass on the side the crossing obstacle is heading to.
            blocked |= direction * (self.angles - bearing_obstacle_rad) > 0.

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
                cfg.right_wall_offset_gain_rad_per_m *
                (wall_error_m if self.wall_target_allowed else 0.)
                + cfg.right_wall_heading_gain * wall_heading,
                -cfg.right_wall_max_correction_rad,
                cfg.right_wall_max_correction_rad))
            # A large difference between the mapped route and the measured
            # boundary is evidence of map-pose drift, not an exit cue.
            # The road bends at the tunnel exit; once the map pose is
            # anchored, that disagreement is the curve, not drift.
            if (avoidance_phase != 'PASSING' and not direction_trusted
                    and abs(math.atan2(math.sin(preferred_bearing_rad - wall_bearing),
                                       math.cos(preferred_bearing_rad - wall_bearing)))
                    > cfg.right_wall_route_disagreement_rad):
                wall_follow_weight = max(wall_follow_weight,
                                         cfg.right_wall_drift_weight)
            # A drifted map bearing must not cancel the measured wall
            # clearance correction inside the tunnel. Hand control back to
            # the saved route as the exit waypoint approaches.
            if exit_rejoin:
                # Switch only the heading source at the portal. Do not fade
                # out lateral spacing with it, and do not use ICP cross-track
                # position as steering evidence. PASSING still suppresses the
                # spacing correction so interior obstacles can be avoided.
                offset_bearing = float(np.clip(
                    cfg.right_wall_offset_gain_rad_per_m *
                    (wall_error_m if self.wall_target_allowed else 0.),
                    -cfg.right_wall_max_correction_rad, cfg.right_wall_max_correction_rad))
                preferred_bearing_rad = (
                    (1. - wall_follow_weight) * preferred_bearing_rad
                    + wall_follow_weight * cfg.right_wall_heading_gain * wall_heading
                    + offset_bearing)
            else:
                preferred_bearing_rad = ((1. - wall_follow_weight) * preferred_bearing_rad
                                         + wall_follow_weight * wall_bearing)
            if avoidance_phase == 'WALL_FOLLOW' and wall_clearance < cfg.wall_fast_clearance_m:
                # Do not let the mapped route keep steering into a measured
                # right wall while the exit blend reduces wall-follow weight.
                away = min(cfg.right_wall_max_correction_rad,
                           .05 + cfg.right_wall_offset_gain_rad_per_m *
                           (cfg.wall_fast_clearance_m - wall_clearance))
                preferred_bearing_rad = max(preferred_bearing_rad, away)
        preferred_bearing_rad = max(-steering_limit_rad,
                                    min(steering_limit_rad,
                                        preferred_bearing_rad + wall_bias))
        steer_left = int(np.searchsorted(self.angles, -steering_limit_rad, side='left'))
        steer_right = int(np.searchsorted(self.angles, steering_limit_rad, side='right')) - 1
        swept_rejections = 0
        for left, right in self._runs(~blocked):
            left = max(left, steer_left)
            right = min(right, steer_right)
            if left > right:
                continue
            # A finite range cap prevents empty sky from overwhelming the score.
            depth_m = min(6.0, float(np.min(ranges[left:right + 1])))
            width_m = 2.0 * depth_m * math.sin(min(math.pi,
                (right - left + 1) * self.angle_step_rad) / 2.0)
            # Prefer the interior of a gap over grazing its inflated edge.
            # This is a preference: narrow safe gaps remain available.
            inset_bins = min((right - left) // 2, int(math.ceil(
                math.atan2(cfg.gap_target_inset_m, max(.1, depth_m)) /
                self.angle_step_rad))) if np.any(local) else 0
            desired = float(np.clip(preferred_bearing_rad,
                                   self.angles[left + inset_bins],
                                   self.angles[right - inset_bins]))
            # Smooth the preference BEFORE collision checks, never a checked
            # command afterwards. If unsafe, another checked angle wins.
            dt = getattr(self, '_bearing_dt', None)
            if cfg.bearing_smoothing_tau_s > 0. and dt is not None:
                alpha = 1. - math.exp(-dt / cfg.bearing_smoothing_tau_s)
                previous = self.previous_bearing_rad
                if avoidance_phase == 'WALL_FOLLOW' and wall is not None and not exit_rejoin:
                    # A turn already made by Ego is not wall-measurement
                    # noise. Move the old target into this scan's vehicle
                    # frame before smoothing it; keep passing/exit guidance
                    # and the subsequent collision checks unchanged.
                    previous = self._bearing_in_current_frame()
                desired = previous + alpha * (desired - previous)
            preferred_index = int(round((desired + self.half_fov_rad) /
                                        self.angle_step_rad))
            step_bins = max(1, int(round(math.radians(2.) / self.angle_step_rad)))
            choices = set(range(left, right + 1, step_bins))
            choices.update((left, right, max(left, min(right, preferred_index))))
            angle_rad = None
            for index in sorted(choices, key=lambda i: abs(i - preferred_index)):
                candidate_angle = float(self.angles[index])
                collision = self._swept_body_collision(
                    body_scan_points, candidate_angle)
                if collision is None:
                    angle_rad = candidate_angle
                    break
                swept_rejected_angles += 1
                if swept_rejection is None:
                    point, travel, along, across = collision
                    # The collision checker uses all height-filtered returns.
                    # Show whether this exact return survived wall filtering.
                    retained_in_front_obstacles = bool(len(obstacles) and np.any(
                        np.all(np.isclose(obstacles, point, atol=1e-4), axis=1)))
                    swept_rejection = {
                        'candidate_bearing_deg': round(math.degrees(candidate_angle), 2),
                        'point_lidar_xyz_m': [round(float(v), 3) for v in point],
                        'travel_m': round(travel, 2),
                        'body_along_m': round(along, 3),
                        'body_across_m': round(across, 3),
                        'retained_in_front_obstacles': retained_in_front_obstacles,
                    }
            if angle_rad is None:
                swept_rejections += 1
                continue
            width_weight = (cfg.passing_gap_width_weight
                            if avoidance_phase == 'PASSING' and self.passing_side == 0
                            else .02)
            score = (-abs(angle_rad - preferred_bearing_rad)
                     + width_weight * min(width_m, 3.)
                     - .1 * abs(angle_rad - self.previous_bearing_rad))
            gap_options.append((score, angle_rad, width_m))
        if not gap_options:
            # At a curved tunnel portal, distant wall returns can fill the
            # entire angular grid even though the near corridor is clear.
            # Creep only with a measured side wall and enough distance to
            # stop from the *current* speed. A near blocked road still stops.
            if (wall_follow and self.wall_target_allowed and self.passing_side == 0
                    and wall is not None
                    and wall_clearance >= cfg.right_wall_min_clearance_m
                    and self._swept_body_collision(body_scan_points, float(np.clip(
                        wall_bearing, -cfg.unwalled_route_bearing_limit_rad,
                        cfg.unwalled_route_bearing_limit_rad))) is None
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
                self._remember_bearing(bearing_rad)
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
                            rear_scan_points, tracked_obstacles=tracks_summary,
                            crossing_threats=len(threats))
            if swept_rejection is not None:
                swept_rejection['rejected_angles'] = swept_rejected_angles
                swept_rejection['rejected_gaps'] = swept_rejections
            return stop(('yielding to crossing obstacle' if threats
                         else 'rear/body swept path blocked' if swept_rejections
                         else 'no gap fits vehicle width'), front_clearance_m,
                        obstacle=True)
        _, angle_rad, gap_width_m = max(gap_options, key=lambda item: item[0])
        bearing_rad = angle_rad
        self._remember_bearing(bearing_rad)
        # Extend the selected curvature to braking distance, beyond the
        # short candidate feasibility check. Use the full cloud, including walls.
        speed_horizon = min(cfg.max_range_m, max(cfg.swept_path_horizon_m,
            cfg.max_speed_mps * cfg.reaction_s + cfg.max_speed_mps ** 2 /
            (2. * cfg.brake_mps2) + cfg.side_margin_m + .25))
        path_hit = self._swept_body_collision(body_scan_points, bearing_rad, speed_horizon)
        if path_hit is not None:
            self._current_obstacle_cap = min(
                self._current_obstacle_cap, self._safe_speed_for_clearance(
                    max(0., path_hit[1] - .25 - cfg.side_margin_m),
                    cfg.reaction_s, cfg.brake_mps2))
        if len(forward):
            front_cap = self._safe_speed_for_clearance(
                front_clearance_m - cfg.side_margin_m, cfg.reaction_s, cfg.brake_mps2)
            if front_cap < cfg.max_speed_mps:
                self._current_obstacle_cap = min(self._current_obstacle_cap, front_cap)
        steering_fraction = abs(bearing_rad) / steering_limit_rad
        # Continuous steering cap avoids abrupt 18 -> 14.4 -> 8.1 km/h
        # target jumps as a LiDAR bearing crosses a bucket boundary.
        speed_fraction = float(np.interp(steering_fraction,
                                        [0., .1, .35, .7, 1.],
                                        [1., .8, .45, .3, .3]))
        target_speed_mps = cfg.max_speed_mps * speed_fraction
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
            target_speed_mps = min(target_speed_mps, cfg.wall_min_clearance_speed_mps)
        # Rear-quarter returns are included in the avoidance scan but cannot
        # certify clear distance for forward acceleration.
        observed_range_m = float(np.percentile(forward_radii, 95.))
        if cfg.mapped_corridor_observation and direction_trusted:
            # S4: side returns do not measure visibility along the chosen gap.
            along = points[:, 0]*math.cos(bearing_rad)+points[:, 1]*math.sin(bearing_rad)
            across = -points[:, 0]*math.sin(bearing_rad)+points[:, 1]*math.cos(bearing_rad)
            seen = along[(along > 0.) & (abs(across) <= cfg.vehicle_half_width_m+cfg.side_margin_m)]
            if len(seen):
                observed_range_m = float(np.percentile(seen, 95.))
        safe_clearance_m = min(front_clearance_m, observed_range_m -
                               front_bumper_from_lidar_m) - cfg.side_margin_m
        target_speed_mps = min(target_speed_mps,
                               self._safe_speed_for_clearance(
                                   safe_clearance_m, cfg.reaction_s,
                                   cfg.brake_mps2))
        clamp_stopping_m = (stopping_distance_m
                            if closest_x_m <= cfg.local_planning_range_m
                            else emergency_stopping_m)
        if len(forward) and front_clearance_m <= clamp_stopping_m + cfg.side_margin_m:
            # Near obstacles keep the conservative clamp; distant ones brake
            # through the clearance cap instead of dropping to 2 m/s.
            # Brake while preserving a safe steering target. A zero-speed
            # plan suppresses the target publisher and loses that option.
            target_speed_mps = min(target_speed_mps, cfg.near_obstacle_speed_mps)
        if wall_follow:
            # Slow for alignment with the original mapped route at the exit.
            target_speed_mps = min(target_speed_mps,
                                   cfg.wall_follow_min_speed_mps + wall_follow_weight *
                                   max(0., cfg.max_speed_mps - cfg.wall_follow_min_speed_mps))
        target_speed_mps = min(target_speed_mps, self._current_obstacle_cap)
        return Plan(target_speed_mps, bearing_rad,
                    'gap fits; yielding to crossing obstacle' if threats
                    else 'gap fits', front_clearance_m,
                    gap_width_m, obstacle_points, front_points, nearest_front_x_m,
                    wall_clearance, self.right_wall_target_clearance_m, wall_heading,
                    wall_follow_weight, route_rejoin_remaining_m,
                    avoidance_phase, rear_obstacle_points,
                    self.rear_confirm_scans, rear_scan_points,
                    tuple(float(item[1]) for item in gap_options),
                    tracked_obstacles=tracks_summary,
                    crossing_threats=len(threats))
