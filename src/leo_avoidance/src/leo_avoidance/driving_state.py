"""GPS to LiDAR iterative odometry state machine for the existing controller."""
import math

from .scan_odometry import between, compose, wrap


def bounded_pose_correction(estimate, reference, max_translation_m,
                            max_yaw_rad):
    """Move an estimate toward an observed pose without a discontinuous snap."""
    if (not all(math.isfinite(v) for v in (*estimate, *reference,
                                           max_translation_m, max_yaw_rad))
            or max_translation_m <= 0 or max_yaw_rad <= 0):
        raise ValueError('invalid pose correction')
    dx = reference[0] - estimate[0]
    dy = reference[1] - estimate[1]
    distance = math.hypot(dx, dy)
    fraction = min(1., max_translation_m / distance) if distance else 1.
    yaw_delta = max(-max_yaw_rad,
                    min(max_yaw_rad, wrap(reference[2] - estimate[2])))
    return (estimate[0] + fraction * dx,
            estimate[1] + fraction * dy,
            wrap(estimate[2] + yaw_delta))


class DrivingState:
    GPS = 'GPS'
    BLACKOUT = 'BLACKOUT'
    SIM_EXIT = 'SIM_EXIT'
    HOLD = 'HOLD'

    def __init__(self, matcher, input_timeout_s=.35,
                 max_blackout_distance_m=30., max_rejoin_error_m=2.,
                 max_rejoin_yaw_rad=.25, gps_silence_timeout_s=.75,
                 max_unmatched_distance_m=3., gps_return_grace_s=1.5,
                 icp_forward_weight=.25, icp_lateral_weight=.1):
        self.matcher = matcher
        self.input_timeout_s = float(input_timeout_s)
        self.max_blackout_distance_m = float(max_blackout_distance_m)
        self.max_rejoin_error_m = float(max_rejoin_error_m)
        self.max_rejoin_yaw_rad = float(max_rejoin_yaw_rad)
        self.gps_silence_timeout_s = float(gps_silence_timeout_s)
        self.max_unmatched_distance_m = float(max_unmatched_distance_m)
        self.gps_return_grace_s = float(gps_return_grace_s)
        self.icp_forward_weight = float(icp_forward_weight)
        self.icp_lateral_weight = float(icp_lateral_weight)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.input_timeout_s, self.max_blackout_distance_m,
                    self.max_rejoin_error_m, self.max_rejoin_yaw_rad,
                    self.gps_silence_timeout_s,
                    self.max_unmatched_distance_m)):
            raise ValueError('invalid driving state limits')
        if not math.isfinite(self.gps_return_grace_s) or self.gps_return_grace_s <= 0:
            raise ValueError('invalid GPS return grace')
        if not all(math.isfinite(v) and 0. <= v <= 1. for v in
                   (self.icp_forward_weight, self.icp_lateral_weight)):
            raise ValueError('invalid LiDAR odometry fusion weights')
        self.mode = self.HOLD
        self.reason = 'waiting for GPS, odometry and LiDAR'
        self.gps_pose = None
        self.gps_valid = None
        self.raw_gps = None
        self.avoidance = None
        self.odom_pose = None
        self.previous_odom = None
        self.previous_scan = None
        self.map_pose = None
        self.last_scan_wall = None
        self.scan_valid = False
        self.scan_fault_reason = None
        self.travelled_m = 0.
        self.last_match = None
        self.last_rejoin_error = None
        self.had_blackout = False
        self.unmatched_distance_m = 0.
        self.gps_return_since = None

    def update_gps(self, pose, received):
        if pose is None or len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            return
        self.gps_pose = (tuple(pose), received)

    def update_gps_valid(self, valid, received):
        self.gps_valid = (bool(valid), received)

    def update_raw_gps(self, latitude_deg, longitude_deg, received, status=1):
        finite = (math.isfinite(latitude_deg) and math.isfinite(longitude_deg)
                  and -80 <= latitude_deg <= 84 and -180 <= longitude_deg <= 180)
        kind = ('zero' if finite and latitude_deg == longitude_deg == 0.0 else
                'valid' if finite and status > 0 else 'invalid')
        self.raw_gps = (kind, received)
        if kind == 'zero':
            self.gps_return_since = None
        elif kind == 'valid' and self.had_blackout and self.gps_return_since is None:
            self.gps_return_since = received

    def update_avoidance(self, valid, received):
        self.avoidance = (bool(valid), received)

    def update_odom(self, pose, received):
        if pose is None or len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            return
        self.odom_pose = (tuple(pose), received)

    def _fresh(self, sample, now):
        return sample is not None and 0 <= now - sample[1] <= self.input_timeout_s

    def gps_available(self, now):
        return (self._fresh(self.raw_gps, now) and self.raw_gps[0] == 'valid'
                and self._fresh(self.gps_valid, now) and self.gps_valid[0]
                and self._fresh(self.gps_pose, now))

    def gps_zero(self, now):
        return self._fresh(self.raw_gps, now) and self.raw_gps[0] == 'zero'

    def gps_missing(self, now):
        return (self.raw_gps is not None
                and self.raw_gps[0] in ('valid', 'zero')
                and now - self.raw_gps[1] >= self.gps_silence_timeout_s)

    def scan(self, points_xy, now, odom_sample=None,
             allow_gps_anchor=True, gps_anchor_pose=None):
        odom_sample = self.odom_pose if odom_sample is None else odom_sample
        if not self._fresh(odom_sample, now):
            self.scan_valid = False
            self.reason = 'odometry missing at LiDAR scan'
            self.scan_fault_reason = self.reason
            return
        odom = odom_sample[0]
        if (allow_gps_anchor and self.gps_available(now)
                and not self.had_blackout):
            self.previous_scan = points_xy
            self.previous_odom = odom
            self.map_pose = (self.gps_pose[0] if gps_anchor_pose is None
                             else gps_anchor_pose)
            self.last_scan_wall = now
            self.scan_valid = True
            self.scan_fault_reason = None
            self.travelled_m = 0.
            self.unmatched_distance_m = 0.
            return
        if self.previous_scan is None or self.previous_odom is None or self.map_pose is None:
            self.scan_valid = False
            self.reason = 'no GPS-anchored LiDAR scan'
            self.scan_fault_reason = self.reason
            return
        rear_prior = between(self.previous_odom, odom)
        match = self.matcher.match(self.previous_scan, points_xy, rear_prior)
        self.last_match = match
        if not match.valid:
            # Advance the reference with Ego odometry while braking. Keeping
            # the old scan makes every later ICP attempt span a growing gap.
            motion_m = math.hypot(rear_prior[0], rear_prior[1])
            self.unmatched_distance_m += motion_m
            self.travelled_m += motion_m
            if self.unmatched_distance_m <= self.max_unmatched_distance_m:
                self.map_pose = compose(self.map_pose, rear_prior)
            self.previous_scan = points_xy
            self.previous_odom = odom
            self.scan_valid = False
            self.reason = ('LiDAR ICP failed: %d pairs, %.3f m RMSE; %.2f m '
                           'unmatched motion%s' % (match.pairs, match.rmse_m,
                           self.unmatched_distance_m,
                           ' (recovery limit exceeded)' if self.unmatched_distance_m
                           > self.max_unmatched_distance_m else ''))
            self.scan_fault_reason = self.reason
            return
        if self.unmatched_distance_m > self.max_unmatched_distance_m:
            self.scan_valid = False
            self.reason = 'LiDAR ICP recovery exceeded odometry-only distance limit'
            self.scan_fault_reason = self.reason
            return
        matched_delta = self.matcher.rear_delta_from_lidar(match.delta_lidar)
        # Tunnel walls are repetitive: planar ICP can report a small yaw
        # correction at every scan and accumulate a metre-scale lateral bias.
        # Ego heading is observed directly by MORAI's heading sensor. Keep
        # that yaw and most of its lateral motion; use ICP for scan quality
        # and bounded translation correction instead of integrating ICP yaw.
        rear_delta = (
            rear_prior[0] + self.icp_forward_weight *
            (matched_delta[0] - rear_prior[0]),
            rear_prior[1] + self.icp_lateral_weight *
            (matched_delta[1] - rear_prior[1]),
            rear_prior[2])
        self.travelled_m += math.hypot(rear_delta[0], rear_delta[1])
        self.map_pose = compose(self.map_pose, rear_delta)
        self.previous_scan = points_xy
        self.previous_odom = odom
        self.last_scan_wall = now
        self.scan_valid = True
        self.scan_fault_reason = None
        self.unmatched_distance_m = 0.

    def _select_sim_exit(self, pose):
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            self.mode = self.HOLD
            self.reason = 'invalid MORAI exit pose'
            return self.mode, None, self.reason
        self.map_pose = tuple(pose)
        self.mode = self.SIM_EXIT
        self.reason = 'MORAI ENU exit pose; waypoint steering'
        return self.mode, self.map_pose, self.reason

    def select(self, now, blackout_allowed=True, route_rejoin_ok=False,
               sim_exit_pose=None, sim_exit_active=False):
        if self.gps_available(now):
            gps = self.gps_pose[0]
            corroborated = False
            if self.had_blackout and self.map_pose is not None:
                error = between(self.map_pose, gps)
                self.last_rejoin_error = (math.hypot(error[0], error[1]),
                                          abs(wrap(error[2])))
                if (math.hypot(error[0], error[1]) > self.max_rejoin_error_m
                        or abs(wrap(error[2])) > self.max_rejoin_yaw_rad):
                    if not route_rejoin_ok:
                        if not (sim_exit_active and sim_exit_pose is not None):
                            self.mode = self.HOLD
                            self.reason = 'GPS return disagrees with LiDAR odometry'
                            return self.mode, None, self.reason
                        # Keep the exit waypoint path until GPS and the
                        # mapped route agree; a transient GPS pose must not
                        # interrupt the steering handoff.
                        return self._select_sim_exit(sim_exit_pose)
                    corroborated = True
            self.mode = self.GPS
            self.map_pose = gps
            self.travelled_m = 0.
            self.had_blackout = False
            self.gps_return_since = None
            self.reason = ('fresh GPS map pose; route-corroborated rejoin'
                           if corroborated else 'fresh GPS map pose')
            return self.mode, gps, self.reason
        if sim_exit_pose is not None and self.had_blackout:
            return self._select_sim_exit(sim_exit_pose)
        if sim_exit_active and self.had_blackout:
            self.mode = self.HOLD
            self.reason = 'MORAI exit pose missing or stale'
            return self.mode, None, self.reason
        gps_return_pending = (self.had_blackout and self.raw_gps is not None
                              and self.raw_gps[0] == 'valid'
                              and self.gps_return_since is not None
                              and 0 <= now - self.gps_return_since
                              <= self.gps_return_grace_s)
        if self.map_pose is None:
            self.mode = self.HOLD
            self.reason = 'GPS anchor unavailable for blackout'
            return self.mode, None, self.reason
        if not blackout_allowed:
            self.mode = self.HOLD
            self.reason = 'GPS blackout outside configured tunnel section'
            return self.mode, None, self.reason
        if not (self.gps_zero(now) or self.gps_missing(now) or gps_return_pending):
            self.mode = self.HOLD
            self.reason = 'GPS invalid without zero-coordinate or missing-stream trigger'
            return self.mode, None, self.reason
        if (self.map_pose is None or not self.scan_valid
                or self.last_scan_wall is None
                or not 0 <= now - self.last_scan_wall <= self.input_timeout_s
                or not self._fresh(self.odom_pose, now)):
            self.mode = self.HOLD
            self.reason = 'LiDAR/odometry pose missing or stale'
            return self.mode, None, self.reason
        if self.travelled_m > self.max_blackout_distance_m:
            self.mode = self.HOLD
            self.reason = 'blackout travel limit exceeded'
            return self.mode, None, self.reason
        if not self._fresh(self.avoidance, now) or not self.avoidance[0]:
            self.mode = self.HOLD
            self.reason = 'LiDAR avoidance has no fresh safe gap'
            return self.mode, None, self.reason
        self.mode = self.BLACKOUT
        self.had_blackout = True
        self.reason = 'LiDAR ICP and odometry'
        return self.mode, self.map_pose, self.reason
