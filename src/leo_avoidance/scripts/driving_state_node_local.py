#!/usr/bin/env python3
"""Switch existing map-pose consumers from GPS to LiDAR ICP + Ego odometry."""
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from morai_msgs.msg import GPSMessage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool, Float64, String

from leo_avoidance.driving_state_local import DrivingState
from leo_avoidance.ftg_local import Config, FollowTheGap, Plan
from leo_avoidance.scan_odometry import ScanOdometry
from leo_guard.traffic_local import RouteProgress
from leo_guard.route_file import (anchor_on_route, checkpoint_indices,
                                  load_route_points, load_route_speeds)

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from avoidance_runtime import FrozenConfigTarget, RuntimeSettings  # noqa: E402
from perception_debug import PerceptionDebug, tracked_obstacles_map  # noqa: E402


def locate_or_reacquire(route, x, y, yaw):
    """Windowed route projection; once lost, search the whole route again.

    The raceline follower may drive a lane far from Raceline 1 (CP8-CP11), so
    a projection lost there must not stay stuck at the old route index.
    """
    try:
        return route.locate(x, y, yaw)
    except ValueError:
        if route.index is None:
            raise
        route.index = None
        return route.locate(x, y, yaw)


def s14_ground_interval(waypoint_file, route):
    """Exact CP13-to-CP14 interval, including the shared boundary vertex.

    Route point sector labels lose the duplicated first S14 point. Checkpoint
    indices retain its true location and keep this profile out of S13/S15.
    CSV routes have no sector metadata and retain the original filter.
    """
    if Path(waypoint_file).suffix.lower() != '.json':
        return None
    checkpoints = checkpoint_indices(waypoint_file)
    start, end = (float(route.s[checkpoints[name]]) for name in ('13', '14'))
    if not 0. <= start < end <= route.total:
        raise ValueError('invalid S14 ground-filter interval')
    return start, end


class DrivingStateNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.matcher = ScanOdometry(**{
            name: rospy.get_param('~' + name, default)
            for name, default in (
                ('lidar_to_rear_x_m', 1.4), ('min_z_m', -.8), ('max_z_m', .8),
                ('max_range_m', 18.), ('voxel_m', .25),
                ('max_pair_distance_m', .65), ('min_pairs', 35),
                ('max_rmse_m', .25), ('max_correction_m', .5),
                ('max_correction_rad', .12))})
        self.machine = DrivingState(
            self.matcher,
            input_timeout_s=float(rospy.get_param('~input_timeout_s', .35)),
            max_blackout_distance_m=float(rospy.get_param('~max_blackout_distance_m', 30.)),
            max_rejoin_error_m=float(rospy.get_param('~max_rejoin_error_m', 2.)),
            max_rejoin_yaw_rad=float(rospy.get_param('~max_rejoin_yaw_rad', .25)),
            gps_silence_timeout_s=float(rospy.get_param('~gps_silence_timeout_s', .75)),
            max_unmatched_distance_m=float(rospy.get_param(
                '~max_unmatched_distance_m', 3.)),
            gps_return_grace_s=float(rospy.get_param('~gps_return_grace_s', 1.5)),
            icp_forward_weight=float(rospy.get_param('~icp_forward_weight', .25)),
            icp_lateral_weight=float(rospy.get_param('~icp_lateral_weight', .1)))
        self.config = Config(**{key: rospy.get_param('~' + key, default)
                                for key, default in vars(Config()).items()})
        self.planner = FollowTheGap(self.config)
        self.runtime = RuntimeSettings('leo_driving_state', FrozenConfigTarget(self.planner))
        waypoint_file = rospy.get_param('~waypoint_file')
        route_points = load_route_points(waypoint_file)
        # FTG top speed = Map Tuner point speed at the current route index.
        self.route_speeds_kmh = load_route_speeds(waypoint_file)
        # A safe pass can leave the mapped centerline by several metres.
        # Keep route projection available so the preferred bearing pulls
        # back toward the same waypoint path after the obstacle clears.
        route_max_error_m = float(rospy.get_param('~route_max_error_m', 7.))
        self.route = RouteProgress(route_points, max_error=route_max_error_m,
                                   backward_window=30)
        self.s14_ground_bounds = s14_ground_interval(waypoint_file, self.route)
        self.s14_ground_reference_z_m = float(rospy.get_param(
            '~s14_ground_reference_z_m', -1.61))
        if (not math.isfinite(self.s14_ground_reference_z_m)
                or not -2.5 < self.s14_ground_reference_z_m < -.3):
            raise ValueError('invalid S14 ground reference in LiDAR frame')
        self.blackout_route = RouteProgress(route_points, max_error=route_max_error_m,
                                            backward_window=30)
        self.rejoin_route = RouteProgress(route_points, max_error=2.0)
        self.rejoin_diagnostic = None
        # GPS-denied window as map anchors, projected onto the current route.
        self.blackout_start_index, _ = anchor_on_route(
            route_points, rospy.get_param('~blackout_start_anchor'))
        self.blackout_end_index, _ = anchor_on_route(
            route_points, rospy.get_param('~blackout_end_anchor'))
        # Surveyed tunnel-exit gate (map constant, not simulator state).
        self.exit_rejoin_start_m = float(rospy.get_param('~exit_rejoin_start_m', 35.))
        self.exit_rejoin_full_m = float(rospy.get_param('~exit_rejoin_full_m', 12.))
        self.exit_rejoin_active = False
        exit_anchor = rospy.get_param('~exit_anchor')
        self.exit_xy = np.array((float(exit_anchor['x']), float(exit_anchor['y'])))
        exit_index, _ = anchor_on_route(route_points, exit_anchor)
        self.exit_index = exit_index
        if (not 0 <= exit_index < len(route_points)-1
                or not np.isfinite(self.exit_xy).all()
                or not 0 <= self.exit_rejoin_full_m < self.exit_rejoin_start_m):
            raise ValueError('invalid tunnel exit coordinate or limits')
        tangent = route_points[exit_index+1, :2] - route_points[exit_index, :2]
        segment_length = float(np.linalg.norm(tangent))
        if segment_length < 1e-6:
            raise ValueError('invalid tunnel exit route tangent')
        self.exit_tangent = tangent / segment_length
        projected_m = float(np.clip(
            (self.exit_xy - route_points[exit_index, :2]) @ self.exit_tangent,
            0., segment_length))
        self.exit_route_s = float(self.route.s[exit_index] + projected_m)
        self.exit_route_offset = (self.exit_xy - route_points[exit_index, :2]
                                  - projected_m * self.exit_tangent)
        self.previous_output_pose = None
        self.gps_pose_max_age_s = float(rospy.get_param('~gps_pose_max_age_s', 1.0))
        if (not 0 <= self.blackout_start_index <= self.blackout_end_index
                < len(route_points) or not math.isfinite(self.gps_pose_max_age_s)
                or self.gps_pose_max_age_s <= 0):
            raise ValueError('invalid tunnel blackout indices or GPS pose age')
        self.route_lookahead_m = float(rospy.get_param('~route_lookahead_m', 5.))
        self.target_distance_m = float(rospy.get_param('~target_distance_from_lidar_m', 4.))
        self.speed_timeout_s = float(rospy.get_param('~speed_timeout_s', .35))
        if not all(math.isfinite(v) and v > 0 for v in (
                self.route_lookahead_m, self.target_distance_m, self.speed_timeout_s)):
            raise ValueError('invalid avoidance route/timeout settings')
        self.pending = None
        self.last_cloud_received_wall = None
        self.last_cloud_stamp_age_s = None
        self.last_plan_completed_wall = None
        self.latest_speed = None
        self.latest_plan = None
        self.odom_stamp = None
        self.gps_stamp = None
        self.odom_history = deque(maxlen=100)
        self.gps_history = deque(maxlen=100)
        self.speed_history = deque(maxlen=100)
        self.pose_pub = rospy.Publisher('/leo/driving_pose', PoseStamped, queue_size=1)
        self.valid_pub = rospy.Publisher('/leo/driving_valid', Bool, queue_size=1)
        self.mode_pub = rospy.Publisher('/leo/driving_mode', String, queue_size=1)
        self.status_pub = rospy.Publisher('/leo/driving_status', String, queue_size=1)
        self.target_pub = rospy.Publisher('/leo/avoidance_target', PointStamped, queue_size=1)
        self.speed_pub = rospy.Publisher('/leo/avoidance_speed_limit_kmh', Float64, queue_size=1)
        self.avoidance_status_pub = rospy.Publisher('/leo/avoidance_status', String, queue_size=1)
        self.debug = PerceptionDebug('/leo/ftg_debug')
        self.debug_lidar_z = float(rospy.get_param('~lidar_to_rear_z_m', 1.23))
        self.subs = [
            rospy.Subscriber('/gps', GPSMessage, self.on_raw_gps, queue_size=1),
            rospy.Subscriber('/localization/pose', PoseStamped, self.on_gps_pose, queue_size=1),
            rospy.Subscriber('/localization/valid', Bool, self.on_gps_valid, queue_size=1),
            rospy.Subscriber('/leo/ego_odom', Odometry, self.on_odom, queue_size=1),
            rospy.Subscriber('/competition/ego_speed_kmh', Float64, self.on_speed, queue_size=1),
            rospy.Subscriber('/lidar3D', PointCloud2, self.on_cloud,
                             queue_size=1, buff_size=2 ** 24)]
        threading.Thread(target=self.cloud_worker, daemon=True).start()
        self.timer = rospy.Timer(rospy.Duration(.05), self.tick)

    @staticmethod
    def yaw(q):
        return math.atan2(2 * (q.w*q.z + q.x*q.y),
                          1 - 2 * (q.y*q.y + q.z*q.z))

    @staticmethod
    def fresh_stamp(message, timeout_s=.35):
        age = rospy.Time.now().to_sec() - message.header.stamp.to_sec()
        return math.isfinite(age) and 0 <= age <= timeout_s

    def on_gps_pose(self, message):
        if (message.header.frame_id != 'map'
                or not self.fresh_stamp(message, self.gps_pose_max_age_s)):
            return
        p = message.pose.position
        pose = (p.x, p.y, self.yaw(message.pose.orientation))
        if not all(math.isfinite(value) for value in pose):
            return
        with self.lock:
            self.gps_stamp = message.header.stamp.to_sec()
            received = time.monotonic()
            self.gps_history.append((pose, received, self.gps_stamp))
            self.machine.update_gps(pose, received)

    def on_raw_gps(self, message):
        if not self.fresh_stamp(message):
            return
        with self.lock:
            self.machine.update_raw_gps(message.latitude, message.longitude,
                                        time.monotonic(), message.status)

    def on_speed(self, message):
        speed_mps = float(message.data) / 3.6
        with self.lock:
            sample = (speed_mps, time.monotonic())
            self.latest_speed = sample
            self.speed_history.append(sample)

    def on_gps_valid(self, message):
        with self.lock:
            self.machine.update_gps_valid(message.data, time.monotonic())

    def on_odom(self, message):
        if message.header.frame_id != 'odom' or not self.fresh_stamp(message):
            return
        p = message.pose.pose.position
        pose = (p.x, p.y, self.yaw(message.pose.pose.orientation))
        with self.lock:
            self.odom_stamp = message.header.stamp.to_sec()
            received = time.monotonic()
            self.odom_history.append((pose, received, self.odom_stamp))
            self.machine.update_odom(pose, received)

    def on_cloud(self, message):
        received = time.monotonic()
        stamp_age = rospy.Time.now().to_sec() - message.header.stamp.to_sec()
        with self.lock:
            self.pending = (message, received)
            self.last_cloud_received_wall = received
            self.last_cloud_stamp_age_s = stamp_age

    def route_bearing(self, pose):
        x, y, yaw = pose
        self.route_progress_m, _ = locate_or_reacquire(self.route, x, y, yaw)
        index = self.route.index
        distance_m = 0.
        target = self.route.points[index]
        for offset in range(1, len(self.route.points)):
            next_index = (index + offset) % len(self.route.points)
            next_point = self.route.points[next_index]
            distance_m += float(np.linalg.norm(next_point - target))
            target = next_point
            if distance_m >= self.route_lookahead_m:
                break
        # Approach the photographed tunnel-exit coordinate while preserving
        # the surveyed route's heading and lookahead. The offset is small and
        # is introduced gradually as wall following hands back to waypoints.
        remaining_m = self.exit_route_s - self.route_progress_m
        if 0. <= remaining_m < self.exit_rejoin_start_m:
            blend = float(np.clip(
                (self.exit_rejoin_start_m - remaining_m) /
                (self.exit_rejoin_start_m - self.exit_rejoin_full_m), 0., 1.))
            target = target + blend * self.exit_route_offset
        dx, dy = target[0] - x, target[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if local_x <= 0:
            raise ValueError('route target is behind vehicle')
        return math.atan2(local_y, local_x)

    def ground_reference_for_progress(self, progress):
        bounds = getattr(self, 's14_ground_bounds', None)
        if bounds is not None and bounds[0] <= progress < bounds[1]:
            return self.s14_ground_reference_z_m
        return None

    def exit_rejoin_guidance(self, heading_yaw):
        """Exit-only direction from saved route geometry and captured heading.

        LiDAR position selects route progress, but its drifting lateral
        coordinate never contributes to the heading error. Wall spacing
        remains the independent lateral reference in the planner. Once the
        exit window is entered, keep guidance until select() accepts GPS;
        accumulated position error can put progress beyond the exit early.
        """
        remaining = self.exit_route_s - self.route_progress_m
        if 0. <= remaining < self.exit_rejoin_start_m:
            self.exit_rejoin_active = True
        if not self.exit_rejoin_active:
            return None, 1., None
        progress = self.route_progress_m
        target_s = min(progress + self.route_lookahead_m, float(self.route.s[-1]))
        start = np.array([np.interp(progress, self.route.s, self.route.points[:, axis])
                          for axis in (0, 1)])
        end = np.array([np.interp(target_s, self.route.s, self.route.points[:, axis])
                        for axis in (0, 1)])
        delta = end - start
        if not math.isfinite(heading_yaw) or float(np.linalg.norm(delta)) < 1e-6:
            raise ValueError('invalid exit route direction or heading')
        route_yaw = math.atan2(delta[1], delta[0])
        yaw_error = math.atan2(math.sin(route_yaw - heading_yaw),
                               math.cos(route_yaw - heading_yaw))
        blend = float(np.clip((self.exit_rejoin_start_m - remaining) /
                             (self.exit_rejoin_start_m - self.exit_rejoin_full_m), 0., 1.))
        # Smooth the heading handover at both ends of the existing window.
        blend = blend * blend * (3. - 2. * blend)
        return yaw_error, 1. - blend, remaining

    def route_rejoin_ok(self, gps_pose, map_pose, blackout_index):
        """Accept a drifted LiDAR pose only when GPS agrees with nearby route."""
        self.rejoin_diagnostic = None
        if gps_pose is None or map_pose is None or blackout_index is None:
            return False
        try:
            map_s, map_error = locate_or_reacquire(self.blackout_route, *map_pose)
            # ICP can accumulate several metres of longitudinal error across
            # the tunnel. Search the whole route once GPS returns instead of
            # the normal 10-waypoint rear window around the drifted ICP index.
            self.rejoin_route.index = None
            gps_s, gps_error = self.rejoin_route.locate(*gps_pose)
        except ValueError as error:
            self.rejoin_diagnostic = {'reason': str(error),
                                      'gps_pose': gps_pose,
                                      'lidar_pose': map_pose,
                                      'lidar_route_index': blackout_index}
            return False
        segment = self.rejoin_route.delta[self.rejoin_route.index]
        route_yaw = math.atan2(segment[1], segment[0])
        yaw_error = math.atan2(math.sin(gps_pose[2] - route_yaw),
                               math.cos(gps_pose[2] - route_yaw))
        progress_delta = ((gps_s - map_s + self.rejoin_route.total / 2.)
                          % self.rejoin_route.total - self.rejoin_route.total / 2.)
        allowed = (gps_error <= 1.9 and abs(yaw_error) <= 0.35
                   and -9. <= progress_delta <= 8.)
        self.rejoin_diagnostic = {
            'gps_cross_track_m': gps_error,
            'lidar_cross_track_m': map_error,
            'gps_yaw_error_rad': yaw_error,
            'route_progress_delta_m': progress_delta,
            'gps_route_index': self.rejoin_route.index,
            'lidar_route_index': blackout_index,
            'route_ok': allowed}
        return allowed

    @staticmethod
    def nearest_capture(history, stamp, tolerance_s=.15):
        if not history:
            return None
        sample = min(history, key=lambda item: abs(item[2] - stamp))
        return sample if abs(sample[2] - stamp) <= tolerance_s else None

    @staticmethod
    def nearest_speed(history, received, tolerance_s):
        if not history:
            return None
        sample = min(history, key=lambda item: abs(item[1] - received))
        return sample if abs(sample[1] - received) <= tolerance_s else None

    def cloud_worker(self):
        while not rospy.is_shutdown():
            with self.lock:
                item, self.pending = self.pending, None
            if item is None:
                self.debug.clear_if_stale()
                time.sleep(.01)
                continue
            message, received = item
            points = None
            speed = None
            ground_reference = None
            self.runtime.poll()
            try:
                with self.lock:
                    scan_stamp = message.header.stamp.to_sec()
                    odom_capture = self.nearest_capture(self.odom_history, scan_stamp)
                    odom_sample = ((odom_capture[0], min(odom_capture[1], received))
                                   if odom_capture else None)
                    gps_capture = self.nearest_capture(self.gps_history, scan_stamp)
                    speed = self.nearest_speed(self.speed_history, received,
                                               self.speed_timeout_s)
                    anchoring = (self.machine.gps_available(received)
                                 and not self.machine.had_blackout)
                    gps_anchor_pose = gps_capture[0] if anchoring and gps_capture else None
                if (message.header.frame_id != 'lidar'
                        or not self.fresh_stamp(message)
                        or odom_sample is None
                        or (anchoring and gps_anchor_pose is None)):
                    raise ValueError('LiDAR/odometry frame or capture times do not match')
                points = np.asarray(list(point_cloud2.read_points(
                    message, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=float)
                if time.monotonic() - received > self.machine.input_timeout_s:
                    raise ValueError('LiDAR ICP input became stale during processing')
                xy = self.matcher.prepare(points)
                with self.lock:
                    self.machine.scan(xy, received, odom_sample,
                                      anchoring, gps_anchor_pose)
                    pose = self.machine.map_pose if self.machine.scan_valid else None
                    scan_fault = self.machine.scan_fault_reason
                if pose is None:
                    raise ValueError(scan_fault or 'LiDAR map pose missing/stale')
                if (speed is None or not math.isfinite(speed[0])
                        or speed[0] < 0):
                    raise ValueError('Ego speed missing/stale at LiDAR scan')
                preferred_bearing = self.route_bearing(pose)
                ground_reference = self.ground_reference_for_progress(self.route_progress_m)
                if self.route_speeds_kmh is not None:
                    self.planner.config = replace(
                        self.planner.config, max_speed_mps=float(
                            self.route_speeds_kmh[self.route.index]) / 3.6)
                with self.lock:
                    # A brief GPS fix must not hand steering to a drifted
                    # route position. Exit guidance starts in the surveyed
                    # window and stays active until GPS rejoin is accepted.
                    wall_follow = (self.machine.had_blackout
                                   or self.machine.gps_zero(received)
                                   or self.machine.gps_missing(received))
                    wall_follow = (wall_follow
                                   and self.blackout_start_index <= self.route.index
                                   <= self.blackout_end_index)
                    exit_heading, wall_weight, remaining = (
                        self.exit_rejoin_guidance(odom_sample[0][2]) if wall_follow
                        else (None, 1., None))
                self.planner.debug_capture = self.debug.active()
                plan = self.planner.plan(points, speed[0], preferred_bearing,
                                         wall_follow=wall_follow,
                                         wall_follow_weight=wall_weight,
                                         route_rejoin_remaining_m=remaining,
                                         exit_heading_error_rad=exit_heading,
                                         ego_pose=odom_sample[0],
                                         stamp=scan_stamp,
                                         ground_reference_z_m=ground_reference)
                if time.monotonic() - received > self.machine.input_timeout_s:
                    raise ValueError('LiDAR planning exceeded input age limit')
            except Exception as error:
                self.debug.clear('invalid_perception_input', message.header)
                plan = Plan(0., 0., str(error), 0., 0.)
                with self.lock:
                    if not self.machine.scan_valid or 'LiDAR/odometry frame' in str(error):
                        self.machine.scan_valid = False
                        self.machine.scan_fault_reason = str(error)
                    self.machine.reason = str(error)
            else:
                snapshot = self.planner.debug_snapshot
                if snapshot is not None:
                    details = {key: value for key, value in snapshot.items()
                               if key not in ('points', 'tracked')}
                    self.debug.publish(points, snapshot['points'], message.header, pose,
                                       offset=(self.planner.config.lidar_to_rear_x_m,
                                               0., self.debug_lidar_z),
                                       mode='ftg_filtered', debug_point_stage='ftg_collision_input',
                                       filter_summary='측정 노면 평면 → 충돌 높이·자차 → 전후방·차체·벽·빈 공간 판단',
                                       pose_sync='scan_capture', imu_corrected=False,
                                       tracked_obstacles=len(plan.tracked_obstacles),
                                       obstacles=tracked_obstacles_map(
                                           snapshot.get('tracked', []), pose,
                                           self.planner.config.lidar_to_rear_x_m),
                                       planner_state=plan.avoidance_phase, planner_reason=plan.reason,
                                       **details)
                elif self.debug.active():
                    self.debug.clear('no_perception_evidence', message.header)
            with self.lock:
                self.latest_plan = (plan, received)
                self.last_plan_completed_wall = time.monotonic()
                self.machine.update_avoidance(plan.speed_mps > 0., received,
                                             plan.obstacle_stop)
            if plan.front_points or plan.avoidance_phase in ('PASSING', 'RETURNING'):
                rospy.loginfo_throttle(
                    .5,
                    'FTG plan: reason=%s phase=%s front_points=%d nearest_x=%.2fm '
                    'front_clearance=%.2fm gap_width=%.2fm bearing=%.3frad '
                    'target_speed=%.2fm/s ego_speed=%.2fm/s right_wall=%s '
                    'wall_heading=%s passing_side=%d rear_points=%d '
                    'rear_scan_points=%d rear_confirm=%d',
                    plan.reason, plan.avoidance_phase, plan.front_points,
                    plan.nearest_front_x_m, plan.front_clearance_m,
                    plan.gap_width_m, plan.target_bearing_rad, plan.speed_mps,
                    speed[0] if speed is not None else -1.,
                    plan.right_wall_clearance_m, plan.right_wall_heading_rad,
                    self.planner.passing_side, plan.rear_obstacle_points,
                    plan.rear_scan_points, plan.rear_confirm_scans)
            moving = [track for track in plan.tracked_obstacles if track[5]]
            if moving:
                rospy.logwarn_throttle(
                    .5, 'FTG dynamic obstacles (id, x, y, vx, vy, lateral_scans): '
                    '%s; crossing_threats=%d reason=%s',
                    [track[:5] + track[6:] for track in moving],
                    plan.crossing_threats,
                    plan.reason)
            if plan.reason == 'rear/body swept path blocked' and plan.swept_rejection:
                rospy.logwarn_throttle(
                    2., 'FTG swept rejection: %s',
                    json.dumps(plan.swept_rejection, sort_keys=True))
            if (plan.reason == 'front obstacle inside stopping distance'
                    and points is not None):
                candidates = self.planner.obstacle_candidates(
                    points, ground_reference_z_m=ground_reference)
                front = candidates[(candidates[:, 0] > 0.)
                                   & (np.abs(candidates[:, 1]) <=
                                      self.config.vehicle_half_width_m
                                      + self.config.side_margin_m)]
                if len(front):
                    rospy.logwarn_throttle(
                        2., 'FTG close front points: count=%d x=[%.2f,%.2f] '
                        'y=[%.2f,%.2f] z=[%.2f,%.2f]', len(front),
                        float(np.min(front[:, 0])), float(np.max(front[:, 0])),
                        float(np.min(front[:, 1])), float(np.max(front[:, 1])),
                        float(np.min(front[:, 2])), float(np.max(front[:, 2])))

    def tick(self, _event):
        with self.lock:
            now = time.monotonic()
            map_pose = self.machine.map_pose
            lidar_pose_before_select = map_pose
            blackout_index = None
            blackout_projection_error = None
            if map_pose is not None:
                try:
                    locate_or_reacquire(self.blackout_route, *map_pose)
                    blackout_index = self.blackout_route.index
                except ValueError as error:
                    blackout_projection_error = str(error)
            blackout_allowed = (blackout_index is not None
                                and self.blackout_start_index <= blackout_index
                                <= self.blackout_end_index)
            if not self.machine.had_blackout or not self.machine.gps_available(now):
                self.rejoin_diagnostic = None
            gps_rejoin_ok = (self.machine.had_blackout
                             and self.machine.gps_available(now)
                             and self.route_rejoin_ok(
                                 self.machine.gps_pose[0], map_pose,
                                 blackout_index))
            mode, pose, reason = self.machine.select(
                now, blackout_allowed, gps_rejoin_ok)
            # Snapshot the source stamp under the same lock as select(). A
            # newer GPS callback must not retimestamp this selected pose.
            output_pose_stamp = (getattr(self, 'gps_stamp', None)
                                 if mode == DrivingState.GPS else None)
            if mode == DrivingState.GPS:
                self.exit_rejoin_active = False
            previous_output_pose = self.previous_output_pose
            output_jump_m = (float(np.linalg.norm(
                np.asarray(pose[:2]) - np.asarray(previous_output_pose[:2])))
                if pose is not None and previous_output_pose is not None else None)
            if pose is not None:
                self.previous_output_pose = pose
            if (mode == DrivingState.HOLD
                    and reason == 'GPS blackout outside configured tunnel section'
                    and blackout_projection_error):
                reason = 'LiDAR blackout pose outside route corridor or facing backwards'
                self.machine.reason = reason
            travelled = self.machine.travelled_m
            raw_gps = self.machine.raw_gps
            gps_pose = self.machine.gps_pose
            gps_valid = self.machine.gps_valid
            gps_return_since = self.machine.gps_return_since
            match = self.machine.last_match
            scan_fault = self.machine.scan_fault_reason
            map_pose = self.machine.map_pose
            scan_wall = self.machine.last_scan_wall
            odom_sample = self.machine.odom_pose
            item = self.latest_plan
            speed = self.latest_speed
            last_cloud_received_wall = self.last_cloud_received_wall
            last_cloud_stamp_age_s = self.last_cloud_stamp_age_s
            last_plan_completed_wall = self.last_plan_completed_wall
        cloud_received_age_s = (now - last_cloud_received_wall
                                if last_cloud_received_wall is not None else None)
        plan_completed_age_s = (now - last_plan_completed_wall
                                if last_plan_completed_wall is not None else None)
        plan = item[0] if item and 0 <= now-item[1] <= self.machine.input_timeout_s else None
        speed_mps = (speed[0] if speed and 0 <= now-speed[1] <= self.speed_timeout_s
                     and math.isfinite(speed[0]) and speed[0] >= 0 else None)
        if output_jump_m is not None and output_jump_m > 10.:
            rospy.logwarn('Driving pose jump %.2f m: mode=%s previous=%s output=%s lidar_before_select=%s',
                          output_jump_m, mode, previous_output_pose, pose,
                          lidar_pose_before_select)
        avoidance_valid = (mode == DrivingState.BLACKOUT
                           and pose is not None
                           and plan is not None and plan.speed_mps > 0
                           and speed_mps is not None)
        if avoidance_valid:
            target = PointStamped()
            target.header.stamp = rospy.Time.now()
            target.header.frame_id = 'base_link'
            target.point.x = (self.config.lidar_to_rear_x_m
                              + self.target_distance_m * math.cos(plan.target_bearing_rad))
            target.point.y = self.target_distance_m * math.sin(plan.target_bearing_rad)
            self.target_pub.publish(target)
        self.speed_pub.publish(Float64(data=plan.speed_mps * 3.6 if avoidance_valid else 0.))
        self.avoidance_status_pub.publish(String(data=json.dumps({
            'valid': avoidance_valid,
            'reason': plan.reason if plan else 'LiDAR avoidance plan missing/stale',
            'target_speed_mps': plan.speed_mps if plan else 0.,
            'ego_speed_mps': speed_mps,
            'target_bearing_rad': plan.target_bearing_rad if plan else 0.,
            'candidate_bearings_rad': list(plan.candidate_bearings_rad) if plan else [],
            'swept_rejection': plan.swept_rejection if plan else None,
            'front_clearance_m': plan.front_clearance_m if plan else 0.,
            'gap_width_m': plan.gap_width_m if plan else 0.,
            'obstacle_points': plan.obstacle_points if plan else 0,
            'front_points': plan.front_points if plan else 0,
            'nearest_front_x_m': plan.nearest_front_x_m if plan else 0.,
            'right_wall_clearance_m': plan.right_wall_clearance_m if plan else None,
            'right_wall_target_clearance_m': (
                plan.right_wall_target_clearance_m if plan else None),
            'right_wall_heading_rad': plan.right_wall_heading_rad if plan else None,
            'wall_follow_weight': plan.wall_follow_weight if plan else 0.,
            'avoidance_phase': plan.avoidance_phase if plan else None,
            'rear_scan_points': plan.rear_scan_points if plan else 0,
            'rear_obstacle_points': plan.rear_obstacle_points if plan else 0,
            'rear_confirm_scans': plan.rear_confirm_scans if plan else 0,
            'route_rejoin_remaining_m': (
                plan.route_rejoin_remaining_m if plan else None),
            'crossing_threats': plan.crossing_threats if plan else 0,
            'tracked_obstacles': list(plan.tracked_obstacles) if plan else []})))
        self.mode_pub.publish(String(data=mode))
        self.valid_pub.publish(Bool(data=pose is not None))
        plan_reason = plan.reason if plan else 'LiDAR avoidance plan missing/stale'
        if mode == DrivingState.HOLD:
            rospy.logwarn_throttle(
                2., 'Driving input ages: cloud_received=%s cloud_stamp=%s '
                'plan_completed=%s scan=%s odom=%s plan_reason=%s',
                cloud_received_age_s, last_cloud_stamp_age_s,
                plan_completed_age_s,
                now - scan_wall if scan_wall is not None else None,
                now - odom_sample[1] if odom_sample is not None else None,
                plan_reason)
            rospy.logwarn_throttle(2., 'Driving HOLD: %s; scan: %s; avoidance: %s; front_points: %d; nearest_front_x_m: %.2f; clearance_m: %.2f; right_wall_clearance_m: %s; avoidance_phase: %s; route_index: %s; route_projection_error: %s; map_pose: %s; rejoin: %s; error: %s',
                                   reason, scan_fault, plan_reason,
                                   plan.front_points if plan else 0,
                                   plan.nearest_front_x_m if plan else 0.,
                                   plan.front_clearance_m if plan else 0.,
                                   plan.right_wall_clearance_m if plan else None,
                                   plan.avoidance_phase if plan else None,
                                   blackout_index, blackout_projection_error,
                                   map_pose,
                                   self.rejoin_diagnostic,
                                   self.machine.last_rejoin_error)
        self.status_pub.publish(String(data=json.dumps({
            'mode': mode, 'valid': pose is not None, 'reason': reason,
            'blackout_allowed': blackout_allowed,
            'blackout_route_index': blackout_index,
            'blackout_projection_error': blackout_projection_error,
            'raw_gps_kind': raw_gps[0] if raw_gps else None,
            'gps_pose_age_s': now - gps_pose[1] if gps_pose else None,
            'gps_valid_age_s': now - gps_valid[1] if gps_valid else None,
            'gps_return_elapsed_s': (now - gps_return_since
                                     if gps_return_since is not None else None),
            'gps_rejoin_route_ok': gps_rejoin_ok,
            'gps_rejoin_error': self.machine.last_rejoin_error,
            'gps_rejoin_diagnostic': self.rejoin_diagnostic,
            'avoidance_reason': plan_reason,
            'scan_fault_reason': scan_fault,
            'map_pose': map_pose,
            'lidar_pose_before_select': lidar_pose_before_select,
            'driving_pose': pose,
            'previous_driving_pose': previous_output_pose,
            'driving_pose_jump_m': output_jump_m,
            'scan_age_s': now - scan_wall if scan_wall is not None else None,
            'cloud_received_age_s': cloud_received_age_s,
            'cloud_stamp_age_s': last_cloud_stamp_age_s,
            'plan_completed_age_s': plan_completed_age_s,
            'odom_age_s': now - odom_sample[1] if odom_sample is not None else None,
            'travelled_m': travelled,
            'unmatched_distance_m': self.machine.unmatched_distance_m,
            'icp_pairs': match.pairs if match else None,
            'icp_rmse_m': match.rmse_m if match and math.isfinite(match.rmse_m) else None})))
        if pose is None:
            return
        message = PoseStamped()
        message.header.stamp = (rospy.Time.from_sec(output_pose_stamp)
                                if output_pose_stamp is not None else rospy.Time.now())
        message.header.frame_id = 'map'
        message.pose.position.x, message.pose.position.y = pose[:2]
        message.pose.orientation.z = math.sin(pose[2] / 2.)
        message.pose.orientation.w = math.cos(pose[2] / 2.)
        self.pose_pub.publish(message)


if __name__ == '__main__':
    rospy.init_node('leo_driving_state')
    DrivingStateNode()
    rospy.spin()
