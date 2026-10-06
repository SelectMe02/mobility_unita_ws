#!/usr/bin/env python3
"""Switch existing map-pose consumers from GPS to LiDAR ICP + Ego odometry."""
import json
import math
import threading
import time
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from morai_msgs.msg import GPSMessage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool, Float64, String

from leo_avoidance.driving_state import DrivingState
from leo_avoidance.ftg import Config, FollowTheGap, Plan
from leo_avoidance.scan_odometry import ScanOdometry
from leo_guard.traffic import RouteProgress


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
            gps_return_grace_s=float(rospy.get_param('~gps_return_grace_s', 1.5)))
        self.config = Config(**{key: rospy.get_param('~' + key, default)
                                for key, default in vars(Config()).items()})
        self.planner = FollowTheGap(self.config)
        route_points = np.loadtxt(rospy.get_param('~waypoint_file'))
        self.route = RouteProgress(route_points, backward_window=30)
        self.blackout_route = RouteProgress(route_points, backward_window=30)
        self.rejoin_route = RouteProgress(route_points, max_error=2.0)
        self.rejoin_diagnostic = None
        self.blackout_start_index = int(rospy.get_param('~blackout_start_index', 0))
        self.blackout_end_index = int(rospy.get_param(
            '~blackout_end_index', len(route_points) - 1))
        exit_index = int(rospy.get_param('~sim_exit_waypoint_index', 4053))
        self.exit_xy = np.array((float(rospy.get_param('~sim_exit_x_m', -78.13)),
                                 float(rospy.get_param('~sim_exit_y_m', -546.24))))
        self.exit_radius_m = float(rospy.get_param('~sim_exit_trigger_radius_m', 15.))
        self.exit_cross_track_m = float(rospy.get_param('~sim_exit_cross_track_limit_m', 5.))
        if (not 0 <= exit_index < len(route_points)-1
                or not np.isfinite(self.exit_xy).all()
                or not math.isfinite(self.exit_radius_m) or self.exit_radius_m <= 0
                or not math.isfinite(self.exit_cross_track_m) or self.exit_cross_track_m <= 0):
            raise ValueError('invalid MORAI exit coordinate or limits')
        tangent = route_points[exit_index+1, :2] - route_points[exit_index, :2]
        if np.linalg.norm(tangent) < 1e-6:
            raise ValueError('invalid MORAI exit route tangent')
        self.exit_tangent = tangent / np.linalg.norm(tangent)
        self.exit_latched = False
        self.latest_ego_pose = None
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
        self.subs = [
            rospy.Subscriber('/gps', GPSMessage, self.on_raw_gps, queue_size=1),
            rospy.Subscriber('/localization/pose', PoseStamped, self.on_gps_pose, queue_size=1),
            rospy.Subscriber('/localization/valid', Bool, self.on_gps_valid, queue_size=1),
            rospy.Subscriber('/leo/ego_odom', Odometry, self.on_odom, queue_size=1),
            rospy.Subscriber('/competition/ego_speed_kmh', Float64, self.on_speed, queue_size=1),
            rospy.Subscriber('/competition/ego_pose', PoseStamped, self.on_ego_pose, queue_size=1),
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

    def on_ego_pose(self, message):
        if message.header.frame_id != 'map' or not self.fresh_stamp(message):
            return
        p = message.pose.position
        pose = (p.x, p.y, self.yaw(message.pose.orientation))
        if not all(math.isfinite(v) for v in pose):
            return
        with self.lock:
            self.latest_ego_pose = (pose, time.monotonic())

    def past_exit_gate(self, pose):
        delta = np.asarray(pose[:2]) - self.exit_xy
        along = float(delta @ self.exit_tangent)
        cross = float(self.exit_tangent[0]*delta[1] - self.exit_tangent[1]*delta[0])
        return (along >= 0 and abs(cross) <= self.exit_cross_track_m
                and float(np.linalg.norm(delta)) <= self.exit_radius_m)

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
        with self.lock:
            self.pending = (message, time.monotonic())

    def route_bearing(self, pose):
        x, y, yaw = pose
        self.route.locate(x, y, yaw)
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
        dx, dy = target[0] - x, target[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if local_x <= 0:
            raise ValueError('route target is behind vehicle')
        return math.atan2(local_y, local_x)

    def route_rejoin_ok(self, gps_pose, map_pose, blackout_index):
        """Accept a drifted LiDAR pose only when GPS agrees with nearby route."""
        self.rejoin_diagnostic = None
        if gps_pose is None or map_pose is None or blackout_index is None:
            return False
        try:
            map_s, map_error = self.blackout_route.locate(*map_pose)
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
                time.sleep(.01)
                continue
            message, received = item
            points = None
            try:
                with self.lock:
                    scan_stamp = message.header.stamp.to_sec()
                    odom_capture = self.nearest_capture(self.odom_history, scan_stamp)
                    odom_sample = ((odom_capture[0], min(odom_capture[1], received))
                                   if odom_capture else None)
                    gps_capture = self.nearest_capture(self.gps_history, scan_stamp)
                    speed = self.nearest_speed(self.speed_history, received,
                                               self.speed_timeout_s)
                    ego_exit_pose = (self.latest_ego_pose[0]
                                     if self.exit_latched and self.machine.had_blackout
                                     and self.latest_ego_pose is not None
                                     and 0 <= received-self.latest_ego_pose[1]
                                     <= self.machine.input_timeout_s else None)
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
                if ego_exit_pose is not None:
                    pose = ego_exit_pose
                else:
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
                plan = self.planner.plan(points, speed[0], preferred_bearing)
                if time.monotonic() - received > self.machine.input_timeout_s:
                    raise ValueError('LiDAR planning exceeded input age limit')
            except Exception as error:
                plan = Plan(0., 0., str(error), 0., 0.)
                with self.lock:
                    if not self.machine.scan_valid or 'LiDAR/odometry frame' in str(error):
                        self.machine.scan_valid = False
                        self.machine.scan_fault_reason = str(error)
                    self.machine.reason = str(error)
            with self.lock:
                self.latest_plan = (plan, received)
                self.machine.update_avoidance(plan.speed_mps > 0., received)

    def tick(self, _event):
        with self.lock:
            now = time.monotonic()
            map_pose = self.machine.map_pose
            ego_item = self.latest_ego_pose
            ego_pose = (ego_item[0] if ego_item is not None
                        and 0 <= now-ego_item[1] <= self.machine.input_timeout_s
                        else None)
            if (self.machine.had_blackout and not self.exit_latched
                    and ego_pose is not None and self.past_exit_gate(ego_pose)):
                self.exit_latched = True
            sim_exit_pose = (ego_pose if self.exit_latched
                             and self.machine.had_blackout else None)
            blackout_index = None
            if map_pose is not None:
                try:
                    self.blackout_route.locate(*map_pose)
                    blackout_index = self.blackout_route.index
                except ValueError:
                    pass
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
                now, blackout_allowed, gps_rejoin_ok, sim_exit_pose,
                self.exit_latched)
            if mode == DrivingState.GPS:
                self.exit_latched = False
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
        plan = item[0] if item and 0 <= now-item[1] <= self.machine.input_timeout_s else None
        speed_mps = (speed[0] if speed and 0 <= now-speed[1] <= self.speed_timeout_s
                     and math.isfinite(speed[0]) and speed[0] >= 0 else None)
        avoidance_valid = (mode in (DrivingState.BLACKOUT, DrivingState.SIM_EXIT)
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
            'front_clearance_m': plan.front_clearance_m if plan else 0.,
            'gap_width_m': plan.gap_width_m if plan else 0.,
            'obstacle_points': plan.obstacle_points if plan else 0,
            'front_points': plan.front_points if plan else 0,
            'nearest_front_x_m': plan.nearest_front_x_m if plan else 0.})))
        self.mode_pub.publish(String(data=mode))
        self.valid_pub.publish(Bool(data=pose is not None))
        plan_reason = plan.reason if plan else 'LiDAR avoidance plan missing/stale'
        if mode == DrivingState.HOLD:
            rospy.logwarn_throttle(2., 'Driving HOLD: %s; scan: %s; avoidance: %s; front_points: %d; nearest_front_x_m: %.2f; clearance_m: %.2f; rejoin: %s; error: %s',
                                   reason, scan_fault, plan_reason,
                                   plan.front_points if plan else 0,
                                   plan.nearest_front_x_m if plan else 0.,
                                   plan.front_clearance_m if plan else 0.,
                                   self.rejoin_diagnostic,
                                   self.machine.last_rejoin_error)
        self.status_pub.publish(String(data=json.dumps({
            'mode': mode, 'valid': pose is not None, 'reason': reason,
            'blackout_allowed': blackout_allowed,
            'blackout_route_index': blackout_index,
            'raw_gps_kind': raw_gps[0] if raw_gps else None,
            'gps_pose_age_s': now - gps_pose[1] if gps_pose else None,
            'gps_valid_age_s': now - gps_valid[1] if gps_valid else None,
            'gps_return_elapsed_s': (now - gps_return_since
                                     if gps_return_since is not None else None),
            'gps_rejoin_route_ok': gps_rejoin_ok,
            'gps_rejoin_error': self.machine.last_rejoin_error,
            'gps_rejoin_diagnostic': self.rejoin_diagnostic,
            'sim_exit_latched': self.exit_latched,
            'ego_pose_age_s': now-ego_item[1] if ego_item is not None else None,
            'avoidance_reason': plan_reason,
            'scan_fault_reason': scan_fault,
            'map_pose': map_pose,
            'scan_age_s': now - scan_wall if scan_wall is not None else None,
            'odom_age_s': now - odom_sample[1] if odom_sample is not None else None,
            'travelled_m': travelled,
            'unmatched_distance_m': self.machine.unmatched_distance_m,
            'icp_pairs': match.pairs if match else None,
            'icp_rmse_m': match.rmse_m if match and math.isfinite(match.rmse_m) else None})))
        if pose is None:
            return
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = 'map'
        message.pose.position.x, message.pose.position.y = pose[:2]
        message.pose.orientation.z = math.sin(pose[2] / 2.)
        message.pose.orientation.w = math.cos(pose[2] / 2.)
        self.pose_pub.publish(message)


if __name__ == '__main__':
    rospy.init_node('leo_driving_state')
    DrivingStateNode()
    rospy.spin()
