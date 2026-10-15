#!/usr/bin/env python3
"""GPS-route obstacle handling: LiDAR tracks -> offset path and speed cap.

Publishes /leo/route_avoidance (JSON): a map-frame path for Pure Pursuit, a
speed cap, the state and the tracked obstacles. It never publishes control;
the follower stops when this output is missing or stale.
"""
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String

from leo_avoidance.road_points import RoadPointConfig, boundary_mask, obstacle_points
from leo_avoidance.obstacle_tracker import ObstacleTracker
from leo_avoidance.route_avoidance import (RouteAvoidance, RouteAvoidanceConfig,
                                           RouteObstacle)
from leo_guard.route_file import load_lane_changes, load_route_points, load_survey

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from avoidance_runtime import FrozenConfigTarget, RuntimeSettings  # noqa: E402


class RouteAvoidanceNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.lidar_to_rear_x_m = float(rospy.get_param('~lidar_to_rear_x_m', 1.4))
        self.input_timeout_s = float(rospy.get_param('~input_timeout_s', .35))
        points = load_route_points(rospy.get_param('~waypoint_file'))[:, :2]
        config = RouteAvoidanceConfig(**{
            key: float(rospy.get_param('~' + key, default))
            for key, default in vars(RouteAvoidanceConfig()).items()})
        # Dashed-line intervals from MGeo (build_route_lane_changes.py); a
        # route without them never leaves its lane, it waits instead.
        lane_file = rospy.get_param('~lane_change_file', '')
        lane_changes = load_lane_changes(lane_file, points) if lane_file else []
        # Crosswalks just past each signal stop line (no crosswalk layer in
        # the map data; at C016 the crosswalk is about 7 m past the line): a
        # person crossing there means stopping before it.
        crosswalks = []
        survey_file = rospy.get_param('~intersection_file', '')
        if survey_file:
            for stop in load_survey(survey_file, points):
                if stop.get('controlled') is True:
                    crosswalks.append((stop['s'] + float(rospy.get_param('~crosswalk_start_m', 1.)),
                                       stop['s'] + float(rospy.get_param('~crosswalk_end_m', 10.))))
        self.planner = RouteAvoidance(points, config, lane_changes, crosswalks)
        self.runtime = RuntimeSettings('leo_route_avoidance', FrozenConfigTarget(self.planner))
        # Open-road returns: local ground per grid cell, not the tunnel's single
        # road plane (which turns a road rising ahead into obstacles).
        self.road_points = RoadPointConfig(**{
            key: (tuple(rospy.get_param('~road_' + key, list(default)))
                  if isinstance(default, tuple) else
                  float(rospy.get_param('~road_' + key, default)))
            for key, default in vars(RoadPointConfig(
                lidar_to_rear_x_m=self.lidar_to_rear_x_m)).items()})
        self.tracker = ObstacleTracker(
            self.lidar_to_rear_x_m, region_x_m=(-10., 60.),
            region_half_width_m=float(rospy.get_param('~track_half_width_m', 10.)),
            max_size_m=float(rospy.get_param('~track_max_size_m', 8.)))
        self.pose = None
        self.speed = None
        self.signal = None
        self.pending = None
        self.pub = rospy.Publisher('/leo/route_avoidance', String, queue_size=1)
        rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'),
                         PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(rospy.get_param('~speed_topic', '/competition/ego_speed_kmh'),
                         Float64, self.on_speed, queue_size=1)
        rospy.Subscriber(rospy.get_param('~signal_status_topic', '/leo/signal_status'),
                         String, self.on_signal, queue_size=1)
        rospy.Subscriber(rospy.get_param('~cloud_topic', '/lidar3D'), PointCloud2,
                         self.on_cloud, queue_size=1, buff_size=2 ** 24)
        threading.Thread(target=self.worker, daemon=True).start()

    def on_pose(self, message):
        if message.header.frame_id != 'map':
            return
        p, q = message.pose.position, message.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if all(math.isfinite(v) for v in (p.x, p.y, yaw)):
            with self.lock:
                self.pose = ((p.x, p.y, yaw), time.monotonic())

    def on_speed(self, message):
        if math.isfinite(message.data):
            with self.lock:
                self.speed = (message.data / 3.6, time.monotonic())

    def on_signal(self, message):
        try:
            status = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.signal = (status, time.monotonic())

    def on_cloud(self, message):
        with self.lock:
            self.pending = (message, time.monotonic())

    def fresh(self, item, now, timeout=None):
        timeout = self.input_timeout_s if timeout is None else timeout
        return item[0] if item and 0 <= now - item[1] <= timeout else None

    def worker(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.runtime.poll()
            with self.lock:
                pending, self.pending = self.pending, None
            if pending is not None:
                try:
                    self.process(*pending)
                except Exception as error:  # keep the node alive; output goes stale
                    rospy.logwarn_throttle(2., 'Route avoidance failed: %s', error)
            rate.sleep()

    def drop_road_boundaries(self, points, pose):
        """Remove guardrails, fences and walls along the road.

        The tunnel FTG finds its walls; on open road the route frame makes a
        roadside barrier a thin line at nearly constant offset, also on bends.
        """
        if not len(points):
            return points
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        rear_x = points[:, 0] + self.lidar_to_rear_x_m
        xy = np.column_stack((x0 + c * rear_x - s * points[:, 1],
                              y0 + s * rear_x + c * points[:, 1]))
        route = self.planner.route
        if self.planner.last_s is None:
            lo, hi = 0, len(route.lengths)
            s_ego, _, _ = route.project(x0, y0, lo, hi)
        else:
            s_ego = self.planner.last_s
        lo, hi = route.window(s_ego - 15., s_ego + 70.)
        if len(xy) > 6000:  # keep the projection cost bounded
            keep = np.zeros(len(xy), bool)
            keep[::int(math.ceil(len(xy) / 6000.))] = True
            points, xy = points[keep], xy[keep]
        along, offset = route.project_many(xy, lo, hi)
        return points[~boundary_mask(along, offset)]

    def process(self, message, received):
        now = time.monotonic()
        with self.lock:
            pose = self.fresh(self.pose, now)
            speed = self.fresh(self.speed, now)
            signal = self.fresh(self.signal, now, 1.)
        if message.header.frame_id != 'lidar' or pose is None or speed is None:
            self.tracker.reset()
            rospy.logwarn_throttle(2., 'Route avoidance waiting for LiDAR/pose/speed')
            return
        points = np.asarray(list(point_cloud2.read_points(
            message, field_names=('x', 'y', 'z'), skip_nans=True)), float).reshape(-1, 3)
        points = obstacle_points(points, self.road_points)
        points = self.drop_road_boundaries(points, pose)
        stamp = message.header.stamp.to_sec()
        tracked = self.tracker.update(points, pose, stamp)
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        obstacles = []
        for t in tracked:
            rear_x = t.x_m + self.lidar_to_rear_x_m
            vx, vy = (t.vx_mps, t.vy_mps) if t.velocity_known else (0., 0.)
            obstacles.append(RouteObstacle(
                x0 + c * rear_x - s * t.y_m, y0 + s * rear_x + c * t.y_m,
                radius=.5 * t.size_y_m + .1, half_length=.5 * t.size_x_m + .1,
                vx=c * vx - s * vy, vy=s * vx + c * vy,
                # Lateral motion seen in consecutive scans already predicts a
                # crossing, before the track is confirmed dynamic.
                dynamic=t.dynamic or abs(t.lateral_scans) >= 2,
                track_id=t.track_id))
        stop_line, permitted = None, True
        if signal is not None:
            stop_line = signal.get('next_controlled_stop_distance_m')
            permitted = (signal.get('state') == 'clearing'
                         or signal.get('decision') == 'go'
                         or signal.get('release_stable') is True)
        decision = self.planner.update((x0, y0), speed, obstacles,
                                       stop_line_distance_m=stop_line,
                                       stop_line_permitted=permitted)
        cap = decision.speed_cap_mps
        self.pub.publish(String(data=json.dumps(dict(
            stamp=stamp, state=decision.state, reason=decision.reason,
            speed_cap_kmh=None if math.isinf(cap) else round(cap * 3.6, 2),
            offset_m=round(decision.offset_now_m, 2),
            path=[[round(x, 3), round(y, 3)] for x, y in decision.path],
            obstacles=[dict(s=round(o['s'], 1), d=round(o['d'], 2), r=round(o['r'], 2),
                            l=round(o['l'], 2), v_s=round(o['v_s'], 2),
                            v_d=round(o['v_d'], 2), moving=o['moving'], id=o['id'])
                       for o in decision.obstacles]))))
        if decision.state != 'FOLLOW':
            rospy.loginfo_throttle(.5, 'Route avoidance %s: %s (cap %s km/h, offset %.2f m)',
                                   decision.state, decision.reason,
                                   'none' if math.isinf(cap) else '%.1f' % (cap * 3.6),
                                   decision.offset_now_m)


if __name__ == '__main__':
    rospy.init_node('leo_route_avoidance')
    RouteAvoidanceNode()
    rospy.spin()
