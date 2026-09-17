#!/usr/bin/env python3
"""Observe waypoint tracking; publishes visualization and metrics only."""
import copy
import math
import threading
import time
from collections import deque

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

from unita_visualization.path_metrics import nearest_segment


class WaypointVisualizer:
    def __init__(self):
        self.lock = threading.RLock()
        self.frame = rospy.get_param('~map_frame', 'map')
        self.timeout = float(rospy.get_param('~pose_timeout', 1.0))
        self.min_distance = float(rospy.get_param('~trajectory_spacing', 0.2))
        self.max_points = int(rospy.get_param('~max_trajectory_points', 10000))
        self.window_size = int(rospy.get_param('~error_window_samples', 200))
        self.position_reset_distance = float(rospy.get_param('~position_reset_distance', 10.0))
        if (not math.isfinite(self.timeout) or self.timeout <= 0
                or not math.isfinite(self.min_distance) or self.min_distance < 0
                or self.max_points < 2 or self.window_size < 1
                or not math.isfinite(self.position_reset_distance) or self.position_reset_distance <= 0):
            raise ValueError('invalid visualization limits')
        self.reference = None
        self.points = []
        self.pose = None
        self.pose_time = 0
        self.valid = False
        self.valid_time = 0
        self.track = deque(maxlen=self.max_points)
        self.errors = deque(maxlen=self.window_size)
        self.last_stamp = None
        self.gap = False
        self.markers_pub = rospy.Publisher('/waypoint_debug/markers', MarkerArray, queue_size=1)
        self.track_pub = rospy.Publisher('/waypoint_debug/trajectory', Path, queue_size=1, latch=True)
        self.reference_pub = rospy.Publisher('/waypoint_debug/reference', Path, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher('/waypoint_debug/status', String, queue_size=1, latch=True)
        self.metric_pubs = {key: rospy.Publisher('/waypoint_debug/' + key, Float64, queue_size=1)
                            for key in ('cross_track_error', 'heading_error_deg', 'goal_distance', 'progress_m')}
        self.subs = [
            rospy.Subscriber(rospy.get_param('~path_topic', '/waypoint_path'), Path, self.path_callback, queue_size=1),
            rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'), PoseStamped, self.pose_callback, queue_size=1),
            rospy.Subscriber(rospy.get_param('~valid_topic', '/localization/valid'), Bool, self.valid_callback, queue_size=1)]
        self.timer = rospy.Timer(rospy.Duration(0.1), self.update, reset=True)

    def path_callback(self, message):
        with self.lock:
            points = [(p.pose.position.x, p.pose.position.y) for p in message.poses]
            if (message.header.frame_id != self.frame or len(points) < 2
                    or not all(math.isfinite(v) for p in points for v in p)
                    or not any(a != b for a, b in zip(points, points[1:]))):
                self.reference = None
                self.points = []
                empty = Path()
                empty.header.frame_id = self.frame
                self.reference_pub.publish(empty)
                rospy.logwarn('Invalid reference path or frame mismatch; expected %s', self.frame)
                return
            if points != self.points:
                self.track.clear()
                self.errors.clear()
                self.last_stamp = None
            self.points = points
            self.reference = message
            self.reference_pub.publish(message)

    def pose_callback(self, message):
        with self.lock:
            q = message.pose.orientation
            values = (message.pose.position.x, message.pose.position.y, q.x, q.y, q.z, q.w)
            if (message.header.frame_id != self.frame or not all(math.isfinite(v) for v in values)
                    or sum(v * v for v in values[2:]) < 1e-12):
                self.pose = None
                return
            self.pose = message
            self.pose_time = time.monotonic()

    def valid_callback(self, message):
        with self.lock:
            self.valid = message.data
            self.valid_time = time.monotonic()
            if not message.data:
                self.gap = True

    def marker(self, ident, kind, color):
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'waypoint_tracking'
        marker.id = ident
        marker.type = kind
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1
        marker.lifetime = rospy.Duration(0.5)
        return marker

    def update(self, event):
        with self.lock:
            self.update_locked()

    def update_locked(self):
        now = time.monotonic()
        fresh = (self.valid and self.pose is not None
                 and now - self.pose_time <= self.timeout and now - self.valid_time <= self.timeout)
        if fresh:
            age = (rospy.Time.now() - self.pose.header.stamp).to_sec()
            fresh = 0 <= age <= self.timeout
        if not fresh or self.reference is None:
            reason = 'WAIT: valid localization' if not fresh else 'WAIT: /waypoint_path in ' + self.frame
            self.status_pub.publish(String(data=reason))
            delete = Marker()
            delete.action = Marker.DELETEALL
            self.markers_pub.publish(MarkerArray(markers=[delete]))
            self.gap = True
            return
        pose = self.pose
        reset_position = (self.track and math.hypot(
            pose.pose.position.x - self.track[-1].pose.position.x,
            pose.pose.position.y - self.track[-1].pose.position.y) > self.position_reset_distance)
        if (self.gap or reset_position
                or (self.last_stamp is not None and pose.header.stamp < self.last_stamp)):
            self.track.clear()  # Avoid drawing a false line across GPS loss/reset.
            self.errors.clear()
            self.last_stamp = None
            self.gap = False
        x, y = pose.pose.position.x, pose.pose.position.y
        q = pose.pose.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        qx, qy, qz, qw = (v / norm for v in (q.x, q.y, q.z, q.w))
        yaw = math.atan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))
        px, py, error, heading, progress, index = nearest_segment(self.points, x, y, yaw)
        if pose.header.stamp != self.last_stamp:
            self.errors.append(error)
            if (not self.track or math.hypot(x-self.track[-1].pose.position.x,
                                            y-self.track[-1].pose.position.y) >= self.min_distance):
                self.track.append(copy.deepcopy(pose))
            self.last_stamp = pose.header.stamp
        track = Path()
        track.header = pose.header
        track.poses = list(self.track)
        self.track_pub.publish(track)
        goal_distance = math.hypot(x-self.points[-1][0], y-self.points[-1][1])
        metrics = dict(cross_track_error=error, heading_error_deg=math.degrees(heading),
                       goal_distance=goal_distance, progress_m=progress)
        for key, value in metrics.items():
            self.metric_pubs[key].publish(Float64(data=value))
        rms = math.sqrt(sum(e*e for e in self.errors) / len(self.errors))
        peak = max(abs(e) for e in self.errors)
        arrow = self.marker(0, Marker.ARROW, (0.2, 1, 0.2))
        arrow.points = [Point(x=x, y=y, z=.3), Point(x=x+3*math.cos(yaw), y=y+3*math.sin(yaw), z=.3)]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = .2, .5, .7
        line = self.marker(1, Marker.LINE_STRIP, (1, .2, 1))
        line.scale.x = .12
        line.points = [Point(x=x, y=y, z=.15), Point(x=px, y=py, z=.15)]
        nearest = self.marker(2, Marker.SPHERE, (1, .2, 1))
        nearest.pose.position = Point(x=px, y=py, z=.15)
        nearest.scale.x = nearest.scale.y = nearest.scale.z = .4
        start = self.marker(3, Marker.SPHERE, (.2, .6, 1))
        start.pose.position = Point(x=self.points[0][0], y=self.points[0][1], z=.2)
        start.scale.x = start.scale.y = start.scale.z = .8
        goal = self.marker(4, Marker.SPHERE, (1, .2, .2))
        goal.pose.position = Point(x=self.points[-1][0], y=self.points[-1][1], z=.2)
        goal.scale.x = goal.scale.y = goal.scale.z = .8
        text = self.marker(5, Marker.TEXT_VIEW_FACING, (1, 1, 1))
        text.pose.position = Point(x=x, y=y, z=3)
        text.scale.z = .8
        text.text = ('CTE: {:+.2f} m | Heading: {:+.1f} deg\n'
                     'Window RMS: {:.2f} m | Max: {:.2f} m (n={})\n'
                     'Along: {:.1f} m | Goal: {:.1f} m | Segment: {}').format(
                         error, math.degrees(heading), rms, peak, len(self.errors), progress, goal_distance, index)
        self.markers_pub.publish(MarkerArray(markers=[arrow, line, nearest, start, goal, text]))
        self.status_pub.publish(String(data='OK: valid localization and reference'))


if __name__ == '__main__':
    rospy.init_node('waypoint_visualizer')
    try:
        node = WaypointVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
