#!/usr/bin/env python3
"""Alternative UDP owner: same /ctrl_cmd during GPS and LiDAR odometry states."""
import json
import math
import os
import sys
import time

import rospkg
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float64, String

from leo_avoidance.blackout_watchdog import BlackoutWatchdog

sys.path.insert(0, os.path.join(rospkg.RosPack().get_path('control'), 'scripts'))
from competition_udp_control import CompetitionUDPControl  # noqa: E402


class BlackoutUDPControl(CompetitionUDPControl):
    def __init__(self):
        super().__init__()
        timeout = float(rospy.get_param('~fallback_timeout_s', .35))
        distance = float(rospy.get_param('~max_blackout_distance_m', 20.))
        margin = float(rospy.get_param('~stopline_safety_margin_m', 10.))
        entry_speed = float(rospy.get_param('~max_blackout_entry_speed_mps', 2.))
        silence = float(rospy.get_param('~gps_silence_timeout_s', .75))
        self.watchdog = BlackoutWatchdog(
            self.watchdog.command_timeout, self.watchdog.sensor_timeout,
            timeout, distance, margin, entry_speed, silence)
        self.subscribers.extend([
            rospy.Subscriber('/lidar3D', PointCloud2, self.lidar_callback,
                             queue_size=1, buff_size=2 ** 24),
            rospy.Subscriber('/leo/ego_odom', Odometry, self.odom_callback, queue_size=1),
            rospy.Subscriber('/leo/driving_mode', String, self.mode_callback, queue_size=1),
            rospy.Subscriber('/leo/driving_valid', Bool, self.valid_callback, queue_size=1),
            rospy.Subscriber('/leo/avoidance_status', String,
                             self.avoidance_callback, queue_size=1),
            rospy.Subscriber('/competition/ego_speed_kmh', Float64,
                             self.speed_callback, queue_size=1),
            rospy.Subscriber('/leo/signal_status', String,
                             self.guard_callback, queue_size=1)])
        rospy.logwarn('LiDAR odometry fallback uses the same Leo signal guard /ctrl_cmd; one UDP owner.')

    def set_enabled(self, request):
        result = super().set_enabled(request)
        if not request.data:
            with self.lock:
                if isinstance(self.watchdog, BlackoutWatchdog):
                    self.watchdog.disarm()
        return result

    def gps_callback(self, message):
        super().gps_callback(message)
        latitude_deg, longitude_deg = message.latitude, message.longitude
        with self.lock:
            if isinstance(self.watchdog, BlackoutWatchdog):
                self.watchdog.set_raw_gps(latitude_deg, longitude_deg,
                                          message.status, self.stamp(message),
                                          time.monotonic())

    def lidar_callback(self, message):
        stamp = message.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        valid = (message.header.frame_id == 'lidar' and message.width * message.height > 0
                 and len(message.data) > 0 and math.isfinite(age)
                 and 0 <= age <= self.watchdog.fallback_timeout)
        with self.lock:
            self.watchdog.set_lidar(valid, stamp, time.monotonic())

    def odom_callback(self, message):
        stamp = message.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        valid = (message.header.frame_id == 'odom' and math.isfinite(age)
                 and 0 <= age <= self.watchdog.fallback_timeout)
        with self.lock:
            self.watchdog.set_odom(valid, stamp, time.monotonic())

    def mode_callback(self, message):
        with self.lock:
            self.watchdog.set_drive_mode(message.data, time.monotonic())

    def valid_callback(self, message):
        with self.lock:
            self.watchdog.set_drive_valid(message.data, time.monotonic())

    def avoidance_callback(self, message):
        try:
            status = json.loads(message.data)
            valid = status.get('valid') is True
        except (ValueError, TypeError):
            valid = False
        with self.lock:
            self.watchdog.set_avoidance_valid(valid, time.monotonic())

    def speed_callback(self, message):
        with self.lock:
            self.watchdog.set_speed(message.data / 3.6, time.monotonic())

    def guard_callback(self, message):
        try:
            status = json.loads(message.data)
            armed = status.get('armed') is True and status.get('state') != 'hold'
            distance = float(status.get('next_controlled_stop_distance_m',
                                        status.get('stop_distance')))
        except (ValueError, TypeError, KeyError):
            armed, distance = False, math.nan
        with self.lock:
            self.watchdog.set_guard(armed, distance, time.monotonic())


if __name__ == '__main__':
    rospy.init_node('competition_udp_control')
    BlackoutUDPControl().run()
