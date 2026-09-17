#!/usr/bin/env python3
"""Publish planar map pose from MORAI GPS and IMU orientation."""
import math
import threading
import time

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from morai_msgs.msg import GPSMessage
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from unita_localization.estimator import GPSIMUEstimator


class GPSIMULocalization:
    def __init__(self):
        self.lock = threading.Lock()
        self.estimator = GPSIMUEstimator(
            utm_zone=int(rospy.get_param('~utm_zone', 52)),
            south=rospy.get_param('~utm_south', False),
            timeout=float(rospy.get_param('~sensor_timeout', 1.0)),
            max_skew=float(rospy.get_param('~max_sensor_skew', 0.2)),
            yaw_offset=float(rospy.get_param('~yaw_offset', 0.0)),
            require_gps_fix=rospy.get_param('~require_gps_fix', True))
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        if not self.map_frame or not self.base_frame or self.map_frame == self.base_frame:
            raise ValueError('map_frame and base_frame must be nonempty and distinct')
        rate = float(rospy.get_param('~publish_rate', 20.0))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('publish_rate must be positive and finite')
        self.pose_pub = rospy.Publisher(
            rospy.get_param('~pose_topic', '/localization/pose'), PoseStamped, queue_size=1)
        self.valid_pub = rospy.Publisher(
            rospy.get_param('~valid_topic', '/localization/valid'), Bool, queue_size=1)
        self.tf = (tf2_ros.TransformBroadcaster()
                   if rospy.get_param('~publish_tf', False) else None)
        self.gps_sub = rospy.Subscriber(rospy.get_param('~gps_topic', '/gps'),
                                        GPSMessage, self.gps_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber(rospy.get_param('~imu_topic', '/imu'),
                                        Imu, self.imu_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.publish, reset=True)
        rospy.loginfo('GPS/IMU localization: %s + %s -> %s (frame=%s)',
                      self.gps_sub.resolved_name, self.imu_sub.resolved_name,
                      self.pose_pub.resolved_name, self.map_frame)

    @staticmethod
    def stamp(message):
        # Accept unstamped sources, but use receive time only in that case.
        return (message.header.stamp if message.header.stamp != rospy.Time()
                else rospy.Time.now()).to_sec()

    def gps_callback(self, message):
        with self.lock:
            try:
                self.estimator.update_gps(
                    message.latitude, message.longitude, message.eastOffset,
                    message.northOffset, message.status, self.stamp(message), time.monotonic())
            except ValueError as error:
                rospy.logwarn_throttle(2, 'Invalid GPS: %s', error)
        if message.eastOffset == message.northOffset == 0:
            rospy.logwarn_throttle(30, 'GPS map offsets are zero; verify the MORAI map origin.')

    def imu_callback(self, message):
        q = message.orientation
        with self.lock:
            try:
                self.estimator.update_imu(
                    (q.x, q.y, q.z, q.w), message.orientation_covariance[0] != -1,
                    self.stamp(message), time.monotonic())
            except ValueError as error:
                rospy.logwarn_throttle(2, 'Invalid IMU: %s', error)

    def publish(self, event):
        with self.lock:
            state, reason = self.estimator.snapshot(rospy.Time.now().to_sec(), time.monotonic())
        self.valid_pub.publish(Bool(data=state is not None))
        if state is None:
            rospy.logwarn_throttle(2, 'Localization unavailable: %s', reason)
            return
        x, y, yaw, stamp = state
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time.from_sec(stamp)
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        self.pose_pub.publish(pose)
        if self.tf is not None:
            transform = TransformStamped()
            transform.header = pose.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.rotation = pose.pose.orientation
            self.tf.sendTransform(transform)


if __name__ == '__main__':
    rospy.init_node('gps_imu_localization')
    try:
        node = GPSIMULocalization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except ValueError as error:
        rospy.logfatal('Invalid localization configuration: %s', error)
        raise SystemExit(1)
