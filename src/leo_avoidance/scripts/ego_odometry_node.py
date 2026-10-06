#!/usr/bin/env python3
"""Build relative odometry from MORAI Ego UDP speed and heading, without GPS."""
import math
import time

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

from leo_avoidance.dead_reckoning import DeadReckoner


class EgoOdometryNode:
    def __init__(self):
        self.integrator = DeadReckoner(
            max_interval_s=float(rospy.get_param('~max_heading_interval_s', .6)))
        self.speed = None
        self.speed_timeout_s = float(rospy.get_param('~speed_timeout_s', .35))
        self.pub = rospy.Publisher('/leo/ego_odom', Odometry, queue_size=1)
        self.subs = [
            rospy.Subscriber('/competition/ego_speed_kmh', Float64, self.on_speed, queue_size=1),
            rospy.Subscriber('/competition/heading_imu', Imu, self.on_heading, queue_size=1)]

    def on_speed(self, message):
        value = message.data / 3.6
        self.speed = (value, time.monotonic()) if math.isfinite(value) and value >= 0 else None

    def on_heading(self, message):
        stamp = message.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        q = message.orientation
        norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if (not math.isfinite(age) or not 0 <= age <= self.speed_timeout_s
                or not math.isfinite(norm) or norm < .5 or norm > 1.5
                or self.speed is None
                or not 0 <= time.monotonic() - self.speed[1] <= self.speed_timeout_s):
            return
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        try:
            x, y, yaw = self.integrator.update(self.speed[0], yaw, stamp)
        except ValueError as error:
            rospy.logwarn_throttle(2., 'Ego odometry unavailable: %s', error)
            return
        odom = Odometry()
        odom.header.stamp = message.header.stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw / 2.)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.)
        odom.twist.twist.linear.x = self.speed[0]
        self.pub.publish(odom)


if __name__ == '__main__':
    rospy.init_node('leo_ego_odometry')
    EgoOdometryNode()
    rospy.spin()
