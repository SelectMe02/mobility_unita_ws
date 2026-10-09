#!/usr/bin/env python3
"""Record pose sources and header stamps to CSV for blackout drift analysis.

One row per message: topic, receive time (ROS now), header stamp, x, y, yaw,
plus the driving mode/travel from /leo/driving_status. Analyse the file with
odometry_drift_analysis.py.
"""
import csv
import json
import math
import sys
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Float64, String


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Recorder:
    def __init__(self, path):
        self.file = open(path, 'w', newline='')
        self.out = csv.writer(self.file)
        # rospy runs each subscription in its own thread.
        self.lock = threading.Lock()
        self.out.writerow(['topic', 'recv_s', 'stamp_s', 'x', 'y', 'yaw',
                           'value', 'mode', 'travelled_m'])
        self.mode = ''
        self.travelled = ''
        rospy.Subscriber('/localization/pose', PoseStamped, self.pose('/localization/pose'))
        rospy.Subscriber('/competition/ego_pose', PoseStamped, self.pose('/competition/ego_pose'))
        rospy.Subscriber('/leo/driving_pose', PoseStamped, self.pose('/leo/driving_pose'))
        rospy.Subscriber('/leo/ego_odom', Odometry, self.odom)
        rospy.Subscriber('/competition/heading_imu', Imu, self.imu)
        rospy.Subscriber('/lidar3D', PointCloud2, self.cloud, queue_size=5)
        rospy.Subscriber('/competition/ego_speed_kmh', Float64, self.speed)
        rospy.Subscriber('/leo/driving_status', String, self.status)
        rospy.on_shutdown(self.file.close)

    def row(self, topic, stamp, x='', y='', yaw='', value=''):
        line = [topic, '%.4f' % rospy.Time.now().to_sec(),
                '%.4f' % stamp if stamp else '', x, y, yaw, value]
        with self.lock:
            self.out.writerow(line + [self.mode, self.travelled])

    def pose(self, topic):
        def callback(message):
            p = message.pose.position
            self.row(topic, message.header.stamp.to_sec(), '%.3f' % p.x, '%.3f' % p.y,
                     '%.5f' % yaw_of(message.pose.orientation))
        return callback

    def odom(self, message):
        p = message.pose.pose.position
        self.row('/leo/ego_odom', message.header.stamp.to_sec(), '%.3f' % p.x,
                 '%.3f' % p.y, '%.5f' % yaw_of(message.pose.pose.orientation),
                 '%.3f' % message.twist.twist.linear.x)

    def imu(self, message):
        self.row('/competition/heading_imu', message.header.stamp.to_sec(),
                 yaw='%.5f' % yaw_of(message.orientation))

    def cloud(self, message):
        self.row('/lidar3D', message.header.stamp.to_sec())

    def speed(self, message):
        self.row('/competition/ego_speed_kmh', None, value='%.3f' % message.data)

    def status(self, message):
        try:
            data = json.loads(message.data)
        except ValueError:
            return
        self.mode = data.get('mode', '')
        self.travelled = '%.2f' % data.get('travelled_m', 0.)
        map_pose = data.get('map_pose')
        if map_pose:
            self.row('map_pose', None, '%.3f' % map_pose[0], '%.3f' % map_pose[1],
                     '%.5f' % map_pose[2])


if __name__ == '__main__':
    rospy.init_node('odometry_drift_recorder', anonymous=True)
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('__') else 'drift.csv'
    Recorder(path)
    rospy.loginfo('Recording blackout drift diagnostics to %s', path)
    rospy.spin()
