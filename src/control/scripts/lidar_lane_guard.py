#!/usr/bin/env python3
"""Hold or soften waypoint commands when LiDAR sees a painted road line."""

import copy
import json
import math
import threading
import time

import numpy as np
import rospy
from morai_msgs.msg import CtrlCmd
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String

from unita_control.lidar_lane_guard import choose_steering, extract_marking_lines


class LidarLaneGuard:
    def __init__(self):
        self.lock = threading.Lock()
        self.command = None
        self.command_at = None
        self.lines = []
        self.cloud_at = None
        self.speed = None
        self.speed_at = None
        self.max_speed = float(rospy.get_param('~max_speed_kmh', 20.0))
        self.uncertain_speed = float(rospy.get_param('~uncertain_speed_kmh', 12.0))
        self.timeout = float(rospy.get_param('~sensor_timeout', 0.5))
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.max_speed, self.uncertain_speed, self.timeout))
                or self.uncertain_speed > self.max_speed):
            raise ValueError('invalid LiDAR guard speed/timeout')
        self.publisher = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.status = rospy.Publisher('/lane_guard/status', String, queue_size=1,
                                      latch=True)
        rospy.Subscriber('/ctrl_cmd_unchecked', CtrlCmd, self.command_cb, queue_size=1)
        rospy.Subscriber('/lidar3D', PointCloud2, self.cloud_cb, queue_size=1,
                         buff_size=4000000)
        rospy.Subscriber('/competition/ego_speed_kmh', Float64, self.speed_cb,
                         queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self.publish_guarded)

    def command_cb(self, message):
        with self.lock:
            self.command, self.command_at = message, time.monotonic()

    def speed_cb(self, message):
        with self.lock:
            self.speed, self.speed_at = float(message.data), time.monotonic()

    def cloud_cb(self, message):
        fields = {field.name: field.offset for field in message.fields}
        if (message.is_bigendian or message.point_step < 16
                or not all(name in fields for name in ('x', 'y', 'z', 'intensity'))):
            with self.lock:
                self.lines, self.cloud_at = [], None
            return
        dtype = np.dtype({'names': ('x', 'y', 'z', 'intensity'),
                          'formats': ('<f4', '<f4', '<f4', '<f4'),
                          'offsets': tuple(fields[name] for name in
                                           ('x', 'y', 'z', 'intensity')),
                          'itemsize': message.point_step})
        raw = np.frombuffer(message.data, dtype=dtype,
                            count=message.width * message.height)
        points = np.column_stack(tuple(raw[name] for name in
                                       ('x', 'y', 'z', 'intensity')))
        try:
            lines = extract_marking_lines(points)
        except (ValueError, TypeError) as error:
            rospy.logwarn_throttle(3, 'LiDAR lane extraction failed: %s', error)
            with self.lock:
                self.lines, self.cloud_at = [], None
            return
        with self.lock:
            self.lines, self.cloud_at = lines, time.monotonic()

    @staticmethod
    def full_stop():
        command = CtrlCmd()
        command.longlCmdType = 1
        command.accel = 0.0
        command.brake = 1.0
        command.velocity = 0.0
        command.steering = 0.0
        return command

    def publish_guarded(self, _event):
        now = time.monotonic()
        with self.lock:
            command, command_at = self.command, self.command_at
            lines, cloud_at = self.lines, self.cloud_at
            speed, speed_at = self.speed, self.speed_at
        stale = (command_at is None or cloud_at is None or speed_at is None
                 or now - command_at > self.timeout
                 or now - cloud_at > self.timeout
                 or now - speed_at > self.timeout)
        if stale or speed is None or not math.isfinite(speed):
            self.publisher.publish(self.full_stop())
            self.status.publish(String(data='BRAKE: stale command, LiDAR or speed'))
            return
        steering, clearance = choose_steering(command.steering, lines)
        if clearance < 0.0:
            self.publisher.publish(self.full_stop())
            self.status.publish(String(data='BRAKE: predicted line intersection'))
            return

        guarded = copy.deepcopy(command)
        guarded.steering = steering
        cap = self.max_speed if lines else self.uncertain_speed
        if clearance < 0.35:
            cap = min(cap, 8.0)
        elif clearance < 1.0:
            cap = min(cap, 15.0)
        if guarded.longlCmdType == 2:
            guarded.velocity = min(guarded.velocity, cap)
        if speed >= cap - 1.0:
            guarded.longlCmdType = 1
            guarded.velocity = 0.0
            guarded.accel = 0.0
            guarded.brake = max(guarded.brake,
                                min(0.45, max(0.0, speed - cap) * 0.14))
        self.publisher.publish(guarded)
        detail = {'state': 'GUARD' if lines else 'UNCERTAIN',
                  'lines': len(lines), 'speed_cap_kmh': cap,
                  'clearance_m': round(clearance, 2) if math.isfinite(clearance) else None,
                  'steering_delta_rad': round(steering - command.steering, 3)}
        self.status.publish(String(data=json.dumps(detail, sort_keys=True)))


if __name__ == '__main__':
    rospy.init_node('lidar_lane_guard')
    try:
        LidarLaneGuard()
        rospy.spin()
    except (ValueError, rospy.ROSInterruptException) as error:
        rospy.logfatal('LiDAR lane guard failed: %s', error)
        raise SystemExit(1)
