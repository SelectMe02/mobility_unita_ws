#!/usr/bin/env python3
"""One receiver process per camera/GPS/IMU, configured by sensor name."""
import math
import socket

import cv2
import numpy as np
import rospy
from morai_msgs.msg import GPSMessage
from sensor_msgs.msg import Image, Imu

from sensor_bridge.protocol import CameraAssembler, parse_gps, parse_imu


class SensorReceiver:
    def __init__(self):
        name = rospy.get_param('~sensor')
        config = rospy.get_param('~sensors/' + name)
        self.kind = config['kind']
        self.frame_id = config['frame_id']
        self.use_sensor_stamp = rospy.get_param('~use_sensor_stamp', False)
        self.assembler = CameraAssembler()
        self.east_offset = float(rospy.get_param('~east_offset', 0.0))
        self.north_offset = float(rospy.get_param('~north_offset', 0.0))
        if not all(math.isfinite(v) for v in (self.east_offset, self.north_offset)):
            raise ValueError('GPS map offsets must be finite')
        if self.kind == 'gps' and self.east_offset == self.north_offset == 0:
            rospy.logwarn('GPS UDP has no map offsets; east_offset/north_offset '
                          'are zero. Set map offsets before waypoint driving.')
        message_type = {'camera': Image, 'imu': Imu, 'gps': GPSMessage}[self.kind]
        self.publisher = rospy.Publisher(config['topic'], message_type, queue_size=1)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rospy.on_shutdown(self.socket.close)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.socket.settimeout(1.0)
        bind_ip = rospy.get_param('~bind_ip', '0.0.0.0')
        self.source_ip = rospy.get_param('~source_ip', '')
        port = int(config['destination_port'])
        self.socket.bind((bind_ip, port))
        rospy.loginfo('%s: UDP %s:%d -> %s (%s)', name, bind_ip, port,
                      config['topic'], message_type._type)

    def header(self, message, stamp=None):
        message.header.frame_id = self.frame_id
        message.header.stamp = (rospy.Time(*stamp)
                                if self.use_sensor_stamp and stamp is not None
                                else rospy.Time.now())

    def publish_packet(self, packet):
        if self.kind == 'camera':
            result = self.assembler.feed(packet)
            if result is None:
                return
            stamp, jpeg = result
            pixels = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError('JPEG decoding failed')
            message = Image()
            self.header(message, stamp)
            message.height, message.width = pixels.shape[:2]
            message.encoding = 'bgr8'
            message.is_bigendian = 0
            message.step = message.width * 3
            message.data = pixels.tobytes()
            self.publisher.publish(message)
        elif self.kind == 'imu':
            stamp, values = parse_imu(packet)
            message = Imu()
            self.header(message, stamp)
            (message.orientation.w, message.orientation.x,
             message.orientation.y, message.orientation.z) = values[:4]
            (message.angular_velocity.x, message.angular_velocity.y,
             message.angular_velocity.z) = values[4:7]
            (message.linear_acceleration.x, message.linear_acceleration.y,
             message.linear_acceleration.z) = values[7:]
            self.publisher.publish(message)
        else:
            for lat, lon, altitude, quality in parse_gps(packet):
                message = GPSMessage()
                self.header(message)
                message.latitude, message.longitude = lat, lon
                message.altitude = altitude
                message.eastOffset = self.east_offset
                message.northOffset = self.north_offset
                message.status = quality
                self.publisher.publish(message)

    def run(self):
        while not rospy.is_shutdown():
            try:
                packet, address = self.socket.recvfrom(65535)
            except socket.timeout:
                rospy.logwarn_throttle(10, 'No UDP packet for %s in the last second',
                                       self.publisher.resolved_name)
                continue
            except OSError:
                if rospy.is_shutdown():
                    break
                raise
            if self.source_ip and address[0] != self.source_ip:
                continue
            try:
                self.publish_packet(packet)
            except (ValueError, UnicodeError, cv2.error) as error:
                rospy.logwarn_throttle(5, 'Discarding invalid %s packet: %s',
                                       self.kind, error)


if __name__ == '__main__':
    rospy.init_node('udp_sensor_bridge')
    try:
        SensorReceiver().run()
    except rospy.ROSInterruptException:
        pass
    except (OSError, ValueError, KeyError) as error:
        rospy.logfatal('Sensor bridge startup/runtime failure: %s', error)
        raise SystemExit(1)
