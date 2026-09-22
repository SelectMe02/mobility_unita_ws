#!/usr/bin/env python3
"""UDP-only SIM interface; explicitly requests AutoMode even from Manual."""
import ipaddress
import math
import socket
import threading
import time

import rospy
from morai_msgs.msg import CtrlCmd, GPSMessage
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, String, UInt8
from std_srvs.srv import SetBool, SetBoolResponse
from unita_control.competition_protocol import (
    CompetitionCommandEncoder, StatusFreshness, decode_ego_status)
from unita_control.udp_protocol import CommandWatchdog


class CompetitionUDPControl:
    def __init__(self):
        address = ipaddress.IPv4Address(rospy.get_param('~morai_ip'))
        if address.is_unspecified or address.is_multicast or str(address) == '255.255.255.255':
            raise ValueError('morai_ip must be the SIM PC unicast IPv4')
        self.destination = (str(address), int(rospy.get_param('~morai_port', 9093)))
        status_port = int(rospy.get_param('~status_port', 9092))
        local_port = int(rospy.get_param('~local_port', 0))
        if (not 1 <= self.destination[1] <= 65535 or not 1 <= status_port <= 65535
                or not 0 <= local_port <= 65535 or local_port == status_port):
            raise ValueError('invalid control/status/source UDP port')
        self.heading_source = rospy.get_param('~heading_source', 'ego_status')
        if self.heading_source not in ('ego_status', 'imu'):
            raise ValueError('heading_source must be ego_status or imu')
        self.yaw_offset = float(rospy.get_param('~ego_yaw_offset_deg', 0))
        self.rate = float(rospy.get_param('~send_rate', 20))
        if not math.isfinite(self.yaw_offset) or not math.isfinite(self.rate) or not 1 <= self.rate <= 200:
            raise ValueError('invalid yaw offset / send rate (1..200 Hz)')
        self.encoder = CompetitionCommandEncoder(
            gear=int(rospy.get_param('~gear', 4)),
            max_steering_deg=float(rospy.get_param('~max_steering_deg', 36.25)),
            steering_sign=float(rospy.get_param('~steering_sign', 1)),
            max_speed_kmh=float(rospy.get_param('~max_speed_kmh', 30)))
        self.watchdog = CommandWatchdog(
            float(rospy.get_param('~command_timeout', .5)),
            float(rospy.get_param('~sensor_timeout', 1)))
        self.freshness = StatusFreshness(float(rospy.get_param('~status_timeout', 1)))
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.enabled = True
        self.was_active = False
        self.last_state = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((rospy.get_param('~bind_ip', '0.0.0.0'), local_port))
        self.status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status_sock.bind((rospy.get_param('~bind_ip', '0.0.0.0'), status_port))
        self.status_sock.setblocking(False)
        self.status_pub = rospy.Publisher('/control/udp_status', String, queue_size=1, latch=True)
        self.mode_pub = rospy.Publisher('/control/sim_mode', UInt8, queue_size=1, latch=True)
        self.ego_pub = rospy.Publisher('/competition/ego_status', String, queue_size=1, latch=True)
        self.speed_pub = rospy.Publisher('/competition/ego_speed_kmh', Float64, queue_size=1)
        self.heading_pub = (rospy.Publisher('/competition/heading_imu', Imu, queue_size=1)
                            if self.heading_source == 'ego_status' else None)
        self.subscribers = [
            rospy.Subscriber(rospy.get_param('~ctrl_topic', '/ctrl_cmd'), CtrlCmd, self.command_callback, queue_size=1),
            rospy.Subscriber(rospy.get_param('~gps_topic', '/gps'), GPSMessage, self.gps_callback, queue_size=1)]
        if self.heading_source == 'imu':
            self.subscribers.append(rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1))
        self.enable_service = rospy.Service('~set_enabled', SetBool, self.set_enabled)
        rospy.on_shutdown(self.stopped.set)
        rospy.loginfo('COMPETITION UDP: SIM=%s:%d, Ego status listen=%d, heading=%s, ctrl_mode=2, gear=%d; NO ROS bridge/service',
                      *self.destination, status_port, self.heading_source, self.encoder.gear)
        rospy.loginfo('Automatic takeover enabled. Stop this launch or call ~set_enabled false to return to keyboard mode.')

    def set_enabled(self, request):
        with self.lock:
            self.enabled = request.data
            self.watchdog.set_command(None, time.monotonic())
        return SetBoolResponse(success=True, message='UDP takeover enabled' if request.data else 'UDP takeover disabled; releasing to keyboard')

    def command_callback(self, message):
        try:
            packet = self.encoder.encode(message.longlCmdType, message.velocity,
                                         message.acceleration, message.accel,
                                         message.brake, message.steering)
        except ValueError as error:
            packet = None
            rospy.logwarn_throttle(2, 'Rejected CtrlCmd: %s', error)
        with self.lock:
            self.watchdog.set_command(packet, time.monotonic())

    @staticmethod
    def stamp(message):
        return (message.header.stamp if message.header.stamp != rospy.Time() else rospy.Time.now()).to_sec()

    def gps_callback(self, message):
        values = (message.latitude, message.longitude, message.eastOffset, message.northOffset)
        valid = (all(math.isfinite(value) for value in values) and message.status > 0
                 and -80 <= message.latitude <= 84 and -180 <= message.longitude <= 180
                 and not message.latitude == message.longitude == 0)
        with self.lock:
            self.watchdog.set_sensor('gps', valid, self.stamp(message), time.monotonic())

    def imu_callback(self, message):
        q = message.orientation
        values = (q.x, q.y, q.z, q.w)
        norm = math.sqrt(sum(value * value for value in values))
        valid = (all(math.isfinite(value) for value in values) and math.isfinite(norm)
                 and norm >= 1e-6 and message.orientation_covariance[0] != -1)
        with self.lock:
            self.watchdog.set_sensor('imu', valid, self.stamp(message), time.monotonic())

    def receive_status(self):
        for _ in range(256):
            try:
                packet, source = self.status_sock.recvfrom(4096)
            except BlockingIOError:
                break
            if source[0] != self.destination[0]:
                rospy.logwarn_throttle(5, 'Ignoring Ego status from unexpected IP %s', source[0])
                continue
            try:
                status = decode_ego_status(packet)
            except ValueError as error:
                rospy.logwarn_throttle(5, 'Rejected Ego status: %s', error)
                continue
            received = time.monotonic()
            with self.lock:
                changed = self.freshness.update(status.stamp, received)
            self.mode_pub.publish(UInt8(data=status.mode))
            self.speed_pub.publish(Float64(data=abs(status.speed_kmh)))
            self.ego_pub.publish(String(data='mode={} gear={} speed={:.2f}km/h yaw={:.2f}deg steer={:.2f}deg wheelbase={:.3f}m pos={}'.format(
                status.mode, status.gear, status.speed_kmh, status.yaw_deg, status.steering_deg, status.wheelbase, status.position)))
            if changed and self.heading_pub is not None:
                yaw = math.radians(status.yaw_deg + self.yaw_offset)
                message = Imu()
                # Heading from Ego UDP, NOT the physical IMU sensor. Arrival time
                # shares the GPS bridge clock, including SIM Sync Mode timestamps.
                message.header.stamp = rospy.Time.now()
                message.header.frame_id = 'ego_heading'
                message.orientation.z, message.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                message.angular_velocity_covariance[0] = -1
                message.linear_acceleration_covariance[0] = -1
                self.heading_pub.publish(message)
                with self.lock:
                    self.watchdog.set_sensor('imu', True, message.header.stamp.to_sec(), received)

    def run(self):
        try:
            while not self.stopped.is_set() and not rospy.is_shutdown():
                self.receive_status()
                with self.lock:
                    if self.enabled:
                        if self.freshness.fresh(time.monotonic()):
                            self.was_active = True
                            packet, state = self.watchdog.select(time.monotonic(), rospy.Time.now().to_sec(), self.encoder.stop())
                            # No mode gate: AutoMode=2 is explicitly requested even
                            # when the last reported mode was Manual.
                            self.sock.sendto(packet, self.destination)
                        elif self.was_active:
                            packet, state = self.encoder.stop(), 'BRAKE: missing/stale/frozen Ego UDP status'
                            self.sock.sendto(packet, self.destination)
                        else:
                            # Until the correct Publisher is verified, do not force
                            # AutoMode and do not interfere with SIM I/Q keyboard input.
                            state = 'PAUSED: waiting for first valid Ego UDP status; no control packets'
                    else:
                        if self.was_active:
                            for _ in range(3):
                                self.sock.sendto(self.encoder.stop(), self.destination)
                                self.sock.sendto(self.encoder.release(), self.destination)
                            self.was_active = False
                        state = 'PAUSED: competition takeover disabled; keyboard mode requested'
                self.status_pub.publish(String(data=state))
                if state != self.last_state:
                    rospy.loginfo('%s', state)
                    self.last_state = state
                self.stopped.wait(1 / self.rate)
        finally:
            try:
                if self.was_active:
                    for _ in range(3):
                        self.sock.sendto(self.encoder.stop(), self.destination)
                    for _ in range(3):
                        self.sock.sendto(self.encoder.release(), self.destination)
            finally:
                self.status_sock.close()
                self.sock.close()


if __name__ == '__main__':
    rospy.init_node('competition_udp_control')
    try:
        CompetitionUDPControl().run()
    except rospy.ROSInterruptException:
        pass
    except (ValueError, OSError) as error:
        rospy.logfatal('Competition UDP control failed: %s', error)
        raise SystemExit(1)
