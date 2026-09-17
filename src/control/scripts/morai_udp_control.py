#!/usr/bin/env python3
"""Forward ROS CtrlCmd to MORAI's UDP Cmd Control receiving endpoint."""
import ipaddress
import math
import socket
import threading
import time

import rospy
from morai_msgs.msg import CtrlCmd, GPSMessage
from morai_msgs.srv import MoraiEventCmdSrv, MoraiEventCmdSrvRequest
from sensor_msgs.msg import Imu
from std_msgs.msg import String, UInt8
from unita_control.udp_protocol import (CommandEncoder, CommandWatchdog, ModeGate,
                                      decode_status_mode, normalize_service_mode)


class MoraiUDPControl:
    def __init__(self):
        self.encoder = CommandEncoder(
            gear=int(rospy.get_param('~gear', 4)),
            max_steering_deg=float(rospy.get_param('~max_steering_deg', 36.25)),
            steering_sign=float(rospy.get_param('~steering_sign', 1.0)),
            max_speed_kmh=float(rospy.get_param('~max_speed_kmh', 30)))
        self.watchdog = CommandWatchdog(
            command_timeout=float(rospy.get_param('~command_timeout', .5)),
            sensor_timeout=float(rospy.get_param('~sensor_timeout', 1)))
        self.rate = float(rospy.get_param('~send_rate', 20))
        if not math.isfinite(self.rate) or not 1 <= self.rate <= 200:
            raise ValueError('send_rate must be 1..200 Hz')
        ip = rospy.get_param('~morai_ip', '127.0.0.1')
        address = ipaddress.IPv4Address(ip)
        if address.is_unspecified or address.is_multicast or str(address) == '255.255.255.255':
            raise ValueError('morai_ip must be the SIM PC unicast IPv4 address')
        port = int(rospy.get_param('~morai_port', 9093))
        local_port = int(rospy.get_param('~local_port', 0))
        if not 1 <= port <= 65535 or not 0 <= local_port <= 65535:
            raise ValueError('invalid UDP port')
        self.destination = (str(address), port)
        self.mode_gate = ModeGate(float(rospy.get_param('~status_timeout', .3)))
        self.mode_source = rospy.get_param('~mode_source', 'service')
        if self.mode_source not in ('service', 'udp'):
            raise ValueError('mode_source must be service or udp')
        self.mode_service = rospy.get_param('~mode_service', '/Service_MoraiEventCmd')
        status_port = int(rospy.get_param('~status_port', 9092))
        if not 1 <= status_port <= 65535 or status_port == local_port:
            raise ValueError('status_port must be 1..65535 and differ from local_port')
        self.status_sock = None
        if self.mode_source == 'udp':
            self.status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.status_sock.bind((rospy.get_param('~bind_ip', '0.0.0.0'), status_port))
            self.status_sock.setblocking(False)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((rospy.get_param('~bind_ip', '0.0.0.0'), local_port))
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.stop_packet = self.encoder.stop()
        self.last_state = None
        self.status = rospy.Publisher('/control/udp_status', String, queue_size=1, latch=True)
        self.mode_pub = rospy.Publisher('/control/sim_mode', UInt8, queue_size=1, latch=True)
        self.subs = [
            rospy.Subscriber(rospy.get_param('~ctrl_topic', '/ctrl_cmd'), CtrlCmd, self.command_callback, queue_size=1),
            rospy.Subscriber(rospy.get_param('~gps_topic', '/gps'), GPSMessage, self.gps_callback, queue_size=1),
            rospy.Subscriber(rospy.get_param('~imu_topic', '/imu'), Imu, self.imu_callback, queue_size=1)]
        rospy.on_shutdown(self.stopped.set)
        rospy.loginfo('MORAI UDP control: %s -> %s:%d, source port=%d, gear=%d, max steering=%.2f deg',
                      self.subs[0].resolved_name, ip, port, self.sock.getsockname()[1],
                      self.encoder.gear, math.degrees(self.encoder.max_steering_rad))
        if self.mode_source == 'udp':
            rospy.loginfo('Reading SIM mode from Ego Vehicle Status UDP port %d', status_port)
        else:
            rospy.loginfo('Reading SIM mode from %s with option=0 (query only); no status UDP required',
                          self.mode_service)
            self.mode_thread = threading.Thread(target=self.query_service_mode, daemon=True)
            self.mode_thread.start()
        if address.is_loopback:
            rospy.logwarn('morai_ip is loopback; use the SIM PC IP if MORAI runs on another PC.')

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
        return (message.header.stamp if message.header.stamp != rospy.Time()
                else rospy.Time.now()).to_sec()

    def gps_callback(self, message):
        values = (message.latitude, message.longitude, message.eastOffset, message.northOffset)
        valid = (all(math.isfinite(v) for v in values) and message.status > 0
                 and -80 <= message.latitude <= 84 and -180 <= message.longitude <= 180
                 and not message.latitude == message.longitude == 0)
        with self.lock:
            self.watchdog.set_sensor('gps', valid, self.stamp(message), time.monotonic())

    def imu_callback(self, message):
        q = message.orientation
        values = (q.x, q.y, q.z, q.w)
        norm = math.sqrt(sum(v*v for v in values))
        valid = (message.orientation_covariance[0] != -1 and math.isfinite(norm)
                 and norm >= 1e-6 and all(math.isfinite(v) for v in values))
        with self.lock:
            self.watchdog.set_sensor('imu', valid, self.stamp(message), time.monotonic())

    def update_mode(self, mode, stamp, received, publish=True):
        with self.lock:
            if self.mode_gate.update(mode, stamp, received):
                self.watchdog.set_command(None, received)
        if publish:
            self.mode_pub.publish(UInt8(data=mode))

    def query_service_mode(self):
        # Calls may block on a lost rosbridge response. The separate send loop
        # still expires the mode and stops sending; this thread is a daemon.
        while not self.stopped.is_set() and not rospy.is_shutdown():
            try:
                rospy.wait_for_service(self.mode_service, timeout=.2)
                query = MoraiEventCmdSrvRequest()
                query.request.option = 0  # No mode/gear/lamps/pause write bits.
                query.request.gear = -1
                started = time.monotonic()
                result = rospy.ServiceProxy(self.mode_service, MoraiEventCmdSrv)(query)
                received = time.monotonic()
                if (not self.stopped.is_set() and not rospy.is_shutdown()
                        and received - started <= self.mode_gate.timeout):
                    mode = normalize_service_mode(result.response.ctrl_mode)
                    self.update_mode(mode, received, received)
            except (rospy.ROSException, rospy.ServiceException, OSError) as error:
                if not self.stopped.is_set() and not rospy.is_shutdown():
                    rospy.logwarn_throttle(5, 'SIM mode query failed: %s', error)
            self.stopped.wait(1.0 / self.rate)

    def receive_status(self, publish=True):
        if self.status_sock is None:
            return
        # Bounded drain keeps the send/watchdog loop responsive under a burst.
        for _ in range(256):
            try:
                packet, source = self.status_sock.recvfrom(4096)
            except BlockingIOError:
                break
            if source[0] != self.destination[0]:
                rospy.logwarn_throttle(5, 'Ignoring Ego status from unexpected IP %s', source[0])
                continue
            try:
                mode, stamp = decode_status_mode(packet)
            except ValueError as error:
                rospy.logwarn_throttle(5, 'Rejected Ego status: %s', error)
                continue
            self.update_mode(mode, stamp, time.monotonic(), publish)

    def run(self):
        try:
            # Wall-clock loop also brakes when /clock stalls.
            while not self.stopped.is_set() and not rospy.is_shutdown():
                self.receive_status()
                with self.lock:
                    permitted, state = self.mode_gate.select(time.monotonic())
                    if permitted:
                        packet, state = self.watchdog.select(time.monotonic(), rospy.Time.now().to_sec(), self.stop_packet)
                        self.sock.sendto(packet, self.destination)
                    else:
                        self.watchdog.set_command(None, time.monotonic())
                self.status.publish(String(data=state))
                if state != self.last_state:
                    rospy.loginfo('%s', state)
                    self.last_state = state
                self.stopped.wait(1.0 / self.rate)
        finally:
            # Never force AutoMode back on after the driver has selected manual.
            try:
                self.receive_status(publish=False)
                with self.lock:
                    if self.mode_gate.select(time.monotonic())[0]:
                        for _ in range(3):
                            self.sock.sendto(self.stop_packet, self.destination)
            except OSError:
                pass
            if self.status_sock is not None:
                self.status_sock.close()
            self.sock.close()


if __name__ == '__main__':
    rospy.init_node('morai_udp_control')
    try:
        node = MoraiUDPControl()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except (ValueError, OSError) as error:
        rospy.logfatal('MORAI UDP control failed: %s', error)
        raise SystemExit(1)
