import math
import struct
import importlib.util
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from unita_control.competition_protocol import (
    CompetitionCommandEncoder, StatusFreshness, decode_ego_status)
from unita_control.udp_protocol import CommandEncoder, CommandWatchdog


def status_packet(mode=1, size=229, stamp=(100, 123)):
    packet = bytearray(size)
    packet[:11] = b'#MoraiInfo$'
    struct.pack_into('<i', packet, 11, size - 29)
    struct.pack_into('<ii', packet, 27, *stamp)
    packet[35:37] = bytes((mode, 4))
    struct.pack_into('<f', packet, 37, 10)
    struct.pack_into('<f', packet, 69, 2.7)
    struct.pack_into('<3f', packet, 77, -131.5, -428, 28.5)
    struct.pack_into('<f', packet, 97, 61.3)
    struct.pack_into('<f', packet, 137, -5)
    packet[-2:] = b'\r\n'
    return packet


class CompetitionTests(unittest.TestCase):
    def test_ego_status_never_exposes_simulator_position(self):
        for script in ('competition_udp_control.py', 'competition_udp_control_local.py'):
            spec = importlib.util.spec_from_file_location(
                'control_no_position_' + script[:-3],
                Path(__file__).parents[1] / 'scripts' / script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            packet = status_packet(mode=2)
            struct.pack_into('<3f', packet, 77, -78.13, -546.24, 30.01)
            node = module.CompetitionUDPControl.__new__(module.CompetitionUDPControl)
            node.destination = ('127.0.0.1', 9093)
            node.status_sock = MagicMock()
            node.status_sock.recvfrom.side_effect = [
                (bytes(packet), ('127.0.0.1', 9094)), BlockingIOError()]
            node.lock = threading.Lock()
            node.freshness = StatusFreshness()
            node.mode_pub, node.speed_pub, node.ego_pub = MagicMock(), MagicMock(), MagicMock()
            node.heading_pub = None
            node.diagnostic_log = True
            node.diagnostic_raw_pub, node.diagnostic_status_pub = MagicMock(), MagicMock()
            node.yaw_offset = 0.
            with patch.object(module.rospy.Time, 'now', return_value=module.rospy.Time(100)):
                node.receive_status()
            self.assertFalse(hasattr(node, 'ego_pose_pub'))
            published = ' '.join(str(call) for pub in (node.ego_pub, node.diagnostic_raw_pub,
                                                      node.diagnostic_status_pub)
                                 for call in pub.publish.call_args_list)
            self.assertNotIn('pos', published.replace('position', 'pos'))
            self.assertNotIn(struct.pack('<3f', -78.13, -546.24, 30.01).hex(), published)

    def test_new_command_requests_auto_and_old_command_keeps_mode(self):
        args = (1, 0, 0, .4, 0, .1)
        old = CommandEncoder().encode(*args)
        new = CompetitionCommandEncoder().encode(*args)
        self.assertEqual(new[30], 2)
        self.assertEqual(old[30], 0)
        self.assertEqual(new[:30] + new[31:], old[:30] + old[31:])

    def test_auto_stop_and_keyboard_release(self):
        encoder = CompetitionCommandEncoder()
        self.assertEqual(struct.unpack_from('<3b5f', encoder.stop(), 30), (2, 4, 1, 0, 0, 0, 1, 0))
        self.assertEqual(encoder.release()[30], 1)
        for mode in (2, 3):
            with self.assertRaisesRegex(ValueError, 'longlCmdType=1'):
                encoder.encode(mode, 0, 0, 0, 0, .2)

    def test_both_status_layouts_have_same_heading_fields(self):
        for size in (181, 229):
            status = decode_ego_status(status_packet(size=size))
            self.assertEqual((status.mode, status.gear, status.stamp), (1, 4, (100, 123)))
            self.assertFalse(hasattr(status, 'position'))
            self.assertAlmostEqual(status.yaw_deg, 61.3, places=4)
            self.assertAlmostEqual(status.wheelbase, 2.7, places=4)
            self.assertEqual(status.steering_deg, -5)

    def test_non_finite_status_fields_are_rejected(self):
        for offset in (37, 69, 97, 137):
            for value in (math.nan, math.inf):
                packet = status_packet()
                struct.pack_into('<f', packet, offset, value)
                with self.assertRaises(ValueError):
                    decode_ego_status(packet)

    def test_position_bytes_are_never_read(self):
        packet = status_packet()
        struct.pack_into('<3f', packet, 77, math.nan, math.nan, math.nan)
        decode_ego_status(packet)

    def test_diagnostic_publishers_and_truth_pose_are_gated(self):
        spec = importlib.util.spec_from_file_location('control_log_gate',
            Path(__file__).parents[1] / 'scripts/competition_udp_control.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for enabled in (False, True):
            node = module.CompetitionUDPControl.__new__(module.CompetitionUDPControl)
            node.destination = ('127.0.0.1', 9093)
            node.status_sock = MagicMock()
            node.status_sock.recvfrom.side_effect = [(status_packet(), ('127.0.0.1', 9096)), BlockingIOError()]
            node.lock, node.freshness = threading.Lock(), StatusFreshness()
            node.diagnostic_log, node.yaw_offset = enabled, 0.
            for name in ('mode_pub', 'speed_pub', 'ego_pub', 'diagnostic_status_pub', 'diagnostic_raw_pub'):
                setattr(node, name, MagicMock())
            node.heading_pub = None
            with patch.object(module.rospy.Time, 'now', return_value=module.rospy.Time(100)):
                node.receive_status()
            self.assertEqual(node.diagnostic_raw_pub.publish.called, enabled)
            self.assertEqual(node.diagnostic_status_pub.publish.called, enabled)
            self.assertNotIn('pos=', node.ego_pub.publish.call_args.args[0].data)

    def test_malformed_status_is_rejected(self):
        for packet in (status_packet()[:-1], bytes(229)):
            with self.assertRaises(ValueError):
                decode_ego_status(packet)
        collision = bytearray(181)
        collision[:15] = b'#CollisionData$'
        with self.assertRaisesRegex(ValueError, 'CollisionData'):
            decode_ego_status(collision)

    def test_frozen_status_expires_and_reset_recovers(self):
        tracker = StatusFreshness(.3)
        self.assertFalse(tracker.fresh(10))
        self.assertTrue(tracker.update((100, 0), 10))
        self.assertFalse(tracker.update((100, 0), 10.2))
        self.assertFalse(tracker.fresh(10.4))
        self.assertTrue(tracker.update((1, 0), 10.4))
        self.assertTrue(tracker.fresh(10.5))
        self.assertFalse(tracker.fresh(10.3))
        for timeout in (0, -1, math.inf):
            with self.assertRaises(ValueError):
                StatusFreshness(timeout)

    def test_manual_status_does_not_block_auto_request(self):
        status = decode_ego_status(status_packet(mode=1))
        tracker = StatusFreshness()
        tracker.update(status.stamp, 10)
        watchdog = CommandWatchdog()
        encoder = CompetitionCommandEncoder()
        watchdog.set_command(encoder.encode(1, 0, 0, .4, 0, .1), 10)
        watchdog.set_sensor('gps', True, 100, 10)
        watchdog.set_sensor('imu', True, 100, 10)
        self.assertTrue(tracker.fresh(10.1))
        packet, state = watchdog.select(10.1, 100.1, encoder.stop())
        self.assertTrue(state.startswith('SENDING'))
        self.assertEqual(packet[30], 2)

    def test_competition_retains_limits_and_input_validation(self):
        encoder = CompetitionCommandEncoder(steering_sign=-1)
        packet = encoder.encode(1, 0, 0, .4, 0, 2)
        fields = struct.unpack_from('<5f', packet, 33)
        self.assertEqual((fields[0], fields[1], fields[3], fields[4]), (0, 0, 0, -1))
        self.assertAlmostEqual(fields[2], .4)
        with self.assertRaises(ValueError):
            encoder.encode(1, math.nan, 0, 0, 0, 0)

    def test_ioniq5_wheel_angle_normalization(self):
        encoder = CompetitionCommandEncoder(max_steering_deg=40.0)
        packet = encoder.encode(1, 0, 0, .2, 0, math.radians(20))
        self.assertAlmostEqual(struct.unpack_from('<f', packet, 49)[0], .5)
        self.assertEqual(packet[32], 1)


if __name__ == '__main__':
    unittest.main()
