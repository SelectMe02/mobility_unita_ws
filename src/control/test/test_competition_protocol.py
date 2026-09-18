import math
import struct
import unittest

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
    def test_new_command_requests_auto_and_old_command_keeps_mode(self):
        args = (2, 10, 0, 0, 0, .1)
        old = CommandEncoder().encode(*args)
        new = CompetitionCommandEncoder().encode(*args)
        self.assertEqual(new[30], 2)
        self.assertEqual(old[30], 0)
        self.assertEqual(new[:30] + new[31:], old[:30] + old[31:])

    def test_auto_stop_and_keyboard_release(self):
        encoder = CompetitionCommandEncoder()
        self.assertEqual(struct.unpack_from('<3b5f', encoder.stop(), 30), (2, 4, 1, 0, 0, 0, 1, 0))
        self.assertEqual(encoder.release()[30], 1)
        self.assertEqual(encoder.encode(2, 0, 0, 0, 0, .2), encoder.stop())

    def test_both_status_layouts_have_same_heading_fields(self):
        for size in (181, 229):
            status = decode_ego_status(status_packet(size=size))
            self.assertEqual((status.mode, status.gear, status.stamp), (1, 4, (100, 123)))
            self.assertEqual(status.position, (-131.5, -428, 28.5))
            self.assertAlmostEqual(status.yaw_deg, 61.3, places=4)
            self.assertAlmostEqual(status.wheelbase, 2.7, places=4)
            self.assertEqual(status.steering_deg, -5)

    def test_non_finite_status_fields_are_rejected(self):
        for offset in (37, 69, 77, 81, 85, 97, 137):
            for value in (math.nan, math.inf):
                packet = status_packet()
                struct.pack_into('<f', packet, offset, value)
                with self.assertRaises(ValueError):
                    decode_ego_status(packet)

    def test_malformed_status_is_rejected(self):
        for packet in (status_packet()[:-1], bytes(229)):
            with self.assertRaises(ValueError):
                decode_ego_status(packet)

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
        watchdog.set_command(encoder.encode(2, 10, 0, 0, 0, .1), 10)
        watchdog.set_sensor('gps', True, 100, 10)
        watchdog.set_sensor('imu', True, 100, 10)
        self.assertTrue(tracker.fresh(10.1))
        packet, state = watchdog.select(10.1, 100.1, encoder.stop())
        self.assertTrue(state.startswith('SENDING'))
        self.assertEqual(packet[30], 2)

    def test_competition_retains_limits_and_input_validation(self):
        encoder = CompetitionCommandEncoder(steering_sign=-1)
        packet = encoder.encode(2, 40, 0, 0, 0, 2)
        self.assertEqual(struct.unpack_from('<5f', packet, 33), (30, 0, 0, 0, -1))
        with self.assertRaises(ValueError):
            encoder.encode(2, math.nan, 0, 0, 0, 0)


if __name__ == '__main__':
    unittest.main()
