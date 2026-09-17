import math
import struct
import unittest

from unita_control.udp_protocol import (CommandEncoder, CommandWatchdog, ModeGate,
                                      decode_status_mode, normalize_service_mode)


class ModeTests(unittest.TestCase):
    def test_service_modes_use_different_values_from_udp(self):
        self.assertEqual(normalize_service_mode(1), 1)  # Keyboard
        self.assertEqual(normalize_service_mode(3), 2)  # External
        self.assertEqual(normalize_service_mode(6), 0)  # Built-in
        for mode in (0, 2, 4, 5, 7, -1, 255):
            self.assertEqual(normalize_service_mode(mode), 0)

    def frame(self, mode=2, size=181):
        packet = bytearray(size)
        packet[:11] = b'#MoraiInfo$'
        struct.pack_into('<i', packet, 11, size - 29)
        struct.pack_into('<ii', packet, 27, 100, 123)
        packet[35:37] = bytes((mode, 4))
        packet[-2:] = b'\r\n'
        return packet

    def test_status_base_and_tire_extension(self):
        for size in (181, 229):
            self.assertEqual(decode_status_mode(self.frame(size=size)), (2, (100, 123)))

    def test_reject_malformed_status(self):
        frames = [self.frame()[:-1], bytearray(181)]
        for offset, value in ((11, 200), (31, 1000000000)):
            frame = self.frame()
            struct.pack_into('<i', frame, offset, value)
            frames.append(frame)
        for frame in frames:
            with self.assertRaises(ValueError):
                decode_status_mode(frame)

    def test_manual_builtin_unknown_never_permit_even_brakes(self):
        gate = ModeGate()
        self.assertFalse(gate.select(100)[0])
        for mode in (1, 3, 0, 255):
            gate.update(mode, (100, 0), 100)
            self.assertFalse(gate.select(100.1)[0])
        gate.update(2, (100, 1), 100.1)
        self.assertTrue(gate.select(100.2)[0])
        gate.update(1, (100, 2), 100.2)
        self.assertFalse(gate.select(100.2)[0])

    def test_frozen_status_timeout_and_recovery(self):
        gate = ModeGate(.3)
        gate.update(2, (100, 0), 100)
        gate.update(2, (100, 0), 100.2)
        self.assertFalse(gate.select(100.4)[0])
        gate.update(2, (100, 1), 100.4)
        self.assertTrue(gate.select(100.4)[0])
        self.assertFalse(gate.select(100.8)[0])

    def test_mode_change_invalidates_previous_command(self):
        gate = ModeGate()
        self.assertTrue(gate.update(2, (100, 0), 100))
        self.assertFalse(gate.update(2, (100, 1), 100.1))
        self.assertTrue(gate.update(1, (100, 2), 100.2))
        self.assertTrue(gate.update(2, (100, 3), 100.3))


class ProtocolTests(unittest.TestCase):
    def test_wire_fields(self):
        packet = CommandEncoder().encode(2, 20, 0, 0, 0, math.radians(18.125))
        self.assertEqual(len(packet), 55)
        self.assertEqual(packet[:14], b'#MoraiCtrlCmd$')
        self.assertEqual(struct.unpack_from('<4i', packet, 14), (23, 0, 0, 0))
        self.assertEqual(struct.unpack_from('<3b', packet, 30), (0, 4, 2))
        self.assertEqual(struct.unpack_from('<5f', packet, 33), (20, 0, 0, 0, .5))
        self.assertEqual(packet[-2:], b'\r\n')

    def test_limits_and_steering_sign(self):
        packet = CommandEncoder(steering_sign=-1).encode(2, 40, 0, 0, 0, 2)
        self.assertEqual(struct.unpack_from('<5f', packet, 33), (30, 0, 0, 0, -1))

    def test_stop_and_zero_velocity(self):
        encoder = CommandEncoder()
        self.assertEqual(encoder.encode(2, 0, 0, 0, 0, .5), encoder.stop())
        self.assertEqual(struct.unpack_from('<3b5f', encoder.stop(), 30), (0, 4, 1, 0, 0, 0, 1, 0))

    def test_q_manual_transition_between_poll_and_packet_is_not_overridden(self):
        encoder = CommandEncoder()
        gate = ModeGate()
        gate.update(normalize_service_mode(3), 100, 100)
        self.assertTrue(gate.select(100.01)[0])
        # Q changes the SIM to manual after the last external-mode response.
        # The next command (or watchdog brake) must not request external mode.
        for packet in (encoder.encode(2, 20, 0, 0, 0, .1), encoder.stop()):
            sim_mode = 1
            requested_mode = packet[30]
            sim_mode = {1: 1, 2: 3, 3: 6}.get(requested_mode, sim_mode)
            self.assertEqual(sim_mode, 1)
        gate.update(normalize_service_mode(1), 100.02, 100.02)
        self.assertFalse(gate.select(100.02)[0])

    def test_throttle_acceleration_modes(self):
        encoder = CommandEncoder()
        packet = encoder.encode(1, 0, 0, .2, .3, 0)
        self.assertEqual(packet[32], 1)
        self.assertAlmostEqual(struct.unpack_from('<f', packet, 41)[0], .2)
        self.assertAlmostEqual(struct.unpack_from('<f', packet, 45)[0], .3)
        packet = encoder.encode(3, 0, 1.5, 0, 0, 0)
        self.assertEqual(struct.unpack_from('<f', packet, 37)[0], 1.5)

    def test_invalid_commands(self):
        encoder = CommandEncoder()
        for arguments in ((0,20,0,0,0,0), (2,float('nan'),0,0,0,0),
                          (2,20,0,0,0,float('inf')), (1,0,0,2,0,0),
                          (1,0,0,0,-1,0), (2,-20,0,0,0,0)):
            with self.assertRaises(ValueError):
                encoder.encode(*arguments)

    def test_invalid_configuration(self):
        for kwargs in ({'gear':6}, {'max_steering_deg':0}, {'steering_sign':0},
                       {'max_speed_kmh':float('inf')}):
            with self.assertRaises(ValueError):
                CommandEncoder(**kwargs)


class WatchdogTests(unittest.TestCase):
    def ready(self):
        watchdog = CommandWatchdog()
        watchdog.set_command(b'drive', 100)
        watchdog.set_sensor('gps', True, 10, 100)
        watchdog.set_sensor('imu', True, 10, 100)
        return watchdog

    def test_no_command_or_missing_sensor(self):
        w = CommandWatchdog()
        self.assertEqual(w.select(100, 10, b'stop')[0], b'stop')
        w.set_command(b'drive',100)
        self.assertEqual(w.select(100, 10, b'stop')[0], b'stop')

    def test_command_timeout_and_invalid_replacement(self):
        w = self.ready()
        self.assertEqual(w.select(100.1,10.1,b'stop')[0],b'drive')
        self.assertEqual(w.select(100.6,10.1,b'stop')[0],b'stop')
        w.set_command(None,100)
        self.assertEqual(w.select(100.1,10.1,b'stop')[0],b'stop')

    def test_sensor_loss_stale_clock_and_recovery(self):
        w = self.ready()
        w.set_command(b'drive',101.2)
        self.assertEqual(w.select(101.2,10,b'stop')[0],b'stop')
        w.set_sensor('gps',True,11.2,101.2)
        w.set_sensor('imu',True,11.2,101.2)
        self.assertEqual(w.select(101.2,11.2,b'stop')[0],b'drive')
        self.assertEqual(w.select(101.2,10,b'stop')[0],b'stop')
        w.set_sensor('gps',False,11.2,101.2)
        self.assertEqual(w.select(101.2,11.2,b'stop')[0],b'stop')


if __name__ == '__main__':
    unittest.main()
