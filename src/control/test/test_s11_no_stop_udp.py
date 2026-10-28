import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from competition_udp_control_local import CompetitionUDPControl


class S11UDPTests(unittest.TestCase):
    def controller(self):
        n = CompetitionUDPControl.__new__(CompetitionUDPControl)
        n.simulator_s11_no_stop = True
        n.s11_sector = 11
        n.s11_command = (b'cruise', 10.)
        n.s11_speed_kmh, n.s11_steering = 60., .1
        n.encoder = MagicMock()
        n.encoder.encode.return_value = b'fallback'
        n.freshness = MagicMock()
        n.freshness.fresh.return_value = True
        n.lock = threading.Lock()
        return n

    def test_guard_and_watchdog_stops_yield_to_fresh_follower(self):
        n = self.controller()
        self.assertEqual(n.simulator_s11_packet(b'stop', 'BRAKE', 10.1)[0], b'cruise')

    def test_fallback_decelerates_but_releases_at_floor(self):
        n = self.controller()
        for speed in (60., 31., 30., 20.):
            n.s11_speed_kmh = speed
            self.assertEqual(n.simulator_s11_packet(b'stop', 'BRAKE', 11.)[0], b'fallback')
            args = n.encoder.encode.call_args.args
            self.assertEqual(args[0], 1)
            self.assertEqual(args[5], .1)
            self.assertEqual(args[4] > 0., speed > 31.)
            self.assertEqual(args[3] > 0., speed < 30.)

    def test_frozen_speed_cannot_keep_braking(self):
        n = self.controller()
        n.freshness.fresh.return_value = False
        n.simulator_s11_packet(b'stop', 'BRAKE', 10.1)
        args = n.encoder.encode.call_args.args
        self.assertEqual((args[3], args[4]), (.2, 0.))

    def test_sector_entry_discards_previous_sector_command(self):
        n = self.controller()
        n.s11_sector = 10
        n.s11_tracking_callback(SimpleNamespace(data='{"sector":11}'))
        self.assertIsNone(n.s11_command)
        self.assertEqual(n.simulator_s11_packet(b'stop', 'BRAKE', 10.1)[0], b'fallback')

    def test_sector_exit_and_disabled_policy_restore_original_packet(self):
        n = self.controller()
        n.s11_tracking_callback(SimpleNamespace(data='{"sector":12}'))
        self.assertEqual(n.simulator_s11_packet(b'stop', 'BRAKE', 10.1), (b'stop', 'BRAKE'))
        n.s11_sector = 11
        n.simulator_s11_no_stop = False
        self.assertEqual(n.simulator_s11_packet(b'stop', 'BRAKE', 10.1), (b'stop', 'BRAKE'))

    def test_fallback_matches_launch_braking_strength(self):
        import xml.etree.ElementTree as ET
        launch = Path(__file__).resolve().parents[2] / 'unita_launch/launch/semi.launch'
        node = ET.parse(launch).find("node[@name='gps_waypoint_follower']")
        gain = float(node.find("param[@name='sector11_brake_gain']").get('value'))
        maximum = float(node.find("param[@name='sector11_max_cruise_brake']").get('value'))
        n = self.controller()
        for speed, expected in ((60., .65), (40., .36), (35., .16), (31., 0.), (30., 0.)):
            with self.subTest(speed=speed):
                n.s11_speed_kmh = speed
                n.simulator_s11_packet(b'stop', 'BRAKE', 11.)
                brake = n.encoder.encode.call_args.args[4]
                self.assertAlmostEqual(brake, expected)
                self.assertAlmostEqual(brake, min(maximum, gain*max(0., speed-31.)))
