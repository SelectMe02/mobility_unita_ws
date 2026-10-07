"""Check that the UDP gate receives the next controlled stop line distance."""
import importlib.util
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from std_msgs.msg import String


spec = importlib.util.spec_from_file_location(
    'blackout_udp_control_test',
    Path(__file__).parents[1] / 'scripts/blackout_udp_control.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BlackoutControlNodeTests(unittest.TestCase):
    def test_uncontrolled_entrance_uses_next_controlled_signal(self):
        node = module.BlackoutUDPControl.__new__(module.BlackoutUDPControl)
        node.lock = threading.Lock()
        node.watchdog = MagicMock()
        node.guard_callback(String(data=json.dumps({
            'armed': True, 'state': 'clear', 'stop_distance': 7.4,
            'next_controlled_stop_distance_m': 516.4})))
        armed, distance, _ = node.watchdog.set_guard.call_args[0]
        self.assertTrue(armed)
        self.assertAlmostEqual(distance, 516.4)

        node.guard_callback(String(data=json.dumps({
            'armed': True, 'state': 'hold',
            'next_controlled_stop_distance_m': 516.4})))
        armed, _, _ = node.watchdog.set_guard.call_args[0]
        self.assertFalse(armed)


if __name__ == '__main__':
    unittest.main()
