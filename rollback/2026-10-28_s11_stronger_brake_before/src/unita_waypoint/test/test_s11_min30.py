"""MORAI S11 no-stop policy at the final command boundary."""
import unittest
from unittest.mock import MagicMock
import test_driving_stability as stability
from pedal_guard import PedalGuard


class S11Min30Tests(unittest.TestCase):
    def follower(self, sector=11):
        n = stability.StabilityTests().cruise_follower(sector)
        n.simulator_s11_no_stop = True
        n.publish_tracking_state = MagicMock()
        n.publish_local_path = MagicMock()
        n.target_speed_pub = MagicMock()
        return n

    def test_all_lower_targets_are_clamped_in_both_control_modes(self):
        for mode in ('pid', 'velocity'):
            for target in (0., 5., 29., 30., 50.):
                n = self.follower()
                n.longitudinal_control_mode = mode
                n.publish_control(.1, target)
                self.assertEqual(n.requested_target_speed_kmh, max(30., target))
                cmd = n.ctrl_pub.publish.call_args.args[0]
                self.assertEqual(cmd.steering, .1)
                if mode == 'velocity':
                    self.assertEqual(cmd.velocity, max(30., target))

    def test_stop_requests_keep_moving_and_track_available_route(self):
        for reason in ('planner_blocked', 'obstacle_perception_stale',
                       'gps_or_heading_stale', 'guard_speed_zero_or_stale',
                       'no_forward_waypoint', 'goal_reached'):
            n = self.follower()
            n.stop_reason = reason
            n.path_ready = True
            n.find_lookahead_point = MagicMock(return_value=(0, 5., 1., 5.1))
            n.calculate_steering = MagicMock(return_value=.15)
            n.publish_full_stop()
            self.assertEqual(n.requested_target_speed_kmh, 30.)
            self.assertEqual(n.ctrl_pub.publish.call_args.args[0].steering, .15)
            self.assertLess(n.ctrl_pub.publish.call_args.args[0].brake, 1.)
            n.publish_tracking_state.assert_called_once_with()

    def test_no_feedback_preserves_competition_pedal_format(self):
        for speed in (None, 60.):
            n = self.follower()
            n.current_speed_kmh = speed
            n.stop_reason = 'speed_feedback_stale'
            n.publish_full_stop()
            cmd = n.ctrl_pub.publish.call_args.args[0]
            self.assertEqual(cmd.longlCmdType, 1)
            self.assertEqual(n.requested_target_speed_kmh, 30.)
            if speed is None:
                self.assertEqual(cmd.brake, 0.)
                self.assertGreater(cmd.accel, 0.)

    def test_brake_releases_near_floor_even_with_windup_and_previous_hold(self):
        for speed in (0., 20., 29., 30., 31.):
            n = self.follower()
            n.current_speed_kmh = speed
            n.speed_pid.update = MagicMock(return_value=-1.)
            n.pedal_guard = PedalGuard()
            n.pedal_guard.stop(0.)
            n.last_decision.state = 'BLOCKED'
            n.local_planner.lane_end_imminent = True
            n.publish_control(0., 0.)
            cmd = n.ctrl_pub.publish.call_args.args[0]
            self.assertEqual(cmd.brake, 0.)
            self.assertFalse(n.pedal_guard.holding)
            if speed < 30.:
                self.assertGreater(cmd.accel, 0.)

    def test_caps_and_runtime_floor_cannot_lower_30(self):
        n = self.follower()
        n.last_decision.speed_cap_kmh = 0.
        n.local_planner.sector_min_speed_kmh = 0.
        n.local_planner.lane_end_imminent = True
        self.assertEqual(n.sector_speed_floor(0.), 30.)
        self.assertEqual(n.sector_speed_floor(55.), 55.)

    def test_other_sectors_and_disabled_option_retain_full_stop(self):
        for sector, enabled in ((9, True), (10, True), (12, True), (11, False)):
            n = self.follower(sector)
            n.simulator_s11_no_stop = enabled
            n.publish_full_stop()
            self.assertEqual(n.ctrl_pub.publish.call_args.args[0].brake, 1.)
            self.assertEqual(n.requested_target_speed_kmh, 0.)


if __name__ == '__main__':
    unittest.main()
