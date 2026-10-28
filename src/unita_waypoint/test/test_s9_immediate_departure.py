import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_avoidance_core import LocalAvoidancePlanner, Obstacle


class ImmediateDepartureTests(unittest.TestCase):
    def planner(self, sector=9):
        p = LocalAvoidancePlanner([(float(x), 0.) for x in range(101)], [sector]*101, {},
            roundabout_sector=sector, roundabout_conflict_point=(30., 0.),
            roundabout_approach_points=[(30., 30.), (30., 0.), (45., 0.)])
        p.roundabout_yield_seen = True
        p.roundabout_threat_id = 7
        p.roundabout_last_clear_stamp = 1.
        return p

    def cleared(self, **kw):
        return Obstacle(36., 0., .8, vx=5., track_id=7, measurement_stamp=2., **kw)

    def step(self, p, obstacles, stamp=2.):
        return p.plan(10, (10., 0.), 0., obstacles, desired_speed_mps=5., observation_stamp=stamp)

    def test_fresh_measured_pass_releases_in_first_scan(self):
        d = self.step(self.planner(), [self.cleared()])
        self.assertEqual(d.reason, 'roundabout_gap_accepted')

    def test_dropout_predicted_stale_and_wrong_id_keep_confirmation(self):
        stale = self.cleared(); stale.measurement_stamp = 1.
        other = self.cleared(); other.track_id = 8
        for obstacles in ([], [self.cleared(predicted=True)], [stale], [other]):
            self.assertEqual(self.step(self.planner(), obstacles).state, 'ROUNDABOUT_APPROACH')

    def test_same_frame_does_not_release(self):
        o = self.cleared(); o.measurement_stamp = 1.
        self.assertEqual(self.step(self.planner(), [o], stamp=1.).state, 'ROUNDABOUT_APPROACH')

    def test_occupied_merge_or_next_car_still_yields(self):
        for obstacles in ([Obstacle(30., 1., .8, track_id=7)],
                          [self.cleared(), Obstacle(30., 10., .8, vy=-5., track_id=8)]):
            decision = self.planner().plan(20, (20., 0.), 0., obstacles,
                desired_speed_mps=5., observation_stamp=2.)
            self.assertEqual(decision.state, 'ROUNDABOUT_YIELD')

    def test_other_configured_sectors_keep_two_scans(self):
        for sector in range(1, 15):
            if sector != 9:
                self.assertEqual(self.step(self.planner(sector), [self.cleared()]).state,
                                 'ROUNDABOUT_APPROACH')

    def test_first_throttle_is_on_release_tick_only_in_s9(self):
        from gps_waypoint_follower import GPSWaypointFollower
        from pedal_guard import PedalGuard
        for sector in (8, 9, 10):
            for demand in (1., -1.):
                n = GPSWaypointFollower.__new__(GPSWaypointFollower)
                n.longitudinal_control_mode, n.control_rate = 'pid', 20.
                n.last_pid_update, n.current_speed_kmh = 1., 0.
                n.speed_pid = MagicMock(); n.speed_pid.update.return_value = demand
                n.ctrl_pub, n.sector11_cruise_brake = MagicMock(), None
                n.lane_change_reason = 'roundabout_gap_accepted'
                n.local_planner = SimpleNamespace(clear_hold_s=.4, restart_accel_rate=.5)
                n.route_sectors, n.current_waypoint_index = [sector], 0
                n.pedal_guard = PedalGuard(); n.pedal_guard.stop(1.)
                with patch('gps_waypoint_follower.time.monotonic', return_value=1.05):
                    n.publish_control(0., 30.)
                cmd = n.ctrl_pub.publish.call_args.args[0]
                if demand < 0.:
                    self.assertEqual((cmd.accel, cmd.brake), (0., 1.))
                else:
                    self.assertAlmostEqual(cmd.accel, .05 if sector == 9 else 0.)
                    self.assertEqual(cmd.brake, 0.)


if __name__ == '__main__':
    unittest.main()
