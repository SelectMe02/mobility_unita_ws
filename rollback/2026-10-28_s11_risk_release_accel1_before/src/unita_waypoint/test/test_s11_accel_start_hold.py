import unittest
from unittest.mock import MagicMock, patch
import test_s11_lane_change_accel as pedals
import test_s11_entry_slowdown as entry
import test_highway_planner as f


class AccelStartHoldTests(unittest.TestCase):
    def test_small_following_speed_increment_defers_lane_change(self):
        p=f.planner(s11_forward_only=True)
        p._idm_cap=MagicMock(return_value=61.8)
        d=p.plan(60,(60.,0.),60./3.6,[],desired_speed_mps=100./3.6)
        self.assertIsNone(p.transition)
        self.assertEqual(d.reason,'wait_acceleration_room')
        p._idm_cap.return_value=70.
        d=p.plan(60,(60.,0.),60./3.6,[],desired_speed_mps=100./3.6)
        self.assertEqual(p.transition['target'],2)
        self.assertGreaterEqual(d.speed_cap_kmh,65.)

    def test_map_limit_does_not_require_exceeding_limit_to_change(self):
        p=f.planner(s11_forward_only=True)
        p.plan(60,(60.,0.),60./3.6,[],desired_speed_mps=60./3.6)
        self.assertIsNotNone(p.transition)

    def test_entry_latch_holds_r2_without_returning_to_r1(self):
        p=entry.planner();p.s11_entry_guard.active=True
        d=p.plan(60,entry.point(0.,3.6),30./3.6,[],desired_speed_mps=100./3.6)
        self.assertIsNone(p.transition)
        self.assertEqual(p.current_lane,2)
        self.assertEqual(d.speed_cap_kmh,30.)
        self.assertTrue(all(abs(entry.f.core.s11_entry_coordinates(x)[1]-3.6)<1e-6 for x in d.path))

    def test_boost_continues_for_exactly_one_second_after_completion(self):
        n=pedals.LaneChangeAccelerationTests().follower()
        n.sector11_post_change_accel_s=1.
        n.local_planner.highway_change_end=None
        self.assertTrue(n.s11_acceleration_phase(10.))
        n.local_planner.transition=None;n.local_planner.highway_change_end=100.
        self.assertTrue(n.s11_acceleration_phase(11.))
        self.assertTrue(n.s11_acceleration_phase(11.999))
        self.assertFalse(n.s11_acceleration_phase(12.))
        self.assertFalse(n.s11_acceleration_phase(12.1))

    def test_post_completion_pedal_uses_fast_ramp_and_brake_still_wins(self):
        n=pedals.LaneChangeAccelerationTests().follower()
        n.local_planner.highway_change_end=None
        with patch('gps_waypoint_follower.time.monotonic',return_value=10.):n.publish_control(0.,80.)
        n.local_planner.transition=None;n.local_planner.highway_change_end=100.
        with patch('gps_waypoint_follower.time.monotonic',return_value=10.05):n.publish_control(0.,80.)
        self.assertAlmostEqual(n.ctrl_pub.publish.call_args.args[0].accel,.2)
        n.speed_pid.update.return_value=-1.
        with patch('gps_waypoint_follower.time.monotonic',return_value=10.1):n.publish_control(0.,30.)
        self.assertEqual(n.ctrl_pub.publish.call_args.args[0].accel,0.)
        self.assertGreater(n.ctrl_pub.publish.call_args.args[0].brake,0.)

    def test_dropout_or_sector_exit_does_not_create_tail_boost(self):
        n=pedals.LaneChangeAccelerationTests().follower()
        n.local_planner.highway_change_end=None
        n.s11_acceleration_phase(10.)
        n.local_planner.transition=None
        self.assertFalse(n.s11_acceleration_phase(10.1))
        n.s11_accel_until=20.;n.route_sectors=[12]
        self.assertFalse(n.s11_acceleration_phase(10.2))
        self.assertEqual(n.s11_accel_until,0.)


if __name__=='__main__':unittest.main()
