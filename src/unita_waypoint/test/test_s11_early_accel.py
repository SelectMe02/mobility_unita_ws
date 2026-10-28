"""S11 approval must reserve speed headroom instead of capping at entry speed."""
import unittest
from unittest.mock import MagicMock, patch
import test_highway_planner as f
import test_s11_lane_change_accel as pedals


class EarlyAccelTests(unittest.TestCase):
    def planner(self, **kwargs):
        return f.planner(s11_forward_only=True, highway_keep_lane_priority=True, **kwargs)

    def test_first_approved_command_has_ten_kmh_headroom_and_positive_pedal(self):
        for lane in (1,2):
            p=self.planner()
            d=p.plan(60,(60.,f.LANE_Y[lane]),60./3.6,[],desired_speed_mps=100./3.6)
            self.assertEqual(p.transition['target'],lane+1)
            self.assertAlmostEqual(d.speed_cap_kmh,70.)
            n=pedals.LaneChangeAccelerationTests().follower()
            n.local_planner=p
            from gps_waypoint_follower import PIDController
            n.speed_pid=PIDController(.08,.002,.01)
            with patch('gps_waypoint_follower.time.monotonic',return_value=10.):
                n.publish_control(0.,d.speed_cap_kmh)
            cmd=n.ctrl_pub.publish.call_args.args[0]
            self.assertGreater(cmd.accel,0.)
            self.assertEqual(cmd.brake,0.)

    def test_map_speed_limits_acceleration_target(self):
        p=self.planner()
        d=p.plan(60,(60.,0.),60./3.6,[],desired_speed_mps=65./3.6)
        self.assertAlmostEqual(min(65.,d.speed_cap_kmh),65.)
        self.assertLessEqual(p._length_speed(3.6,p.transition['length'])*3.6,65.+1e-6)

    def test_longer_geometry_respects_same_lateral_acceleration(self):
        new=self.planner();old=f.planner()
        for p in (old,new):p.plan(60,(60.,0.),60./3.6,[],desired_speed_mps=100./3.6)
        self.assertGreater(new.transition['length'],old.transition['length'])
        self.assertAlmostEqual(new.transition['length']/old.transition['length'],70./60.)

    def test_short_available_lane_does_not_force_ten_kmh_boost(self):
        p=self.planner(r2_end=115)
        d=p.plan(60,(60.,0.),60./3.6,[],desired_speed_mps=100./3.6)
        if p.transition is not None:
            self.assertLess(d.speed_cap_kmh,70.)
            self.assertLessEqual(p.transition['length'],47.)

    def test_new_planning_speed_is_checked_for_collision_before_approval(self):
        p=self.planner()
        speeds=[]
        def conflict(path,speed,obstacles):
            speeds.append(speed)
            return (1.,None,10.) if speed>60./3.6+.01 else None
        p._highway_path_conflict=MagicMock(side_effect=conflict)
        d=p.plan(60,(60.,0.),60./3.6,[],desired_speed_mps=100./3.6)
        self.assertTrue(any(s>60./3.6 for s in speeds))
        if p.transition is not None:self.assertLessEqual(d.speed_cap_kmh,60.)

    def test_front_and_rear_obstacles_still_block_unsafe_change(self):
        p=self.planner()
        p.plan(60,(60.,0.),60./3.6,[f.car(60.,2,20.),f.car(80.,1,10.)],desired_speed_mps=100./3.6)
        self.assertIsNone(p.transition)


if __name__=='__main__':unittest.main()
