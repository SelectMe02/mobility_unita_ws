"""Stronger S11 pedals preserve final speed caps and braking priority."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET
import test_s11_min30 as fixture


class LaneChangeAccelerationTests(unittest.TestCase):
    def follower(self, state='LANE_CHANGING', sector=11):
        n = fixture.S11Min30Tests().follower(sector)
        n.local_planner.transition = None if state is None else dict(state=state)
        node = ET.parse(Path(__file__).resolve().parents[2]/'unita_launch/launch/semi.launch').find("node[@name='gps_waypoint_follower']")
        for name in ('sector11_lane_change_accel_gain', 'sector11_lane_change_accel_rate'):
            setattr(n,name,float(node.find("param[@name='%s']" % name).get('value')))
        n.speed_pid.update = MagicMock(return_value=.2)
        return n

    def test_positive_pedal_demand_is_amplified_with_faster_ramp(self):
        for state in ('LANE_CHANGING','RETURNING','OVERTAKING'):
            n = self.follower(state)
            for step,expected in enumerate((.05,.1,.15,.2)):
                with patch('gps_waypoint_follower.time.monotonic',return_value=10.+step*.05):
                    n.publish_control(0.,80.)
                cmd=n.ctrl_pub.publish.call_args.args[0]
                self.assertAlmostEqual(cmd.accel,expected)
                self.assertEqual(cmd.brake,0.)
                self.assertEqual(n.requested_target_speed_kmh,80.)

    def test_full_demand_reaches_one_in_one_second(self):
        n=self.follower();n.speed_pid.update.return_value=1.
        for step in range(20):
            with patch('gps_waypoint_follower.time.monotonic',return_value=10.+step*.05):
                n.publish_control(0.,80.)
        self.assertAlmostEqual(n.ctrl_pub.publish.call_args.args[0].accel,1.)

    def test_s11_general_ramp_is_fast_and_other_sectors_keep_old_ramp(self):
        for state,sector in ((None,11),('CANCELLING',11),('LANE_CHANGING',9),('LANE_CHANGING',12)):
            n=self.follower(state,sector)
            n.publish_control(0.,80.)
            self.assertAlmostEqual(n.ctrl_pub.publish.call_args.args[0].accel,.05 if sector==11 else .025)

    def test_braking_cancels_throttle_without_delay(self):
        n=self.follower()
        n.publish_control(0.,80.)
        n.speed_pid.update.return_value=-1.
        n.publish_control(0.,30.)
        cmd=n.ctrl_pub.publish.call_args.args[0]
        self.assertEqual(cmd.accel,0.)
        self.assertGreater(cmd.brake,0.)
        self.assertEqual(n.requested_target_speed_kmh,30.)

    def test_coast_and_at_target_do_not_get_extra_throttle(self):
        for demand,target in ((0.,80.),(.2,60.),(.2,55.)):
            n=self.follower();n.speed_pid.update.return_value=demand
            n.publish_control(0.,target)
            self.assertAlmostEqual(n.ctrl_pub.publish.call_args.args[0].accel,0. if demand==0. else .05)
            self.assertEqual(n.requested_target_speed_kmh,target)


if __name__=='__main__':unittest.main()
