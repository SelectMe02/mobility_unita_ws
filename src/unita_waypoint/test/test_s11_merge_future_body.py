import unittest
import math
import test_s11_entry_slowdown as e
import test_highway_planner as f

class FutureMergeBodyTests(unittest.TestCase):
    def run_case(self, side, side_speed=0., confirmed=True):
        p=e.planner(merge_control=True)
        x,y=e.point(0.,side)
        tx,ty=f.core.TANGENT
        o=f.core.Obstacle(x,y,.7,length=.6,width=1.2,yaw=math.atan2(ty,tx),
                          vx=15.*tx-side_speed*ty,vy=15.*ty+side_speed*tx,
                          confirmed=confirmed,velocity_reliable=True,track_id=7)
        d=p.plan(40,e.point(-20.),20.,[o],desired_speed_mps=25.,observation_stamp=1.)
        return d

    def test_converging_vehicle_triggers_before_current_boxes_overlap(self):
        self.assertLess(self.run_case(3.2,-.5).speed_cap_kmh,72.)

    def test_parallel_or_departing_vehicle_does_not_trigger(self):
        for side_speed in (0.,.5):
            self.assertGreaterEqual(self.run_case(3.2,side_speed).speed_cap_kmh,90.)

    def test_confirmed_partial_body_is_not_treated_as_tiny_car(self):
        self.assertLess(self.run_case(2.6).speed_cap_kmh,72.)
        self.assertGreaterEqual(self.run_case(2.6,confirmed=False).speed_cap_kmh,90.)

    def test_far_lane_stays_clear_even_with_small_lateral_motion(self):
        self.assertGreaterEqual(self.run_case(7.2,-.5).speed_cap_kmh,90.)

if __name__=='__main__':unittest.main()
