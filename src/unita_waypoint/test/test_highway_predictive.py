import unittest
import test_highway_planner as f

class PredictiveTests(unittest.TestCase):
    def test_side_car_crosses_before_lane_label_changes(self):
        p=f.planner()
        path=[(float(x),0.) for x in range(60)]
        o=f.car(15.,2,5.)
        o.vy=-2.
        o.lane_stable=True
        o.lane_boundary_crossing=False
        self.assertIsNotNone(p._highway_path_conflict(path,10.,[o]))
        self.assertLess(p._highway_predictive_cap(path,10.,[o]),36.)
    def test_parallel_car_clear_of_body_does_not_brake(self):
        p=f.planner()
        path=[(float(x),0.) for x in range(60)]
        self.assertIsNone(p._highway_path_conflict(path,10.,[f.car(3.,2,10.)]))
    def test_fast_rear_car_intercepts_merge(self):
        p=f.planner()
        path=[(float(x),min(3.6,x*.18)) for x in range(60)]
        self.assertIsNotNone(p._highway_path_conflict(path,10.,[f.car(-10.,2,20.)]))
    def test_body_overlap_not_only_centres(self):
        p=f.planner()
        o=f.car(5.,1,0.);o.y=1.8
        self.assertIsNotNone(p._highway_path_conflict([(0.,0.),(30.,0.)],5.,[o]))
    def test_unknown_vehicle_blocks_change(self):
        p=f.planner()
        o=f.car(10.,2,0.);o.confirmed=False;o.velocity_reliable=False
        path=[(float(x),min(3.6,x*.3)) for x in range(40)]
        self.assertIsNotNone(p._highway_path_conflict(path,8.,[o]))
    def test_repeated_same_observation_does_not_confirm_lane_gain(self):
        p=f.planner()
        lead=f.car(45.,3,12.)
        for _ in range(6):
            d=p.plan(0,(0.,f.LANE_Y[3]),20.,[lead],desired_speed_mps=f.CRUISE,observation_stamp=10.)
            self.assertIsNone(p.transition)
        d=p.plan(0,(0.,f.LANE_Y[3]),20.,[lead],desired_speed_mps=f.CRUISE,observation_stamp=10.5)
        self.assertIsNotNone(p.transition)
    def test_mild_slowdown_keeps_lane(self):
        p=f.planner()
        d=p.plan(0,(0.,f.LANE_Y[3]),27.,[f.car(55.,3,27.)],desired_speed_mps=f.CRUISE)
        self.assertIsNone(p.transition)
        self.assertGreater(d.speed_cap_kmh,80.)
