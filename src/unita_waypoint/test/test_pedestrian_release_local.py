import unittest
import test_predictive_safety_local as fixtures
from leo_avoidance.route_avoidance import RouteObstacle

class ReleaseTests(unittest.TestCase):
    def test_separate_people_do_not_share_ids_or_entry_side(self):
        p=fixtures.PredictiveSafetyTests().route()
        for i in range(4):
            p.update((0.,0.),4.,[RouteObstacle(20.,3.-i*.3,.3,track_id=1),
                                RouteObstacle(27.,-3.+i*.3,.3,track_id=2)])
        self.assertEqual(len(p.pedestrian_holds),2)
        self.assertEqual([h['ids'] for h in p.pedestrian_holds],[{1},{2}])
        self.assertEqual([h['entry_side'] for h in p.pedestrian_holds],[1.,-1.])
    def test_new_id_reacquired_at_predicted_position_and_clear_in_three_scans(self):
        p=fixtures.PredictiveSafetyTests().route()
        for t,d in [(1.,-2.),(1.1,-1.7),(1.2,-1.4)]:
            p.update((0.,0.),4.,[RouteObstacle(20.,d,.3,vy=2.,dynamic=True,track_id=1)],stamp=t)
        self.assertIsNotNone(p.pedestrian_hold)
        p.update((0.,0.),0.,[],stamp=2.)
        for i in range(3):
            p.update((0.,0.),0.,[RouteObstacle(20.,2.6+.2*i,.3,vy=2.,dynamic=True,track_id=9)],stamp=3.2+.1*i)
            if i<2:self.assertIsNotNone(p.pedestrian_hold)
        self.assertIsNone(p.pedestrian_hold)
    def test_missing_frames_do_not_count_as_clearance(self):
        p=fixtures.PredictiveSafetyTests().route()
        for t,d in [(1.,-2.),(1.1,-1.7),(1.2,-1.4)]:
            p.update((0.,0.),4.,[RouteObstacle(20.,d,.3,vy=2.,dynamic=True,track_id=1)],stamp=t)
        p.update((0.,0.),0.,[RouteObstacle(20.,2.6,.3,vy=2.,dynamic=True,track_id=1)],stamp=3.2)
        for i in range(10):p.update((0.,0.),0.,[],stamp=3.3+i*.1)
        self.assertIsNotNone(p.pedestrian_hold)
        self.assertEqual(p.pedestrian_hold['clear_scans'],0)
