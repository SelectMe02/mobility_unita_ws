import unittest
from test_predictive_safety_local import PredictiveSafetyTests
from leo_avoidance.route_avoidance import RouteObstacle

class CandidateTests(unittest.TestCase):
    def planner(self):
        return PredictiveSafetyTests().route()
    def obs(self,d):
        return [RouteObstacle(20.,d,.25,track_id=1)]
    def test_one_jitter_slows_then_releases_when_stationary_outside(self):
        p=self.planner()
        p.update((0.,0.),6.,self.obs(-2.5))
        result=p.update((0.,0.),6.,self.obs(-2.22))
        self.assertTrue(p.pedestrian_hold['tentative'])
        self.assertGreater(result.speed_cap_mps,0.)
        self.assertLessEqual(result.speed_cap_mps,10./3.6)
        for _ in range(6):p.update((0.,0.),0.,self.obs(-2.5))
        self.assertIsNone(p.pedestrian_hold)
    def test_approach_promotes_and_missing_does_not_release(self):
        p=self.planner()
        for d in (3.,2.7,2.4,2.1,1.):p.update((0.,0.),6.,self.obs(d))
        self.assertTrue(p.pedestrian_hold['confirmed_crossing'])
        for _ in range(100):p.update((0.,0.),0.,[])
        self.assertIsNotNone(p.pedestrian_hold)
        for _ in range(20):p.update((0.,0.),0.,self.obs(-4.))
        self.assertIsNone(p.pedestrian_hold)
