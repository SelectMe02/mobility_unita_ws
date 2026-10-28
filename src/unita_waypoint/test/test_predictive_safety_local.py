import unittest
import test_route_pass_ownership_local as fixture
from leo_avoidance.route_avoidance import RouteAvoidance, RouteAvoidanceConfig, RouteObstacle
from local_avoidance_core_local import Obstacle
from obstacle_motion import update_motion


class PredictiveSafetyTests(unittest.TestCase):
    def test_stationary_broad_object_needs_sustained_motion(self):
        state = {}
        for i in range(10):
            state = update_motion(state, i*.1, (0., 0., 1.4, .1))
        for i in range(1, 4):
            x = .09*i
            state = update_motion(state, .9+i*.1, (x, 0., x+1.4, .1))
            self.assertFalse(state['prediction_confirmed'])
            self.assertEqual(state['vx'], 0.)
        for i in range(4, 20):
            x = .09*i
            state = update_motion(state, .9+i*.1, (x, 0., x+1.4, .1))
        self.assertTrue(state['prediction_confirmed'])
        self.assertGreater(state['vx'], .8)

    def route(self, protected=True):
        p=RouteAvoidance([(float(x),0.) for x in range(401)], RouteAvoidanceConfig(),
                        [dict(start_s=0.,end_s=400.,side='left',offset_m=3.5)])
        if protected:p.set_pedestrian_hold_ranges([(0.,200.)])
        return p

    def test_unconfirmed_approach_brakes_and_loss_does_not_release(self):
        p=self.route()
        p.update((0.,0.),6.8,[RouteObstacle(15.,3.,.25,track_id=1)])
        decision=p.update((1.,0.),6.8,[RouteObstacle(15.,2.6,.25,track_id=1)])
        self.assertIsNotNone(p.pedestrian_hold)
        self.assertLess(decision.speed_cap_mps,6.8)
        for _ in range(100):
            decision=p.update((1.,0.),0.,[])
            self.assertIsNotNone(p.pedestrian_hold)
        # Reacquisition on the far side provides affirmative clearance.
        for _ in range(20):
            p.update((1.,0.),0.,[RouteObstacle(15.,-4.,.25,track_id=1)])
        self.assertIsNotNone(p.pedestrian_hold)

    def test_stationary_outside_person_and_other_regions_do_not_latch(self):
        for protected in (True,False):
            p=self.route(protected)
            for lateral in ((3.,3.,3.) if protected else (3.,2.6,2.2)):
                p.update((0.,0.),6.,[RouteObstacle(15.,lateral,.25,track_id=1)])
            self.assertIsNone(p.pedestrian_hold)

    def test_blocked_pass_in_protected_zone_keeps_existing_path(self):
        p=self.route()
        block=RouteObstacle(35.,0.,1.,track_id=1)
        p.update((10.,0.),5.,[block])
        active=p.active
        self.assertIsNotNone(active)
        p.update((11.,0.),5.,[block,RouteObstacle(25.,3.5,.6,track_id=2)])
        self.assertIs(p.active,active)

    def test_far_collision_slows_but_near_collision_stops(self):
        p=fixture.RoutePassOwnershipTests().passing().local_planner
        path=[(float(x),0.) for x in range(100)]
        far=p.check_route_pass(path,3.,[Obstacle(18.,0.,.5)])
        self.assertEqual(far.state,'PASSING')
        self.assertEqual(far.reason,'route_pass_obstacle_ahead')
        self.assertGreater(far.speed_cap_kmh,0.)
        self.assertLessEqual(far.speed_cap_kmh,p.pass_speed_kmh)
        near=p.check_route_pass(path,6.,[Obstacle(6.,0.,.5)])
        self.assertEqual(near.state,'BLOCKED')

if __name__=='__main__':unittest.main()
