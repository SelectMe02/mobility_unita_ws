import unittest
from local_avoidance_core_local import LocalAvoidancePlanner, Obstacle

class CrossingIntervalTests(unittest.TestCase):
    def planner(self):
        return LocalAvoidancePlanner([(float(x),0.) for x in range(100)], [6]*100,
            {2:[(float(x),-3.5) for x in range(100)]})

    def test_dynamic_crossing_and_static_dropout_do_not_start_pass(self):
        p=self.planner()
        for t,kind,vy in [(0.,'dynamic',1.),(.1,'dynamic',1.),(.2,'static',0.)]:
            d=p.plan(0,(0.,0.),0.,[Obstacle(10., .1+t, .5,vy=vy,kind=kind)],
                     crossing_yield=True,stamp=t)
            self.assertIsNone(p.transition)
            self.assertNotIn(d.state,('OVERTAKING','LANE_CHANGING'))
        d=p.plan(0,(0.,0.),0.,[],crossing_yield=True,stamp=.3)
        self.assertEqual(d.state,'RACELINE')
        self.assertGreater(d.speed_cap_kmh,0.)

    def test_static_obstacle_still_passes_inside_and_outside_interval(self):
        for enabled in (True,False):
            p=self.planner()
            p.plan(0,(0.,0.),0.,[Obstacle(18.,0.,.5,kind='static')],
                   crossing_yield=enabled,stamp=0.)
            self.assertIsNotNone(p.transition)

    def test_leaving_interval_clears_veto_memory(self):
        p=self.planner()
        p.plan(0,(0.,0.),0.,[Obstacle(10.,0.,.5,vy=1.,kind='dynamic')],crossing_yield=True,stamp=0.)
        p.plan(0,(0.,0.),0.,[],crossing_yield=False,stamp=.1)
        self.assertEqual(p._crossing_pass_veto,[])

    def test_static_dropout_at_last_position_not_only_predicted_endpoint(self):
        p=self.planner()
        p.plan(0,(0.,0.),0.,[Obstacle(18.,0.,.5,vy=2.,kind='dynamic')],
               crossing_yield=True,stamp=1.)
        d=p.plan(0,(0.,0.),0.,[Obstacle(18.,0.,.5,kind='static')],
                 crossing_yield=True,stamp=2.5)
        self.assertIsNone(p.transition)
        self.assertEqual(d.reason,'crossing_yield_no_overtake')
        p.plan(0,(0.,0.),0.,[Obstacle(18.,0.,.5,kind='static')],
               crossing_yield=True,stamp=3.1)
        self.assertIsNotNone(p.transition)

    def test_measured_lateral_motion_before_dynamic_label_vetoes_pass(self):
        p=self.planner()
        p.plan(0,(0.,0.),0.,[Obstacle(18.,0.,.5,vy=2.,kind='unknown')],
               crossing_yield=True,stamp=1.)
        p.plan(0,(0.,0.),0.,[Obstacle(18.,0.,.5,kind='static')],
               crossing_yield=True,stamp=1.2)
        self.assertIsNone(p.transition)

    def test_separate_static_fragment_during_crossing_does_not_start_pass(self):
        p=self.planner()
        p.plan(0,(0.,0.),0.,[Obstacle(15.,-1.,.4,vy=2.,kind='dynamic'),
                             Obstacle(20.,0.,.4,kind='static')],crossing_yield=True,stamp=1.)
        self.assertIsNone(p.transition)
