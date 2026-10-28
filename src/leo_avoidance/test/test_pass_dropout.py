"""An established static pass retains obstacle clearance through missed scans."""
import unittest
from leo_avoidance.route_avoidance import RouteAvoidance, RouteAvoidanceConfig, RouteObstacle


class PassDropoutTests(unittest.TestCase):
    def test_missing_block_does_not_shorten_pass_or_return_before_rear_clear(self):
        planner = RouteAvoidance([(float(x), 0.) for x in range(150)],
            RouteAvoidanceConfig(), [dict(start_s=0., end_s=149., side='left', offset_m=3.5)])
        block = RouteObstacle(35., 0., 1., half_length=2.)
        decision = planner.update((10., 0.), 5., [block])
        self.assertEqual(decision.state, 'PASSING')
        hold_end, end = planner.active.hold_end_s, planner.active.end_s
        for x in (20., 30., 35., 37.):
            decision = planner.update((x, planner.active.offset_at(x)), 5., [])
            self.assertEqual(decision.state, 'PASSING')
            self.assertEqual(planner.active.hold_end_s, hold_end)
            self.assertEqual(planner.active.end_s, end)
        self.assertGreaterEqual(hold_end, 37. + planner.config.rear_extent_m)
        planner.update((end + 1., 0.), 5., [])
        self.assertIsNone(planner.active)
