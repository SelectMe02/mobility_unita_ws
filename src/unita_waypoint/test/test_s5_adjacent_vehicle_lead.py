"""S5 lane-keeping must not tail stable parallel traffic with noisy boxes."""
import unittest
from dataclasses import replace
from local_avoidance_core import LocalAvoidancePlanner, Obstacle

class S5AdjacentVehicleTests(unittest.TestCase):
    def planner(self, sector=5):
        route = [(float(i), 0.) for i in range(100)]
        return LocalAvoidancePlanner(route, [sector]*100,
            {2: [(float(i), -3.5) for i in range(100)]})

    def car(self, **kw):
        return replace(Obstacle(6., -2.6, 3., length=4., width=3.6,
            kind='dynamic', confirmed=True, velocity_reliable=False,
            raceline=2, lane_sector=5, lane_stable=True,
            lane_boundary_crossing=True, track_id=48), **kw)

    def test_no_stop_or_lane_change_for_fluctuating_adjacent_vehicle(self):
        p = self.planner()
        for width in (1.7, 3.5, 1.7, 3.6, 4.5):
            d = p.plan(0, (0., 0.), 4., [self.car(width=width)], desired_speed_mps=7.)
            self.assertEqual(d.state, 'RACELINE')
            self.assertGreater(d.speed_cap_kmh, 0.)
            self.assertIsNone(p.transition)

    def test_current_lane_or_uncertain_vehicle_keeps_lead_protection(self):
        for car in (self.car(raceline=1), self.car(lane_stable=False),
                    self.car(lane_sector=4), self.car(confirmed=False),
                    self.car(velocity_reliable=True, vy=1.),
                    self.car(y=-.5), self.car(length=.6, width=.6, y=0.)):
            p = self.planner()
            d = p.plan(0, (0., 0.), 4., [car], desired_speed_mps=7.)
            self.assertNotEqual(d.state, 'RACELINE')

    def test_pedestrian_and_static_object_not_exempt(self):
        for car in (self.car(kind='static'), self.car(length=.6)):
            p = self.planner()
            d = p.plan(0, (0., 0.), 4., [car], desired_speed_mps=7.)
            self.assertNotEqual(d.state, 'RACELINE')

    def test_other_sectors_unchanged(self):
        for sector in (4, 6, 11):
            p = self.planner(sector)
            d = p.plan(0, (0., 0.), 4., [self.car()], desired_speed_mps=7.)
            self.assertNotEqual(d.state, 'RACELINE')

    def test_transition_checks_still_see_adjacent_vehicle(self):
        p = self.planner()
        p.transition = dict(source=1, target=2, start=0., length=30.,
                            state='OVERTAKING', locked=False)
        d = p.plan(0, (0., 0.), 4., [self.car()], desired_speed_mps=7.)
        self.assertLess(d.speed_cap_kmh, 25.)

if __name__ == '__main__':
    unittest.main()
