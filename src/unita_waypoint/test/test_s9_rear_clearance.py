import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_avoidance_core import LocalAvoidancePlanner, Obstacle


class RearClearanceTests(unittest.TestCase):
    def planner(self, sector=9):
        p = LocalAvoidancePlanner([(float(x), 0.) for x in range(101)], [sector]*101, {},
            roundabout_sector=sector, roundabout_conflict_point=(30., 0.),
            roundabout_approach_points=[(30., 30.), (30., 0.), (45., 0.)])
        p.roundabout_yield_seen = True
        p.roundabout_threat_id = 7
        p.roundabout_last_clear_stamp = 1.
        return p

    def car(self, **kw):
        args = dict(x=34.4, y=0., radius=2.5, length=4.6, width=1.9,
                    vx=5., track_id=7, measurement_stamp=2.)
        args.update(kw)
        return Obstacle(**args)

    def step(self, p, obstacles, speed=0.):
        return p.plan(20, (20., 0.), speed, obstacles, desired_speed_mps=5., observation_stamp=2.)

    def test_full_rear_clearance_releases_before_old_exit_buffer(self):
        d = self.step(self.planner(), [self.car()])
        self.assertEqual(d.reason, 'roundabout_gap_accepted')
        self.assertGreater(d.speed_cap_kmh, 0.)
        self.assertLess(d.speed_cap_kmh, float('inf'))

    def test_insufficient_clearance_stale_fragment_and_prediction_keep_yield(self):
        for car in (self.car(x=32.9), self.car(predicted=True),
                    self.car(measurement_stamp=1.), self.car(length=1.),
                    self.car(width=.3)):
            self.assertEqual(self.step(self.planner(), [car]).state, 'ROUNDABOUT_YIELD')

    def test_high_speed_and_other_sectors_preserve_existing_exit_margin(self):
        self.assertEqual(self.step(self.planner(), [self.car()], speed=2.).state, 'ROUNDABOUT_YIELD')
        for sector in range(1, 15):
            if sector != 9:
                self.assertEqual(self.step(self.planner(sector), [self.car()]).state, 'ROUNDABOUT_YIELD')

    def test_next_car_still_blocks(self):
        d = self.step(self.planner(), [self.car(), Obstacle(30., 10., 1., vy=-5., track_id=8)])
        self.assertEqual(d.state, 'ROUNDABOUT_YIELD')

    def test_path_obstacle_blocks_first_and_latched_departure(self):
        p = self.planner()
        obstacles = [self.car(), Obstacle(23., 0., 1.)]
        self.assertEqual(self.step(p, obstacles).speed_cap_kmh, 0.)
        self.assertTrue(p.roundabout_gap_accepted)
        self.assertEqual(self.step(p, obstacles).speed_cap_kmh, 0.)


if __name__ == '__main__':
    unittest.main()
