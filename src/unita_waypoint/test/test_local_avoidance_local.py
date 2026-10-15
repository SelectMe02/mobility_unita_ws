import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from local_avoidance_core_local import (LocalAvoidancePlanner, Obstacle, s_curve,  # noqa: E402
                                        s_curve_blend)

RADIUS = 3. / math.tan(math.radians(40.))   # competition wheelbase and full lock


def planner(**kwargs):
    route = [(0., float(y)) for y in range(100)]
    return LocalAvoidancePlanner(route, [1] * len(route), {},
                                 min_turn_radius_m=RADIUS, **kwargs)


class LaneChangeShapeTests(unittest.TestCase):
    def test_slow_lane_change_is_two_full_lock_arcs(self):
        p = planner()
        for kmh in (0., 10., 15.):
            self.assertEqual(p._merge_shape(kmh / 3.6), 'arc')
            self.assertAlmostEqual(p._merge_length((0., 0.), (3.6, 0.), kmh / 3.6), 6.21, 2)
        # Two lanes at once: arcs capped at 60 degrees plus a straight.
        self.assertAlmostEqual(p._merge_length((0., 0.), (7.2, 0.), 0.), 8.285, 2)

    def test_arc_profile_is_continuous_and_reaches_the_lane(self):
        values = [s_curve_blend(i / 100., 3.6, RADIUS) for i in range(101)]
        self.assertEqual(values[0], 0.)
        self.assertAlmostEqual(values[-1], 1.)
        self.assertAlmostEqual(values[50], .5, 3)
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))
        self.assertLess(max(b - a for a, b in zip(values, values[1:])), .05)
        length, arc_long, _, heading = s_curve(3.6, RADIUS)
        self.assertLessEqual(heading, math.radians(60.) + 1e-9)
        # The first arc follows the turn radius exactly.
        x = arc_long / 2.
        y = s_curve_blend(x / length, 3.6, RADIUS) * 3.6
        self.assertAlmostEqual(y, RADIUS - math.sqrt(RADIUS ** 2 - x ** 2))

    def test_fast_lane_change_is_limited_by_lateral_acceleration(self):
        p = planner()
        for kmh, expected in ((20., 10.54), (30., 15.81), (60., 31.62)):
            v = kmh / 3.6
            self.assertEqual(p._merge_shape(v), 'smooth')
            length = p._merge_length((0., 0.), (3.6, 0.), v)
            self.assertAlmostEqual(length, expected, 2)
            self.assertLessEqual(6. * 3.6 * v * v / length ** 2, 6. + 1e-9)
        # Never tighter than the steering allows for a smoothstep.
        self.assertAlmostEqual(p._merge_length((0., 0.), (3.6, 0.), 15.1 / 3.6),
                               math.sqrt(6. * 3.6 * RADIUS))

    def test_shape_limits_are_configurable_and_validated(self):
        p = planner(arc_lane_change_max_kmh=10., lane_change_lateral_accel_mps2=4.)
        self.assertEqual(p._merge_shape(12. / 3.6), 'smooth')
        with self.assertRaises(ValueError):
            planner(lane_change_lateral_accel_mps2=0.)
        with self.assertRaises(ValueError):
            planner(arc_lane_change_max_kmh=-1.)


def r2_planner():
    """Straight +Y route on x=0 with R2 3.6 m to its right for y 20..60 only."""
    route = [(0., float(y)) for y in range(200)]
    r2 = {i: (3.6, float(i)) for i in range(20, 61)}
    return LocalAvoidancePlanner(route, [1] * len(route), {2: list(r2.values())},
                                 lane_indices={2: r2, 3: {}}, min_turn_radius_m=RADIUS)


class ReturnToR1Tests(unittest.TestCase):
    def test_returns_to_r1_after_r2_has_ended(self):
        p = r2_planner()
        p.departed_preferred, p.previous_preferred_lane = True, 1
        d = p.plan(80, (3.6, 80.), 0., [], ego_yaw=math.pi / 2)
        self.assertEqual(d.state, 'RETURNING')
        self.assertEqual(p.transition['shape'], 'arc')
        self.assertAlmostEqual(p.transition['length'], 6.21, 2)
        end = [pt for pt in d.path if pt[1] >= 80. + 6.3][0]
        self.assertAlmostEqual(end[0], 0., 1)

    def test_passed_static_obstacle_does_not_hold_the_return(self):
        p = r2_planner()
        p.departed_preferred, p.previous_preferred_lane, p.current_lane = True, 1, 2
        passed = Obstacle(0., 46., .5, kind='static')
        d = p.plan(50, (3.6, 50.), 3., [passed], ego_yaw=math.pi / 2)
        self.assertEqual(d.state, 'RETURNING')

    def test_vehicle_closing_from_behind_still_holds_the_return(self):
        p = r2_planner()
        p.departed_preferred, p.previous_preferred_lane, p.current_lane = True, 1, 2
        car = Obstacle(0., 44., .9, vx=0., vy=8., kind='dynamic')
        d = p.plan(50, (3.6, 50.), 3., [car], ego_yaw=math.pi / 2)
        self.assertNotEqual(d.state, 'RETURNING')

    def test_blocked_return_keeps_the_offset_and_moves_on(self):
        p = r2_planner()
        p.departed_preferred, p.previous_preferred_lane = True, 1
        parked = Obstacle(0., 84., .9, kind='static')       # where the return would land
        d = p.plan(80, (3.6, 80.), 0., [parked], ego_yaw=math.pi / 2)
        self.assertEqual((d.state, d.reason), ('LANE_WAIT', 'return_blocked_holding_offset'))
        self.assertGreater(d.speed_cap_kmh, 0.)
        self.assertTrue(all(abs(x - 3.6) < .01 for x, _ in d.path))

    def test_blocked_return_and_blocked_offset_slows_then_waits_behind(self):
        parked_on_r1 = Obstacle(0., 84., .9, kind='static')
        for ahead_y, stops in ((88., False), (84.5, True)):
            p = r2_planner()
            p.departed_preferred, p.previous_preferred_lane = True, 1
            d = p.plan(80, (3.6, 80.), 0., [parked_on_r1,
                                           Obstacle(3.6, ahead_y, .9, kind='static')],
                       ego_yaw=math.pi / 2)
            self.assertEqual((d.state, d.reason), ('TAILING', 'return_blocked_waiting'))
            if stops:
                self.assertEqual(d.speed_cap_kmh, 0.)
            else:
                self.assertTrue(0. < d.speed_cap_kmh < 20.)
