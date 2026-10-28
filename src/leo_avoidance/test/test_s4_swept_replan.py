"""S4 swept footprint search: reproduce 22:34 route-frame observations."""
import math
import unittest
from leo_avoidance.route_avoidance import RouteAvoidance, RouteObstacle


class S4SweptTests(unittest.TestCase):
    def planner(self):
        p = RouteAvoidance([(float(x), 0.) for x in range(401)])
        p.set_static_pass_zone(297.906, 315.906, 339.906, 346.742, -3.2)
        return p

    def scene(self):
        return [RouteObstacle(316.5, -3.43, .21, half_length=.18),
                RouteObstacle(329.3, 1.32, .92, half_length=.55)]

    def test_logged_small_object_blocks_strip_but_not_selected_path(self):
        p = self.planner()
        obs = self.scene()
        items = p._to_route(obs, 305.3)
        self.assertFalse(p._lane_clear(items, -3.2, 305.3, 346.7, 5.6)[0])
        d = p.update((305.3, 0.), 5.6, obs)
        self.assertEqual(d.state, 'PASSING')
        self.assertTrue(p._s4_swept_check(p.active, items, 305.3, 5.6)[0])
        self.assertLess(d.speed_cap_mps, 5.6)

    def test_stopped_at_logged_position_finds_partial_safe_offset(self):
        p = self.planner()
        d = p.update((319.99, 0.), 0., self.scene())
        self.assertEqual(d.state, 'PASSING')
        self.assertLess(abs(p.active.offset_m), 3.2)
        self.assertGreater(d.speed_cap_mps, 0.)
        self.assertAlmostEqual(p.active.offset_at(319.99), 0.)
        self.assertLessEqual(p.active.end_s, 346.742)

    def test_selected_path_body_overlap_blocks_even_if_centres_clear(self):
        p = self.planner()
        p.update((305.3, 0.), 5., self.scene())
        profile = p.active
        obj = RouteObstacle(330., profile.offset_m+.8, .4, half_length=1.)
        items = p._to_route([obj], 305.3)
        self.assertFalse(p._s4_swept_check(profile, items, 305.3, 5.)[0])

    def test_real_blockage_preserves_entry_room_earlier_than_bumper_stop(self):
        p = self.planner()
        obs = self.scene()+[RouteObstacle(325., -2., 2., half_length=3.)]
        d = p.update((310., 0.), 5., obs)
        self.assertIsNone(p.active)
        self.assertEqual(d.state, 'WAIT_PASS')
        self.assertIn('preserving entry room', d.reason)
        self.assertLess(d.speed_cap_mps, 5.)

    def test_selected_pass_survives_missed_observations(self):
        p = self.planner()
        p.update((305.3, 0.), 5., self.scene())
        active = p.active
        for s in (310., 315., 320., 325., 330.):
            p.update((s, active.offset_at(s)), 3., [])
            self.assertIs(p.active, active)
        self.assertLessEqual(active.end_s, 346.742)

    def test_new_mover_intersecting_body_still_stops(self):
        p = self.planner()
        p.update((305.3, 0.), 5., self.scene())
        d = p.update((320., p.active.offset_at(320.)), 3.,
                     [RouteObstacle(322., p.active.offset_at(322.), .8,
                                    dynamic=True, vx=1., half_length=1.)])
        self.assertEqual(d.speed_cap_mps, 0.)

    def test_outside_s4_does_not_enable_new_search(self):
        p = self.planner()
        d = p.update((200., 0.), 5., [RouteObstacle(225., 0., .8)])
        self.assertEqual(d.state, 'WAIT_SOLID_LINE')
        self.assertIsNone(p.active)

    def test_selected_paths_also_pass_follower_safety_guard(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src/unita_waypoint/scripts'))
        from local_avoidance_core_local import LocalAvoidancePlanner, Obstacle
        for x, speed in ((305.3, 5.6), (319.99, 0.)):
            p = self.planner()
            d = p.update((x, 0.), speed, self.scene())
            guard = LocalAvoidancePlanner([(float(x), 0.) for x in range(401)],
                                         [4]*401, {})
            result = guard.check_route_pass(d.path, speed,
                [Obstacle(o.x, o.y, o.radius) for o in self.scene()])
            self.assertEqual(result.state, 'PASSING')
            self.assertGreater(result.speed_cap_kmh, 0.)
