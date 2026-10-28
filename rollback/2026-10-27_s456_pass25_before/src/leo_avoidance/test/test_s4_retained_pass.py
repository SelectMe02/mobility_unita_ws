"""S4-only retained static footprint and bounded smooth pass regressions."""
import importlib.util
from pathlib import Path
import unittest
from leo_avoidance.route_avoidance import RouteAvoidance, RouteObstacle


class S4PassTests(unittest.TestCase):
    def planner(self):
        p = RouteAvoidance([(x, 0.) for x in range(401)])
        p.set_static_pass_zone(98., 116., 140., 147., -3.2)
        return p

    def test_missing_block_keeps_pass_until_rear_clears(self):
        p = self.planner()
        d = p.update((100., 0.), 8., [RouteObstacle(129., 0., .8, half_length=.3)])
        self.assertEqual(d.state, 'PASSING')
        active = p.active
        self.assertTrue(active.quintic)
        for x in range(101, 147):
            d = p.update((float(x), active.offset_at(x)), 3., [])
            self.assertEqual(p.active, active)
            if x <= 132:
                self.assertTrue(p.static_memory)
            if active.full_s <= x <= active.hold_end_s:
                self.assertAlmostEqual(d.offset_now_m, -3.2)
        p.update((148., 0.), 3., [])
        self.assertIsNone(p.active)
        self.assertFalse(p.static_memory)
        self.assertFalse(p.static_pass_managed)

    def test_wait_does_not_clear_on_missed_detection(self):
        p = self.planner()
        # Detection too late to create a bounded-curvature pull-out.
        block = RouteObstacle(129., 0., .8, half_length=.3)
        p.update((120., 0.), 7., [block])
        self.assertIsNone(p.active)
        d = p.update((121., 0.), 0., [])
        self.assertEqual(d.state, 'WAIT_PASS')
        self.assertEqual(d.speed_cap_mps, 0.)
        self.assertTrue(p.static_memory)

    def test_blocked_adjacent_lane_never_becomes_clear_by_dropout(self):
        p = self.planner()
        p.update((100., 0.), 5., [RouteObstacle(129., 0., .8),
                                RouteObstacle(130., -3.2, .8)])
        self.assertIsNone(p.active)
        d = p.update((101., 0.), 5., [])
        self.assertIsNone(p.active)
        self.assertEqual(d.state, 'WAIT_PASS')

    def test_new_obstacle_does_not_cancel_existing_pass(self):
        p = self.planner()
        p.update((100., 0.), 5., [RouteObstacle(129., 0., .8, half_length=.3)])
        active = p.active
        d = p.update((110., active.offset_at(110.)), 5., [RouteObstacle(130., -3.2, .8)])
        self.assertIs(p.active, active)
        self.assertEqual(d.state, 'PASSING')
        self.assertLess(d.speed_cap_mps, 20./3.6)

    def test_scope_does_not_allow_pass_elsewhere(self):
        p = self.planner()
        d = p.update((10., 0.), 5., [RouteObstacle(35., 0., .8)])
        self.assertEqual(d.state, 'WAIT_SOLID_LINE')
        self.assertFalse(p.static_memory)
        self.assertFalse(p.static_pass_managed)

    def test_reset_clears_retained_state(self):
        p = self.planner()
        p.update((100., 0.), 5., [RouteObstacle(129., 0., .8)])
        p.reset()
        self.assertFalse(p.static_memory)
        self.assertIsNone(p.active)

    def test_real_s4_map_and_collision_location(self):
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('s4_node', root/'src/leo_avoidance/scripts/route_avoidance_node.py')
        node = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(node)
        file = root/'src/unita_waypoint/config/map_tunner_project.json'
        p = RouteAvoidance(node.load_route_points(file)[:, :2])
        node.configure_s4_static_pass(p, file)
        start, full_start, full_end, end, offset = p.static_pass_zone
        self.assertAlmostEqual(offset, -3.2, places=2)
        xy, _ = p.route.pose(start+1., 0.)
        obstacle_xy, _ = p.route.pose(329.3, 1.3)
        d = p.update(xy, 8., [RouteObstacle(*obstacle_xy, .8, half_length=.25)])
        self.assertEqual(d.state, 'PASSING')
        self.assertGreaterEqual(p.active.full_s, full_start)
        self.assertLessEqual(p.active.hold_end_s, full_end)
        self.assertLessEqual(p.active.end_s, end)
        self.assertLess(d.speed_cap_mps, 20./3.6)

class S4OwnershipTests(unittest.TestCase):
    def helper(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src/unita_waypoint/test'))
        import test_route_pass_ownership_local as cases
        case = cases.RoutePassOwnershipTests()
        obj = case.passing()
        obj.route_avoidance['static_pass_managed'] = True
        return case, obj

    def test_s4_owns_even_with_old_saved_lane_transition(self):
        case, obj = self.helper()
        obj.local_planner.transition = dict(source=1, target=2, state='OVERTAKING',
            start=38., length=14., lateral=3.5, shape='smooth')
        obj.local_planner.active_lane = 2
        case.run_loop(obj)
        self.assertTrue(obj.debug_trace['route_pass_owns'])
        self.assertEqual(obj.debug_trace['local_path_source'], 'route_avoidance')
        self.assertIsNone(obj.local_planner.transition)

    def test_s4_wait_cap_survives_empty_local_observations(self):
        case, obj = self.helper()
        obj.route_avoidance.update(state='WAIT_PASS', pass_active=False, speed_cap_kmh=0.)
        obj.obstacles = []
        case.run_loop(obj)
        self.assertTrue(obj.debug_trace['route_pass_owns'])
        self.assertEqual(obj.stop_reason, 'planned_speed_zero')
        self.assertFalse(obj.commands)
