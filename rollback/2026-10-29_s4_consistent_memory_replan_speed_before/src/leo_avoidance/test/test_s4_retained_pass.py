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
        self.assertLessEqual(d.speed_cap_mps, 20./3.6)

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
        self.assertLessEqual(d.speed_cap_mps, 20./3.6)

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


class StaleMemoryTests(unittest.TestCase):
    def planner(self):
        p = RouteAvoidance([(x, 0.) for x in range(401)])
        p.set_static_pass_zone(98., 116., 140., 147., -3.2)
        return p

    def test_moving_source_removes_old_footprint_but_keeps_live_obstacle(self):
        p = self.planner()
        p.update((120., 0.), 0., [RouteObstacle(129., 0., .8, track_id=86)])
        mover = RouteObstacle(132., 3., .8, vx=2., vy=1., dynamic=True, track_id=86)
        p.update((120., 0.), 0., [mover])
        self.assertTrue(p.static_memory)
        result = p.update((120., 0.), 0., [mover])
        self.assertFalse(p.static_memory)
        self.assertTrue(any(o['id'] == 86 and o['moving'] for o in result.obstacles))

    def test_confirmed_free_space_releases_wait_without_track_identity(self):
        p = self.planner()
        p.update((120., 0.), 0., [RouteObstacle(129., 0., .8, track_id=2)])
        for _ in range(8):
            result = p.update((120., 0.), 0., [], static_clearance_observer=lambda h: True)
        self.assertFalse(p.static_memory)
        self.assertEqual(result.state, 'FOLLOW')
        self.assertGreater(result.speed_cap_mps, 0.)

    def test_present_static_object_vetoes_clear_and_duplicate_timestamps(self):
        p = self.planner()
        block = RouteObstacle(129., 0., .8, track_id=2)
        for _ in range(8):
            p.update((120., 0.), 0., [block], static_clearance_observer=lambda h: True)
        self.assertTrue(p.static_memory)
        for _ in range(8):
            p.update((120., 0.), 0., [], stamp=2., static_clearance_observer=lambda h: True)
        self.assertTrue(p.static_memory)

    def test_mover_does_not_erase_another_observed_static_object(self):
        p = self.planner()
        p.update((120., 0.), 0., [RouteObstacle(129., 0., .8, track_id=86)])
        for _ in range(3):
            p.update((120., 0.), 0., [
                RouteObstacle(132., 3., .8, vx=2., vy=1., dynamic=True, track_id=86),
                RouteObstacle(129., 0., .8, track_id=90)])
        self.assertTrue(p.static_memory)

    def test_brief_motion_noise_and_occlusion_preserve_memory(self):
        p = self.planner()
        p.update((120., 0.), 0., [RouteObstacle(129., 0., .8, track_id=86)])
        p.update((120., 0.), 0., [RouteObstacle(130., 0., .8, vx=2., dynamic=True, track_id=86)])
        for _ in range(20):
            p.update((120., 0.), 0., [], static_clearance_observer=lambda h: False)
        self.assertTrue(p.static_memory)


class S4MotionMemoryTests(unittest.TestCase):
    def planner(self):
        p = RouteAvoidance([(x, 0.) for x in range(401)])
        p.set_static_pass_zone(98., 116., 140., 147., -3.2)
        return p

    def test_pre_s4_lead_vehicle_cannot_leave_static_trail_on_dropout(self):
        p = self.planner()
        for i in range(3):
            p.update((97., 0.), 0., [RouteObstacle(125.+i*.2, 0., .8, vx=2., dynamic=True, track_id=44)])
        d = p.update((100., 0.), 0., [RouteObstacle(125.6, 0., .8, track_id=44)])
        self.assertFalse(p.static_memory)
        self.assertTrue(any(o['id'] == 44 for o in d.obstacles))

    def test_measured_translation_does_not_grow_permanent_static_wall(self):
        p = self.planner()
        for i in range(8):
            p.update((120., 0.), 0., [RouteObstacle(128.+i*.2, i*.15, .8,
                     half_length=.5, track_id=84)])
        self.assertFalse(p.static_memory)

    def test_different_static_sources_never_merge_and_dropout_keeps_both(self):
        p = self.planner()
        p.update((120., 0.), 0., [RouteObstacle(129., 0., .8, track_id=44),
                                RouteObstacle(130., .3, .8, track_id=84)])
        self.assertEqual(len(p.static_memory), 2)
        p.update((120., 0.), 0., [])
        self.assertEqual(len(p.static_memory), 2)
        self.assertTrue(all(len(h['source_ids']) == 1 for h in p.static_memory))

    def test_changing_visible_face_is_not_translation(self):
        p = self.planner()
        for i in range(8):
            # One edge stays fixed while the visible box grows.
            p.update((120., 0.), 0., [RouteObstacle(129.+i*.1, 0., .8,
                     half_length=.5+i*.1, track_id=4)])
        self.assertTrue(p.static_memory)
        self.assertNotIn(4, p.s4_mobile_until)

    def test_stationary_vehicle_regains_static_protection_and_reset_clears_history(self):
        p = self.planner()
        for _ in range(3):
            p.update((120., 0.), 0., [RouteObstacle(129., 0., .8, vx=2., dynamic=True, track_id=4)])
        for _ in range(32):
            p.update((120., 0.), 0., [RouteObstacle(129., 0., .8, track_id=4)])
        self.assertTrue(p.static_memory)
        p.reset()
        self.assertFalse(p.s4_motion_history)
        self.assertFalse(p.s4_mobile_until)


class S4AdaptivePathTests(unittest.TestCase):
    def test_recorded_staggered_obstacles_replan_inside_saved_corridor(self):
        from leo_avoidance.route_avoidance import _Pass
        p = RouteAvoidance([(0., 0.), (500., 0.)])
        p.set_static_pass_zone(297.906, 315.906, 339.906, 346.742, -3.2)
        p.active = _Pass(310.4306, 324.4306, 332.3648, 346.3648, -2.08, 328.2756, True)
        start = 319.2156
        old_offset = p.active.offset_at(start)
        live = [RouteObstacle(316.6575, -3.7998, .2015, half_length=.194, track_id=1),
                RouteObstacle(328.3894, .09885, .6964, half_length=.7933, track_id=9),
                RouteObstacle(333.9269, -5.0044, .7767, half_length=.6089, track_id=8),
                RouteObstacle(342.0356, -5.2218, .5212, half_length=.1825, track_id=10)]
        p.static_memory = [dict(s=328.5764, d=.4487, r=1.2062, l=.9984, id=9,
                               moving=False, v_s=0., v_d=0., remembered=True, source_ids=[9]),
                           dict(s=334.3255, d=-4.8135, r=1.2501, l=1.0274, id=8,
                               moving=False, v_s=0., v_d=0., remembered=True, source_ids=[8])]
        for i in range(14):
            result = p.update((start, old_offset), 0., live, stamp=10.+.1*i)
        self.assertTrue(p.active.knots)
        self.assertGreater(result.speed_cap_mps, 0.)
        self.assertAlmostEqual(p.active.offset_at(start), old_offset, places=4)
        self.assertTrue(p._s4_swept_check(p.active, result.obstacles, start, 0.)[0])
        self.assertAlmostEqual(p.active.offset_at(p.static_pass_zone[3]), 0.)
        self.assertTrue(p.static_memory)  # evidence is retained for dropout
        dropout = p.update((start, old_offset), 0., [], stamp=11.5)
        self.assertTrue(any(o.get('remembered') for o in dropout.obstacles))

    def test_search_cannot_go_outside_corridor_or_through_complete_blockage(self):
        from leo_avoidance.s4_path_search import search_offsets
        p = RouteAvoidance([(0., 0.), (100., 0.)])
        obstacle = dict(s=15., d=-1.6, l=2., r=4., v_s=0., v_d=0.)
        self.assertIsNone(search_offsets(0., 35., 0., 0., -3.2, [obstacle], p.config))
