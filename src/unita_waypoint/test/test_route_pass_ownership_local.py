"""S5: a route pass must not be vetoed by an unused R1 correction."""
import json
import time
import unittest
from std_msgs.msg import String
import test_return_ownership_local as ownership

module = ownership.module


class RoutePassOwnershipTests(ownership.ReturnOwnershipTests):
    def passing(self):
        obj = self.follower('PASSING', 20., 'passing static block')
        obj.local_planner.active_lane = None
        obj.current_x, obj.current_y = 40., 1.6
        path = [(40., 1.6), (43., 2.6)] + [(float(x), 3.5) for x in range(46, 110)]
        obj.route_avoidance_callback(String(data=json.dumps(dict(
            path=path, state='PASSING', reason='passing static block',
            pass_active=True, speed_cap_kmh=20., traffic_speed_cap_kmh=None))))
        obj.obstacles = [module.Obstacle(55., 0., 1.)]
        return obj

    def test_dropout_does_not_start_unused_r1_return(self):
        obj = self.passing()
        for obstacles in (obj.obstacles, [], [], obj.obstacles):
            obj.obstacles = obstacles
            self.run_loop(obj)
            self.assertIsNone(obj.local_planner.transition)
            self.assertTrue(obj.debug_trace['route_pass_owns'])
            self.assertEqual(obj.debug_trace['local_path_source'], 'route_avoidance')
            self.assertIsNone(obj.stop_reason)

    def test_safe_route_recovers_from_obsolete_blocked_offset_return(self):
        obj = self.passing()
        obj.current_speed_kmh = 0.
        obj.local_planner.transition = dict(source=1, target=1, state='RETURNING',
            start=38., length=7., offset=1.6, lateral=1.6, shape='smooth')
        self.run_loop(obj)
        self.assertIsNone(obj.local_planner.transition)
        self.assertIsNone(obj.stop_reason)
        self.assertTrue(obj.commands)

    def test_selected_path_blocker_stops_and_is_rechecked(self):
        obj = self.passing()
        obj.obstacles = [module.Obstacle(44., 3.5, 1., kind='static')]
        self.run_loop(obj)
        self.assertEqual(obj.stop_reason, 'planner_blocked:route_pass_obstructed')
        self.assertFalse(obj.commands)
        obj.obstacles = [module.Obstacle(48., 7., 1., kind='static')]
        self.run_loop(obj)
        self.assertIsNone(obj.stop_reason)

    def test_moving_crossing_of_selected_path_is_not_ignored(self):
        obj = self.passing()
        obj.obstacles = [module.Obstacle(44., 4.7, .4, vy=-1., kind='dynamic')]
        self.run_loop(obj)
        self.assertEqual(obj.stop_reason, 'planner_blocked:route_pass_obstructed')

    def test_yield_keeps_pass_ownership_and_brake(self):
        obj = self.passing()
        obj.route_avoidance.update(state='YIELD', speed_cap_kmh=0., traffic_speed_cap_kmh=0.)
        self.run_loop(obj)
        self.assertTrue(obj.debug_trace['route_pass_owns'])
        self.assertIsNone(obj.local_planner.transition)
        self.assertEqual(obj.stop_reason, 'planned_speed_zero')
        self.assertFalse(obj.commands)

    def test_stale_obstacles_still_stop(self):
        obj = self.passing()
        obj.obstacle_received = time.monotonic() - 2.
        self.run_loop(obj)
        self.assertEqual(obj.stop_reason, 'obstacle_perception_stale')

    def test_pass_completion_restores_normal_planning(self):
        obj = self.passing()
        self.run_loop(obj)
        obj.route_avoidance.update(pass_active=False, state='FOLLOW')
        obj.current_y = 0.
        self.run_loop(obj)
        self.assertFalse(obj.debug_trace['route_pass_owns'])

    def test_invalid_ownership_flag_rejected(self):
        obj = self.passing()
        obj.route_avoidance_callback(String(data=json.dumps(dict(
            path=[(0., 0.), (1., 0.)], speed_cap_kmh=20., pass_active='false'))))
        self.assertIsNone(obj.route_avoidance)


class S4ShortTargetTests(unittest.TestCase):
    def test_short_forward_target_is_s4_opt_in(self):
        obj = module.GPSWaypointFollower.__new__(module.GPSWaypointFollower)
        obj.current_x = obj.current_y = obj.current_yaw = 0.
        obj.current_waypoint_index = 0
        obj.lookahead_distance = 5.
        path = [(0.,0.),(1.,.1),(3.,.2)]
        self.assertIsNone(obj.local_target(path))
        target = obj.local_target(path, allow_short=True)
        self.assertEqual(target[1:3], (3.,.2))
        self.assertIsNone(obj.local_target([(-1.,0.)], allow_short=True))
