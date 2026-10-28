"""Displayed paths must match the selected steering source and current tick."""
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import test_loop_tracking as primary_tests
import test_loop_tracking_local as leo_tests
from local_avoidance_core import Decision, LocalAvoidancePlanner, Obstacle
from local_avoidance_core_local import LocalAvoidancePlanner as LeoPlanner, Obstacle as LeoObstacle
from tracking_debug import local_plan_trace, saved_route_window


class TrackingDebugTests(unittest.TestCase):
    def test_saved_route_window_wraps_only_for_closed_route(self):
        points = [(0., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.)]
        self.assertEqual(saved_route_window(points, 3, True, 15.),
                         [(0., 10.), (0., 0.), (10., 0.)])
        self.assertEqual(saved_route_window(points, 3, False, 15.), points[3:])
        self.assertEqual(saved_route_window([], 0, True), [])

    def test_trace_copies_executed_branch_without_changing_planner(self):
        route = [(float(i), 0.) for i in range(150)]
        p = LocalAvoidancePlanner(route, [11]*len(route), {}, highway_sectors=(11,))
        obstacles = [Obstacle(30., 0., 1.)]
        result = p.plan(0, route[0], 5., obstacles, desired_speed_mps=10.)
        trace = local_plan_trace(p, result, obstacles)
        self.assertEqual(trace['planner_mode'], 'highway')
        self.assertEqual(trace['planner_reason'], result.reason)
        self.assertEqual(trace['planner_state'], result.state)
        self.assertEqual(trace['obstacle_horizon_m'], p.highway_obstacle_horizon_m)
        json.dumps(trace, allow_nan=False)
        p.transition = dict(source=1, target=2, length=25.)
        trace = local_plan_trace(p, result, obstacles)
        trace['transition']['target'] = 3
        self.assertEqual(p.transition['target'], 2)

    def test_leo_obstacles_do_not_have_track_confirmation_fields(self):
        route = [(float(i), 0.) for i in range(150)]
        planner = LeoPlanner(route, [1]*len(route), {})
        obstacles = [LeoObstacle(30., 0., 1.)]
        result = planner.plan(0, route[0], 5., obstacles)
        trace = local_plan_trace(planner, result, obstacles)
        self.assertEqual(trace['planner_mode'], 'local_avoidance')
        self.assertIsNone(trace['confirmed_obstacle_count'])
        self.assertEqual(trace['planner_obstacle_input_count'], 1)

    def make_follower(self, test_module):
        fixture = test_module.LoopTrackingTests()
        follower = fixture.follower([(float(i), 0.) for i in range(100)], loop=False)
        follower.local_path_pub = MagicMock()
        follower.tracking_state_pub = MagicMock()
        follower.drive_mode = None
        return fixture, follower

    def run_loop(self, fixture, follower, test_module):
        with patch.object(test_module.module.rospy.Time, 'now',
                          return_value=test_module.module.rospy.Time.from_sec(100.)):
            fixture.run_loop(follower)

    def path_points(self, follower):
        path = follower.local_path_pub.publish.call_args[0][0]
        return [(pose.pose.position.x, pose.pose.position.y) for pose in path.poses]

    def state(self, follower):
        return json.loads(follower.tracking_state_pub.publish.call_args[0][0].data)

    def test_both_followers_show_saved_route_when_it_supplies_steering(self):
        for test_module in (primary_tests, leo_tests):
            with self.subTest(follower=test_module.__name__):
                fixture, follower = self.make_follower(test_module)
                self.run_loop(fixture, follower, test_module)
                self.assertEqual(self.path_points(follower), follower.waypoints[:61])
                self.assertEqual(self.state(follower)['local_path_source'], 'saved_route')
                self.assertEqual(self.state(follower)['pursuit_target'], {'x': 5., 'y': 0.})
                self.assertEqual(follower.commands[-1], (0., 20.))

    def test_route_avoidance_path_is_not_replaced_by_saved_route(self):
        fixture, follower = self.make_follower(leo_tests)
        follower.route_avoidance_topic = '/leo/route_avoidance'
        path = [(float(i), 3.5 if i >= 4 else 0.) for i in range(100)]
        follower.route_avoidance = dict(path=path, speed_cap_kmh=12.,
                                         state='PASSING', reason='confirmed_blocker')
        follower.route_avoidance_received = time.monotonic()
        self.run_loop(fixture, follower, leo_tests)
        self.assertEqual(self.path_points(follower), path)
        self.assertEqual(self.state(follower)['local_path_source'], 'route_avoidance')
        self.assertEqual(self.state(follower)['route_reason'], 'confirmed_blocker')
        self.assertGreater(follower.commands[-1][0], 0.)

    def test_semi_path_reports_the_steering_owner_when_both_planners_run(self):
        for local_owns in (False, True):
            with self.subTest(local_owns=local_owns):
                fixture, follower = self.make_follower(leo_tests)
                route_path = [(float(i), 1.) for i in range(100)]
                local_path = [(float(i), 3.5) for i in range(100)]
                follower.route_avoidance_topic = '/leo/route_avoidance'
                follower.route_avoidance = dict(path=route_path, speed_cap_kmh=12.,
                    state='PASSING', reason='route_obstacle')
                follower.route_avoidance_received = time.monotonic()
                follower.enable_local_avoidance = True
                follower.local_planner = LeoPlanner(follower.waypoints, [1]*100, {})
                follower.local_planner.active_lane = 2 if local_owns else None
                follower.local_planner.plan = MagicMock(return_value=Decision(
                    'LANE_CHANGING', local_path, 16., reason='local_obstacle'))
                follower.obstacles = []
                follower.obstacle_received = time.monotonic()
                follower.obstacle_stamp = 100.
                self.run_loop(fixture, follower, leo_tests)
                state = self.state(follower)
                self.assertEqual(state['planner_mode'], 'route_and_local_avoidance')
                self.assertEqual(state['local_path_source'],
                                 'local_planner' if local_owns else 'route_avoidance')
                self.assertEqual(self.path_points(follower), local_path if local_owns else route_path)
                # Steering ownership alone does not discard an unspecified
                # route obstacle cap; only a known static path wait may yield.
                self.assertEqual(follower.commands[-1][1], 12.)
                self.assertEqual(state['route_reason'], 'route_obstacle')
                self.assertEqual(state['planner_reason'],
                                 'local_obstacle' if local_owns else 'selected_route_pass')
                if local_owns:
                    follower.local_planner.plan.assert_called_once()
                else:
                    follower.local_planner.plan.assert_not_called()

    def test_waypoint_only_zone_reports_bypassed_avoidance(self):
        fixture, follower = self.make_follower(leo_tests)
        follower.enable_local_avoidance = True
        follower.route_avoidance_topic = '/leo/route_avoidance'
        follower.waypoint_only_ranges = [(0, 20)]
        self.run_loop(fixture, follower, leo_tests)
        self.assertEqual(self.state(follower)['planner_mode'], 'waypoint_only')
        self.assertEqual(self.state(follower)['local_path_source'], 'saved_route')
        self.assertEqual(follower.commands[-1][1], 20.)

    def test_stop_clears_previous_path_and_reports_invalid(self):
        for test_module in (primary_tests, leo_tests):
            with self.subTest(follower=test_module.__name__):
                fixture, follower = self.make_follower(test_module)
                self.run_loop(fixture, follower, test_module)
                follower.ctrl_pub = MagicMock()
                follower.speed_pid = MagicMock()
                follower.publish_full_stop = test_module.module.GPSWaypointFollower.publish_full_stop.__get__(follower)
                with patch.object(test_module.module.rospy.Time, 'now',
                                  return_value=test_module.module.rospy.Time.from_sec(100.)):
                    follower.publish_full_stop()
                self.assertEqual(self.path_points(follower), [])
                self.assertFalse(self.state(follower)['valid'])
                self.assertEqual(self.state(follower)['local_path_source'], 'none')

    def test_stale_perception_tick_does_not_reuse_previous_decision(self):
        for test_module in (primary_tests, leo_tests):
            with self.subTest(follower=test_module.__name__):
                fixture, follower = self.make_follower(test_module)
                self.run_loop(fixture, follower, test_module)
                follower.debug_trace.update(planner_reason='old_lane_change',
                    transition=dict(source=1, target=2), planner_state='LANE_CHANGING')
                follower.enable_local_avoidance = True
                follower.obstacle_received = None
                follower.ctrl_pub = MagicMock()
                follower.speed_pid = MagicMock()
                follower.publish_full_stop = test_module.module.GPSWaypointFollower.publish_full_stop.__get__(follower)
                self.run_loop(fixture, follower, test_module)
                state = self.state(follower)
                self.assertFalse(state['valid'])
                self.assertEqual(state['stop_reason'], 'obstacle_perception_stale')
                self.assertNotIn('planner_reason', state)
                self.assertNotIn('transition', state)
                self.assertNotIn('pursuit_target', state)
                self.assertEqual(self.path_points(follower), [])

    def test_blocked_decision_remains_diagnosable_after_path_is_cleared(self):
        for test_module, planner_type in ((primary_tests, LocalAvoidancePlanner),
                                          (leo_tests, LeoPlanner)):
            with self.subTest(follower=test_module.__name__):
                fixture, follower = self.make_follower(test_module)
                follower.enable_local_avoidance = True
                follower.local_planner = planner_type(follower.waypoints, [1]*100, {})
                follower.local_planner.plan = MagicMock(return_value=Decision(
                    'BLOCKED', [], 0., reason='preferred_saved_path_missing'))
                follower.obstacles = []
                follower.obstacle_received = time.monotonic()
                follower.obstacle_stamp = 100.
                follower.obstacle_horizon_pub = MagicMock()
                follower.ctrl_pub = MagicMock()
                follower.speed_pid = MagicMock()
                follower.publish_full_stop = test_module.module.GPSWaypointFollower.publish_full_stop.__get__(follower)
                self.run_loop(fixture, follower, test_module)
                state = self.state(follower)
                self.assertFalse(state['valid'])
                self.assertEqual(state['planner_state'], 'BLOCKED')
                self.assertEqual(state['planner_reason'], 'preferred_saved_path_missing')
                self.assertEqual(state['stop_reason'], 'planner_blocked:preferred_saved_path_missing')
                self.assertEqual(state['local_path_source'], 'none')
                self.assertEqual(self.path_points(follower), [])

    def test_blackout_target_is_not_misrepresented_as_generated_path(self):
        fixture, follower = self.make_follower(leo_tests)
        now = time.monotonic()
        follower.drive_pose_topic = '/leo/driving_pose'
        follower.drive_pose_received = follower.drive_valid_received = follower.drive_mode_received = now
        follower.drive_pose_ready = follower.drive_valid = True
        follower.drive_mode = 'BLACKOUT'
        follower.avoidance_target = (4., 2.)
        follower.avoidance_speed_kmh = 10.
        follower.avoidance_target_received = follower.avoidance_speed_received = now
        self.run_loop(fixture, follower, leo_tests)
        self.assertEqual(self.path_points(follower), [])
        self.assertEqual(self.state(follower)['planner_mode'], 'external_blackout_target')
        self.assertEqual(self.state(follower)['pursuit_target'], {'x': 4., 'y': 2.})


if __name__ == '__main__':
    unittest.main()
