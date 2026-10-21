"""Keep the collision-checked saved-lane return selected until R1 is reached."""
import json
import time
import unittest

from std_msgs.msg import String

import test_loop_tracking_local as tracking_tests

module = tracking_tests.module


class ReturnOwnershipTests(unittest.TestCase):
    def follower(self, state='WAIT_PASS', cap=0.,
                 reason='static block ahead beyond pass planning', traffic_cap=None):
        route = [(float(i), 0.) for i in range(200)]
        obj = tracking_tests.LoopTrackingTests().follower(route)
        obj.current_x, obj.current_y = 40., 3.5
        obj.current_waypoint_index = 40
        obj.first_nearest_search = False
        obj.current_speed_kmh = 10.
        obj.enable_local_avoidance = True
        lane = {i: (float(i), 3.5) for i in range(200)}
        obj.local_planner = module.LocalAvoidancePlanner(
            route, [4] * len(route), {2: list(lane.values())}, lane_indices={2: lane})
        obj.local_planner.active_lane = 2
        obj.obstacles = []
        obj.obstacle_received = time.monotonic()
        obj.obstacle_stamp = 100.
        obj.route_avoidance_topic = '/leo/route_avoidance'
        obj.route_avoidance_callback(String(data=json.dumps(dict(
            path=route[40:110], speed_cap_kmh=cap, traffic_speed_cap_kmh=traffic_cap,
            state=state, reason=reason))))
        return obj

    def run_loop(self, obj):
        tracking_tests.LoopTrackingTests().run_loop(obj)

    def test_return_keeps_local_path_and_speed_past_static_route_zero_cap(self):
        # The recorded failure at bag +90.052 s had this exact route state,
        # with active_lane cleared as the offset-based R2 -> R1 return began.
        obj = self.follower()
        self.run_loop(obj)

        self.assertEqual(obj.local_planner.transition['state'], 'RETURNING')
        self.assertEqual(obj.local_planner.transition['source'], 2)
        self.assertEqual(obj.local_planner.transition['target'], 1)
        self.assertIn('offset', obj.local_planner.transition)
        self.assertIsNone(obj.local_planner.active_lane)
        self.assertEqual(obj.debug_trace['local_path_source'], 'local_planner')
        self.assertEqual(obj.commands[-1][1], 20.)
        self.assertIsNone(obj.stop_reason)

        # Crossing the nearest-lane boundary alone must not hand over early.
        obj.current_x, obj.current_y = 44., 1.2
        self.run_loop(obj)
        self.assertIsNotNone(obj.local_planner.transition)
        self.assertEqual(obj.debug_trace['local_path_source'], 'local_planner')
        self.assertEqual(obj.commands[-1][1], 20.)

        # Once the return is complete, route planning resumes including its cap.
        obj.current_x, obj.current_y = 50., 0.
        self.run_loop(obj)
        self.assertIsNone(obj.local_planner.transition)
        self.assertEqual(obj.stop_reason, 'planned_speed_zero')
        self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_route_safety_stops_still_apply_during_return(self):
        for state, reason in (
                ('YIELD', 'predicted crossing at s+3.0 m'),
                ('YIELD', 'mover beside the bumper'),
                ('CROSSWALK_STOP', 'person on the crosswalk'),
                ('EMERGENCY', 'emergency stop'),
                ('WAIT_PASS', 'red stop line inside the pass'),
                ('WAIT_PASS', 'left lane occupied at s+3.0 m'),
                ('UNRECOGNIZED_STATE', 'new safety condition')):
            with self.subTest(state=state, reason=reason):
                obj = self.follower(state=state, reason=reason)
                self.run_loop(obj)
                self.assertEqual(obj.debug_trace['planner_state'], 'RETURNING')
                self.assertEqual(obj.stop_reason, 'planned_speed_zero')
                self.assertFalse(obj.commands)
                self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_moving_traffic_speed_limit_still_applies_during_return(self):
        obj = self.follower('FOLLOW_LEAD', 7., 'following a lead mover')
        self.run_loop(obj)
        self.assertEqual(obj.debug_trace['local_path_source'], 'local_planner')
        self.assertEqual(obj.commands[-1][1], 7.)

    def test_static_zero_cap_cannot_hide_simultaneous_traffic_limit(self):
        for traffic_cap in (0., 7.):
            with self.subTest(traffic_cap=traffic_cap):
                obj = self.follower(traffic_cap=traffic_cap)
                self.run_loop(obj)
                self.assertEqual(obj.debug_trace['planner_state'], 'RETURNING')
                if traffic_cap == 0.:
                    self.assertEqual(obj.stop_reason, 'planned_speed_zero')
                    self.assertFalse(obj.commands)
                    self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)
                else:
                    self.assertEqual(obj.commands[-1][1], 7.)

    def test_legacy_route_output_keeps_full_cap_until_producer_is_updated(self):
        obj = self.follower()
        legacy = dict(obj.route_avoidance)
        del legacy['traffic_speed_cap_kmh']
        obj.route_avoidance_callback(String(data=json.dumps(legacy)))
        self.run_loop(obj)
        self.assertEqual(obj.stop_reason, 'planned_speed_zero')
        self.assertFalse(obj.commands)
        self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_invalid_traffic_limit_invalidates_route_input(self):
        for traffic_cap in (-1., float('nan'), 'invalid'):
            with self.subTest(traffic_cap=traffic_cap):
                obj = self.follower()
                data = dict(obj.route_avoidance, traffic_speed_cap_kmh=traffic_cap)
                obj.route_avoidance_callback(String(data=json.dumps(data)))
                self.run_loop(obj)
                self.assertEqual(obj.stop_reason, 'route_avoidance_stale')
                self.assertFalse(obj.commands)
                self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_signal_guard_can_still_stop_return(self):
        obj = self.follower()
        obj.guard_limit_topic = '/leo/speed_limit_kmh'
        obj.guard_limit_received = time.monotonic()
        obj.guard_speed_limit = 0.
        self.run_loop(obj)
        self.assertEqual(obj.stop_reason, 'guard_speed_zero_or_stale')
        self.assertFalse(obj.commands)
        self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_cp11_handoff_keeps_static_route_stop_even_with_separate_traffic_cap(self):
        obj = self.follower()
        obj.r1_handoff_ranges = [(0, 45)]
        obj.r1_handoff_pending = False
        obj.r1_handoff_end_index = None
        self.run_loop(obj)
        self.assertTrue(obj.debug_trace['r1_handoff_active'])
        self.assertEqual(obj.stop_reason, 'planned_speed_zero')
        self.assertFalse(obj.commands)
        self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_return_requires_fresh_route_and_obstacle_inputs(self):
        for stale_input, stop_reason in (
                ('route_avoidance_received', 'route_avoidance_stale'),
                ('obstacle_received', 'obstacle_perception_stale')):
            with self.subTest(stale_input=stale_input):
                obj = self.follower()
                setattr(obj, stale_input, time.monotonic() - 1.)
                self.run_loop(obj)
                self.assertEqual(obj.stop_reason, stop_reason)
                self.assertFalse(obj.commands)
                self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_collision_blocked_return_still_stops(self):
        obj = self.follower()
        obj.local_planner.transition = dict(source=2, target=1, start=40.,
            length=10., state='RETURNING', shape='smooth', lateral=3.5, offset=3.5)
        obj.obstacles = [module.Obstacle(41., 3.5, .8, kind='static')]
        self.run_loop(obj)
        self.assertEqual(obj.stop_reason, 'planner_blocked:transition_obstructed')
        self.assertFalse(obj.commands)
        self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_r1_offset_correction_keeps_route_ownership(self):
        obj = self.follower()
        obj.local_planner.active_lane = None
        obj.local_planner.transition = dict(source=1, target=1, offset=.8,
                                           state='RETURNING')
        self.assertFalse(obj.local_planner_owns_steering())


if __name__ == '__main__':
    unittest.main()
