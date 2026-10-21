"""CP11 controller ownership must hand Leo an R1 plan, with safety intact."""
import json
import math
import time
import unittest
from unittest.mock import patch

from std_msgs.msg import String

import test_loop_tracking_local as tracking_tests

module = tracking_tests.module


def planner():
    route = [(float(x), 0.) for x in range(200)]
    # Adjacent saved lanes end before the handoff, as in the CP11 stop log.
    lanes = {number: {i: (float(i), y) for i in range(20, 61)}
             for number, y in ((2, 3.6), (3, 7.2))}
    return module.LocalAvoidancePlanner(
        route, [11] * len(route),
        {number: list(points.values()) for number, points in lanes.items()},
        lane_indices=lanes)


def stale_transition():
    return dict(source=3, target=2, start=40., length=15.,
                state='OVERTAKING', shape='smooth', lateral=3.6)


class R1HandoffPlannerTests(unittest.TestCase):
    def test_handoff_clears_transition_to_ended_lane_and_keeps_progress(self):
        p = planner()
        p.transition = stale_transition()
        p.active_lane = 2
        p.departed_preferred = p.off_r1 = True
        p.previous_preferred_lane = 2
        before = p.plan(80, (80., 0.), 3., [], ego_yaw=0.)
        self.assertEqual((before.state, before.reason),
                         ('BLOCKED', 'transition_obstructed'))
        self.assertEqual(before.path, [])
        p.lap_offset = p.total_length
        progress = (p.previous_arc, p.lap_offset)

        p.reset_for_r1_handoff()

        self.assertIsNone(p.transition)
        self.assertIsNone(p.active_lane)
        self.assertFalse(p.departed_preferred)
        self.assertFalse(p.off_r1)
        self.assertIsNone(p.previous_preferred_lane)
        self.assertEqual((p.previous_arc, p.lap_offset), progress)
        after = p.plan(80, (80., 0.), 3., [], ego_yaw=0., force_r1=True)
        self.assertEqual(after.state, 'RACELINE')
        self.assertEqual(p.effective_preferred_lane, 1)
        self.assertEqual(after.speed_cap_kmh, math.inf)
        self.assertIsNone(p.transition)

    def test_handoff_follows_r1_instead_of_open_adjacent_lane(self):
        obstacle = module.Obstacle(50., 0., .9, kind='static')
        normal = planner()
        ordinary = normal.plan(30, (30., 0.), 3., [obstacle], ego_yaw=0.)
        self.assertEqual(ordinary.state, 'OVERTAKING')
        self.assertEqual(normal.transition['target'], 2)

        protected = planner()
        decision = protected.plan(30, (30., 0.), 3., [obstacle],
                                  ego_yaw=0., force_r1=True)
        self.assertEqual(decision.state, 'TAILING')
        self.assertGreater(decision.speed_cap_kmh, 0.)
        self.assertTrue(math.isfinite(decision.speed_cap_kmh))
        self.assertIsNone(protected.transition)
        self.assertIsNone(protected.active_lane)
        self.assertTrue(all(abs(y) < 1e-9 for _, y in decision.path))

    def test_r1_protection_keeps_emergency_obstacle_stop(self):
        p = planner()
        obstacle = module.Obstacle(34., 0., .9, kind='static')
        decision = p.plan(30, (30., 0.), 3., [obstacle],
                          ego_yaw=0., force_r1=True)
        self.assertEqual(decision.speed_cap_kmh, 0.)
        self.assertIsNone(p.transition)
        self.assertEqual(p.effective_preferred_lane, 1)

    def test_forced_plan_rejects_stale_non_r1_transition_directly(self):
        p = planner()
        p.transition = stale_transition()
        decision = p.plan(80, (80., 0.), 3., [], ego_yaw=0., force_r1=True)
        self.assertEqual(decision.state, 'RACELINE')
        self.assertIsNone(p.transition)

    def test_actual_offset_returns_to_r1_without_requiring_ended_r2(self):
        p = planner()
        p.transition = stale_transition()
        decision = p.plan(80, (80., 3.6), 3., [], ego_yaw=0., force_r1=True)
        self.assertEqual(decision.state, 'RETURNING')
        self.assertEqual(p.transition['target'], 1)
        self.assertAlmostEqual(decision.path[0][1], 3.6)
        self.assertAlmostEqual(decision.path[-1][1], 0.)
        transition_start = p.transition['start']
        p.plan(81, (81., 3.5), 3., [], ego_yaw=0., force_r1=True)
        self.assertEqual(p.transition['start'], transition_start)

    def test_forced_return_still_checks_approaching_vehicle_on_r1(self):
        p = planner()
        car = module.Obstacle(44., 0., .9, vx=8., vy=0., kind='dynamic')
        decision = p.plan(50, (50., 3.6), 3., [car], ego_yaw=0., force_r1=True)
        self.assertNotEqual(decision.state, 'RETURNING')
        self.assertIsNone(p.transition)

    def test_disabled_protection_preserves_configured_lane_preference(self):
        p = planner()
        p.preferred_lane_indices = {30: 2}
        normal = p.plan(30, (30., 0.), 3., [], ego_yaw=0.)
        self.assertEqual(normal.state, 'LANE_CHANGING')
        self.assertEqual(p.transition['target'], 2)

        p = planner()
        p.preferred_lane_indices = {30: 2}
        protected = p.plan(30, (30., 0.), 3., [], ego_yaw=0., force_r1=True)
        self.assertEqual(protected.state, 'RACELINE')
        self.assertEqual(p.effective_preferred_lane, 1)
        self.assertIsNone(p.transition)


class R1HandoffFollowerTests(unittest.TestCase):
    def follower(self, index=30):
        obj = tracking_tests.LoopTrackingTests().follower(
            [(float(x), 0.) for x in range(200)], loop=True)
        obj.current_waypoint_index = index
        obj.current_x = float(index)
        obj.first_nearest_search = False
        obj.enable_local_avoidance = True
        obj.local_planner = planner()
        obj.obstacles = []
        obj.obstacle_received = time.monotonic()
        obj.obstacle_stamp = 100.
        obj.r1_handoff_ranges = [(10, 60)]
        obj.r1_handoff_pending = False
        obj.r1_handoff_end_index = None
        obj.handoff_guard_status = None
        obj.handoff_guard_received = None
        return obj

    def guard(self, obj, index, state='clear', armed=True, received=None):
        received = time.monotonic() if received is None else received
        with patch.object(module.time, 'monotonic', return_value=received):
            obj.handoff_guard_callback(String(data=json.dumps(dict(
                state=state, armed=armed, route_index=index))))

    def decision(self, state='RACELINE'):
        return module.LocalAvoidancePlanner._decision(
            None, state, [], math.inf, reason='test')

    def enter(self, obj, now):
        obj.current_waypoint_index = 30
        self.guard(obj, 30, state='passthrough', received=now)
        self.assertTrue(obj.prepare_r1_handoff(now))
        self.assertTrue(obj.r1_handoff_pending)

    def test_passthrough_clears_unused_transition_on_every_tick(self):
        obj = self.follower()
        now = time.monotonic()
        self.guard(obj, 30, state='passthrough', received=now)
        for _ in range(2):
            obj.local_planner.transition = stale_transition()
            obj.local_planner.active_lane = 2
            self.assertTrue(obj.prepare_r1_handoff(now))
            self.assertIsNone(obj.local_planner.transition)
            self.assertIsNone(obj.local_planner.active_lane)
        self.assertEqual(obj.r1_handoff_end_index, 60)

    def test_own_route_index_protects_while_guard_update_lags(self):
        obj = self.follower()
        now = time.monotonic()
        self.guard(obj, 9, received=now)
        self.assertTrue(obj.prepare_r1_handoff(now))
        obj.current_waypoint_index = 61
        self.guard(obj, 60, state='passthrough', received=now)
        obj.local_planner.transition = stale_transition()
        self.assertTrue(obj.prepare_r1_handoff(now))
        self.assertIsNone(obj.local_planner.transition)
        obj.complete_r1_handoff(now, self.decision())
        self.assertTrue(obj.r1_handoff_pending)

    def test_release_requires_fresh_armed_guard_beyond_cp11(self):
        for state, armed, guard_index, age in (
                ('clear', True, 61, .6), ('clear', False, 61, 0.),
                ('hold', True, 61, 0.), ('passthrough', True, 60, 0.),
                ('clear', True, 60, 0.)):
            with self.subTest(state=state, armed=armed, index=guard_index, age=age):
                obj = self.follower()
                now = time.monotonic()
                self.enter(obj, now)
                obj.current_waypoint_index = 61
                self.guard(obj, guard_index, state, armed, now-age)
                self.assertTrue(obj.prepare_r1_handoff(now))
                obj.complete_r1_handoff(now, self.decision())
                self.assertTrue(obj.r1_handoff_pending)

    def test_release_waits_until_vehicle_acquires_r1(self):
        for lane, off_r1, transition, state in (
                (2, False, None, 'RACELINE'),
                (1, True, None, 'RACELINE'),
                (1, False, dict(target=1), 'RETURNING'),
                (1, False, None, 'BLOCKED')):
            with self.subTest(lane=lane, off_r1=off_r1, state=state):
                obj = self.follower()
                now = time.monotonic()
                self.enter(obj, now)
                obj.current_waypoint_index = 61
                self.guard(obj, 61, received=now)
                self.assertTrue(obj.prepare_r1_handoff(now))
                obj.local_planner.current_lane = lane
                obj.local_planner.off_r1 = off_r1
                obj.local_planner.transition = transition
                obj.complete_r1_handoff(now, self.decision(state))
                self.assertTrue(obj.r1_handoff_pending)

    def test_guard_hold_without_route_index_prevents_release(self):
        obj = self.follower()
        now = time.monotonic()
        self.enter(obj, now)
        obj.current_waypoint_index = 61
        self.guard(obj, 61, received=now)
        with patch.object(module.time, 'monotonic', return_value=now):
            obj.handoff_guard_callback(String(data=json.dumps(dict(
                state='hold', armed=True, reason='localization unavailable'))))
        self.assertEqual(obj.handoff_guard_status['state'], 'hold')
        obj.complete_r1_handoff(now, self.decision())
        self.assertTrue(obj.r1_handoff_pending)

    def test_guard_cannot_release_while_follower_is_still_before_cp11(self):
        obj = self.follower()
        now = time.monotonic()
        self.enter(obj, now)
        self.guard(obj, 61, received=now)
        obj.complete_r1_handoff(now, self.decision())
        self.assertTrue(obj.r1_handoff_pending)

    def test_completes_after_r1_acquisition_and_rearms_next_lap(self):
        obj = self.follower()
        now = time.monotonic()
        for lap in range(2):
            self.enter(obj, now+lap)
            obj.current_waypoint_index = 61
            self.guard(obj, 61, received=now+lap)
            self.assertTrue(obj.prepare_r1_handoff(now+lap))
            obj.complete_r1_handoff(now+lap, self.decision())
            self.assertFalse(obj.r1_handoff_pending)
            self.assertFalse(obj.prepare_r1_handoff(now+lap))

    def test_route_offset_cannot_override_r1_but_speed_cap_still_applies(self):
        obj = self.follower()
        obj.local_planner.transition = stale_transition()
        obj.route_avoidance_topic = '/leo/route_avoidance'
        obj.route_avoidance_callback(String(data=json.dumps(dict(
            path=[[float(i), 3.5] for i in range(30, 90)],
            speed_cap_kmh=12., state='PASSING', reason='offset'))))

        tracking_tests.LoopTrackingTests().run_loop(obj)

        self.assertTrue(obj.commands)
        self.assertAlmostEqual(obj.commands[-1][0], 0., places=6)
        self.assertEqual(obj.commands[-1][1], 12.)
        self.assertIsNone(obj.local_planner.transition)
        self.assertEqual(obj.local_planner.effective_preferred_lane, 1)

    def test_zero_route_speed_cap_still_stops_during_handoff(self):
        obj = self.follower()
        obj.route_avoidance_topic = '/leo/route_avoidance'
        obj.route_avoidance_callback(String(data=json.dumps(dict(
            path=[[float(i), 3.5] for i in range(30, 90)],
            speed_cap_kmh=0., state='WAITING', reason='obstacle'))))

        tracking_tests.LoopTrackingTests().run_loop(obj)

        self.assertFalse(any(speed > 0. for _, speed in obj.commands))
        self.assertTrue(obj.commands or obj.ctrl_pub.publish.called)

    def test_stale_perception_still_holds_brake_during_handoff(self):
        obj = self.follower()
        obj.obstacle_received = time.monotonic()-1.

        tracking_tests.LoopTrackingTests().run_loop(obj)

        self.assertFalse(obj.commands)
        self.assertEqual(obj.stop_reason, 'obstacle_perception_stale')
        self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_stale_route_planner_still_holds_brake_during_handoff(self):
        obj = self.follower()
        obj.route_avoidance_topic = '/leo/route_avoidance'
        obj.route_avoidance = None
        obj.route_avoidance_received = None

        tracking_tests.LoopTrackingTests().run_loop(obj)

        self.assertFalse(obj.commands)
        self.assertEqual(obj.stop_reason, 'route_avoidance_stale')
        self.assertEqual(obj.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_teleport_outside_handoff_drops_previous_lap_pending_state(self):
        obj = self.follower(index=0)
        obj.r1_handoff_pending = True
        obj.r1_handoff_end_index = 60
        obj.last_tracking_position = (100., 0.)
        self.guard(obj, 60, state='passthrough')

        tracking_tests.LoopTrackingTests().run_loop(obj)

        self.assertFalse(obj.r1_handoff_pending)
        self.assertIsNone(obj.handoff_guard_status)
        self.assertTrue(obj.commands)


if __name__ == '__main__':
    unittest.main()
