import importlib.util
import math
from pathlib import Path
import threading
from collections import deque
import unittest
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/local_avoidance_core.py'
spec = importlib.util.spec_from_file_location('local_avoidance_core', SCRIPT)
core = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = core
spec.loader.exec_module(core)
detector_spec = importlib.util.spec_from_file_location(
    'lidar_obstacles', SCRIPT.parent / 'lidar_obstacles.py')
detector = importlib.util.module_from_spec(detector_spec)
detector_spec.loader.exec_module(detector)


def planner(lanes=True):
    route = [(float(i), 0.) for i in range(100)]
    adjacent = {2: [(float(i), 3.5) for i in range(100)]} if lanes else {}
    return core.LocalAvoidancePlanner(route, [11]*len(route), adjacent)


class LocalAvoidanceTests(unittest.TestCase):
    def test_passes_static_obstacle_and_returns_when_gap_is_clear(self):
        p = planner()
        obstacle = core.Obstacle(25., 0., 1., kind='static')
        decision = p.plan(0, (0., 0.), 6., [obstacle])
        self.assertEqual(decision.state, 'OVERTAKING')
        self.assertTrue(decision.path)
        self.assertGreater(decision.path[25][1], 3.)
        self.assertAlmostEqual(decision.path[-1][1], 3.5, places=3)
        self.assertEqual(decision.obstacle_kind, 'static')
        # No scheduled merge in front of the passed vehicle. Replan after it
        # is behind the ego, with the rear clearance checked separately.
        recovery = p.plan(40, (40., 3.5), 5., [obstacle])
        self.assertEqual(recovery.state, 'RETURNING')
        self.assertAlmostEqual(recovery.path[-1][1], 0., places=3)

    def test_no_adjacent_route_trails_and_can_accelerate_from_rest(self):
        p = planner(lanes=False)
        decision = p.plan(0, (0., 0.), 0., [core.Obstacle(25., 0., 1.)])
        self.assertEqual(decision.state, 'TAILING')
        self.assertGreater(decision.speed_cap_kmh, 0.)
        self.assertEqual(decision.path, [])

    def test_blocked_adjacent_route_does_not_pass(self):
        p = planner()
        decision = p.plan(0, (0., 0.), 6., [
            core.Obstacle(25., 0., 1., kind='dynamic'),
            core.Obstacle(25., 3.5, 1., kind='static')])
        self.assertEqual(decision.state, 'TAILING')
        self.assertEqual(decision.obstacle_kind, 'dynamic')

    def test_near_obstacle_does_not_start_late_lane_change(self):
        p = planner()
        decision = p.plan(0, (0., 0.), 10., [core.Obstacle(7., 0., 1.)])
        self.assertEqual(decision.state, 'TAILING')
        self.assertLess(decision.speed_cap_kmh, 20.)

    def test_clear_fast_pass_has_no_lane_change_speed_cap(self):
        p = planner()
        decision = p.plan(0, (0., 0.), 14., [core.Obstacle(30., 0., 1.)])
        self.assertEqual(decision.state, 'OVERTAKING')
        self.assertTrue(math.isinf(decision.speed_cap_kmh))
        continued = p.plan(5, decision.path[5], 14., [core.Obstacle(30., 0., 1.)])
        self.assertEqual(continued.state, 'OVERTAKING')
        self.assertTrue(math.isinf(continued.speed_cap_kmh))

    def test_fast_lane_change_has_longer_smoother_path(self):
        p = self.preferred_planner()
        slow = p.plan(0, (0., 0.), 5., [])
        slow_length = p.transition['length']
        p.transition = None
        fast = p.plan(0, (0., 0.), 25., [])
        self.assertEqual(fast.state, 'LANE_CHANGING')
        self.assertAlmostEqual(p.transition['length'], 25.*p.lane_change_smoothness_s)
        self.assertGreater(p.transition['length'], slow_length)
        self.assertLess(fast.path[20][1], slow.path[20][1])
        self.assertLess(fast.path[20][1], fast.path[-1][1])

    def test_clear_preferred_lane_change_and_return_keep_speed_profile(self):
        p = self.preferred_planner()
        decision = p.plan(0, (0., 0.), 25., [])
        self.assertEqual(decision.state, 'LANE_CHANGING')
        self.assertTrue(math.isinf(decision.speed_cap_kmh))
        p.transition = None
        p.preferred_lane = 1
        p.departed_preferred = True
        returned = p.plan(30, (30., 7.), 25., [])
        self.assertEqual(returned.state, 'RETURNING')
        self.assertTrue(math.isinf(returned.speed_cap_kmh))

    def test_blocked_fast_pass_still_uses_obstacle_stopping_cap(self):
        p = planner()
        decision = p.plan(0, (0., 0.), 25., [
            core.Obstacle(15., 0., 1.), core.Obstacle(15., 3.5, 1.)])
        self.assertEqual(decision.state, 'TAILING')
        self.assertLess(decision.speed_cap_kmh, 25.*3.6)

    def test_prediction_uses_actual_lane_change_speed(self):
        p = self.preferred_planner()
        # Parallel traffic stays alongside at this speed. Clamping the ego's
        # predicted speed to 20 km/h would incorrectly let this car move away.
        decision = p.plan(0, (0., 0.), 20., [
            core.Obstacle(0., 3.5, 1., vx=20., kind='dynamic')])
        self.assertEqual(decision.state, 'LANE_WAIT')
        self.assertEqual(decision.reason, 'predicted_path_occupied')

    def test_horizon_edit_changes_threat_decision_and_visualized_distance(self):
        p = planner()
        p.obstacle_horizon_m = 20.
        obstacle = core.Obstacle(30., 0., 1.)
        self.assertEqual(p.plan(0, (0., 0.), 14., [obstacle]).state, 'RACELINE')
        self.assertEqual(p.obstacle_horizon_path[-1], (20., 0.))
        p.obstacle_horizon_m = 40.
        self.assertEqual(p.plan(0, (0., 0.), 14., [obstacle]).state, 'OVERTAKING')
        path = p.obstacle_horizon_path
        self.assertAlmostEqual(sum(core.distance(a, b) for a, b in zip(path, path[1:])), 40.)

    def test_curved_horizon_is_arc_distance_and_clips_inside_segment(self):
        self.assertEqual(core.path_prefix([(0., 0.), (3., 0.), (3., 4.)], 5.),
                         [(0., 0.), (3., 0.), (3., 2.)])

    def test_replanning_starts_at_current_offset_and_returns(self):
        p = planner()
        obstacle = core.Obstacle(25., 0., 1.)
        self.assertEqual(p.plan(0, (0., 0.), 6., [obstacle]).state, 'OVERTAKING')
        continuation = p.plan(10, (10., 2.), 6., [obstacle])
        self.assertEqual(continuation.state, 'OVERTAKING')
        self.assertAlmostEqual(continuation.path[0][1], 2., places=2)
        for _ in range(6):
            recovery = p.plan(31, (31., 3.5), 5., [])
        self.assertEqual(recovery.state, 'RETURNING')
        self.assertAlmostEqual(recovery.path[0][1], 3.5, places=2)
        self.assertAlmostEqual(recovery.path[-1][1], 0., places=2)

    def test_blocked_r3_reselects_r2_from_actual_vehicle_position(self):
        p = planner()
        p.lanes[3] = [(float(i), 7.) for i in range(100)]
        p.active_lane = 3
        decision = p.plan(0, (0., 7.), 5., [
            core.Obstacle(30., 0., .8), core.Obstacle(30., 7., .8)])
        self.assertEqual(decision.state, 'OVERTAKING')
        self.assertEqual(p.active_lane, 2)
        self.assertAlmostEqual(decision.path[0][1], 7.)

    def test_map_frame_tracking_distinguishes_moving_and_parked(self):
        node = detector.LidarObstacles.__new__(detector.LidarObstacles)
        node.dynamic_speed_threshold_mps = .8
        node.tracks = {}
        node.next_id = 1
        for frame in range(4):
            tracked = node._update_tracks([
                (20.+frame*.25, 0., .8), (30., 4., .8)], frame*.1)
        self.assertEqual(tracked[0]['kind'], 'dynamic')
        self.assertEqual(tracked[1]['kind'], 'static')

    def test_clustered_lidar_returns_form_individual_objects(self):
        first = np.array([[10.+i*.08, j*.08] for i in range(5) for j in range(5)])
        second = first + [4., 0.]
        self.assertEqual(len(detector.clusters(np.vstack((first, second)))), 2)

    def test_cloud_pose_interpolates_at_capture_time(self):
        node = detector.LidarObstacles.__new__(detector.LidarObstacles)
        node.lock = threading.Lock()
        node.max_pose_age = .15
        node.poses = deque([(1., (10., 0., 0.)), (1.1, (11., 0., 0.))])
        self.assertAlmostEqual(node.pose_at(1.05)[0], 10.5)
        self.assertIsNone(node.pose_at(1.2))

    def preferred_planner(self):
        route = [(float(i), 0.) for i in range(250)]
        lanes = {2: [(x, 3.5) for x, _ in route],
                 3: [(x, 7.) for x, _ in route]}
        return core.LocalAvoidancePlanner(route, [11]*len(route), lanes, preferred_lane=3)

    def test_rear_and_side_roi_retains_traffic_and_removes_ego_body(self):
        points = np.array([[-30., 7., 1.], [-5., 0., 1.], [1., 3.5, 1.],
                           [25., 0., 1.], [1.4, 0., 1.23], [-.5, 0., .5],
                           [-56., 3.5, 1.], [20., 0., 0.]])
        self.assertEqual(detector.road_obstacle_mask(points).tolist(),
                         [True, True, True, True, False, False, False, False])

    def test_long_rear_truck_is_kept_and_circles_cover_observed_returns(self):
        truck = np.array([(x, y) for x in np.linspace(-30., -15., 61)
                          for y in (-1., 0., 1.)])
        pieces = detector.object_circles(truck)
        self.assertGreater(len(pieces), 1)
        for x, y in truck:
            self.assertTrue(any(math.hypot(x-cx, y-cy) <= r+1e-6
                                for cx, cy, r in pieces))

    def test_prefers_r3_and_transition_end_does_not_move_when_replanning(self):
        p = self.preferred_planner()
        d = p.plan(0, (0., 0.), 5., [])
        self.assertEqual(d.state, 'LANE_CHANGING')
        self.assertEqual(p.active_lane, 3)
        end = p.transition['start']+p.transition['length']
        continuing = p.plan(5, d.path[5], 5., [])
        self.assertEqual(p.transition['start']+p.transition['length'], end)
        self.assertEqual(continuing.path[0], d.path[5])
        settled = p.plan(26, (26., 7.), 5., [])
        self.assertEqual(settled.state, 'RACELINE')
        self.assertEqual(p.active_lane, 3)
        self.assertTrue(all(abs(y-7.) < .01 for _, y in settled.path))

    def test_fast_rear_car_blocks_preferred_lane_change(self):
        p = self.preferred_planner()
        d = p.plan(0, (0., 0.), 5., [core.Obstacle(-35., 7., 1., vx=18., kind='dynamic')])
        self.assertEqual(d.state, 'LANE_WAIT')
        self.assertEqual(d.reason, 'rear_or_crossed_lane_busy')
        self.assertIsNone(p.active_lane)
        self.assertIsNone(p.transition)
        self.assertEqual(d.path, [])
        self.assertGreater(d.speed_cap_kmh, 0.)

    def test_unknown_rear_velocity_waits_for_tracking_to_converge(self):
        p = self.preferred_planner()
        d = p.plan(0, (0., 0.), 5., [core.Obstacle(-40., 7., 1.)])
        self.assertEqual(d.state, 'LANE_WAIT')
        self.assertIsNone(p.transition)

    def test_close_slow_rear_car_blocks_and_distant_slow_car_allows_change(self):
        p = self.preferred_planner()
        close = p.plan(0, (0., 0.), 5., [core.Obstacle(-7., 7., 1., vx=3., kind='dynamic')])
        self.assertEqual(close.state, 'LANE_WAIT')
        clear = p.plan(0, (0., 0.), 5., [core.Obstacle(-40., 7., 1., vx=3., kind='dynamic')])
        self.assertEqual(clear.state, 'LANE_CHANGING')
        self.assertEqual(p.active_lane, 3)

    def test_intermediate_lane_rear_traffic_blocks_two_lane_merge(self):
        p = self.preferred_planner()
        d = p.plan(0, (0., 0.), 5., [core.Obstacle(-30., 3.5, 1., vx=18., kind='dynamic')])
        self.assertEqual(d.state, 'LANE_WAIT')
        self.assertIsNone(p.transition)

    def test_side_by_side_moving_vehicle_blocks_swept_body(self):
        p = self.preferred_planner()
        d = p.plan(0, (0., 0.), 5., [core.Obstacle(0., 3.5, 1., vx=5., kind='dynamic')])
        self.assertEqual(d.state, 'LANE_WAIT')
        self.assertIsNone(p.transition)

    def test_r3_dynamic_leader_passes_in_r2_then_waits_and_returns_to_r3(self):
        p = self.preferred_planner()
        d = p.plan(0, (0., 7.), 5., [core.Obstacle(25., 7., 1., vx=2., kind='dynamic')])
        self.assertEqual(d.state, 'OVERTAKING')
        self.assertEqual(p.active_lane, 2)
        held = p.plan(20, (20., 3.5), 5., [core.Obstacle(35., 7., 1., vx=2., kind='dynamic')])
        self.assertEqual(p.active_lane, 2)
        self.assertTrue(held.path)
        near = p.plan(50, (50., 3.5), 5., [core.Obstacle(45., 7., 1., vx=2., kind='dynamic')])
        self.assertEqual(near.state, 'LANE_WAIT')
        self.assertEqual(p.active_lane, 2)
        returned = p.plan(65, (65., 3.5), 5., [core.Obstacle(45., 7., 1., vx=2., kind='dynamic')])
        self.assertEqual(returned.state, 'RETURNING')
        self.assertEqual(p.active_lane, 3)
        self.assertAlmostEqual(returned.path[0][1], 3.5)
        self.assertAlmostEqual(returned.path[-1][1], 7.)
        settled = p.plan(85, (85., 7.), 5., [])
        self.assertEqual(settled.state, 'RACELINE')
        self.assertEqual(p.active_lane, 3)

    def test_no_free_passing_gap_tails_in_r3_without_switching_to_r1(self):
        p = self.preferred_planner()
        d = p.plan(0, (0., 7.), 5., [core.Obstacle(25., 7., 1., vx=2., kind='dynamic'),
                    core.Obstacle(-20., 3.5, 1., vx=18., kind='dynamic'),
                    core.Obstacle(-20., 0., 1., vx=18., kind='dynamic')])
        self.assertEqual(d.state, 'TAILING')
        self.assertEqual(p.active_lane, 3)
        self.assertTrue(all(abs(y-7.) < .01 for _, y in d.path))

    def test_r3_ends_returns_before_leaving_saved_coverage(self):
        p = self.preferred_planner()
        p.lane_indices = {n: {i: point for i, point in enumerate(lane) if i <= 70}
                          for n, lane in p.lanes.items()}
        d = p.plan(45, (45., 7.), 5., [])
        self.assertEqual(p.effective_preferred_lane, 1)
        self.assertIsNone(p.active_lane)
        self.assertIsNotNone(p.transition)
        self.assertAlmostEqual(d.path[0][1], 7.)
        self.assertAlmostEqual(d.path[-1][1], 0.)

    def test_rear_track_survives_brief_occlusion_but_expires(self):
        node = detector.LidarObstacles.__new__(detector.LidarObstacles)
        node.dynamic_speed_threshold_mps = .8
        node.tracks, node.next_id = {}, 1
        for i in range(4):
            tracked = node._update_tracks([(-40.+i*3., 7., 1.)], i*.1)
        self.assertEqual(tracked[0]['kind'], 'dynamic')
        missed = node._update_tracks([], .4)
        self.assertEqual(missed[0]['id'], tracked[0]['id'])
        self.assertGreater(missed[0]['x'], tracked[0]['x'])
        self.assertEqual(node._update_tracks([], .9), [])

    def test_collision_prediction_is_not_frozen_after_three_seconds(self):
        p = self.preferred_planner()
        path = [(float(i), 7.) for i in range(35)]
        self.assertFalse(p._collision_free(path,
                         [core.Obstacle(-30., 7., 1., vx=10., kind='dynamic')], 5.))


if __name__ == '__main__':
    unittest.main()
