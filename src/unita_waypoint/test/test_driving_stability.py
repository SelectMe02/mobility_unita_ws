"""Calibration, transient returns, near hazards and restart gate regressions."""
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import rospy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ground_segmentation import segment_ground
from local_avoidance_core import LocalAvoidancePlanner, Obstacle
from lidar_obstacles import LidarObstacles
from pedal_guard import PedalGuard
from gps_waypoint_follower import GPSWaypointFollower, PIDController


def planner():
    route = [(float(i), 0.) for i in range(150)]
    return LocalAvoidancePlanner(route, [11]*len(route),
                                 {3: [(x, 7.) for x, _ in route]}, preferred_lane=3)


class StabilityTests(unittest.TestCase):
    def test_shifted_road_is_confirmed_without_removing_low_object_or_roof(self):
        x, y = np.meshgrid(np.arange(5., 40., .4), np.arange(-2.5, 2.6, .4))
        road = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -.38)))
        objects = np.array([[10., 0., -.23], [12., 0., .8]])
        points = np.vstack((road, objects))
        old = segment_ground(points, np.ones(len(points), dtype=bool))
        ground, mask, status = segment_ground(points, np.ones(len(points), dtype=bool), reference_z=-.38)
        self.assertEqual(old[2]['planes'], 0)
        self.assertGreater(status['planes'], 0)
        self.assertTrue(ground[:len(road)].all())
        self.assertTrue(mask[-2:].all())
        self.assertFalse(ground[-2:].any())
        roof = road.copy(); roof[:, 2] = .8
        self.assertFalse(segment_ground(roof, np.ones(len(roof), dtype=bool), reference_z=-.38)[0].any())

    def test_small_track_requires_observation_time_and_bad_size_match_gets_new_id(self):
        n = LidarObstacles.__new__(LidarObstacles)
        n.tracks, n.next_id, n.dynamic_speed_threshold_mps = {}, 1, .8
        n.track_confirm_time_s = .2
        for i in range(3):
            o = n._update_tracks([(20.+i*.01, 0., .15)], 1.+i*.01)[0]
        self.assertFalse(o['confirmed'])
        self.assertFalse(o['velocity_reliable'])
        o = n._update_tracks([(20.2, 0., .15)], 1.25)[0]
        self.assertTrue(o['confirmed'])
        old_id = o['id']
        o = n._update_tracks([(20.2, 0., 1.5)], 1.35)[0]
        self.assertNotEqual(o['id'], old_id)
        self.assertFalse(o['confirmed'])

    def test_unstable_track_does_not_extrapolate_into_another_lane(self):
        n = LidarObstacles.__new__(LidarObstacles)
        n.tracks, n.next_id, n.dynamic_speed_threshold_mps = {}, 1, .8
        for i, x in enumerate([20., 20.2, 19.8, 21.]):
            o = n._update_tracks([(x, 0., .5)], 1.+i*.1)[0]
        self.assertTrue(o['confirmed'])
        self.assertFalse(o['velocity_reliable'])
        predicted = n._update_tracks([], 1.4)[0]
        self.assertEqual(predicted['x'], o['x'])

    def test_far_single_frame_return_does_not_start_pass_but_close_one_is_protected(self):
        p = planner();p.preferred_lane = 1
        far = p.plan(0, (0., 0.), 3., [Obstacle(35., 0., .15, confirmed=False, velocity_reliable=False)])
        self.assertEqual(far.state, 'RACELINE')
        self.assertIsNone(p.transition)
        path = [(float(i), 0.) for i in range(50)]
        urgent = p.transition_decision(path, [Obstacle(3., 0., .15, confirmed=False,
                                                     velocity_reliable=False)], 3., 8., 'RETURNING')
        self.assertEqual(urgent.state, 'BLOCKED')
        self.assertEqual(urgent.reason, 'transition_emergency')

    def test_distant_collision_slows_and_near_collision_stops(self):
        p = planner();path = [(float(i), 0.) for i in range(100)]
        far = p.transition_decision(path, [Obstacle(35., 0., 1.)], 5., 10., 'RETURNING')
        self.assertEqual(far.reason, 'transition_slowing')
        self.assertGreater(far.speed_cap_kmh, 0.)
        near = p.transition_decision(path, [Obstacle(7., 0., 1.)], 5., 10., 'RETURNING')
        self.assertEqual((near.state, near.speed_cap_kmh), ('BLOCKED', 0.))
        self.assertEqual(near.reason, 'transition_emergency')
        clear = p.transition_decision(path, [], 25., 10., 'RETURNING')
        self.assertTrue(math.isinf(clear.speed_cap_kmh))

    def test_stopping_distance_can_require_emergency_before_ttc_threshold(self):
        p = planner();path = [(float(i), 0.) for i in range(150)]
        d = p.transition_decision(path, [Obstacle(80., 0., 1.)], 25., 8., 'OVERTAKING')
        self.assertGreater(p.collision_risk.time_s, p.emergency_ttc_s)
        self.assertEqual(d.reason, 'transition_emergency')

    def test_clear_release_requires_time_and_distinct_sensor_frames(self):
        guard = PedalGuard();guard.stop(0.)
        for t in [.1, .2, .5, 1.]:
            self.assertEqual(guard.command(1., t, 10.)[:2], (0., 1.))
        self.assertEqual(guard.command(1., 1.1, 10.1)[:2], (0., 1.))
        accel, brake, _ = guard.command(1., 1.2, 10.2)
        self.assertGreater(accel, 0.)
        self.assertLessEqual(accel, .051)
        self.assertEqual(brake, 0.)
        guard.stop(1.25)
        self.assertEqual(guard.command(1., 1.3, 10.3)[:2], (0., 1.))

    def test_throttle_ramps_brake_is_immediate_and_clock_reset_restarts_confirmation(self):
        guard = PedalGuard()
        a = guard.command(1., 1.)[0]
        b = guard.command(1., 1.1)[0]
        self.assertLessEqual(b-a, .051)
        self.assertEqual(guard.command(-1., 1.15)[:2], (0., 1.))
        self.assertEqual(guard.command(-1., 1.2, brake_limit=.35)[:2], (0., .35))
        guard.stop(2.)
        guard.command(1., 2.1, 10.)
        guard.command(1., 2.3, 10.1)
        self.assertEqual(guard.command(1., 2.6, 9.)[:2], (0., 1.))

    def test_follower_soft_brake_limit_never_limits_full_stop(self):
        n = GPSWaypointFollower.__new__(GPSWaypointFollower)
        n.longitudinal_control_mode, n.control_rate = 'pid', 20.
        n.last_pid_update, n.current_speed_kmh = None, 50.
        n.speed_pid = PIDController(.08, .002, .01)
        n.ctrl_pub = MagicMock();n.lane_change_reason = 'transition_slowing'
        n.publish_control(0., 10.)
        self.assertAlmostEqual(n.ctrl_pub.publish.call_args.args[0].brake, .35)
        n.publish_full_stop()
        self.assertEqual(n.ctrl_pub.publish.call_args.args[0].brake, 1.)


if __name__ == '__main__':
    unittest.main()
