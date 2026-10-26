"""Calibration, transient returns, near hazards and restart gate regressions."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import rospy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ground_segmentation import segment_ground
from local_avoidance_core import LocalAvoidancePlanner, Obstacle
from lidar_obstacles import LidarObstacles, road_surface_evidence, road_surface_in_box
from lane_corridor import LaneCorridor
from pedal_guard import CruiseBrake, PedalGuard
from gps_waypoint_follower import GPSWaypointFollower, PIDController


def planner():
    route = [(float(i), 0.) for i in range(150)]
    return LocalAvoidancePlanner(route, [11]*len(route),
                                 {3: [(x, 7.) for x, _ in route]}, preferred_lane=3)


class StabilityTests(unittest.TestCase):
    def test_s10_saved_40_ignores_obstacles_and_restores_them_on_exit(self):
        n = GPSWaypointFollower.__new__(GPSWaypointFollower)
        n.route_sectors = [10, 10, 10, 11, 9]
        n.project_speed_limits_kmh = [55., 40., 55., 40., 40.]
        n.debug_trace = {}
        n.obstacles = [Obstacle(12., 0., 1., kind='static'),
                       Obstacle(15., 0., 1., kind='dynamic')]
        route = [(float(i), 0.) for i in range(150)]
        p = LocalAvoidancePlanner(route, [10]*len(route), {})
        for speed in (0., 25., 40., 55.):
            n.current_speed_kmh = speed
            for index in (0, 1, 2, 3, 4, -1, 5):
                n.current_waypoint_index = index
                selected = n.planning_obstacles()
                ignored = index == 1
                self.assertEqual(n.debug_trace['s10_40_lidar_ignored'], ignored)
                decision = p.plan(0, (0., 0.), 8., selected)
                if ignored:
                    self.assertEqual(selected, [])
                    self.assertTrue(math.isinf(decision.speed_cap_kmh))
                    self.assertTrue(math.isinf(decision.obstacle_distance_m))
                else:
                    self.assertIs(selected, n.obstacles)
                    self.assertLess(decision.speed_cap_kmh, 40.)
        del n.project_speed_limits_kmh
        n.current_waypoint_index = 1
        self.assertIs(n.planning_obstacles(), n.obstacles)

    def labelled_tracker(self, sector=11):
        n = LidarObstacles.__new__(LidarObstacles)
        n.tracks, n.next_id, n.dynamic_speed_threshold_mps = {}, 1, .8
        n.lane_corridor = LaneCorridor(dict(frame_id='map', raceline_width=3.6,
            loop=False, lanes=[dict(sector=sector, raceline=i,
                points=[[-50., (i-1)*3.6], [100., (i-1)*3.6]]) for i in (1, 2, 3)]))
        return n

    def detection(self, x=20., y=3.6, width=1.9):
        return (x, y, math.hypot(4.5, width)/2, 4.5, width, 0.)

    def test_midpoint_lane_needs_distinct_observations_and_elapsed_time(self):
        n = self.labelled_tracker()
        for stamp in (1., 1.01, 1.02):
            o = n._update_tracks([self.detection()], stamp)[0]
            self.assertEqual(o['raceline'], 2)
            self.assertFalse(o['lane_stable'])
        o = n._update_tracks([self.detection()], 1.21)[0]
        self.assertTrue(o['lane_stable'])
        self.assertFalse(o['lane_boundary_crossing'])

    def test_stable_neighbour_track_does_not_steal_another_lanes_detection(self):
        n = self.labelled_tracker()
        for stamp in (1., 1.1, 1.2):
            previous = n._update_tracks([self.detection()], stamp)[0]
        objects = n._update_tracks([self.detection(y=7.2)], 1.4)
        current = next(o for o in objects if not o['predicted'])
        self.assertNotEqual(current['id'], previous['id'])
        self.assertEqual(current['raceline'], 3)
        self.assertFalse(current['lane_stable'])

    def test_ambiguous_midpoints_do_not_count_toward_stable_label(self):
        n = self.labelled_tracker()
        for stamp in (1., 1.1, 1.2):
            o = n._update_tracks([self.detection(y=1.8)], stamp)[0]
            self.assertFalse(o['lane_stable'])
        o = n._update_tracks([self.detection(y=1.6)], 1.3)[0]
        self.assertFalse(o['lane_stable'])
        for stamp in (1.4, 1.5):
            o = n._update_tracks([self.detection(y=1.6)], stamp)[0]
        self.assertTrue(o['lane_stable'])
        self.assertTrue(o['lane_boundary_crossing'])

    def test_highway_occlusion_keeps_measured_width_and_longitudinal_uncertainty(self):
        n = self.labelled_tracker()
        for stamp in (1., 1.1, 1.2):
            n._update_tracks([self.detection()], stamp)
        o = n._update_tracks([], 1.5)[0]
        self.assertTrue(o['predicted'])
        self.assertTrue(o['lane_stable'])
        self.assertAlmostEqual(o['width'], 1.9)
        self.assertAlmostEqual(o['raw_width'], 2.2)
        self.assertAlmostEqual(o['length'], 4.8)
        self.assertEqual(n._update_tracks([], 1.61), [])
        # A prediction moving toward the boundary loses the narrow-box rule.
        crossing = self.labelled_tracker()
        for stamp in (1., 1.1, 1.2):
            o = crossing._update_tracks([self.detection()], stamp)[0]
        crossing.tracks[o['id']].update(vy=4., velocity_reliable=True)
        o = crossing._update_tracks([], 1.5)[0]
        self.assertTrue(o['lane_boundary_crossing'])
        self.assertAlmostEqual(o['width'], 2.2)
        normal = self.labelled_tracker(sector=10)
        for stamp in (1., 1.1, 1.2):
            normal._update_tracks([self.detection()], stamp)
        self.assertAlmostEqual(normal._update_tracks([], 1.5)[0]['width'], 2.2)

    def test_actual_box_growth_and_boundary_intrusion_are_not_smoothed_away(self):
        n = self.labelled_tracker()
        for stamp in (1., 1.1, 1.2):
            n._update_tracks([self.detection()], stamp)
        o = n._update_tracks([self.detection(width=4.8)], 1.3)[0]
        self.assertTrue(o['lane_boundary_crossing'])
        self.assertEqual(o['width'], 4.8)
        self.assertGreater(n._update_tracks([], 1.5)[0]['width'], 4.8)

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

    def test_external_clearance_confirmation_releases_without_second_hold(self):
        guard = PedalGuard();guard.stop(0.)
        guard.release(.1)
        accel, brake, state = guard.command(1., .15, 10.)
        self.assertGreater(accel, 0.)
        self.assertEqual(brake, 0.)
        self.assertEqual(state, 'drive')

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

    def test_roundabout_accepted_gap_skips_duplicate_pedal_hold(self):
        n = GPSWaypointFollower.__new__(GPSWaypointFollower)
        n.longitudinal_control_mode, n.control_rate = 'pid', 20.
        n.last_pid_update, n.current_speed_kmh = None, 0.
        n.speed_pid = PIDController(.08, .002, .01)
        n.ctrl_pub, n.sector11_cruise_brake = MagicMock(), None
        n.lane_change_reason = 'roundabout_gap_accepted'
        n.local_planner = MagicMock(clear_hold_s=.4, restart_accel_rate=.5,
                                    avoidance_brake_limit=.35)
        n.enable_local_avoidance, n.obstacle_stamp = True, 10.
        n.pedal_guard = PedalGuard();n.pedal_guard.stop(0.)
        n.publish_control(0., 30.)
        command = n.ctrl_pub.publish.call_args.args[0]
        self.assertFalse(n.pedal_guard.holding)
        self.assertEqual(command.brake, 0.)

    def cruise_follower(self, sector=11):
        n = GPSWaypointFollower.__new__(GPSWaypointFollower)
        n.longitudinal_control_mode, n.control_rate = 'pid', 20.
        n.last_pid_update, n.current_speed_kmh = None, 60.
        n.speed_pid = PIDController(.08, .002, .01)
        n.ctrl_pub = MagicMock()
        n.current_waypoint_index, n.route_sectors = 0, [sector]
        n.sector11_cruise_brake = CruiseBrake()
        n.local_planner = SimpleNamespace(lane_end_imminent=False)
        n.last_decision = SimpleNamespace(state='RACELINE',
            obstacle_distance_m=math.inf, speed_cap_kmh=math.inf)
        return n

    def test_s11_cruise_brake_ramps_up_and_releases_at_target(self):
        n = self.cruise_follower()
        brakes = []
        for step in range(15):
            with patch('gps_waypoint_follower.time.monotonic', return_value=10.+step*.05):
                n.publish_control(0., 40.)
            cmd = n.ctrl_pub.publish.call_args.args[0]
            brakes.append(cmd.brake)
            self.assertEqual(cmd.accel, 0.)
        self.assertAlmostEqual(brakes[0], .025)
        self.assertAlmostEqual(max(brakes), .25)
        self.assertTrue(all(0. <= b-a <= .025001 for a, b in zip(brakes, brakes[1:])))
        self.assertEqual(n.pedal_reason, 'sector11_cruise_brake')
        # A negative PID integral must not keep braking through the target.
        n.speed_pid.update = MagicMock(return_value=-1.)
        for speed, expected in ((43., .05), (42., .025), (41., 0.), (40., 0.), (39., 0.)):
            n.current_speed_kmh = speed
            n.publish_control(0., 40.)
            self.assertAlmostEqual(n.ctrl_pub.publish.call_args.args[0].brake, expected)

    def test_highway_merge_curvature_cap_is_not_raised_by_sector_floor_or_boost(self):
        n = self.cruise_follower()
        n.local_planner = SimpleNamespace(
            lane_end_imminent=False, highway_active=True, min_speed_sectors=(11,),
            sector_min_speed_kmh=40., lane_change_speed_boost_kmh=10.,
            transition={'state': 'LANE_CHANGING'})
        n.current_speed_kmh = 12.
        n.target_speed_for_index = lambda index: 100.
        n.last_decision.speed_cap_kmh = 13.
        self.assertEqual(n.sector_speed_floor(13.), 13.)
        n.last_decision.speed_cap_kmh = math.inf
        self.assertEqual(n.sector_speed_floor(13.), 40.)

    def test_highway_ordinary_following_brake_ramps_but_emergency_is_immediate(self):
        n = self.cruise_follower()
        n.sector11_cruise_brake = CruiseBrake(gain=.02, max_brake=.22, rise_rate=.3)
        n.local_planner.highway_active = True
        n.local_planner.highway_emergency = False
        n.last_decision.obstacle_distance_m = 20.
        n.last_decision.speed_cap_kmh = 40.
        n.speed_pid.update = MagicMock(return_value=-1.)
        n.publish_control(0., 40.)
        self.assertAlmostEqual(n.ctrl_pub.publish.call_args.args[0].brake, .015)
        n.local_planner.highway_emergency = True
        n.publish_control(0., 10.)
        self.assertEqual(n.ctrl_pub.publish.call_args.args[0].brake, 1.)

    def test_road_ring_evidence_does_not_mark_low_obstacles_or_unknown_ground(self):
        road = np.array([-.39, -.38, -.37])
        self.assertTrue(road_surface_evidence(road, -.38, True))
        # Logged S11 ground ring: retained for 0.4 s after a flat return at
        # -0.441..-0.430 m, just below the calibrated -0.38 m road height.
        self.assertTrue(road_surface_evidence([-.441, -.431, -.430], -.38, True))
        self.assertFalse(road_surface_evidence(road, -.38, False))
        self.assertFalse(road_surface_evidence(road+.1, -.38, True))
        self.assertFalse(road_surface_evidence([-.38, -.3, -.2], -.38, True))
        # A car face can have road-height and high returns at identical XY.
        # XY voxelization must not hide its vertical surface from this gate.
        xy = np.array([[10., -.2], [10., 0.], [10., .2], [10., 0.]])
        box = (10., 0., .4, .2, .6, 0.)
        self.assertFalse(road_surface_in_box(xy, np.array([-.38, -.38, -.38, .8]),
                                              box, -.38, True))
        n = LidarObstacles.__new__(LidarObstacles)
        n.tracks, n.next_id, n.dynamic_speed_threshold_mps = {}, 1, .8
        marked = n._update_tracks([(20., 0., .4, .3, .6, 0., True)], 1.)[0]
        self.assertTrue(marked['road_surface'])
        self.assertTrue(n._update_tracks([], 1.1)[0]['road_surface'])
        # A new elevated detection on the same track must clear the flag.
        elevated = n._update_tracks([(20., 0., .4, .3, .6, 0., False)], 1.2)[0]
        self.assertFalse(elevated['road_surface'])

    def test_s11_hazards_and_other_sectors_bypass_cruise_brake(self):
        for condition in ('other_sector', 'obstacle', 'lane_end', 'finite_planner_cap',
                          'guard_cap', 'external_cap', 'blackout'):
            with self.subTest(condition=condition):
                n = self.cruise_follower()
                n.speed_pid.update = MagicMock(return_value=-1.)
                if condition == 'other_sector': n.route_sectors = [10]
                if condition == 'obstacle': n.last_decision.obstacle_distance_m = 10.
                if condition == 'lane_end': n.local_planner.lane_end_imminent = True
                if condition == 'finite_planner_cap': n.last_decision.speed_cap_kmh = 20.
                if condition == 'guard_cap': n.guard_speed_limit = 20.
                if condition == 'external_cap': n.avoidance_speed_kmh = 20.
                if condition == 'blackout': n.drive_mode = 'BLACKOUT'
                n.publish_control(0., 40.)
                self.assertEqual(n.ctrl_pub.publish.call_args.args[0].brake, 1.)
                self.assertEqual(n.sector11_cruise_brake.brake, 0.)

    def test_s11_full_stop_and_clearance_confirmation_remain_immediate(self):
        n = self.cruise_follower()
        n.publish_control(0., 40.)
        n.publish_full_stop()
        self.assertEqual(n.ctrl_pub.publish.call_args.args[0].brake, 1.)
        self.assertEqual(n.sector11_cruise_brake.brake, 0.)
        n.publish_control(0., 70.)
        self.assertEqual(n.ctrl_pub.publish.call_args.args[0].brake, 1.)
        self.assertEqual(n.pedal_reason, 'clearance_confirmation')

    def test_invalid_cruise_brake_settings_are_rejected(self):
        for kwargs in ({'gain': math.nan}, {'max_brake': 1.1}, {'rise_rate': 0.},
                       {'deadband_kmh': -1.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CruiseBrake(**kwargs)


if __name__ == '__main__':
    unittest.main()
