"""Motion regressions for changing LiDAR faces and real moving traffic."""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from obstacle_motion import box_bounds, update_motion
from lidar_obstacles import LidarObstacles, object_box
from lidar_obstacles_local import LidarObstacles as LocalLidarObstacles
from lidar_obstacles_local import object_circles as local_object_circles
from local_avoidance_core_local import LocalAvoidancePlanner, Obstacle


def detector(cls):
    node = cls.__new__(cls)
    node.tracks, node.next_id = {}, 1
    node.dynamic_speed_threshold_mps = .8
    node.track_confirm_time_s = .2
    return node


def translated_bounds(x=0., y=0.):
    return (x-2., y-1., x+2., y+1.)


def detection(cls, bounds):
    xmin, ymin, xmax, ymax = bounds
    x, y = (xmin+xmax)/2., (ymin+ymax)/2.
    length, width = xmax-xmin, ymax-ymin
    radius = math.hypot(length, width)/2.
    if cls is LidarObstacles:
        return (x, y, radius, length, width, 0.)
    return (x, y, radius, tuple(bounds))


class MotionEstimatorTests(unittest.TestCase):
    def replay(self, samples, threshold=.8):
        previous, results = {}, []
        for stamp, bounds in samples:
            motion = update_motion(previous, stamp, bounds, threshold)
            results.append(motion)
            previous = dict(motion, stamp=stamp)
        return results

    def test_rotated_box_bounds_enclose_physical_footprint(self):
        np.testing.assert_allclose(box_bounds(10., 20., 4., 2., math.pi/2.),
                                   (9., 18., 11., 22.), atol=1e-10)
        half_extent = 3./math.sqrt(2.)
        np.testing.assert_allclose(box_bounds(0., 0., 4., 2., math.pi/4.),
                                   (-half_extent, -half_extent,
                                    half_extent, half_extent), atol=1e-10)

    def test_growing_visible_face_does_not_move_stationary_object(self):
        # The rear face stays at x=18. As the car passes, more of the
        # object's side appears. Its observed centre moves 1.6 m in .8 s.
        samples = [(i*.1, (18., -1., 18.8+i*.4, 1.)) for i in range(9)]
        results = self.replay(samples)
        self.assertNotIn('dynamic', [r['motion_kind'] for r in results])
        self.assertEqual(results[-1]['motion_kind'], 'static')
        self.assertLess(math.hypot(results[-1]['vx'], results[-1]['vy']), .3)

    def test_true_translation_is_dynamic_by_fourth_scan(self):
        for vx, vy in ((2.5, 0.), (0., 1.5), (30., 0.)):
            with self.subTest(vx=vx, vy=vy):
                results = self.replay([(i*.1, translated_bounds(vx*i*.1, vy*i*.1))
                                       for i in range(4)])
                self.assertEqual(results[-1]['motion_kind'], 'dynamic')
                self.assertTrue(results[-1]['velocity_reliable'])
                self.assertAlmostEqual(results[-1]['vx'], vx, delta=.2)
                self.assertAlmostEqual(results[-1]['vy'], vy, delta=.2)
                # Candidate motion cannot enter prediction one scan before
                # the same evidence is sufficient to classify the object.
                self.assertEqual((results[-2]['vx'], results[-2]['vy']), (0., 0.))
                self.assertGreater(math.hypot(results[-2]['candidate_vx'],
                                             results[-2]['candidate_vy']), .8)

    def test_one_scan_position_jump_is_not_dynamic_evidence(self):
        samples = [(i*.1, translated_bounds(.65 if i == 6 else 0., 0.))
                   for i in range(16)]
        results = self.replay(samples)
        self.assertNotIn('dynamic', [r['motion_kind'] for r in results])
        self.assertEqual(results[-1]['motion_kind'], 'static')
        self.assertTrue(all(r['vx'] == r['vy'] == 0. for r in results))

    def test_sustained_slow_motion_is_predicted_without_promoting_short_drift(self):
        for speed in (.36, .48, -.36, -.48):
            with self.subTest(speed=speed):
                results = self.replay([(i*.1, translated_bounds(0., speed*i*.1))
                                       for i in range(25)])
                self.assertEqual(results[-1]['motion_kind'], 'static')
                self.assertAlmostEqual(results[-1]['candidate_vy'], speed)
                confirmed = [i for i, value in enumerate(results) if value['prediction_confirmed']]
                self.assertTrue(confirmed)
                first = confirmed[0]
                # Broad footprints initially judged stationary require a
                # longer consistent run; sustained real motion still wins.
                self.assertEqual(results[first]['prediction_consistent_scans'], 6)
                self.assertEqual(results[first-1]['vy'], 0.)
                for value in results[first:]:
                    self.assertTrue(value['prediction_confirmed'])
                    self.assertAlmostEqual(value['vy'], speed)
                    self.assertLess(value['prediction_stop_threshold'], abs(speed))

    def test_back_and_forth_measurement_jumps_do_not_confirm_translation(self):
        results = self.replay([(i*.1, translated_bounds(.35 if i % 2 else -.35))
                               for i in range(30)])
        self.assertFalse(any(value['prediction_confirmed'] for value in results))
        self.assertTrue(all(value['vx'] == value['vy'] == 0. for value in results))

    def test_stopped_object_can_become_static_then_move_again(self):
        samples = []
        for i in range(36):
            x = 2.5*i*.1 if i <= 9 else 2.25
            if i >= 25:
                x += 2.5*(i-24)*.1
            samples.append((i*.1, translated_bounds(x, 0.)))
        results = self.replay(samples)
        self.assertEqual(results[8]['motion_kind'], 'dynamic')
        self.assertEqual(results[22]['motion_kind'], 'static')
        self.assertEqual(results[-1]['motion_kind'], 'dynamic')
        self.assertTrue(results[-1]['velocity_reliable'])
        self.assertFalse(results[22]['prediction_confirmed'])
        self.assertEqual((results[22]['vx'], results[22]['vy']), (0., 0.))
        self.assertTrue(results[-1]['prediction_confirmed'])
        self.assertAlmostEqual(results[-1]['vx'], 2.5, delta=.2)


class DetectorMotionTests(unittest.TestCase):
    def test_recorded_lane_change_uses_confirmed_motion_without_relaxing_collision_geometry(self):
        fixture = json.loads((Path(__file__).parent / 'data' /
                              'static_lane_change_20261021.json').read_text())
        node, observed = detector(LocalLidarObstacles), []
        for frame in fixture['frames']:
            output = node._update_tracks(frame['detections'], frame['stamp'])
            measured = [item for item in output if not item.get('predicted', False)]
            self.assertEqual(len(measured), len(frame['detections']))
            for item, raw in zip(measured, frame['detections']):
                self.assertEqual((item['x'], item['y'], item['radius']), tuple(raw[:3]))
            observed.append(output)
        planner = LocalAvoidancePlanner([(0., 0.), (100., 0.)], [1, 1], {},
                                        **fixture['planner_parameters'])
        self.assertEqual(len(fixture['cases']), 9)
        for case in fixture['cases']:
            original = [Obstacle(**item) for item in case['original_obstacles']]
            revised = [Obstacle(*(item[key] for key in ('x', 'y', 'radius', 'vx', 'vy')),
                                kind=item['kind']) for item in observed[case['frame_index']]]
            self.assertFalse(planner._collision_free(case['path'], original,
                                                     case['speed_mps'], case['horizon_s']))
            self.assertTrue(planner._collision_free(case['path'], revised,
                                                    case['speed_mps'], case['horizon_s']))

    def test_recorded_static_obstacle_remains_static_while_passing(self):
        fixture_path = Path(__file__).parent / 'data' / 'static_obstacle_20261021.json'
        with fixture_path.open() as stream:
            fixture = json.load(stream)
        target = fixture['metadata']['stationary_target']
        for name, cls in (('main', LidarObstacles), ('local', LocalLidarObstacles)):
            with self.subTest(detector=name):
                node, kinds = detector(cls), []
                for stamp, detections in fixture[name]:
                    output = node._update_tracks(detections, stamp)
                    measured = [t for t in output if not t.get('predicted', False)]
                    self.assertEqual(len(measured), len(detections))
                    for actual, supplied in zip(measured, detections):
                        # Stabilizing motion classification must preserve
                        # every measured obstacle's collision geometry.
                        for field, value in zip(('x', 'y', 'radius'), supplied[:3]):
                            self.assertEqual(actual[field], value)
                        if name == 'main':
                            for field, value in zip(('length', 'width', 'yaw'), supplied[3:6]):
                                self.assertEqual(actual[field], value)
                        if (stamp >= target['after_s'] and
                                math.hypot(actual['x']-target['center'][0],
                                           actual['y']-target['center'][1]) < target['radius_m']):
                            kinds.append(actual['kind'])
                # The original captures reported 16/20 dynamic samples.
                # Requiring confirmed static results also rejects fixes
                # that merely hide every uncertain result as unknown.
                self.assertGreaterEqual(len(kinds), 80)
                self.assertGreaterEqual(kinds.count('static'), 40)
                self.assertEqual(kinds.count('dynamic'), 0)

    def test_both_detectors_keep_static_face_growth_static(self):
        for cls in (LidarObstacles, LocalLidarObstacles):
            with self.subTest(detector=cls.__module__):
                node, observed = detector(cls), []
                for i in range(9):
                    bounds = (18., -1., 18.8+i*.4, 1.)
                    tracks = node._update_tracks([detection(cls, bounds)], 10.+i*.1)
                    observed.append(next(t for t in tracks if not t.get('predicted', False)))
                self.assertEqual(len({t['id'] for t in observed}), 1)
                self.assertNotIn('dynamic', [t['kind'] for t in observed])
                self.assertEqual(observed[-1]['kind'], 'static')
                self.assertTrue(all(t['vx'] == t['vy'] == 0. for t in observed))
                # Motion stabilization must not shrink the occupied space.
                self.assertAlmostEqual(observed[-1]['x'], 20.)
                self.assertAlmostEqual(observed[-1]['radius'], math.hypot(4., 2.)/2.)
                if cls is LidarObstacles:
                    self.assertEqual((observed[-1]['length'], observed[-1]['width']), (4., 2.))

    def test_both_detectors_track_crossing_and_fast_rear_traffic(self):
        for cls in (LidarObstacles, LocalLidarObstacles):
            for vx, vy in ((2.5, 0.), (0., 1.5), (30., 0.)):
                with self.subTest(detector=cls.__module__, vx=vx, vy=vy):
                    node, observed = detector(cls), []
                    for i in range(4):
                        bounds = translated_bounds(-40.+vx*i*.1, 7.+vy*i*.1)
                        tracks = node._update_tracks([detection(cls, bounds)], 10.+i*.1)
                        observed.append(next(t for t in tracks if not t.get('predicted', False)))
                    self.assertEqual(len({t['id'] for t in observed}), 1)
                    self.assertEqual(observed[-1]['kind'], 'dynamic')
                    self.assertTrue(observed[-1]['velocity_reliable'])
                    self.assertAlmostEqual(observed[-1]['vx'], vx, delta=.2)
                    self.assertAlmostEqual(observed[-1]['vy'], vy, delta=.2)
                    # A real car must continue to occupy its predicted
                    # position through a brief missing LiDAR scan.
                    missed = node._update_tracks([], 10.4)[0]
                    self.assertEqual(missed['id'], observed[-1]['id'])
                    self.assertTrue(missed['velocity_reliable'])
                    self.assertAlmostEqual(missed['x'], observed[-1]['x']+.1*vx, delta=.03)
                    self.assertAlmostEqual(missed['y'], observed[-1]['y']+.1*vy, delta=.03)

    def test_live_threshold_change_keeps_track_and_confirmed_velocity(self):
        for cls in (LidarObstacles, LocalLidarObstacles):
            with self.subTest(detector=cls.__module__):
                node = detector(cls)
                for i in range(9):
                    before = node._update_tracks(
                        [detection(cls, translated_bounds(1.5*i*.1))], 10.+i*.1)[0]
                self.assertEqual(before['kind'], 'dynamic')
                node.dynamic_speed_threshold_mps = 2.
                after = node._update_tracks(
                    [detection(cls, translated_bounds(1.5*.9))], 10.9)[0]
                self.assertEqual(after['id'], before['id'])
                self.assertEqual(after['kind'], 'static')
                self.assertAlmostEqual(after['candidate_vx'], before['vx'], delta=.1)
                self.assertAlmostEqual(after['vx'], before['vx'], delta=.1)
                self.assertTrue(after['prediction_confirmed'])
                missed = node._update_tracks([], 11.)[0]
                self.assertTrue(missed['prediction_confirmed'])
                self.assertAlmostEqual(missed['x'], after['x']+.15, delta=.03)

    def test_both_detectors_keep_slow_forward_and_reverse_motion_through_occlusion(self):
        for cls in (LidarObstacles, LocalLidarObstacles):
            for speed in (.36, .48, -.36, -.48):
                with self.subTest(detector=cls.__module__, speed=speed):
                    node = detector(cls)
                    observed = []
                    for i in range(25):
                        observed.append(node._update_tracks(
                            [detection(cls, translated_bounds(speed*i*.1))], 10.+i*.1)[0])
                    self.assertEqual(len({item['id'] for item in observed}), 1)
                    self.assertEqual(observed[-1]['kind'], 'static')
                    self.assertTrue(observed[-1]['prediction_confirmed'])
                    self.assertAlmostEqual(observed[-1]['vx'], speed)
                    missed = node._update_tracks([], 12.5)[0]
                    self.assertTrue(missed['prediction_confirmed'])
                    self.assertAlmostEqual(missed['x'], observed[-1]['x']+.1*speed)

    def test_unreliable_jump_does_not_extrapolate_occluded_obstacle(self):
        for cls in (LidarObstacles, LocalLidarObstacles):
            with self.subTest(detector=cls.__module__):
                node = detector(cls)
                for i in range(8):
                    x = .65 if i == 7 else 0.
                    seen = node._update_tracks(
                        [detection(cls, translated_bounds(x))], 10.+i*.1)[0]
                self.assertNotEqual(seen['kind'], 'dynamic')
                missed = node._update_tracks([], 10.9)[0]
                self.assertAlmostEqual(missed['x'], seen['x'], delta=.03)
                self.assertAlmostEqual(missed['y'], seen['y'], delta=.03)

    def test_unconfirmed_object_still_blocks_occupied_path(self):
        node = detector(LocalLidarObstacles)
        tracked = node._update_tracks([detection(LocalLidarObstacles,
                                                translated_bounds(6., 0.))], 10.)[0]
        self.assertEqual(tracked['kind'], 'unknown')
        self.assertFalse(tracked['prediction_confirmed'])
        obstacle = Obstacle(*(tracked[key] for key in ('x', 'y', 'radius', 'vx', 'vy')),
                            kind=tracked['kind'])
        planner = LocalAvoidancePlanner([(0., 0.), (30., 0.)], [1, 1], {})
        self.assertFalse(planner._collision_free([(0., 0.), (10., 0.)], [obstacle], 3.))

    def test_local_ego_rotation_preserves_stationary_world_bounds(self):
        # An asymmetric L-shaped visible face shifts its body-axis box
        # centre as ego rotates, although every physical return is static.
        world = np.array([[18., -1.], [18., 1.], [22., 1.],
                          [18., 0.], [20., 1.]])
        node, observed = detector(LocalLidarObstacles), []
        for i in range(9):
            yaw = -.4+i*.1
            pose = (i*.5, .1*i, yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            body = (world-np.asarray(pose[:2])) @ np.array([[c, -s], [s, c]])
            detections = local_object_circles(body, pose=pose)
            self.assertEqual(len(detections), 1)
            np.testing.assert_allclose(detections[0][3], (18., -1., 22., 1.), atol=1e-10)
            tracks = node._update_tracks(detections, 10.+i*.1)
            observed.append(next(t for t in tracks if not t.get('predicted', False)))
        self.assertEqual(len({t['id'] for t in observed}), 1)
        self.assertNotIn('dynamic', [t['kind'] for t in observed])
        self.assertEqual(observed[-1]['kind'], 'static')

    def test_main_detector_observed_box_remains_unchanged(self):
        group = np.array([[18., -1.], [18., 1.], [22., 1.]])
        measured = object_box(group, (1., 0.))
        node = detector(LidarObstacles)
        for i in range(5):
            tracked = node._update_tracks([measured], 10.+i*.1)[0]
        for name, expected in zip(('x', 'y', 'radius', 'length', 'width', 'yaw'), measured):
            self.assertAlmostEqual(tracked[name], expected)
        self.assertEqual(tracked['kind'], 'static')


if __name__ == '__main__':
    unittest.main()
