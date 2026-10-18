"""Live panel edits reach perception/planning and report applied values."""
import copy
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import rospy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from avoidance_runtime import RuntimeSettings, SETTINGS
import avoidance_runtime
from gps_waypoint_follower import GPSWaypointFollower
from lidar_obstacles import LidarObstacles, road_obstacle_mask, object_circles
from local_avoidance_core import LocalAvoidancePlanner, Obstacle
from map_debug import MapDebug, String
from map_tunner_core import load_project, save_project


class LiveAvoidanceTest(unittest.TestCase):
    def setUp(self):
        self.settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.settings_dir.cleanup)
        directory = patch.object(avoidance_runtime, 'PERSISTENCE_DIR', Path(self.settings_dir.name))
        directory.start()
        self.addCleanup(directory.stop)
        master = patch('rospy.get_master')
        proxy = master.start()
        proxy.return_value.getParam.side_effect = lambda key: (1, '', rospy.get_param(key, {}))
        self.addCleanup(master.stop)

    def test_applied_settings_survive_new_runtime_and_invalid_edits_do_not_save(self):
        planner = LocalAvoidancePlanner([(float(i), 0.) for i in range(100)], [11]*100, {})
        params = {}
        def get_param(name, default=None):
            return params.copy() if name == '~' else params.get(name.lstrip('~'), default)
        def set_param(name, value):
            params[name.rsplit('/', 1)[-1].lstrip('~')] = value
        with patch('rospy.get_param', side_effect=get_param), \
                patch('rospy.set_param', side_effect=set_param), \
                patch('rospy.Publisher', return_value=MagicMock()), \
                patch('rospy.get_name', return_value='/gps_waypoint_follower'):
            runtime = RuntimeSettings('gps_waypoint_follower', planner)
            params['lane_change_lateral_accel_mps2'] = 3.5
            runtime.poll()
            saved_file = Path(self.settings_dir.name) / 'avoidance_settings_gps_waypoint_follower.json'
            self.assertEqual(json.loads(saved_file.read_text())['lane_change_lateral_accel_mps2'], 3.5)
            params['lane_change_lateral_accel_mps2'] = -1.
            runtime.next_poll = 0.
            runtime.poll()
            self.assertEqual(json.loads(saved_file.read_text())['lane_change_lateral_accel_mps2'], 3.5)
            params.clear()
            restarted = LocalAvoidancePlanner([(float(i), 0.) for i in range(100)], [11]*100, {})
            RuntimeSettings('gps_waypoint_follower', restarted)
            self.assertEqual(restarted.lane_change_lateral_accel, 3.5)
            self.assertEqual(params['lane_change_lateral_accel_mps2'], 3.5)

    def test_detector_settings_survive_restart(self):
        params = {}
        with patch('rospy.get_param', side_effect=lambda name, default=None:
                   params.copy() if name == '~' else params.get(name.lstrip('~'), default)), \
                patch('rospy.set_param', side_effect=lambda name, value:
                      params.__setitem__(name.rsplit('/', 1)[-1].lstrip('~'), value)), \
                patch('rospy.Publisher', return_value=MagicMock()), \
                patch('rospy.get_name', return_value='/lidar_obstacles'):
            detector = LidarObstacles.__new__(LidarObstacles)
            runtime = RuntimeSettings('lidar_obstacles', detector)
            params['front_range_m'] = 37.
            params['ground_plane_tolerance_m'] = .06
            runtime.poll()
            params.clear()
            restarted = LidarObstacles.__new__(LidarObstacles)
            RuntimeSettings('lidar_obstacles', restarted)
            self.assertEqual((restarted.front_range, restarted.ground_plane_tolerance_m),
                             (37., .06))

    def test_live_height_edits_change_candidates_and_reject_invalid_range(self):
        detector, debug, params = LidarObstacles.__new__(LidarObstacles), self.debug(), {}
        detector.lock, detector.imus = threading.Lock(), []
        detector.body_from_lidar = np.eye(3)
        detector.lidar_x, detector.lidar_y, detector.lidar_z = 1.4, 0., 1.23
        detector.front_range = detector.rear_range = 55.
        detector.side_range, detector.half_width = 12., .946
        detector.front_extent, detector.rear_extent = 3.845, .790
        detector.imu_max_skew_s = .05
        detector.ground_plane_tolerance_m = .04
        detector.obstacle_min_height_m, detector.obstacle_max_height_m = .22, 2.6
        # No IMU: strict absolute height gate, including both boundaries.
        points = np.array([[20., 0., z] for z in (.1, .22, .4, .8, 1., 2.6)])
        raw = points-[1.4, 0., 1.23]
        self.assertEqual(detector.filter_ground(raw, 10.)[2].tolist(),
                         [False, False, True, True, True, False])
        def get_param(name, default=None):
            return params.copy() if name == '~' else params.get(name.lstrip('~'), default)
        def set_param(name, value):
            params[name.rsplit('/', 1)[-1].lstrip('~')] = value
        with patch('rospy.get_param', side_effect=get_param), \
                patch('rospy.set_param', side_effect=set_param), \
                patch('rospy.Publisher', return_value=MagicMock()), \
                patch('rospy.get_name', return_value='/lidar_obstacles'), \
                patch('time.monotonic', return_value=100.):
            runtime = RuntimeSettings('lidar_obstacles', detector)
            runtime.poll()
            debug.on_config(runtime.publisher.publish.call_args.args[0], 'lidar_obstacles')
            for name, value in [('obstacle_min_height_m', .3), ('obstacle_max_height_m', .9)]:
                debug.on_parameter_edit(String(data=json.dumps(dict(name=name, value=value))))
            runtime.next_poll = 0.
            runtime.poll()
            debug.on_config(runtime.publisher.publish.call_args.args[0], 'lidar_obstacles')
            rows, _ = debug.parameter_status(100.)
            for name in ('obstacle_min_height_m', 'obstacle_max_height_m'):
                row = next(r for r in rows if r['name'] == name)
                self.assertEqual(row['value'], params[name])
                self.assertTrue(row['applied'])
            self.assertEqual(detector.filter_ground(raw, 10.)[2].tolist(),
                             [False, False, True, True, False, False])
            params['obstacle_min_height_m'] = .9
            runtime.next_poll = 0.
            runtime.poll()
            self.assertEqual((detector.obstacle_min_height_m, detector.obstacle_max_height_m), (.3, .9))
            report = json.loads(runtime.publisher.publish.call_args.args[0].data)
            self.assertIn('obstacle_min_height_m', report['errors'])
            params['obstacle_min_height_m'] = 0.
            runtime.next_poll = 0.
            runtime.poll()
            self.assertTrue(detector.filter_ground(raw, 10.)[2][0])

    def test_cluster_size_edits_change_filter_and_reject_inverted_limits(self):
        detector = LidarObstacles.__new__(LidarObstacles)
        params = {}
        with patch('rospy.get_param', side_effect=lambda n, d=None: params if n == '~' else d), \
                patch('rospy.set_param'), patch('rospy.Publisher', return_value=MagicMock()), \
                patch('rospy.get_name', return_value='/lidar_obstacles'):
            runtime = RuntimeSettings('lidar_obstacles', detector)
            group = np.array([[0., 0.], [2., 0.], [4., 0.]])
            self.assertTrue(object_circles(group, detector.cluster_min_size_m, detector.cluster_max_size_m))
            params.update(cluster_min_size_m=.5, cluster_max_size_m=3.)
            runtime.poll()
            self.assertFalse(object_circles(group, detector.cluster_min_size_m, detector.cluster_max_size_m))
            params.update(cluster_min_size_m=4.)
            runtime.next_poll = 0.
            runtime.poll()
            self.assertEqual((detector.cluster_min_size_m, detector.cluster_max_size_m), (.5, 3.))
            report = json.loads(runtime.publisher.publish.call_args.args[0].data)
            self.assertIn('cluster_min_size_m', report['errors'])

    def debug(self):
        debug = MapDebug.__new__(MapDebug)
        debug.config_lock = threading.Lock()
        debug.config_snapshots = {}
        debug.pending_parameters = {}
        debug.parameter_error = ''
        debug.saved_lane_corridor = SimpleNamespace(width=3.6)
        return debug

    def test_edit_is_acknowledged_only_after_planner_applies_it(self):
        planner = LocalAvoidancePlanner([(float(i), 0.) for i in range(100)], [11]*100, {})
        params = {}
        debug = self.debug()
        def get_param(name, default=None):
            return params.copy() if name == '~' else params.get(name.lstrip('~'), default)
        def set_param(name, value):
            params[name.rsplit('/', 1)[-1].lstrip('~')] = value
        with patch('rospy.get_param', side_effect=get_param), \
                patch('rospy.set_param', side_effect=set_param), \
                patch('rospy.get_name', return_value='/gps_waypoint_follower'), \
                patch('rospy.Publisher', return_value=MagicMock()), \
                patch('time.monotonic', return_value=100.):
            runtime = RuntimeSettings('gps_waypoint_follower', planner)
            runtime.poll()
            debug.on_config(runtime.publisher.publish.call_args.args[0], 'gps_waypoint_follower')
            debug.on_parameter_edit(String(data=json.dumps(dict(name='obstacle_horizon_m', value=15.))))
            before, _ = debug.parameter_status(100.)
            row = next(r for r in before if r['name'] == 'obstacle_horizon_m')
            self.assertEqual((row['value'], row['pending_value']), (45., 15.))
            self.assertEqual(planner.plan(0, (0., 0.), 10., [Obstacle(25., 0., 1.)]).state, 'TAILING')
            runtime.next_poll = 0.
            runtime.poll()
            debug.on_config(runtime.publisher.publish.call_args.args[0], 'gps_waypoint_follower')
            rows, _ = debug.parameter_status(100.)
        row = next(r for r in rows if r['name'] == 'obstacle_horizon_m')
        self.assertEqual(row['value'], 15.)
        self.assertIsNone(row['pending_value'])
        self.assertTrue(row['applied'])
        self.assertEqual(planner.plan(0, (0., 0.), 10., [Obstacle(25., 0., 1.)]).state, 'RACELINE')
        self.assertEqual(planner.obstacle_horizon_path[-1], (15., 0.))
        # Every exposed planner setting changes its actual attribute, not just rosparam.
        params.update({name: spec[1]*2 for name, spec in SETTINGS.items()
                       if spec[0] == 'gps_waypoint_follower'})
        runtime.next_poll = 0.
        with patch('rospy.get_param', return_value=params), patch('rospy.get_name', return_value='/gps_waypoint_follower'):
            runtime.poll()
        for name, spec in runtime.settings.items():
            self.assertEqual(getattr(planner, spec[2]), params[name])

    def test_detector_roi_changes_and_side_range_remains_user_value(self):
        detector = LidarObstacles.__new__(LidarObstacles)
        detector.front_range = detector.rear_range = 55.
        detector.side_range = 12.
        detector.dynamic_speed_threshold_mps = .8
        params = {}
        with patch('rospy.get_param', side_effect=lambda n, d=None: params if n == '~' else d), \
                patch('rospy.set_param'), patch('rospy.Publisher', return_value=MagicMock()), \
                patch('rospy.get_name', return_value='/lidar_obstacles'):
            runtime = RuntimeSettings('lidar_obstacles', detector)
            params.update(front_range_m=25., rear_range_m=20., side_range_m=5.)
            runtime.poll()
        points = np.array([[30., 0., 1.], [-30., 0., 1.], [15., 6., 1.], [15., 3., 1.]])
        self.assertEqual(road_obstacle_mask(points, detector.front_range, detector.rear_range,
                                          detector.side_range).tolist(), [False, False, False, True])
        detector.lock, detector.tracks = threading.Lock(), {}
        geometry = dict(frame_id='map', raceline_width=4., loop=False,
                        lanes=[dict(sector=1, raceline=1, points=[[0., 0.], [50., 0.]])])
        with patch('rospy.set_param'), patch('rospy.loginfo'):
            detector.on_lane_geometry(String(data=json.dumps(geometry)))
        self.assertEqual(detector.side_range, 5.)

    def test_invalid_edit_does_not_mutate_live_value_and_stale_ack_is_not_applied(self):
        debug = self.debug()
        for name, value in [('front_range_m', 0), ('avoidance_margin_m', -1),
                            ('obstacle_horizon_m', math.inf), ('rear_range_m', 'abc'),
                            ('pass_speed_kmh', 0), ('raceline_width', 4)]:
            with self.subTest(name=name, value=value), patch('rospy.set_param') as set_param:
                debug.on_parameter_edit(String(data=json.dumps(dict(name=name, value=value))))
                set_param.assert_not_called()
                self.assertTrue(debug.parameter_error)
        debug.config_snapshots['lidar_obstacles'] = (
            dict(node='/lidar_obstacles', values=dict(front_range_m=55.)), 100.)
        rows, _ = debug.parameter_status(101.1)
        self.assertFalse(next(r for r in rows if r['name'] == 'front_range_m')['applied'])
        self.assertFalse(next(r for r in rows if r['name'] == 'raceline_width')['editable'])

    def test_live_speed_threshold_edit_reclassifies_existing_track(self):
        detector = LidarObstacles.__new__(LidarObstacles)
        detector.front_range = detector.rear_range = 55.
        detector.side_range, detector.dynamic_speed_threshold_mps = 12., .8
        detector.tracks = {1: dict(x=20., y=0., radius=.8, vx=1., vy=0., hits=3, stamp=99.)}
        detector.next_id = 2
        debug, params = self.debug(), {}
        def get_param(name, default=None):
            return params.copy() if name == '~' else params.get(name.lstrip('~'), default)
        def set_param(name, value):
            params[name.rsplit('/', 1)[-1].lstrip('~')] = value
        with patch('rospy.get_param', side_effect=get_param), \
                patch('rospy.set_param', side_effect=set_param), \
                patch('rospy.get_name', return_value='/lidar_obstacles'), \
                patch('rospy.Publisher', return_value=MagicMock()), \
                patch('time.monotonic', return_value=100.):
            runtime = RuntimeSettings('lidar_obstacles', detector)
            runtime.poll()
            debug.on_config(runtime.publisher.publish.call_args.args[0], 'lidar_obstacles')
            before = detector._update_tracks([(20.1, 0., .8)], 99.1)[0]
            self.assertEqual(before['kind'], 'dynamic')
            debug.on_parameter_edit(String(data=json.dumps(dict(
                name='dynamic_speed_threshold_mps', value=1.5))))
            runtime.next_poll = 0.
            runtime.poll()
            debug.on_config(runtime.publisher.publish.call_args.args[0], 'lidar_obstacles')
            rows, _ = debug.parameter_status(100.)
            after = detector._update_tracks([(20.2, 0., .8)], 99.2)[0]
        self.assertEqual(after['id'], before['id'])
        self.assertAlmostEqual(after['vx'], before['vx'])
        self.assertEqual(after['kind'], 'static')
        row = next(r for r in rows if r['name'] == 'dynamic_speed_threshold_mps')
        self.assertEqual(row['value'], 1.5)
        self.assertTrue(row['applied'])
        self.assertIsNone(row['pending_value'])

    def test_speed_threshold_boundary_and_unknown_tracks(self):
        for vx, vy, hits, expected in ((1.5, 0., 3, 'static'),
                                       (1.51, 0., 3, 'dynamic'),
                                       (1.2, 1., 3, 'dynamic'),
                                       (3., 0., 1, 'unknown')):
            with self.subTest(vx=vx, vy=vy, hits=hits):
                detector = LidarObstacles.__new__(LidarObstacles)
                detector.dynamic_speed_threshold_mps = 1.5
                detector.tracks = {1: dict(x=20., y=0., radius=.8,
                                          vx=vx, vy=vy, hits=hits, stamp=99.)}
                detector.next_id = 2
                actual = detector._update_tracks([(20.+vx*.5, vy*.5, .8)], 99.5)[0]
                self.assertEqual(actual['kind'], expected)

    def test_invalid_rosparam_preserves_last_good_planner_value(self):
        planner = LocalAvoidancePlanner([(0., 0.), (100., 0.)], [11, 11], {})
        with patch('rospy.get_param', side_effect=lambda n, d=None: d), \
                patch('rospy.set_param'), patch('rospy.Publisher', return_value=MagicMock()):
            runtime = RuntimeSettings('gps_waypoint_follower', planner)
        with patch('rospy.get_param', return_value={'avoidance_decel_mps2': 0}), \
                patch('rospy.get_name', return_value='/gps_waypoint_follower'):
            runtime.poll()
        self.assertEqual(planner.decel, 3.)
        report = json.loads(runtime.publisher.publish.call_args.args[0].data)
        self.assertIn('avoidance_decel_mps2', report['errors'])

    def test_tuner_saved_width_updates_follower_geometry_without_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'project.json'
            project = load_project(ROOT / 'config/map_tunner_project.json')
            save_project(path, project)
            follower = GPSWaypointFollower.__new__(GPSWaypointFollower)
            follower.waypoint_file = str(path)
            follower.waypoint_format = 'local_xy'
            follower.raceline_config = str(ROOT / 'config/raceline_layout.yaml')
            follower.raceline_width_override = 9.  # Must not override saved tuner width.
            follower.lane_geometry_pub = MagicMock()
            with patch('rospy.set_param'):
                follower.load_waypoints()
            original = copy.deepcopy(follower.driving_lane_geometry['lanes'])
            project['generated_racelines']['settings']['raceline_width'] = 4.2
            save_project(path, project)
            with patch('rospy.set_param'), patch('rospy.loginfo'):
                follower.sync_tuner_width()
            self.assertEqual(follower.raceline_width, 4.2)
            self.assertEqual(follower.lane_corridor.width, 4.2)
            published = json.loads(follower.lane_geometry_pub.publish.call_args.args[0].data)
            self.assertEqual(published['raceline_width'], 4.2)
            self.assertEqual(published['lanes'], original)

    def test_horizon_marker_follows_curved_path_and_expires(self):
        debug = self.debug()
        debug.horizon_pub = MagicMock()
        data = dict(frame_id='map', stamp=100., horizon_m=5., half_width_m=1.296,
                    points=[[0., 0.], [3., 0.], [3., 2.]])
        with patch('time.monotonic', return_value=200.), \
                patch('rospy.Time.now', return_value=rospy.Time(100)):
            debug.on_horizon(String(data=json.dumps(data)))
            debug.publish_horizon(200.)
            markers = debug.horizon_pub.publish.call_args.args[0].markers
            label = next(m for m in markers if m.ns == 'obstacle_horizon_label')
            self.assertEqual((label.pose.position.x, label.pose.position.y), (3., 2.))
            self.assertEqual(label.text, 'obstacle_horizon 5.0 m')
            self.assertTrue(all(m.lifetime.to_sec() == .5 for m in markers))
            debug.horizon_pub.reset_mock()
            debug.publish_horizon(201.)
            debug.horizon_pub.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
