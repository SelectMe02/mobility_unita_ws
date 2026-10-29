#!/usr/bin/env python3
"""Verify saved geometry reaches RViz, including appended points and live saves."""
import copy
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import map_debug
from perception_debug import PerceptionDebug, xyz_cloud
from map_tunner_core import load_project, save_project, ProjectEditor


class MapDebugProjectTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'project.json'
        self.saved = load_project(ROOT / 'config/map_tunner_project.json')
        # A saved R1 coordinate edit must not trigger regeneration of R3.
        self.saved['sectors'][10]['racelines']['1'][20]['x'] += 1.0
        save_project(self.path, self.saved)
        with patch.object(map_debug, 'DEBUG_PROJECT_FILE', self.path), \
                patch.object(map_debug.rospy, 'get_param', side_effect=lambda name, default=None:
                          str(self.path) if name == '~project_file' else default), \
                patch.object(map_debug.rospy, 'Publisher', return_value=MagicMock()), \
                patch.object(map_debug.rospy, 'Subscriber'), \
                patch.object(map_debug.rospy, 'Timer'), \
                patch.object(map_debug.rospy, 'loginfo'), \
                patch.object(map_debug, 'AerialMap'), \
                patch.object(map_debug.MapDebug, 'publish_static'):
            self.debug = map_debug.MapDebug()
        self.addCleanup(self.debug.mesh_directory.cleanup)
        self.debug.marker_pub = MagicMock()

    def publish(self):
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(0)):
            self.debug.publish_paths()
        return self.debug.marker_pub.publish.call_args.args[0].markers

    def test_active_launch_map_switches_both_directions_without_debug_restart(self):
        import os
        extreme = Path(self.folder.name) / 'extreme.json'
        changed = copy.deepcopy(self.saved)
        changed['sectors'][0]['racelines']['1'][0]['speed_kmh'] = 59
        save_project(extreme, changed)
        stamp = self.path.stat().st_mtime_ns
        os.utime(extreme, ns=(stamp, stamp))
        for path, expected in ((extreme, 59), (self.path, self.saved['sectors'][0]['racelines']['1'][0]['speed_kmh'])):
            self.debug.next_project_check = 0.
            with patch.object(map_debug.rospy, 'get_param', return_value=str(path)), \
                    patch.object(self.debug, 'publish_paths') as publish, \
                    patch.object(self.debug, 'log_project_counts'):
                self.debug.reload_project_if_changed()
            self.assertEqual(self.debug.project_file, path.resolve())
            self.assertEqual(self.debug.project['sectors'][0]['racelines']['1'][0]['speed_kmh'], expected)
            publish.assert_called_once()

    def test_startup_uses_exact_saved_json_and_publishes_entire_raceline3(self):
        self.assertEqual(self.debug.project, self.saved)
        markers = self.publish()
        line = next(m for m in markers if m.ns == 'sector_paths' and m.id == 113)
        dots = next(m for m in markers if m.ns == 'sector_points' and m.id == 113)
        expected = [(p['x'], p['y']) for p in self.saved['sectors'][10]['racelines']['3']]
        self.assertEqual([(p.x, p.y) for p in line.points], expected)
        self.assertEqual([(p.x, p.y) for p in dots.points], expected)
        r1 = next(m for m in markers if m.ns == 'sector_paths' and m.id == 111)
        self.assertGreater(line.points[0].z, r1.points[0].z)
        self.assertFalse(any(m.ns == 'cp11_finish_line' for m in markers))

    def test_save_reloads_appended_points_and_republishes(self):
        editor = ProjectEditor(self.saved)
        editor.append_continuous_points(11, 3, (70, -140), (70, -200), 10)
        save_project(self.path, editor.project)
        self.debug.project_stamp = -1
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(0)), \
                patch.object(map_debug.rospy, 'loginfo'):
            self.debug.reload_project_if_changed()
        self.assertEqual(self.debug.project, editor.project)
        markers = self.debug.marker_pub.publish.call_args.args[0].markers
        line = next(m for m in markers if m.ns == 'sector_paths' and m.id == 113)
        self.assertEqual(len(line.points), len(editor.lane(11, 3)))
        self.assertEqual((line.points[-1].x, line.points[-1].y), (70, -200))

    def test_invalid_save_keeps_previous_display_and_retries(self):
        self.path.write_text('{')
        self.debug.project_stamp = -1
        before = copy.deepcopy(self.debug.project)
        with patch.object(map_debug.rospy, 'logwarn_throttle'):
            self.debug.reload_project_if_changed()
        self.assertEqual(self.debug.project, before)
        self.debug.marker_pub.publish.assert_not_called()
        save_project(self.path, self.saved)
        self.debug.next_project_check = 0
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(0)), \
                patch.object(map_debug.rospy, 'loginfo'):
            self.debug.reload_project_if_changed()
        self.debug.marker_pub.publish.assert_called_once()

    def test_live_geometry_never_overrides_saved_paths_even_after_tuner_save(self):
        geometry = map_debug.project_lane_geometry(self.saved, 4.)
        for lane in geometry['lanes']:
            for point in lane['points']:
                point[0] += 100.
        self.debug.on_lane_geometry(map_debug.String(data=json.dumps(geometry)))
        self.debug.marker_pub.publish.assert_not_called()
        markers = self.publish()
        line = next(m for m in markers if m.ns == 'sector_paths' and m.id == 113)
        self.assertEqual(line.points[0].x, self.saved['sectors'][10]['racelines']['3'][0]['x'])
        self.test_save_reloads_appended_points_and_republishes()

    def test_debug_honors_project_parameter_and_keeps_original_default(self):
        self.assertEqual(self.debug.project_file, self.path.resolve())
        # Default stays on the original project until an active launch is detected.
        self.assertEqual(str(map_debug.DEBUG_PROJECT_FILE),
                         '/home/unita/unita_ws/src/unita_waypoint/config/map_tunner_project.json')

    def status(self, state):
        self.debug.tracking = (state, 100.)
        self.debug.valid = True
        self.debug.pose = (70., 100., 0.)
        self.debug.pose_time = 100.
        with patch.object(map_debug.time, 'monotonic', return_value=100.1), \
                patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(100)), \
                patch.object(self.debug, 'reload_project_if_changed'):
            self.debug.refresh(None)
        return json.loads(self.debug.status_pub.publish.call_args.args[0].data)

    def test_ground_status_reaches_panel_and_expires_in_both_clocks(self):
        state = dict(stamp=100., frame_id='map', mode='ground_filtered',
                     pitch_deg=-4.2, roll_deg=.5, imu_skew_s=.02,
                     residual_p95_m=.015, planes=8, removed_points=900, obstacle_points=35)
        with patch.object(map_debug.time, 'monotonic', return_value=100.):
            self.debug.on_ground_status(map_debug.String(data=json.dumps(state)))
        self.assertEqual(self.status(dict(valid=True, sector=1, raceline=1))['ground_filter'], state)
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(100.1)):
            self.assertIsNone(self.debug.ground_status(100.6))
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(101.)):
            self.assertIsNone(self.debug.ground_status(100.1))
        state['pitch_deg'] = float('nan')
        self.debug.on_ground_status(map_debug.String(data=json.dumps(state)))
        self.assertEqual(self.debug.ground_snapshot[0]['pitch_deg'], -4.2)

    def test_follower_priority_and_actual_lane_reach_panel_without_inference(self):
        state = dict(valid=True, sector=11, raceline=3, local_lane=3,
                     current_raceline=3, preferred_raceline=3,
                     preferred_raceline_reason='priority_interval', waypoint_index=2530,
                     finish_line_y=-217.855, finish_line_rear_y=-217.1,
                     finish_line_passed=False)
        status = self.status(state)
        for field, value in state.items():
            if field not in ('valid', 'finish_line_y', 'finish_line_rear_y', 'finish_line_passed'):
                self.assertEqual(status[field], value)
        self.assertEqual(status['raceline_width'], self.debug.saved_lane_corridor.width)
        state.update(local_lane=None, raceline=1, preferred_raceline=1)
        status = self.status(state)
        self.assertEqual(status['current_raceline'], 3)
        self.assertEqual(status['preferred_raceline'], 1)

    def test_invalid_tracking_never_displays_old_priority(self):
        status = self.status(dict(valid=False, sector=11, raceline=3,
                                 current_raceline=3, preferred_raceline=1))
        self.assertIsNone(status['preferred_raceline'])
        self.assertIsNone(status['current_raceline'])
        self.assertIsNone(status['raceline'])

    def test_debug_vehicle_uses_controller_pose_from_same_message_as_priority(self):
        state = dict(valid=True, sector=11, raceline=3, local_lane=3,
                     current_raceline=3, preferred_raceline=3,
                     frame_id='map', x=73., y=89.3, yaw=-1.5)
        status = self.status(state)
        self.assertEqual((status['x'], status['y']), (73., 89.3))
        self.assertEqual(status['preferred_raceline'], 3)

    def test_guard_selects_actual_controller_and_hold_clears_source(self):
        self.assertEqual(self.debug.select_source(100.), 'raceline')
        self.debug.guard_snapshot = (dict(state='passthrough'), 100.)
        self.assertEqual(self.debug.select_source(100.1), 'raceline')
        self.debug.guard_snapshot = (dict(state='clearing'), 100.)
        self.assertEqual(self.debug.select_source(100.1), 'leo')
        self.debug.guard_snapshot = (dict(state='hold'), 100.)
        self.assertIsNone(self.debug.select_source(100.1))
        self.debug.guard_snapshot = (dict(state='passthrough'), 100.)
        self.assertIsNone(self.debug.select_source(100.6))

    def test_stopped_decision_stays_visible_without_usable_tracking(self):
        status = self.status(dict(valid=False, sector=11, raceline=3,
            planner_mode='highway', planner_state='BLOCKED', planner_reason='no_safe_gap'))
        self.assertEqual(status['decision']['planner_reason'], 'no_safe_gap')
        self.assertIsNone(status['current_raceline'])

    def test_route_collision_and_tunnel_perception_follow_executed_branch(self):
        self.debug.active_source = 'leo'
        state = dict(valid=True, sector=4, raceline=1,
                     planner_mode='route_and_local_avoidance', local_path_source='route_avoidance')
        self.debug.source_snapshots['leo']['tracking'] = (state, 100.)
        with patch.object(map_debug.time, 'monotonic', return_value=100.1):
            self.assertEqual(self.debug.perception_source(), 'route')
            self.debug.on_perception_view(map_debug.String(data='collision'))
            self.assertEqual(self.debug.perception_source(), 'leo')
            state['planner_mode'] = 'external_blackout_target'
            self.assertEqual(self.debug.perception_source(), 'ftg')

    def test_actual_path_elevation_and_expiry_and_stop_clear(self):
        self.debug.local_path_pub = MagicMock()
        self.debug.map_local_path_pub = MagicMock()
        self.debug.local_path_marker_pub = MagicMock()
        self.debug.map_path_marker_pub = MagicMock()
        path = map_debug.NavPath()
        path.header = map_debug.Header(stamp=map_debug.rospy.Time(100), frame_id='map')
        pose = map_debug.PoseStamped()
        pose.pose.position.x, pose.pose.position.y = 2., 3.
        next_pose = copy.deepcopy(pose)
        next_pose.pose.position.x = 4.
        path.poses = [pose, next_pose]
        self.debug.tracking = (dict(valid=True, sector=11, raceline=1), 100.)
        with patch.object(map_debug.time, 'monotonic', return_value=100.1), \
                patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(100.1)):
            self.debug.on_local_path(path)
            self.debug.publish_local_path(100.1)
            result = self.debug.local_path_pub.publish.call_args.args[0]
            self.assertEqual((result.poses[0].pose.position.x, result.poses[0].pose.position.y), (2., 3.))
            self.assertAlmostEqual(result.poses[0].pose.position.z, -.31)
            map_result = self.debug.map_local_path_pub.publish.call_args.args[0]
            self.assertEqual((map_result.poses[0].pose.position.x, map_result.poses[0].pose.position.y), (2., 3.))
            self.assertEqual(map_result.poses[0].pose.position.z, 1.4)
            self.assertEqual(self.debug.map_path_marker_pub.publish.call_args.args[0].action,
                             map_debug.Marker.ADD)
            self.assertEqual(path.poses[0].pose.position.z, 0.)
            self.debug.tracking[0]['valid'] = False
            self.debug.publish_local_path(100.2)
            self.assertEqual(self.debug.local_path_pub.publish.call_args.args[0].poses, [])
            self.assertEqual(self.debug.local_path_marker_pub.publish.call_args.args[0].action,
                             map_debug.Marker.DELETE)

    def test_selected_cloud_source_never_reuses_stale_points(self):
        self.debug.used_points_pub = MagicMock()
        self.debug.discarded_points_pub = MagicMock()
        cloud = map_debug.point_cloud2.create_cloud_xyz32(
            map_debug.Header(stamp=map_debug.rospy.Time(100), frame_id='map'), [(1., 2., 3.)])
        self.debug.source_snapshots['raceline']['used'] = (cloud, 100.)
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(100.1)):
            self.debug.publish_debug_clouds(100.1)
            self.assertEqual(self.debug.used_points_pub.publish.call_args.args[0], cloud)
            self.debug.publish_debug_clouds(100.7)
            self.assertEqual(self.debug.used_points_pub.publish.call_args.args[0].width, 0)

    def test_vehicle_box_uses_user_dimensions_and_rear_axle_origin(self):
        from map_debug_geometry import vehicle_box_corners
        corners = vehicle_box_corners((10., 20., math.pi*.5), 3.845, .79, .946, -.38, 2.434)
        self.assertAlmostEqual(max(p[0] for p in corners)-min(p[0] for p in corners), 1.892)
        self.assertAlmostEqual(max(p[1] for p in corners)-min(p[1] for p in corners), 4.635)
        self.assertAlmostEqual(max(p[2] for p in corners)-min(p[2] for p in corners), 2.434)
        self.assertAlmostEqual(max(p[1] for p in corners), 23.845)

    def test_old_source_stamp_rejected_despite_fresh_delivery(self):
        state = dict(valid=True, sector=11, raceline=1, stamp=98.)
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(100)):
            self.debug.on_tracking_state(map_debug.String(data=json.dumps(state)))
            self.assertIsNone(self.debug.tracking)
            pose = map_debug.PoseStamped()
            pose.header = map_debug.Header(stamp=map_debug.rospy.Time(98), frame_id='map')
            pose.pose.orientation.w = 1.
            self.debug.on_pose(pose)
            self.assertIsNone(self.debug.pose)


class MapDebugObstacleBoxTest(unittest.TestCase):
    def make_debug(self, source='raceline'):
        debug = map_debug.MapDebug.__new__(map_debug.MapDebug)
        debug.obstacle_pub = MagicMock()
        debug.obstacle_marker_ids = set()
        debug.ground_z = -.38
        debug.lidar_obstacle_pub = MagicMock()
        debug.lane_corridor = MagicMock()
        debug.lane_corridor.contains = lambda points: [True]*len(points)
        debug.saved_lane_corridor = debug.lane_corridor
        debug.source_snapshots = {key: {} for key in ('raceline', 'leo', 'route', 'ftg')}
        debug.active_source = source
        debug.perception_source = lambda: source
        return debug

    def show_obstacle(self, debug, item, source='raceline', received=10., now=10.):
        with patch.object(map_debug.rospy.Time, 'now', return_value=map_debug.rospy.Time(10)), \
                patch.object(map_debug.time, 'monotonic', return_value=received):
            debug.on_obstacles(map_debug.String(data=json.dumps(dict(
                stamp=10., frame_id='map', obstacles=[item]))), source)
            debug.publish_obstacles(now)
        return {m.ns: m for m in debug.lidar_obstacle_pub.publish.call_args.args[0].markers}

    def test_rectangle_marker_uses_detector_box(self):
        debug = self.make_debug()
        item = dict(id=4, x=1., y=2., radius=2.49, vx=0., vy=0., kind='dynamic',
                    length=4.6, width=1.9, yaw=1.)
        self.show_obstacle(debug, item)
        markers = debug.obstacle_pub.publish.call_args.args[0].markers
        box = next(m for m in markers if m.ns == 'obstacle_boxes')
        self.assertAlmostEqual(box.scale.x, 4.6)
        self.assertAlmostEqual(box.scale.y, 1.9)
        self.assertAlmostEqual(2.*math.atan2(box.pose.orientation.z, box.pose.orientation.w), 1.)
        edges = next(m for m in markers if m.ns == 'obstacle_edges')
        corner = edges.points[0]
        self.assertAlmostEqual(math.hypot(corner.x-1., corner.y-2.), math.hypot(2.3, .95))

    def test_measured_cube_edges_and_label_share_z_without_repeated_corner_shift(self):
        for kind in ('static', 'dynamic', 'unknown'):
            with self.subTest(kind=kind):
                debug = self.make_debug()
                item = dict(id=4, x=1., y=2., radius=2.49, vx=0., vy=0., kind=kind,
                            length=4.6, width=1.9, yaw=1., z_min=.23, z_max=2.23)
                markers = self.show_obstacle(debug, item)
                box = markers['obstacle_boxes']
                self.assertEqual(box.action, map_debug.Marker.ADD)
                self.assertAlmostEqual(box.pose.position.z-box.scale.z*.5, .23)
                self.assertAlmostEqual(box.pose.position.z+box.scale.z*.5, 2.23)
                edges = markers['obstacle_edges'].points
                self.assertEqual(len(edges), 24)
                self.assertTrue(all(min(abs(p.z-.23), abs(p.z-2.23)) < 1e-9 for p in edges))
                self.assertAlmostEqual(markers['obstacle_labels'].pose.position.z, 2.53)
                aerial = {m.ns: m for m in debug.obstacle_pub.publish.call_args.args[0].markers}
                self.assertEqual(aerial['obstacle_boxes'].pose.position.z, 3.75)
                self.assertEqual({p.z for p in aerial['obstacle_edges'].points}, {3., 4.5})

    def test_every_source_uses_its_matching_scan_and_rotated_footprint(self):
        item = dict(id=4, x=1., y=2., radius=2.49, vx=0., vy=0., kind='static',
                    length=4.6, width=1.0, yaw=math.pi*.5)
        cloud = xyz_cloud(map_debug.Header(stamp=map_debug.rospy.Time(10), frame_id='map'),
                          [[1., 0., .7], [1.4, 4., 2.8], [2., 2., 99.]])
        for source in ('raceline', 'leo', 'route', 'ftg'):
            with self.subTest(source=source):
                debug = self.make_debug(source)
                with patch.object(map_debug.time, 'monotonic', return_value=10.):
                    debug.on_debug_cloud(cloud, (source, 'used'))
                    # A newer cloud must not replace the obstacle's scan.
                    newer = xyz_cloud(map_debug.Header(stamp=map_debug.rospy.Time(10.1),
                                                       frame_id='map'), [[1., 2., 90.]])
                    debug.on_debug_cloud(newer, (source, 'used'))
                markers = self.show_obstacle(debug, item, source)
                box = markers['obstacle_boxes']
                self.assertEqual(box.action, map_debug.Marker.ADD)
                self.assertAlmostEqual(box.pose.position.z-box.scale.z*.5, .7, places=6)
                self.assertAlmostEqual(box.pose.position.z+box.scale.z*.5, 2.8, places=6)

    def test_missing_wrong_source_wrong_stamp_and_stale_heights_only_draw_footprint(self):
        item = dict(id=4, x=1., y=2., radius=2., vx=0., vy=0., kind='static',
                    z_min=float('nan'), z_max=5.)
        for source, stamp, received in (('leo', 10., 10.), ('raceline', 10.1, 10.),
                                        ('raceline', 10., 9.)):
            with self.subTest(source=source, stamp=stamp, received=received):
                debug = self.make_debug()
                cloud = xyz_cloud(map_debug.Header(stamp=map_debug.rospy.Time(stamp),
                                                   frame_id='map'), [[1., 2., 99.]])
                with patch.object(map_debug.time, 'monotonic', return_value=received):
                    debug.on_debug_cloud(cloud, (source, 'used'))
                markers = self.show_obstacle(debug, item)
                self.assertEqual(markers['obstacle_boxes'].action, map_debug.Marker.DELETE)
                for point in markers['obstacle_edges'].points:
                    self.assertAlmostEqual(point.z, debug.ground_z+.03)

    def test_single_observed_height_remains_flat_at_measured_z(self):
        debug = self.make_debug()
        item = dict(id=4, x=1., y=2., radius=2., vx=0., vy=0., kind='static',
                    z_min=.8, z_max=.8)
        markers = self.show_obstacle(debug, item)
        self.assertEqual(markers['obstacle_boxes'].action, map_debug.Marker.DELETE)
        self.assertEqual({p.z for p in markers['obstacle_edges'].points}, {.8})
        self.assertAlmostEqual(markers['obstacle_labels'].pose.position.z, 1.1)

    def test_debug_bounds_apply_mount_offset_once_before_display_decimation(self):
        publishers = [MagicMock() for _ in range(3)]
        for publisher in publishers:
            publisher.get_num_connections.return_value = 1
        with patch('perception_debug.rospy.Publisher', side_effect=publishers), \
                patch('perception_debug.rospy.on_shutdown'), \
                patch('perception_debug.rospy.get_param', side_effect=lambda name, default: default):
            debug = PerceptionDebug('/test/perception_debug')
        self.addCleanup(debug.close)
        debug.max_points = 1
        completed = threading.Event()
        debug.status_pub.publish.side_effect = lambda _: completed.set()
        points = np.array([[10., 0., -1.], [10.5, .1, 1.], [50., 5., 99.]])
        original = points.copy()
        obstacle = dict(id=4, x=100., y=31.65, length=2., width=.5, yaw=math.pi*.5)
        debug.publish(points, points, map_debug.Header(stamp=map_debug.rospy.Time(10)),
                      (100., 20., math.pi*.5), offset=(1.4, 0., 1.23), obstacles=[obstacle])
        self.assertTrue(completed.wait(2.))
        state = json.loads(debug.status_pub.publish.call_args.args[0].data)
        self.assertAlmostEqual(state['obstacles'][0]['z_min'], .23)
        self.assertAlmostEqual(state['obstacles'][0]['z_max'], 2.23)
        self.assertEqual(debug.used_pub.publish.call_args.args[0].width, 1)
        self.assertEqual(state['debug_clouds']['used']['total'], 3)
        self.assertEqual(state['cloud_stamp'], 10.)
        self.assertNotIn('z_min', obstacle)
        np.testing.assert_array_equal(points, original)


if __name__ == '__main__':
    unittest.main()
