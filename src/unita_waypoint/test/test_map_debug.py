#!/usr/bin/env python3
"""Verify saved geometry reaches RViz, including appended points and live saves."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import map_debug
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
                          '/tmp/ignored_project_override.json' if name == '~project_file' else default), \
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
        finish = next(m for m in markers if m.ns == 'cp11_finish_line')
        self.assertEqual([p.y for p in finish.points], [-217.855053961277]*2)
        self.assertGreater(abs(finish.points[1].x-finish.points[0].x), 20.)

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

    def test_debug_uses_fixed_project_even_when_ros_parameter_requests_another_file(self):
        self.assertEqual(self.debug.project_file, self.path.resolve())
        # The production constant is the user's explicitly required path.
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

    def test_follower_priority_and_actual_lane_reach_panel_without_inference(self):
        state = dict(valid=True, sector=11, raceline=3, local_lane=3,
                     current_raceline=3, preferred_raceline=3,
                     preferred_raceline_reason='priority_interval', waypoint_index=2530,
                     finish_line_y=-217.855, finish_line_rear_y=-217.1,
                     finish_line_passed=False)
        status = self.status(state)
        for field, value in state.items():
            if field != 'valid':
                self.assertEqual(status[field], value)
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


if __name__ == '__main__':
    unittest.main()
