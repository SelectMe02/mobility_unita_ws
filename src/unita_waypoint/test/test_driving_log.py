"""Archive consistency, diagnostics gating, and explicit perception waits."""
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import rospy
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, String

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from driving_log_recorder import DrivingLogRecorder, record_command, BAG_TOPICS, json_safe
from gps_waypoint_follower import GPSWaypointFollower
from lidar_obstacles import LidarObstacles
from morai_msgs.msg import CtrlCmd
from map_debug import MapDebug


class DrivingLogTest(unittest.TestCase):
    def test_recorder_archives_parameters_events_and_finalizes_bag(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            params = {'lidar_obstacles': {'ground_plane_tolerance_m': .04}}
            process = MagicMock(pid=98765)
            process.poll.return_value = None
            def get_param(name, default=None):
                return str(directory) if name == '~directory' else params if name == '/' else default
            with patch('rospy.get_param', side_effect=get_param), \
                    patch('rospy.get_master') as master, \
                    patch('rospy.Time.now', return_value=rospy.Time(100.)), \
                    patch('rospy.get_published_topics', return_value=[('/imu', 'sensor_msgs/Imu')]), \
                    patch('rospy.Subscriber'), patch('rospy.Timer'), patch('rospy.on_shutdown'), \
                    patch('rospy.loginfo'), patch('rosnode.get_node_names', return_value=['/imu_udp_bridge']), \
                    patch('subprocess.Popen', return_value=process) as popen, patch('os.killpg') as kill:
                master.return_value.getParam.side_effect = lambda _: (1, '', params)
                recorder = DrivingLogRecorder()
                self.assertEqual(popen.call_args.args[0], record_command(directory))
                self.assertIn('/lidar3D', BAG_TOPICS)
                self.assertIn('/diagnostics/simulator/ego_status_udp', BAG_TOPICS)
                self.assertIn('/ctrl_cmd', BAG_TOPICS)
                recorder.on_json(String(data=json.dumps(dict(reason='obstacle_perception_stale'))),
                                 '/waypoint_debug/control_decision')
                recorder.health(None)
                # Parameter edits are preserved in history, not just latest YAML.
                params['lidar_obstacles']['ground_plane_tolerance_m'] = .06
                recorder.health(None)
                recorder.close()
                kill.assert_called_once()
                process.wait.assert_called_once()
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertTrue(manifest['diagnostic_only'])
            self.assertIn('closed', manifest)
            self.assertIn('/lidar3D', manifest['missing_topics'])
            rows = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['data']['reason'], 'obstacle_perception_stale')
            history = [row for row in rows if row['topic'] == 'parameters']
            self.assertEqual([row['data']['lidar_obstacles']['ground_plane_tolerance_m'] for row in history], [.04, .06])
            self.assertTrue((directory / 'config/map_tunner_project.json').is_file())

    def test_nonfinite_values_are_normalized_without_changing_text(self):
        self.assertEqual(json_safe(dict(value=math.inf, message='NaN Infinity')), dict(value=None, message='NaN Infinity'))

    def test_control_log_contains_stop_reason_and_actual_pedals(self):
        n = GPSWaypointFollower.__new__(GPSWaypointFollower)
        n.control_decision_pub = MagicMock()
        n.stop_reason = 'obstacle_perception_stale'
        n.current_speed_kmh = 15.
        n.diagnostic_speed_limits = dict(profile_kmh=60., local_planner_kmh=0.)
        cmd = CtrlCmd(longlCmdType=1, brake=1.)
        with patch('rospy.Time.now', return_value=rospy.Time(100.)):
            n.publish_control_decision(cmd, 0., n.stop_reason)
        data = json.loads(n.control_decision_pub.publish.call_args.args[0].data)
        self.assertEqual(data['reason'], n.stop_reason)
        self.assertEqual(data['command']['brake'], 1.)
        self.assertEqual(data['speed_limits']['local_planner_kmh'], 0.)
        n.control_decision_pub = None
        n.publish_control_decision(cmd, 0., 'disabled')

    def test_pose_wait_is_reported_instead_of_silently_dropping_cloud(self):
        n = LidarObstacles.__new__(LidarObstacles)
        n.lock = threading.Lock()
        n.pending = point_cloud2.create_cloud_xyz32(Header(stamp=rospy.Time(100.), frame_id='lidar'), [])
        n.lane_corridor = object()
        n.pose_at = lambda _: None
        n.ground_status_pub = MagicMock()
        n.poses, n.imus = [], []
        with patch('rospy.Time.now', return_value=rospy.Time(100.)):
            n.process_pending(None)
        data = json.loads(n.ground_status_pub.publish.call_args.args[0].data)
        self.assertEqual(data['mode'], 'waiting_pose')
        self.assertEqual(data['cloud_stamp'], 100.)
        self.assertEqual(data['pose_samples'], 0)

    def test_green_map_projection_keeps_xy_stamp_and_displays_below_map_road(self):
        n = MapDebug.__new__(MapDebug)
        n.ground_display_pub = MagicMock()
        message = point_cloud2.create_cloud_xyz32(Header(stamp=rospy.Time(100.), frame_id='map'),
                                                [(12., 3., -.16)])
        n.on_ground_points(message)
        projected = n.ground_display_pub.publish.call_args.args[0]
        point = next(point_cloud2.read_points(projected))
        self.assertAlmostEqual(point[2], .15)
        self.assertEqual(point[:2], (12., 3.))
        self.assertEqual(projected.header.stamp, message.header.stamp)
        self.assertAlmostEqual(next(point_cloud2.read_points(message))[2], -.16)


class DrivingLogSessionTest(unittest.TestCase):
    def test_run_folder_is_date_named_unique_and_matched_by_session(self):
        from datetime import datetime
        import driving_log_session as session
        with tempfile.TemporaryDirectory() as folder:
            now = datetime(2026, 10, 15, 23, 28, 0)
            first = session.new_run_directory(folder, now)
            second = session.new_run_directory(folder, now)
            self.assertEqual(first.name, 'run_20261015_232800')
            self.assertEqual(second.name, 'run_20261015_232800_2')
            params = {}
            fake = MagicMock()
            fake.set_param.side_effect = params.__setitem__
            fake.get_param.side_effect = lambda name, default=None: params.get(name, default)
            self.assertIsNone(session.announced_directory(fake, 'launch_a'))
            session.announce(fake, first, 'launch_a')
            self.assertEqual(session.announced_directory(fake, 'launch_a'), first)
            # A folder announced by an earlier launch is never reused.
            self.assertIsNone(session.announced_directory(fake, 'launch_b'))

    def test_raceline_log_buffers_until_recorder_announces_folder(self):
        import driving_log_session as session
        from raceline_test_follower import RacelineTestFollower
        with tempfile.TemporaryDirectory() as folder:
            params = {}
            node = RacelineTestFollower.__new__(RacelineTestFollower)
            node.diagnostic_file = None
            node.last_diagnostic_key, node.last_diagnostic_time = None, 0.
            node.diagnostic_pending = dict(explicit='', session='launch_a', root=folder,
                                           deadline=float('inf'), buffer=[])
            with patch('raceline_test_follower.rospy.get_param',
                       side_effect=lambda name, default=None: params.get(name, default)), \
                    patch('raceline_test_follower.rospy.Time.now', return_value=rospy.Time(5.)), \
                    patch('raceline_test_follower.rospy.on_shutdown'), \
                    patch('raceline_test_follower.rospy.loginfo'):
                node._diagnostic('start', force=True)
                self.assertIsNone(node.diagnostic_file)
                run = session.new_run_directory(folder)
                params.update({session.DIRECTORY_PARAM: str(run), session.SESSION_PARAM: 'launch_a'})
                node._diagnostic('drive', force=True)
                node.diagnostic_file.close()
            rows = [json.loads(line) for path in run.glob('raceline_*.jsonl')
                    for line in path.read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['start', 'drive'])


if __name__ == '__main__':
    unittest.main()
