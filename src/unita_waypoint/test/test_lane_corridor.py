"""Roadside rejection, continuous lane widths and shared geometry consumers."""
import json
import math
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, String

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from lane_corridor import LaneCorridor
from lidar_obstacles import LidarObstacles
from gps_waypoint_follower import GPSWaypointFollower
from map_debug import MapDebug


def geometry(width=3.2, lanes=None):
    return dict(frame_id='map', raceline_width=width, loop=False,
                lanes=lanes or [dict(sector=1, raceline=1, points=[[-50., 0.], [50., 0.]])])


class LaneCorridorTests(unittest.TestCase):
    def test_boundary_is_half_full_width_and_interpolates_between_waypoints(self):
        c = LaneCorridor(geometry())
        self.assertEqual(c.contains([[0., 1.6], [0., -1.6], [0., 1.601],
                                     [0., -1.601], [51., 0.]]).tolist(),
                         [True, True, False, False, False])
        wide = LaneCorridor(geometry(width=4.))
        self.assertTrue(wide.contains([[0., 1.9]])[0])
        self.assertFalse(c.contains([[0., 1.9]])[0])

    def test_distance_is_perpendicular_for_rotated_lanes(self):
        lane = dict(sector=1, raceline=1, points=[[-20., -20.], [20., 20.]])
        c = LaneCorridor(geometry(lanes=[lane]))
        normal = np.array([-1., 1.])/math.sqrt(2.)
        self.assertEqual(c.contains([normal*1.59, normal*1.61]).tolist(), [True, False])

    def test_all_three_lanes_are_kept_and_roadside_is_rejected(self):
        c = LaneCorridor(geometry(lanes=[dict(sector=11, raceline=n,
             points=[[-50., (n-1)*3.2], [50., (n-1)*3.2]]) for n in (1, 2, 3)]))
        self.assertEqual(c.contains([[0., 0.], [-20., 3.2], [10., 6.4],
                                     [0., 8.001], [0., -1.601]]).tolist(),
                         [True, True, True, False, False])

    def test_separate_adjacent_lane_intervals_are_not_joined(self):
        lanes = [dict(sector=1, raceline=1, points=[[-50., 0.], [50., 0.]]),
                 dict(sector=1, raceline=2, points=[[-20., 4.], [-10., 4.]]),
                 dict(sector=2, raceline=2, points=[[10., 4.], [20., 4.]])]
        c = LaneCorridor(geometry(lanes=lanes))
        self.assertEqual(c.contains([[-15., 4.], [0., 4.], [15., 4.]]).tolist(),
                         [True, False, True])

    def test_primary_lane_remains_continuous_across_sector_boundary(self):
        c = LaneCorridor(geometry(lanes=[
            dict(sector=1, raceline=1, points=[[-10., 0.], [-.25, 0.]]),
            dict(sector=2, raceline=1, points=[[.25, 0.], [10., 0.]])]))
        self.assertTrue(c.contains([[0., 1.5]])[0])

    def test_invalid_geometry_rejected(self):
        for width in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                LaneCorridor(geometry(width))

    def lidar_node(self, corridor):
        n = LidarObstacles.__new__(LidarObstacles)
        n.lock = threading.Lock()
        n.lane_corridor = corridor
        n.lidar_x = n.lidar_y = n.lidar_z = 0.
        n.front_range = n.rear_range = 55.
        n.side_range = 12.
        n.dynamic_speed_threshold_mps = .8
        n.cluster_min_size_m, n.cluster_max_size_m = .2, 24.
        n.obstacle_min_height_m, n.obstacle_max_height_m = .22, 2.6
        n.body_from_lidar = n.body_from_imu = np.eye(3)
        n.imus = []
        n.imu_max_skew_s, n.ground_plane_tolerance_m, n.ground_max_slope_deg = .05, .04, 20.
        n.ground_status_pub = MagicMock()
        n.ground_points_pub = n.obstacle_points_pub = MagicMock()
        n.ground_points_pub.get_num_connections.return_value = 0
        n.half_width, n.front_extent, n.rear_extent = .946, 3.845, .790
        n.tracks, n.next_id = {}, 1
        n.pub = MagicMock()
        n.pose_at = MagicMock(return_value=(0., 0., 0.))
        return n

    def test_lidar_filters_roadside_before_tracking_including_rear(self):
        n = self.lidar_node(LaneCorridor(geometry()))
        points = [(x+dx, y+dy, 1.) for x, y in ((20., 0.), (-20., .5), (20., 5.))
                  for dx in (-.3, 0., .3) for dy in (-.3, 0., .3)]
        n.pending = point_cloud2.create_cloud_xyz32(Header(stamp=rospy.Time(100), frame_id='lidar'), points)
        with patch('lidar_obstacles.rospy.Time.now', return_value=rospy.Time(100)):
            n.process_pending(None)
        published = json.loads(n.pub.publish.call_args.args[0].data)['obstacles']
        self.assertEqual(len(published), 2)
        self.assertTrue(any(o['x'] < 0. for o in published))
        self.assertTrue(all(abs(o['y']) <= 1.6 for o in published))
        self.assertEqual(len(n.tracks), 2)

    def test_prediction_leaving_lane_is_deleted_and_empty_cloud_clears(self):
        n = self.lidar_node(LaneCorridor(geometry()))
        n.tracks = {1: dict(x=20., y=0., radius=.8, vx=0., vy=10., hits=4, stamp=99.8,
                            confirmed=True, velocity_reliable=True)}
        n.pending = point_cloud2.create_cloud_xyz32(Header(stamp=rospy.Time(100), frame_id='lidar'), [])
        with patch('lidar_obstacles.rospy.Time.now', return_value=rospy.Time(100)):
            n.process_pending(None)
        self.assertEqual(json.loads(n.pub.publish.call_args.args[0].data)['obstacles'], [])
        self.assertEqual(n.tracks, {})

    def test_planning_and_debug_reject_outside_center_even_with_large_radius(self):
        corridor = LaneCorridor(geometry())
        f = GPSWaypointFollower.__new__(GPSWaypointFollower)
        f.lane_corridor = corridor
        d = MapDebug.__new__(MapDebug)
        d.lane_corridor, d.obstacle_snapshot = corridor, None
        objects = [dict(id=i, x=20., y=y, radius=3., vx=0., vy=0., kind='static')
                   for i, y in enumerate((0., 4.), 1)]
        msg = String(data=json.dumps(dict(stamp=100., frame_id='map', obstacles=objects)))
        with patch('rospy.Time.now', return_value=rospy.Time(100)):
            f.obstacles_callback(msg)
            d.on_obstacles(msg)
        self.assertEqual(len(f.obstacles), 1)
        self.assertEqual(len(d.obstacle_snapshot[0]), 1)
        self.assertEqual(d.obstacle_snapshot[0][0][0], 1)

    def test_width_update_reaches_perception_and_debug_and_clears_old_tracks(self):
        n = self.lidar_node(LaneCorridor(geometry()))
        n.tracks = {1: dict(x=0., y=0.)}
        d = MapDebug.__new__(MapDebug)
        d.obstacle_snapshot = ([], 0., 0.)
        d.publish_paths = MagicMock()
        msg = String(data=json.dumps(geometry(width=4.)))
        with patch('rospy.set_param'), patch('rospy.loginfo'):
            n.on_lane_geometry(msg)
            d.on_lane_geometry(msg)
        self.assertEqual(n.lane_corridor.width, 4.)
        self.assertEqual(d.lane_corridor.width, 4.)
        self.assertTrue(n.lane_corridor.contains([[0., 1.9]])[0])
        self.assertTrue(d.lane_corridor.contains([[0., 1.9]])[0])
        self.assertEqual(n.tracks, {})
        self.assertIsNone(d.obstacle_snapshot)
        d.publish_paths.assert_not_called()

    def test_outside_debug_detection_deletes_previously_visible_box(self):
        d = MapDebug.__new__(MapDebug)
        d.lane_corridor = LaneCorridor(geometry())
        d.obstacle_snapshot, d.obstacle_marker_ids = None, set()
        d.obstacle_pub = MagicMock()
        obj = dict(id=1, x=20., y=0., radius=.8, vx=0., vy=0., kind='static')
        with patch('rospy.Time.now', return_value=rospy.Time(100)), \
                patch('map_debug.time.monotonic', return_value=10.):
            d.on_obstacles(String(data=json.dumps(dict(stamp=100., frame_id='map', obstacles=[obj]))))
            d.publish_obstacles(10.)
            self.assertEqual(d.obstacle_marker_ids, {1})
            obj['y'] = 4.
            d.on_obstacles(String(data=json.dumps(dict(stamp=100., frame_id='map', obstacles=[obj]))))
            d.publish_obstacles(10.)
        self.assertEqual(d.obstacle_marker_ids, set())
        self.assertTrue(all(m.action == m.DELETE for m in d.obstacle_pub.publish.call_args.args[0].markers))


if __name__ == '__main__':
    unittest.main()
