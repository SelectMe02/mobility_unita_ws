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
from lane_corridor import LaneCorridor, SectorLaneFilter
from lidar_obstacles import LidarObstacles
from lidar_obstacles_local import LidarObstacles as LocalLidarObstacles
from gps_waypoint_follower import GPSWaypointFollower
from map_debug import MapDebug


def geometry(width=3.2, lanes=None):
    return dict(frame_id='map', raceline_width=width, loop=False,
                lanes=lanes or [dict(sector=1, raceline=1, points=[[-50., 0.], [50., 0.]])])


def sector_geometry():
    return geometry(width=3.6, lanes=[
        dict(sector=11, raceline=1, points=[[-100., 0.], [-10., 0.]]),
        dict(sector=11, raceline=2, points=[[-100., 3.6], [0., 3.6]]),
        dict(sector=12, raceline=1, points=[[0., 0.], [50., 0.]]),
        dict(sector=13, raceline=1, points=[[50., 0.], [120., 0.]])])


class LaneCorridorTests(unittest.TestCase):
    def test_sector_filter_activates_on_projected_interval_not_nearest_vertex(self):
        f = SectorLaneFilter(sector_geometry())
        for x, expected in ((-50., False), (-.001, False), (0., True),
                            (.001, True), (49.999, True), (50., False), (70., False)):
            self.assertEqual(f.active((x, 0.)), expected, x)
        self.assertFalse(f.active((float('nan'), 0.)))
        # The incoming join is 70 m long. Its points must not acquire S12's
        # policy merely because the closest saved vertex belongs to S12.
        g = geometry(width=3.6, lanes=[
            dict(sector=11, raceline=1, points=[[-80., 0.], [-70., 0.]]),
            dict(sector=12, raceline=1, points=[[0., 0.], [10., 0.]]),
            dict(sector=13, raceline=1, points=[[10., 0.], [20., 0.]])])
        f = SectorLaneFilter(g)
        self.assertFalse(f.active((-1., 0.)))
        self.assertTrue(f.active((0., 0.)))
        self.assertTrue(f.active((9.999, 0.)))
        self.assertFalse(f.active((10., 0.)))

    def test_sector_filter_has_exact_width_and_keeps_next_sector_ahead(self):
        g = sector_geometry()
        f = SectorLaneFilter(g)
        self.assertEqual(g['raceline_width'], 3.6)
        self.assertEqual(f.corridor.width, 3.2)
        self.assertEqual(f.corridor.contains([
            [20., 1.6], [20., -1.6], [20., 1.7], [20., -1.7],
            [70., 1.5], [-5., 0.], [-20., 3.6]]).tolist(),
            [True, True, False, False, True, False, False])

    def test_sector_filter_missing_sector_and_other_sectors_keep_original_corridor(self):
        missing = SectorLaneFilter(geometry())
        self.assertIsNone(missing.corridor)
        self.assertFalse(missing.active((0., 0.)))
        g = sector_geometry()
        original, policy = LaneCorridor(g), SectorLaneFilter(g)
        cloud = [[-20., 3.6], [-20., 1.7], [20., 1.7], [70., 1.7]]
        for pose in ((-20., 0.), (70., 0.)):
            selected = policy.corridor if policy.active(pose) else original
            self.assertIs(selected, original)
            np.testing.assert_array_equal(selected.contains(cloud), original.contains(cloud))

    def test_box_midpoint_labels_three_lanes_and_tracks_boundary_intrusion(self):
        c = LaneCorridor(geometry(width=3.6, lanes=[dict(sector=11, raceline=n,
             points=[[-50., (n-1)*3.6], [50., (n-1)*3.6]]) for n in (1, 2, 3)]))
        for n in (1, 2, 3):
            label = c.label_box(20., (n-1)*3.6, 4.5, 1.9, 0.)
            self.assertEqual((label['lane_sector'], label['raceline']), (11, n))
            self.assertTrue(label['lane_label_confident'])
            self.assertFalse(label['lane_boundary_crossing'])
        crossing = c.label_box(20., 1.2, 4.5, 1.9, 0.)
        self.assertEqual(crossing['raceline'], 1)
        self.assertTrue(crossing['lane_boundary_crossing'])
        angled = c.label_box(20., 0., 4.5, 1.9, math.pi/4.)
        self.assertTrue(angled['lane_boundary_crossing'])
        edge = c.label_box(20., 1.8, 4.5, 1.9, 0.)
        self.assertFalse(edge['lane_label_confident'])

    def test_lane_label_never_extrapolates_finite_lane_or_accepts_roadside(self):
        c = LaneCorridor(geometry(lanes=[
            dict(sector=11, raceline=1, points=[[-50., 0.], [50., 0.]]),
            dict(sector=11, raceline=2, points=[[-20., 3.2], [-10., 3.2]]),
            dict(sector=12, raceline=2, points=[[10., 3.2], [20., 3.2]])]))
        self.assertEqual(c.label_box(-15., 3.2, 4., 1.8, 0.)['raceline'], 2)
        for x, y in ((0., 3.2), (60., 0.), (0., 10.), (float('nan'), 0.)):
            label = c.label_box(x, y, 4., 1.8, 0.)
            self.assertEqual(label['raceline'], 0)
            self.assertFalse(label['lane_label_confident'])

    def test_boundary_is_half_full_width_and_interpolates_between_waypoints(self):
        c = LaneCorridor(geometry())
        self.assertEqual(c.contains([[0., 1.6], [0., -1.6], [0., 1.601],
                                     [0., -1.601], [51., 0.]]).tolist(),
                         [True, True, False, False, False])
        wide = LaneCorridor(geometry(width=4.))
        self.assertTrue(wide.contains([[0., 1.9]])[0])
        self.assertFalse(c.contains([[0., 1.9]])[0])

    def test_depth_is_distance_from_the_edge_and_constant_along_a_bend(self):
        c = LaneCorridor(geometry(width=3.6))
        np.testing.assert_allclose(c.depth([[0., 1.7], [0., 0.], [0., -1.5]]), [.1, 1.8, .3])
        np.testing.assert_allclose(c.depth([[0., 1.9]]), [-.1])
        self.assertEqual(c.depth([[60., 0.]]).tolist(), [-np.inf])
        # R=50 m bend: a barrier 1.7 m outside the centre line is 0.1 m deep everywhere.
        arc = [[50.*math.sin(a), 50.-50.*math.cos(a)] for a in np.linspace(0., 1., 60)]
        bend = LaneCorridor(geometry(width=3.6, lanes=[dict(sector=1, raceline=1, points=arc)]))
        rail = [[51.7*math.sin(a), 50.-51.7*math.cos(a)] for a in np.linspace(.1, .9, 40)]
        np.testing.assert_allclose(bend.depth(rail), .1, atol=.01)

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

    def lidar_node(self, corridor, node_type=LidarObstacles):
        n = node_type.__new__(node_type)
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

    def test_both_lidars_narrow_points_before_clustering_only_in_sector12(self):
        g = sector_geometry()
        # One centre object, one straddling the boundary, one entirely outside,
        # and one object in the next sector. The straddling object's inner
        # returns must survive even though its outer returns are removed.
        points = [(x+dx, y+dy, 1.)
                  for x, y, offsets in ((20., 0., (-.3, 0., .3)),
                                         (25., 1.55, (-.1, 0., .1)),
                                         (30., 1.75, (-.04, 0., .04)),
                                         (40., 0., (-.3, 0., .3)))
                  for dx in (-.3, 0., .3) for dy in offsets]
        for node_type in (LidarObstacles, LocalLidarObstacles):
            with self.subTest(node=node_type.__module__):
                n = self.lidar_node(LaneCorridor(g), node_type)
                n.sector_lane_filter = SectorLaneFilter(g)
                n.pose_at.return_value = (20., 0., 0.)
                n.debug = MagicMock()
                n.pending = point_cloud2.create_cloud_xyz32(
                    Header(stamp=rospy.Time(100), frame_id='lidar'), points)
                with patch('rospy.Time.now', return_value=rospy.Time(100)):
                    n.process_pending(None)
                objects = json.loads(n.pub.publish.call_args.args[0].data)['obstacles']
                self.assertEqual(len(objects), 3)
                self.assertTrue(any(o['x'] > 55. for o in objects))
                self.assertFalse(any(48. < o['x'] < 52. for o in objects))
                self.assertTrue(any(43. < o['x'] < 47. and o['y'] <= 1.6 for o in objects))
                if node_type is LidarObstacles:
                    status = json.loads(n.ground_status_pub.publish.call_args.args[0].data)
                else:
                    status = n.debug.publish.call_args.kwargs
                self.assertTrue(status['s12_filter_active'])
                self.assertEqual(status['lane_filter_width_m'], 3.2)
                self.assertEqual(n.lane_corridor.width, 3.6)

    def test_both_lidars_remove_outside_tracks_before_associating_inside_returns(self):
        for node_type in (LidarObstacles, LocalLidarObstacles):
            with self.subTest(node=node_type.__module__):
                n = self.lidar_node(LaneCorridor(sector_geometry()), node_type)
                n.sector_lane_filter = SectorLaneFilter(sector_geometry())
                n.pose_at.return_value = (20., 0., 0.)
                n.tracks = {99: dict(x=40., y=1.7, radius=.5, vx=3., vy=0.,
                                    hits=5, stamp=99.9, confirmed=True,
                                    velocity_reliable=True)}
                n.next_id = 100
                points = [(20.+dx, y, 1.) for dx in (-.3, 0., .3)
                          for y in (1.3, 1.4, 1.5)]
                n.pending = point_cloud2.create_cloud_xyz32(
                    Header(stamp=rospy.Time(100), frame_id='lidar'), points)
                with patch('rospy.Time.now', return_value=rospy.Time(100)):
                    n.process_pending(None)
                objects = json.loads(n.pub.publish.call_args.args[0].data)['obstacles']
                self.assertEqual([o['id'] for o in objects], [100])
                self.assertEqual(objects[0]['vx'], 0.)
                self.assertNotIn(99, n.tracks)

    def test_both_lidars_keep_other_sector_detection_results_identical(self):
        g = sector_geometry()
        for node_type in (LidarObstacles, LocalLidarObstacles):
            for ego_x in (-30., 70.):
                with self.subTest(node=node_type.__module__, ego_x=ego_x):
                    published = []
                    for enabled in (False, True):
                        n = self.lidar_node(LaneCorridor(g), node_type)
                        if enabled:
                            n.sector_lane_filter = SectorLaneFilter(g)
                        n.pose_at.return_value = (ego_x, 0., 0.)
                        points = [(20.+dx, y+dy, 1.)
                                  for y in (0., 1.7, 3.6)
                                  for dx in (-.3, 0., .3) for dy in (-.03, 0., .03)]
                        n.pending = point_cloud2.create_cloud_xyz32(
                            Header(stamp=rospy.Time(100), frame_id='lidar'), points)
                        with patch('rospy.Time.now', return_value=rospy.Time(100)):
                            n.process_pending(None)
                        published.append(json.loads(n.pub.publish.call_args.args[0].data))
                    self.assertEqual(published[0], published[1])
                    self.assertTrue(any(o['y'] > 1.6 for o in published[1]['obstacles']))

    def test_both_lidars_prune_predictions_crossing_sector12_width(self):
        for node_type in (LidarObstacles, LocalLidarObstacles):
            with self.subTest(node=node_type.__module__):
                n = self.lidar_node(LaneCorridor(sector_geometry()), node_type)
                n.sector_lane_filter = SectorLaneFilter(sector_geometry())
                n.pose_at.return_value = (20., 0., 0.)
                n.tracks = {1: dict(x=40., y=1.5, radius=.5, vx=0., vy=2.,
                                   hits=5, stamp=99.9, confirmed=True,
                                   velocity_reliable=True)}
                n.pending = point_cloud2.create_cloud_xyz32(
                    Header(stamp=rospy.Time(100), frame_id='lidar'), [])
                with patch('rospy.Time.now', return_value=rospy.Time(100)):
                    n.process_pending(None)
                self.assertEqual(json.loads(n.pub.publish.call_args.args[0].data)['obstacles'], [])
                self.assertEqual(n.tracks, {})

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

    def test_follower_keeps_roundabout_lane_only_near_s9_merge(self):
        f = GPSWaypointFollower.__new__(GPSWaypointFollower)
        f.lane_corridor = LaneCorridor(geometry())
        f.roundabout_obstacle_corridor = LaneCorridor(geometry(
            width=4.5, lanes=[dict(sector=9, raceline=1,
                                  points=[[-50., 5.], [50., 5.]])]))
        f.roundabout_obstacle_conflict_point = (20., 0.)
        f.roundabout_obstacle_sector = 9
        f.roundabout_obstacle_monitor_radius_m = 35.
        f.route_sectors = [9]
        f.current_waypoint_index = 0
        f.current_x, f.current_y = 0., 0.
        objects = [dict(id=i, x=20., y=y, radius=.8, vx=0., vy=0., kind='unknown')
                   for i, y in enumerate((0., 5., 10.), 1)]
        message = String(data=json.dumps(dict(stamp=100., frame_id='map', obstacles=objects)))
        with patch('rospy.Time.now', return_value=rospy.Time(100)):
            f.obstacles_callback(message)
        self.assertEqual([o.track_id for o in f.obstacles], [1, 2])

        # The same adjacent-lane object remains filtered in every other sector
        # and after Ego leaves the local CP8->CP9 monitor radius.
        with patch('rospy.Time.now', return_value=rospy.Time(100)):
            f.route_sectors = [8]
            f.obstacles_callback(message)
        self.assertEqual([o.track_id for o in f.obstacles], [1])
        with patch('rospy.Time.now', return_value=rospy.Time(100)):
            f.route_sectors = [9]
            f.current_x = -20.
            f.obstacles_callback(message)
        self.assertEqual([o.track_id for o in f.obstacles], [1])

    def test_lane_metadata_reaches_planner_and_visible_box_label(self):
        corridor = LaneCorridor(geometry())
        f = GPSWaypointFollower.__new__(GPSWaypointFollower)
        f.lane_corridor = corridor
        d = MapDebug.__new__(MapDebug)
        d.lane_corridor, d.obstacle_snapshot = corridor, None
        d.obstacle_marker_ids, d.obstacle_pub = set(), MagicMock()
        d.ground_z, d.lidar_obstacle_pub = -.1, MagicMock()
        item = dict(id=7, x=20., y=0., radius=2.5, length=4.5, width=1.9,
                    yaw=0., vx=10., vy=0., kind='dynamic', raceline=1,
                    lane_sector=11, lane_stable=True, lane_boundary_crossing=False,
                    raw_width=2.2, predicted=True, measurement_stamp=99.8)
        message = String(data=json.dumps(dict(stamp=100., frame_id='map', obstacles=[item])))
        with patch('rospy.Time.now', return_value=rospy.Time(100)):
            f.obstacles_callback(message)
            d.on_obstacles(message)
            d.publish_obstacles(d.obstacle_snapshot[1])
        o = f.obstacles[0]
        self.assertEqual((o.track_id, o.raceline, o.lane_sector), (7, 1, 11))
        self.assertTrue(o.lane_stable)
        self.assertFalse(o.lane_boundary_crossing)
        self.assertTrue(o.predicted)
        markers = d.obstacle_pub.publish.call_args.args[0].markers
        label = next(m for m in markers if m.ns == 'obstacle_labels')
        self.assertIn('R1', label.text)
        self.assertNotIn('?', label.text)

    def test_width_update_reaches_perception_and_debug_and_clears_old_tracks(self):
        n = self.lidar_node(LaneCorridor(geometry()))
        n.tracks = {1: dict(x=0., y=0.)}
        d = MapDebug.__new__(MapDebug)
        d.obstacle_snapshot = ([], 0., 0.)
        d.source_snapshots = {'raceline': {}}
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
        d.ground_z, d.lidar_obstacle_pub = -.1, MagicMock()
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
