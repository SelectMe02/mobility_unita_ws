"""S4 through CP6 reuse the ground filter without removing the passing lane."""
import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, String
from test_lane_corridor import LaneCorridorTests, LocalLidarObstacles, LaneCorridor


class SectorGroundTests(unittest.TestCase):
    def run_scene(self, ego_x):
        geometry = dict(frame_id='map', raceline_width=3.6, loop=False, lanes=[
            dict(sector=sector, raceline=lane,
                 points=[[start, y], [start+100., y]])
            for sector, start in ((3, -100.), (4, 0.), (5, 100.), (6, 200.), (7, 300.))
            for lane, y in ((1, 0.), (2, -3.6))])
        n = LaneCorridorTests().lidar_node(LaneCorridor(geometry), LocalLidarObstacles)
        n.debug = MagicMock()
        n.debug.active.return_value = True
        with patch('lidar_obstacles_local.rospy.set_param'), patch('lidar_obstacles_local.rospy.loginfo'):
            n.on_lane_geometry(String(data=json.dumps(geometry)))
        n.pose_at.return_value = (ego_x, 0., 0.)
        x, y = np.meshgrid(np.arange(8., 32., .3), np.arange(-5., 1., .3))
        ground = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .25)))
        # A real raised obstacle in the right passing lane must remain.
        x, y = np.meshgrid(np.arange(19.5, 20.6, .2), np.arange(-4., -3.1, .2))
        box = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 1.)))
        n.pending = point_cloud2.create_cloud_xyz32(
            Header(stamp=rospy.Time(10.), frame_id='lidar'), np.vstack((ground, box)))
        with patch('lidar_obstacles_local.rospy.Time.now', return_value=rospy.Time(10.)):
            n.process_pending(None)
        args, status = n.debug.publish.call_args
        return args[1], status, json.loads(n.pub.publish.call_args.args[0].data)['obstacles']

    def test_ground_removed_and_right_lane_obstacle_retained_through_cp6(self):
        for x in (50., 150., 250., 299.):
            with self.subTest(ego_x=x):
                points, status, obstacles = self.run_scene(x)
                self.assertTrue(status['local_ground_filter_active'])
                self.assertTrue(len(points) > 10)
                self.assertTrue(np.all(points[:, 2] > .9))
                self.assertTrue(any(abs(o['x']-(x+20.)) < 1. and o['y'] < -3. for o in obstacles))

    def test_other_sectors_keep_existing_height_filter(self):
        for x in (-50., 301., 350.):
            points, status, _ = self.run_scene(x)
            self.assertFalse(status['local_ground_filter_active'])
            self.assertTrue(np.any(points[:, 2] < .3))


if __name__ == '__main__':
    unittest.main()
