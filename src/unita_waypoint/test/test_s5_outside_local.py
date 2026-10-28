import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, String
import test_lane_corridor as fixtures
from lidar_obstacles_local import S5Lane2ObservationZone


def geometry():
    lanes = [dict(sector=s, raceline=1, points=[[x, 0.], [x+100., 0.]])
             for s, x in ((4, -100.), (5, 0.), (6, 100.), (7, 200.), (8, 300.))]
    lanes += [dict(sector=5, raceline=2, points=[[20., -3.6], [100., -3.6]]),
              dict(sector=6, raceline=2, points=[[100., -3.6], [140., -3.6]]),
              dict(sector=8, raceline=2, points=[[300., -3.6], [400., -3.6]])]
    return fixtures.geometry(lanes=lanes)


class OutsideObservationTests(unittest.TestCase):
    def test_zone_stops_at_lane2_endpoint_not_sector_end_or_later_lane2(self):
        zone = S5Lane2ObservationZone(geometry())
        for x, active in ((-1., False), (0., False), (19.99, False), (20., True), (99., True), (139.99, True),
                          (140., False), (180., False), (350., False)):
            self.assertEqual(zone.active((x, 0.)), active, x)
        g = geometry()
        g['lanes'] = [l for l in g['lanes'] if not (l['sector']==5 and l['raceline']==2)]
        self.assertFalse(S5Lane2ObservationZone(g).active((50., 0.)))

    def test_only_zone_observes_outside_and_exit_discards_outside_tracks(self):
        g = geometry()
        n = fixtures.LaneCorridorTests().lidar_node(fixtures.LaneCorridor(g), fixtures.LocalLidarObstacles)
        n.debug = MagicMock()
        n.debug.active.return_value = True
        with patch('lidar_obstacles_local.rospy.set_param'), patch('lidar_obstacles_local.rospy.loginfo'):
            n.on_lane_geometry(String(data=json.dumps(g)))
        # Ground support plus a small raised object on the left, outside all lanes.
        x,y = np.meshgrid(np.arange(8., 13., .25),np.arange(2., 6., .25))
        ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        raised = np.array([[10.,4.,1.],[10.3,4.,1.],[10.6,4.,1.],[10.3,4.3,1.]])
        for i,(ego,enabled) in enumerate(((-50.,False),(50.,True),(139.,True),(140.,False),(350.,False))):
            stamp=10.+i*.1
            n.pose_at.return_value=(ego,0.,0.)
            n.pending=point_cloud2.create_cloud_xyz32(Header(stamp=rospy.Time(stamp),frame_id='lidar'),np.vstack((ground,raised)))
            with patch('lidar_obstacles_local.rospy.Time.now',return_value=rospy.Time(stamp)):
                n.process_pending(None)
            obstacles=json.loads(n.pub.publish.call_args.args[0].data)['obstacles']
            self.assertEqual(bool(obstacles),enabled,ego)
            self.assertEqual(n.debug.publish.call_args.kwargs['s5_lane2_outside_observation'],enabled)
            if not enabled:
                self.assertFalse(n.tracks)


if __name__=='__main__':
    unittest.main()
