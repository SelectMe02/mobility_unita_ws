import unittest
from local_avoidance_core_local import LocalAvoidancePlanner, Obstacle

class ActiveS5AdjacentTests(unittest.TestCase):
    def planner(self, sector=5):
        route=[(float(x), 0.) for x in range(100)]
        return LocalAvoidancePlanner(route, [sector]*100,
            {2:[(float(x), -3.5) for x in range(100)]})

    def car(self, **kw):
        values=dict(x=6., y=-3.3, radius=2.8, vx=2.7, kind='dynamic',
            length=4., raceline=2, lane_sector=5, lane_stable=True,
            confirmed=True, velocity_reliable=True)
        values.update(kw)
        return Obstacle(**values)

    def test_adjacent_large_circle_does_not_block_any_keep_lane_check(self):
        p=self.planner()
        for radius in (2.2, 3., 4.5, 2.2):
            d=p.plan(0, (0.,0.), 5., [self.car(radius=radius)])
            self.assertEqual(d.state, 'RACELINE')
            self.assertGreater(d.speed_cap_kmh, 0.)
            self.assertIsNone(p.transition)

    def test_our_lane_unknown_labels_pedestrian_and_approach_still_protected(self):
        for kw in [dict(raceline=1),dict(lane_stable=False),dict(confirmed=False),
                   dict(y=0.),dict(length=.6),dict(kind='unknown'),dict(vy=1.)]:
            d=self.planner().plan(0, (0.,0.), 5., [self.car(**kw)])
            self.assertNotEqual(d.state, 'RACELINE',kw)

    def test_other_sectors_unchanged(self):
        for sector in (4,6,11):
            d=self.planner(sector).plan(0,(0.,0.),5.,[self.car()])
            self.assertNotEqual(d.state,'RACELINE')

    def test_lane_change_still_checks_adjacent_car(self):
        p=self.planner()
        p.transition=dict(source=1,target=2,start=0.,length=20.,state='OVERTAKING',
                          shape='smooth',lateral=3.5)
        d=p.plan(0,(0.,0.),5.,[self.car()])
        self.assertNotEqual(d.state,'RACELINE')
        self.assertLess(d.speed_cap_kmh,25.)

    def test_same_lane_car_is_followed_then_released_when_it_leaves(self):
        p=self.planner()
        for x in (18., 12., 8.):
            d=p.plan(0,(0.,0.),5.,[self.car(x=x,y=0.,radius=1.,raceline=1)])
            self.assertEqual(d.state,'TAILING')
            self.assertEqual(d.reason,'s5_follow_vehicle')
            self.assertIsNone(p.transition)
        d=p.plan(0,(0.,0.),5.,[self.car(x=8.)])
        self.assertEqual(d.state,'RACELINE')
        self.assertIsNone(p.transition)

    def test_reliable_forward_motion_survives_static_label_dropout(self):
        d=self.planner().plan(0,(0.,0.),5.,[
            self.car(x=18.,y=0.,radius=1.,raceline=1,kind='static')])
        self.assertEqual(d.state,'TAILING')
        self.assertEqual(d.reason,'s5_follow_vehicle')

class ActiveS5InputTests(unittest.TestCase):
    def test_callback_preserves_lane_and_vehicle_evidence(self):
        import json
        from unittest.mock import Mock, patch
        import rospy
        from std_msgs.msg import String
        from gps_waypoint_follower_local import GPSWaypointFollower
        n=GPSWaypointFollower.__new__(GPSWaypointFollower)
        n.lane_corridor=Mock()
        n.lane_corridor.contains.return_value=[True]
        n.waypoint_only_ranges=[]
        car=dict(x=6.,y=-3.5,radius=3.,vx=2.,vy=0.,kind='dynamic',
                 length=4.,raceline=2,lane_sector=5,lane_stable=True,
                 confirmed=True,velocity_reliable=True)
        with patch('gps_waypoint_follower_local.rospy.Time.now',return_value=rospy.Time(10.)):
            n.obstacles_callback(String(data=json.dumps(dict(stamp=10.,frame_id='map',obstacles=[car]))))
        self.assertEqual(n.obstacles[0].raceline,2)
        self.assertEqual(n.obstacles[0].length,4.)
        self.assertTrue(n.obstacles[0].lane_stable)
