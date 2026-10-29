"""Actual Leo producer -> follower: lane metadata must survive the wire."""
import dataclasses
import unittest
import numpy as np
from lidar_obstacles_local import LidarObstacles, object_circles
from lane_corridor import LaneCorridor
from local_avoidance_core_local import LocalAvoidancePlanner, Obstacle

class LocalMetadataTests(unittest.TestCase):
    def setUp(self):
        self.node=LidarObstacles.__new__(LidarObstacles)
        self.node.tracks={};self.node.next_id=1
        self.node.lane_corridor=LaneCorridor(dict(frame_id='map',raceline_width=3.5,
            lanes=[dict(sector=5,raceline=i,points=[[0.,y],[100.,y]]) for i,y in [(1,0.),(2,-3.5)]]))
        self.p=LocalAvoidancePlanner([(float(x),0.) for x in range(100)],[5]*100,
            {2:[(float(x),-3.5) for x in range(100)]})

    def observe(self, y=-3.3, length=4.):
        for i in range(15):
            x=8.+i*.3
            group=np.array([[x+dx,y+dy] for dx in (-length/2,length/2) for dy in (-.7,.7)])
            objects=self.node._update_tracks(object_circles(group,(0.,0.,0.)),i*.1)
        return objects[0]

    def decision(self,o):
        keys={f.name for f in dataclasses.fields(Obstacle)}
        return self.p.plan(0,(0.,0.),5.,[Obstacle(**{k:v for k,v in o.items() if k in keys})])

    def test_parallel_car_metadata_reaches_actual_planner(self):
        o=self.observe()
        self.assertEqual(o['raceline'],2)
        self.assertTrue(o['lane_stable']);self.assertTrue(o['confirmed'])
        self.assertAlmostEqual(o['length'],4.)
        self.assertEqual(o['kind'],'dynamic')
        self.assertEqual(self.decision(o).state,'RACELINE')
        missing={k:v for k,v in o.items() if k not in ('length','confirmed','raceline','lane_stable','lane_sector')}
        self.assertEqual(self.decision(missing).state,'TAILING')

    def test_same_lane_car_is_followed(self):
        d=self.decision(self.observe(y=0.))
        self.assertEqual(d.state,'TAILING');self.assertEqual(d.reason,'s5_follow_vehicle')

    def test_short_dropout_preserves_geometry_and_lane(self):
        o=self.observe()
        predicted=self.node._update_tracks([],1.5)[0]
        self.assertTrue(predicted['predicted']);self.assertTrue(predicted['lane_stable'])
        self.assertEqual(predicted['length'],o['length'])
        self.assertEqual(self.decision(predicted).state,'RACELINE')
        self.assertEqual(self.node._update_tracks([],2.),[])

    def test_small_crosser_is_not_excluded_as_vehicle(self):
        o=self.observe(y=0.,length=.5)
        self.assertLess(o['length'],2.5)
        self.assertNotEqual(self.decision(o).state,'RACELINE')

if __name__=='__main__':unittest.main()
