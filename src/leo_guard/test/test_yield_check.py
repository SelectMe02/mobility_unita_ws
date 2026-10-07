import unittest
import numpy as np
from leo_guard.yield_check import corridor_observation


class YieldCheckTests(unittest.TestCase):
    def ground(self):
        return np.array([(x,y,-.344) for x in np.arange(3,35,.3) for y in np.arange(-3,3,.3)])
    def route(self):return np.c_[np.arange(0,36,2),np.zeros(18)]
    def test_observed_ground_clear_but_empty_cloud_never_clear(self):
        self.assertTrue(corridor_observation(self.ground(),self.route())['clear'])
        self.assertFalse(corridor_observation(np.empty((0,3)),self.route())['clear'])
    def test_one_small_elevated_return_blocks_corridor(self):
        p=np.r_[self.ground(),[[12.,.5,.1]]]
        s=corridor_observation(p,self.route())
        self.assertFalse(s['clear']);self.assertAlmostEqual(s['obstacle_distance'],12.)
    def test_roadside_pole_outside_corridor_does_not_block(self):
        p=np.r_[self.ground(),[[12.,4.,1.]]]
        self.assertTrue(corridor_observation(p,self.route())['clear'])
    def test_missing_ground_band_holds_even_without_obstacle(self):
        p=self.ground();p=p[(p[:,0]<12)|(p[:,0]>=20)]
        self.assertFalse(corridor_observation(p,self.route())['valid'])

    def test_duplicate_waypoints_at_link_joins_are_supported(self):
        r=self.route();r=np.insert(r,5,r[5],axis=0)
        self.assertTrue(corridor_observation(self.ground(),r)['clear'])
    def test_unobserved_ground_inside_twenty_metre_horizon_blocks(self):
        p=self.ground();p=p[p[:,0]<12]
        self.assertFalse(corridor_observation(p,self.route())['valid'])

    def test_body_pitch_plane_with_obstacle_outliers(self):
        p=self.ground();p[:,2]+=.07*p[:,0]-.03*p[:,1]
        self.assertTrue(corridor_observation(p,self.route())['clear'])
        p=np.r_[p,[[12.,.5,2.],[10.,.2,1.]]]
        self.assertFalse(corridor_observation(p,self.route())['clear'])
    def test_implausible_ground_slope_cannot_grant_clearance(self):
        p=self.ground();p[:,2]+=.4*p[:,0]
        self.assertFalse(corridor_observation(p,self.route())['clear'])
