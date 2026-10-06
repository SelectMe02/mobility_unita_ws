import unittest
import numpy as np
from leo_guard.core import PaintAccumulator,project_ground_pixels

class CameraTemporalTest(unittest.TestCase):
    def test_motion_compensation_and_expiration(self):
        a=PaintAccumulator();p=np.array([[10.,2.,0.,110.]])
        a.add(p,1.,1.,(0.,0.,0.))
        self.assertTrue(np.allclose(a.current(1.1,(2.,0.,0.))[0,:2],[8.,2.]))
        self.assertTrue(np.allclose(a.current(1.1,(0.,0.,np.pi/2))[0,:2],[2.,-10.]))
        self.assertEqual(len(a.current(2.,(0.,0.,0.))),0)
    def test_pose_skew_and_clock_reset(self):
        a=PaintAccumulator();p=np.zeros((1,4))
        with self.assertRaises(ValueError):a.add(p,1.,1.1,(0,0,0))
        a.add(p,1.,1.,(0,0,0))
        with self.assertRaises(ValueError):a.add(p,.5,.5,(0,0,0))
        self.assertEqual(len(a.current(1.,(0,0,0))),0)
    def test_ground_projection_and_horizon(self):
        # Optical right -> vehicle right, optical down -> groundward.
        k=np.array([[100,0,50],[0,100,50],[0,0,1.]])
        t=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,1.],[0,0,0,1.]])
        p=project_ground_pixels(np.array([[50.,60.],[60.,60.],[50.,40.],[50.,50.]]),k,t)
        self.assertEqual(len(p),2);self.assertTrue(np.allclose(p,[[10,0,0],[10,-1,0]]))
    def test_bad_calibration_rejected(self):
        with self.assertRaises(ValueError):project_ground_pixels(np.zeros((2,2)),np.eye(3),np.eye(3))

if __name__=='__main__':unittest.main()
