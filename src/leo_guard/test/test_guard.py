import unittest
import numpy as np
from leo_guard.core import SignalLatch, signal_permission, swept_clearance, speed_for_clearance, extract_scene, constrain_command

G={'wheelbase':3.,'front_extent':3.845,'rear_overhang':.79,'half_width':.946}
def lane(width=3.5,xmin=-2.,xmax=65.):
    return dict(valid=True,x_min=xmin,x_max=xmax,left=dict(coeff=[0,0,width/2],uncertainty=.1),
                right=dict(coeff=[0,0,-width/2],uncertainty=.1),obstacle_distance=None)
def light(label,confidence=.9):return dict(label=label,confidence=confidence,bbox_normalized=[.4,.2,.5,.3])

class GuardTest(unittest.TestCase):
    def test_signal_direction(self):
        self.assertFalse(signal_permission([light('traffic_light_green')],'left',[0,0,1,1])[0])
        self.assertTrue(signal_permission([light('traffic_light_left')],'left',[0,0,1,1])[0])
        self.assertFalse(signal_permission([light('traffic_light_left')],'straight',[0,0,1,1])[0])
        self.assertTrue(signal_permission([light('traffic_light_green_left')],'straight',[0,0,1,1])[0])
    def test_conflicts_and_unknown(self):
        for d in ([],[light('traffic_light_yellow')],[light('traffic_light_red')],
                  [light('traffic_light_green'),light('traffic_light_red')],
                  [light('traffic_light_green',.6)],[light('unknown')]):
            self.assertFalse(signal_permission(d,'straight',[0,0,1,1])[0])
    def test_single_frame_never_releases(self):
        s=SignalLatch(.6,1.)
        for t in (0,.2,.6,.9):self.assertFalse(s.update(True,'left',1,t))
    def test_red_cancels_stable_green(self):
        s=SignalLatch(.6,.4)
        self.assertFalse(s.update(True,'green',1,0))
        self.assertFalse(s.update(True,'green',2,.3))
        self.assertTrue(s.update(True,'green',3,.6))
        self.assertFalse(s.update(False,'red',4,.61))
        self.assertFalse(s.update(True,'green',5,.62))
    def test_stale_latch_and_maneuver_change(self):
        s=SignalLatch(.6,.4)
        for i,t in enumerate((0,.3,.6)):s.update(True,'straight',i,t)
        self.assertFalse(s.update(True,'straight',4,1.1))
        self.assertFalse(s.update(True,'left',5,1.2))
    def test_body_and_high_speed_horizon(self):
        self.assertTrue(swept_clearance(lane(),60/3.6,0,G,.4,3,.25)[0])
        self.assertFalse(swept_clearance(lane(xmax=35),100/3.6,0,G,.4,3,.25)[0])
        self.assertFalse(swept_clearance(lane(width=2.4),0,0,G,.4,3,.25)[0])
        self.assertFalse(swept_clearance(lane(xmin=5),0,0,G,.4,3,.25)[0])
    def test_outer_front_corner(self):
        self.assertFalse(swept_clearance(lane(),10,.2,G,.4,3,.25)[0])
    def test_braking_bound(self):
        for distance in (0,1,10,40):
            v=speed_for_clearance(distance,3,.4)
            self.assertAlmostEqual(v*.4+v*v/6,distance)
    def test_no_context_or_validation(self):
        context=dict(state='clear',association_verified=True)
        self.assertFalse(constrain_command(lane(),0,0,G,context,[],False)[0])
        self.assertFalse(constrain_command(lane(),0,0,G,None,[],True)[0])
    def test_stop_line_front_extent(self):
        context=dict(state='approach',association_verified=True,release_stable=False,stop_distance=4.)
        self.assertFalse(constrain_command(lane(),0,0,G,context,[],True)[0])
        context['stop_distance']=15.
        permitted,cap,_=constrain_command(lane(),0,0,G,context,[],True)
        self.assertTrue(permitted);self.assertLess(cap,8.)
    def test_obstacle_does_not_authorize_line_crossing(self):
        scene=lane();scene['obstacle_distance']=4.
        self.assertFalse(constrain_command(scene,0,0,G,dict(state='clear',association_verified=True),[],True)[0])
    def test_observed_pair_without_synthetic_boundary(self):
        rng=np.random.RandomState(9)
        x=rng.uniform(-5,30,2000);y=rng.uniform(-5,5,2000)
        ground=np.c_[x,y,np.zeros(2000),np.full(2000,10.)]
        x=np.linspace(-5,30,200)
        left=np.c_[x,1.75+np.zeros(200),np.zeros(200),np.full(200,110.)]
        right=left.copy();right[:,1]=-1.75
        scene=extract_scene(np.r_[ground,left,right],0)
        self.assertTrue(scene['valid']);self.assertGreater(scene['x_max'],29)
        self.assertFalse(extract_scene(np.r_[ground,left],0)['valid'])
    def test_missing_intensity_and_nan(self):
        with self.assertRaises(ValueError):extract_scene(np.zeros((200,3)),0)
        self.assertFalse(extract_scene(np.full((200,4),np.nan),0)['valid'])

    def test_outer_dense_marking_cannot_hide_center_line(self):
        x=np.linspace(-5,30,200)
        paint=np.r_[np.c_[x,np.full(200,3.5),np.zeros(200),np.full(200,110)],
                    np.c_[x,np.full(200,-1.75),np.zeros(200),np.full(200,110)],
                    np.c_[x,np.full(200,.3),np.zeros(200),np.full(200,110)]]
        scene=extract_scene(paint,0.)
        self.assertFalse(scene['valid']);self.assertIn('inside',scene['reason'])

if __name__=='__main__':unittest.main()
