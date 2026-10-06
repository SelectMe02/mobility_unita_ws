"""ROS message tests without launching a master or permitting simulator motion."""
import importlib.util
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import MagicMock,patch
from morai_msgs.msg import CtrlCmd
from std_msgs.msg import String
from leo_guard.core import SignalLatch
spec=importlib.util.spec_from_file_location('guard_node',Path(__file__).parents[1]/'scripts/guard_node.py')
node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)

class NodeTest(unittest.TestCase):
    def guard(self):
        g=node.Guard.__new__(node.Guard);g.lock=threading.Lock();g.inputs={}
        g.timeout=.35;g.context_timeout=.35;g.signal_timeout=1.;g.approved=True
        g.geometry=dict(wheelbase=3.,front_extent=3.845,rear_overhang=.79,half_width=.946)
        g.latency=.4;g.decel=3.;g.margin=.25;g.roi=[0,0,1,1];g.latch=SignalLatch()
        g.pub=MagicMock();g.status=MagicMock();g.cap=MagicMock()
        now=time.monotonic();cmd=CtrlCmd();cmd.longlCmdType=1;cmd.accel=.5
        scene=dict(valid=True,x_min=-2.,x_max=65.,left=dict(coeff=[0,0,1.75],uncertainty=.1),
                   right=dict(coeff=[0,0,-1.75],uncertainty=.1),obstacle_distance=None)
        for key,value in dict(command=cmd,speed=0.,lane=scene,localization=True,
                              traffic=dict(state='clear',association_verified=True,_source_deadline=now+1.)).items():
            g.inputs[key]=(value,now)
        return g
    def test_missing_each_required_input_brakes(self):
        for key in ('command','speed','lane','localization','traffic'):
            g=self.guard();g.inputs.pop(key);g.tick(None)
            self.assertEqual(g.pub.publish.call_args[0][0].brake,1.,key)
            self.assertEqual(g.cap.publish.call_args[0][0].data,0.,key)
    def test_stale_each_required_input_brakes(self):
        for key in ('command','speed','lane','localization','traffic'):
            g=self.guard();value,stamp=g.inputs[key];g.inputs[key]=(value,stamp-2.)
            g.tick(None);self.assertEqual(g.pub.publish.call_args[0][0].brake,1.,key)
    def test_clear_verified_corridor_passes_control(self):
        g=self.guard();g.tick(None);self.assertEqual(g.pub.publish.call_args[0][0].accel,.5)
    def test_unverified_calibration_never_moves(self):
        g=self.guard();g.approved=False;g.tick(None)
        self.assertEqual(g.pub.publish.call_args[0][0].accel,0.)
        self.assertEqual(g.pub.publish.call_args[0][0].brake,1.)
    def test_red_or_missing_signal_stops_before_front_extent(self):
        g=self.guard();g.inputs['traffic'][0].update(state='approach',stop_distance=4.,maneuver='left')
        g.tick(None);self.assertEqual(g.pub.publish.call_args[0][0].brake,1.)
    def test_source_age_cannot_be_reset_by_publication(self):
        g=self.guard()
        with patch.object(node.rospy.Time,'now',return_value=MagicMock(to_sec=lambda:100.)):
            g.json_input('signals',String(data=json.dumps(dict(stamp=90.,valid=True,frame=1,detections=[]))))
        self.assertIsNone(g.inputs['signals'][0])
    def test_nan_command_brakes(self):
        g=self.guard();g.inputs['command'][0].accel=float('nan');g.tick(None)
        self.assertEqual(g.pub.publish.call_args[0][0].brake,1.)

if __name__=='__main__':unittest.main()
