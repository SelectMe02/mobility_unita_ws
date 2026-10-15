"""Exercise the actual command gate with ROS messages, without vehicle motion."""
import importlib.util
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from morai_msgs.msg import CtrlCmd
from leo_guard.traffic_local import RouteProgress, TrafficDecision

spec = importlib.util.spec_from_file_location('signal_node_test', Path(__file__).parents[1] / 'scripts/signal_guard_node_local.py')
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


class SignalNodeTests(unittest.TestCase):
    def guard(self):
        g = node.SignalGuard.__new__(node.SignalGuard)
        g.lock = threading.RLock(); g.inputs = {}; g.armed = True
        g.pose_fault_since = None; g.pose_fault_grace_s = 2.
        g.front_axle = 3.; g.half_width = .946
        g.route = RouteProgress([[0, 0], [100, 0], [200, 0]])
        stop = dict(id='right', s=50., clearance_m=30., maneuver='right_unprotected',
                    controlled=True, reviewed=True, signal_roi=[0, 0, 1, 1])
        g.decision = TrafficDecision([stop], 200.)
        for name in ('pub', 'cap', 'status', 'camera'): setattr(g, name, MagicMock())
        return g

    def step(self, g, t, speed=0.):
        cmd = CtrlCmd(); cmd.longlCmdType = 1; cmd.accel = .5
        signals = dict(valid=True, frame=t, deadline=t+1., detections=[
            dict(label='traffic_light_green', confidence=.9, bbox_normalized=[.2, .2, .8, .8])])
        for key, value in dict(pose=(45., 0., 0.), valid=True, speed=speed,
                               command=cmd, signals=signals).items():
            g.inputs[key] = (value, t)
        with patch.object(node.time, 'monotonic', return_value=t), patch.object(node.rospy, 'loginfo_throttle'):
            g.tick(None)
        return g.pub.publish.call_args[0][0]

    def release(self, g):
        for i in range(9): self.step(g, i*.25)

    def test_right_green_goes_without_a_full_stop(self):
        # The GPS route turns on green; route obstacles are the avoidance planner's.
        g = self.guard()
        for i in range(9): output = self.step(g, i*.25, speed=3.)
        self.assertGreater(output.accel, 0.)
        self.assertEqual(output.brake, 0.)
        self.assertAlmostEqual(g.cap.publish.call_args[0][0].data, 18.)

    def test_arm_reports_missing_input_names(self):
        g = self.guard()
        g.inputs['speed'] = (0., 1.)
        with patch.object(node.time, 'monotonic', return_value=1.):
            result = g.set_armed(type('Request', (), {'data': True})())
        self.assertFalse(result.success)
        self.assertIn('driving_valid', result.message)
        self.assertIn('driving_pose', result.message)
        self.assertNotIn('ego_speed', result.message)

    def test_short_pose_gap_brakes_and_preserves_arm_then_times_out(self):
        g = self.guard(); self.step(g, 1.)
        g.inputs['valid'] = (False,1.1)
        with patch.object(node.time,'monotonic',return_value=1.1):g.tick(None)
        status=json.loads(g.status.publish.call_args[0][0].data)
        self.assertTrue(g.armed);self.assertEqual(status['state'],'hold')
        self.assertEqual(g.pub.publish.call_args[0][0].brake,1.)
        self.step(g,1.2)
        self.assertTrue(g.armed)
        g.inputs['valid'] = (False,2.)
        with patch.object(node.time,'monotonic',return_value=2.):g.tick(None)
        g.inputs['valid'] = (False,4.1)
        with patch.object(node.time,'monotonic',return_value=4.1):g.tick(None)
        self.assertFalse(g.armed)


if __name__ == '__main__': unittest.main()
