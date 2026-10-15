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
        for name in ('pub', 'cap', 'status', 'camera', 'runtime'): setattr(g, name, MagicMock())
        g.route_max_error_m = g.blackout_route_max_error_m = 2.
        g.blackout, g.passthrough = [], []
        g.passthrough_max_error_m, g.in_passthrough = 15., False
        g.auto_arm, g.auto_arm_hold_s, g.ready_since = False, 1., None
        g.lanes = None
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
        # No fixed turn speed: the follower's Map Tuner speed is the target.
        self.assertEqual(g.cap.publish.call_args[0][0].data, float('inf'))

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

    def passthrough_command(self):
        cmd = CtrlCmd(); cmd.longlCmdType = 1; cmd.accel = .7; cmd.steering = .1
        return cmd

    def test_passthrough_segment_forwards_raceline_command_unchanged(self):
        g = self.guard()
        g.passthrough = [(0, 1)]  # whole test route
        g.inputs['passthrough_command'] = (self.passthrough_command(), 1.)
        # A red light ahead does not matter inside the pass-through segment.
        output = self.step(g, 1.)
        self.assertEqual((output.accel, output.brake, output.steering), (.7, 0., .1))
        status = json.loads(g.status.publish.call_args[0][0].data)
        self.assertEqual(status['state'], 'passthrough')
        self.assertTrue(g.armed)

    def test_passthrough_uses_wide_corridor_and_leaving_it_resets_signal_state(self):
        g = self.guard()
        g.passthrough = [(0, 0)]
        g.inputs['passthrough_command'] = (self.passthrough_command(), 1.)
        g.route = RouteProgress([[0, 0], [30, 0], [100, 0], [200, 0]])
        g.route.index = 0  # tracked into the segment (0-30 m)
        g.inputs['pose'] = ((20., 6., 0.), 1.)  # raceline lane, 6 m off the route
        for key, value in dict(valid=True, speed=5.).items(): g.inputs[key] = (value, 1.)
        with patch.object(node.time, 'monotonic', return_value=1.):
            g.tick(None)
        self.assertTrue(g.armed and g.in_passthrough)
        g.decision.last_s = 20.
        self.step(g, 1.1)  # back on the route, 45 m: normal signal gate
        self.assertTrue(g.armed)
        self.assertFalse(g.in_passthrough)

    def test_blackout_window_widens_the_corridor(self):
        g = self.guard()
        g.route_max_error_m, g.blackout_route_max_error_m = 2., 7.
        g.blackout = [(0, 1)]
        g.route.index = 0
        self.assertEqual(g.corridor(), 7.)
        g.blackout = []
        self.assertEqual(g.corridor(), 2.)

    def test_auto_arm_after_inputs_stay_fresh(self):
        g = self.guard(); g.armed = False; g.auto_arm = True
        for t in (1., 1.5, 2.1):
            for key, value in dict(pose=(45., 0., 0.), valid=True, speed=0.).items():
                g.inputs[key] = (value, t)
            with patch.object(node.time, 'monotonic', return_value=t), \
                    patch.object(node.rospy, 'loginfo'):
                g.try_auto_arm(t)
            self.assertEqual(g.armed, t >= 2.)

    def test_corridor_is_measured_from_the_lane_being_driven(self):
        g = self.guard(); g.route_max_error_m = 3.
        on_r2 = (50., -3.7, 0.)  # saved Raceline 2, 3.7 m right of Raceline 1
        with self.assertRaises(ValueError):
            g.track(on_r2, 1.)
        g.route.index = None
        g.lanes = node.cKDTree([[x, -3.6] for x in range(0, 200)])
        progress, error = g.track(on_r2, 1.)
        self.assertAlmostEqual(progress, 50.); self.assertLess(error, .2)
        g.route.index = None
        with self.assertRaises(ValueError):  # beyond Raceline 2 as well
            g.track((50., -8., 0.), 1.)

    def test_route_avoidance_offset_counts_as_driven_lane(self):
        g = self.guard(); g.route_max_error_m = 2.
        g.inputs['route_offset'] = (3.5, 1.)  # passing one lane to the left
        progress, error = g.track((50., 3.6, 0.), 1.)
        self.assertAlmostEqual(error, .1)
        g.route.index = None
        g.inputs['route_offset'] = (3.5, 0.)  # stale offset: plain Raceline 1 corridor
        with self.assertRaises(ValueError):
            g.track((50., 3.6, 0.), 1.)


if __name__ == '__main__': unittest.main()
