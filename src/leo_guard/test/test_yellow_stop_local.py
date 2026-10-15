import math
import unittest

from leo_guard.yellow_stop import (decide_signal, nearest_ahead,
                                   project_stopline, stopping_speed_mps,
                                   camera_signal_state)
from leo_guard.traffic_local import TrafficDecision


class YellowStopTests(unittest.TestCase):
    def test_stopline_distance_follows_the_route_on_a_curve(self):
        # Ego heads 60 degrees off the line's straight bearing on a bend; the
        # surveyed line is 20 m ahead along the route.
        stop = dict(id='node1', s=60., traffic_light_id='light1',
                    maneuver='straight', controlled=True, reviewed=True,
                    signal_roi=[.2, .1, .8, .6], clearance_m=15.)
        controller = TrafficDecision([stop], 1000., stoplines=[
            dict(idx='node1', x=10., y=17., traffic_light_id='light1')])
        cap, detail = controller.decide(40., None, 1., pose=(0., 0., 0.),
                                        speed_mps=5., ros_now=1.)
        self.assertAlmostEqual(detail['d_m'], 20. - 3.)
        self.assertEqual(detail['state'], 'approach')
        self.assertGreater(cap, 0.)

    def test_curved_approach_uses_route_distance_before_stopline_projection(self):
        stop = dict(id='node1', s=60., traffic_light_id='light1',
                    maneuver='straight', controlled=True, reviewed=True,
                    signal_roi=[.2, .1, .8, .6], clearance_m=15.,
                    signal_activation_distance_m=15.)
        controller = TrafficDecision([stop], 1000., stoplines=[
            dict(idx='node1', x=60., y=10., traffic_light_id='light1')])
        cap, detail = controller.decide(0., None, 1.,
                                        pose=(0., 0., 0.), speed_mps=0., ros_now=1.)
        self.assertGreater(cap, 0.)
        self.assertEqual(detail['state'], 'approach')
        self.assertNotEqual(detail['reason'], 'outside stop-line lane tolerance')

    def test_projection_uses_front_bumper_and_rejects_other_lane_or_past(self):
        stops = [
            dict(idx='current', x=12., y=0., traffic_light_id='t1'),
            dict(idx='other_lane', x=8., y=4., traffic_light_id='t1'),
            dict(idx='behind', x=-1., y=0., traffic_light_id='t1'),
            dict(idx='other_light', x=6., y=0., traffic_light_id='t2'),
        ]
        self.assertEqual(project_stopline(stops[0], 2., 0., 0., 3.), (7., 0.))
        selected = nearest_ahead(stops, 2., 0., 0., 't1', front_offset_m=3.,
                                 lateral_tolerance_m=2.)
        self.assertEqual(selected[3]['idx'], 'current')
        self.assertAlmostEqual(selected[0], 7.)
        self.assertIsNone(nearest_ahead(stops, 10., 0., 0., 't1', node_idx='current',
                                        front_offset_m=3.))

    def test_yellow_go_stop_and_dilemma_stop(self):
        self.assertEqual(decide_signal('yellow', 8., 10., 2.)[:3],
                         ('go', True, False))
        self.assertEqual(decide_signal('yellow', 40., 10., 2.)[:3],
                         ('stop', False, True))
        decision, can_go, can_stop, target = decide_signal('yellow', 12., 10., .5)
        self.assertEqual((decision, can_go, can_stop), ('stop', False, False))
        self.assertTrue(math.isfinite(target))

    def test_red_overrides_possible_passage_and_profile_stops_at_buffer(self):
        decision, _, _, target = decide_signal('red', 8., 10., 2.)
        self.assertEqual(decision, 'stop')
        self.assertGreater(target, 0.)
        self.assertEqual(stopping_speed_mps(.5), 0.)
        self.assertEqual(stopping_speed_mps(-1.), 0.)

    def test_optional_acceleration_is_explicit(self):
        self.assertFalse(decide_signal('yellow', 1., 0., 2.)[1])
        self.assertTrue(decide_signal('yellow', 1., 0., 2.,
                                      use_acceleration=True)[1])

    def test_camera_region_conflict_is_unknown(self):
        roi = [.2, .1, .8, .6]
        def detection(label):
            return dict(label=label, confidence=.95,
                        bbox_normalized=[.4, .2, .6, .3])
        self.assertEqual(camera_signal_state([detection('traffic_light_yellow')], roi),
                         'yellow')
        self.assertEqual(camera_signal_state([detection('traffic_light_yellow'),
                                              detection('traffic_light_green')], roi),
                         'unknown')

    def controller(self):
        stop = dict(id='node1', s=50., traffic_light_id='light1',
                    maneuver='straight', controlled=True, reviewed=True,
                    signal_roi=[.2, .1, .8, .6], clearance_m=15.)
        return TrafficDecision([stop], 1000., stoplines=[
            dict(idx='node1', x=50., y=0., traffic_light_id='light1')])

    @staticmethod
    def signal(label, frame, stamp):
        return dict(valid=True, frame=frame, stamp=stamp, detections=[dict(
            label=label, confidence=.95, bbox_normalized=[.4, .2, .6, .3])])

    def test_green_goes_yellow_and_red_always_stop(self):
        controller = self.controller()
        cap, detail = controller.decide(42., self.signal('traffic_light_green', 1, 1.), 1.,
                                         pose=(42., 0., 0.), speed_mps=10., ros_now=1.)
        self.assertEqual(detail['decision'], 'go')
        self.assertEqual(cap, float('inf'))
        # Yellow right after green still stops, however close the line is.
        cap, detail = controller.decide(42., self.signal('traffic_light_yellow', 2, 1.1), 1.1,
                                         pose=(42., 0., 0.), speed_mps=10., ros_now=1.1)
        self.assertEqual(detail['decision'], 'stop')
        self.assertLess(cap, 10.)
        cap, detail = controller.decide(42., self.signal('traffic_light_red', 3, 1.2), 1.2,
                                         pose=(42., 0., 0.), speed_mps=10., ros_now=1.2)
        self.assertEqual(detail['decision'], 'stop')

    def test_green_crossing_continues_once_the_front_is_over_the_line(self):
        controller = self.controller()
        _, detail = controller.decide(48., self.signal('traffic_light_green', 1, 1.), 1.,
                                       pose=(48., 0., 0.), speed_mps=10., ros_now=1.)
        self.assertEqual(detail['decision'], 'go')
        _, detail = controller.decide(52., self.signal('traffic_light_yellow', 2, 1.4), 1.4,
                                       pose=(52., 0., 0.), speed_mps=10., ros_now=1.4)
        self.assertEqual(detail['state'], 'clearing')
        cap, detail = controller.decide(56., self.signal('traffic_light_red', 3, 1.8), 1.8,
                                         pose=(56., 0., 0.), speed_mps=10., ros_now=1.8)
        self.assertEqual(detail['state'], 'clearing')
        self.assertGreater(cap, 0.)

