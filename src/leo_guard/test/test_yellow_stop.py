import math
import unittest

from leo_guard.yellow_stop import (decide_signal, nearest_ahead,
                                   project_stopline, stopping_speed_mps,
                                   camera_signal_state)
from leo_guard.traffic import TrafficDecision


class YellowStopTests(unittest.TestCase):
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

    def test_mapped_yellow_requires_recent_green_and_lane_match(self):
        stop = dict(id='node1', s=50., traffic_light_id='light1',
                    maneuver='straight', controlled=True, reviewed=True,
                    signal_roi=[.2, .1, .8, .6], clearance_m=15.)
        nodes = [dict(idx='node1', x=50., y=0., traffic_light_id='light1')]
        controller = TrafficDecision([stop], 1000., stoplines=nodes)
        def signal(label, frame, stamp):
            return dict(valid=True, frame=frame, stamp=stamp, detections=[dict(
                label=label, confidence=.95,
                bbox_normalized=[.4, .2, .6, .3])])
        yellow = signal('traffic_light_yellow', 1, 1.)
        cap, detail = controller.decide(42., yellow, 1.,
                                         pose=(42., 0., 0.), speed_mps=10., ros_now=1.)
        self.assertEqual(detail['decision'], 'stop')
        for index in range(4):
            now = 2. + index * .25
            controller.decide(42., signal('traffic_light_green', index+2, now), now,
                              pose=(42., 0., 0.), speed_mps=10., ros_now=now)
        cap, detail = controller.decide(42., signal('traffic_light_yellow', 7, 3.), 3.,
                                         pose=(42., 0., 0.), speed_mps=10., ros_now=3.)
        self.assertEqual(detail['decision'], 'go')
        self.assertTrue(detail['can_go'])
        self.assertGreater(cap, 0.)
        cap, detail = controller.decide(42., signal('traffic_light_red', 8, 3.1), 3.1,
                                         pose=(42., 0., 0.), speed_mps=10., ros_now=3.1)
        self.assertEqual(detail['decision'], 'stop')
        self.assertIsNone(controller.approach_go_selected)
        cap, detail = controller.decide(42., yellow, 3.2,
                                         pose=(42., 4., 0.), speed_mps=10., ros_now=3.2)
        self.assertEqual(cap, 0.)
        self.assertIn('lane tolerance', detail['reason'])

    def test_yellow_go_clears_only_after_front_crosses(self):
        stop = dict(id='node1', s=50., traffic_light_id='light1',
                    maneuver='straight', controlled=True, reviewed=True,
                    signal_roi=[.2, .1, .8, .6], clearance_m=15.)
        controller = TrafficDecision([stop], 1000., stoplines=[
            dict(idx='node1', x=50., y=0., traffic_light_id='light1')])
        def signal(label, frame, stamp):
            return dict(valid=True, frame=frame, stamp=stamp, detections=[dict(
                label=label, confidence=.95,
                bbox_normalized=[.4, .2, .6, .3])])
        for index in range(4):
            now = 1. + index * .2
            controller.decide(42., signal('traffic_light_green', index+1, now), now,
                              pose=(42., 0., 0.), speed_mps=10., ros_now=now)
        _, detail = controller.decide(42., signal('traffic_light_yellow', 5, 1.8), 1.8,
                                       pose=(42., 0., 0.), speed_mps=10., ros_now=1.8)
        self.assertEqual(detail['decision'], 'go')
        self.assertIsNone(controller.committed)
        _, detail = controller.decide(50., signal('traffic_light_red', 6, 2.), 2.,
                                       pose=(50., 0., 0.), speed_mps=10., ros_now=2.)
        self.assertEqual(detail['state'], 'clearing')
        self.assertIsNotNone(controller.committed)

    def test_yellow_respects_leo_camera_and_head_size_review(self):
        stop = dict(id='node1', s=50., traffic_light_id='light1',
                    maneuver='straight', controlled=True, reviewed=True,
                    signal_camera='up', signal_min_width=.1,
                    signal_activation_distance_m=12.,
                    signal_roi=[.2, .1, .8, .6], clearance_m=15.)
        controller = TrafficDecision([stop], 1000., stoplines=[
            dict(idx='node1', x=50., y=0., traffic_light_id='light1')])

        def signal(label, frame, stamp, camera='up', width=.2):
            return dict(valid=True, frame=frame, stamp=stamp, camera=camera,
                        detections=[dict(label=label, confidence=.95,
                            bbox_normalized=[.5-width/2, .2, .5+width/2, .3])])

        for i in range(4):
            now = 1. + i*.2
            controller.decide(42., signal('traffic_light_green', i, now,
                                           camera='front'), now,
                              pose=(42., 0., 0.), speed_mps=10., ros_now=now)
        _, detail = controller.decide(42., signal('traffic_light_yellow', 4, 1.8),
                                       1.8, pose=(42., 0., 0.), speed_mps=10.,
                                       ros_now=1.8)
        self.assertEqual(detail['decision'], 'stop')

        for i in range(4):
            now = 2. + i*.2
            controller.decide(42., signal('traffic_light_green', 5+i, now,
                                           width=.02), now,
                              pose=(42., 0., 0.), speed_mps=10., ros_now=now)
        _, detail = controller.decide(42., signal('traffic_light_yellow', 9, 2.8),
                                       2.8, pose=(42., 0., 0.), speed_mps=10.,
                                       ros_now=2.8)
        self.assertEqual(detail['decision'], 'stop')

        for i in range(4):
            now = 3. + i*.2
            controller.decide(42., signal('traffic_light_green', 10+i, now), now,
                              pose=(42., 0., 0.), speed_mps=10., ros_now=now)
        _, detail = controller.decide(42., signal('traffic_light_yellow', 14, 3.8),
                                       3.8, pose=(42., 0., 0.), speed_mps=10.,
                                       ros_now=3.8)
        self.assertEqual(detail['decision'], 'go')


if __name__ == '__main__':
    unittest.main()
