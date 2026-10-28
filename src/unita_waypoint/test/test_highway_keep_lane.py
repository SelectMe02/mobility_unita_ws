"""Production highway lane retention, including preferred-lane candidates."""
import unittest
import test_highway_planner as f


def planner(**kwargs):
    return f.planner(highway_keep_lane_priority=True, **kwargs)


class KeepLaneTests(unittest.TestCase):
    def test_default_is_keep_lane_priority(self):
        p = f.core.LocalAvoidancePlanner([(0., 0.), (100., 0.)], [11, 11], {})
        self.assertTrue(p.highway_keep_lane_priority)

    def test_free_road_does_not_chase_preferred_lane(self):
        p = planner()
        for stamp in (0., 1., 2., 10.):
            d = p.plan(0, (0., 0.), 24., [], desired_speed_mps=f.CRUISE,
                       observation_stamp=stamp)
            self.assertIsNone(p.transition)
            self.assertEqual(d.reason, 'keep_lane_no_passing_need')

    def test_mild_following_holds_both_preferred_and_nonpreferred_lanes(self):
        for lane in (1, 2, 3):
            p = planner()
            for stamp in (0., 2., 5.):
                p.plan(0, (0., f.LANE_Y[lane]), 26., [f.car(55., lane, 25.5)],
                       desired_speed_mps=f.CRUISE, observation_stamp=stamp)
                self.assertIsNone(p.transition)

    def test_clear_moving_lead_gain_requires_continuous_time(self):
        for lane in (1, 3):
            p = planner()
            for stamp in (10., 10., 10.5, 11.1):
                p.plan(0, (0., f.LANE_Y[lane]), 20., [f.car(45., lane, 12.)],
                       desired_speed_mps=f.CRUISE, observation_stamp=stamp)
                self.assertIsNone(p.transition)
            p.plan(0, (0., f.LANE_Y[lane]), 20., [f.car(45., lane, 12.)],
                   desired_speed_mps=f.CRUISE, observation_stamp=11.3)
            self.assertIsNotNone(p.transition)

    def test_interrupted_gain_does_not_accumulate(self):
        p = planner()
        for stamp, obs in [(1., [f.car(45., 3, 12.)]), (2., []),
                           (2.1, [f.car(45., 3, 12.)]), (2.5, [f.car(45., 3, 12.)])]:
            p.plan(0, (0., 7.2), 20., obs, desired_speed_mps=f.CRUISE,
                   observation_stamp=stamp)
            self.assertIsNone(p.transition)

    def test_static_block_can_still_be_passed(self):
        p = planner()
        p.plan(0, (0., 7.2), 20., [f.car(50., 3)], desired_speed_mps=f.CRUISE)
        self.assertIsNotNone(p.transition)

    def test_ending_lane_still_merges_without_waiting_for_gain(self):
        p = planner(r3_end=120)
        p.plan(60, (60., 7.2), 15., [], desired_speed_mps=f.CRUISE,
               observation_stamp=0.)
        self.assertIsNotNone(p.transition)
        self.assertEqual(p.transition['target'], 2)
