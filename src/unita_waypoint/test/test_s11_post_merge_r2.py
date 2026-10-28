import unittest
from unittest.mock import MagicMock
import test_highway_planner as f
import test_s11_entry_slowdown as e

class PostMergeR2Tests(unittest.TestCase):
    def test_small_positive_speed_margin_allowed_only_after_r1_merge(self):
        p=f.planner(s11_forward_only=True)
        p._idm_cap=MagicMock(return_value=31.8)
        p.plan(60,(60.,0.),30./3.6,[],desired_speed_mps=100./3.6)
        self.assertIsNone(p.transition)
        p.s11_r1_merge_complete=True
        d=p.plan(60,(60.,0.),30./3.6,[],desired_speed_mps=100./3.6)
        self.assertEqual(p.transition['target'],2)
        self.assertAlmostEqual(d.speed_cap_kmh,31.8)

    def test_deceleration_requirement_still_blocks(self):
        p=f.planner(s11_forward_only=True)
        p.s11_r1_merge_complete=True
        p._idm_cap=MagicMock(return_value=29.)
        p.plan(60,(60.,0.),30./3.6,[],desired_speed_mps=100./3.6)
        self.assertIsNone(p.transition)

    def test_close_target_vehicle_still_blocks_after_merge(self):
        p=f.planner(s11_forward_only=True)
        p.s11_r1_merge_complete=True
        p.plan(60,(60.,0.),30./3.6,[f.car(60.,2,30./3.6)],desired_speed_mps=100./3.6)
        self.assertIsNone(p.transition)

    def test_r2_to_r3_keeps_five_kmh_requirement(self):
        p=f.planner(s11_forward_only=True)
        p.s11_r1_merge_complete=True
        p._idm_cap=MagicMock(return_value=31.8)
        p.plan(60,(60.,3.6),30./3.6,[],desired_speed_mps=100./3.6)
        self.assertIsNone(p.transition)

    def test_real_entry_sets_completion_only_after_join(self):
        p=e.planner(merge_control=True)
        p.plan(60,e.point(0.),10.,[],desired_speed_mps=25.,observation_stamp=1.)
        self.assertFalse(p.s11_r1_merge_complete)
        self.assertIsNone(p.transition)
        p.plan(100,e.point(40.),10.,[],desired_speed_mps=25.,observation_stamp=2.)
        self.assertTrue(p.s11_r1_merge_complete)
        self.assertEqual(p.transition['target'],2)

if __name__=='__main__':unittest.main()
