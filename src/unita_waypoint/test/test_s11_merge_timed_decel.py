import unittest
import test_highway_planner as f

class TimedMergeDecelTests(unittest.TestCase):
    def setUp(self):
        self.p=f.planner(s11_forward_only=True)

    def test_feasible_distant_merge_uses_gradual_target(self):
        immediate=80./5.*3.6
        cap=self.p._s11_merge_timed_cap(80.,20.,5.,immediate)
        self.assertGreater(cap,immediate)
        self.assertLess(cap,72.)

    def test_near_entry_and_unknown_clearance_keep_original_braking(self):
        self.assertEqual(self.p._s11_merge_timed_cap(20.,20.,3.,30.),30.)
        self.assertEqual(self.p._s11_merge_timed_cap(80.,20.,float('inf'),30.),30.)

    def test_infeasible_yield_does_not_delay_braking(self):
        self.assertEqual(self.p._s11_merge_timed_cap(40.,23.,6.,30.),30.)

    def test_floor_speed_does_not_accelerate_toward_conflict(self):
        self.assertEqual(self.p._s11_merge_timed_cap(60.,30./3.6,5.,30.),30.)

    def test_spatial_response_envelope_bounds_target(self):
        import math
        cap=self.p._s11_merge_timed_cap(80.,20.,5.,57.6)
        envelope=math.sqrt((30./3.6)**2+6.*(80.-20.*.6-5.))*3.6
        self.assertLessEqual(cap,envelope)

if __name__=='__main__':unittest.main()
