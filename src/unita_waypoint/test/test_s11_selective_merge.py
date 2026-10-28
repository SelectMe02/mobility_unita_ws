import unittest
import test_highway_planner as f
import test_s11_entry_slowdown as e

class SelectiveMergeTests(unittest.TestCase):
    def test_empty_entry_waits_for_roi_and_three_fresh_scans_without_speed_cap(self):
        p=e.planner()
        self.assertFalse(p._s11_entry_observation_ready(e.point(-45.),1.))
        for stamp in (1.1,1.1,1.2):
            self.assertFalse(p._s11_entry_observation_ready(e.point(-20.),stamp))
        self.assertTrue(p._s11_entry_observation_ready(e.point(-20.),1.4))
        self.assertFalse(p._s11_entry_observation_ready(e.point(-20.),2.))
        p=e.planner()
        d=p.plan(30,e.point(-30.),15.,[],desired_speed_mps=25.,observation_stamp=1.)
        self.assertIsNone(p.transition)
        self.assertEqual(d.reason,'s11_wait_entry_observation')
        self.assertGreaterEqual(d.speed_cap_kmh,15.*3.6)

    def test_entry_predicts_confirmed_vehicle_at_arrival_not_unrelated_lane(self):
        o=e.npc(track_id=1)
        o.x,o.y=e.point(f.core.LENGTH-25.)
        o.vx,o.vy=(a*10. for a in f.core.TANGENT)
        self.assertTrue(f.core.EntryGuard().update(11,1,e.point(0.),[o],10.))
        o.x,o.y=e.point(f.core.LENGTH-25.,3.6)
        self.assertFalse(f.core.EntryGuard().update(11,1,e.point(0.),[o],10.))
        o.x,o.y=e.point(f.core.LENGTH+25.)
        self.assertFalse(f.core.EntryGuard().update(11,1,e.point(0.),[o],10.))

    def test_far_from_entry_no_new_scan_wait(self):
        p=f.planner(s11_forward_only=True)
        d=p.plan(60,(60.,0.),15.,[],desired_speed_mps=25.)
        self.assertIsNotNone(p.transition)

    def reason(self, obstacle):
        p=f.planner(s11_forward_only=True)
        return p._s11_close_merge_reason(60,(60.,3.6),3,{3:(60.,7.2)},10.,
                                        dict(length=45.),[obstacle])

    def test_close_unknown_rear_waits_but_distant_one_does_not(self):
        o=f.car(46.,3);o.kind='unknown';o.velocity_reliable=False
        self.assertEqual(self.reason(o),'s11_wait_rear_motion')
        o.x=5.
        self.assertEqual(self.reason(o),'')

    def test_close_fast_rear_needs_gap_through_completion(self):
        self.assertEqual(self.reason(f.car(44.,3,16.5)),'s11_wait_rear_margin')
        self.assertEqual(self.reason(f.car(44.,3,10.)), '')

    def test_side_body_checked_even_with_wrong_lane_label(self):
        o=f.car(60.,3,10.);o.raceline=1;o.lane_stable=False
        self.assertEqual(self.reason(o),'s11_wait_side_clearance')
        o.y=0.
        self.assertEqual(self.reason(o),'')

if __name__=='__main__':unittest.main()
