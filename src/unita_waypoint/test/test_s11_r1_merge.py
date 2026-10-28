import unittest
import test_s11_entry_slowdown as e
import test_highway_planner as f

class R1MergeTests(unittest.TestCase):
    def plan(self,p,along=0.,obs=(),stamp=1.,speed=20.):
        return p.plan(int(along+60),e.point(along),speed,list(obs),
                      desired_speed_mps=25.,observation_stamp=stamp)

    def test_clear_entry_keeps_speed_and_r1_then_releases_change(self):
        p=e.planner(merge_control=True)
        d=self.plan(p)
        self.assertIsNone(p.transition)
        self.assertEqual(d.reason,'s11_r1_merge_align')
        self.assertGreaterEqual(d.speed_cap_kmh,90.)
        d=self.plan(p,40.,stamp=2.)
        self.assertIsNotNone(p.transition)
        self.assertEqual(p.transition['target'],2)

    def test_near_merge_vehicle_yields_even_if_label_says_r2(self):
        p=e.planner(merge_control=True);o=e.npc(track_id=7,raceline=2,lane_stable=True,lane_sector=11)
        d=self.plan(p,obs=[o])
        self.assertLessEqual(d.speed_cap_kmh,30.)
        self.assertEqual(d.reason,'s11_r1_merge_yield')
        self.assertIsNone(p.transition)

    def test_recent_dropout_keeps_yield_but_expires(self):
        p=e.planner(merge_control=True);o=e.npc(track_id=7)
        self.plan(p,obs=[o])
        d=self.plan(p,stamp=1.5)
        self.assertEqual(d.speed_cap_kmh,30.)
        self.assertIsNone(p.transition)
        d=self.plan(p,stamp=2.)
        self.assertGreaterEqual(d.speed_cap_kmh,90.)

    def test_separated_lane_does_not_slow_entry(self):
        p=e.planner(merge_control=True)
        for side in (3.6,7.2):
            d=self.plan(p,obs=[e.npc(side,track_id=7)])
            self.assertGreaterEqual(d.speed_cap_kmh,90.)

    def test_outside_geographic_zone_preserves_planner(self):
        p=f.planner(s11_forward_only=True)
        p.plan(60,(60.,0.),20.,[],desired_speed_mps=25.)
        self.assertIsNotNone(p.transition)

    def test_existing_commitment_not_reversed(self):
        p=e.planner(merge_control=True)
        self.plan(p,40.,stamp=1.)
        tr=p.transition
        self.assertIsNotNone(tr)
        # Local controller must not override an already approved transition.
        self.assertIsNone(p._s11_r1_merge(60,e.point(0.),20.,25.,p._forward(60,80.),0.,[],2.))
        self.assertIs(p.transition,tr)

if __name__=='__main__':unittest.main()
