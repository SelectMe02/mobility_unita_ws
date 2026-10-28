import unittest
import test_s11_entry_slowdown as f


class RiskReleaseTests(unittest.TestCase):
    def test_clear_lane_releases_before_entry_zone_ends(self):
        p=f.planner();o=f.npc(track_id=7)
        p.plan(60,f.point(0.),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=1.)
        for stamp in (1.1,1.2):
            d=p.plan(60,f.point(0.,3.6),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=stamp)
            self.assertEqual(d.speed_cap_kmh,30.)
        d=p.plan(60,f.point(0.,3.6),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=1.3)
        self.assertFalse(p.s11_entry_guard.active)
        self.assertGreater(d.speed_cap_kmh,30.)

    def test_approved_change_releases_on_clear_remaining_path(self):
        p=f.planner();o=f.npc(track_id=7)
        for stamp in (.5,.7,1.):
            p.plan(60,f.point(0.),30./3.6,[],desired_speed_mps=100./3.6,observation_stamp=stamp)
        self.assertIsNotNone(p.transition)
        target=p.transition['target']
        p.plan(61,f.point(1.),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=1.1)
        self.assertTrue(p.s11_entry_guard.active)
        # Positively observe the same vehicle well behind the remaining path.
        o.x,o.y=f.point(-30.)
        for stamp in (1.2,1.4):
            d=p.plan(62,f.point(2.),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=stamp)
        self.assertFalse(p.s11_entry_guard.active)
        self.assertEqual(p.transition['target'],target)
        self.assertGreater(d.speed_cap_kmh,30.)
        o.x,o.y=f.point(f.f.core.LENGTH)
        d=p.plan(62,f.point(2.),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=1.5)
        self.assertTrue(p.s11_entry_guard.active)
        self.assertLessEqual(d.speed_cap_kmh,30.)

    def test_missing_or_predicted_blocker_does_not_release(self):
        for missing in (True,False):
            p=f.planner();o=f.npc(track_id=7)
            p.plan(60,f.point(0.),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=1.)
            o.predicted=True
            for stamp in (1.1,1.5,2.):
                d=p.plan(60,f.point(0.,3.6),30./3.6,[] if missing else [o],desired_speed_mps=100./3.6,observation_stamp=stamp)
                self.assertEqual(d.speed_cap_kmh,30.)

    def test_current_path_threat_keeps_braking(self):
        p=f.planner();o=f.npc(track_id=7)
        for stamp in (1.,1.2,1.4):
            d=p.plan(60,f.point(0.),30./3.6,[o],desired_speed_mps=100./3.6,observation_stamp=stamp)
            self.assertEqual(d.speed_cap_kmh,30.)
            self.assertTrue(p.s11_entry_guard.active)

    def test_clear_confirmation_needs_fresh_samples_and_resets_on_risk(self):
        g=f.f.core.EntryGuard();g.active=True
        self.assertFalse(g.release_if_clear(True,1.))
        self.assertFalse(g.release_if_clear(True,1.))
        self.assertFalse(g.release_if_clear(False,1.2))
        self.assertFalse(g.release_if_clear(True,1.3))
        self.assertFalse(g.release_if_clear(True,1.4))
        self.assertTrue(g.release_if_clear(True,1.5))


if __name__=='__main__':unittest.main()
