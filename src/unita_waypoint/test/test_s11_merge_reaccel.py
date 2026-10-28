import unittest
import test_s11_entry_slowdown as e
import test_highway_planner as f

class MergeReaccelTests(unittest.TestCase):
    def sequence(self, movement):
        p=e.planner(merge_control=True)
        p.sector_min_speed_kmh=30.
        results=[]
        for k in range(3):
            o=e.npc(track_id=7,velocity_reliable=False)
            o.x,o.y=e.point(f.core.LENGTH+15.+movement*k)
            o.measurement_stamp=1.+k*.1
            d=p.plan(88,e.point(f.core.LENGTH),30./3.6,[o],
                     desired_speed_mps=100./3.6,observation_stamp=o.measurement_stamp)
            results.append(d.speed_cap_kmh)
            self.assertEqual(o.vx,0.)
            self.assertEqual(o.vy,0.)
            self.assertFalse(o.velocity_reliable)
        return results

    def test_fresh_consistent_motion_releases_false_stationary_following_cap(self):
        caps=self.sequence(1.)
        self.assertLessEqual(caps[0],30.)
        self.assertLessEqual(caps[1],30.)
        self.assertGreater(caps[2],30.)

    def test_stationary_vehicle_still_requires_slowing(self):
        self.assertTrue(all(c<=30. for c in self.sequence(0.)))

    def test_backward_motion_does_not_confirm_forward_clearance(self):
        self.assertTrue(all(c<=30. for c in self.sequence(-1.)))

if __name__=='__main__':unittest.main()
