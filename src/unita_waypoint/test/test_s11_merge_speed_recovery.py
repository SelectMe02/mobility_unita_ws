import unittest
import test_highway_planner as f

class MergeSpeedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.p=f.planner(s11_forward_only=True)
        self.path=[(float(x),0.) for x in range(80)]

    def cap(self,car,normal=31.9,v=30./3.6,others=()):
        return self.p._s11_merge_recovery_cap(self.path,v,100./3.6,
                  (car.x,car,car.vx),[car]+list(others),normal)

    def test_clear_moving_front_gap_allows_meaningful_recovery_target(self):
        self.assertAlmostEqual(self.cap(f.car(30.,1,10.)),40.)

    def test_close_or_stationary_front_still_limits_speed(self):
        self.assertEqual(self.cap(f.car(8.,1,10.)),31.9)
        self.assertEqual(self.cap(f.car(30.,1,0.)),31.9)

    def test_uncertain_motion_and_brake_demand_not_overridden(self):
        car=f.car(30.,1,10.);car.velocity_reliable=False
        self.assertEqual(self.cap(car),31.9)
        car.velocity_reliable=True
        self.assertEqual(self.cap(car,normal=28.),28.)
        self.assertEqual(self.cap(car,normal=52.,v=50./3.6),52.)

    def test_other_obstacle_blocks_acceleration(self):
        self.assertEqual(self.cap(f.car(30.,1,10.),others=[f.car(15.,1,0.)]),31.9)

if __name__=='__main__':unittest.main()
