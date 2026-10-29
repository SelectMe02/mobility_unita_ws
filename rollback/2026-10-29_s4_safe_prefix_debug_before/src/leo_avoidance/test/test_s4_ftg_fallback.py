import unittest
from unittest.mock import Mock
import numpy as np
from leo_avoidance.route_avoidance import RouteAvoidance, Decision, _Pass
from leo_avoidance.s4_ftg_fallback import S4Fallback, Profile
from leo_avoidance.ftg_local import Plan

class S4FallbackTests(unittest.TestCase):
    def setUp(self):
        self.p = RouteAvoidance([(0., 0.), (100., 0.)])
        self.p.set_static_pass_zone(0., 10., 80., 95., -3.2)
        self.p.static_pass_managed = True
        self.p.last_s = 20.
        self.f = S4Fallback()
        self.f.ftg = Mock()
        self.f.ftg.plan.return_value = Plan(5., 0., 'gap fits', 20., 4.)
        self.points = np.array([[float(x), -7., .2] for x in range(25)])
        self.d = Decision('WAIT_PASS', [], 0., 'S4 no path')
        self.pose = (20., -1., 0.)

    def tick(self, t, d=None, **kw):
        return self.f.apply(self.p, self.d if d is None else d, self.points,
                            self.pose, 0., t, **kw)

    def enter(self):
        for t in (1., 1.2, 1.41):
            result = self.tick(t)
        return result

    def test_persistent_failure_enters_and_keeps_safe_path(self):
        self.assertIs(self.tick(1.), self.d)
        self.assertIs(self.tick(1.2), self.d)
        d = self.tick(1.41)
        self.assertEqual(self.f.mode, 'ftg')
        self.assertGreater(d.speed_cap_mps, 0.)
        profile = self.f.profile
        self.f.ftg.plan.return_value = Plan(5., -.3, 'gap fits', 20., 4.)
        self.tick(1.5)
        self.assertIs(self.f.profile, profile)

    def test_single_failure_and_non_s4_do_not_switch(self):
        self.tick(1.)
        clear = Decision('FOLLOW', [(20., 0.), (40., 0.)], 5., 'clear')
        self.tick(1.2, clear)
        self.tick(1.4)
        self.assertEqual(self.f.mode, 'route')
        self.p.static_pass_managed = False
        self.assertIs(self.tick(2.), self.d)

    def test_live_or_remembered_obstacle_blocks_fallback(self):
        for remembered in (False, True):
            self.setUp()
            self.d.obstacles = [dict(s=24., d=-1., l=2., r=2.,
                v_s=0., v_d=0., moving=False, id=1, remembered=remembered)]
            self.assertEqual(self.enter().speed_cap_mps, 0.)

    def test_untracked_scan_points_block_fallback(self):
        self.points = np.vstack((self.points, [[3., 0., .2]]))
        self.assertEqual(self.enter().speed_cap_mps, 0.)

    def test_traffic_and_movers_are_never_bypassed(self):
        self.enter()
        self.assertEqual(self.tick(1.5, permitted=False).speed_cap_mps, 0.)
        self.d.traffic_speed_cap_mps = 0.
        self.assertEqual(self.tick(1.6).speed_cap_mps, 0.)
        self.d.traffic_speed_cap_mps = float('inf')
        self.d.obstacles = [dict(moving=True)]
        self.assertEqual(self.tick(1.7).speed_cap_mps, 0.)

    def test_clear_flicker_does_not_return(self):
        self.enter()
        self.p.active = _Pass(0., 10., 70., 90., -1., 30.)
        clear = Decision('PASSING', [(20., -1.), (90., 0.)], 5., 'clear')
        for i in range(20):
            self.tick(1.5+i*.1, clear if i%2 else self.d)
        self.assertEqual(self.f.mode, 'ftg')

    def test_stable_recovery_uses_continuous_bridge_before_return(self):
        self.enter()
        self.p.active = _Pass(0., 10., 70., 90., -1., 30.)
        clear = Decision('PASSING', [(20., -1.), (90., 0.)], 5., 'clear')
        for i in range(15):
            self.tick(1.5+i*.1, clear)
        self.assertEqual(self.f.mode, 'recover')
        self.assertAlmostEqual(self.f.profile.offset_at(20.), -1.)
        self.assertAlmostEqual(self.f.profile.curve(20., 1), 0.)
        self.p.last_s = self.f.join
        self.pose = (self.p.last_s, -1., 0.)
        self.assertIs(self.tick(3., clear), clear)
        self.assertEqual(self.f.mode, 'route')

    def test_duplicate_or_gapped_scan_does_not_reuse_motion(self):
        self.enter()
        self.assertEqual(self.tick(1.41).speed_cap_mps, 0.)
        self.assertIs(self.tick(3.), self.d)
        self.assertEqual(self.f.mode, 'route')

    def test_boundary_and_curvature_rejection(self):
        for profile in (Profile([20., 36.], [-1., -5.], [0., 0.]),
                        Profile([20., 22.], [-1., -3.], [0., 0.])):
            self.assertFalse(self.f.safe(self.p, profile, [], 0., self.points, self.pose))

if __name__ == '__main__':
    unittest.main()
