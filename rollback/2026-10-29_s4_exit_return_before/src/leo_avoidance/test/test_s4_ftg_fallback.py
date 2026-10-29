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

    def test_distant_blocker_allows_stoppable_prefix(self):
        profile = Profile([20., 60.], [-1., -1.], [0., 0.])
        self.d.obstacles = [dict(s=38., d=-1., l=2., r=1.,
            v_s=0., v_d=0., moving=False, id=7, remembered=True)]
        result = self.f.follow_prefix(self.p, self.d, profile, self.points,
                                      self.pose, 0., 'LOCAL_PATH')
        self.assertGreater(result.speed_cap_mps, 0.)
        self.assertLess(result.path[-1][0], 38.)
        v = result.speed_cap_mps
        cfg = self.p.config
        self.assertLessEqual(v*cfg.reaction_s+v*v/(2*cfg.brake_mps2),
                             self.f.status['safe_distance_m']-.75+.01)
        self.assertIn('remembered', self.f.status['reason'])

    def test_near_raw_point_stops_even_with_retained_path(self):
        profile = Profile([20., 60.], [-1., -1.], [0., 0.])
        self.points = np.vstack((self.points, [[3., 0., .2]]))
        result = self.f.follow_prefix(self.p, self.d, profile, self.points,
                                      self.pose, 0., 'LOCAL_PATH')
        self.assertEqual(result.speed_cap_mps, 0.)
        self.assertEqual(self.f.status['state'], 'STOP')

    def test_clear_return_end_connects_to_route_without_end_braking(self):
        profile = Profile([20., 24.], [-.1, 0.], [0., 0.])
        result = self.f.follow_prefix(self.p, self.d, profile, self.points,
                                      self.pose, 0., 'LOCAL_PATH')
        self.assertGreater(result.path[-1][0], 30.)
        self.assertEqual(result.path[-1][1], 0.)
        self.assertAlmostEqual(result.speed_cap_mps, self.p.config.pass_speed_kmh/3.6)

    def test_under_five_meters_uses_stopping_distance_not_binary_stop(self):
        profile = Profile([20., 60.], [-1., -1.], [0., 0.])
        self.f.clear_prefix = lambda *args: (23.75, 'observed obstacle #1')
        result = self.f.follow_prefix(self.p, self.d, profile, self.points,
                                      self.pose, 0., 'LOCAL_PATH')
        self.assertGreater(result.speed_cap_mps, 0.)
        self.assertEqual(result.path[-1][0], 23.75)

    def test_obstacle_on_continuation_is_still_checked(self):
        profile = Profile([20., 24.], [-.1, 0.], [0., 0.])
        self.d.obstacles = [dict(s=29.,d=0.,l=1.,r=1.,moving=False,id=1,v_s=0.,v_d=0.)]
        result = self.f.follow_prefix(self.p, self.d, profile, self.points,
                                      self.pose, 0., 'LOCAL_PATH')
        self.assertLess(result.path[-1][0], 29.)
        self.assertIn('obstacle',self.f.status['reason'])

    def test_arming_ftg_does_not_replace_better_local_prefix(self):
        self.p.active = _Pass(0., 10., 70., 90., -1., 30.)
        self.d = Decision('PASSING', [(20., -1.), (90., 0.)], 0., 'far conflict')
        normal = self.p.active
        self.f.clear_prefix = lambda p, profile, *args: (
            (28., 'far obstacle') if profile is normal else
            ((26., 'short FTG') if profile is not None else (20., 'no path')))
        for t in (1., 1.2, 1.41, 1.6):
            result = self.tick(t)
            self.assertGreater(result.speed_cap_mps, 0.)
            self.assertEqual(self.f.status['source'], 'LOCAL_PATH')

if __name__ == '__main__':
    unittest.main()

class ReturnBoundaryTests(S4FallbackTests):
    def test_small_inward_connection_passes_without_ftg(self):
        profile = Profile([20.,28.,40.],[-3.29168,-3.,0.],[.08,.1,0.])
        pose=(20.,-3.29168,.08)
        end, reason=self.f.clear_prefix(self.p,profile,[],0.,self.points,pose)
        self.assertGreater(end,28.)
        self.assertNotEqual(reason,'saved corridor boundary')

    def test_outward_large_or_nonreturning_connection_is_rejected(self):
        for values, slopes in [([-3.5,-3.,0.],[.1,.1,0.]),
                               ([-3.29,-3.29,-3.29],[0.,0.,0.]),
                               ([-3.29,-3.,0.],[-.1,.1,0.])]:
            profile=Profile([20.,28.,40.],values,slopes)
            end,reason=self.f.clear_prefix(self.p,profile,[],0.,self.points,(20.,values[0],0.))
            self.assertEqual(end,20.)
            self.assertEqual(reason,'saved corridor boundary')

    def test_return_exception_does_not_skip_obstacles(self):
        profile=Profile([20.,28.,40.],[-3.29,-3.,0.],[.08,.1,0.])
        obstacle=dict(s=21.,d=-3.29,l=1.,r=.5,id=1,moving=False,v_s=0.,v_d=0.)
        end,reason=self.f.clear_prefix(self.p,profile,[obstacle],0.,self.points,(20.,-3.29,.08))
        self.assertEqual(end,20.)
        self.assertIn('obstacle',reason)

    def test_final_local_stop_arms_fallback_even_if_original_moves(self):
        self.p.active=_Pass(10.,15.,30.,45.,-3.2,25.)
        self.f.anchored_route=Mock(return_value=self.p.active)
        stopped=Decision('PASSING',[(20.,-3.2),(30.,-3.2)],0.,'boundary')
        self.f.follow_prefix=Mock(return_value=stopped)
        self.f.ftg.plan.return_value=Plan(0.,0.,'no gap',0.,0.)
        moving=Decision('PASSING',[(20.,-3.2),(30.,-3.2)],5.,'passing')
        for t in (1.,1.2,1.41):self.tick(t,moving)
        self.assertEqual(self.f.mode,'ftg')
