"""Extreme-only tunnel policy; raw collision evidence is never filtered away."""
import json
import math
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np
import yaml
from leo_avoidance.ftg_local import Config, FollowTheGap

ROOT = Path(__file__).resolve().parents[3]


def config(**kwargs):
    values = yaml.safe_load((ROOT/'src/leo_avoidance/config/avoidance_local.yaml').read_text())
    values.update(yaml.safe_load((ROOT/'src/leo_avoidance/config/avoidance_extreme.yaml').read_text()))
    values = {k:v for k,v in values.items() if k in vars(Config())}
    values.update(max_speed_mps=55./3.6)
    values.update(kwargs)
    return Config(**values)


def walls():
    x = np.arange(1., 30., .1)
    return np.array([[a, b, 0.] for a in x for b in (-3.846, 3.846)])


class ExtremeTest(unittest.TestCase):
    def setUp(self):
        self.time = 0.
        self.p = FollowTheGap(config(), clock=lambda:self.time)

    def scan(self, extra=(), speed=2., **kwargs):
        self.time += .1
        cloud = np.vstack((walls(), np.array(extra).reshape(-1,3)))
        return self.p.plan(cloud, speed, wall_follow=True, stamp=self.time, **kwargs)

    def test_wall_follow_30_even_zero_direction_weight(self):
        for speed in (30.,40.,55.):
            self.p.config = replace(self.p.config, max_speed_mps=speed/3.6)
            for _ in range(40):
                a = self.scan(wall_follow_weight=0.)
                self.assertEqual(a.avoidance_phase,'WALL_FOLLOW')
                self.assertLessEqual(a.speed_mps,30./3.6)
            self.assertGreaterEqual(a.speed_mps,27./3.6)

    def test_noise_side_rear_wall_fragments_do_not_start_passing(self):
        for extra in ([[12.,0.,0.]], [[10.,2.6,0.],[10.3,2.6,0.],[10.6,2.6,0.]],
                      [[-2.,0.,0.],[-2.,.3,0.],[-2.,-.3,0.]],
                      [[10.,3.4,0.],[10.3,3.4,0.],[10.6,3.4,0.]]):
            for _ in range(3):self.assertEqual(self.scan(extra).avoidance_phase,'WALL_FOLLOW')

    def test_confirm_clear_recovery_and_reappearance(self):
        box = [[12.,y,0.] for y in (-.3,0.,.3)]
        self.assertEqual(self.scan(box).avoidance_phase,'AVOID_PENDING')
        a = self.scan(box)
        self.assertEqual(a.avoidance_phase,'PASSING')
        self.assertLessEqual(a.speed_mps,25./3.6)
        self.assertEqual(self.scan().avoidance_phase,'PASSING')
        self.assertEqual(self.scan().avoidance_phase,'PASSING')
        self.assertEqual(self.scan().avoidance_phase,'RECOVERY')
        self.assertEqual(self.scan(box).avoidance_phase,'PASSING')
        for _ in range(3): a=self.scan()
        self.assertEqual(a.avoidance_phase,'RECOVERY')
        for _ in range(5): a=self.scan()
        self.assertEqual(a.avoidance_phase,'WALL_FOLLOW')

    def test_imminent_single_hit_stops_immediately(self):
        a = self.scan([[2.7,0.,0.]], speed=8.)
        self.assertEqual(a.speed_mps,0.)
        self.assertTrue(a.obstacle_stop)

    def test_blocked_gap_stops(self):
        a=self.scan([[4.,y,0.] for y in np.arange(-3.8,3.9,.1)],speed=8.)
        self.assertEqual(a.speed_mps,0.)

    def test_scan_duplicates_and_gaps_do_not_confirm(self):
        box=np.array([[12.,y,0.] for y in (-.3,0.,.3)])
        self.p._scan_stamp=1.
        self.assertEqual(self.p._blackout_phase(box,(None,None),0.),'AVOID_PENDING')
        self.assertEqual(self.p._blackout_phase(box,(None,None),0.),'AVOID_PENDING')
        self.p._scan_stamp=2.
        self.assertEqual(self.p._blackout_phase(box,(None,None),0.),'AVOID_PENDING')
        self.p._scan_stamp=2.1
        self.assertEqual(self.p._blackout_phase(box,(None,None),0.),'PASSING')

    def test_recovery_acceleration_and_all_phase_caps(self):
        self.scan([[2.7,0.,0.]],speed=8.)
        previous=0.
        for _ in range(100):
            a=self.scan()
            self.assertLessEqual(a.speed_mps-previous,.12+1e-8)
            self.assertLessEqual(a.speed_mps,30./3.6)
            previous=a.speed_mps
        self.assertGreaterEqual(previous,27./3.6)

    def test_gps_normal_and_legacy_not_capped(self):
        legacy=FollowTheGap(replace(config(),blackout_speed_policy=False))
        extreme=FollowTheGap(config())
        a=legacy.plan(walls(),1.,wall_follow=False)
        b=extreme.plan(walls(),1.,wall_follow=False)
        self.assertEqual(a,b)
        self.assertGreater(b.speed_mps,30./3.6)

    def test_blackout_publication_cap_at_portal(self):
        import runpy
        helpers=runpy.run_path(str(ROOT/'src/leo_avoidance/test/test_rejoin_route_local.py'))
        state=helpers['exit_tick_state']()
        state.config=config()
        state.latest_plan=(replace(state.latest_plan[0],speed_mps=55./3.6),10.)
        helpers['tick_at_capture'](state)
        self.assertLessEqual(state.speed_pub.publish.call_args[0][0].data,30.+1e-9)
        report=json.loads(state.avoidance_status_pub.publish.call_args[0][0].data)
        self.assertLessEqual(report['target_speed_mps'],30./3.6)

    def test_extreme_preserves_side_intrusion_stop(self):
        import runpy
        helpers=runpy.run_path(str(ROOT/'src/leo_avoidance/test/test_ftg_local.py'))
        test=helpers['test_tunnel_side_intrusion_reaches_stop_plan_but_not_route']
        # Run the same protected intrusion scenario with the new policy enabled.
        old=test.__globals__['Config']
        try:
            test.__globals__['Config']=lambda **kw: replace(old(**kw),blackout_speed_policy=True)
            test()
        finally:
            test.__globals__['Config']=old

    def test_map_only_55_points_changed(self):
        old=json.loads((ROOT/'src/unita_waypoint/config/map_tunner_project.json').read_text())
        new=json.loads((ROOT/'src/unita_waypoint/config/map_extreme_project.json').read_text())
        for sector in old['sectors']:
            for points in sector['racelines'].values():
                for p in points:
                    if p.get('speed_kmh')==55:p['speed_kmh']=59
        self.assertEqual(old,new)

if __name__=='__main__':unittest.main()
