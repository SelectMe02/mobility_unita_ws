import math
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import test_highway_planner as f


def point(along, lateral=0.):
    c=f.core
    return (c.APPROACH[0]+c.TANGENT[0]*along-c.TANGENT[1]*lateral,
            c.APPROACH[1]+c.TANGENT[1]*along+c.TANGENT[0]*lateral)


def npc(lateral=0., **kw):
    x,y=point(f.core.LENGTH,lateral)
    return f.core.Obstacle(x,y,2.,length=4.5,width=1.9,
                          yaw=math.atan2(f.core.TANGENT[1],f.core.TANGENT[0]),**kw)


def planner(merge_control=False):
    route=[point(i-60.) for i in range(201)]
    lanes={n:[point(i-60.,(n-1)*3.6) for i in range(201)] for n in (2,3)}
    indices={n:dict(enumerate(line)) for n,line in lanes.items()}
    p = f.core.LocalAvoidancePlanner(route,[11]*len(route),lanes,
                                        lane_indices=indices,highway_sectors=(11,))
    if not merge_control:
        # Legacy EntryGuard tests also cover a pre-existing committed maneuver.
        p._s11_r1_merge = lambda *args: None
    return p


class EntrySlowdownTests(unittest.TestCase):
    def test_requested_approach_detects_npc_and_holds_r1_at_30(self):
        p=planner()
        d=p.plan(60,point(0.),85./3.6,[npc()],desired_speed_mps=100./3.6)
        self.assertEqual(d.reason,'s11_entry_slowdown')
        self.assertEqual(d.speed_cap_kmh,30.)
        self.assertIsNone(p.transition)
        self.assertTrue(all(abs(f.core.s11_entry_coordinates(pt)[1])<1e-6 for pt in d.path))

    def test_monitor_starts_before_requested_approach(self):
        p=planner()
        d=p.plan(30,point(-30.),80./3.6,[npc()],desired_speed_mps=100./3.6)
        self.assertEqual(d.speed_cap_kmh,30.)
        self.assertIsNone(p.transition)

    def test_no_npc_does_not_apply_entry_speed_limit(self):
        p=planner()
        d=p.plan(60,point(0.),80./3.6,[],desired_speed_mps=100./3.6)
        self.assertFalse(p.s11_entry_guard.active)
        self.assertNotEqual(d.reason,'s11_entry_slowdown')

    def test_other_lane_road_and_outside_region_do_not_trigger(self):
        for o in (npc(3.6),npc(7.2),npc(road_surface=True),
                  npc(lane_stable=True,lane_sector=11,raceline=2,lane_boundary_crossing=False)):
            p=planner();p.plan(60,point(0.),20.,[o],desired_speed_mps=25.)
            self.assertFalse(p.s11_entry_guard.active)
        o=npc();o.x,o.y=point(f.core.LENGTH+15.)
        self.assertFalse(f.core.EntryGuard().update(11,1,point(0.),[o]))

    def test_new_or_occluded_npc_keeps_30_until_entry_passed(self):
        p=planner();o=npc(confirmed=False,velocity_reliable=False)
        p.plan(60,point(0.),20.,[o],desired_speed_mps=25.)
        d=p.plan(90,point(30.),10.,[],desired_speed_mps=25.)
        self.assertEqual(d.speed_cap_kmh,30.)
        self.assertIsNone(p.transition)
        p.plan(100,point(40.),10.,[],desired_speed_mps=25.)
        self.assertFalse(p.s11_entry_guard.active)
        self.assertIsNotNone(p.transition)
        self.assertEqual(p.transition['target'],2)

    def test_already_approved_change_is_slowed_without_reversal(self):
        p=planner()
        for stamp in (1.,1.1,1.3):
            p.plan(60,point(0.),20.,[],desired_speed_mps=25.,observation_stamp=stamp)
        self.assertIsNotNone(p.transition)
        target=p.transition['target']
        d=p.plan(61,point(1.),20.,[npc()],desired_speed_mps=25.)
        self.assertLessEqual(d.speed_cap_kmh,30.)
        self.assertEqual(p.transition['target'],target)
        self.assertNotEqual(p.transition['state'],'CANCELLING')

    def test_other_sector_or_completed_lane_does_not_trigger(self):
        for sector,lane in ((10,1),(12,1),(11,2),(11,3)):
            self.assertFalse(f.core.EntryGuard().update(sector,lane,point(0.),[npc()]))

    def test_lidar_front_override_preserves_points_at_50m(self):
        from collections import deque
        import threading
        import numpy as np
        from lidar_obstacles import LidarObstacles
        n=LidarObstacles.__new__(LidarObstacles)
        n.body_from_lidar=np.eye(3);n.lidar_x=1.4;n.lidar_y=0.;n.lidar_z=1.23
        n.lock=threading.Lock();n.imus=deque();n.imu_max_skew_s=.05
        n.front_range=30.;n.rear_range=55.;n.side_range=12.
        n.half_width=.946;n.front_extent=3.845;n.rear_extent=.790
        n.obstacle_min_height_m=.22;n.obstacle_max_height_m=2.6
        n.ground_plane_tolerance_m=.04
        raw=np.array([[48.6,2.,-.23]])
        _,_,normal,_=n.filter_ground(raw,1.)
        _,_,entry,_=n.filter_ground(raw,1.,front_range=65.)
        self.assertFalse(normal[0]);self.assertTrue(entry[0])
        self.assertEqual(n.front_range,30.)


if __name__=='__main__':unittest.main()
