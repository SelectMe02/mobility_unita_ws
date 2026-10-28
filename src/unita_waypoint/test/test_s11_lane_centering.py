"""Regression for the 97 m merge / 48 m moving correction seen in 17:03 bag."""
import math
import unittest
import test_highway_planner as f


class LaneCenteringTests(unittest.TestCase):
    def planner(self):
        return f.planner(s11_forward_only=True, highway_keep_lane_priority=True)

    def transition(self, source=1, target=2):
        return dict(source=source, target=target, start=0., length=97.,
                    shape='quintic', highway=True, state='LANE_CHANGING',
                    speed_cap_kmh=85., s11_forward_only=True)

    def test_correction_expires_at_original_endpoint(self):
        p = self.planner()
        for source,target in ((1,2),(2,3)):
            tr = self.transition(source,target)
            for index in (40,70,90,96):
                ego=(float(index),f.LANE_Y[source]+1.)
                path=p._path(p._forward(index,80.),ego,target,float(index),tr)
                self.assertAlmostEqual(path[0][1],ego[1])
                for x,y in path:
                    if x>=97.:
                        self.assertAlmostEqual(y,f.LANE_Y[target])

    def test_after_endpoint_entire_path_matches_saved_lane_even_when_offset(self):
        p=self.planner()
        for source,target in ((1,2),(2,3)):
            tr=self.transition(source,target)
            for index in (97,110,160):
                path=p._path(p._forward(index,80.),(float(index),f.LANE_Y[target]-1.8),
                             target,float(index),tr)
                self.assertTrue(path)
                self.assertTrue(all(abs(y-f.LANE_Y[target])<1e-9 for x,y in path))

    def test_capture_does_not_require_source_lane_to_still_exist(self):
        p=self.planner()
        tr=self.transition(2,3)
        p.lane_indices[2].pop(110)
        path=p._path(p._forward(110,60.),(110.,5.4),3,110.,tr)
        self.assertGreater(len(path),2)
        self.assertTrue(all(abs(y-7.2)<1e-9 for x,y in path))

    def test_capture_then_five_metre_centerline_hold_before_next_change(self):
        p=self.planner();p.transition=self.transition();p.s11_reached_lane=1
        d=p.plan(110,(110.,2.),8.333,[],desired_speed_mps=f.CRUISE)
        self.assertEqual(d.reason,'s11_centerline_capture')
        self.assertEqual(p.transition['target'],2)
        d=p.plan(112,(112.,3.4),8.333,[],desired_speed_mps=f.CRUISE)
        self.assertIsNone(p.transition)
        self.assertEqual(d.reason,'s11_centerline_hold')
        self.assertTrue(all(abs(y-3.6)<1e-9 for x,y in d.path))
        p.plan(116,(116.,3.6),8.333,[],desired_speed_mps=f.CRUISE)
        self.assertIsNone(p.transition)
        p.plan(118,(118.,3.6),8.333,[],desired_speed_mps=f.CRUISE)
        self.assertEqual((p.transition['source'],p.transition['target']),(2,3))

    def test_30kmh_pure_pursuit_converges_after_endpoint(self):
        # Kinematic steering check, not a MORAI actuator/vehicle validation.
        p=self.planner();tr=self.transition()
        x,y,yaw=110.,1.8,0.
        speed,dt,lookahead=30./3.6,.05,.78*30./3.6
        for _ in range(72):
            path=p._path(p._forward(int(x),60.),(x,y),2,x,tr)
            for px,py in path:
                dx,dy=px-x,py-y
                forward=math.cos(yaw)*dx+math.sin(yaw)*dy
                lateral=-math.sin(yaw)*dx+math.cos(yaw)*dy
                ld=math.hypot(dx,dy)
                if forward>0 and ld>=lookahead:
                    curvature=2.*lateral/(ld*ld)
                    break
            x+=speed*math.cos(yaw)*dt;y+=speed*math.sin(yaw)*dt
            yaw+=speed*curvature*dt
        self.assertLess(abs(y-3.6),.35)


if __name__=='__main__':unittest.main()
