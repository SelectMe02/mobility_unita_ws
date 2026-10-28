import unittest
import test_highway_planner as f


class RearRiskAccelerationTests(unittest.TestCase):
    def setUp(self):
        self.p=f.planner(s11_forward_only=True)
        self.path=[(float(x),3.6+3.6*min(x/40.,1.)) for x in range(81)]
        self.rear=f.car(-16.,3,16.5)

    def cap(self, obstacles, upper=50.):
        return self.p._s11_rear_acceleration_cap(0,(0.,3.6),self.path,10.,upper,obstacles)

    def test_rear_conflict_prefers_bounded_acceleration(self):
        self.assertEqual(self.cap([self.rear]),50.)
        self.assertIsNotNone(self.p._highway_path_conflict(self.path,10.,[self.rear]))

    def test_front_conflict_prevents_acceleration(self):
        self.assertIsNone(self.cap([self.rear,f.car(23.,3,0.)]))

    def test_uncertain_or_predicted_rear_does_not_trigger(self):
        for field in ('confirmed','velocity_reliable'):
            setattr(self.rear,field,False)
            self.assertIsNone(self.cap([self.rear]))
            setattr(self.rear,field,True)
        self.rear.predicted=True
        self.assertIsNone(self.cap([self.rear]))

    def test_speed_ceiling_and_nonclosing_car(self):
        self.assertIsNone(self.cap([self.rear],36.))
        self.rear.vx=9.
        self.assertIsNone(self.cap([self.rear]))

    def test_gradual_acceleration_still_detects_overlap(self):
        constant=self.p._highway_path_conflict(self.path,10.,[self.rear])
        accelerated=self.p._highway_path_conflict(self.path,10.,[self.rear],target_speed=50./3.6)
        self.assertIsNotNone(accelerated)
        self.assertGreater(accelerated[0],constant[0])

    def test_committed_transition_uses_acceleration_reason(self):
        from unittest.mock import patch
        p=self.p
        p.transition=dict(source=2,target=3,start=0.,length=40.,state='LANE_CHANGING',
                          s11_forward_only=True,speed_cap_kmh=50.)
        ahead=p._forward(0,80.)
        with patch.object(p,'_path',return_value=self.path):
            d=p._s11_forward_transition(0,(0.,3.6),10.,100./3.6,0.,ahead,
                                       {2:(0.,3.6),3:(0.,7.2)},[self.rear],100.)
        self.assertEqual(d.reason,'s11_rear_risk_accelerate')
        self.assertEqual(d.speed_cap_kmh,50.)
        self.assertEqual(p.transition['target'],3)


if __name__=='__main__':unittest.main()
