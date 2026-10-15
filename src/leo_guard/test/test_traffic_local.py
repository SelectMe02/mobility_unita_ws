import unittest
from leo_guard.traffic_local import TrafficDecision, RouteProgress


class TrafficTests(unittest.TestCase):
    def test_route_reacquires_gps_eight_metres_behind_lidar_progress(self):
        points = [(i * .5, 0.) for i in range(100)]
        route = RouteProgress(points, backward_window=30)
        route.index = 80
        progress, error = route.locate(32., .2, 0.)
        self.assertIn(route.index, (63, 64))
        self.assertAlmostEqual(progress, 32.)
        self.assertAlmostEqual(error, .2)

    def controller(self, maneuver='straight', reviewed=True):
        stop=dict(id='first',s=50.,maneuver=maneuver,controlled=True,
                  reviewed=reviewed,signal_roi=[.3,.05,.7,.6],clearance_m=12.)
        return TrafficDecision([stop],1000.)

    def signals(self,label,frame):
        return dict(valid=True,frame=frame,detections=[dict(
            label=label,confidence=.95,bbox_normalized=[.45,.2,.55,.3])])

    def release(self,c,label,progress=40.):
        result=None
        for i in range(4):result=c.decide(progress,self.signals(label,i),i*.25)
        return result

    def test_red_stops_before_line_and_missing_signal_does_not_release(self):
        c=self.controller()
        cap,_=c.decide(45.,self.signals('traffic_light_red',1),0.)
        self.assertEqual(cap,0.)
        cap,_=c.decide(45.,None,.2);self.assertEqual(cap,0.)

    def test_green_releases_straight_but_never_left(self):
        c=self.controller();cap,detail=self.release(c,'traffic_light_green')
        self.assertEqual(cap,float('inf'));self.assertTrue(detail['release_stable'])
        c=self.controller('left');_,detail=self.release(c,'traffic_light_green')
        self.assertFalse(detail['release_stable'])
        _,detail=self.release(c,'traffic_light_left');self.assertTrue(detail['release_stable'])

    def test_duplicate_green_image_cannot_release(self):
        c=self.controller()
        for i in range(20):_,d=c.decide(45.,self.signals('traffic_light_green',1),i*.1)
        self.assertFalse(d['release_stable'])

    def test_unreviewed_head_never_releases_on_green(self):
        c=self.controller(reviewed=False);cap,d=self.release(c,'traffic_light_green',45.)
        self.assertEqual(cap,0.);self.assertFalse(d['release_stable'])

    def test_distant_green_in_same_roi_cannot_release_current_stop(self):
        c=self.controller();c.stops[0]['signal_min_width']=.04
        for i in range(4):
            signals=self.signals('traffic_light_green',i)
            signals['detections'][0]['bbox_normalized']=[.49,.24,.51,.25]
            cap,d=c.decide(45.,signals,i*.25)
        self.assertEqual(cap,0.);self.assertFalse(d['release_stable'])

    def test_signal_from_outside_surveyed_distance_cannot_build_release_latch(self):
        c=self.controller();c.stops[0]['signal_activation_distance_m']=12.
        _,d=self.release(c,'traffic_light_green',30.)
        self.assertFalse(d['release_stable'])
        # Inside the surveyed distance one green frame releases at once.
        _,d=c.decide(39.,self.signals('traffic_light_green',5),1.)
        self.assertTrue(d['release_stable'])
        for i in range(4):
            _,d=c.decide(39.,self.signals('traffic_light_green',6+i),1.25+i*.25)
        self.assertTrue(d['release_stable'])

    def test_unprotected_right_requires_separate_yield_logic(self):
        c=self.controller('right_unprotected');cap,d=self.release(c,'traffic_light_green',45.)
        self.assertEqual(cap,0.);self.assertFalse(d['release_stable'])

    def test_right_only_green_plus_clearance_releases_without_fixed_turn_speed(self):
        # The Map Tuner point speed is the turn speed; the gate adds none.
        c=self.controller('right_unprotected')
        for i in range(4):
            cap,d=c.decide(45.,self.signals('traffic_light_green',i),i*.25,yield_clear=True)
        self.assertTrue(d['release_stable']);self.assertEqual(cap,float('inf'))
        cap,d=c.decide(45.,self.signals('traffic_light_green',5),1.,yield_clear=False)
        self.assertFalse(d['release_stable']);self.assertEqual(cap,0.)

    def test_right_red_and_left_arrow_never_release_even_with_clearance(self):
        for label in ('traffic_light_red','traffic_light_left'):
            c=self.controller('right_unprotected')
            for i in range(4):
                cap,d=c.decide(45.,self.signals(label,i),i*.25,yield_clear=True)
            self.assertFalse(d['release_stable']);self.assertEqual(cap,0.)

    def test_unpermitted_crossing_does_not_skip_to_next_intersection(self):
        c=self.controller();c.decide(49.,None,0.)
        cap,d=c.decide(51.,None,.1)
        self.assertEqual(cap,0.);self.assertEqual(d['state'],'hold')

    def test_finish_crossing_when_red_changes_after_front_enters(self):
        c=self.controller();self.release(c,'traffic_light_green',50.)
        cap,d=c.decide(51.,self.signals('traffic_light_red',5),1.)
        self.assertGreater(cap,0.);self.assertEqual(d['state'],'clearing')

    def test_braking_margin_does_not_grant_early_intersection_commit(self):
        c=self.controller();self.release(c,'traffic_light_green',45.)
        self.assertIsNone(c.committed)
        cap,d=c.decide(45.,self.signals('traffic_light_red',5),1.)
        self.assertEqual(cap,0.)

    def test_teleport_invalidates_crossing_permission(self):
        c=self.controller();self.release(c,'traffic_light_green',45.)
        with self.assertRaises(ValueError):c.decide(200.,None,1.)
        self.assertIsNone(c.committed)

    def test_measured_front_axle_commits_only_at_line_and_clears_full_exit(self):
        c=self.controller()
        self.release(c,'traffic_light_green',46.)
        c.decide(46.,self.signals('traffic_light_green',4),1.,front_offset=3.)
        self.assertIsNone(c.committed)
        c.decide(47.,self.signals('traffic_light_green',5),1.25,front_offset=3.)
        self.assertIsNotNone(c.committed)
        c.decide(54.,self.signals('traffic_light_red',6),1.4,front_offset=3.)
        _,d=c.decide(60.,self.signals('traffic_light_red',6),1.5,front_offset=3.)
        self.assertEqual(d['state'],'clearing')
        _,d=c.decide(63.,None,1.75,front_offset=3.)
        self.assertNotEqual(d['state'],'clearing')

    def test_far_camera_wait_point_does_not_commit_before_front_crosses(self):
        c=self.controller();c.stops[0]['stop_margin_m']=17.
        cap,_=c.decide(33.,self.signals('traffic_light_red',1),0.,front_offset=3.)
        self.assertEqual(cap,0.)
        self.release(c,'traffic_light_green',33.)
        self.assertIsNone(c.committed)

    def test_pose_outside_route_or_reverse_is_rejected(self):
        r=RouteProgress([[0,0],[10,0],[20,0]])
        self.assertAlmostEqual(r.locate(5.,.2,0.)[0],5.)
        with self.assertRaises(ValueError):r.locate(5.,3.,0.)
        with self.assertRaises(ValueError):r.locate(5.,0.,3.141592653589793)

    def test_blackout_clearance_skips_only_reviewed_uncontrolled_stop(self):
        stops = [dict(id='tunnel_entrance',s=50.,maneuver='straight',
                      controlled=False,reviewed=True,clearance_m=120.),
                 dict(id='next_signal',s=190.,maneuver='straight',
                      controlled=True,reviewed=True,signal_roi=[.3,.05,.7,.6],
                      clearance_m=12.)]
        decision = TrafficDecision(stops,1000.)
        self.assertAlmostEqual(decision.distance_to_next_controlled_stop(45.,3.),142.)
        self.assertAlmostEqual(decision.distance_to_next_controlled_stop(185.,3.),2.)

    def test_other_camera_green_cannot_release_or_preserve_latch(self):
        c=self.controller();c.stops[0]['signal_camera']='up'
        _,d=self.release(c,'traffic_light_green',45.)
        self.assertFalse(d['release_stable']);self.assertEqual(d['signal_camera'],'up')
        for i in range(4):
            s=self.signals('traffic_light_green',10+i);s['camera']='up'
            _,d=c.decide(45.,s,2.+i*.25)
        self.assertTrue(d['release_stable'])
        cap,d=c.decide(45.,self.signals('traffic_light_green',20),3.)
        self.assertEqual(cap,0.);self.assertFalse(d['release_stable'])
        s=self.signals('traffic_light_green',21);s['camera']='up'
        _,d=c.decide(45.,s,3.25)
        self.assertTrue(d['release_stable'])  # green on the right camera goes again

    def test_reviewed_uncontrolled_merge_has_no_fixed_speed(self):
        c=self.controller();c.stops[0].update(controlled=False)
        cap,d=c.decide(10.,None,0.);self.assertEqual(cap,float('inf'))
        for i in range(1,6):cap,d=c.decide(10.+8*i,None,i*.1)
        self.assertEqual(cap,float('inf'));self.assertEqual(d['state'],'clear')

    def test_unreviewed_merge_does_not_use_speed_cap_to_bypass_review(self):
        c=self.controller(reviewed=False);c.stops[0].update(controlled=False)
        cap,d=c.decide(45.,None,0.);self.assertEqual(cap,0.);self.assertFalse(d['release_stable'])

if __name__=='__main__':unittest.main()


class CrossingSpeedTests(unittest.TestCase):
    STOP = dict(id='first', s=100., maneuver='straight', controlled=True,
                reviewed=True, signal_roi=[.3, .05, .7, .6], clearance_m=20.)

    def drive_to(self, controller, start, end, green=True):
        """Advance 2 m per 0.25 s with a fresh camera frame each step."""
        caps = {}
        progress, now, frame = start, 0., 0
        while progress <= end:
            frame += 1
            signals = dict(valid=True, frame=frame, detections=[dict(
                label='traffic_light_green', confidence=.95,
                bbox_normalized=[.45, .2, .55, .3])]) if green else None
            caps[progress] = controller.decide(progress, signals, now)
            progress += 2.
            now += .25
        return caps

    def test_green_crossing_has_no_fixed_speed(self):
        # Intersection speeds come from the Map Tuner point speeds.
        c = TrafficDecision([dict(self.STOP)], 1000.)
        caps = self.drive_to(c, 40., 170.)
        for progress in (60., 76., 110., 154., 168.):
            self.assertEqual(caps[progress][0], float('inf'))
            self.assertNotIn('crossing_limit_kmh', caps[progress][1])


class MissedSignalHoldTests(unittest.TestCase):
    STOP = dict(id='first', s=100., traffic_light_id='light1', maneuver='straight',
                controlled=True, reviewed=True, signal_roi=[.3, .05, .7, .6],
                clearance_m=20.)
    # Camera results 0.2 s apart on the 2026-10-15 C016 approach:
    # G = green_left in the head ROI, M = no detection.
    LOGGED = 'GGMMGGMGGMGGG'

    def controller(self, hold, stoplines=True):
        lines = [dict(idx='first', x=100., y=0., traffic_light_id='light1')]
        return TrafficDecision([dict(self.STOP)], 1000., missed_signal_hold_s=hold,
                               stoplines=lines if stoplines else None)

    def frame(self, kind, number):
        heads = dict(M=[], G=[('traffic_light_green_left', .95)],
                     R=[('traffic_light_red', .95)], Y=[('traffic_light_yellow', .95)],
                     U=[('traffic_light_green', .5)])[kind]
        return dict(valid=True, frame=number, detections=[dict(
            label=label, confidence=score, bbox_normalized=[.45, .2, .55, .3])
            for label, score in heads])

    def run_pattern(self, c, pattern, progress=85.):
        """Guard steps at 20 Hz; each camera result stays latest for 0.2 s."""
        results = []
        for step in range(len(pattern) * 4):
            now = step * .05
            number = step // 4
            signals = self.frame(pattern[number], number)
            results.append((pattern[number], c.decide(progress, signals, now,
                pose=(85., 0., 0.), speed_mps=8., ros_now=now)))
        return results

    def test_logged_flicker_keeps_green_and_old_behaviour_brakes(self):
        for stoplines in (True, False):
            held = self.run_pattern(self.controller(.8, stoplines), self.LOGGED)
            self.assertTrue(all(cap == float('inf') for _, (cap, _) in held))
            self.assertTrue(any(detail.get('reason') == 'permit held through missed detection'
                                or detail.get('signal_held') for _, (_, detail) in held))
            old = self.run_pattern(self.controller(0., stoplines), self.LOGGED)
            self.assertTrue(any(cap < float('inf') for kind, (cap, _) in old if kind == 'M'))

    def test_red_or_yellow_after_green_stops_at_once(self):
        for kind in 'RY':
            results = self.run_pattern(self.controller(.8), 'GG' + kind)
            self.assertTrue(all(cap < float('inf') for k, (cap, _) in results if k == kind))

    def test_miss_longer_than_hold_stops(self):
        results = self.run_pattern(self.controller(.8), 'GMMMMMM')
        self.assertLess(results[-1][1][0], float('inf'))

    def test_uncertain_head_or_stale_camera_is_not_held(self):
        c = self.controller(.8)
        self.run_pattern(c, 'GG')
        cap, _ = c.decide(85., self.frame('U', 9), .45, pose=(85., 0., 0.),
                          speed_mps=8., ros_now=.45)
        self.assertLess(cap, float('inf'))
        c = self.controller(.8)
        self.run_pattern(c, 'GG')
        cap, _ = c.decide(85., None, .45, pose=(85., 0., 0.), speed_mps=8., ros_now=.45)
        self.assertLess(cap, float('inf'))
        cap, _ = c.decide(85., self.frame('M', 9), .5, pose=(85., 0., 0.),
                          speed_mps=8., ros_now=.5)
        self.assertLess(cap, float('inf'))

    def test_miss_without_earlier_green_is_not_held(self):
        results = self.run_pattern(self.controller(.8), 'MMG')
        self.assertTrue(all(cap < float('inf') for k, (cap, _) in results if k == 'M'))

    def test_invalid_hold_rejected(self):
        for value in (-.1, float('nan')):
            with self.assertRaises(ValueError):
                TrafficDecision([dict(self.STOP)], 1000., missed_signal_hold_s=value)
