"""S11: R1 -> R2 -> R3 only, including in-progress hazard handling."""
import unittest
import test_highway_planner as f


def planner(**kwargs):
    return f.planner(s11_forward_only=True, highway_keep_lane_priority=True, **kwargs)


class ForwardOnlyTests(unittest.TestCase):
    def test_clear_road_prioritizes_next_lane_despite_keep_lane_policy(self):
        for lane in (1, 2):
            p = planner()
            p.plan(10, (10., f.LANE_Y[lane]), 15., [], desired_speed_mps=f.CRUISE)
            self.assertEqual((p.transition['source'], p.transition['target']), (lane, lane+1))
            self.assertTrue(p.transition['s11_forward_only'])

    def test_r3_never_selects_r2_even_with_front_obstacle_or_lane_end(self):
        for end in (None, 160):
            p = planner(r3_end=end)
            for stamp in (1., 2., 3., 4.):
                d = p.plan(60, (60., 7.2), 15., [f.car(80., 3)],
                           desired_speed_mps=f.CRUISE, observation_stamp=stamp)
                self.assertIsNone(p.transition)
                self.assertTrue(all(abs(y-7.2) < .01 for _, y in d.path))
                self.assertLess(d.speed_cap_kmh, 54.)

    def test_blocked_r3_does_not_send_r2_vehicle_back_to_r1(self):
        p = planner()
        for stamp in (1., 2., 3., 4.):
            p.plan(60, (60., 3.6), 15., [f.car(80., 2), f.car(60., 3)],
                   desired_speed_mps=f.CRUISE, observation_stamp=stamp)
            self.assertIsNone(p.transition)
            self.assertEqual(p.current_lane, 2)

    def test_rear_gap_checks_still_block_unsafe_forward_change(self):
        p = planner()
        p.plan(60, (60., 0.), 15., [f.car(55., 2, 30.)], desired_speed_mps=f.CRUISE)
        self.assertIsNone(p.transition)

    def test_approved_change_does_not_cancel_for_front_side_or_rear_hazard(self):
        for obstacles in ([f.car(70., 2)], [f.car(61., 2, 15.)], [f.car(50., 2, 30.)]):
            p = planner()
            p.plan(60, (60., 0.), 15., [], desired_speed_mps=f.CRUISE)
            original = dict(p.transition)
            for stamp in (1., 1.1, 1.2, 1.3, 2.):
                d = p.plan(60, (60., 0.), 15., obstacles, desired_speed_mps=f.CRUISE,
                           observation_stamp=stamp)
                self.assertEqual(p.transition, original)
                self.assertNotEqual(d.state, 'CANCELLING')
                self.assertTrue(d.path)

    def test_complete_sequence_without_backward_transition(self):
        p = planner()
        ego = (60., 0.)
        seen = set()
        for step in range(160):
            d = p.plan(int(ego[0]), ego, 15., [], desired_speed_mps=f.CRUISE,
                       observation_stamp=step*.1)
            if p.transition:
                seen.add((p.transition['source'], p.transition['target']))
                self.assertEqual(p.transition['target'], p.transition['source']+1)
                ego = f.point_at(d.path, 1.5)
            elif p.current_lane == 3:
                break
            else:
                ego = f.point_at(d.path, 1.5) if d.path else (ego[0]+1.5, ego[1])
        self.assertEqual(seen, {(1, 2), (2, 3)})
        self.assertEqual(p.current_lane, 3)
        self.assertLessEqual(abs(ego[1]-7.2), .35)

    def test_progress_end_alone_is_not_completion(self):
        p = planner()
        p.plan(60, (60., 0.), 15., [], desired_speed_mps=f.CRUISE)
        original = dict(p.transition)
        end = int(original['start']+original['length'])+1
        p.plan(end, (float(end), 1.), 15., [], desired_speed_mps=f.CRUISE)
        self.assertEqual(p.transition, original)

    def test_nearest_lane_jitter_cannot_reopen_completed_lower_lane(self):
        p = planner()
        p.s11_reached_lane = 3
        d = p.plan(60, (60., 5.2), 15., [], desired_speed_mps=f.CRUISE)
        self.assertEqual(p.current_lane, 3)
        self.assertIsNone(p.transition)
        self.assertGreater(d.path[-1][1], 7.)

    def test_temporary_missing_target_preserves_approved_target(self):
        p = planner()
        p.plan(60, (60., 0.), 15., [], desired_speed_mps=f.CRUISE)
        original = dict(p.transition)
        p.lane_indices[2].pop(60)
        d = p.plan(60, (60., 0.), 15., [], desired_speed_mps=f.CRUISE)
        self.assertEqual(p.transition, original)
        self.assertTrue(d.path)

    def test_other_highway_sector_keeps_previous_direction_policy(self):
        p = planner(r3_end=160)
        p.sectors = [10]*len(p.sectors)
        p.highway_sectors = (10,)
        p.plan(60, (60., 7.2), 15., [], desired_speed_mps=f.CRUISE)
        self.assertEqual(p.transition['target'], 2)
        self.assertNotIn('s11_forward_only', p.transition)


if __name__ == '__main__':
    unittest.main()
