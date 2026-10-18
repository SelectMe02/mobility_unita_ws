"""Sector 11: compact following, MOBIL lane changes, R3 priority."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/local_avoidance_core.py'
spec = importlib.util.spec_from_file_location('local_avoidance_core', SCRIPT)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)

ROAD = 2000
LANE_Y = {1: 0., 2: 3.6, 3: 7.2}
CRUISE = 100./3.6


def planner(highway=True, r3_end=None, r2_end=None, sector_10_until=0, **kwargs):
    route = [(float(i), 0.) for i in range(ROAD)]
    ends = {2: r2_end or ROAD, 3: r3_end or ROAD}
    lanes = {n: [(float(i), LANE_Y[n]) for i in range(ends[n])] for n in (2, 3)}
    indices = {n: {i: (float(i), LANE_Y[n]) for i in range(ends[n])} for n in (2, 3)}
    sectors = [10 if i < sector_10_until else 11 for i in range(ROAD)]
    return core.LocalAvoidancePlanner(route, sectors, lanes, lane_indices=indices,
                                      highway_sectors=(11,) if highway else (), **kwargs)


def car(x, lane, speed=0., kind='dynamic'):
    return core.Obstacle(x, LANE_Y[lane], 1.2, vx=speed, kind=kind if speed else 'static',
                         length=4.5, width=1.9)


def point_at(path, distance_m):
    remaining = distance_m
    for a, b in zip(path, path[1:]):
        length = core.distance(a, b)
        if length >= remaining and length > 0.:
            t = remaining/length
            return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))
        remaining -= length
    return path[-1]


def drive(p, ego, speed, traffic, seconds, dt=.1, desired=CRUISE):
    """Ego follows each decision path; its speed tracks the planner cap."""
    log = []
    for k in range(int(seconds/dt)):
        obstacles = traffic(k*dt)
        index = max(0, min(ROAD-2, int(round(ego[0]))))
        decision = p.plan(index, ego, speed, obstacles, desired_speed_mps=desired)
        log.append((decision, ego, speed, p.current_lane, obstacles))
        target = min(desired, decision.speed_cap_kmh/3.6)
        speed = max(0., speed+max(-6.*dt, min(2.*dt, target-speed)))
        ego = (point_at(decision.path, speed*dt) if len(decision.path) >= 2
               else (ego[0]+speed*dt, ego[1]))
    return log, ego, speed


class HighwayPlannerTests(unittest.TestCase):
    def test_stable_neighbour_in_margin_is_not_a_straight_lane_leader(self):
        p = planner()
        p.highway_active = True
        # Ego is offset toward R2; the body clears it by 30.4 cm, inside
        # the former 35 cm margin. R2's box remains inside its own lane.
        path = [(0., 1.4), (55., 1.4)]
        neighbour = car(20., 2, speed=15.)
        self.assertIsNotNone(p._lead(path, path[0], [neighbour], 55., lane=1))
        neighbour.raceline, neighbour.lane_sector = 2, 11
        neighbour.lane_stable, neighbour.lane_boundary_crossing = True, False
        self.assertIsNone(p._lead(path, path[0], [neighbour], 55., lane=1))
        # Same-lane following and lane-change envelopes keep the old margin.
        self.assertIsNotNone(p._lead(path, path[0], [neighbour], 55., lane=2))
        p.transition = {'target': 2}
        self.assertIsNotNone(p._lead(path, path[0], [neighbour], 55., lane=1))

    def test_lane_label_never_hides_body_overlap_or_boundary_crossing(self):
        p = planner()
        p.highway_active = True
        neighbour = car(20., 2, speed=0.)
        neighbour.raceline, neighbour.lane_sector = 2, 11
        neighbour.lane_stable, neighbour.lane_boundary_crossing = True, False
        path = [(0., 1.8), (55., 1.8)]
        lead = p._lead(path, path[0], [neighbour], 55., lane=1)
        self.assertIsNotNone(lead)
        self.assertLess(p._idm_cap(CRUISE, CRUISE, lead), CRUISE*3.6)
        path = [(0., 1.4), (55., 1.4)]
        neighbour.lane_boundary_crossing = True
        self.assertIsNotNone(p._lead(path, path[0], [neighbour], 55., lane=1))
        neighbour.lane_boundary_crossing = False
        neighbour.lane_stable = False
        self.assertIsNotNone(p._lead(path, path[0], [neighbour], 55., lane=1))
        neighbour.lane_stable = True
        p.highway_active = False
        self.assertIsNotNone(p._lead(path, path[0], [neighbour], 55., lane=1))

    def test_free_road_moves_to_r3_at_full_speed(self):
        p = planner()
        decision = p.plan(0, (0., 0.), CRUISE, [], desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'LANE_CHANGING')
        self.assertEqual(p.transition['target'], 2)
        self.assertEqual(p.transition['shape'], 'quintic')
        # No speed precondition and no slow-down for a clear change.
        self.assertTrue(math.isinf(decision.speed_cap_kmh))
        log, ego, _ = drive(p, (0., 0.), CRUISE, lambda t: [], 20.)
        self.assertAlmostEqual(ego[1], LANE_Y[3], delta=.2)
        self.assertEqual(p.current_lane, 3)
        self.assertEqual(log[-1][0].state, 'RACELINE')
        self.assertNotIn('BLOCKED', {d.state for d, *_ in log})

    def test_quintic_change_starts_without_a_lateral_kink(self):
        p = planner()
        decision = p.plan(0, (0., 0.), CRUISE, [], desired_speed_mps=CRUISE)
        start = point_at(decision.path, 5.)
        self.assertLess(start[1], .01)
        # Lateral acceleration limit sets the length at speed.
        lateral = LANE_Y[2]
        expected = CRUISE*math.sqrt(core.QUINTIC_PEAK_CURVATURE*lateral/p.lane_change_lateral_accel)
        self.assertAlmostEqual(p.transition['length'], expected, delta=1.)

    def test_static_obstacle_in_r3_is_passed_at_speed_without_stopping(self):
        p = planner()
        obstacle = car(150., 3)
        log, ego, _ = drive(p, (0., LANE_Y[3]), CRUISE, lambda t: [obstacle], 14.)
        states = [d.state for d, *_ in log]
        self.assertIn('OVERTAKING', states)
        self.assertNotIn('BLOCKED', states)
        self.assertGreater(min(s for _, _, s, _, _ in log), 10.)
        self.assertGreater(ego[0], 200.)
        # Back in R3 after the pass.
        self.assertAlmostEqual(ego[1], LANE_Y[3], delta=.3)

    def test_close_static_obstacle_shortens_change_and_slows_instead_of_waiting(self):
        p = planner()
        decision = p.plan(0, (0., LANE_Y[3]), CRUISE, [car(50., 3)], desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'OVERTAKING')
        self.assertLess(p.transition['length'], CRUISE*3.)
        self.assertLess(decision.speed_cap_kmh, 100.)
        self.assertGreater(decision.speed_cap_kmh, 30.)

    def test_occupied_neighbour_tails_static_obstacle_without_planner_stop(self):
        p = planner()
        decision = p.plan(0, (0., LANE_Y[3]), 15., [car(50., 3), car(1., 2, 15.)],
                          desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'TAILING')
        self.assertGreater(decision.speed_cap_kmh, 0.)

    def test_fast_follower_in_target_lane_blocks_overtake_until_it_passes(self):
        p = planner()
        leader = car(35., 3, 10.)
        decision = p.plan(0, (0., LANE_Y[3]), 15., [leader, car(-12., 2, 30.)],
                          desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'TAILING')
        self.assertEqual(decision.reason, 'mobil_unsafe')
        decision = p.plan(0, (0., LANE_Y[3]), 15., [leader, car(-50., 2, 15.)],
                          desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'OVERTAKING')

    def test_slightly_slower_leader_is_followed_in_r3(self):
        p = planner()
        decision = p.plan(0, (0., LANE_Y[3]), 27., [car(50., 3, 27.)], desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'TAILING')
        self.assertIsNone(p.transition)

    def test_slow_leader_is_overtaken_then_r3_is_resumed(self):
        p = planner()
        traffic = lambda t: [car(60.+15.*t, 3, 15.)]
        log, ego, _ = drive(p, (0., LANE_Y[3]), 20., traffic, 30.)
        states = [d.state for d, *_ in log]
        self.assertIn('OVERTAKING', states)
        self.assertIn('RETURNING', states)
        self.assertNotIn('BLOCKED', states)
        self.assertGreater(ego[0], 60.+15.*30.)
        self.assertAlmostEqual(ego[1], LANE_Y[3], delta=.3)
        # IDM following: no full brake to a standstill.
        self.assertGreater(min(s for _, _, s, _, _ in log), 10.)

    def test_ending_r3_is_left_even_against_the_r3_bias(self):
        p = planner(r3_end=250)
        log, ego, _ = drive(p, (0., LANE_Y[3]), CRUISE, lambda t: [], 12.)
        self.assertAlmostEqual(ego[1], LANE_Y[2], delta=.3)
        self.assertGreater(ego[0], 250.)

    def test_change_is_aborted_when_target_lane_is_invaded_early(self):
        p = planner()
        leader = car(45., 3, 12.)
        decision = p.plan(0, (0., LANE_Y[3]), 20., [leader], desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'OVERTAKING')
        intruder = car(-9., 2, 32.)
        for step in range(p.cancel_confirm_scans):
            x = 1.+step
            decision = p.plan(int(x), (x, LANE_Y[3]-.2), 20., [leader, intruder],
                              desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'CANCELLING')
        self.assertEqual(p.transition['target'], 3)

    def test_outside_highway_sector_keeps_original_logic(self):
        scenario = dict(index=0, ego_xy=(0., LANE_Y[3]), speed_mps=8.,
                        obstacles=[car(30., 3)], desired_speed_mps=CRUISE)
        old = planner(highway=False, sector_10_until=500).plan(**scenario)
        new = planner(sector_10_until=500).plan(**scenario)
        self.assertEqual((old.state, old.speed_cap_kmh, old.reason, old.path),
                         (new.state, new.speed_cap_kmh, new.reason, new.path))

    def test_unconfirmed_rear_clutter_does_not_block_the_r3_change(self):
        # 2026-10-18 21:29 run: small unconfirmed returns behind at the R2
        # edge were assumed at road speed and blocked or cancelled every change.
        p = planner()
        clutter = [core.Obstacle(-7., 2.4, .3, kind='unknown', confirmed=False,
                                 velocity_reliable=False, length=.3, width=2.),
                   core.Obstacle(-15., 2.6, .3, kind='unknown', confirmed=False,
                                 velocity_reliable=False, length=.7, width=.7)]
        decision = p.plan(0, (0., 0.), 25., clutter, desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'LANE_CHANGING')
        self.assertEqual(p.transition['target'], 2)

    def test_unconfirmed_wide_patch_ahead_does_not_cancel_the_change(self):
        p = planner()
        p.plan(0, (0., 0.), 25., [], desired_speed_mps=CRUISE)
        patch = core.Obstacle(40., 4., 5., kind='unknown', confirmed=False,
                              velocity_reliable=False, length=1.2, width=10.5)
        for step in range(p.cancel_confirm_scans+2):
            x = 2.+2.*step
            decision = p.plan(int(x), (x, .3), 25., [patch], desired_speed_mps=CRUISE)
        self.assertNotEqual(decision.state, 'CANCELLING')
        self.assertEqual(p.transition['target'], 2)

    def test_cancel_into_an_ending_lane_restarts_when_a_merge_fits(self):
        # The run stopped for good while CANCELLING back into ending R1.
        p = planner(r3_end=None, r2_end=None)
        p.route_lanes = [1]*ROAD
        # R1 ends at x=124, R2/R3 continue. At x=110 a short, slow
        # one-lane merge still fits within the steering limit (~10.6 m).
        p.lane_indices[1] = {}
        p.route = list(p.route)
        p.plan(100, (100., 0.), 3., [], desired_speed_mps=CRUISE)
        p.transition = dict(source=2, target=1, start=p._progress_at(100, (100., 0.)),
                            length=30., state='CANCELLING', shape='quintic', highway=True,
                            speed_cap_kmh=None)
        p._lane_point = (lambda lane, point, index, orig=p._lane_point:
                         None if lane == 1 and index > 124 else orig(lane, point, index))
        decision = p.plan(110, (110., 0.), 0., [], desired_speed_mps=CRUISE)
        self.assertNotEqual(decision.state, 'CANCELLING')
        self.assertGreater(decision.speed_cap_kmh, 0.)
        self.assertEqual(p.transition['target'], 2)
        self.assertLessEqual(p.transition['length'], 14.)
        self.assertLessEqual(decision.speed_cap_kmh,
                             p._length_speed(LANE_Y[2], p.transition['length'])*3.6)
        log, ego, _ = drive(p, (110., 0.), 0., lambda t: [], 20.)
        self.assertGreater(ego[0], 150.)
        self.assertEqual(p.current_lane, 3)

    def test_insufficient_steering_room_at_lane_end_keeps_stop(self):
        p = planner()
        original = p._lane_point
        p._lane_point = lambda lane, point, index: (
            None if lane == 1 and index > 118 else original(lane, point, index))
        decision = p.plan(110, (110., 0.), 0., [], desired_speed_mps=CRUISE)
        self.assertIsNone(p.transition)
        self.assertEqual(decision.speed_cap_kmh, 0.)

    def test_unconfirmed_near_obstacle_brakes_on_first_change_scan(self):
        p = planner()
        obstacle = core.Obstacle(4., 0., .5, confirmed=False,
                                 velocity_reliable=False, length=.5, width=1.)
        decision = p.plan(0, (0., 0.), 15., [obstacle], desired_speed_mps=CRUISE)
        self.assertEqual(decision.speed_cap_kmh, 0.)

    def test_unconfirmed_car_beside_target_lane_keeps_immediate_gap(self):
        p = planner()
        obstacle = car(0., 2, 15.)
        obstacle.confirmed = obstacle.velocity_reliable = False
        decision = p.plan(0, (0., 0.), 15., [obstacle], desired_speed_mps=CRUISE)
        self.assertIsNone(p.transition)
        self.assertEqual(decision.reason, 'lane_change_gap')

    def test_lane_only_crossed_ignores_its_far_slow_leader(self):
        p = planner()
        # R2 leader far ahead and only slightly slower: R1 -> R2 -> R3 still goes.
        decision = p.plan(0, (0., 0.), 15., [car(54., 2, 13.)], desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'LANE_CHANGING')

    def test_invalid_highway_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            planner(idm_time_gap_s=0.)

    def test_same_speed_leader_with_three_metre_gap_does_not_brake(self):
        for speed in (30./3.6, 60./3.6, CRUISE):
            p = planner(sector_min_speed_kmh=30.)
            x = p.front_extent+2.25+3.1
            decision = p.plan(0, (0., LANE_Y[3]), speed,
                              [car(x, 3, speed), car(0., 2, speed)],
                              desired_speed_mps=CRUISE)
            self.assertGreaterEqual(decision.speed_cap_kmh+1e-8, speed*3.6)
            self.assertFalse(p.highway_emergency)

    def test_ordinary_following_has_thirty_kmh_floor(self):
        p = planner(sector_min_speed_kmh=30.)
        leader = car(p.front_extent+2.25+15., 3, 5.)
        decision = p.plan(0, (0., LANE_Y[3]), 30./3.6,
                          [leader, car(0., 2, 30./3.6)], desired_speed_mps=CRUISE)
        self.assertGreaterEqual(decision.speed_cap_kmh, 30.)
        self.assertFalse(p.highway_emergency)

    def test_collision_envelope_overrides_floor_for_a_stopped_close_car(self):
        p = planner(sector_min_speed_kmh=30.)
        for gap in (1., .2):
            obstacle = car(p.front_extent+2.25+gap, 3)
            decision = p.plan(0, (0., LANE_Y[3]), 20.,
                              [obstacle, car(0., 2, 20.)], desired_speed_mps=CRUISE)
            self.assertLess(decision.speed_cap_kmh, 30.)
            self.assertTrue(p.highway_emergency)
        self.assertEqual(decision.speed_cap_kmh, 0.)

    def test_target_lane_accepts_same_speed_cars_with_three_metre_bumper_gaps(self):
        p = planner(sector_min_speed_kmh=30.)
        front = car(p.front_extent+2.25+3.1, 2, 20.)
        rear = car(-p.rear_extent-2.25-3.1, 2, 20.)
        decision = p.plan(0, (0., 0.), 20., [front, rear], desired_speed_mps=CRUISE)
        self.assertEqual(decision.state, 'LANE_CHANGING')
        self.assertEqual(p.transition['target'], 2)

    def test_target_lane_rejects_either_bumper_gap_below_three_metres(self):
        for front_gap, rear_gap in ((2.9, 10.), (10., 2.9)):
            p = planner(sector_min_speed_kmh=30.)
            front = car(p.front_extent+2.25+front_gap, 2, 20.)
            rear = car(-p.rear_extent-2.25-rear_gap, 2, 20.)
            p.plan(0, (0., 0.), 20., [front, rear], desired_speed_mps=CRUISE)
            self.assertIsNone(p.transition)

    def test_confirmed_road_surface_evidence_is_used_only_on_highway(self):
        road = core.Obstacle(4., 0., .4, confirmed=False, velocity_reliable=False,
                             length=.3, width=.6, road_surface=True)
        p = planner(sector_min_speed_kmh=30.)
        decision = p.plan(0, (0., 0.), 20., [road], desired_speed_mps=CRUISE)
        self.assertGreaterEqual(decision.speed_cap_kmh, 30.)
        self.assertEqual(p.highway_road_returns_ignored, 1)
        normal = planner(highway=False)
        decision = normal.plan(0, (0., 0.), 20., [road], desired_speed_mps=CRUISE)
        self.assertEqual(decision.speed_cap_kmh, 0.)

    def test_low_speed_lane_change_is_planned_for_the_cruise_floor(self):
        p = planner(sector_min_speed_kmh=30.)
        decision = p.plan(0, (0., 0.), 2., [], desired_speed_mps=CRUISE)
        self.assertGreaterEqual(decision.speed_cap_kmh, 30.)


if __name__ == '__main__':
    unittest.main()
