import importlib.util
import math
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/local_avoidance_core.py'
spec = importlib.util.spec_from_file_location('roundabout_local_avoidance_core', SCRIPT)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def planner(sector=9):
    route = [(float(x), 0.) for x in range(101)]
    return core.LocalAvoidancePlanner(
        route, [sector]*len(route), {},
        roundabout_conflict_point=(30., 0.),
        roundabout_approach_points=[(30., 30.), (30., 0.), (45., 0.)],
        roundabout_sector=9, roundabout_conflict_radius_m=3.5,
        roundabout_monitor_distance_m=30., roundabout_approach_width_m=4.5,
        roundabout_unknown_speed_kmh=20., roundabout_time_gap_s=2.5,
        roundabout_clear_scans=3)


def circulating(y=10., reliable=True, track_id=7):
    return core.Obstacle(30., y, .8, vy=-20./3.6, kind='dynamic',
                         confirmed=reliable, velocity_reliable=reliable,
                         length=4.6, width=1.9, yaw=-math.pi/2., track_id=track_id)


def test_circulating_car_with_overlapping_arrival_makes_ego_yield():
    p = planner()
    decision = p.plan(10, (10., 0.), 5., [circulating()], desired_speed_mps=5.)
    assert decision.state == 'ROUNDABOUT_YIELD'
    assert decision.reason == 'roundabout_conflict'
    assert decision.obstacle_kind == 'dynamic'
    assert math.isfinite(decision.speed_cap_kmh)
    assert p.roundabout_state == 'YIELD'
    assert p.roundabout_threat_id == 7
    assert p.roundabout_actor_eta_s is not None


def test_new_left_track_uses_twenty_kmh_until_velocity_is_reliable():
    p = planner()
    new_track = circulating(reliable=False, track_id=11)
    new_track.vx = new_track.vy = 0.
    decision = p.plan(10, (10., 0.), 5., [new_track], desired_speed_mps=5.)
    assert decision.state == 'ROUNDABOUT_YIELD'
    assert p.roundabout_threat_id == 11


def test_clear_entry_is_released_after_three_scans_without_changing_speed():
    p = planner()
    decisions = [p.plan(10, (10., 0.), 5., [], desired_speed_mps=5.)
                 for _ in range(3)]
    assert [d.state for d in decisions[:2]] == ['ROUNDABOUT_APPROACH']*2
    assert decisions[2].state == 'RACELINE'
    assert math.isinf(decisions[2].speed_cap_kmh)
    assert p.roundabout_state == 'ENTRY_CLEAR'
    assert not p.roundabout_gap_accepted
    # A clear road seen before Ego has yielded is not a departure commitment;
    # a vehicle appearing later during the original approach must still stop it.
    later = p.plan(10, (10., 0.), 0., [circulating()], desired_speed_mps=5.)
    assert later.state == 'ROUNDABOUT_YIELD'


def test_clear_confirmation_counts_distinct_lidar_frames_only():
    p = planner()
    repeated = [p.plan(10, (10., 0.), 5., [], desired_speed_mps=5.,
                       observation_stamp=10.) for _ in range(5)]
    assert all(d.state == 'ROUNDABOUT_APPROACH' for d in repeated)
    assert p.roundabout_clear_votes == 1
    p.plan(10, (10., 0.), 5., [], desired_speed_mps=5., observation_stamp=10.1)
    released = p.plan(10, (10., 0.), 5., [], desired_speed_mps=5.,
                      observation_stamp=10.2)
    assert released.state == 'RACELINE'
    assert p.roundabout_state == 'ENTRY_CLEAR'


def test_vehicle_outside_circulating_corridor_keeps_existing_behavior():
    p = planner()
    p.roundabout_clear_votes = p.roundabout_clear_scans
    unrelated = core.Obstacle(15., 15., 1., vy=-5., kind='dynamic')
    decision = p.plan(10, (10., 0.), 5., [unrelated], desired_speed_mps=5.)
    assert decision.state == 'RACELINE'
    assert math.isinf(decision.speed_cap_kmh)


def test_large_arrival_gap_keeps_normal_roundabout_entry():
    p = planner()
    p.roundabout_clear_votes = p.roundabout_clear_scans
    late_car = circulating(30.)
    decision = p.plan(20, (20., 0.), 5., [late_car], desired_speed_mps=5.)
    assert decision.state != 'ROUNDABOUT_YIELD'
    assert p.roundabout_state == 'ENTRY_CLEAR'
    assert math.isinf(decision.speed_cap_kmh)


def test_car_that_clears_well_before_ego_does_not_hold_entry():
    p = planner()
    p.roundabout_clear_votes = p.roundabout_clear_scans
    clearing = circulating(3.)
    decision = p.plan(0, (0., 0.), 5., [clearing], desired_speed_mps=5.)
    assert decision.state != 'ROUNDABOUT_YIELD'
    assert p.roundabout_state == 'ENTRY_CLEAR'


def test_guard_is_disabled_in_every_other_sector():
    p = planner(sector=8)
    decision = p.plan(10, (10., 0.), 5., [circulating()], desired_speed_mps=5.)
    assert decision.state == 'RACELINE'
    assert p.roundabout_state == 'OUTSIDE'


def test_stopped_ego_accepts_clear_gap_after_yield_and_ignores_following_car():
    p = planner()
    first = p.plan(10, (10., 0.), 5., [circulating()], desired_speed_mps=5.,
                   observation_stamp=1.)
    assert first.state == 'ROUNDABOUT_YIELD'
    assert p.roundabout_yield_seen

    released = None
    for stamp in (2., 3., 4.):
        released = p.plan(10, (10., 0.), 0., [], desired_speed_mps=5.,
                          observation_stamp=stamp)
    assert released.state == 'ROUNDABOUT_ENTRY'
    assert released.reason == 'roundabout_gap_accepted'
    assert p.roundabout_gap_accepted

    following = p.plan(10, (10., 0.), 0., [circulating(y=15., track_id=12)],
                       desired_speed_mps=5., observation_stamp=5.)
    assert following.state == 'ROUNDABOUT_ENTRY'
    assert p.roundabout_state == 'ENTRY_CLEAR'
    assert p.roundabout_threat_id is None


def test_clear_gap_is_not_latched_while_ego_is_still_moving():
    p = planner()
    p.plan(10, (10., 0.), 5., [circulating()], desired_speed_mps=5.,
           observation_stamp=1.)
    for stamp in (2., 3., 4.):
        p.plan(10, (10., 0.), 2., [], desired_speed_mps=5.,
               observation_stamp=stamp)
    assert not p.roundabout_gap_accepted
    following = p.plan(10, (10., 0.), 2., [circulating(y=15., track_id=12)],
                       desired_speed_mps=5., observation_stamp=5.)
    assert following.state == 'ROUNDABOUT_YIELD'


def test_front_past_merge_boundary_is_committed_without_guard_brake():
    p = planner()
    decision = p.plan(24, (24., 0.), 5., [circulating(4.)], desired_speed_mps=5.)
    assert decision.state != 'ROUNDABOUT_YIELD'
    assert p.roundabout_state == 'COMMITTED'



def _yield_then_clear(p, ego_speed, desired=30./3.6):
    p.plan(10, (10., 0.), ego_speed, [circulating()], desired_speed_mps=desired,
           observation_stamp=1.)
    assert p.roundabout_yield_seen
    for stamp in (2., 3., 4.):
        p.plan(10, (10., 0.), ego_speed, [], desired_speed_mps=desired,
               observation_stamp=stamp)
    assert p.roundabout_clear_votes == p.roundabout_clear_scans


def _car_with_lead(p, lead_s, desired=30./3.6, track_id=31):
    # Place a following car so its merge ETA is ego ETA + lead_s.
    ego_eta = max(0., 20.-p.front_extent)/desired
    remaining = (ego_eta+lead_s)*20./3.6+p.roundabout_conflict_radius_m+.8
    return circulating(y=remaining, track_id=track_id)


def test_rolling_ego_enters_ahead_of_following_car_after_first_car_cleared():
    p = planner()
    _yield_then_clear(p, ego_speed=10./3.6)
    decision = p.plan(10, (10., 0.), 10./3.6, [_car_with_lead(p, 1.63)],
                      desired_speed_mps=30./3.6, observation_stamp=5.)
    assert decision.state != 'ROUNDABOUT_YIELD'
    assert p.roundabout_state == 'ENTRY_CLEAR'


def test_ego_first_margin_still_yields_to_a_tighter_following_car():
    p = planner()
    _yield_then_clear(p, ego_speed=10./3.6)
    decision = p.plan(10, (10., 0.), 10./3.6, [_car_with_lead(p, 1.2)],
                      desired_speed_mps=30./3.6, observation_stamp=5.)
    assert decision.state == 'ROUNDABOUT_YIELD'


def test_first_approach_keeps_symmetric_gap_for_ego_first_arrival():
    p = planner()
    p.roundabout_clear_votes = p.roundabout_clear_scans
    decision = p.plan(10, (10., 0.), 10./3.6, [_car_with_lead(p, 1.63)],
                      desired_speed_mps=30./3.6)
    assert decision.state == 'ROUNDABOUT_YIELD'


def test_crawling_ego_keeps_symmetric_gap_after_yield():
    p = planner()
    _yield_then_clear(p, ego_speed=.6)
    assert not p.roundabout_gap_accepted
    decision = p.plan(10, (10., 0.), .6, [_car_with_lead(p, 1.63)],
                      desired_speed_mps=30./3.6, observation_stamp=5.)
    assert decision.state == 'ROUNDABOUT_YIELD'


def test_following_car_arriving_before_ego_still_needs_full_gap():
    p = planner()
    _yield_then_clear(p, ego_speed=10./3.6)
    decision = p.plan(10, (10., 0.), 10./3.6, [_car_with_lead(p, -1.)],
                      desired_speed_mps=30./3.6, observation_stamp=5.)
    assert decision.state == 'ROUNDABOUT_YIELD'


def test_car_still_inside_merge_blocks_rolling_ego_after_yield():
    p = planner()
    _yield_then_clear(p, ego_speed=10./3.6)
    decision = p.plan(10, (10., 0.), 10./3.6, [circulating(y=1., track_id=40)],
                      desired_speed_mps=30./3.6, observation_stamp=5.)
    assert decision.state == 'ROUNDABOUT_YIELD'


def test_yield_release_uses_two_distinct_clear_scans_while_still_rolling():
    p = planner()
    p.plan(10, (10., 0.), 2., [circulating()], desired_speed_mps=5.,
           observation_stamp=1.)
    for _ in range(4):
        held = p.plan(10, (10., 0.), 1., [], desired_speed_mps=5.,
                      observation_stamp=2.)
        assert held.state == 'ROUNDABOUT_APPROACH'
    released = p.plan(10, (10., 0.), 1., [], desired_speed_mps=5.,
                      observation_stamp=2.1)
    assert released.reason == 'roundabout_rolling_entry'
    assert math.isinf(released.speed_cap_kmh)
    assert not p.roundabout_gap_accepted


def test_low_speed_rolling_release_rechecks_tight_following_car():
    p = planner()
    _yield_then_clear(p, ego_speed=3./3.6)
    released = p.plan(10, (10., 0.), 3./3.6, [_car_with_lead(p, 1.63)],
                      desired_speed_mps=30./3.6, observation_stamp=5.)
    assert released.reason == 'roundabout_rolling_entry'
    assert not p.roundabout_gap_accepted
    blocked = p.plan(10, (10., 0.), 3./3.6, [_car_with_lead(p, 1.2)],
                     desired_speed_mps=30./3.6, observation_stamp=6.)
    assert blocked.state == 'ROUNDABOUT_YIELD'


def test_creeping_over_commit_boundary_cannot_cancel_rejected_gap():
    p = planner()
    first = p.plan(22, (22., 0.), 1., [circulating()], desired_speed_mps=5.,
                   observation_stamp=1.)
    assert first.state == 'ROUNDABOUT_YIELD'
    over_boundary = p.plan(24, (24., 0.), .8, [circulating()],
                           desired_speed_mps=5., observation_stamp=1.1)
    assert over_boundary.state == 'ROUNDABOUT_YIELD'
    assert over_boundary.speed_cap_kmh == 0.


def test_rolling_release_preserves_path_obstacle_stop():
    p = planner()
    _yield_then_clear(p, ego_speed=1.)
    blocked = p.plan(10, (10., 0.), 1., [core.Obstacle(13., 0., 1.)],
                     desired_speed_mps=5., observation_stamp=5.)
    assert blocked.speed_cap_kmh == 0.
