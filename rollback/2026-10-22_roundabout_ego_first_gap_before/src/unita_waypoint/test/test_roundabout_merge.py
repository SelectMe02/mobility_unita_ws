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

