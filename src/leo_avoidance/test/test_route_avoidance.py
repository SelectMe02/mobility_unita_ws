import math

import numpy as np

from leo_avoidance.route_avoidance import (RouteAvoidance, RouteAvoidanceConfig,
                                           RouteObstacle, stopping_speed)

# Straight route along +x; left (+d) is +y.
ROUTE = [(float(x), 0.) for x in range(0, 401, 2)]


# Dashed line to a same-direction lane 3.5 m left along the whole test route.
LEFT_LANE = [dict(start_s=0., end_s=400., side='left', offset_m=3.5)]


def planner(lane_changes=LEFT_LANE, **overrides):
    return RouteAvoidance(ROUTE, RouteAvoidanceConfig(**overrides), lane_changes)


def offset_at(decision, x):
    return min(decision.path, key=lambda p: abs(p[0] - x))[1]


def drive(avoid, obstacles, start_x, end_x, speed=5.5, step=1.):
    """Advance Ego along its own planned path; return every decision."""
    decisions = []
    x, y = start_x, 0.
    while x < end_x:
        decision = avoid.update((x, y), speed, obstacles)
        decisions.append((x, y, decision))
        if decision.speed_cap_mps < .3:
            break
        x += step
        y = offset_at(decision, x)
    return decisions


def test_static_block_in_lane_is_passed_on_clear_opposite_lane_and_rejoined():
    avoid = planner()
    block = RouteObstacle(60., 0., 1.2)
    decisions = drive(avoid, [block], 0., 120.)
    assert decisions[-1][0] >= 119.  # never stopped
    beside = [y for x, y, _ in decisions if 57. <= x + 3.845 and x - .79 <= 61.2]
    assert beside and min(beside) > 3.  # whole body on the opposite lane beside it
    assert abs(decisions[-1][1]) < .05  # back on the route afterwards
    assert {d.state for _, _, d in decisions} <= {'FOLLOW', 'PASSING', 'WAIT_PASS'}


def test_oncoming_car_on_opposite_lane_makes_ego_wait_behind_the_block():
    avoid = planner()
    block = RouteObstacle(50., 0., 1.2)
    oncoming = RouteObstacle(90., 3.5, 1., vx=-8., dynamic=True, half_length=2.4)
    decision = avoid.update((10., 0.), 5.5, [block, oncoming])
    assert decision.state == 'WAIT_PASS'
    assert max(abs(y) for _, y in decision.path) < 1e-6
    gap = 50. - 1.2 - (10. + 3.845) - 5.
    assert math.isclose(decision.speed_cap_mps, stopping_speed(gap, 3., .4))


def test_pass_starts_once_the_oncoming_car_has_gone_by():
    avoid = planner()
    block = RouteObstacle(50., 0., 1.2)
    gone = RouteObstacle(-20., 3.5, 1., vx=-8., dynamic=True, half_length=2.4)
    decision = avoid.update((10., 0.), 5.5, [block, gone])
    assert decision.state == 'PASSING'


def test_stopped_close_behind_a_block_pulls_out_slowly_when_clear():
    avoid = planner()
    block = RouteObstacle(25., 0., 1.2)
    decision = avoid.update((15., 0.), 0., [block])
    assert decision.state == 'PASSING'
    assert decision.speed_cap_mps <= 7. / 3.6 + 1e-9


def test_red_stop_line_inside_the_pass_keeps_ego_behind_the_block():
    avoid = planner()
    block = RouteObstacle(50., 0., 1.2)
    decision = avoid.update((10., 0.), 5.5, [block], stop_line_distance_m=45.,
                            stop_line_permitted=False)
    assert decision.state == 'WAIT_PASS'
    decision = avoid.update((10., 0.), 5.5, [block], stop_line_distance_m=45.,
                            stop_line_permitted=True)
    assert decision.state == 'PASSING'


def test_pedestrian_walking_into_the_lane_is_yielded_to_before_it_enters():
    avoid = planner()
    # 1.9 m/s toward the lane from 4.5 m right, 20 m ahead of the bumper.
    ped = RouteObstacle(33.845, -4.5, .3, vy=1.9, dynamic=True)
    decision = avoid.update((10., 0.), 5.5, [ped])
    assert decision.state == 'YIELD'
    ahead = 33.845 - .3 - (10. + 3.845)
    assert math.isclose(decision.speed_cap_mps, stopping_speed(ahead - 3., 3., .4))
    # Closer in, the same walker forces a stop well before its line.
    decision = avoid.update((26., 0.), 5.5, [ped])
    assert decision.state == 'YIELD' and decision.speed_cap_mps < 2.


def test_pedestrian_walking_away_or_standing_at_the_kerb_is_ignored():
    avoid = planner()
    away = RouteObstacle(33.845, -4.5, .3, vy=-1.9, dynamic=True)
    standing = RouteObstacle(30., -2.6, .3)
    decision = avoid.update((10., 0.), 5.5, [away, standing])
    assert decision.state == 'FOLLOW'
    assert decision.speed_cap_mps == math.inf


def test_crossing_pedestrian_that_clears_before_arrival_does_not_slow_ego():
    avoid = planner()
    # Already leaving the lane on the far side.
    ped = RouteObstacle(40., 2.2, .3, vy=1.9, dynamic=True)
    decision = avoid.update((10., 0.), 5.5, [ped])
    assert decision.state == 'FOLLOW'


def test_slow_lead_vehicle_is_followed_not_passed():
    avoid = planner()
    lead = RouteObstacle(40., 0., 1., vx=2., dynamic=True, half_length=2.4)
    decision = avoid.update((10., 0.), 8., [lead])
    assert decision.state == 'FOLLOW_LEAD'
    assert max(abs(y) for _, y in decision.path) < 1e-6
    ahead = 40. - 2.4 - 13.845
    assert math.isclose(decision.speed_cap_mps, stopping_speed(ahead - 6., 3., .4) + 2.)


def test_far_static_block_still_limits_speed_to_a_stoppable_value():
    avoid = planner(plan_distance_m=20.)
    block = RouteObstacle(60., 0., 1.2)
    decision = avoid.update((0., 0.), 16., [block])
    gap = 60. - 1.2 - 3.845 - 5.
    assert math.isclose(decision.speed_cap_mps, stopping_speed(gap, 3., .4))


def test_oncoming_car_on_its_own_lane_does_not_stop_ego():
    avoid = planner()
    oncoming = RouteObstacle(40., 3.5, 1., vx=-8., dynamic=True, half_length=2.4)
    decision = avoid.update((10., 0.), 8., [oncoming])
    assert decision.state == 'FOLLOW'
    assert decision.speed_cap_mps == math.inf


def test_oncoming_car_appearing_while_pulling_out_returns_ego_to_its_lane():
    avoid = planner()
    block = RouteObstacle(50., 0., 1.2)
    assert avoid.update((10., 0.), 5.5, [block]).state == 'PASSING'
    oncoming = RouteObstacle(80., 3.5, 1., vx=-8., dynamic=True, half_length=2.4)
    decision = avoid.update((14., .5), 5.5, [block, oncoming])
    assert decision.state == 'WAIT_PASS'
    assert max(abs(y) for _, y in decision.path) < 1e-6


def test_solid_line_beside_the_block_means_waiting_behind_it():
    avoid = planner(lane_changes=[])
    block = RouteObstacle(50., 0., 1.2)
    decision = avoid.update((10., 0.), 5.5, [block])
    assert decision.state == 'WAIT_SOLID_LINE'
    assert max(abs(y) for _, y in decision.path) < 1e-6
    gap = 50. - 1.2 - 13.845 - 5.
    assert math.isclose(decision.speed_cap_mps, stopping_speed(gap, 3., .4))


def test_dashed_line_ending_before_the_return_also_means_waiting():
    avoid = planner(lane_changes=[dict(start_s=0., end_s=55., side='left', offset_m=3.5)])
    decision = avoid.update((10., 0.), 5.5, [RouteObstacle(50., 0., 1.2)])
    assert decision.state == 'WAIT_SOLID_LINE'


def test_only_the_dashed_right_lane_is_used_for_the_pass():
    avoid = planner(lane_changes=[dict(start_s=0., end_s=400., side='right', offset_m=-3.7)])
    decisions = drive(avoid, [RouteObstacle(60., 0., 1.2)], 0., 120.)
    assert decisions[-1][0] >= 119.
    assert min(y for _, y, _ in decisions) < -3.5
    assert max(y for _, y, _ in decisions) < 1e-6


def test_occupied_right_lane_falls_back_to_the_clear_left_lane():
    avoid = planner(lane_changes=[dict(start_s=0., end_s=400., side='right', offset_m=-3.7),
                                  dict(start_s=0., end_s=400., side='left', offset_m=3.5)])
    parked_right = RouteObstacle(58., -3.7, 1.)
    decision = avoid.update((10., 0.), 5.5, [RouteObstacle(50., 0., 1.2), parked_right])
    assert decision.state == 'PASSING'
    assert max(y for _, y in decision.path) > 3.


def test_dashed_intervals_that_touch_cover_one_pass():
    avoid = planner(lane_changes=[dict(start_s=0., end_s=45., side='left', offset_m=3.5),
                                  dict(start_s=45.5, end_s=400., side='left', offset_m=3.6)])
    assert avoid.pass_offsets(10., 90.) == [3.5]


def closed_loop_jaywalk(speed_kmh, trigger_ahead_m, detect_delay_s=.8, decel=3., dt=.05):
    """Ego on the route at speed; a walker 4.5 m right starts crossing when Ego's
    rear axle is trigger_ahead_m from its line. Returns the smallest bumper gap
    while the walker is inside Ego's body band: negative is a hit, None means
    Ego had fully passed or the walker had crossed before they could meet."""
    avoid = planner(lane_changes=[])
    ped_x, ped_y, ped_v = 120., -4.5, 7. / 3.6
    x, v, t, t_start = 0., speed_kmh / 3.6, 0., None
    closest = None
    while t < 20.:
        if t_start is None and ped_x - x <= trigger_ahead_m:
            t_start = t
        walking = t_start is not None and ped_y < 6.
        if walking:
            ped_y = -4.5 + ped_v * (t - t_start)
        seen_moving = walking and t - t_start >= detect_delay_s
        ped = RouteObstacle(ped_x, ped_y, .3, vy=ped_v if seen_moving else 0.,
                            dynamic=seen_moving)
        cap = avoid.update((x, 0.), v, [ped]).speed_cap_mps
        target = min(speed_kmh / 3.6, cap)
        v = max(target, v - decel * dt) if target < v else target
        x += v * dt
        t += dt
        if abs(ped_y) < .946 + .3 and ped_x + .3 >= x - .79:
            gap = max(-1., ped_x - .3 - (x + 3.845))  # -1: inside the body length
            closest = gap if closest is None else min(closest, gap)
    return closest


def test_the_crash_jaywalker_is_stopped_for_at_crossing_speed():
    # Same walker as the 2026-10-13 C016 crash (starts 26 m ahead), but Ego
    # holds the 30 km/h intersection crossing speed.
    gap = closed_loop_jaywalk(30., 26.)
    assert gap is not None and gap > 2.


def test_the_crash_jaywalker_is_not_avoidable_at_full_speed():
    # The crash itself: about 48 km/h with the walker starting 26 m ahead.
    gap = closed_loop_jaywalk(48., 26.)
    assert gap is not None and gap < 0.


def crosswalk_run(walker_y0, walker_vy, until_s=25., crosswalk_x=58.):
    """Ego from rest 58 m before a crosswalk (stop line + 3 m) at up to 30 km/h.
    Returns (closest bumper gap to the crosswalk while stopped, time moving again)."""
    avoid = RouteAvoidance(ROUTE, RouteAvoidanceConfig(), LEFT_LANE,
                           crosswalks=[(crosswalk_x - 2., crosswalk_x + 3.)])
    x, v, t, dt = 0., 0., 0., .05
    stopped_gap, resumed = None, None
    while t < until_s:
        y = walker_y0 + walker_vy * t
        moving = walker_vy != 0. and t >= .8
        decision = avoid.update((x, 0.), v, [RouteObstacle(
            crosswalk_x, y, .3, vy=walker_vy if moving else 0., dynamic=moving)])
        target = min(30. / 3.6, decision.speed_cap_mps)
        v = min(target, v + 2. * dt) if target >= v else max(target, v - 3. * dt)
        x += v * dt
        t += dt
        if v < .1 and t > 1. and stopped_gap is None:
            stopped_gap = crosswalk_x - 2. - (x + 3.845)
        if stopped_gap is not None and resumed is None and v > 1.:
            resumed = t
    return stopped_gap, resumed, x


def test_walker_crossing_global_route_makes_ego_stop():
    # The pedestrian approaches and crosses the global route.
    gap, resumed, _ = crosswalk_run(-6., .5)
    assert gap is not None and 0. < gap < 4.
    assert resumed is not None


def test_person_standing_off_the_road_or_walking_away_does_not_stop_ego():
    gap, _, x = crosswalk_run(7., 0.)
    assert gap is None and x > 100.
    gap, _, x = crosswalk_run(6., 4. / 3.6)
    assert gap is None and x > 100.


def test_stationary_small_fixture_in_other_lane_does_not_block_crosswalk():
    avoid = RouteAvoidance(ROUTE, RouteAvoidanceConfig(), LEFT_LANE,
                           crosswalks=[(56., 61.)])
    fixture = RouteObstacle(58., -3.4, .3)
    decision = avoid.update((44., 0.), 0., [fixture])
    assert decision.state != 'CROSSWALK_STOP'
    walker = RouteObstacle(58., -3.4, .3, vy=1., dynamic=True)
    decision = avoid.update((44., 0.), 0., [walker])
    assert decision.state != 'CROSSWALK_STOP'
    walker_in_route = RouteObstacle(58., 0., .3, vy=1., dynamic=True)
    decision = avoid.update((44., 0.), 0., [walker_in_route])
    assert decision.state == 'CROSSWALK_STOP'


def test_traffic_along_the_lane_over_the_crosswalk_is_trailed_not_stopped_for():
    avoid = RouteAvoidance(ROUTE, RouteAvoidanceConfig(), LEFT_LANE, crosswalks=[(56., 61.)])
    lead = RouteObstacle(58., 0., 1., vx=5., dynamic=True, half_length=2.4)
    decision = avoid.update((30., 0.), 8., [lead])
    assert decision.state == 'FOLLOW_LEAD'
    oncoming = RouteObstacle(58., 3.5, 1., vx=-8., dynamic=True, half_length=2.4)
    scooter = RouteObstacle(58., 0., .4, vx=4., dynamic=True, half_length=1.)
    assert avoid.update((30., 0.), 8., [oncoming]).state == 'FOLLOW'
    assert avoid.update((30., 0.), 8., [scooter]).state == 'FOLLOW_LEAD'
    # Someone crossing on the same crosswalk still stops Ego.
    walker = RouteObstacle(58., 1., .3, vy=1.1, dynamic=True)
    assert avoid.update((30., 0.), 8., [lead, walker]).state == 'CROSSWALK_STOP'


def test_closed_lap_path_continues_across_the_start_line():
    # Square lap of 400 m whose last point returns to the start.
    side = np.arange(0., 100., 1.)
    lap = np.vstack([np.column_stack((side, np.zeros_like(side))),
                     np.column_stack((np.full_like(side, 100.), side)),
                     np.column_stack((100. - side, np.full_like(side, 100.))),
                     np.column_stack((np.zeros_like(side), 100. - side)),
                     [[0., 0.]]])
    planner = RouteAvoidance(lap, RouteAvoidanceConfig())
    assert planner.loop_length == 400.
    before = planner.update((0., 3.), 10., [])  # 3 m before the start line
    assert before.path and before.path[-1][0] > 40.  # path runs on past the start
    after = planner.update((2., 0.), 10., [])  # crossed the start line
    assert planner.last_s < 5.
    assert after.path and after.path[-1][0] > 40.
