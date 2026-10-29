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


def test_r1_only_zone_disables_offset_path_but_keeps_obstacle_braking():
    p = planner()
    p.r1_only_ranges = [(0., 100.)]
    d = p.update((0., 0.), 5., [RouteObstacle(25., 0., 1.)])
    assert p.active is None
    assert all(abs(y) < 1e-6 for x,y in d.path)
    assert math.isfinite(d.speed_cap_mps)
    p.reset()
    p.update((101., 0.), 5., [RouteObstacle(140., 0., 1.)])
    assert p.active is not None


def test_r1_only_zone_keeps_dynamic_crossing_stop():
    p = planner()
    p.r1_only_ranges = [(0., 100.)]
    d = p.update((0., 0.), 5., [RouteObstacle(5., 0., .4, vy=-2., dynamic=True)])
    assert d.speed_cap_mps == 0.
    assert p.active is None


def test_s5_vehicle_initial_static_label_and_motion_dropout_do_not_start_pass():
    avoid = planner()
    avoid.vehicle_follow_ranges = [(0., 100.)]
    for i in range(12):
        dynamic = 2 <= i <= 6
        car = RouteObstacle(40.+i*.3, 0., 1., vx=3. if dynamic else 0.,
                            dynamic=dynamic, track_id=7, half_length=2.)
        decision = avoid.update((0., 0.), 5., [car], stamp=i*.1)
        assert avoid.active is None
        assert all(abs(y)<1e-6 for x,y in decision.path)


def test_s5_confirmed_static_vehicle_remains_avoidable():
    avoid = planner()
    avoid.vehicle_follow_ranges = [(0., 100.)]
    for i in range(9):
        avoid.update((0.,0.), 5., [RouteObstacle(40.,0.,1.,track_id=7,half_length=2.)],stamp=i*.1)
    assert avoid.active is not None


def test_s5_small_crossers_keep_identical_braking():
    for size in (.25, .5):
        a,b = planner(),planner()
        b.vehicle_follow_ranges = [(0.,100.)]
        for i in range(8):
            person=RouteObstacle(12.,2.-i*.3,size,vy=-2.,dynamic=True,track_id=7,half_length=size)
            before=a.update((0.,0.),5.,[person],stamp=i*.1)
            after=b.update((0.,0.),5.,[person],stamp=i*.1)
            assert before.state==after.state
            assert before.speed_cap_mps==after.speed_cap_mps
            assert before.path==after.path


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


def test_static_wait_does_not_create_a_separate_traffic_constraint():
    decision = planner(lane_changes=[]).update(
        (10., 0.), 5.5, [RouteObstacle(16., 0., 1.2)])
    assert decision.state == 'WAIT_SOLID_LINE'
    assert decision.speed_cap_mps == 0.
    assert math.isinf(decision.traffic_speed_cap_mps)


def test_static_zero_cap_cannot_hide_an_occupied_crosswalk():
    avoid = RouteAvoidance(ROUTE, RouteAvoidanceConfig(), [], crosswalks=[(15., 20.)])
    decision = avoid.update((10., 0.), 5.5,
        [RouteObstacle(16., 0., 1.2), RouteObstacle(17., .5, .3)])
    assert decision.state == 'WAIT_SOLID_LINE'
    assert decision.speed_cap_mps == 0.
    assert decision.traffic_speed_cap_mps == 0.


def test_static_zero_cap_cannot_hide_a_predicted_crossing():
    decision = planner(lane_changes=[]).update((10., 0.), 5.5,
        [RouteObstacle(16., 0., 1.2), RouteObstacle(17., 3., .3, vy=-2., dynamic=True)])
    assert decision.state == 'WAIT_SOLID_LINE'
    assert decision.speed_cap_mps == 0.
    assert decision.traffic_speed_cap_mps == 0.


def test_static_zero_cap_keeps_the_independent_moving_traffic_limit():
    decision = planner(lane_changes=[]).update((10., 0.), 5.5,
        [RouteObstacle(16., 0., 1.2),
         RouteObstacle(22., 0., 1., vx=2., dynamic=True, half_length=2.)])
    assert decision.state == 'WAIT_SOLID_LINE'
    assert decision.speed_cap_mps == 0.
    expected = stopping_speed(22. - 2. - 13.845 - 6., 3., .4) + 2.
    assert math.isclose(decision.traffic_speed_cap_mps, expected)


def test_solid_line_static_wait_keeps_a_red_stop_line_constraint():
    decision = planner(lane_changes=[]).update(
        (10., 0.), 5.5, [RouteObstacle(16., 0., 1.2)],
        stop_line_distance_m=10., stop_line_permitted=False)
    assert decision.state == 'WAIT_SOLID_LINE'
    assert decision.speed_cap_mps == 0.
    assert decision.traffic_speed_cap_mps == 0.


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


def test_waypoint_only_zone_ignores_a_block_inside_it_but_not_outside():
    # Tollgate (Map Tuner points 3496-3572): booths there are cleared by the route.
    for block_x, expected in ((60., 'FOLLOW'), (120., 'PASSING')):
        avoid = planner()
        avoid.ignored_s_ranges = [(50., 80.)]
        decision = avoid.update((30., 0.), 5.5, [RouteObstacle(block_x, 0., 1.2)])
        if expected == 'FOLLOW':
            assert decision.state == 'FOLLOW' and math.isinf(decision.speed_cap_mps)
        else:
            assert decision.state != 'FOLLOW'


# Crossing-pedestrian hold: only inside the configured zone (s 100..200 here).
def hold_planner(zone=((100., 200.),)):
    avoid = RouteAvoidance(ROUTE, RouteAvoidanceConfig(), [])
    avoid.set_pedestrian_hold_ranges(zone)
    return avoid


def person(x, y, vy=0., moving=True, track_id=220):
    return RouteObstacle(x, y, .35, vx=0., vy=vy, dynamic=moving,
                         track_id=track_id, half_length=.3)


def approach(avoid, ego_x, speed, x, y_far, y_near, vy, track_id=220):
    """Three updates of a person stepping toward the route (y_far -> y_near)."""
    for y in (y_far + (y_far - y_near), y_far):
        avoid.update((ego_x, 0.), speed, [person(x, y, vy=vy, track_id=track_id)])
    return avoid.update((ego_x, 0.), speed, [person(x, y_near, vy=vy, track_id=track_id)])


def test_velocity_dropout_beside_the_band_keeps_the_pedestrian_stop():
    # Logged S7 case: walking in from the right, then one update with no
    # velocity just outside the band. It must not read as a clear road.
    for zone, held in ((((100., 200.),), True), ((), False)):
        avoid = hold_planner(zone)
        first = approach(avoid, 128., 6.7, 140., -3.1, -2.9, 1.2)
        assert first.state == 'YIELD'
        dropout = avoid.update((128.4, 0.), 3.6, [person(140., -2.24, moving=False)])
        if held:
            assert dropout.state == 'YIELD'
            assert dropout.speed_cap_mps <= first.speed_cap_mps
            assert dropout.traffic_speed_cap_mps == dropout.speed_cap_mps
        else:
            assert dropout.state == 'FOLLOW'      # unchanged outside the zone


def test_pedestrian_stop_is_a_full_stop_when_planned_braking_is_short():
    avoid = hold_planner()
    decision = approach(avoid, 128., 7.9, 139., -3.0, -2.8, 1.9)
    assert decision.state == 'YIELD'
    assert decision.speed_cap_mps == 0.
    # Full braking holds until Ego has stopped, even as the stop gets easier.
    slower = avoid.update((131., 0.), 2.5, [person(139., -1.8, vy=1.9)])
    assert slower.speed_cap_mps == 0.


def test_crossing_that_needs_no_braking_starts_no_hold():
    avoid = hold_planner()
    decision = approach(avoid, 100., 3., 118., -3.5, -3.3, 1.)
    assert decision.state == 'YIELD' and decision.speed_cap_mps > 3.
    assert avoid.pedestrian_hold is None


def test_roadside_clutter_with_a_stale_velocity_starts_no_hold():
    # Logged S5 clutter: velocity toward the route, position moving away.
    avoid = hold_planner()
    for y in (-1.83, -1.7, -1.5, -1.84, -2.08):    # tiny, approaching once
        decision = avoid.update((128., 0.), 5.5, [RouteObstacle(
            140., y, .17, vx=0., vy=2.0, dynamic=True, track_id=128, half_length=.12)])
    assert avoid.pedestrian_hold is None
    assert decision.speed_cap_mps > 0.


def test_pedestrian_hold_releases_after_the_person_is_seen_clear():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1.0, 1.5)
    assert avoid.pedestrian_hold is not None
    # Advance the observation clock for the crossing, then require three
    # measured clear observations beyond the vehicle envelope.
    avoid.observation_time += 2.
    states = [avoid.update((128., 0.), 0., [person(140., 2.5, vy=1.5)]).state
              for _ in range(3)]
    assert states == ['YIELD', 'YIELD', 'FOLLOW']
    assert avoid.pedestrian_hold is None


def test_brief_track_loss_does_not_release_the_pedestrian_stop():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1.0, 1.5)
    for _ in range(4):
        assert avoid.update((128., 0.), 0., []).state == 'YIELD'
    # Re-acquired under a new id next to the last position: same person.
    assert avoid.update((128., 0.), 0., [person(140.5, -.3, vy=1.5, track_id=301)]).state == 'YIELD'
    assert 301 in avoid.pedestrian_hold['ids']


def test_person_waiting_at_the_kerb_eventually_releases_the_hold():
    avoid = hold_planner()
    assert approach(avoid, 128., 6., 140., -2.8, -2.6, 1.5).state == 'YIELD'
    assert avoid.pedestrian_hold is not None
    states = [avoid.update((128., 0.), 0., [person(140., -2.6, moving=False)]).state
              for _ in range(30)]
    assert states[-2] == 'YIELD' and states[-1] == 'FOLLOW'


def test_vehicles_and_lane_traffic_are_not_held_as_pedestrians():
    avoid = hold_planner()
    for y in (-2.9, -2.6):
        avoid.update((128., 0.), 4., [RouteObstacle(140., y, 1., vx=0., vy=1.5, dynamic=True,
                                                    track_id=5, half_length=2.3)])
    assert avoid.pedestrian_hold is None
    avoid = hold_planner()
    for x in (139., 140.):
        avoid.update((128., 0.), 4., [RouteObstacle(x, 0., .4, vx=3., vy=0., dynamic=True,
                                                    track_id=6, half_length=.6)])
    assert avoid.pedestrian_hold is None


def test_pedestrian_hold_ends_once_ego_is_past_the_person():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1.0, 1.5)
    assert avoid.update((141., 0.), 3., []).state == 'FOLLOW'
    assert avoid.pedestrian_hold is None


def test_person_unseen_at_the_bumper_keeps_the_stop_until_seen_clear():
    # Logged S5 case: right in front of the bumper the person drops out of
    # the LiDAR tracks. That is not a clear road.
    avoid = hold_planner()
    approach(avoid, 128., 0., 136., -1.3, -1.0, 1.5)
    states = [avoid.update((128., 0.), 0., []).state for _ in range(29)]
    assert set(states) == {'YIELD'}
    # Seen clear on the far side: released after the usual confirmation.
    states = [avoid.update((128., 0.), 0., [person(136., 3.5, vy=1.5)]).state
              for _ in range(5)]
    assert states[-1] == 'FOLLOW'


def test_person_last_seen_clear_then_lost_requires_observed_clearance():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1.0, 1.5)
    avoid.observation_time += 2.
    avoid.update((128., 0.), 0., [person(140., 3.5, vy=1.5)])
    states = [avoid.update((128., 0.), 0., []).state for _ in range(4)]
    assert states[-1] == 'YIELD'
    for _ in range(5):
        decision = avoid.update((128., 0.), 0., [person(140., 3.5, vy=1.5)])
    assert decision.state == 'FOLLOW'


def test_lost_pedestrian_requires_measured_clear_space_before_release():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1., 1.5)
    for _ in range(40):
        decision = avoid.update((128., 0.), 0., [], clearance_observer=lambda h, o: False)
    assert decision.state == 'YIELD'
    # Fresh positive observations, not elapsed time alone, release the cap.
    for _ in range(8):
        decision = avoid.update((128., 0.), 0., [], clearance_observer=lambda h, o: True)
    assert avoid.pedestrian_hold is None
    assert decision.state == 'FOLLOW'
    assert decision.speed_cap_mps > 0.


def test_visibility_release_restarts_after_occlusion_and_duplicate_stamp():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1., 1.5)
    start = avoid.observation_time + 1.
    for _ in range(10):
        avoid.update((128., 0.), 0., [], stamp=start, clearance_observer=lambda h, o: True)
    assert avoid.pedestrian_hold is not None
    for i in range(4):
        avoid.update((128., 0.), 0., [], stamp=start+.1*(i+1), clearance_observer=lambda h, o: True)
    avoid.update((128., 0.), 0., [], stamp=start+.5, clearance_observer=lambda h, o: False)
    assert avoid.pedestrian_hold['visible_clear_scans'] == 0
    assert avoid.pedestrian_hold is not None


def test_visible_space_does_not_override_a_matching_person_or_moving_ego():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1., 1.5)
    for _ in range(40):
        avoid.update((128., 1.), 0., [person(140., -.5, vy=0.)],
                     clearance_observer=lambda h, o: True)
    assert avoid.pedestrian_hold is not None
    for _ in range(20):
        avoid.update((128., 0.), 2., [], clearance_observer=lambda h, o: True)
    assert avoid.pedestrian_hold is not None


def test_old_hold_can_reconnect_to_person_near_last_footprint():
    avoid = hold_planner()
    approach(avoid, 128., 0., 140., -1.3, -1., 1.5)
    for _ in range(40):
        avoid.update((128., 0.), 0., [])
    avoid.update((128., 0.), 0., [person(140., -.5, vy=0., track_id=999)])
    assert 999 in avoid.pedestrian_hold['ids']


def test_observed_clearance_rejects_empty_sparse_and_occluded_scans():
    from leo_avoidance.observed_clearance import rectangle_observed_clear
    corners = [[5., -1.], [5., 1.], [7., 1.], [7., -1.]]
    a = np.deg2rad(np.arange(-20., 20., .1))
    clear = np.column_stack((15.*np.cos(a), 15.*np.sin(a), np.zeros(len(a))))
    assert rectangle_observed_clear(clear, corners)
    assert not rectangle_observed_clear([], corners)
    assert not rectangle_observed_clear(clear[::30], corners)
    assert not rectangle_observed_clear(np.vstack((clear, [[3., 0., 0.]])), corners)
    assert not rectangle_observed_clear(np.vstack((clear, [[6., 0., 0.]])), corners)


def candidate_boundary_hold():
    # 2026-10-29 15:44:08: candidate 53 associated with 33, just below .6 m.
    return dict(ids={33, 53}, s=446.949249, d=-3.250417, l=.598334,
                cap=0., full_brake=False, clear_scans=0, static_scans=0,
                lost_scans=0, near=False, confirmed_crossing=False,
                tentative=True, approach_scans=0, stationary_scans=3,
                initial_d=-2.473532, last_r=.568877, last_l=.598334,
                seen_at=10., vd=0.)


def boundary_observation(**changes):
    result = dict(id=33, s=446.95228, d=-3.2564, r=.575039, l=.601724,
                  moving=False, v_s=0., v_d=0.)
    result.update(changes)
    return result


def test_recorded_size_boundary_candidate_reconnects_and_releases():
    p = hold_planner()
    p.pedestrian_hold = candidate_boundary_hold()
    p.observation_time = 10.1
    assert p._single_pedestrian_hold_cap([boundary_observation()], 0., 444.5376,
                                       448.3826, lambda s: 0.) is None
    assert p.pedestrian_hold is None


def test_size_hysteresis_does_not_admit_new_large_objects_or_large_growth():
    p = hold_planner()
    hold = candidate_boundary_hold()
    assert not p._held_person_sized(hold, boundary_observation(id=999))
    assert not p._held_person_sized(hold, boundary_observation(l=1.2))
    assert not p._person_sized(boundary_observation())


def test_matching_candidate_inside_path_upgrades_instead_of_releasing():
    p = hold_planner()
    hold = candidate_boundary_hold()
    hold.update(d=-1.8, initial_d=-1.8, stationary_scans=3)
    p.pedestrian_hold = hold
    p.observation_time = 10.1
    cap = p._single_pedestrian_hold_cap([boundary_observation(d=-1.8)], 0., 444.5376,
                                       448.3826, lambda s: 0.)
    assert cap == 0.
    assert p.pedestrian_hold['confirmed_crossing']
    assert not p.pedestrian_hold['tentative']


def test_duplicate_scan_cannot_finish_candidate_stationary_release():
    p = hold_planner()
    p.pedestrian_hold = candidate_boundary_hold()
    for _ in range(10):
        p.observation_time = 10.
        p._single_pedestrian_hold_cap([boundary_observation()], 0., 444.5376,
                                     448.3826, lambda s: 0.)
    assert p.pedestrian_hold is not None


def test_unconfirmed_outward_candidate_releases_only_after_observed_departure():
    p = hold_planner()
    h = candidate_boundary_hold()
    h.update(stationary_scans=0, d=-3., initial_d=-3.)
    p.pedestrian_hold = h
    for i in range(1, 4):
        p.observation_time = 10.+i*.1
        cap = p._single_pedestrian_hold_cap(
            [boundary_observation(d=-3.-.1*i, moving=True, v_d=-1.)],
            0., 444.5376, 448.3826, lambda s: 0.)
        if i < 3:
            assert cap is not None
    assert p.pedestrian_hold is None


def test_missing_candidate_still_requires_evidence_not_timeout():
    p = hold_planner()
    p.pedestrian_hold = candidate_boundary_hold()
    for i in range(1, 100):
        p.observation_time = 10.+i*.1
        p._single_pedestrian_hold_cap([], 0., 444.5376, 448.3826, lambda s: 0.)
    assert p.pedestrian_hold is not None
