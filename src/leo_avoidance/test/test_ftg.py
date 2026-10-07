import math

import numpy as np

from leo_avoidance.ftg import Config, FollowTheGap
from leo_avoidance.ftg_visualization import (cluster_boxes,
                                             line_candidate_points,
                                             relative_lidar_xy)


def test_pose_diagnostic_uses_current_vehicle_heading():
    assert np.allclose(relative_lidar_xy((10., 12.), (10., 10., 0.)),
                       (-1.4, 2.))
    assert np.allclose(relative_lidar_xy((8., 10.), (10., 10., math.pi/2.)),
                       (-1.4, 2.))


def scan_points(obstacles, radius_m=12.0):
    """Two far side returns plus specified obstacle x,y points."""
    a = np.deg2rad([-55.0, 55.0])
    walls = np.column_stack((radius_m * np.cos(a), radius_m * np.sin(a),
                             np.zeros(len(a))))
    return np.vstack((walls, np.asarray(obstacles, dtype=float).reshape(-1, 3)))


def tunnel_walls(right_distance_m, left_distance_m=3.):
    x = np.linspace(1., 13., 121)
    return np.vstack((
        np.column_stack((x, np.full(len(x), -right_distance_m),
                         np.zeros(len(x)))),
        np.column_stack((x, np.full(len(x), left_distance_m),
                         np.zeros(len(x))))))


def test_tunnel_walls_are_track_boundaries_but_box_remains_obstacle():
    walls = tunnel_walls(3.846, 3.846)
    box = np.array([[7., y, 0.] for y in np.linspace(-.6, .6, 11)])
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=5.))
    assert len(planner.obstacle_candidates(walls)) == 0
    open_plan = planner.plan(walls, 1., wall_follow=True)
    assert open_plan.speed_mps > 0.
    assert open_plan.obstacle_points == 0
    assert open_plan.front_points == 0
    assert open_plan.right_wall_clearance_m is not None
    assert len(planner.obstacle_candidates(np.vstack((walls, box)))) == len(box)
    box_plan = planner.plan(np.vstack((walls, box)), 1., wall_follow=True)
    assert box_plan.front_points == len(box)
    assert box_plan.avoidance_phase == 'PASSING'
    assert box_plan.candidate_bearings_rad
    assert any(math.isclose(angle, box_plan.target_bearing_rad, abs_tol=1e-9)
               for angle in box_plan.candidate_bearings_rad)


def test_curved_tunnel_wall_is_boundary_but_inner_box_remains_obstacle():
    x = np.linspace(1., 19.8, 190)
    bend = .008 * x * x
    walls = np.vstack((
        np.column_stack((x, 5. + bend, np.zeros(len(x)))),
        np.column_stack((x, -5. + bend, np.zeros(len(x))))))
    box = np.array([[9., y, 0.] for y in np.linspace(-.5, .5, 9)])
    planner = FollowTheGap(Config(max_range_m=30.))
    assert len(planner.obstacle_candidates(walls)) == 0
    kept = planner.obstacle_candidates(np.vstack((walls, box)))
    assert len(kept) == len(box)


def test_short_side_object_is_not_classified_as_tunnel_wall():
    x = np.linspace(5., 8., 30)
    object_points = np.column_stack((x, np.full(len(x), -3.),
                                     np.zeros(len(x))))
    assert len(FollowTheGap().obstacle_candidates(object_points)) == len(x)


def test_panelled_wall_profile_survives_global_fit_failure():
    x = np.linspace(1., 24., 240)
    # Each panel is locally smooth, but the full side is not one line/curve.
    y = -4.2 + .12 * x + .55 * np.sin(x / 2.)
    wall = np.column_stack((x, y, np.zeros(len(x))))
    box = np.array([[10., side, 0.] for side in np.linspace(-.6, .6, 9)])
    planner = FollowTheGap(Config(max_range_m=30.))
    kept = planner.obstacle_candidates(np.vstack((wall, box)))
    assert len(kept) == len(box)


def test_thick_tunnel_panels_do_not_make_giant_obstacle_boxes():
    x = np.linspace(1., 24., 140)
    right = np.vstack([np.column_stack((x, np.full(len(x), -y),
                                        np.zeros(len(x))))
                       for y in (3.6, 4.0, 4.4)])
    left = np.vstack([np.column_stack((x, np.full(len(x), y),
                                       np.zeros(len(x))))
                      for y in (3.6, 4.0, 4.4)])
    box = np.array([[8., y, 0.] for y in np.linspace(-.6, .6, 11)])
    planner = FollowTheGap(Config(max_range_m=30.))
    kept = planner.obstacle_candidates(np.vstack((right, left, box)))
    assert len(kept) == len(box)
    plan = planner.plan(np.vstack((right, left, box)), 1., wall_follow=True)
    assert plan.right_wall_clearance_m is not None
    assert plan.front_points == len(box)


def test_connected_side_wall_survives_sparse_strip_sampling():
    x = np.ravel(np.arange(1., 26.)[:, None]
                 + np.array([.05, .3, .55, .8])[None, :])
    wall = np.column_stack((x, np.full(len(x), -3.8), np.zeros(len(x))))
    box = np.array([[8., y, 0.] for y in np.linspace(-.6, .6, 9)])
    planner = FollowTheGap(Config(max_range_m=30.))
    assert planner._wall_fit(wall, -1.) is None
    assert planner._wall_profile(wall, -1.) is None
    kept, fits = planner._track_obstacles(np.vstack((wall, box)))
    assert len(kept) == len(box)
    assert fits[0] is not None


def test_track_boundary_limits_route_bearing_without_wall_bubbles():
    walls = tunnel_walls(3.846, 3.846)
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=5.))
    plan = planner.plan(walls, 1., .5, wall_follow=True,
                        wall_follow_weight=0.)
    assert plan.speed_mps > 0.
    assert abs(plan.target_bearing_rad) < .3


def test_blackout_right_wall_spacing_steers_away_and_then_back():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=5.))
    initial = planner.plan(tunnel_walls(3.846, 3.846), 1., wall_follow=True)
    assert math.isclose(initial.right_wall_clearance_m, 2.9, abs_tol=.05)
    assert math.isclose(initial.right_wall_target_clearance_m, 2.9, abs_tol=.01)
    assert abs(initial.target_bearing_rad) < .02

    near = planner.plan(tunnel_walls(2.4, 3.6), 1., wall_follow=True)
    assert near.target_bearing_rad > .02  # left, away from right wall
    assert near.right_wall_clearance_m < initial.right_wall_clearance_m
    far = None
    for _ in range(8):
        far = planner.plan(tunnel_walls(4.1, 2.5), 1., wall_follow=True)
    assert far.target_bearing_rad < -.02  # right, toward saved spacing
    assert math.isclose(far.right_wall_target_clearance_m,
                        initial.right_wall_target_clearance_m, abs_tol=.01)


def test_blackout_wall_spacing_overrides_drifted_route_until_exit():
    walls = tunnel_walls(2.4, 3.6)
    planner = FollowTheGap(Config(max_range_m=30.))
    tunnel = planner.plan(walls, 1., -.4, wall_follow=True,
                          wall_follow_weight=1.)
    exit_plan = planner.plan(walls, 1., -.4, wall_follow=True,
                             wall_follow_weight=0.)
    assert tunnel.right_wall_clearance_m < 2.9
    assert tunnel.target_bearing_rad > 0.
    assert exit_plan.target_bearing_rad > 0.
    assert exit_plan.wall_follow_weight >= .9


def test_missing_exit_wall_limits_uncorroborated_route_turn():
    planner = FollowTheGap(Config(max_range_m=30.))
    side_returns = scan_points([], radius_m=12.)
    plan = planner.plan(side_returns, 1., -.6, wall_follow=True,
                        wall_follow_weight=0.)
    assert plan.speed_mps > 0.
    assert abs(plan.target_bearing_rad) <= math.radians(7.) + .01


def test_distant_tunnel_end_returns_allow_wall_guided_crawl_only_with_braking_room():
    walls = tunnel_walls(3.546, 3.546)
    far = np.array([[18., y, 0.] for y in np.linspace(-5., 5., 201)])
    cloud = np.vstack((walls, far))
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    plan = planner.plan(cloud, 6.7, wall_follow=True)
    assert plan.reason == 'distant returns; wall-guided crawl'
    assert 0. < plan.speed_mps <= 2.
    assert plan.front_points > 0 and plan.front_clearance_m > 15.
    assert plan.avoidance_phase == 'WALL_FOLLOW'
    # A real close blockage must not use the distant-return exception.
    close = far.copy()
    close[:, 0] = 8.
    blocked = planner.plan(np.vstack((walls, close)), 6.7,
                           wall_follow=True)
    assert blocked.speed_mps == 0.
    fast = FollowTheGap(Config(max_range_m=30., max_speed_mps=12.)).plan(
        cloud, 10., wall_follow=True)
    assert fast.speed_mps == 0.


def test_right_wall_target_resets_after_blackout():
    planner = FollowTheGap(Config(max_range_m=30.))
    planner.plan(tunnel_walls(3.), 0., wall_follow=True)
    gps_plan = planner.plan(tunnel_walls(2.4, 3.6), 0., wall_follow=False)
    assert gps_plan.right_wall_target_clearance_m is None
    assert abs(gps_plan.target_bearing_rad) < .02


def test_short_right_object_cannot_replace_long_wall_measurement():
    planner = FollowTheGap(Config(max_range_m=30.))
    obstacle = np.array([[x, -1.5, 0.] for x in np.linspace(4., 5., 30)])
    plan = planner.plan(np.vstack((tunnel_walls(3.), obstacle)), 0.,
                        wall_follow=True)
    assert math.isclose(plan.right_wall_clearance_m, 3. - .946, abs_tol=.1)
    assert len(planner.obstacle_candidates(
        np.vstack((tunnel_walls(3.), obstacle)))) >= len(obstacle)


def test_exit_fades_wall_spacing_and_limits_speed_for_route_alignment():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=30./3.6))
    walls = tunnel_walls(2.4, 3.6)
    tunnel = planner.plan(walls, 1., .08, wall_follow=True,
                          wall_follow_weight=1., route_rejoin_remaining_m=35.)
    exit_plan = planner.plan(walls, 1., .08, wall_follow=True,
                             wall_follow_weight=0., route_rejoin_remaining_m=12.)
    assert tunnel.target_bearing_rad > exit_plan.target_bearing_rad
    assert abs(exit_plan.target_bearing_rad - .08) < .02
    assert exit_plan.speed_mps <= 5.
    assert exit_plan.right_wall_target_clearance_m == 2.9
    assert exit_plan.route_rejoin_remaining_m == 12.


def test_right_wall_spacing_band_has_no_offset_correction():
    for clearance in (2.82, 2.9, 2.98):
        planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=5.))
        plan = planner.plan(tunnel_walls(clearance + .946, 3.846), 1.,
                            wall_follow=True)
        assert 2.8 <= plan.right_wall_clearance_m <= 3.0
        assert abs(plan.target_bearing_rad) < .02


def test_post_pass_search_ignores_distant_wall_but_keeps_near_blocker():
    walls = tunnel_walls(3.846, 3.846)
    distant = np.array([[28., y, 0.] for y in np.linspace(-10., 10., 101)])
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    planner.avoidance_active = True
    planner.passing_side = 1
    plan = planner.plan(np.vstack((walls, distant)), 2., wall_follow=True)
    assert plan.reason == 'gap fits'
    assert plan.front_points > 0
    assert plan.speed_mps <= planner.config.no_return_speed_mps
    assert plan.target_bearing_rad >= 0.
    near = distant.copy()
    near[:, 0] = 3.
    blocked = planner.plan(np.vstack((walls, near)), 0., wall_follow=True)
    assert blocked.speed_mps == 0.


def test_post_pass_twenty_metre_search_sees_distant_box():
    walls = tunnel_walls(3.846, 3.846)
    box = np.array([[18., y, 0.] for y in np.linspace(-1., 1., 41)])
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    planner.avoidance_active = True
    planner.passing_side = 1
    plan = planner.plan(np.vstack((walls, box)), 2., wall_follow=True)
    assert plan.avoidance_phase == 'PASSING'
    assert plan.target_bearing_rad > .1
    assert plan.speed_mps <= planner.config.no_return_speed_mps


def test_wall_spacing_waits_for_rear_lidar_detection_after_pass():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=5.))
    walls = tunnel_walls(2.4, 3.6)
    box = np.array([[7., y, 0.] for y in np.linspace(-.6, .6, 11)])
    rear = np.array([[-8., y, 0.] for y in (1.8, 1.9, 2., 2.1)])
    passing = planner.plan(np.vstack((walls, box)), 1., -.3,
                           wall_follow=True)
    assert passing.avoidance_phase == 'PASSING'
    assert passing.wall_follow_weight == 0.
    assert passing.rear_obstacle_points == 0
    opposite = -.3 if passing.target_bearing_rad > 0. else .3
    still_passing = planner.plan(walls, 1., opposite, wall_follow=True)
    assert still_passing.avoidance_phase == 'PASSING'
    assert still_passing.wall_follow_weight == 0.
    assert still_passing.target_bearing_rad * passing.target_bearing_rad >= 0.
    first_rear = planner.plan(np.vstack((walls, rear)), 1., -.3,
                              wall_follow=True)
    assert first_rear.avoidance_phase == 'PASSING'
    assert first_rear.rear_obstacle_points == 4
    assert first_rear.rear_confirm_scans == 1
    resumed = planner.plan(np.vstack((walls, rear)), 1., -.3,
                           wall_follow=True)
    assert resumed.avoidance_phase == 'WALL_FOLLOW'
    assert resumed.wall_follow_weight == 1.
    assert resumed.target_bearing_rad > 0.


def test_committed_pass_does_not_reverse_with_obstacle_still_ahead():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=5.))
    walls = tunnel_walls(3.546, 3.546)
    box = np.array([[7., y, 0.] for y in np.linspace(-.6, .6, 11)])
    planner.avoidance_active = True
    planner.passing_side = 1
    left_pass = planner.plan(np.vstack((walls, box)), 1., -.35,
                             wall_follow=True)
    assert left_pass.avoidance_phase == 'PASSING'
    assert left_pass.speed_mps == 0. or left_pass.target_bearing_rad >= 0.
    planner.passing_side = -1
    right_pass = planner.plan(np.vstack((walls, box)), 1., .35,
                              wall_follow=True)
    assert right_pass.avoidance_phase == 'PASSING'
    assert right_pass.speed_mps == 0. or right_pass.target_bearing_rad <= 0.


def test_tunnel_walls_and_rear_side_returns_do_not_end_pass():
    planner = FollowTheGap(Config(max_range_m=30.))
    walls = tunnel_walls(3.546, 3.546)
    first = np.array([[7., y, 0.] for y in np.linspace(-.6, .6, 11)])
    planner.plan(np.vstack((walls, first)), 1., wall_follow=True)
    # A wall can lie behind the sensor, but is outside the close rear cone.
    rear_walls = np.array([[-x, side * 3.546, 0.]
                           for x in np.linspace(8., 16., 30)
                           for side in (-1., 1.)])
    # A nearer flank return is outside the +/-15 degree rear cone.
    flank = np.array([[-3., y, 0.] for y in (1.8, 1.9, 2., 2.1)])
    for _ in range(3):
        plan = planner.plan(np.vstack((walls, rear_walls, flank)), 1.,
                            wall_follow=True)
        assert plan.avoidance_phase == 'PASSING'
        assert plan.rear_scan_points > 0
        assert plan.rear_obstacle_points == 0


def test_rear_points_before_a_front_pass_do_not_change_wall_following():
    planner = FollowTheGap(Config(max_range_m=30.))
    walls = tunnel_walls(3.546, 3.546)
    rear = np.array([[-8., y, 0.] for y in (1.8, 1.9, 2., 2.1)])
    plan = planner.plan(np.vstack((walls, rear)), 1., wall_follow=True)
    assert plan.rear_obstacle_points == 4
    assert plan.avoidance_phase == 'WALL_FOLLOW'
    assert len(planner.rear_obstacle_candidates(np.vstack((walls, rear)))) == 4
    assert len(planner.obstacle_candidates(rear)) == 0


def test_two_rear_points_or_overhead_returns_cannot_end_pass():
    planner = FollowTheGap(Config(max_range_m=30.))
    walls = tunnel_walls(3.546, 3.546)
    front = np.array([[7., y, 0.] for y in np.linspace(-.6, .6, 11)])
    planner.plan(np.vstack((walls, front)), 1., wall_follow=True)
    two_rear = np.array([[-8., y, 0.] for y in (1.9, 2.)])
    roof = np.array([[-8., y, 2.] for y in (1.8, 1.9, 2., 2.1)])
    for _ in range(3):
        plan = planner.plan(np.vstack((walls, two_rear, roof)), 1.,
                            wall_follow=True)
        assert plan.avoidance_phase == 'PASSING'
        assert plan.rear_obstacle_points == 2
        assert plan.rear_confirm_scans == 0


def test_empty_cloud_stops():
    plan = FollowTheGap().plan(np.empty((0, 3)), 0.0)
    assert plan.speed_mps == 0 and 'empty' in plan.reason


def test_open_corridor_keeps_forward_heading():
    plan = FollowTheGap().plan(scan_points([]), 0.0)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad) < 1e-9


def test_clear_long_range_scan_allows_thirty_kmh():
    points = scan_points([], radius_m=29.)
    plan = FollowTheGap(Config(max_range_m=30., max_speed_mps=30./3.6)).plan(
        points, 3.)
    assert math.isclose(plan.speed_mps, 30./3.6, abs_tol=.01)


def test_short_observed_range_does_not_allow_thirty_kmh():
    points = scan_points([], radius_m=15.)
    plan = FollowTheGap(Config(max_range_m=30., max_speed_mps=30./3.6)).plan(
        points, 3.)
    assert 0. < plan.speed_mps < 30./3.6


def test_right_wall_nearby_slows_and_steers_left():
    cfg = Config(max_range_m=30., max_speed_mps=30./3.6)
    far = scan_points([], radius_m=29.)
    right_wall = np.array([[x, -1.5, 0.] for x in np.linspace(1., 8., 30)])
    plan = FollowTheGap(cfg).plan(np.vstack((far, right_wall)), 1.)
    assert 0. < plan.speed_mps < 5.
    assert plan.target_bearing_rad > 0.


def test_obstacle_clearance_increase_expands_the_blocked_angle():
    points = scan_points([(7., 0., 0.)])
    old = FollowTheGap(Config(side_margin_m=.25)).plan(points, 0.)
    widened = FollowTheGap(Config(side_margin_m=.4)).plan(points, 0.)
    assert abs(widened.target_bearing_rad) > abs(old.target_bearing_rad)
    assert math.isclose(Config().max_target_bearing_rad,
                        math.radians(35.), abs_tol=1e-12)


def test_open_corridor_follows_route_bearing():
    plan = FollowTheGap().plan(scan_points([]), 0.0, preferred_bearing_rad=.2)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad - .2) < .02


def test_center_obstacle_selects_side_gap():
    plan = FollowTheGap().plan(scan_points([(7.0, 0.0, 0.0)]), 0.0)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad) > 0.1
    assert plan.gap_width_m > 0


def test_low_box_is_kept_while_road_return_is_excluded():
    walls = scan_points([])
    low_box = np.array([[7., y, -1.05] for y in np.linspace(-.7, .7, 9)])
    ground = np.array([[5., 0., -1.23]])
    plan = FollowTheGap().plan(np.vstack((walls, low_box, ground)), 0.)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad) > .1
    assert plan.front_points == len(low_box)
    assert plan.nearest_front_x_m == 7.


def test_dense_tunnel_road_is_not_a_front_obstacle():
    xs, ys = np.meshgrid(np.linspace(1., 12., 36), np.linspace(-3., 3., 17))
    road = np.column_stack((xs.ravel(), ys.ravel(),
                            -1.23 + .01 * xs.ravel()))
    walls = scan_points([])
    planner = FollowTheGap(Config(max_speed_mps=5.))
    plan = planner.plan(np.vstack((road, walls)), 2.)
    assert plan.speed_mps > 0
    assert plan.front_points == 0
    assert abs(plan.target_bearing_rad) < .02


def test_short_tunnel_road_patch_is_removed_without_hiding_low_box():
    xs, ys = np.meshgrid(np.linspace(9., 11.8, 18),
                         np.linspace(-3., 3., 13))
    road = np.column_stack((xs.ravel(), ys.ravel(),
                            -1.23 + .01 * xs.ravel()))
    box = np.array([[10.5, y, -.92] for y in np.linspace(-.7, .7, 9)])
    planner = FollowTheGap()
    kept = planner.obstacle_candidates(np.vstack((road, scan_points([]), box)))
    assert len(kept) == len(scan_points([])) + len(box)
    assert np.allclose(kept[-len(box):], box)


def test_wide_flat_low_obstacle_cannot_define_its_own_ground_plane():
    xs, ys = np.meshgrid(np.linspace(7., 10., 15),
                         np.linspace(-1.5, 1.5, 11))
    pallet = np.column_stack((xs.ravel(), ys.ravel(),
                              np.full(xs.size, -.85)))
    planner = FollowTheGap()
    assert len(planner.obstacle_candidates(pallet)) == len(pallet)


def test_low_box_survives_road_plane_rejection():
    xs, ys = np.meshgrid(np.linspace(1., 12., 36), np.linspace(-3., 3., 17))
    road = np.column_stack((xs.ravel(), ys.ravel(),
                            -1.23 + .01 * xs.ravel()))
    box = np.array([[7., y, -1.0] for y in np.linspace(-.7, .7, 9)])
    planner = FollowTheGap(Config(max_speed_mps=5.))
    plan = planner.plan(np.vstack((road, scan_points([]), box)), 0.)
    assert plan.front_points == len(box)
    assert plan.nearest_front_x_m == 7.
    assert abs(plan.target_bearing_rad) > .1


def test_tunnel_wall_and_vehicle_selects_inflated_side_corridor():
    xs = np.linspace(1., 15., 100)
    walls = np.vstack((
        np.column_stack((xs, np.full(len(xs), 4.), np.zeros(len(xs)))),
        np.column_stack((xs, np.full(len(xs), -4.), np.zeros(len(xs))))))
    vehicle = np.array([[14., y, 0.] for y in np.linspace(-.9, .9, 15)])
    plan = FollowTheGap(Config(max_range_m=15., max_speed_mps=5.)).plan(
        np.vstack((walls, vehicle)), 5., preferred_bearing_rad=.1)
    assert plan.speed_mps > 0
    assert plan.target_bearing_rad > 0.1
    assert plan.gap_width_m < 2 * (0.946 + .25)


def test_passable_tunnel_box_uses_local_gap_instead_of_far_wall_ray():
    xs = np.linspace(1., 30., 300)
    walls = np.vstack((
        np.column_stack((xs, np.full(len(xs), 3.5), np.zeros(len(xs)))),
        np.column_stack((xs, np.full(len(xs), -3.5), np.zeros(len(xs))))))
    box = np.array([[7., y, 0.] for y in np.linspace(-.7, .7, 15)])
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=30./3.6))
    plan = planner.plan(np.vstack((walls, box)), 5.)
    assert plan.reason == 'gap fits'
    assert 0. < plan.speed_mps <= 2.  # brake without dropping steering target
    assert abs(plan.target_bearing_rad) > .2
    assert math.isclose(plan.nearest_front_x_m, 7.)


def test_tunnel_gap_narrower_than_inflated_vehicle_still_stops():
    xs = np.linspace(1., 30., 300)
    walls = np.vstack((
        np.column_stack((xs, np.full(len(xs), 3.), np.zeros(len(xs)))),
        np.column_stack((xs, np.full(len(xs), -3.), np.zeros(len(xs))))))
    box = np.array([[7., y, 0.] for y in np.linspace(-.7, .7, 15)])
    plan = FollowTheGap(Config(max_range_m=30.)).plan(
        np.vstack((walls, box)), 0.)
    assert plan.speed_mps == 0.
    assert plan.reason == 'no gap fits vehicle width'
    assert plan.candidate_bearings_rad == ()


def test_blocked_route_bearing_chooses_matching_side():
    planner = FollowTheGap()
    plan = planner.plan(scan_points([(7.0, 0.0, 0.0)]), 0.0,
                        preferred_bearing_rad=.15)
    assert plan.speed_mps > 0
    assert plan.target_bearing_rad > 0


def test_side_obstacle_keeps_target_near_curved_route():
    plan = FollowTheGap().plan(
        scan_points([(5.0, -4.0, 0.0)]), 0.0,
        preferred_bearing_rad=-0.5)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad + 0.5) < 0.35


def test_close_obstacle_stops_at_current_speed():
    plan = FollowTheGap().plan(scan_points([(3.0, 0.0, 0.0)]), 1.5)
    assert plan.speed_mps == 0
    assert 'stopping distance' in plan.reason


def test_no_vehicle_width_gap_stops():
    cfg = Config(max_range_m=5.0)
    planner = FollowTheGap(cfg)
    angles = np.linspace(-math.radians(65), math.radians(65), 131)
    dense_wall = np.column_stack((2 * np.cos(angles), 2 * np.sin(angles),
                                  np.zeros(len(angles))))
    plan = planner.plan(dense_wall, 0.0)
    assert plan.speed_mps == 0


def test_unreachable_side_gap_does_not_clip_into_blocked_heading():
    angles = np.linspace(-.4, .4, 81)
    barrier = np.column_stack((8. * np.cos(angles),
                               8. * np.sin(angles), np.zeros(len(angles))))
    plan = FollowTheGap(Config(max_target_bearing_rad=.35)).plan(barrier, 0.)
    assert plan.speed_mps == 0


def test_road_below_vehicle_collision_height_is_excluded():
    plan = FollowTheGap().plan(
        np.vstack((scan_points([]), [[5., 0., -2.]])), 0.0)
    assert plan.obstacle_points == 2
    assert plan.speed_mps > 0


def test_tunnel_roof_above_vehicle_height_does_not_block_route():
    planner = FollowTheGap()
    points = scan_points([(7., 0., 2.)])
    plan = planner.plan(points, 0.)
    assert plan.front_points == 0
    assert plan.obstacle_points == 2
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad) < .02
    assert len(planner.obstacle_candidates(points)) == 2


def test_low_box_remains_in_vehicle_collision_height():
    plan = FollowTheGap().plan(scan_points([(7., 0., -1.05)]), 0.)
    assert plan.front_points == 1
    assert abs(plan.target_bearing_rad) > .1


def test_tunnel_arch_retains_walls_but_not_overhead_returns():
    xs, ys = np.meshgrid(np.linspace(5., 12., 12),
                         np.linspace(-4., 4., 33))
    # Roof is high at the road center, while the side wall extends to road level.
    z = 2.0 - 2.0 * (np.abs(ys) / 4.) ** 2
    arch = np.column_stack((xs.ravel(), ys.ravel(), z.ravel()))
    planner = FollowTheGap()
    kept = planner.obstacle_candidates(arch)
    assert 0 < len(kept) < len(arch)
    assert np.all(np.abs(kept[:, 1]) > 2.)
    assert planner.plan(arch, 0.).front_points == 0


def test_side_returns_allow_cautious_exit_when_forward_sector_has_no_hits():
    angles = np.deg2rad(np.linspace(75., 85., 20))
    points = np.column_stack((10. * np.cos(angles),
                              10. * np.sin(angles), np.zeros(len(angles))))
    plan = FollowTheGap(Config(front_fov_deg=130., max_speed_mps=5.)).plan(points, 2., .1)
    assert plan.speed_mps == 2.
    assert abs(plan.target_bearing_rad - .1) < 1e-9
    assert 'cautious' in plan.reason


def test_sparse_side_returns_still_stop():
    points = np.array([[1., 10., 0.]] * 5)
    plan = FollowTheGap(Config(front_fov_deg=130., max_speed_mps=5.)).plan(points, 2.)
    assert plan.speed_mps == 0.


def test_blackout_scan_uses_only_minus_135_to_plus_135_degrees():
    bearings = np.deg2rad([-140., -110., 0., 110., 140.])
    points = np.column_stack((10. * np.cos(bearings),
                              10. * np.sin(bearings), np.zeros(len(bearings))))
    planner = FollowTheGap()
    plan = planner.plan(points, 0.)
    assert math.isclose(math.degrees(planner.half_fov_rad), 135.)
    assert plan.obstacle_points == 3
    assert plan.front_points == 1
    assert len(planner.obstacle_candidates(points)) == plan.obstacle_points


def test_visualization_boxes_keep_all_ftg_hits_including_single_point():
    cloud = np.array([[5., -.1, 0.], [5.2, 0., .1],
                      [9., 2., -.5], [9.2, 2.1, -.4],
                      [14., -3., 0.]])
    planner = FollowTheGap()
    candidates = planner.obstacle_candidates(cloud)
    boxes = cluster_boxes(candidates)
    assert len(candidates) == planner.plan(cloud, 0.).obstacle_points
    assert sorted(box[2] for box in boxes) == [1, 2, 2]
    assert sum(box[2] for box in boxes) == len(candidates)


def test_visualization_oriented_box_fits_slanted_returns_tightly():
    x = np.linspace(2., 12., 80)
    slanted = np.column_stack((x, .45 * x + 1., np.zeros(len(x))))
    axis_box = cluster_boxes(slanted)[0]
    fitted = cluster_boxes(slanted, oriented=True)[0]
    assert fitted[2] == len(slanted)
    assert fitted[5][0] * fitted[5][1] < .1 * np.prod(axis_box[1][:2] - axis_box[0][:2])
    assert abs(fitted[6]) > .1
    line = line_candidate_points(fitted[7], fitted[5], fitted[6])
    assert line is not None and len(line) > 3
    assert np.ptp(line[:, 0]) > 8.


def test_visualization_compact_obstacle_stays_box():
    x, y = np.meshgrid(np.linspace(4., 5., 5), np.linspace(-.5, .5, 5))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    box = cluster_boxes(points, oriented=True)[0]
    assert line_candidate_points(box[7], box[5], box[6]) is None


def test_rear_quadrant_returns_cannot_authorize_no_forward_crawl():
    bearings = np.deg2rad(np.linspace(100., 120., 20))
    points = np.column_stack((10. * np.cos(bearings),
                              10. * np.sin(bearings), np.zeros(len(bearings))))
    plan = FollowTheGap(Config(max_range_m=5.)).plan(points, 0.)
    assert plan.speed_mps == 0.
    assert plan.reason == 'no forward returns'


def test_steering_speed_cap_changes_continuously():
    angles = np.deg2rad([-55., 55.])
    walls = np.column_stack((12. * np.cos(angles), 12. * np.sin(angles),
                             np.zeros(len(angles))))
    planner = FollowTheGap(Config(max_speed_mps=5.))
    speeds = [planner.plan(walls, 1., bearing).speed_mps
              for bearing in (.02, .03, .04, .05)]
    assert all(a >= b for a, b in zip(speeds, speeds[1:]))
    assert max(a - b for a, b in zip(speeds, speeds[1:])) < .5
