import math

import numpy as np

from leo_avoidance.ftg import Config, FollowTheGap


def scan_points(obstacles, radius_m=12.0):
    """Two far side returns plus specified obstacle x,y points."""
    a = np.deg2rad([-55.0, 55.0])
    walls = np.column_stack((radius_m * np.cos(a), radius_m * np.sin(a),
                             np.zeros(len(a))))
    return np.vstack((walls, np.asarray(obstacles, dtype=float).reshape(-1, 3)))


def test_empty_cloud_stops():
    plan = FollowTheGap().plan(np.empty((0, 3)), 0.0)
    assert plan.speed_mps == 0 and 'empty' in plan.reason


def test_open_corridor_keeps_forward_heading():
    plan = FollowTheGap().plan(scan_points([]), 0.0)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad) < 1e-9


def test_open_corridor_follows_route_bearing():
    plan = FollowTheGap().plan(scan_points([]), 0.0, preferred_bearing_rad=.2)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad - .2) < .02


def test_center_obstacle_selects_side_gap():
    plan = FollowTheGap().plan(scan_points([(7.0, 0.0, 0.0)]), 0.0)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad) > 0.1
    assert plan.gap_width_m > 0


def test_low_box_and_ground_returns_are_both_included():
    walls = scan_points([])
    low_box = np.array([[7., y, -1.05] for y in np.linspace(-.7, .7, 9)])
    ground = np.array([[5., 0., -1.23]])
    plan = FollowTheGap().plan(np.vstack((walls, low_box, ground)), 0.)
    assert plan.speed_mps > 0
    assert abs(plan.target_bearing_rad) > .1
    assert plan.front_points == len(low_box) + 1
    assert plan.nearest_front_x_m == 5.


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


def test_ground_only_returns_are_not_height_filtered():
    plan = FollowTheGap().plan(scan_points([]) + [0, 0, -2], 0.0)
    assert plan.obstacle_points == 2
    assert plan.speed_mps > 0


def test_high_returns_are_not_height_filtered():
    plan = FollowTheGap().plan(scan_points([(7., 0., 2.)]), 0.)
    assert plan.front_points == 1
    assert abs(plan.target_bearing_rad) > .1


def test_side_returns_allow_cautious_exit_when_forward_sector_has_no_hits():
    angles = np.deg2rad(np.linspace(75., 85., 20))
    points = np.column_stack((10. * np.cos(angles),
                              10. * np.sin(angles), np.zeros(len(angles))))
    plan = FollowTheGap(Config(max_speed_mps=5.)).plan(points, 2., .1)
    assert plan.speed_mps == 2.
    assert abs(plan.target_bearing_rad - .1) < 1e-9
    assert 'cautious' in plan.reason


def test_sparse_side_returns_still_stop():
    points = np.array([[1., 10., 0.]] * 5)
    plan = FollowTheGap(Config(max_speed_mps=5.)).plan(points, 2.)
    assert plan.speed_mps == 0.


def test_steering_speed_cap_changes_continuously():
    angles = np.deg2rad([-55., 55.])
    walls = np.column_stack((12. * np.cos(angles), 12. * np.sin(angles),
                             np.zeros(len(angles))))
    planner = FollowTheGap(Config(max_speed_mps=5.))
    speeds = [planner.plan(walls, 1., bearing).speed_mps
              for bearing in (.02, .03, .04, .05)]
    assert all(a >= b for a, b in zip(speeds, speeds[1:]))
    assert max(a - b for a, b in zip(speeds, speeds[1:])) < .5
