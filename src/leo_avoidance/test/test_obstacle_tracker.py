import math

import numpy as np

from leo_avoidance.ftg import Config, FollowTheGap
from leo_avoidance.obstacle_tracker import ObstacleTracker

LIDAR_TO_REAR_M = 1.4


def box(x, y, half_m=.3):
    grid = np.linspace(-half_m, half_m, 7)
    return np.array([[x + dx, y + dy, 0.] for dx in grid for dy in grid])


def tunnel_walls(half_width_m=3.95):
    x = np.linspace(1., 13., 121)
    return np.vstack((
        np.column_stack((x, np.full(len(x), -half_width_m), np.zeros(len(x)))),
        np.column_stack((x, np.full(len(x), half_width_m), np.zeros(len(x))))))


def lidar_xy(world_xy, ego_pose):
    """World point in the LiDAR frame of a rear-axle Ego pose."""
    dx, dy = world_xy[0] - ego_pose[0], world_xy[1] - ego_pose[1]
    c, s = math.cos(ego_pose[2]), math.sin(ego_pose[2])
    return c * dx + s * dy - LIDAR_TO_REAR_M, -s * dx + c * dy


def test_static_box_stays_static_while_ego_drives_past():
    tracker = ObstacleTracker(LIDAR_TO_REAR_M)
    for step in range(12):
        t = .1 * step
        ego = (5. * t, 0., 0.)
        tracks = tracker.update(box(*lidar_xy((15., 2.), ego)), ego, t)
    assert len(tracks) == 1
    assert tracks[0].velocity_known and not tracks[0].dynamic
    assert math.hypot(tracks[0].vx_mps, tracks[0].vy_mps) < .1


def test_static_box_stays_static_while_ego_turns():
    tracker = ObstacleTracker(LIDAR_TO_REAR_M)
    for step in range(12):
        t = .1 * step
        ego = (3. * t, 0., .4 * t)
        tracks = tracker.update(box(*lidar_xy((12., 1.), ego)), ego, t)
    assert len(tracks) == 1 and not tracks[0].dynamic


def test_crossing_box_is_dynamic_with_lateral_velocity_in_lidar_frame():
    tracker = ObstacleTracker(LIDAR_TO_REAR_M)
    for step in range(8):
        t = .1 * step
        ego = (0., 0., 0.)
        tracks = tracker.update(box(*lidar_xy((10., -3. + 1.5 * t), ego)),
                                ego, t)
    assert len(tracks) == 1 and tracks[0].dynamic
    assert math.isclose(tracks[0].vy_mps, 1.5, abs_tol=.1)
    assert abs(tracks[0].vx_mps) < .1


def test_velocity_is_rotated_into_current_lidar_frame():
    tracker = ObstacleTracker(LIDAR_TO_REAR_M)
    ego = (0., 0., math.pi / 2.)
    for step in range(8):
        t = .1 * step
        # Moving toward world -x is moving toward LiDAR +y at this heading.
        tracks = tracker.update(box(*lidar_xy((2. - 1.5 * t, 10.), ego)),
                                ego, t)
    assert tracks[0].dynamic
    assert math.isclose(tracks[0].vy_mps, 1.5, abs_tol=.1)


def test_missing_pose_or_scan_gap_resets_tracks():
    tracker = ObstacleTracker(LIDAR_TO_REAR_M)
    ego = (0., 0., 0.)
    for step in range(8):
        tracker.update(box(10., -3. + .15 * step), ego, .1 * step)
    assert tracker.tracks
    assert tracker.update(box(10., 0.), None, .8) == []
    assert not tracker.tracks
    for step in range(8):
        tracker.update(box(10., -3. + .15 * step), ego, 1. + .1 * step)
    tracks = tracker.update(box(10., -1.65), ego, 3.)
    assert len(tracks) == 1 and not tracks[0].velocity_known


def run_crossing(planner, ego_speed_mps=3., obstacle_vy_mps=1.5, steps=12,
                 start_world=(14., -2.6)):
    """Ego drives +x in a straight tunnel; a box crosses right to left."""
    plans = []
    for step in range(steps):
        t = .1 * step
        ego = (ego_speed_mps * t, 0., 0.)
        obstacle = (start_world[0], start_world[1] + obstacle_vy_mps * t)
        cloud = np.vstack((tunnel_walls(), box(*lidar_xy(obstacle, ego))))
        planner._clock = lambda t=t: t
        plans.append(planner.plan(cloud, ego_speed_mps, wall_follow=True,
                                  ego_pose=ego, stamp=t))
    return plans


def test_crossing_obstacle_is_yielded_to_instead_of_passed_on_its_side():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    plans = run_crossing(planner)
    yielding = [plan for plan in plans if plan.crossing_threats]
    assert yielding, [plan.reason for plan in plans]
    for plan in yielding:
        assert 'yielding to crossing obstacle' in plan.reason
        # Never steer left, toward where the obstacle is heading.
        assert plan.speed_mps == 0. or plan.target_bearing_rad <= .05
    assert any(track[5] for track in yielding[-1].tracked_obstacles)


def test_close_crossing_obstacle_stops_as_obstacle_stop():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    plans = run_crossing(planner, start_world=(6.5, -2.6), steps=10)
    stops = [plan for plan in plans
             if plan.reason == 'yielding to crossing obstacle']
    assert stops, [plan.reason for plan in plans]
    assert all(plan.speed_mps == 0. and plan.obstacle_stop for plan in stops)


def test_static_side_box_is_not_a_crossing_threat():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    plans = run_crossing(planner, obstacle_vy_mps=0., start_world=(14., -1.))
    assert not any(plan.crossing_threats for plan in plans)
    assert not any(track[5] for plan in plans
                   for track in plan.tracked_obstacles)


def test_box_leaving_the_lane_is_not_a_threat():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    # Already left of the lane and still moving left.
    plans = run_crossing(planner, start_world=(14., 2.2))
    assert not any(plan.crossing_threats for plan in plans)


def test_planning_without_pose_keeps_previous_behaviour():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    plan = planner.plan(np.vstack((tunnel_walls(), box(8., 0.))), 2.,
                        wall_follow=True)
    assert plan.tracked_obstacles == () and plan.crossing_threats == 0


def test_noisy_static_box_with_growing_visible_face_is_not_yielded_to():
    """Logged case: a static in-lane box read as moving while approached."""
    rng = np.random.default_rng(3)
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    plans = []
    for step in range(30):
        t = .1 * step
        ego = (5. * t, 0., 0.)
        x, y = lidar_xy((20., -.6), ego)
        # Only the near face first; the far face appears when close.
        depth_m = .3 if x > 12. else 1.6
        jitter = rng.normal(0., .12, 2)
        face = np.array([[x + dx + jitter[0], y + dy + jitter[1], 0.]
                         for dx in np.linspace(0., depth_m, 5)
                         for dy in np.linspace(-.5, .5, 7)])
        planner._clock = lambda t=t: t
        plans.append(planner.plan(np.vstack((tunnel_walls(), face)), 5.,
                                  wall_follow=True, ego_pose=ego, stamp=t))
    assert not any(plan.crossing_threats for plan in plans), [
        (plan.reason, plan.tracked_obstacles) for plan in plans
        if plan.crossing_threats]
    assert not any('yielding' in plan.reason for plan in plans)


def test_single_lateral_spike_does_not_yield():
    tracker = ObstacleTracker(LIDAR_TO_REAR_M)
    ego = (0., 0., 0.)
    ys = [-1., -1., -1., -1., -.8, -1., -1., -1., -1.]
    for step, y in enumerate(ys):
        tracks = tracker.update(box(10., y), ego, .1 * step)
        assert abs(tracks[0].lateral_scans) < 3


def test_zero_speed_from_held_obstacle_cap_is_obstacle_stop():
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    planner._clock = lambda: 0.
    planner._held_obstacle_cap = 0.
    planner._obstacle_cap_until = 1.
    plan = planner.plan(tunnel_walls(), 2., wall_follow=True)
    assert plan.speed_mps == 0. and plan.obstacle_stop


def test_object_closing_in_beside_the_car_is_not_yielded_to():
    """Logged case: braking from 4.9 m/s for a return beside the car."""
    planner = FollowTheGap(Config(max_range_m=30., max_speed_mps=8.33))
    plans = []
    for step in range(14):
        t = .1 * step
        ego = (0., 0., 0.)
        # Beside the body (LiDAR x 1.2 m, front bumper at 2.445 m), moving in.
        cloud = np.vstack((tunnel_walls(), box(1.2, 3. - 1.2 * t)))
        planner._clock = lambda t=t: t
        plans.append(planner.plan(cloud, 4., wall_follow=True,
                                  ego_pose=ego, stamp=t))
    assert any(track[5] for plan in plans for track in plan.tracked_obstacles)
    assert not any(plan.crossing_threats for plan in plans)
    assert not any('yielding' in plan.reason for plan in plans)
