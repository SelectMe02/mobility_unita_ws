import math

import numpy as np

from leo_avoidance.dead_reckoning import DeadReckoner
from leo_avoidance.driving_state import DrivingState, bounded_pose_correction
from leo_avoidance.scan_odometry import Match, ScanOdometry, inverse, transform


def landmarks():
    return np.random.default_rng(9).uniform([-7, -5], [12, 5], (500, 2))


def test_exit_pose_correction_converges_without_ten_metre_jump():
    pose = (-72.82, -546.34, -2.62)
    actual = (-65.09, -538.97, -2.61)
    for _ in range(12):
        next_pose = bounded_pose_correction(pose, actual, 1., .05)
        assert math.hypot(next_pose[0]-pose[0], next_pose[1]-pose[1]) <= 1.000001
        pose = next_pose
    assert math.hypot(pose[0]-actual[0], pose[1]-actual[1]) < .01


def test_iterative_lidar_odometry_recovers_motion_with_odom_prior():
    matcher = ScanOdometry()
    previous = landmarks()
    actual_rear = (.15, .02, .03)
    current = transform(previous, inverse(matcher.lidar_delta_from_rear(actual_rear)))
    result = matcher.match(previous, current, (.14, .02, .029))
    assert result.valid and result.pairs > 100 and result.rmse_m < .01
    estimated = matcher.rear_delta_from_lidar(result.delta_lidar)
    assert np.allclose(estimated, actual_rear, atol=.01)


def test_sparse_cloud_cannot_authorize_blackout():
    matcher = ScanOdometry()
    try:
        matcher.prepare(np.zeros((10, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError('sparse cloud was accepted')


def test_gps_to_blackout_uses_lidar_match_then_returns_to_gps():
    matcher = ScanOdometry()
    state = DrivingState(matcher)
    previous = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(previous, 10.)
    assert state.select(10.)[0] == state.GPS
    state.update_gps_valid(False, 10.1)
    state.update_raw_gps(0., 0., 10.1, status=0)
    state.update_avoidance(True, 10.1)
    state.update_odom((.15, .02, .03), 10.1)
    current = transform(previous, inverse(matcher.lidar_delta_from_rear((.15, .02, .03))))
    state.scan(current, 10.1)
    mode, pose, _ = state.select(10.1)
    assert mode == state.BLACKOUT
    assert np.allclose(pose, (100.15, 200.02, .03), atol=.02)
    state.update_gps((100.15, 200.02, .03), 10.2)
    state.update_raw_gps(37., 127., 10.2)
    state.update_gps_valid(True, 10.2)
    assert state.select(10.2)[0] == state.GPS


def test_tunnel_icp_yaw_bias_does_not_move_estimate_toward_wall():
    matcher = ScanOdometry()
    state = DrivingState(matcher)
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    state.select(10.)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_avoidance(True, 10.1)
    matcher.match = lambda *_: Match((1., .4, .05), 100, .02, True)
    matcher.rear_delta_from_lidar = lambda delta: delta
    for step in range(1, 11):
        t = 10. + step * .1
        state.update_raw_gps(0., 0., t)
        state.update_avoidance(True, t)
        state.update_odom((float(step), 0., 0.), t)
        state.scan(cloud, t)
        assert state.select(t)[0] == state.BLACKOUT
    assert abs(state.map_pose[2]) < 1e-9
    assert math.isclose(state.map_pose[1] - 200., .4, abs_tol=.01)


def test_gps_return_waits_briefly_for_localization_pose():
    matcher = ScanOdometry()
    state = DrivingState(matcher, gps_return_grace_s=1.5)
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    state.select(10.)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_avoidance(True, 10.1)
    state.update_odom((.1, 0., 0.), 10.1)
    state.scan(cloud, 10.1)
    assert state.select(10.1)[0] == state.BLACKOUT
    state.update_raw_gps(37., 127., 10.2)
    state.update_odom((.1, 0., 0.), 10.2)
    state.scan(cloud, 10.2)
    state.update_avoidance(True, 10.2)
    assert state.select(10.2)[0] == state.BLACKOUT
    state.update_gps((100.1, 200., 0.), 10.3)
    state.update_gps_valid(True, 10.3)
    assert state.select(10.3)[0] == state.GPS


def test_gps_rejoin_succeeds_after_exit_lidar_features_disappear():
    state = DrivingState(ScanOdometry())
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    state.select(10.)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_avoidance(True, 10.1)
    state.update_odom((0., 0., 0.), 10.1)
    state.scan(cloud, 10.1)
    assert state.select(10.1)[0] == state.BLACKOUT
    state.scan_valid = False
    state.update_raw_gps(37., 127., 10.2)
    state.update_gps((100.2, 200., 0.), 10.2)
    state.update_gps_valid(True, 10.2)
    assert state.select(10.2)[0] == state.GPS


def test_blackout_without_gps_anchor_holds():
    state = DrivingState(ScanOdometry())
    state.update_raw_gps(0., 0., 10.)
    assert state.select(10., blackout_allowed=False) == (
        state.HOLD, None, 'GPS anchor unavailable for blackout')


def test_blackout_outside_tunnel_section_holds():
    state = DrivingState(ScanOdometry())
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    assert state.select(10.)[0] == state.GPS
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    assert state.select(10.1, blackout_allowed=False) == (
        state.HOLD, None, 'GPS blackout outside configured tunnel section')


def test_failed_icp_scan_references_next_scan_without_authorizing_motion():
    matcher = ScanOdometry()
    state = DrivingState(matcher)
    previous = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(previous, 10.)
    state.select(10.)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_avoidance(True, 10.1)
    real_match = matcher.match
    matcher.match = lambda *_: Match((0., 0., 0.), 0, math.inf, False)
    state.update_odom((.5, 0., 0.), 10.1)
    state.scan(transform(previous, (-.5, 0., 0.)), 10.1)
    assert state.select(10.1)[0] == state.HOLD
    assert math.isclose(state.map_pose[0], 100.5, abs_tol=.01)
    assert math.isclose(state.unmatched_distance_m, .5, abs_tol=.01)
    matcher.match = real_match
    state.update_odom((1., 0., 0.), 10.2)
    state.scan(transform(previous, (-1., 0., 0.)), 10.2)
    mode, pose, _ = state.select(10.2)
    assert mode == state.BLACKOUT
    assert math.isclose(pose[0], 101., abs_tol=.02)
    assert state.unmatched_distance_m == 0.


def test_stale_scan_holds_brake_state():
    matcher = ScanOdometry()
    state = DrivingState(matcher)
    state.update_gps((0., 0., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(landmarks(), 10.)
    state.select(10.)
    state.update_gps_valid(False, 10.5)
    state.update_raw_gps(0., 0., 10.5)
    state.update_avoidance(True, 10.5)
    assert state.select(10.5)[0] == state.HOLD


def test_gps_rejoin_mismatch_holds_until_route_corroborates():
    matcher = ScanOdometry()
    state = DrivingState(matcher)
    previous = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(previous, 10.)
    state.select(10.)
    state.update_gps_valid(False, 10.1)
    state.update_raw_gps(0., 0., 10.1)
    state.update_avoidance(True, 10.1)
    state.update_odom((.15, 0., 0.), 10.1)
    state.scan(transform(previous, (-.15, 0., 0.)), 10.1)
    assert state.select(10.1)[0] == state.BLACKOUT
    state.update_gps((110., 200., 0.), 10.2)
    state.update_raw_gps(37., 127., 10.2)
    state.update_gps_valid(True, 10.2)
    assert state.select(10.2)[0] == state.HOLD
    state.reason = 'too few independent LiDAR features'
    assert state.select(10.21) == (
        state.HOLD, None, 'GPS return disagrees with LiDAR odometry')
    assert state.select(10.22, route_rejoin_ok=True)[0] == state.GPS


def test_route_corroborated_gps_rejoin_replaces_drifted_lidar_pose():
    state = DrivingState(ScanOdometry())
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    assert state.select(10.)[0] == state.GPS
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_avoidance(True, 10.1)
    state.update_odom((0., 0., 0.), 10.1)
    state.scan(cloud, 10.1)
    assert state.select(10.1)[0] == state.BLACKOUT
    state.update_gps((103., 200., 0.), 10.2)
    state.update_raw_gps(37., 127., 10.2)
    state.update_gps_valid(True, 10.2)
    mode, pose, reason = state.select(10.2, route_rejoin_ok=True)
    assert mode == state.GPS and pose == (103., 200., 0.)
    assert state.map_pose == pose and not state.had_blackout
    assert 'route-corroborated' in reason
    assert state.last_rejoin_error[0] > state.max_rejoin_error_m


def test_sim_exit_uses_ego_pose_for_waypoint_steering_until_gps_returns():
    state = DrivingState(ScanOdometry())
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    state.select(10.)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_avoidance(True, 10.1)
    state.update_odom((0., 0., 0.), 10.1)
    state.scan(cloud, 10.1)
    assert state.select(10.1)[0] == state.BLACKOUT
    mode, pose, _ = state.select(10.2, sim_exit_pose=(102., 201., .1),
                                 sim_exit_active=True)
    assert mode == state.SIM_EXIT and pose == (102., 201., .1)
    assert state.select(10.3, sim_exit_active=True)[0] == state.HOLD
    state.update_gps((110., 201., .1), 10.35)
    state.update_raw_gps(37., 127., 10.35)
    state.update_gps_valid(True, 10.35)
    mode, pose, _ = state.select(10.35, sim_exit_pose=(102.2, 201., .1),
                                 sim_exit_active=True)
    assert mode == state.SIM_EXIT and pose == (102.2, 201., .1)
    state.update_gps((102., 201., .1), 10.4)
    state.update_raw_gps(37., 127., 10.4)
    state.update_gps_valid(True, 10.4)
    assert state.select(10.4, sim_exit_active=True)[0] == state.GPS


def test_rejoin_checks_pose_after_temporary_icp_hold():
    matcher = ScanOdometry()
    state = DrivingState(matcher)
    previous = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(previous, 10.)
    state.select(10.)
    state.update_gps_valid(False, 10.1)
    state.update_raw_gps(0., 0., 10.1)
    state.update_avoidance(True, 10.1)
    state.update_odom((.15, 0., 0.), 10.1)
    state.scan(transform(previous, (-.15, 0., 0.)), 10.1)
    assert state.select(10.1)[0] == state.BLACKOUT
    state.scan_valid = False
    assert state.select(10.2)[0] == state.HOLD
    state.update_gps((110., 200., 0.), 10.21)
    state.update_raw_gps(37., 127., 10.21)
    state.update_gps_valid(True, 10.21)
    assert state.select(10.21)[0] == state.HOLD


def test_dead_reckoner_uses_speed_and_yaw_not_gps():
    odom = DeadReckoner()
    assert odom.update(1., 0., 10.) == (0., 0., 0.)
    x, y, yaw = odom.update(1., 0., 10.1)
    assert math.isclose(x, .1, abs_tol=1e-6)
    assert math.isclose(y, 0., abs_tol=1e-6) and yaw == 0.


def test_tunnel_forward_motion_ignores_biased_wall_icp_translation():
    matcher = ScanOdometry()
    state = DrivingState(matcher, icp_forward_weight=0.)
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    matcher.match = lambda *_: Match((.8, 0., 0.), 100, .01, True)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_odom((.5, 0., 0.), 10.1)
    state.scan(cloud, 10.1)
    assert math.isclose(state.map_pose[0], 100.5, abs_tol=1e-6)


def test_dead_reckoner_keeps_odom_frame_after_heading_gap():
    odom = DeadReckoner(max_interval_s=.25)
    odom.update(2., 0., 10.)
    odom.update(2., 0., 10.1)
    try:
        odom.update(2., 0., 10.5)
    except ValueError:
        pass
    else:
        raise AssertionError('heading gap should withhold odometry')
    assert math.isclose(odom.pose[0], .2, abs_tol=1e-6)
    assert math.isclose(odom.update(2., 0., 10.6)[0], .4, abs_tol=1e-6)


def test_dead_reckoner_integrates_short_heading_delay():
    odom = DeadReckoner(max_interval_s=.6)
    odom.update(2., 0., 10.)
    assert math.isclose(odom.update(2., 0., 10.4)[0], .8, abs_tol=1e-6)


def test_failed_icp_does_not_integrate_beyond_recovery_limit():
    matcher = ScanOdometry()
    state = DrivingState(matcher, max_unmatched_distance_m=1.)
    previous = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(previous, 10.)
    state.select(10.)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    matcher.match = lambda *_: Match((0., 0., 0.), 0, math.inf, False)
    state.update_odom((2., 0., 0.), 10.1)
    state.scan(previous, 10.1)
    assert state.scan_valid is False
    assert state.map_pose == (100., 200., 0.)
    assert 'recovery limit exceeded' in state.scan_fault_reason


def test_invalid_nonzero_gps_never_enters_blackout():
    state = DrivingState(ScanOdometry())
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(landmarks(), 10.)
    assert state.select(10.)[0] == state.GPS
    state.update_gps_valid(False, 10.1)
    state.update_raw_gps(37., 127., 10.1)
    state.update_avoidance(True, 10.1)
    assert state.select(10.1)[0] == state.HOLD
    state.update_raw_gps(0., 127., 10.1)
    assert state.select(10.1)[0] == state.HOLD


def test_zero_gps_requires_safe_gap_and_fresh_zero_sample():
    state = DrivingState(ScanOdometry())
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(landmarks(), 10.)
    state.select(10.)
    state.update_raw_gps(0., 0., 10.1)
    state.update_gps_valid(False, 10.1)
    state.update_odom((0., 0., 0.), 10.1)
    state.scan(landmarks(), 10.1)
    state.update_avoidance(False, 10.1)
    assert state.select(10.1)[0] == state.HOLD
    state.update_avoidance(True, 10.1)
    assert state.select(10.1)[0] == state.BLACKOUT
    assert state.select(10.5)[0] == state.HOLD


def test_missing_gps_stream_enters_blackout_after_debounce():
    state = DrivingState(ScanOdometry(), gps_silence_timeout_s=.75)
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    assert state.select(10.)[0] == state.GPS
    state.update_odom((0., 0., 0.), 10.5)
    state.scan(cloud, 10.5)
    state.update_avoidance(True, 10.5)
    assert state.select(10.5)[0] == state.HOLD
    state.update_odom((0., 0., 0.), 10.8)
    state.scan(cloud, 10.8)
    state.update_avoidance(True, 10.8)
    assert state.select(10.8)[0] == state.BLACKOUT


def test_nonzero_no_fix_holds_even_if_stream_later_stops():
    state = DrivingState(ScanOdometry(), gps_silence_timeout_s=.75)
    cloud = landmarks()
    state.update_gps((100., 200., 0.), 10.)
    state.update_raw_gps(37., 127., 10.)
    state.update_gps_valid(True, 10.)
    state.update_odom((0., 0., 0.), 10.)
    state.scan(cloud, 10.)
    state.select(10.)
    state.update_raw_gps(37., 127., 10.1, status=0)
    state.update_gps_valid(False, 10.1)
    state.update_odom((0., 0., 0.), 11.)
    state.scan(cloud, 11.)
    state.update_avoidance(True, 11.)
    assert state.select(11.)[0] == state.HOLD
