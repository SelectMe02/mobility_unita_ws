"""Route checks used to recover GPS after LiDAR drift at a tunnel exit."""
import importlib.util
import json
from pathlib import Path
from collections import deque
import threading
from unittest.mock import Mock, patch

import numpy as np

from leo_guard.traffic_local import RouteProgress
from leo_avoidance.ftg_local import Config, FollowTheGap
from leo_avoidance.driving_state_local import DrivingState
from leo_avoidance.scan_odometry import ScanOdometry


path = Path(__file__).parents[1] / 'scripts/driving_state_node_local.py'
spec = importlib.util.spec_from_file_location('rejoin_node_test', path)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


def test_s14_ground_profile_uses_exact_shared_checkpoint_boundaries_only():
    import json
    project_file = Path(__file__).parents[2] / 'unita_waypoint/config/map_tunner_project.json'
    project = json.loads(project_file.read_text())
    route = RouteProgress(node.load_route_points(project_file), max_error=7.)
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    state.s14_ground_bounds = node.s14_ground_interval(project_file, route)
    state.s14_ground_reference_z_m = -1.61
    start, end = state.s14_ground_bounds
    line = project['sectors'][13]['racelines']['1']
    for point, expected in ((line[0], start), (line[-1], end)):
        index = np.argmin(np.linalg.norm(route.points - (point['x'], point['y']), axis=1))
        assert abs(route.s[index] - expected) < 1e-8
    assert state.ground_reference_for_progress(start - 1e-6) is None
    assert state.ground_reference_for_progress(start) == -1.61
    assert state.ground_reference_for_progress(end - 1e-6) == -1.61
    assert state.ground_reference_for_progress(end) is None
    # Check every saved segment, including the rest of S15's GPS-denied window.
    for progress in (route.s[:-1] + route.s[1:]) / 2.:
        reference = state.ground_reference_for_progress(float(progress))
        assert reference == (-1.61 if start <= progress < end else None)
    assert state.ground_reference_for_progress(float('nan')) is None


def test_csv_without_sector_metadata_keeps_original_ground_filter():
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    route = RouteProgress(np.array([[0., 0.], [1., 0.], [2., 0.]]))
    state.s14_ground_bounds = node.s14_ground_interval('route.csv', route)
    assert state.s14_ground_bounds is None
    assert state.ground_reference_for_progress(1.) is None


def test_gps_rejoin_accepts_nearby_route_aligned_fix():
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    points = np.column_stack((np.arange(0., 101., .5), np.zeros(202)))
    state.blackout_route = RouteProgress(points)
    state.rejoin_route = RouteProgress(points, max_error=2.0)
    assert state.route_rejoin_ok((53., .4, 0.), (50., .3, 0.), 100)
    assert state.route_rejoin_ok((53., 1.8, 0.), (50., .3, 0.), 100)
    assert state.rejoin_diagnostic['gps_cross_track_m'] == 1.8
    assert not state.route_rejoin_ok((53., 1.95, 0.), (50., .3, 0.), 100)
    assert not state.rejoin_diagnostic['route_ok']
    assert not state.route_rejoin_ok((53., 3.5, 0.), (50., .3, 0.), 100)
    assert not state.route_rejoin_ok((70., .4, 0.), (50., .3, 0.), 100)
    assert not state.route_rejoin_ok((53., .4, .7), (50., .3, 0.), 100)


def test_logged_tunnel_exit_rejoins_after_eight_metre_icp_drift():
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    points = np.loadtxt(Path(__file__).parents[2] /
                        'unita_waypoint/config/waypoints.csv')
    state.blackout_route = RouteProgress(points)
    state.rejoin_route = RouteProgress(points, max_error=2.0)
    lidar = (-91.04113461701691, -549.9376672888812, -2.9718400176214845)
    gps = (-83.12988601776306, -548.9538922733627, -2.9936428697593134)
    state.blackout_route.index = 4080
    assert state.route_rejoin_ok(gps, lidar, 4080)
    assert state.rejoin_diagnostic['gps_route_index'] == 4064
    assert -9. < state.rejoin_diagnostic['route_progress_delta_m'] < -7.



def test_safe_gap_returns_toward_predicted_waypoints_after_pass():
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    points = np.column_stack((np.arange(0., 101., .5), np.zeros(202)))
    state.route = RouteProgress(points, max_error=4.0, backward_window=30)
    state.route_lookahead_m = 5.0
    state.exit_route_s = 1000.
    state.exit_route_offset = np.zeros(2)
    state.exit_rejoin_start_m = 35.
    state.exit_rejoin_full_m = 12.
    planner = FollowTheGap(Config(max_speed_mps=5.0))
    walls = np.array([[8., -4., 0.], [8., 4., 0.]])
    obstacle = np.array([[7., y, 0.] for y in np.linspace(-.6, .6, 7)])

    route_bearing = state.route_bearing((10., 0., 0.))
    passing = planner.plan(np.vstack((walls, obstacle)), 0., route_bearing)
    assert passing.speed_mps > 0.
    assert abs(passing.target_bearing_rad) > .1

    # Once the obstacle is behind and the car is 2.5 m off the route,
    # the original waypoint path must become the preferred safe heading.
    return_bearing = state.route_bearing((16., -2.5, 0.))
    returning = planner.plan(walls, 1., return_bearing)
    assert return_bearing > .35
    assert returning.speed_mps > 0.
    assert returning.target_bearing_rad > .1


def test_exit_waypoint_aims_toward_photo_coordinate():
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    points = np.column_stack((np.arange(0., 101., .5), np.zeros(202)))
    state.route = RouteProgress(points, max_error=4.)
    state.route_lookahead_m = 5.
    state.exit_route_s = 60.
    state.exit_route_offset = np.array((0., 1.))
    state.exit_rejoin_start_m = 35.
    state.exit_rejoin_full_m = 12.
    far_bearing = state.route_bearing((20., 0., 0.))
    near_bearing = state.route_bearing((48., 0., 0.))
    assert abs(far_bearing) < 1e-12
    assert near_bearing > 0.


def test_route_projection_lost_on_another_lane_is_reacquired():
    # CP8-CP11 runs on Raceline 3, far from this Raceline 1 projection; the
    # windowed search must not stay stuck at the old index afterwards.
    points = np.column_stack((np.arange(0., 400., .5), np.zeros(800)))
    route = RouteProgress(points, max_error=7.0, backward_window=30)
    route.locate(10., 0., 0.)
    progress, error = node.locate_or_reacquire(route, 300., .5, 0.)
    assert abs(progress - 300.) < 1e-6 and error == .5


def exit_state():
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    along = np.arange(0., 101., .5)
    yaw = -.2
    points = np.column_stack((along*np.cos(yaw), along*np.sin(yaw)))
    state.route = RouteProgress(points, max_error=7., backward_window=30)
    state.route_lookahead_m = 5.
    state.exit_route_s = 80.
    state.exit_route_offset = np.zeros(2)
    state.exit_rejoin_start_m = 35.
    state.exit_rejoin_full_m = 12.
    state.exit_rejoin_active = False
    state.exit_progress_m = None
    state.exit_pose_correction = None
    state.exit_wall_pose_max_offset_m = 1.5
    state.exit_wall_pose_max_step_m = .75
    state.exit_wall_pose_extension_m = 25.
    return state


def test_exit_guidance_is_disabled_outside_the_existing_exit_window():
    state = exit_state()
    for progress in (0., 44.9, 45., 80.1, 100.):
        state.route_progress_m = progress
        assert state.exit_rejoin_guidance(0.) == (None, 1., None)
    for progress in (45.1, 56., 68., 80.):
        state.route_progress_m = progress
        heading, weight, remaining = state.exit_rejoin_guidance(0.)
        assert abs(heading + .2) < 1e-9
        assert remaining == 80. - progress
        assert 0. <= weight < 1.
        if progress >= 68.:
            assert weight == 0.


def test_exit_direction_uses_sensor_heading_without_lateral_pose_correction():
    results = []
    bearings = []
    for lateral_drift in (-3., 0., 3.):
        state = exit_state()
        forward = np.array((np.cos(-.2), np.sin(-.2)))
        normal = np.array((-forward[1], forward[0]))
        xy = forward * 68. + normal * lateral_drift
        bearings.append(state.route_bearing((*xy, 0.)))
        results.append(state.exit_rejoin_guidance(0.)[0])
    assert max(bearings) - min(bearings) > .5
    assert np.allclose(results, [-.2, -.2, -.2])
    state = exit_state()
    state.route_progress_m = 68.
    assert abs(state.exit_rejoin_guidance(-.15)[0] + .05) < 1e-9


def test_started_exit_guidance_stays_active_past_the_estimated_exit():
    state = exit_state()
    for progress in (68., 79.9, 80., 80.1, 85., 95.):
        state.route_progress_m = progress
        heading, weight, remaining = state.exit_rejoin_guidance(-.15)
        assert abs(heading + .05) < 1e-9
        assert weight == 0.
        assert remaining == 80. - progress


def test_exit_guidance_does_not_follow_a_post_portal_bend_early():
    state = exit_state()
    # Straight to s=80, then an immediate left bend. Drifted progress must
    # continue to use the straight surveyed portal tangent until GPS returns.
    points = np.array([[0., 0.], [75., 0.], [80., 0.], [80., 20.]])
    state.route = RouteProgress(points, max_error=7.)
    state.exit_route_s = 80.
    state.route_progress_m = 70.
    first = state.exit_rejoin_guidance(0.)[0]
    state.route_progress_m = 90.
    later = state.exit_rejoin_guidance(0.)[0]
    assert abs(first) < 1e-12
    assert abs(later) < 1e-12


def test_s14_exit_wall_correction_preserves_progress_yaw_and_wall_offset():
    state = exit_state()
    state.machine = type('Machine', (), {'had_blackout': True})()
    state.route_progress_m = 70.
    state.exit_rejoin_active = True
    tangent = state.route.delta[0] / np.linalg.norm(state.route.delta[0])
    normal = np.array((-tangent[1], tangent[0]))
    xy = tangent * 70. + normal * 6.8
    pose = (float(xy[0]), float(xy[1]), -.2)
    plan = node.Plan(5., 0., 'gap fits', 20., 5.,
                     right_wall_clearance_m=2.6,
                     right_wall_target_clearance_m=2.9,
                     right_wall_heading_rad=0.,
                     avoidance_phase='WALL_FOLLOW')
    corrected = state.correct_exit_pose_from_wall(pose, plan)
    assert corrected[2] == pose[2]
    assert np.linalg.norm(np.asarray(corrected[:2]) - np.asarray(pose[:2])) <= .75 + 1e-12
    # Repeated independent wall observations converge to route center plus
    # the measured -0.3 m offset, without changing along-route progress.
    for _ in range(20):
        corrected = state.correct_exit_pose_from_wall(corrected, plan)
    along = corrected[0]*tangent[0] + corrected[1]*tangent[1]
    lateral = tangent[0]*corrected[1] - tangent[1]*corrected[0]
    assert abs(along - 70.) < 1e-8
    assert abs(lateral + .3) < 1e-8
    assert state.exit_pose_correction['method'] == 's14_exit_wall'


def test_exit_wall_correction_never_changes_pose_with_obstacle_evidence():
    state = exit_state();state.machine = type('Machine', (), {'had_blackout': True})()
    state.route_progress_m = 70.;state.exit_rejoin_active = True
    pose = (70., 6.8, -.2)
    base = dict(right_wall_clearance_m=2.6,right_wall_target_clearance_m=2.9,
                right_wall_heading_rad=0.,avoidance_phase='PASSING')
    for plan in (node.Plan(5.,0.,'passing',20.,5.,front_points=1,**base),
                 node.Plan(0.,0.,'stop',2.,0.,obstacle_stop=True,**base),
                 node.Plan(5.,0.,'crossing',20.,5.,crossing_threats=1,**base)):
        assert state.correct_exit_pose_from_wall(pose,plan) == pose


def exit_tick_state():
    """Exercise the real node tick and GPS acceptance without ROS transport."""
    state = exit_state()
    state.lock = threading.RLock()
    state.machine = DrivingState(ScanOdometry(), max_blackout_distance_m=200.)
    xy = state.route.points[162]
    pose = (*xy, -.2)
    state.machine.map_pose = pose
    state.machine.mode = DrivingState.BLACKOUT
    state.machine.had_blackout = True
    state.machine.scan_valid = True
    state.machine.last_scan_wall = 10.
    state.machine.update_raw_gps(0., 0., 10.)
    state.machine.update_odom(pose, 10.)
    state.machine.update_avoidance(True, 10.)
    state.blackout_route = RouteProgress(state.route.points, max_error=7.)
    state.rejoin_route = RouteProgress(state.route.points, max_error=2.)
    state.blackout_start_index = 0
    state.blackout_end_index = 190
    state.rejoin_diagnostic = None
    state.previous_output_pose = None
    state.latest_plan = (node.Plan(5., 0., 'test safe plan', 20., 5.), 10.)
    state.latest_speed = (5., 10.)
    state.speed_timeout_s = .35
    state.last_cloud_received_wall = 10.
    state.last_cloud_stamp_age_s = 0.
    state.last_plan_completed_wall = 10.
    state.config = Config()
    state.target_distance_m = 4.
    for name in ('pose', 'valid', 'mode', 'status', 'target', 'speed',
                 'avoidance_status'):
        setattr(state, name + '_pub', Mock())
    state.route_progress_m = 68.
    state.exit_rejoin_guidance(0.)
    return state


def tick_at_capture(state):
    with patch.object(node.time, 'monotonic', return_value=10.), \
            patch.object(node.rospy.Time, 'now', return_value=node.rospy.Time(10.)), \
            patch.object(node.rospy, 'logwarn_throttle'):
        state.tick(None)


def test_exit_guidance_only_resets_after_gps_rejoin_is_accepted():
    state = exit_tick_state()
    # A fresh but disagreeing fix must keep both BLACKOUT and exit guidance.
    state.machine.update_raw_gps(37., 127., 10.)
    state.machine.update_gps_valid(True, 10.)
    state.machine.update_gps((90., 0., -.2), 10.)
    tick_at_capture(state)
    assert state.machine.gps_available(10.)
    assert state.machine.mode == DrivingState.BLACKOUT
    state.route_progress_m = 85.
    assert state.exit_rejoin_guidance(0.)[0] is not None
    # Only the existing select() decision releases the latch.
    state.machine.update_gps(state.machine.map_pose, 10.)
    tick_at_capture(state)
    assert state.machine.mode == DrivingState.GPS
    assert not state.exit_rejoin_active
    assert state.exit_rejoin_guidance(0.) == (None, 1., None)
    # No stale latch on the next approach; activation uses the same window.
    state.route_progress_m = 44.
    assert state.exit_rejoin_guidance(0.) == (None, 1., None)
    state.route_progress_m = 68.
    assert state.exit_rejoin_guidance(0.)[0] is not None


def test_latched_exit_guidance_does_not_bypass_sensor_fault_stop():
    state = exit_tick_state()
    state.machine.last_scan_wall = 9.
    tick_at_capture(state)
    assert state.machine.mode == DrivingState.HOLD
    assert state.exit_rejoin_active
    assert state.valid_pub.publish.call_args[0][0].data is False
    assert state.speed_pub.publish.call_args[0][0].data == 0.
    state.target_pub.publish.assert_not_called()


def test_latched_exit_guidance_does_not_extend_allowed_blackout_section():
    state = exit_tick_state()
    state.blackout_end_index = 120
    tick_at_capture(state)
    assert state.machine.mode == DrivingState.HOLD
    assert state.machine.reason == 'GPS blackout outside configured tunnel section'
    assert state.valid_pub.publish.call_args[0][0].data is False
    assert state.speed_pub.publish.call_args[0][0].data == 0.
    state.target_pub.publish.assert_not_called()


def test_gps_output_preserves_selected_sample_stamp_across_new_callback():
    state = exit_tick_state()
    state.machine.had_blackout = False
    state.machine.update_raw_gps(37., 127., 10.)
    state.machine.update_gps_valid(True, 10.)
    state.gps_pose_max_age_s = 1.
    state.gps_history = deque(maxlen=100)
    sample = node.PoseStamped()
    sample.header.frame_id = 'map'
    sample.header.stamp = node.rospy.Time.from_sec(9.85)
    sample.pose.position.x, sample.pose.position.y = state.machine.map_pose[:2]
    sample.pose.orientation.w = 1.
    with patch.object(node.time, 'monotonic', return_value=10.), \
            patch.object(node.rospy.Time, 'now', return_value=node.rospy.Time(10.)):
        state.on_gps_pose(sample)
    first_pose = state.machine.gps_pose[0]
    # Publish callbacks run after the lock is released. Simulate a newer GPS
    # arriving then: the selected coordinates must retain their old stamp.
    newer = node.PoseStamped()
    newer.header.frame_id = 'map'
    newer.header.stamp = node.rospy.Time.from_sec(9.95)
    newer.pose.position.x, newer.pose.position.y = first_pose[0] + .1, first_pose[1]
    newer.pose.orientation.w = 1.
    state.mode_pub.publish.side_effect = lambda _: state.on_gps_pose(newer)
    tick_at_capture(state)
    message = state.pose_pub.publish.call_args[0][0]
    assert abs(message.header.stamp.to_sec() - 9.85) < 1e-8
    assert message.pose.position.x == first_pose[0]
    assert abs(state.gps_stamp - 9.95) < 1e-8


def test_blackout_output_keeps_existing_time_convention():
    state = exit_tick_state()
    state.gps_stamp = 3.
    tick_at_capture(state)
    assert state.machine.mode == DrivingState.BLACKOUT
    assert state.pose_pub.publish.call_args[0][0].header.stamp.to_sec() == 10.


def continuous_exit_state():
    state = exit_tick_state()
    state.machine.map_pose = (90., -17., 0.)
    state.machine.update_raw_gps(37., 127., 9.5)
    state.machine.update_raw_gps(37., 127., 10.)
    state.machine.update_gps_valid(True, 10.)
    state.route = RouteProgress([[0., 0.], [100., 0.], [200., 0.]], max_error=7.)
    state.rejoin_route = RouteProgress(state.route.points, max_error=2.)
    state.exit_route_s = 80.
    state.exit_rejoin_active = True
    state.gps_history = deque([((81. + i*.5, 2.5, 0.), 9.6+i*.1, 9.6+i*.1)
                               for i in range(5)], maxlen=100)
    state.machine.update_gps(state.gps_history[-1][0], 10.)
    state.latest_speed = (5., 10.)
    return state


def test_continuous_exit_gps_recovers_despite_unprojectable_icp():
    state = continuous_exit_state()
    assert not state.route_rejoin_ok(state.machine.gps_pose[0], state.machine.map_pose, 0)
    assert state.exit_gps_rejoin_ok(10.)
    assert state.rejoin_diagnostic['method'] == 'continuous_exit_gps'
    mode, pose, _ = state.machine.select(10., blackout_allowed=False, route_rejoin_ok=True)
    assert mode == DrivingState.GPS and pose == state.gps_history[-1][0]


def test_real_tick_publishes_continuous_rejoin_event_and_gps_pose():
    state = continuous_exit_state()
    state.blackout_route = RouteProgress(state.route.points, max_error=7.)
    state.gps_stamp = 10.
    state.blackout_end_index = 0
    tick_at_capture(state)
    assert state.machine.mode == DrivingState.GPS
    status = json.loads(state.status_pub.publish.call_args[0][0].data)
    assert status['gps_rejoin_stamp'] == 10.
    assert status['gps_rejoin_diagnostic']['method'] == 'continuous_exit_gps'
    assert not state.exit_rejoin_active


def test_exit_gps_recovery_rejects_unstable_or_wrong_route_samples():
    for fault in ('blip','stale','gap','repeat','jump','far','heading','inside','unlatched','speed'):
        state = continuous_exit_state()
        if fault == 'blip': state.machine.gps_return_since = 9.85
        elif fault == 'stale': state.machine.gps_pose = (state.machine.gps_pose[0], 9.)
        elif fault == 'gap': del state.gps_history[2];del state.gps_history[1]
        elif fault == 'repeat': state.gps_history=deque([(p,t,9.) for p,t,s in state.gps_history])
        elif fault == 'jump': state.gps_history[2]=((90.,2.5,0.),9.8,9.8)
        elif fault in ('far','heading','inside'):
            state.gps_history=deque([((x if fault!='inside' else x-40.,
                                      y if fault!='far' else 4.,
                                      yaw if fault!='heading' else .6),t,s)
                                     for (x,y,yaw),t,s in state.gps_history])
        elif fault == 'unlatched': state.exit_rejoin_active=False
        elif fault == 'speed': state.latest_speed=(0.,10.)
        assert not state.exit_gps_rejoin_ok(10.),fault
