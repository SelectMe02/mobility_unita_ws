"""Route checks used to recover GPS after LiDAR drift at a tunnel exit."""
import importlib.util
from pathlib import Path

import numpy as np

from leo_guard.traffic_local import RouteProgress
from leo_avoidance.ftg_local import Config, FollowTheGap


path = Path(__file__).parents[1] / 'scripts/driving_state_node_local.py'
spec = importlib.util.spec_from_file_location('rejoin_node_test', path)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


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
