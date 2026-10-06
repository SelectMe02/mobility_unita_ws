"""Route checks used to recover GPS after LiDAR drift at a tunnel exit."""
import importlib.util
from pathlib import Path

import numpy as np

from leo_guard.traffic import RouteProgress


path = Path(__file__).parents[1] / 'scripts/driving_state_node.py'
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


def test_photo_enu_coordinate_crosses_exit_gate():
    state = node.DrivingStateNode.__new__(node.DrivingStateNode)
    state.exit_xy = np.array((-78.13, -546.24))
    tangent = np.array((-78.54114889744686, -546.1438570211373)) - np.array(
        (-78.07642846488123, -545.9593735619987))
    state.exit_tangent = tangent / np.linalg.norm(tangent)
    state.exit_radius_m = 15.
    state.exit_cross_track_m = 5.
    assert not state.past_exit_gate((-77.5, -546., 0.))
    assert state.past_exit_gate((-78.13, -546.24, 0.))
    assert state.past_exit_gate((-83., -548., 0.))
    assert not state.past_exit_gate((-110., -555., 0.))
