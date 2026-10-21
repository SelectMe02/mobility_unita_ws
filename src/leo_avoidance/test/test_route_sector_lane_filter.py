"""S12 alone clips route perception to the saved 3.2 m lane before clustering."""
import importlib.util
import json
import math
from pathlib import Path
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from leo_avoidance.obstacle_tracker import ObstacleTracker, TrackedObstacle, _Track
from leo_avoidance.route_avoidance import RouteAvoidance


spec = importlib.util.spec_from_file_location(
    'route_sector_lane_filter_test_node',
    Path(__file__).resolve().parents[1] / 'scripts' / 'route_avoidance_node.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def geometry(y=0., width=5.):
    return dict(frame_id='map', raceline_width=width, loop=False, lanes=[
        dict(sector=11, raceline=1, points=[[-40., y], [0., y]]),
        dict(sector=12, raceline=1, points=[[0., y], [40., y]]),
        dict(sector=13, raceline=1, points=[[40., y], [100., y]])])


def node():
    obj = module.RouteAvoidanceNode.__new__(module.RouteAvoidanceNode)
    obj.lock = threading.Lock()
    obj.closed, obj.input_epoch = False, 0
    obj.input_timeout_s = .35
    obj.pose, obj.speed, obj.signal = ((10., 0., 0.), 10.1), (3., 10.1), None
    obj.lidar_to_rear_x_m, obj.debug_lidar_z = 1.4, 1.23
    obj.road_points = module.RoadPointConfig()
    obj.sector_lane_filter_enabled = True
    obj.sector_lane_filter = module.SectorLaneFilter(geometry(), sector=12, width=3.2)
    obj.tracker = ObstacleTracker(1.4, region_x_m=(-10., 60.), region_half_width_m=10.)
    obj.planner = RouteAvoidance(np.column_stack((np.arange(-40., 101.), np.zeros(141))))
    obj.planner.update = MagicMock(wraps=obj.planner.update)
    obj.debug, obj.pub = MagicMock(), MagicMock()
    obj.debug.active.return_value = True
    obj.boundary_debug_counts = {}
    obj.drop_road_boundaries = MagicMock(side_effect=lambda points, pose: points)
    return obj


def cluster(x, y):
    return [(x + dx, y + dy, 1.) for dx in (-.15, 0., .15)
            for dy in (-.15, 0., .15)]


def run_scan(obj, points, pose=(10., 0., 0.)):
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = 'lidar', rospy.Time.from_sec(10.05)
    with patch.object(module.time, 'monotonic', return_value=10.12), \
            patch.object(module.rospy.Time, 'now', return_value=rospy.Time.from_sec(10.12)), \
            patch.object(module.rospy, 'loginfo_throttle'), \
            patch.object(module.point_cloud2, 'read_points', return_value=points), \
            patch.object(module, 'obstacle_points', side_effect=lambda points, config: points):
        obj.process(msg, 10.06, pose, 0)


def test_only_s12_filters_outside_points_and_preserves_real_lane_obstacle():
    points = cluster(15., 0.) + cluster(15., 3.2)
    for x, expected in ((-10., 2), (10., 1), (50., 2)):
        obj = node()
        run_scan(obj, points, (x, 0., 0.))
        planned = obj.planner.update.call_args.args[2]
        assert len(planned) == expected
        assert any(abs(o.y) < .01 for o in planned)
        output = json.loads(obj.pub.publish.call_args.args[0].data)
        assert len(output['obstacles']) == expected
        used = obj.drop_road_boundaries.call_args.args[0]
        assert len(used) == 9 * expected
        debug = obj.debug.publish.call_args
        np.testing.assert_allclose(debug.args[0], points)
        np.testing.assert_allclose(debug.args[1], used)
        assert debug.kwargs['sector_lane_filter_active'] == (x == 10.)
        assert debug.kwargs['lane_removed_points'] == (9 if x == 10. else 0)
        assert debug.kwargs['tracked_obstacles'] == expected
        if x != 10.:
            baseline = node()
            baseline.sector_lane_filter = None
            run_scan(baseline, points, (x, 0., 0.))
            assert output == json.loads(baseline.pub.publish.call_args.args[0].data)


def test_gate_uses_capture_pose_and_releases_at_exact_s13_boundary():
    for capture_x, expected in ((-.01, 2), (0., 1), (39.9, 1), (40., 2)):
        obj = node()
        # Latest pose is deliberately elsewhere; scan capture pose owns the gate.
        obj.pose = ((-10., 0., 0.), 10.1)
        run_scan(obj, cluster(15., 0.) + cluster(15., 3.2), (capture_x, 0., 0.))
        assert len(obj.planner.update.call_args.args[2]) == expected


def test_boundary_points_are_clipped_before_they_change_cluster_extent():
    obj = node()
    points = [(x, y, 1.) for x in (14.9, 15., 15.1)
              for y in (1.45, 1.55, 1.65, 1.75, 1.85)]
    run_scan(obj, points)
    planned = obj.planner.update.call_args.args[2]
    assert len(planned) == 1
    assert math.isclose(planned[0].y, 1.5, abs_tol=1e-9)
    assert math.isclose(planned[0].radius, .15, abs_tol=1e-9)
    assert obj.debug.publish.call_args.kwargs['lane_removed_points'] == 9
    assert len(obj.drop_road_boundaries.call_args.args[0]) == 6


def test_world_transform_uses_capture_yaw_and_lidar_to_rear_offset():
    obj = node()
    # Facing north on an eastbound saved lane: lidar y maps to negative world x.
    world = np.asarray(cluster(25., 0.) + cluster(25., 3.2))
    points = np.column_stack((world[:, 1] - 1.4, 10. - world[:, 0], world[:, 2]))
    obj.tracker.region_half_width_m = 30.
    run_scan(obj, points, (10., 0., math.pi / 2.))
    planned = obj.planner.update.call_args.args[2]
    assert len(planned) == 1
    assert math.isclose(planned[0].x, 25., abs_tol=1e-9)
    assert math.isclose(planned[0].y, 0., abs_tol=1e-9)


def test_outside_histories_and_predictions_are_pruned_only_in_s12():
    for pose_x in (-10., 10., 50.):
        obj = node()
        obj.tracker.tracks = [
            _Track(101, history=[(9.9, 25., 3.2)]),
            _Track(102, history=[(9.9, 25., 1.5)], velocity=(0., 2.), velocity_known=True),
            _Track(103, history=[(9.9, 25., 0.)]),
            _Track(104, history=[(9.9, -10., 0.)])]
        run_scan(obj, [], (pose_x, 0., 0.))
        assert [t.track_id for t in obj.tracker.tracks] == (
            [103] if pose_x == 10. else [101, 102, 103, 104])
        assert obj.debug.publish.call_args.kwargs['lane_removed_tracks'] == (
            3 if pose_x == 10. else 0)


def test_outside_box_center_cannot_reenter_planner_after_clustering():
    obj = node()
    # On a bend, a box midpoint can lie outside even when its points were inside.
    tracks = [TrackedObstacle(i, 15., y, 2., 2., 0., 0., False, False)
              for i, y in ((101, 0.), (102, 3.2))]
    def clustered(*args):
        obj.tracker.tracks = [_Track(t.track_id, history=[(10.05, 26.4, t.y_m)])
                              for t in tracks]
        return tracks
    obj.tracker.update = MagicMock(side_effect=clustered)
    run_scan(obj, cluster(15., 0.))
    assert [o.track_id for o in obj.planner.update.call_args.args[2]] == [101]
    assert [t.track_id for t in obj.tracker.tracks] == [101]
    assert obj.debug.publish.call_args.kwargs['lane_removed_tracks'] == 1


def test_geometry_update_invalidates_inflight_s12_output_but_not_other_sectors():
    for pose_x in (-10., 10., 50.):
        obj = node()
        original_update = obj.planner.update
        def update(*args, **kwargs):
            decision = original_update(*args, **kwargs)
            obj.on_lane_geometry(String(data=json.dumps(geometry(y=.1))))
            return decision
        obj.planner.update = MagicMock(side_effect=update)
        run_scan(obj, cluster(15., 0.), (pose_x, 0., 0.))
        assert obj.pub.publish.call_count == (0 if pose_x == 10. else 1)
        if pose_x == 10.:
            assert obj.debug.clear.call_args.args[0] == 'lane_geometry_changed'


def test_geometry_updates_keep_fixed_width_and_invalid_update_keeps_last_good_filter():
    obj = node()
    original = obj.sector_lane_filter
    obj.on_lane_geometry(String(data=json.dumps(geometry(y=.5, width=8.))))
    updated = obj.sector_lane_filter
    assert updated is not original
    assert updated.corridor.width == 3.2
    assert updated.corridor.contains([(20., 2.0), (20., 2.2)]).tolist() == [True, False]
    with patch.object(module.rospy, 'logwarn_throttle'):
        obj.on_lane_geometry(String(data='invalid json'))
        obj.on_lane_geometry(String(data=json.dumps(dict(geometry(), frame_id='lidar'))))
        obj.on_lane_geometry(String(data=json.dumps(dict(geometry(), lanes=[]))))
        obj.on_lane_geometry(String(data=json.dumps(dict(
            geometry(), lanes=[geometry()['lanes'][0]]))))
    assert obj.sector_lane_filter is updated
    obj.sector_lane_filter_enabled = False
    obj.on_lane_geometry(String(data=json.dumps(geometry(y=2.))))
    assert obj.sector_lane_filter is updated


def test_initial_project_geometry_does_not_wait_for_topic_and_csv_is_unchanged():
    project = dict(sectors=[dict(number=lane['sector'], racelines={
        '1': [dict(x=x, y=y) for x, y in lane['points']]})
        for lane in geometry()['lanes']])
    params = {'~raceline_config': '/tmp/chosen-raceline.yaml', '~raceline_width': 7.}
    with patch.object(module, 'load_project', return_value=project) as load, \
            patch.object(module, 'raceline_layout_settings', return_value=7.) as settings, \
            patch.object(module.rospy, 'get_param', side_effect=lambda key, default=None:
                         params.get(key, default)):
        assert module.load_sector_lane_filter('/tmp/route.csv') is None
        load.assert_not_called()
        loaded = module.load_sector_lane_filter('/tmp/route.json')
        settings.assert_called_once_with(project, '/tmp/chosen-raceline.yaml', 7.)
        assert loaded.active((10., 0.))
        assert loaded.corridor.width == 3.2
        assert loaded.corridor.contains([(20., 1.6), (20., 1.61)]).tolist() == [True, False]
