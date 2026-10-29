"""Route perception must transform a scan using its stamped capture pose."""
from collections import deque
import importlib.util
import json
import math
from pathlib import Path
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2


spec = importlib.util.spec_from_file_location(
    'route_avoidance_node_pose_test',
    Path(__file__).resolve().parents[1] / 'scripts' / 'route_avoidance_node.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def node():
    obj = module.RouteAvoidanceNode.__new__(module.RouteAvoidanceNode)
    obj.lock = threading.Lock()
    obj.poses = deque(maxlen=100)
    obj.pose = None
    obj.max_pose_gap_s = .15
    obj.input_timeout_s = .35
    obj.pending = deque(maxlen=8)
    obj.last_selected_stamp = None
    obj.input_epoch = obj.processed_epoch = 0
    obj.last_worker_ros_time = None
    obj.closed = False
    obj.tracker, obj.planner, obj.debug, obj.pub = (MagicMock() for _ in range(4))
    return obj


def pose(stamp, x, y, yaw):
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = rospy.Time.from_sec(stamp)
    msg.pose.position.x, msg.pose.position.y = x, y
    msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return msg


def cloud(stamp):
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = 'lidar', rospy.Time.from_sec(stamp)
    return msg


@contextmanager
def at(stamp):
    with patch.object(module.time, 'monotonic', return_value=stamp), \
            patch.object(module.rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
        yield


def test_capture_pose_interpolates_translation_and_wrapped_yaw():
    obj = node()
    obj.on_pose(pose(10., 0., 1., math.radians(179)))
    obj.on_pose(pose(10.1, 2., 3., math.radians(-179)))
    x, y, yaw = obj.pose_at(10.05)
    assert math.isclose(x, 1., abs_tol=1e-6)
    assert math.isclose(y, 2., abs_tol=1e-6)
    assert math.isclose(abs(yaw), math.pi, abs_tol=1e-6)
    assert obj.pose_at(10.) == obj.poses[0][1]


def test_capture_pose_requires_bracketing_samples_and_rejects_pose_gaps():
    obj = node()
    obj.on_pose(pose(10., 0., 0., 0.))
    assert obj.pose_at(10.01) is None
    assert obj.pose_at(9.99) is None
    obj.on_pose(pose(10.3, 3., 0., 0.))
    assert obj.pose_at(10.1) is None
    obj.on_pose(pose(2., 20., 0., 0.))
    assert len(obj.poses) == 1 and obj.pose_at(10.) is None


def test_cloud_waits_for_capture_pose_then_expires_without_tracking():
    obj = node()
    msg = cloud(10.05)
    with at(10.07):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.on_cloud(msg)
    assert obj.take_ready_scan(10.1, 10.1) is None
    assert list(obj.pending) == [(msg, 10.07)]
    obj.tracker.update.assert_not_called()
    obj.tracker.reset.assert_not_called()
    assert obj.take_ready_scan(10.41, 10.41) is None
    assert not obj.pending
    obj.tracker.update.assert_not_called()


def test_new_cloud_does_not_overwrite_scan_waiting_for_following_pose():
    obj = node()
    first, newer = cloud(10.05), cloud(10.15)
    with at(10.06):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.on_cloud(first)
    assert obj.take_ready_scan(10.1, 10.1) is None
    with at(10.16):
        obj.on_cloud(newer)
    with at(10.17):
        obj.on_pose(pose(10.1, 1., 0., 0.))
    selected = obj.take_ready_scan(10.18, 10.18)
    assert selected[0] is first
    assert math.isclose(selected[2][0], .5, abs_tol=1e-6)
    assert [m for m, _ in obj.pending] == [newer]
    with at(10.25):
        obj.on_pose(pose(10.2, 2., 0., 0.))
    assert obj.take_ready_scan(10.26, 10.26)[0] is newer
    assert not obj.pending


def test_selects_latest_ready_scan_and_keeps_newer_unready_scan():
    obj = node()
    messages = [cloud(t) for t in (10.02, 10.05, 10.15)]
    with at(10.16):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.on_pose(pose(10.1, 1., 0., 0.))
        for msg in messages:
            obj.on_cloud(msg)
    assert obj.take_ready_scan(10.17, 10.17)[0] is messages[1]
    assert [m for m, _ in obj.pending] == [messages[2]]
    assert obj.take_ready_scan(10.18, 10.18) is None


def test_duplicates_consumed_and_out_of_order_scans_are_not_reprocessed():
    obj = node()
    first, newer = cloud(10.05), cloud(10.08)
    with at(10.1):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.on_pose(pose(10.1, 1., 0., 0.))
        for msg in (newer, first, newer):
            obj.on_cloud(msg)
    assert len(obj.pending) == 2
    assert obj.take_ready_scan(10.12, 10.12)[0] is newer
    with at(10.13):
        obj.on_cloud(newer)
        obj.on_cloud(first)
    assert not obj.pending and obj.take_ready_scan(10.14, 10.14) is None


def test_waiting_queue_is_bounded_and_capture_timeout_is_not_extended():
    obj = node()
    with at(10.2):
        for i in range(20):
            obj.on_cloud(cloud(10. + i * .01))
    assert len(obj.pending) == 8
    assert obj.take_ready_scan(10.6, 10.6) is None
    assert not obj.pending


def test_pose_time_jump_invalidates_pending_and_inflight_epoch():
    obj = node()
    with at(10.1):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.on_pose(pose(10.1, 1., 0., 0.))
        obj.on_cloud(cloud(10.05))
    selected = obj.take_ready_scan(10.11, 10.11)
    with at(10.12):
        obj.on_cloud(cloud(10.12))
        obj.on_pose(pose(2., 20., 0., 0.))
        obj.process(*selected)
    assert not obj.pending and len(obj.poses) == 1
    obj.pub.publish.assert_not_called()
    obj.tracker.update.assert_not_called()
    obj.take_ready_scan(10.13, 10.13)
    obj.tracker.reset.assert_called_once()
    obj.planner.reset.assert_called_once()


def test_ros_clock_rewind_drops_previous_epoch_and_accepts_new_inputs():
    obj = node()
    assert obj.take_ready_scan(100., 10.) is None
    with at(100.01):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.on_cloud(cloud(10.02))
    assert obj.take_ready_scan(100.02, 2.) is None
    assert not obj.pending and not obj.poses
    with at(100.03):
        obj.on_pose(pose(2., 0., 0., 0.))
        obj.on_pose(pose(2.1, 1., 0., 0.))
        msg = cloud(2.05)
        obj.on_cloud(msg)
    assert obj.take_ready_scan(100.04, 2.1)[0] is msg


def test_shutdown_clears_pending_and_rejects_further_input_and_output():
    obj = node()
    with at(10.1):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.on_pose(pose(10.1, 1., 0., 0.))
        obj.on_cloud(cloud(10.05))
    selected = obj.take_ready_scan(10.11, 10.11)
    obj.close()
    with at(10.12):
        obj.on_cloud(cloud(10.12))
        obj.on_pose(pose(10.2, 2., 0., 0.))
        obj.process(*selected)
    assert not obj.pending and not obj.poses
    assert obj.take_ready_scan(10.13, 10.13) is None
    obj.pub.publish.assert_not_called()


def processing_node():
    obj = node()
    obj.pose = ((0., 0., 0.), 10.1)
    obj.speed, obj.signal = (3., 10.1), None
    obj.lidar_to_rear_x_m, obj.debug_lidar_z = 1.4, 1.23
    obj.road_points = module.RoadPointConfig()
    obj.drop_road_boundaries = lambda points, pose: points
    obj.boundary_debug_counts = {}
    obj.tracker.update.return_value = []
    obj.planner.update.return_value = SimpleNamespace(
        speed_cap_mps=math.inf, state='FOLLOW', reason='clear',
        traffic_speed_cap_mps=math.inf,
        offset_now_m=0., path=[(0., 0.), (1., 0.)], obstacles=[])
    return obj


def test_processing_expiry_or_clock_rewind_does_not_publish_old_plan():
    for after_ros in (10.5, 2.):
        obj = processing_node()
        clock = [10.12]
        result = obj.planner.update.return_value
        def finish(*args, **kwargs):
            clock[0] = after_ros
            return result
        obj.planner.update.side_effect = finish
        with patch.object(module.time, 'monotonic', return_value=10.12), \
                patch.object(module.rospy.Time, 'now', side_effect=lambda: rospy.Time.from_sec(clock[0])), \
                patch.object(module.point_cloud2, 'read_points', return_value=[]):
            obj.process(cloud(10.05), 10.06, (0., 0., 0.), 0)
        obj.pub.publish.assert_not_called()
        obj.debug.clear.assert_called_with('processing_expired', obj.debug.clear.call_args[0][1])


def test_pose_epoch_change_during_processing_prevents_old_plan_publish():
    obj = processing_node()
    obj.poses.append((10.1, (0., 0., 0.)))
    result = obj.planner.update.return_value
    def reset_pose(*args, **kwargs):
        obj.on_pose(pose(2., 10., 0., 0.))
        return result
    obj.planner.update.side_effect = reset_pose
    with at(10.12), patch.object(module.point_cloud2, 'read_points', return_value=[]):
        obj.process(cloud(10.05), 10.06, (0., 0., 0.), 0)
    assert obj.input_epoch == 1
    obj.pub.publish.assert_not_called()


def test_ready_processing_publishes_capture_stamp_once_without_retimestamp():
    obj = processing_node()
    msg = cloud(10.05)
    with at(10.12), patch.object(module.point_cloud2, 'read_points', return_value=[]):
        obj.process(msg, 10.06, (0., 0., 0.), 0)
    obj.pub.publish.assert_called_once()
    output = json.loads(obj.pub.publish.call_args[0][0].data)
    assert output['stamp'] == msg.header.stamp.to_sec()
    assert output['traffic_speed_cap_kmh'] is None


def test_static_wait_output_keeps_independent_traffic_speed_cap():
    obj = processing_node()
    obj.planner.update.return_value.state = 'WAIT_SOLID_LINE'
    obj.planner.update.return_value.speed_cap_mps = 0.
    obj.planner.update.return_value.traffic_speed_cap_mps = 1.25
    with at(10.12), patch.object(module.point_cloud2, 'read_points', return_value=[]):
        obj.process(cloud(10.05), 10.06, (0., 0., 0.), 0)
    output = json.loads(obj.pub.publish.call_args[0][0].data)
    assert output['speed_cap_kmh'] == 0.
    assert output['traffic_speed_cap_kmh'] == 4.5
