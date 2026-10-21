"""Route perception must transform a scan using its stamped capture pose."""
from collections import deque
import importlib.util
import math
from pathlib import Path
import threading
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
    return obj


def pose(stamp, x, y, yaw):
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = rospy.Time.from_sec(stamp)
    msg.pose.position.x, msg.pose.position.y = x, y
    msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return msg


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
    obj.input_timeout_s = .35
    obj.pending = None
    obj.speed, obj.signal = (3., 100.), None
    obj.tracker, obj.debug = MagicMock(), MagicMock()
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = 'lidar', rospy.Time.from_sec(10.05)
    with patch.object(module.time, 'monotonic', return_value=100.), \
            patch.object(module.rospy.Time, 'now', return_value=rospy.Time.from_sec(10.1)):
        obj.on_pose(pose(10., 0., 0., 0.))
        obj.process(msg, 100.)
    assert obj.pending == (msg, 100.)
    obj.tracker.update.assert_not_called()
    obj.tracker.reset.assert_not_called()
    obj.pending = None  # the worker takes the deferred input
    with patch.object(module.time, 'monotonic', return_value=100.4), \
            patch.object(module.rospy.Time, 'now', return_value=rospy.Time.from_sec(10.45)), \
            patch.object(module.rospy, 'logwarn_throttle'):
        obj.process(msg, 100.)
    assert obj.pending is None
    obj.tracker.reset.assert_called_once()
    obj.tracker.update.assert_not_called()
