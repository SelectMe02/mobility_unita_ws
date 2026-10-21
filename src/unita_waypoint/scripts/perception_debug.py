"""Read-only point-cloud diagnostics shared by the active perception nodes."""
import json
import math
import copy
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, String

from map_debug_geometry import obstacle_z_bounds


def xyz_cloud(header, points):
    """Pack standard XYZ32 without a Python loop per displayed return."""
    xyz = np.asarray(points, dtype='<f4').reshape(-1, 3)
    return PointCloud2(header=header, height=1, width=len(xyz),
                       fields=[PointField(name=name, offset=4*i,
                                          datatype=PointField.FLOAT32, count=1)
                               for i, name in enumerate(('x', 'y', 'z'))],
                       is_bigendian=False, point_step=12, row_step=12*len(xyz),
                       data=xyz.tobytes(), is_dense=False)


def retained_mask(points, selected):
    """Match an unchanged subset, preserving multiplicity after decimation.

    This only reconstructs debug provenance from the algorithm's actual
    output. It neither repeats nor approximates a perception filter.
    """
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    selected = np.ascontiguousarray(selected, dtype=np.float64).reshape(-1, 3)
    result = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(selected):
        return result
    row = np.dtype((np.void, points.dtype.itemsize * 3))
    keys, inverse, counts = np.unique(points.view(row).ravel(), return_inverse=True,
                                     return_counts=True)
    selected_keys = selected.view(row).ravel()
    positions = np.searchsorted(keys, selected_keys)
    valid = positions < len(keys)
    valid[valid] &= keys[positions[valid]] == selected_keys[valid]
    wanted = np.bincount(positions[valid], minlength=len(keys))
    order = np.argsort(inverse, kind='stable')
    rank = np.arange(len(points)) - np.repeat(np.cumsum(counts)-counts, counts)
    result[order] = rank < wanted[inverse[order]]
    return result


def tracked_obstacles_map(tracked, pose, lidar_to_rear_x):
    """Present the tracker's measured LiDAR-frame boxes in the map frame."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    result = []
    for track in tracked:
        x, y = track.x_m+lidar_to_rear_x, track.y_m
        length, width = max(.2, track.size_x_m), max(.2, track.size_y_m)
        result.append(dict(id=track.track_id, x=pose[0]+c*x-s*y,
                           y=pose[1]+s*x+c*y, length=length, width=width,
                           radius=.5*math.hypot(length, width), yaw=pose[2],
                           vx=c*track.vx_mps-s*track.vy_mps,
                           vy=s*track.vx_mps+c*track.vy_mps,
                           velocity_reliable=bool(track.velocity_known),
                           kind=('dynamic' if track.dynamic else
                                 'static' if track.velocity_known else 'unknown')))
    return result


class PerceptionDebug:
    def __init__(self, prefix, status_name='status'):
        self.used_pub = rospy.Publisher(prefix+'/obstacle_points', PointCloud2, queue_size=1)
        self.discarded_pub = rospy.Publisher(prefix+'/discarded_points', PointCloud2, queue_size=1)
        self.status_pub = rospy.Publisher(prefix+'/'+status_name, String, queue_size=1)
        # Zero means every valid point. A positive optional limit is reported
        # explicitly in debug_clouds so decimation is never hidden.
        self.max_points = max(0, int(rospy.get_param('~debug_max_points', 0)))
        self.last_valid_wall = 0.
        self.next_clear_wall = 0.
        self._condition = threading.Condition()
        self._pending = None
        self._inflight = None
        self._generation = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name='perception_debug:'+prefix,
                                        daemon=True)
        rospy.on_shutdown(self.close)
        self._thread.start()

    def active(self):
        return any(pub.get_num_connections() for pub in
                   (self.used_pub, self.discarded_pub, self.status_pub))

    def clear(self, mode, header=None):
        now = time.monotonic()
        stamp = rospy.Time.now() if header is None else header.stamp
        with self._condition:
            if self._closed:
                return
            if now < self.next_clear_wall:
                # Never let an invalidated scan overwrite the previous clear.
                # Keep an already queued clear until the worker consumes it.
                if ((self._pending is not None and self._pending[0] == 'cloud')
                        or (self._pending is None and self._inflight == 'cloud')):
                    self._generation += 1
                    self._pending = None
                return
            self.next_clear_wall = now+.2
            self._generation += 1
            self._pending = ('clear', self._generation, stamp, mode)
            self._condition.notify()

    def clear_if_stale(self, timeout=.5):
        if time.monotonic()-self.last_valid_wall > timeout:
            self.clear('waiting_cloud')

    def publish(self, points, selected, header, pose, offset=(0., 0., 0.), **status):
        """Submit a bounded snapshot without doing visualisation on control.

        Arrays are copied so subsequent caller mutations cannot corrupt the
        worker's scan. Only the latest pending scan is retained; new scans or
        invalid-input clears also invalidate any result already being built.
        """
        self.last_valid_wall = time.monotonic()
        if not self.active():
            return
        try:
            points = np.array(points, dtype=float, copy=True).reshape(-1, 3)
            selected = np.array(selected, dtype=float, copy=True).reshape(-1, 3)
            points.setflags(write=False)
            selected.setflags(write=False)
            stamp = rospy.Time(header.stamp.secs, header.stamp.nsecs)
            snapshot = (points, selected, stamp, tuple(pose), tuple(offset),
                        copy.deepcopy(status), self.last_valid_wall)
            with self._condition:
                if self._closed:
                    return
                self._generation += 1
                self._pending = ('cloud', self._generation, snapshot)
                self._condition.notify()
        except Exception as error:
            rospy.logwarn_throttle(2., 'Perception debug output failed: %s', error)

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._generation += 1
            self._condition.notify_all()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                job, self._pending = self._pending, None
                self._inflight = job[0]
            try:
                if job[0] == 'clear':
                    _, generation, stamp, mode = job
                    empty = xyz_cloud(Header(stamp=stamp, frame_id='map'), [])
                    status = String(data=json.dumps(dict(
                        stamp=stamp.to_sec(), frame_id='map', mode=mode, debug_valid=False)))
                    self._deliver(generation, [(self.used_pub, empty),
                                              (self.discarded_pub, empty),
                                              (self.status_pub, status)])
                else:
                    self._build_clouds(job[1], *job[2])
            except Exception as error:
                # Diagnostics must neither terminate their worker nor escape
                # into a planning callback. A future fresh scan can recover.
                try:
                    rospy.logwarn_throttle(2., 'Perception debug worker failed: %s', error)
                except Exception:
                    pass  # ROS may already be shutting down.
            finally:
                with self._condition:
                    self._inflight = None

    def _deliver(self, generation, messages, submitted=None):
        # Building is expensive and runs outside this lock. Publishing the
        # completed byte buffers is brief; the lock orders scans and clears.
        with self._condition:
            if (self._closed or generation != self._generation
                    or (submitted is not None and time.monotonic()-submitted > .5)):
                return
            for publisher, message in messages:
                if publisher.get_num_connections():
                    publisher.publish(message)

    def _build_clouds(self, generation, points, selected, stamp, pose, offset, status, submitted):
        if not self.active():
            return
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        selected = selected[np.isfinite(selected).all(axis=1)]
        discarded = points[~retained_mask(points, selected)]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        if status.get('obstacles'):
            # Bounds use the full selected scan, before optional display
            # decimation, in exactly the same frame as the published cloud.
            xyz = selected+offset
            mapped = np.column_stack((pose[0]+c*xyz[:, 0]-s*xyz[:, 1],
                                      pose[1]+s*xyz[:, 0]+c*xyz[:, 1], xyz[:, 2]))
            for obstacle in status['obstacles']:
                bounds = obstacle_z_bounds(mapped, obstacle['x'], obstacle['y'],
                                           obstacle['length'], obstacle['width'],
                                           obstacle['yaw'])
                if bounds is not None:
                    obstacle['z_min'], obstacle['z_max'] = bounds
        display = {}
        messages = []
        for name, cloud, pub in (('used', selected, self.used_pub),
                                 ('discarded', discarded, self.discarded_pub)):
            stride = (max(1, int(math.ceil(len(cloud)/self.max_points)))
                      if self.max_points else 1)
            display[name] = dict(total=len(cloud), displayed=int(math.ceil(len(cloud)/stride)),
                                 stride=stride)
            if not pub.get_num_connections():
                continue
            xyz = cloud[::stride]+offset
            mapped = np.column_stack((pose[0]+c*xyz[:, 0]-s*xyz[:, 1],
                                      pose[1]+s*xyz[:, 0]+c*xyz[:, 1], xyz[:, 2]))
            messages.append((pub, xyz_cloud(
                Header(stamp=stamp, frame_id='map'), mapped)))
        status.update(stamp=stamp.to_sec(), cloud_stamp=stamp.to_sec(),
                      frame_id='map', debug_valid=True, input_points=len(finite),
                      invalid_points=int((~finite).sum()), used_points=len(selected),
                      discarded_points=len(discarded), debug_clouds=display)
        messages.append((self.status_pub, String(data=json.dumps(status))))
        self._deliver(generation, messages, submitted)
