#!/usr/bin/env python3
"""Cluster road-height LiDAR returns and track their map-frame motion."""
import json
import math
import threading
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from scipy.spatial import cKDTree
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from lane_corridor import LaneCorridor
from obstacle_motion import box_bounds, update_motion
from perception_debug import PerceptionDebug


def road_obstacle_mask(points, front_range=55., rear_range=55., side_range=12.,
                       half_width=.946, front_extent=3.845, rear_extent=.790):
    """Points use rear-axle coordinates; retain rear traffic, remove ego body."""
    road = ((points[:, 0] > -rear_range) & (points[:, 0] < front_range) &
            (np.abs(points[:, 1]) < side_range) &
            (points[:, 2] > .22) & (points[:, 2] < 2.6))
    ego = ((points[:, 0] >= -rear_extent-.15) &
           (points[:, 0] <= front_extent+.15) &
           (np.abs(points[:, 1]) <= half_width+.15))
    return road & ~ego


def clusters(xy, radius=0.65, minimum=3):
    """Euclidean connected components after a small metric voxel filter."""
    if not len(xy):
        return []
    cell = np.floor(xy / .25).astype(np.int32)
    _, unique = np.unique(cell, axis=0, return_index=True)
    xy = xy[np.sort(unique)]
    tree = cKDTree(xy)
    seen = np.zeros(len(xy), dtype=bool)
    groups = []
    for first in range(len(xy)):
        if seen[first]:
            continue
        queue = [first]
        seen[first] = True
        group = []
        while queue:
            i = queue.pop()
            group.append(i)
            for j in tree.query_ball_point(xy[i], radius):
                if not seen[j]:
                    seen[j] = True
                    queue.append(j)
        if len(group) >= minimum:
            groups.append(xy[group])
    return groups


def object_circles(group, pose=None):
    """Cover observed vehicle returns; split a long truck into bounded pieces."""
    size = np.ptp(group, axis=0)
    if np.max(size) > 24. or np.max(size) < .2:
        return []
    axis = int(np.argmax(size))
    count = max(1, int(math.ceil(size[axis]/5.)))
    labels = np.minimum(count-1, ((group[:, axis]-np.min(group[:, axis])) /
                                 max(size[axis], .01)*count).astype(int))
    result = []
    for label in range(count):
        part = group[labels == label]
        if not len(part):
            continue
        low, high = np.min(part, axis=0), np.max(part, axis=0)
        center = (low+high)*.5
        radius = max(.15, .5*math.hypot(*(high-low)))
        if radius <= 6.:
            if pose is None:
                result.append((center[0], center[1], radius))
            else:
                c, s = math.cos(pose[2]), math.sin(pose[2])
                world = np.column_stack((pose[0]+c*part[:, 0]-s*part[:, 1],
                                         pose[1]+s*part[:, 0]+c*part[:, 1]))
                bounds = (*world.min(axis=0), *world.max(axis=0))
                result.append((pose[0]+c*center[0]-s*center[1],
                               pose[1]+s*center[0]+c*center[1], radius, bounds))
    return result


class LidarObstacles:
    def __init__(self):
        self.lidar_x = float(rospy.get_param('~lidar_to_rear_x_m', 1.4))
        self.lidar_y = float(rospy.get_param('~lidar_to_rear_y_m', 0.))
        self.lidar_z = float(rospy.get_param('~lidar_to_rear_z_m', 1.23))
        self.max_pose_age = float(rospy.get_param('~max_pose_age_s', .15))
        self.dynamic_speed_threshold_mps = float(rospy.get_param('~dynamic_speed_threshold_mps', .8))
        self.front_range = float(rospy.get_param('~front_range_m', 55.))
        self.rear_range = float(rospy.get_param('~rear_range_m', 55.))
        self.side_range = float(rospy.get_param('~side_range_m', 12.))
        self.half_width = float(rospy.get_param('~vehicle_half_width_m', .946))
        self.front_extent = float(rospy.get_param('~front_extent_m', 3.845))
        self.rear_extent = float(rospy.get_param('~rear_extent_m', .790))
        if min(self.front_range, self.rear_range, self.side_range,
               self.half_width, self.front_extent, self.rear_extent) <= 0.:
            raise ValueError('invalid obstacle ROI geometry')
        self.poses = deque(maxlen=30)
        self.pending = None
        self.lock = threading.Lock()
        self.tracks = {}
        self.next_id = 1
        self.lane_corridor = None
        self.pub = rospy.Publisher('/waypoint_debug/obstacles', String, queue_size=1)
        self.debug = PerceptionDebug('/leo/waypoint_debug', 'ground_status')
        rospy.Subscriber('/waypoint_debug/lane_geometry', String,
                         self.on_lane_geometry, queue_size=1, buff_size=2**22)
        rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'),
                         PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(rospy.get_param('~cloud_topic', '/lidar3D'),
                         PointCloud2, self.on_cloud, queue_size=1, buff_size=2**24)
        rospy.Timer(rospy.Duration(.025), self.process_pending)

    def on_lane_geometry(self, message):
        try:
            corridor = LaneCorridor(json.loads(message.data))
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            rospy.logwarn_throttle(5., 'Invalid lane geometry: %s', error)
            with self.lock:
                self.lane_corridor = None
                self.tracks.clear()
            return
        with self.lock:
            self.lane_corridor = corridor
            self.tracks.clear()
            # Keep all three lane strips in view when a wider layout is used.
            self.side_range = max(self.side_range, 3.*corridor.width)
        rospy.set_param('~raceline_width', corridor.width)
        rospy.loginfo('Obstacle lane filter: width %.2f m (+/- %.2f m)',
                      corridor.width, corridor.width*.5)

    def on_pose(self, msg):
        if msg.header.frame_id != 'map':
            return
        p, q = msg.pose.position, msg.pose.orientation
        values = (p.x, p.y, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            return
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        stamp = msg.header.stamp.to_sec()
        with self.lock:
            if self.poses and stamp < self.poses[-1][0]:
                self.poses.clear()
            if not self.poses or stamp > self.poses[-1][0]:
                self.poses.append((stamp, (p.x, p.y, yaw)))

    def pose_at(self, stamp):
        with self.lock:
            samples = list(self.poses)
        for i in range(len(samples)-1):
            before, after = samples[i], samples[i+1]
            if before[0] <= stamp <= after[0]:
                dt = after[0]-before[0]
                if dt > self.max_pose_age:
                    return None
                t = (stamp-before[0])/dt
                a, b = before[1], after[1]
                angle = math.atan2(math.sin(b[2]-a[2]), math.cos(b[2]-a[2]))
                return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*angle)
        if samples and abs(samples[-1][0]-stamp) < .02:
            return samples[-1][1]
        return None

    def _update_tracks(self, detections, stamp):
        self.tracks = {key: track for key, track in self.tracks.items()
                       if stamp - track['stamp'] < .8}
        result = []
        used = set()
        for detection in detections:
            x, y, radius = detection[:3]
            bounds = detection[3] if len(detection) > 3 else box_bounds(x, y, 2.*radius, 2.*radius, 0.)
            matches = [(math.hypot(x-t['x']-t['vx']*(stamp-t['stamp']),
                                  y-t['y']-t['vy']*(stamp-t['stamp'])), key, t)
                       for key, t in self.tracks.items() if key not in used]
            matches.sort()
            # Allow a fast rear car's initial displacement before velocity
            # converges; subsequent association uses its predicted position.
            if matches and matches[0][0] < max(2.5, 35.*(stamp-matches[0][2]['stamp'])):
                _, key, previous = matches[0]
                dt = stamp - previous['stamp']
                if dt <= 0:
                    continue
                hits = min(20, previous['hits']+1)
            else:
                previous = {}
                key = self.next_id
                self.next_id += 1
                hits = 1
            motion = update_motion(previous, stamp, bounds,
                                   getattr(self, 'dynamic_speed_threshold_mps', .8))
            vx, vy = motion['vx'], motion['vy']
            track = dict(x=x, y=y, radius=radius, **motion,
                         hits=hits, stamp=stamp)
            self.tracks[key] = track
            used.add(key)
            result.append(dict(id=key, x=x, y=y, radius=radius, vx=vx, vy=vy,
                               velocity_reliable=motion['velocity_reliable'],
                               motion_bounds=list(bounds), predicted=False,
                               kind=motion['motion_kind']))
        # Preserve a briefly occluded rear/side car instead of treating its
        # lane as clear immediately. Do not extend a stale measurement forever.
        for key, track in self.tracks.items():
            dt = stamp-track['stamp']
            if key not in used and 0. < dt <= .4:
                reliable = track.get('velocity_reliable', False)
                result.append(dict(id=key, x=track['x']+(track['vx']*dt if reliable else 0.),
                                   y=track['y']+(track['vy']*dt if reliable else 0.),
                                   radius=track['radius']+.5*dt,
                                   vx=track['vx'], vy=track['vy'], kind='unknown',
                                   velocity_reliable=reliable, predicted=True))
        return result

    def on_cloud(self, msg):
        with self.lock:
            self.pending = msg

    def process_pending(self, _event):
        debug = getattr(self, 'debug', None)
        with self.lock:
            msg = self.pending
            corridor = self.lane_corridor
        if msg is None:
            if debug is not None:
                debug.clear_if_stale()
            return
        if corridor is None:
            if debug is not None:
                debug.clear('waiting_lane_geometry', msg.header)
            rospy.logwarn_throttle(5., 'Waiting for raceline lane geometry')
            return
        stamp = msg.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        if msg.header.frame_id != 'lidar' or not -.05 <= age <= .35:
            if debug is not None:
                debug.clear('wrong_cloud_frame' if msg.header.frame_id != 'lidar'
                            else 'cloud_time_mismatch', msg.header)
            with self.lock:
                if self.pending is msg:
                    self.pending = None
            return
        pose = self.pose_at(stamp)
        if pose is None:
            if debug is not None:
                debug.clear('waiting_pose', msg.header)
            return
        with self.lock:
            if self.pending is msg:
                self.pending = None
        points = np.asarray(list(point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=float)
        points = points.reshape((-1, 3))
        points[:, 0] += self.lidar_x
        points[:, 1] += self.lidar_y
        points[:, 2] += self.lidar_z
        mask = road_obstacle_mask(points, self.front_range, self.rear_range,
                                  self.side_range, self.half_width,
                                  self.front_extent, self.rear_extent)
        xy = points[mask, :2]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        map_xy = np.column_stack((pose[0]+c*xy[:, 0]-s*xy[:, 1],
                                  pose[1]+s*xy[:, 0]+c*xy[:, 1]))
        # Drop roadside returns before clustering so a fence cannot enlarge
        # or merge with a car that is actually inside a mapped lane.
        inside = corridor.contains(map_xy)
        capture_debug = debug is not None and debug.active()
        debug_selected = points[mask][inside] if capture_debug else None
        debug_counts = dict(candidate_points=int(mask.sum()),
                           lane_removed_points=int((~inside).sum()),
                           other_removed_points=int((~mask).sum()))
        xy = xy[inside]
        if len(xy) > 10000:
            xy = xy[::int(math.ceil(len(xy)/10000.))]
        found = []
        for group in clusters(xy):
            found.extend(object_circles(group, pose))
        mask = corridor.contains([o[:2] for o in found])
        found = [o for o, inside in zip(found, mask) if inside]
        with self.lock:
            if self.lane_corridor is not corridor:
                return
            tracked = self._update_tracks(found, stamp)
            tracked = corridor.filter_obstacles(tracked)
            # Expired predictions and vehicles leaving the lane are discarded
            # from tracking as well as from the published obstacle stream.
            keep = {o['id'] for o in tracked}
            self.tracks = {key: value for key, value in self.tracks.items() if key in keep}
        if debug is not None:
            if capture_debug:
                debug.publish(points, debug_selected, msg.header, pose,
                              mode='local_filtered',
                              debug_point_stage='lane_filtered_input',
                              filter_summary='고정 높이·자차·거리 → 저장 차선 → 군집·추적 (IMU 노면 평면 없음)',
                              imu_corrected=False, obstacle_min_height_m=.22,
                              obstacle_max_height_m=2.6, detected_obstacles=len(found),
                              tracked_obstacles=len(tracked), **debug_counts)
        self.pub.publish(String(data=json.dumps(dict(
            stamp=stamp, frame_id='map', obstacles=tracked))))


if __name__ == '__main__':
    rospy.init_node('lidar_obstacles')
    LidarObstacles()
    rospy.spin()
