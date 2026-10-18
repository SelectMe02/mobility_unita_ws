#!/usr/bin/env python3
"""Cluster road-height LiDAR returns and track their map-frame motion."""
import json
import math
import threading
import time
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from scipy.spatial import cKDTree
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, Imu
from std_msgs.msg import String, Header

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from lane_corridor import LaneCorridor
from avoidance_runtime import RuntimeSettings
from ground_segmentation import (rotation_rpy, normalized_quaternion, quaternion_at,
                                 level_rotation, segment_ground)


def road_obstacle_mask(points, front_range=55., rear_range=55., side_range=12.,
                       half_width=.946, front_extent=3.845, rear_extent=.790,
                       min_height=.22, max_height=2.6):
    """Points use rear-axle coordinates; retain rear traffic, remove ego body."""
    road = ((points[:, 0] > -rear_range) & (points[:, 0] < front_range) &
            (np.abs(points[:, 1]) < side_range) &
            (points[:, 2] > min_height) & (points[:, 2] < max_height))
    ego = ((points[:, 0] >= -rear_extent-.15) &
           (points[:, 0] <= front_extent+.15) &
           (np.abs(points[:, 1]) <= half_width+.15))
    return road & ~ego


def road_roi_mask(points, front_range, rear_range, side_range, half_width, front_extent, rear_extent):
    """Include road seeds below the old height gate, excluding the ego body."""
    x, y, z = points.T
    ego = ((x >= -rear_extent-.15) & (x <= front_extent+.15) &
           (np.abs(y) <= half_width+.15))
    return ((x > -rear_range) & (x < front_range) & (np.abs(y) < side_range) &
            (z > -20.) & (z < 20.) & ~ego & np.isfinite(points).all(axis=1))


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


def object_circles(group, minimum_size=.2, maximum_size=24.):
    """Cover observed vehicle returns; split a long truck into bounded pieces."""
    size = np.ptp(group, axis=0)
    if np.max(size) > maximum_size or np.max(size) < minimum_size:
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
            result.append((center[0], center[1], radius))
    return result


def object_box(group, direction, minimum_size=.2, maximum_size=24.):
    """Rectangle around map-frame returns, aligned with the lane direction.

    LiDAR usually sees one or two faces of a car. Aligning with the lane keeps
    the box stable and gives the true front/rear face and side extent along
    the road. Returns (x, y, radius, length, width, yaw) or None.
    """
    tangent = np.asarray(direction, dtype=float)
    normal = np.array([-tangent[1], tangent[0]])
    longitudinal, lateral = group @ tangent, group @ normal
    low = np.array([longitudinal.min(), lateral.min()])
    high = np.array([longitudinal.max(), lateral.max()])
    size = high-low
    if np.max(size) > maximum_size or np.max(size) < minimum_size:
        return None
    # A single visible face has no depth; keep a thin but finite box.
    length, width = max(.2, float(size[0])), max(.2, float(size[1]))
    middle = (low+high)*.5
    center = tangent*middle[0]+normal*middle[1]
    radius = max(.15, .5*math.hypot(length, width))
    return (float(center[0]), float(center[1]), radius, length, width,
            math.atan2(tangent[1], tangent[0]))


def road_surface_evidence(z, reference_z, ground_confirmed):
    """Flat returns at the calibrated road height, not an elevated object.

    Attach evidence to detections instead of globally removing them: only
    the highway planner opts out of these road-ring false detections.
    """
    z = np.asarray(z, dtype=float)
    return bool(ground_confirmed and len(z) >= 3 and np.isfinite(z).all()
                and np.min(z) >= reference_z-.08
                and np.max(z) <= reference_z+.05 and np.ptp(z) <= .06)


def road_surface_in_box(world_xy, z, box, reference_z, ground_confirmed):
    """Use ALL vertical returns in the footprint, including repeated XY.

    Clustering keeps one XY sample per voxel. Its nearest point alone must
    not certify road when a vehicle face has higher returns at the same XY.
    """
    dx, dy = world_xy[:, 0]-box[0], world_xy[:, 1]-box[1]
    c, s = math.cos(box[5]), math.sin(box[5])
    inside = ((np.abs(dx*c+dy*s) <= box[3]/2.+.003)
              & (np.abs(-dx*s+dy*c) <= box[4]/2.+.003))
    return road_surface_evidence(z[inside], reference_z, ground_confirmed)


class LidarObstacles:
    def __init__(self):
        self.lidar_x = float(rospy.get_param('~lidar_to_rear_x_m', 1.4))
        self.lidar_y = float(rospy.get_param('~lidar_to_rear_y_m', 0.))
        self.lidar_z = float(rospy.get_param('~lidar_to_rear_z_m', 1.23))
        self.max_pose_age = float(rospy.get_param('~max_pose_age_s', .15))
        self.front_range = float(rospy.get_param('~front_range_m', 55.))
        self.rear_range = float(rospy.get_param('~rear_range_m', 55.))
        self.side_range = float(rospy.get_param('~side_range_m', 12.))
        self.dynamic_speed_threshold_mps = float(rospy.get_param('~dynamic_speed_threshold_mps', .8))
        def mount(prefix):
            angles = [float(rospy.get_param('~'+prefix+'_'+axis+'_deg', 0.))
                      for axis in ('roll', 'pitch', 'yaw')]
            if not all(math.isfinite(angle) for angle in angles):
                raise ValueError('non-finite sensor mounting rotation: '+prefix)
            return rotation_rpy(*(math.radians(angle) for angle in angles))
        self.body_from_lidar = mount('lidar_mount')
        self.body_from_imu = mount('imu_mount')
        self.imu_frame_id = rospy.get_param('~imu_frame_id', 'imu')
        self.imus = deque(maxlen=300)
        self.imu_rejection = None
        self.last_imu_frame = None
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
        self.runtime_settings = RuntimeSettings('lidar_obstacles', self)
        self.pub = rospy.Publisher('/waypoint_debug/obstacles', String, queue_size=1)
        self.ground_status_pub = rospy.Publisher('/waypoint_debug/ground_status', String, queue_size=1)
        self.ground_points_pub = rospy.Publisher('/waypoint_debug/ground_points', PointCloud2, queue_size=1)
        self.obstacle_points_pub = rospy.Publisher('/waypoint_debug/obstacle_points', PointCloud2, queue_size=1)
        rospy.Subscriber('/waypoint_debug/lane_geometry', String,
                         self.on_lane_geometry, queue_size=1, buff_size=2**22)
        rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'),
                         PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(rospy.get_param('~cloud_topic', '/lidar3D'),
                         PointCloud2, self.on_cloud, queue_size=1, buff_size=2**24)
        # Always the permitted raw IMU; yaw-only synthesized status is insufficient.
        rospy.Subscriber(rospy.get_param('~ground_imu_topic', '/imu'),
                         Imu, self.on_imu, queue_size=100)
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
            # Legacy circles become the square enclosing the circle.
            length, width, yaw = (detection[3:6] if len(detection) >= 6
                                  else (2.*radius, 2.*radius, 0.))
            road_surface = bool(detection[6]) if len(detection) >= 7 else False
            matches = [(math.hypot(x-t['x']-t['vx']*(stamp-t['stamp']),
                                  y-t['y']-t['vy']*(stamp-t['stamp'])), key, t)
                       for key, t in self.tracks.items() if key not in used
                       and max(radius, t['radius']) <= 3.*max(.15, min(radius, t['radius']))]
            matches.sort()
            # Allow a fast rear car's initial displacement before velocity
            # converges; subsequent association uses its predicted position.
            if matches and matches[0][0] < max(2.5, 35.*(stamp-matches[0][2]['stamp'])):
                _, key, previous = matches[0]
                dt = stamp - previous['stamp']
                if dt <= 0:
                    continue
                vx = .65*previous['vx'] + .35*(x-previous['x'])/dt
                vy = .65*previous['vy'] + .35*(y-previous['y'])/dt
                hits = min(20, previous['hits']+1)
                first_stamp = previous.get('first_stamp', previous['stamp']-(previous['hits']-1)*dt)
                velocities = (previous.get('velocities', [])+[((x-previous['x'])/dt, (y-previous['y'])/dt)])[-3:]
            else:
                key = self.next_id
                self.next_id += 1
                vx = vy = 0.
                hits = 1
                first_stamp, velocities = stamp, []
            observed_s = stamp-first_stamp
            confirmed = hits >= 3 and observed_s+1e-9 >= getattr(self, 'track_confirm_time_s', .2)
            velocity_reliable = (confirmed and len(velocities) >= 2
                                 and np.max(np.linalg.norm(np.asarray(velocities)-np.median(velocities, axis=0), axis=1)) <= 3.)
            track = dict(x=x, y=y, radius=radius, vx=vx, vy=vy,
                         length=length, width=width, yaw=yaw,
                         road_surface=road_surface,
                         hits=hits, stamp=stamp, first_stamp=first_stamp, velocities=velocities,
                         confirmed=confirmed, velocity_reliable=bool(velocity_reliable))
            self.tracks[key] = track
            used.add(key)
            result.append(dict(id=key, x=x, y=y, radius=radius, vx=vx, vy=vy,
                               length=length, width=width, yaw=yaw,
                               road_surface=road_surface,
                               confirmed=confirmed, velocity_reliable=bool(velocity_reliable),
                               hits=hits, observed_s=observed_s, measurement_stamp=stamp,
                               kind=('dynamic' if hits >= 3 and
                                     math.hypot(vx, vy) > self.dynamic_speed_threshold_mps else
                                     'static' if hits >= 3 else 'unknown')))
        # Preserve a briefly occluded rear/side car instead of treating its
        # lane as clear immediately. Do not extend a stale measurement forever.
        for key, track in self.tracks.items():
            dt = stamp-track['stamp']
            if key not in used and 0. < dt <= .4:
                reliable = track.get('velocity_reliable', False)
                result.append(dict(id=key, x=track['x']+(track['vx']*dt if reliable else 0.),
                                   y=track['y']+(track['vy']*dt if reliable else 0.),
                                   radius=track['radius']+.5*dt,
                                   length=track.get('length', 2.*track['radius'])+dt,
                                   width=track.get('width', 2.*track['radius'])+dt,
                                   yaw=track.get('yaw', 0.),
                                   road_surface=track.get('road_surface', False),
                                   vx=track['vx'], vy=track['vy'], kind='unknown',
                                   confirmed=track.get('confirmed', False), velocity_reliable=reliable,
                                   hits=track['hits'], observed_s=track['stamp']-track.get('first_stamp', track['stamp']),
                                   measurement_stamp=track['stamp']))
        return result

    def on_cloud(self, msg):
        with self.lock:
            self.pending = msg

    def on_imu(self, msg):
        self.last_imu_frame = msg.header.frame_id
        if msg.header.frame_id != self.imu_frame_id or msg.orientation_covariance[0] == -1:
            self.imu_rejection = 'wrong_frame' if msg.header.frame_id != self.imu_frame_id else 'orientation_unavailable'
            return
        q = msg.orientation
        try:
            orientation = normalized_quaternion((q.x, q.y, q.z, q.w))
        except ValueError:
            self.imu_rejection = 'invalid_quaternion'
            return
        stamp = msg.header.stamp.to_sec()
        if stamp <= 0.:
            self.imu_rejection = 'invalid_stamp'
            return
        self.imu_rejection = None
        with self.lock:
            if self.imus and stamp < self.imus[-1][0]:
                self.imus.clear()
            if not self.imus or stamp > self.imus[-1][0]:
                self.imus.append((stamp, orientation))

    def filter_ground(self, raw, stamp):
        points = raw @ self.body_from_lidar.T + [self.lidar_x, self.lidar_y, self.lidar_z]
        with self.lock:
            samples = list(self.imus)
        q, skew = quaternion_at(samples, stamp, self.imu_max_skew_s)
        status = dict(stamp=stamp, frame_id='map', imu_corrected=q is not None,
                      cloud_stamp=stamp, imu_skew_s=skew, roll_deg=None, pitch_deg=None,
                      imu_samples=len(samples), imu_rejection=getattr(self, 'imu_rejection', None),
                      imu_frame=getattr(self, 'last_imu_frame', None), input_points=len(raw))
        if q is not None:
            rotation, status['roll_deg'], status['pitch_deg'] = level_rotation(q, self.body_from_imu)
            points = points @ rotation.T
        eligible = road_roi_mask(points, self.front_range, self.rear_range, self.side_range,
                                 self.half_width, self.front_extent, self.rear_extent)
        near = eligible & (np.abs(points[:, 0]) < 12.) & (np.abs(points[:, 1]) < 3.)
        status['near_z_percentiles_m'] = (np.percentile(points[near, 2], [5, 50, 95]).tolist()
                                         if near.any() else None)
        status['roi_points'] = int(eligible.sum())
        # Fit only after fresh IMU correction. Fitting a tilted cloud with the
        # near-ego height anchor could accidentally accept a vehicle surface.
        if q is not None:
            ground, obstacle, detail = segment_ground(points, eligible,
                self.ground_plane_tolerance_m, self.ground_max_slope_deg,
                min_height=self.obstacle_min_height_m, max_height=self.obstacle_max_height_m,
                reference_z=getattr(self, 'ground_reference_z_m', 0.),
                min_plane_height=getattr(self, 'obstacle_min_plane_height_m', None))
            status.update(detail)
        else:
            ground = np.zeros(len(points), dtype=bool)
            height = points[:, 2]-getattr(self, 'ground_reference_z_m', 0.)
            obstacle = eligible & (height > self.obstacle_min_height_m) & (height < self.obstacle_max_height_m)
            status.update(planes=0, supported_points=0, residual_p95_m=None)
        status.update(removed_points=int(ground.sum()), obstacle_points=int(obstacle.sum()),
                      mode=('imu_missing_or_stale' if q is None else
                            'ground_filtered' if status['planes'] else 'ground_unconfirmed'),
                      ground_plane_tolerance_m=self.ground_plane_tolerance_m,
                      imu_max_skew_s=self.imu_max_skew_s,
                      obstacle_min_height_m=self.obstacle_min_height_m,
                      obstacle_max_height_m=self.obstacle_max_height_m,
                      ground_reference_z_m=getattr(self, 'ground_reference_z_m', 0.))
        return points, ground, obstacle, status

    def publish_wait_status(self, mode, msg=None):
        now = time.monotonic()
        if now < getattr(self, 'next_wait_status', 0.):
            return
        self.next_wait_status = now+.2
        ros_now = rospy.Time.now().to_sec()
        with self.lock:
            poses = list(getattr(self, 'poses', []))
            imus = list(getattr(self, 'imus', []))
        cloud_stamp = msg.header.stamp.to_sec() if msg is not None else None
        data = dict(stamp=ros_now, frame_id='map', mode=mode, imu_corrected=False,
                    cloud_stamp=cloud_stamp, cloud_frame=msg.header.frame_id if msg is not None else None,
                    cloud_age_s=ros_now-cloud_stamp if cloud_stamp is not None else None,
                    pose_samples=len(poses), imu_samples=len(imus),
                    pose_nearest_skew_s=min((abs(p[0]-cloud_stamp) for p in poses), default=None)
                        if cloud_stamp is not None else None,
                    imu_nearest_skew_s=min((abs(p[0]-cloud_stamp) for p in imus), default=None)
                        if cloud_stamp is not None else None,
                    imu_rejection=getattr(self, 'imu_rejection', None),
                    imu_frame=getattr(self, 'last_imu_frame', None))
        self.ground_status_pub.publish(String(data=json.dumps(data)))

    def publish_filter_debug(self, points, ground, obstacle, pose, status, header):
        self.ground_status_pub.publish(String(data=json.dumps(status)))
        c, s = math.cos(pose[2]), math.sin(pose[2])
        for mask, publisher in ((ground, self.ground_points_pub), (obstacle, self.obstacle_points_pub)):
            if not publisher.get_num_connections():
                continue
            selected = points[mask][::max(1, int(math.ceil(mask.sum()/15000.)))]
            xyz = np.column_stack((pose[0]+c*selected[:, 0]-s*selected[:, 1],
                                   pose[1]+s*selected[:, 0]+c*selected[:, 1], selected[:, 2]))
            publisher.publish(point_cloud2.create_cloud_xyz32(
                Header(stamp=header.stamp, frame_id='map'), xyz))

    def process_pending(self, _event):
        runtime = getattr(self, 'runtime_settings', None)
        if runtime is not None:
            runtime.poll(dict(raceline_width=self.lane_corridor.width
                              if self.lane_corridor is not None else None))
        with self.lock:
            msg = self.pending
            corridor = self.lane_corridor
        if msg is None:
            if time.monotonic()-getattr(self, 'last_filter_wall', 0.) > .5:
                self.publish_wait_status('waiting_cloud')
            return
        if corridor is None:
            self.publish_wait_status('waiting_lane_geometry', msg)
            rospy.logwarn_throttle(5., 'Waiting for raceline lane geometry')
            return
        stamp = msg.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        if msg.header.frame_id != 'lidar' or not -.05 <= age <= .35:
            self.publish_wait_status('wrong_cloud_frame' if msg.header.frame_id != 'lidar'
                                     else 'cloud_time_mismatch', msg)
            with self.lock:
                if self.pending is msg:
                    self.pending = None
            return
        pose = self.pose_at(stamp)
        if pose is None:
            self.publish_wait_status('waiting_pose', msg)
            return
        with self.lock:
            if self.pending is msg:
                self.pending = None
        filter_start = time.monotonic()
        points = np.asarray(list(point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=float)
        points = points.reshape((-1, 3))
        points, ground, mask, status = self.filter_ground(points, stamp)
        self.last_filter_wall = time.monotonic()
        status['cloud_age_s'] = age
        status['filter_processing_s'] = time.monotonic()-filter_start
        self.publish_filter_debug(points, ground, mask, pose, status, msg.header)
        selected = points[mask]
        xy = selected[:, :2]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        map_xy = np.column_stack((pose[0]+c*xy[:, 0]-s*xy[:, 1],
                                  pose[1]+s*xy[:, 0]+c*xy[:, 1]))
        # Drop roadside returns before clustering so a fence cannot enlarge
        # or merge with a car that is actually inside a mapped lane.
        inside = corridor.contains(map_xy)
        xy, selected = xy[inside], selected[inside]
        support_xy, support_z = map_xy[inside], selected[:, 2]
        if len(xy) > 10000:
            stride = int(math.ceil(len(xy)/10000.))
            xy = xy[::stride]
        found = []
        edge_band = getattr(self, 'lane_edge_band_m', 0.)
        for group in clusters(xy):
            world = np.column_stack((pose[0]+c*group[:, 0]-s*group[:, 1],
                                     pose[1]+s*group[:, 0]+c*group[:, 1]))
            # A guardrail or kerb along the lane line reaches only the outer
            # edge band. Anything there is beyond the ego footprint band, and
            # as a lane-aligned box on a bend it would bulge into the lane.
            if edge_band > 0. and np.max(corridor.depth(world)) < edge_band:
                continue
            box = object_box(world, corridor.direction(world.mean(axis=0)),
                             self.cluster_min_size_m, self.cluster_max_size_m)
            if box is not None:
                road_surface = road_surface_in_box(
                    support_xy, support_z, box, getattr(self, 'ground_reference_z_m', 0.),
                    status.get('imu_corrected', False) and status.get('planes', 0) > 0)
                found.append((*box, road_surface))
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
        self.pub.publish(String(data=json.dumps(dict(
            stamp=stamp, frame_id='map', obstacles=tracked))))


if __name__ == '__main__':
    rospy.init_node('lidar_obstacles')
    LidarObstacles()
    rospy.spin()
