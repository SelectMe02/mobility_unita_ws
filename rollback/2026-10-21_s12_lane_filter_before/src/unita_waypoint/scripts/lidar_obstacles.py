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
from obstacle_motion import box_bounds, update_motion
from avoidance_runtime import RuntimeSettings
from perception_debug import xyz_cloud
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
        self.debug_max_points = max(0, int(rospy.get_param('~debug_max_points', 0)))
        self.runtime_settings = RuntimeSettings('lidar_obstacles', self)
        self.pub = rospy.Publisher('/waypoint_debug/obstacles', String, queue_size=1)
        self.ground_status_pub = rospy.Publisher('/waypoint_debug/ground_status', String, queue_size=1)
        self.ground_points_pub = rospy.Publisher('/waypoint_debug/ground_points', PointCloud2, queue_size=1)
        self.obstacle_points_pub = rospy.Publisher('/waypoint_debug/obstacle_points', PointCloud2, queue_size=1)
        self.discarded_points_pub = rospy.Publisher('/waypoint_debug/discarded_points', PointCloud2, queue_size=1)
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
            corridor = getattr(self, 'lane_corridor', None)
            measured_lane = (corridor.label_box(x, y, length, width, yaw)
                             if corridor is not None else {})
            matches = [(math.hypot(x-t['x']-t['vx']*(stamp-t['stamp']),
                                  y-t['y']-t['vy']*(stamp-t['stamp'])), key, t)
                       for key, t in self.tracks.items() if key not in used
                       and not (measured_lane.get('lane_sector') == 11
                                and measured_lane.get('lane_label_confident', False)
                                and not measured_lane.get('lane_boundary_crossing', True)
                                and t.get('lane_stable', False)
                                and t.get('lane_sector') == 11
                                and t.get('raceline') != measured_lane.get('raceline'))
                       and max(radius, t['radius']) <= 3.*max(.15, min(radius, t['radius']))]
            matches.sort()
            # Allow a fast rear car's initial displacement before velocity
            # converges; subsequent association uses its predicted position.
            if matches and matches[0][0] < max(2.5, 35.*(stamp-matches[0][2]['stamp'])):
                _, key, previous = matches[0]
                dt = stamp - previous['stamp']
                if dt <= 0:
                    continue
                hits = min(20, previous['hits']+1)
                first_stamp = previous.get('first_stamp', previous['stamp']-(previous['hits']-1)*dt)
            else:
                previous = {}
                key = self.next_id
                self.next_id += 1
                hits = 1
                first_stamp = stamp
            observed_s = stamp-first_stamp
            confirmed = hits >= 3 and observed_s+1e-9 >= getattr(self, 'track_confirm_time_s', .2)
            motion = update_motion(previous, stamp, box_bounds(x, y, length, width, yaw),
                                   self.dynamic_speed_threshold_mps)
            vx, vy = motion['vx'], motion['vy']
            velocity_reliable = confirmed and motion['velocity_reliable']
            motion['velocity_reliable'] = bool(velocity_reliable)
            lane = self._track_lane(x, y, length, width, yaw, stamp, previous,
                                    confirmed)
            track = dict(x=x, y=y, radius=radius, **motion,
                         length=length, width=width, yaw=yaw,
                         road_surface=road_surface,
                         hits=hits, stamp=stamp, first_stamp=first_stamp,
                         confirmed=confirmed,
                         **lane)
            self.tracks[key] = track
            used.add(key)
            result.append(dict(id=key, x=x, y=y, radius=radius, vx=vx, vy=vy,
                               candidate_vx=motion['candidate_vx'],
                               candidate_vy=motion['candidate_vy'],
                               prediction_confirmed=motion['prediction_confirmed'],
                               length=length, width=width, yaw=yaw,
                               road_surface=road_surface,
                               confirmed=confirmed, velocity_reliable=bool(velocity_reliable),
                               hits=hits, observed_s=observed_s, measurement_stamp=stamp,
                               raw_width=width, predicted=False, **lane,
                               kind=motion['motion_kind'] if confirmed else 'unknown'))
        # Preserve a briefly occluded rear/side car instead of treating its
        # lane as clear immediately. Do not extend a stale measurement forever.
        for key, track in self.tracks.items():
            dt = stamp-track['stamp']
            if key not in used and 0. < dt <= .4:
                reliable = track.get('velocity_reliable', False)
                x = track['x']+(track['vx']*dt if reliable else 0.)
                y = track['y']+(track['vy']*dt if reliable else 0.)
                length = track.get('length', 2.*track['radius'])+dt
                raw_width = track.get('width', 2.*track['radius'])+dt
                width = raw_width
                # Stable S11 traffic keeps its observed lateral footprint
                # during short occlusions; longitudinal uncertainty remains.
                if (track.get('lane_stable', False) and track.get('lane_sector') == 11
                        and not track.get('lane_boundary_crossing', True)):
                    width = track['width']
                lane = self._predicted_lane(track, x, y, length, width)
                if width != raw_width and (not lane['lane_stable']
                                           or lane['lane_boundary_crossing']):
                    width = raw_width
                    lane = self._predicted_lane(track, x, y, length, width)
                result.append(dict(id=key, x=x, y=y,
                                   radius=(max(.15, .5*math.hypot(length, width))
                                           if width != raw_width else track['radius']+.5*dt),
                                   length=length, width=width,
                                   yaw=track.get('yaw', 0.),
                                   road_surface=track.get('road_surface', False),
                                   vx=track['vx'], vy=track['vy'], kind='unknown',
                                   prediction_confirmed=track.get('prediction_confirmed', False),
                                   confirmed=track.get('confirmed', False), velocity_reliable=reliable,
                                   hits=track['hits'], observed_s=track['stamp']-track.get('first_stamp', track['stamp']),
                                   measurement_stamp=track['stamp'], raw_width=raw_width,
                                   predicted=True, **lane))
        return result

    def _track_lane(self, x, y, length, width, yaw, stamp, previous, confirmed):
        corridor = getattr(self, 'lane_corridor', None)
        if corridor is None:
            return dict(raceline=0, lane_sector=0, lane_stable=False,
                        lane_boundary_crossing=True)
        lane = corridor.label_box(x, y, length, width, yaw)
        candidate = (lane['lane_sector'], lane['raceline'])
        same = (candidate == previous.get('lane_candidate')
                and previous.get('lane_label_confident', False)
                and lane['lane_label_confident'])
        hits = (previous.get('lane_hits', 0)+1 if same else 1) if lane['lane_label_confident'] else 0
        since = previous.get('lane_since', stamp) if same else stamp
        label = (previous.get('lane_sector', 0), previous.get('raceline', 0))
        stable = (candidate[1] != 0 and lane['lane_label_confident']
                  and hits >= 3 and stamp-since+1e-9 >= .2 and confirmed)
        if not label[1] or stable:
            label = candidate
        lane.update(lane_candidate=candidate, lane_hits=hits, lane_since=since,
                    lane_stable=bool(stable and label == candidate))
        if label != candidate:
            lane.update(lane_sector=label[0], raceline=label[1],
                        lane_boundary_crossing=True)
        return lane

    def _predicted_lane(self, track, x, y, length, width):
        corridor = getattr(self, 'lane_corridor', None)
        if corridor is None:
            return dict(raceline=0, lane_sector=0, lane_stable=False,
                        lane_boundary_crossing=True)
        lane = corridor.label_box(x, y, length, width, track.get('yaw', 0.))
        lane['lane_stable'] = bool(track.get('lane_stable', False)
            and lane['lane_label_confident']
            and (lane['lane_sector'], lane['raceline'])
                == (track.get('lane_sector'), track.get('raceline')))
        return lane

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
                    debug_valid=False,
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
        # A stopped sensor, invalid frame, or unsynchronised pose must not
        # leave an old red cloud looking like current perception evidence.
        empty = xyz_cloud(
            Header(stamp=rospy.Time.from_sec(ros_now), frame_id='map'), [])
        for name in ('ground_points_pub', 'obstacle_points_pub', 'discarded_points_pub'):
            publisher = getattr(self, name, None)
            if publisher is not None and publisher.get_num_connections():
                publisher.publish(empty)

    def publish_filter_debug(self, points, ground, obstacle, pose, status, header,
                             discarded=None):
        """Publish the actual lane-filtered perception input at cloud time.

        Red points are the input to clustering and road-surface evidence,
        not a claim that every point became an accepted tracked obstacle.
        All valid points are displayed by default. An optional configured
        limit only reduces visualisation traffic, never detector input.
        """
        c, s = math.cos(pose[2]), math.sin(pose[2])
        clouds = [(ground, self.ground_points_pub, 'ground'),
                  (obstacle, self.obstacle_points_pub, 'used')]
        if discarded is not None:
            clouds.append((discarded, getattr(self, 'discarded_points_pub', None), 'discarded'))
        displayed = {}
        for mask, publisher, name in clouds:
            count = int(mask.sum())
            maximum = getattr(self, 'debug_max_points', 0)
            stride = max(1, int(math.ceil(count/maximum))) if maximum else 1
            displayed[name] = dict(total=count, displayed=int(math.ceil(count/stride)),
                                   stride=stride)
            if publisher is None or not publisher.get_num_connections():
                continue
            selected = points[mask][::stride]
            xyz = np.column_stack((pose[0]+c*selected[:, 0]-s*selected[:, 1],
                                   pose[1]+s*selected[:, 0]+c*selected[:, 1], selected[:, 2]))
            publisher.publish(xyz_cloud(
                Header(stamp=header.stamp, frame_id='map'), xyz))
        status.update(debug_valid=True, debug_point_stage='lane_filtered_input',
                      debug_clouds=displayed)
        self.ground_status_pub.publish(String(data=json.dumps(status)))

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
        selected = points[mask]
        xy = selected[:, :2]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        map_xy = np.column_stack((pose[0]+c*xy[:, 0]-s*xy[:, 1],
                                  pose[1]+s*xy[:, 0]+c*xy[:, 1]))
        # Drop roadside returns before clustering so a fence cannot enlarge
        # or merge with a car that is actually inside a mapped lane.
        inside = corridor.contains(map_xy)
        # Keep indices into the full corrected scan for faithful debug
        # colours. The detector below still receives exactly its old arrays.
        used = np.zeros(len(points), dtype=bool)
        used[np.flatnonzero(mask)[inside]] = True
        finite = np.isfinite(points).all(axis=1)
        discarded = finite & ~used
        status.update(candidate_points=int(mask.sum()), used_points=int(used.sum()),
                      ground_removed_points=int(ground.sum()),
                      lane_removed_points=int((~inside).sum()),
                      other_removed_points=int((finite & ~ground & ~mask).sum()),
                      discarded_points=int(discarded.sum()),
                      invalid_points=int((~finite).sum()))
        xy, selected = xy[inside], selected[inside]
        support_xy, support_z = map_xy[inside], selected[:, 2]
        if len(xy) > 10000:
            stride = int(math.ceil(len(xy)/10000.))
            xy = xy[::stride]
        found = []
        edge_band = getattr(self, 'lane_edge_band_m', 0.)
        status.update(cluster_input_points=len(xy), clusters=0,
                      edge_rejected_clusters=0, size_rejected_clusters=0)
        for group in clusters(xy):
            status['clusters'] += 1
            world = np.column_stack((pose[0]+c*group[:, 0]-s*group[:, 1],
                                     pose[1]+s*group[:, 0]+c*group[:, 1]))
            # A guardrail or kerb along the lane line reaches only the outer
            # edge band. Anything there is beyond the ego footprint band, and
            # as a lane-aligned box on a bend it would bulge into the lane.
            if edge_band > 0. and np.max(corridor.depth(world)) < edge_band:
                status['edge_rejected_clusters'] += 1
                continue
            box = object_box(world, corridor.direction(world.mean(axis=0)),
                             self.cluster_min_size_m, self.cluster_max_size_m)
            if box is not None:
                road_surface = road_surface_in_box(
                    support_xy, support_z, box, getattr(self, 'ground_reference_z_m', 0.),
                    status.get('imu_corrected', False) and status.get('planes', 0) > 0)
                found.append((*box, road_surface))
            else:
                status['size_rejected_clusters'] += 1
        mask = corridor.contains([o[:2] for o in found])
        status['center_rejected_detections'] = int((~mask).sum())
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
        status.update(detected_obstacles=len(found), tracked_obstacles=len(tracked),
                      predicted_obstacles=sum(bool(o.get('predicted', False)) for o in tracked))
        self.publish_filter_debug(points, ground, used, pose, status, msg.header, discarded)
        self.pub.publish(String(data=json.dumps(dict(
            stamp=stamp, frame_id='map', obstacles=tracked))))


if __name__ == '__main__':
    rospy.init_node('lidar_obstacles')
    LidarObstacles()
    rospy.spin()
