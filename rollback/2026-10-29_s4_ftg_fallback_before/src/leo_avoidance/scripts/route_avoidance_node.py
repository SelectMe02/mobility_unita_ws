#!/usr/bin/env python3
"""GPS-route obstacle handling: LiDAR tracks -> offset path and speed cap.

Publishes /leo/route_avoidance (JSON): a map-frame path for Pure Pursuit, a
speed cap, the state and the tracked obstacles. It never publishes control;
the follower stops when this output is missing or stale.
"""
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String

from leo_avoidance.road_points import RoadPointConfig, boundary_mask, obstacle_points
from leo_avoidance.obstacle_tracker import ObstacleTracker
from leo_avoidance.observed_clearance import pedestrian_clearance_observer, static_clearance_observer
from leo_avoidance.route_avoidance import (RouteAvoidance, RouteAvoidanceConfig,
                                           RouteObstacle)
from leo_guard.route_file import (load_lane_changes, load_route_points, load_survey,
                                  point_id_ranges)

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from avoidance_runtime import FrozenConfigTarget, RuntimeSettings  # noqa: E402
from lane_corridor import LaneCorridor, SectorLaneFilter, project_lane_geometry  # noqa: E402
from map_tunner_core import load_project, raceline_layout_settings  # noqa: E402
from perception_debug import PerceptionDebug, tracked_obstacles_map  # noqa: E402


def load_sector_lane_filter(waypoint_file):
    """Use the follower's saved geometry before its latched message arrives."""
    if Path(waypoint_file).suffix.lower() != '.json':
        return None
    project = load_project(waypoint_file)
    config = rospy.get_param('~raceline_config', str(
        Path(rospkg.RosPack().get_path('unita_waypoint')) / 'config/raceline_layout.yaml'))
    width = raceline_layout_settings(project, config, rospy.get_param('~raceline_width', None))
    return SectorLaneFilter(project_lane_geometry(project, width), sector=12, width=3.2)


def pedestrian_hold_points(waypoint_file, sector):
    """Map points bounding the pedestrian hold zone, or None.

    Starts at the sector's first raceline-2 point and runs to
    the end of raceline 2 that continues from that sector, which reaches past
    the next checkpoint.
    """
    if sector <= 0 or Path(waypoint_file).suffix.lower() != '.json':
        return None
    sectors = {int(item['number']): item for item in load_project(waypoint_file)['sectors']}
    if sector not in sectors:
        raise ValueError('pedestrian hold sector %d is not in the project' % sector)
    if not sectors[sector]['racelines'].get('2'):
        return None
    start = sectors[sector]['racelines']['2'][0]
    end = sectors[sector]['racelines']['1'][-1]
    number = sector
    while sectors.get(number, {}).get('racelines', {}).get('2'):
        end = sectors[number]['racelines']['2'][-1]
        number += 1
    return (float(start['x']), float(start['y'])), (float(end['x']), float(end['y']))


def configure_s4_static_pass(planner, waypoint_file):
    if Path(waypoint_file).suffix.lower() != '.json':
        return
    sector = next((s for s in load_project(waypoint_file)['sectors']
                   if int(s['number']) == 4), None)
    if not sector or not sector['racelines'].get('2'):
        return
    route = planner.route
    project = lambda p: route.project(float(p['x']), float(p['y']), 0, len(route.lengths))
    lane = [project(p) for p in sector['racelines']['2']]
    offset = float(np.median([p[1] for p in lane]))
    if max(abs(p[1]-offset) for p in lane) > .35:
        raise ValueError('S4 R2 is not a parallel corridor')
    start_s = project(sector['racelines']['1'][0])[0]
    end_s = project(sector['racelines']['1'][-1])[0]
    planner.set_static_pass_zone(max(start_s, lane[0][0]-18.),
                                lane[0][0], lane[-1][0], end_s, offset)


class RouteAvoidanceNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.lidar_to_rear_x_m = float(rospy.get_param('~lidar_to_rear_x_m', 1.4))
        self.input_timeout_s = float(rospy.get_param('~input_timeout_s', .35))
        self.max_pose_gap_s = float(rospy.get_param('~max_pose_age_s', .15))
        waypoint_file = rospy.get_param('~waypoint_file')
        points = load_route_points(waypoint_file)[:, :2]
        self.sector_lane_filter_enabled = Path(waypoint_file).suffix.lower() == '.json'
        self.sector_lane_filter = load_sector_lane_filter(waypoint_file)
        config = RouteAvoidanceConfig(**{
            key: float(rospy.get_param('~' + key, default))
            for key, default in vars(RouteAvoidanceConfig()).items()})
        # Dashed-line intervals from MGeo (build_route_lane_changes.py); a
        # route without them never leaves its lane, it waits instead.
        lane_file = rospy.get_param('~lane_change_file', '')
        lane_changes = load_lane_changes(lane_file, points) if lane_file else []
        # Crosswalks just past each signal stop line (no crosswalk layer in
        # the map data; at C016 the crosswalk is about 7 m past the line): a
        # person crossing there means stopping before it.
        crosswalks = []
        survey_file = rospy.get_param('~intersection_file', '')
        if survey_file:
            for stop in load_survey(survey_file, points):
                if stop.get('controlled') is True:
                    crosswalks.append((stop['s'] + float(rospy.get_param('~crosswalk_start_m', 1.)),
                                       stop['s'] + float(rospy.get_param('~crosswalk_end_m', 10.))))
        self.planner = RouteAvoidance(points, config, lane_changes, crosswalks)
        configure_s4_static_pass(self.planner, waypoint_file)
        if Path(waypoint_file).suffix.lower() == '.json':
            for sector in load_project(waypoint_file)['sectors']:
                if int(sector['number']) in (4, 5, 6):
                    line = sector['racelines']['1']
                    route = self.planner.route
                    bounds = [route.project(float(p['x']), float(p['y']), 0,
                                            len(route.lengths))[0]
                              for p in (line[0], line[-1])]
                    self.planner.pass_speed_25_ranges.append(tuple(bounds))
        # Same Map Tuner point-ID zones as the follower's waypoint_only_point_ids.
        route = self.planner.route
        self.planner.ignored_s_ranges = []
        for first, last in point_id_ranges(rospy.get_param('~waypoint_file'),
                                           rospy.get_param('~waypoint_only_point_ids', '')):
            s_first = route.project(points[first, 0], points[first, 1], 0, len(route.lengths))[0]
            s_last = route.project(points[last, 0], points[last, 1], 0, len(route.lengths))[0]
            self.planner.ignored_s_ranges.append((s_first, s_last))
            rospy.loginfo('Route avoidance ignores obstacles at s %.1f..%.1f m', s_first, s_last)
        # Crossing-pedestrian hold: only where the competition places them.
        bounds = pedestrian_hold_points(waypoint_file,
                                        int(rospy.get_param('~pedestrian_hold_sector', 0)))
        if bounds is not None:
            base = len(points) - 1
            s_start, s_end = (route.project(x, y, 0, base)[0] for x, y in bounds)
            if not s_start < s_end:
                raise ValueError('invalid pedestrian hold zone')
            self.planner.set_pedestrian_hold_ranges([(s_start, s_end)])
            rospy.loginfo('Route avoidance holds crossing pedestrians at s %.1f..%.1f m',
                          s_start, s_end)
        self.runtime = RuntimeSettings('leo_route_avoidance', FrozenConfigTarget(self.planner))
        # Open-road returns: local ground per grid cell, not the tunnel's single
        # road plane (which turns a road rising ahead into obstacles).
        self.road_points = RoadPointConfig(**{
            key: (tuple(rospy.get_param('~road_' + key, list(default)))
                  if isinstance(default, tuple) else
                  float(rospy.get_param('~road_' + key, default)))
            for key, default in vars(RoadPointConfig(
                lidar_to_rear_x_m=self.lidar_to_rear_x_m)).items()})
        self.tracker = ObstacleTracker(
            self.lidar_to_rear_x_m, region_x_m=(-10., 60.),
            region_half_width_m=float(rospy.get_param('~track_half_width_m', 10.)),
            max_size_m=float(rospy.get_param('~track_max_size_m', 8.)))
        self.pose = None
        self.poses = deque(maxlen=100)
        self.speed = None
        self.signal = None
        # Several scans can await their following pose sample at once. Keep
        # them bounded, and process the newest one whose capture pose is ready.
        self.pending = deque(maxlen=8)
        self.last_selected_stamp = None
        self.input_epoch = 0
        self.processed_epoch = 0
        self.last_worker_ros_time = None
        self.closed = False
        self.pub = rospy.Publisher('/leo/route_avoidance', String, queue_size=1)
        self.debug = PerceptionDebug('/leo/route_debug')
        self.debug_lidar_z = float(rospy.get_param('~lidar_to_rear_z_m', 1.23))
        rospy.Subscriber(rospy.get_param('~lane_geometry_topic', '/leo/waypoint_debug/lane_geometry'),
                         String, self.on_lane_geometry, queue_size=1, buff_size=2 ** 22)
        rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'),
                         PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(rospy.get_param('~speed_topic', '/competition/ego_speed_kmh'),
                         Float64, self.on_speed, queue_size=1)
        rospy.Subscriber(rospy.get_param('~signal_status_topic', '/leo/signal_status'),
                         String, self.on_signal, queue_size=1)
        rospy.Subscriber(rospy.get_param('~cloud_topic', '/lidar3D'), PointCloud2,
                         self.on_cloud, queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self.close)
        threading.Thread(target=self.worker, daemon=True).start()

    def on_lane_geometry(self, message):
        if not getattr(self, 'sector_lane_filter_enabled', True):
            return
        try:
            geometry = json.loads(message.data)
            LaneCorridor(geometry)  # Validate the complete map, including frame and lane IDs.
            lane_filter = SectorLaneFilter(geometry, sector=12, width=3.2)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            # Retain validated geometry; a bad update must not open the lane filter.
            rospy.logwarn_throttle(5., 'Invalid route lane geometry: %s', error)
            return
        with self.lock:
            if not self.closed:
                previous = getattr(self, 'sector_lane_filter', None)
                if (previous is not None and previous.corridor is not None
                        and lane_filter.corridor is None):
                    rospy.logwarn_throttle(5., 'Route lane geometry update has no S12 lane; keeping prior map')
                    return
                self.sector_lane_filter = lane_filter

    def _world_xy(self, xy, pose):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        rear_x = xy[:, 0] + self.lidar_to_rear_x_m
        return np.column_stack((x0 + c * rear_x - s * xy[:, 1],
                                y0 + s * rear_x + c * xy[:, 1]))

    def pedestrian_hold_status(self):
        hold = getattr(self.planner, 'pedestrian_hold', None)
        if not isinstance(hold, dict):
            return None
        return dict(s=round(hold['s'], 1), d=round(hold['d'], 2),
                    phase='candidate' if hold.get('tentative') else 'confirmed',
                    active_hazards=len(self.planner.pedestrian_holds),
                    ids=sorted(int(i) for i in hold['ids']),
                    cap_kmh=None if math.isinf(hold['cap']) else round(hold['cap'] * 3.6, 2),
                    full_brake=hold['full_brake'], clear_scans=hold['clear_scans'],
                    static_scans=hold['static_scans'])

    def _prune_sector_tracks(self, corridor, stamp):
        """Do not reconnect an outside history to a newly clipped lane cluster."""
        kept = []
        for track in self.tracker.tracks:
            measured_stamp, x, y = track.history[-1]
            vx, vy = track.velocity if track.velocity_known else (0., 0.)
            dt = max(0., stamp - measured_stamp)
            if corridor.contains([(x, y), (x + vx * dt, y + vy * dt)]).all():
                kept.append(track)
        removed = len(self.tracker.tracks) - len(kept)
        self.tracker.tracks = kept
        return removed

    def on_pose(self, message):
        if message.header.frame_id != 'map':
            return
        p, q = message.pose.position, message.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        stamp = message.header.stamp.to_sec()
        if all(math.isfinite(v) for v in (p.x, p.y, yaw, stamp)):
            with self.lock:
                if self.closed:
                    return
                if self.poses and stamp < self.poses[-1][0]:
                    self._reset_inputs_locked()
                self.pose = ((p.x, p.y, yaw), time.monotonic())
                if not self.poses or stamp > self.poses[-1][0]:
                    self.poses.append((stamp, (p.x, p.y, yaw)))

    def pose_at(self, stamp):
        """Place each scan with its capture pose, including during a turn."""
        with self.lock:
            samples = list(self.poses)
        return self._pose_from_samples(samples, stamp)

    def _pose_from_samples(self, samples, stamp):
        for sample_stamp, pose in samples:
            if abs(sample_stamp - stamp) < 1e-6:
                return pose
        for (before_stamp, before), (after_stamp, after) in zip(samples, samples[1:]):
            if before_stamp <= stamp <= after_stamp:
                gap = after_stamp - before_stamp
                if gap > self.max_pose_gap_s:
                    return None
                fraction = (stamp - before_stamp) / gap
                yaw_delta = math.atan2(math.sin(after[2] - before[2]),
                                       math.cos(after[2] - before[2]))
                return (before[0] + fraction * (after[0] - before[0]),
                        before[1] + fraction * (after[1] - before[1]),
                        before[2] + fraction * yaw_delta)
        return None

    def on_speed(self, message):
        if math.isfinite(message.data):
            with self.lock:
                self.speed = (message.data / 3.6, time.monotonic())

    def on_signal(self, message):
        try:
            status = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.signal = (status, time.monotonic())

    def on_cloud(self, message):
        stamp = message.header.stamp.to_sec()
        if message.header.frame_id != 'lidar' or not math.isfinite(stamp):
            return
        with self.lock:
            if (self.closed or self.last_selected_stamp is not None
                    and stamp <= self.last_selected_stamp
                    or any(item[0].header.stamp.to_sec() == stamp for item in self.pending)):
                return
            self.pending.append((message, time.monotonic()))

    def _reset_inputs_locked(self):
        self.pending.clear()
        self.poses.clear()
        self.pose = None
        self.last_selected_stamp = None
        self.input_epoch += 1

    def close(self):
        with self.lock:
            self.closed = True
            self._reset_inputs_locked()

    def take_ready_scan(self, now, ros_now):
        """Consume the newest synchronized scan; preserve newer waiting ones."""
        with self.lock:
            if self.closed:
                return None
            if self.last_worker_ros_time is not None and ros_now < self.last_worker_ros_time:
                self._reset_inputs_locked()
            self.last_worker_ros_time = ros_now
            if self.processed_epoch != self.input_epoch:
                # Only the worker mutates these objects. Callback time jumps
                # invalidate an in-flight result through input_epoch instead.
                self.tracker.reset()
                self.planner.reset()
                self.processed_epoch = self.input_epoch
            samples = list(self.poses)
            retained = deque(maxlen=self.pending.maxlen)
            selected = None
            for message, received in self.pending:
                stamp = message.header.stamp.to_sec()
                if (not -.05 <= ros_now - stamp <= self.input_timeout_s
                        or not 0. <= now - received <= self.input_timeout_s
                        or self.last_selected_stamp is not None and stamp <= self.last_selected_stamp):
                    continue
                retained.append((message, received))
                pose = self._pose_from_samples(samples, stamp)
                if pose is not None and (selected is None or stamp > selected[0].header.stamp.to_sec()):
                    selected = (message, received, pose, self.input_epoch)
            if selected is not None:
                self.last_selected_stamp = selected[0].header.stamp.to_sec()
                retained = deque((item for item in retained
                                  if item[0].header.stamp.to_sec() > self.last_selected_stamp),
                                 maxlen=self.pending.maxlen)
            self.pending = retained
            return selected

    def fresh(self, item, now, timeout=None):
        timeout = self.input_timeout_s if timeout is None else timeout
        return item[0] if item and 0 <= now - item[1] <= timeout else None

    def worker(self):
        rate = rospy.Rate(50, reset=True)
        while not rospy.is_shutdown() and not self.closed:
            self.runtime.poll()
            pending = self.take_ready_scan(time.monotonic(), rospy.Time.now().to_sec())
            if pending is not None:
                try:
                    self.process(*pending)
                except Exception as error:  # keep the node alive; output goes stale
                    self.debug.clear('processing_error', pending[0].header)
                    rospy.logwarn_throttle(2., 'Route avoidance failed: %s', error)
            else:
                self.debug.clear_if_stale()
            rate.sleep()

    def drop_road_boundaries(self, points, pose):
        """Remove guardrails, fences and walls along the road.

        The tunnel FTG finds its walls; on open road the route frame makes a
        roadside barrier a thin line at nearly constant offset, also on bends.
        """
        self.boundary_debug_counts = dict(boundary_removed_points=0, sampling_removed_points=0)
        if not len(points):
            return points
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        rear_x = points[:, 0] + self.lidar_to_rear_x_m
        xy = np.column_stack((x0 + c * rear_x - s * points[:, 1],
                              y0 + s * rear_x + c * points[:, 1]))
        route = self.planner.route
        if self.planner.last_s is None:
            lo, hi = 0, len(route.lengths)
            s_ego, _, _ = route.project(x0, y0, lo, hi)
        else:
            s_ego = self.planner.last_s
        lo, hi = route.window(s_ego - 15., s_ego + 70.)
        if len(xy) > 6000:  # keep the projection cost bounded
            keep = np.zeros(len(xy), bool)
            keep[::int(math.ceil(len(xy) / 6000.))] = True
            self.boundary_debug_counts['sampling_removed_points'] = int((~keep).sum())
            points, xy = points[keep], xy[keep]
        along, offset = route.project_many(xy, lo, hi)
        boundary = boundary_mask(along, offset)
        self.boundary_debug_counts['boundary_removed_points'] = int(boundary.sum())
        return points[~boundary]

    def process(self, message, received, pose=None, epoch=None):
        now = time.monotonic()
        stamp = message.header.stamp.to_sec()
        with self.lock:
            if self.closed or rospy.is_shutdown() or epoch is not None and epoch != self.input_epoch:
                return
            if epoch is None:
                epoch = self.input_epoch
            latest_pose = self.fresh(self.pose, now)
            speed = self.fresh(self.speed, now)
            signal = self.fresh(self.signal, now, 1.)
            lane_filter = getattr(self, 'sector_lane_filter', None)
        age = rospy.Time.now().to_sec() - stamp
        if (message.header.frame_id != 'lidar' or latest_pose is None or speed is None
                or not -.05 <= age <= self.input_timeout_s
                or not 0. <= now - received <= self.input_timeout_s):
            self.tracker.reset()
            self.debug.clear('waiting_lidar_pose_speed', message.header)
            rospy.logwarn_throttle(2., 'Route avoidance waiting for LiDAR/pose/speed')
            return
        if pose is None:
            pose = self.pose_at(stamp)
        if pose is None:
            # The worker keeps unsynchronized scans in its bounded queue.
            self.debug.clear('waiting_scan_pose', message.header)
            return
        points = np.asarray(list(point_cloud2.read_points(
            message, field_names=('x', 'y', 'z'), skip_nans=True)), float).reshape(-1, 3)
        raw_points = points
        points = obstacle_points(points, self.road_points)
        road_input_count = len(points)
        lane_active = lane_filter is not None and lane_filter.active(pose[:2])
        lane_removed_points = lane_removed_tracks = 0
        if lane_active:
            inside = lane_filter.corridor.contains(self._world_xy(points[:, :2], pose))
            lane_removed_points = int((~inside).sum())
            points = points[inside]
            lane_removed_tracks = self._prune_sector_tracks(lane_filter.corridor, stamp)
        points = self.drop_road_boundaries(points, pose)
        tracked = self.tracker.update(points, pose, stamp)
        if lane_active and tracked:
            inside = lane_filter.corridor.contains(self._world_xy(
                [(t.x_m, t.y_m) for t in tracked], pose))
            rejected_ids = {t.track_id for t, keep in zip(tracked, inside) if not keep}
            tracked = [t for t, keep in zip(tracked, inside) if keep]
            lane_removed_tracks += len(rejected_ids)
            self.tracker.tracks = [t for t in self.tracker.tracks if t.track_id not in rejected_ids]
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        obstacles = []
        for t in tracked:
            rear_x = t.x_m + self.lidar_to_rear_x_m
            vx, vy = (t.vx_mps, t.vy_mps) if t.velocity_known else (0., 0.)
            obstacles.append(RouteObstacle(
                x0 + c * rear_x - s * t.y_m, y0 + s * rear_x + c * t.y_m,
                radius=.5 * t.size_y_m + .1, half_length=.5 * t.size_x_m + .1,
                vx=c * vx - s * vy, vy=s * vx + c * vy,
                # Lateral motion seen in consecutive scans already predicts a
                # crossing, before the track is confirmed dynamic.
                dynamic=t.dynamic or abs(t.lateral_scans) >= 2,
                track_id=t.track_id))
        stop_line, permitted = None, True
        if signal is not None:
            stop_line = signal.get('next_controlled_stop_distance_m')
            permitted = (signal.get('state') == 'clearing'
                         or signal.get('decision') == 'go'
                         or signal.get('release_stable') is True)
        decision = self.planner.update((x0, y0), speed, obstacles,
                                       stop_line_distance_m=stop_line,
                                       stop_line_permitted=permitted, stamp=stamp,
                                       clearance_observer=pedestrian_clearance_observer(
                                           raw_points, pose, self.planner.route,
                                           self.planner.config, self.lidar_to_rear_x_m),
                                       static_clearance_observer=static_clearance_observer(
                                           raw_points, pose, self.planner.route,
                                           self.lidar_to_rear_x_m))
        cap = decision.speed_cap_mps
        output = String(data=json.dumps(dict(
            stamp=stamp, state=decision.state, reason=decision.reason,
            speed_cap_kmh=None if math.isinf(cap) else round(cap * 3.6, 2),
            traffic_speed_cap_kmh=(None if math.isinf(decision.traffic_speed_cap_mps)
                                   else round(decision.traffic_speed_cap_mps * 3.6, 2)),
            offset_m=round(decision.offset_now_m, 2),
            # A traffic yield may replace PASSING in the displayed state;
            # ownership lasts through the retained pass's return ramp.
            pass_active=self.planner.active is not None,
            static_pass_managed=self.planner.static_pass_managed,
            pedestrian_hold=self.pedestrian_hold_status(),
            path=[[round(x, 3), round(y, 3)] for x, y in decision.path],
            obstacles=[dict(s=round(o['s'], 1), d=round(o['d'], 2), r=round(o['r'], 2),
                            l=round(o['l'], 2), v_s=round(o['v_s'], 2),
                            v_d=round(o['v_d'], 2), moving=o['moving'], id=o['id'])
                       for o in decision.obstacles])))
        with self.lock:
            if self.closed or rospy.is_shutdown() or epoch != self.input_epoch:
                return
            current_filter = getattr(self, 'sector_lane_filter', None)
            if (current_filter is not lane_filter and (lane_active or
                    current_filter is not None and current_filter.active(pose[:2]))):
                self.debug.clear('lane_geometry_changed', message.header)
                return
            if (not -.05 <= rospy.Time.now().to_sec() - stamp <= self.input_timeout_s
                    or not 0. <= time.monotonic() - received <= self.input_timeout_s):
                self.debug.clear('processing_expired', message.header)
                return
            self.pub.publish(output)
        if cap <= .01 and (self.planner.static_pass_managed or self.planner.pedestrian_holds
                           or self.planner.active is not None):
            # rosout is recorded even when the route-specific JSON topic is
            # not in a bag. Keep the evidence needed to diagnose a held stop.
            rospy.loginfo_throttle(2., 'Route held-stop evidence: %s', json.dumps(dict(
                stamp=stamp, reason=decision.reason, route_s=self.planner.last_s,
                static_pass_managed=self.planner.static_pass_managed,
                active_pass=(vars(self.planner.active) if self.planner.active else None),
                obstacles=decision.obstacles,
                pedestrian_holds=[{k: sorted(v) if isinstance(v, set) else v
                                   for k, v in h.items()} for h in self.planner.pedestrian_holds])))
        self.debug.publish(raw_points, points, message.header, pose,
                           offset=(self.lidar_to_rear_x_m, 0., self.debug_lidar_z),
                           mode='route_filtered', debug_point_stage=(
                               'sector_lane_boundary_filtered_input' if lane_active
                               else 'route_boundary_filtered_input'),
                           filter_summary=(
                               '국소 격자 노면·높이·자차·거리 → S12 차선 내부 → 경로변 경계 제거 → 추적'
                               if lane_active else '국소 격자 노면·높이·자차·거리 → 경로변 경계 제거 → 추적'),
                           pose_sync='interpolated_scan_pose', imu_corrected=False,
                           candidate_points=road_input_count,
                           ground_height_roi_removed_points=len(raw_points)-road_input_count,
                           sector_lane_filter_active=bool(lane_active),
                           sector_lane_width_m=3.2 if lane_active else None,
                           lane_removed_points=lane_removed_points,
                           lane_removed_tracks=lane_removed_tracks,
                           tracked_obstacles=len(tracked), planner_state=decision.state,
                           obstacles=(tracked_obstacles_map(tracked, pose, self.lidar_to_rear_x_m)
                                      if self.debug.active() else []),
                           planner_reason=decision.reason, **self.boundary_debug_counts)
        if decision.state != 'FOLLOW':
            rospy.loginfo_throttle(.5, 'Route avoidance %s: %s (cap %s km/h, offset %.2f m)',
                                   decision.state, decision.reason,
                                   'none' if math.isinf(cap) else '%.1f' % (cap * 3.6),
                                   decision.offset_now_m)


if __name__ == '__main__':
    rospy.init_node('leo_route_avoidance')
    RouteAvoidanceNode()
    rospy.spin()
