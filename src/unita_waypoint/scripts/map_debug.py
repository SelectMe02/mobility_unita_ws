#!/usr/bin/env python3
"""K-City map, saved route, vehicle state and detected obstacle boxes."""

import json
import copy
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import rospkg
import rospy
import tf
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from std_msgs.msg import Bool, Float64, String, Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(rospkg.RosPack().get_path('unita_waypoint'))
DEBUG_PROJECT_FILE = Path('/home/unita/unita_ws/src/unita_waypoint/config/map_tunner_project.json')
sys.path.insert(0, str(ROOT / 'scripts'))
from map_tunner_core import (CHECKPOINT_ORDER, load_checkpoints, load_csv,
                             load_project, primary_route, raceline_layout_settings)
from map_tunner_map import AerialMap
from lane_corridor import LaneCorridor, project_lane_geometry
from avoidance_runtime import SETTINGS, validated_value
from map_debug_geometry import (nearby_lane_edges, vehicle_box_corners, BOX_EDGE_PAIRS,
                                obstacle_z_bounds)


def marker(namespace, identifier, kind):
    result = Marker()
    result.header.frame_id = 'map'
    result.header.stamp = rospy.Time.now()
    result.ns = namespace
    result.id = identifier
    result.type = kind
    result.action = Marker.ADD
    result.pose.orientation.w = 1.0
    return result


class MapDebug:
    def __init__(self):
        config = ROOT / 'config'
        checkpoint_path = rospy.get_param('~checkpoint_file', str(config / 'map_tunner_checkpoints.yaml'))
        reference_path = rospy.get_param('~waypoint_file', str(config / 'waypoints.csv'))
        image_path = rospy.get_param('~image_file', str(config / 'kcity_map_sim.png'))
        calibration_path = rospy.get_param('~calibration_file', str(config / 'map_tunner_calibration.yaml'))
        self.project_file = DEBUG_PROJECT_FILE.resolve()
        self.project = load_project(self.project_file)
        self.raceline_config = rospy.get_param('~raceline_config', str(config / 'raceline_layout.yaml'))
        self.raceline_width_override = rospy.get_param('~raceline_width', None)
        width = raceline_layout_settings(self.project, self.raceline_config)
        self.saved_lane_corridor = LaneCorridor(project_lane_geometry(self.project, width))
        self.lane_corridor = self.saved_lane_corridor
        self.live_lane_geometry = None
        self.project_stamp = self.project_file.stat().st_mtime_ns
        self.next_project_check = 0.0
        self.route, self.sectors = primary_route(self.project)
        self.xy = np.asarray([(point['x'], point['y']) for point in self.route])
        self.checkpoints = load_checkpoints(checkpoint_path)
        self.aerial = AerialMap(image_path, calibration_path, self.checkpoints,
                                rospy.get_param('~map_resolution', 0.5))
        self.reference = load_csv(reference_path)
        self.mesh_directory = tempfile.TemporaryDirectory(prefix='map_debug_photo_')
        self.pose = None
        self.pose_time = None
        self.speed = None
        self.speed_time = None
        self.tracking = None
        self.valid = False
        self.last_vehicle_visible = False
        self.obstacle_snapshot = None
        self.obstacle_marker_ids = set()
        self.config_lock = threading.Lock()
        self.config_snapshots = {}
        self.pending_parameters = {}
        self.parameter_error = ''
        self.horizon_snapshot = None
        self.ground_snapshot = None
        self.source_snapshots = {key: {} for key in ('raceline', 'leo', 'route', 'ftg')}
        self.guard_snapshot = None
        self.active_source = 'raceline'
        self.perception_view = 'auto'
        self.perception_subscription_source = None
        self.perception_subscribers = []
        self.local_path_snapshot = None
        self.local_path_visible = False
        self.lidar_scene_visible = False
        self.vehicle_geometry = {
            'front': float(rospy.get_param('~vehicle_front_m', 3.845)),
            'rear': float(rospy.get_param('~vehicle_rear_m', .790)),
            'half_width': float(rospy.get_param('~vehicle_half_width_m', .946)),
        }
        height = rospy.get_param('~vehicle_height_m', 2.434)
        self.vehicle_height = float(height) if height != '' else None
        if (any(not math.isfinite(v) or v <= 0. for v in self.vehicle_geometry.values())
                or self.vehicle_height is not None and
                (not math.isfinite(self.vehicle_height) or self.vehicle_height <= 0.)):
            raise ValueError('invalid debug vehicle dimensions')
        self.ground_z = float(rospy.get_param('~ground_z_m', -.38))
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.photo_pub = rospy.Publisher('/map_debug/photo', Marker, queue_size=1, latch=True)
        self.map_pub = rospy.Publisher('/map_debug/map', OccupancyGrid, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher('/map_debug/markers', MarkerArray, queue_size=1, latch=True)
        self.vehicle_pub = rospy.Publisher('/map_debug/vehicle', MarkerArray, queue_size=1)
        self.obstacle_pub = rospy.Publisher('/map_debug/obstacles', MarkerArray, queue_size=1)
        self.status_pub = rospy.Publisher('/map_debug/status', String, queue_size=1, latch=True)
        self.horizon_pub = rospy.Publisher('/map_debug/obstacle_horizon', MarkerArray, queue_size=1)
        self.ground_display_pub = rospy.Publisher('/map_debug/ground_points', PointCloud2, queue_size=1)
        self.lidar_scene_pub = rospy.Publisher('/map_debug/lidar_scene', MarkerArray, queue_size=1)
        self.lidar_obstacle_pub = rospy.Publisher('/map_debug/lidar_obstacles', MarkerArray, queue_size=1)
        self.local_path_pub = rospy.Publisher('/map_debug/local_path', NavPath, queue_size=1)
        self.map_local_path_pub = rospy.Publisher('/map_debug/map_local_path', NavPath, queue_size=1)
        self.local_path_marker_pub = rospy.Publisher('/map_debug/local_path_markers', Marker, queue_size=1)
        self.map_path_marker_pub = rospy.Publisher('/map_debug/map_local_path_markers', Marker, queue_size=1)
        self.used_points_pub = rospy.Publisher('/map_debug/used_points', PointCloud2, queue_size=1)
        self.discarded_points_pub = rospy.Publisher('/map_debug/discarded_points', PointCloud2, queue_size=1)
        self.publish_static()
        rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'),
                         PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(rospy.get_param('~valid_topic', '/localization/valid'),
                         Bool, self.on_valid, queue_size=1)
        rospy.Subscriber(rospy.get_param('~speed_topic', '/competition/ego_speed_kmh'),
                         Float64, self.on_speed, queue_size=1)
        rospy.Subscriber('/waypoint_debug/tracking_state', String,
                         self.on_tracking_state, queue_size=1)
        rospy.Subscriber(rospy.get_param('~obstacles_topic', '/waypoint_debug/obstacles'),
                         String, self.on_obstacles, queue_size=1)
        rospy.Subscriber('/waypoint_debug/lane_geometry', String,
                         self.on_lane_geometry, queue_size=1, buff_size=2**22)
        for owner in sorted({spec[0] for spec in SETTINGS.values()}):
            rospy.Subscriber('/waypoint_debug/avoidance_config/' + owner, String,
                             self.on_config, callback_args=owner, queue_size=1)
        rospy.Subscriber('/map_debug/avoidance_parameter_edit', String,
                         self.on_parameter_edit, queue_size=10)
        rospy.Subscriber('/waypoint_debug/obstacle_horizon', String,
                         self.on_horizon, queue_size=1)
        rospy.Subscriber('/waypoint_debug/local_path', NavPath,
                         self.on_local_path, queue_size=1)
        rospy.Subscriber('/leo/signal_status', String, self.on_guard_status, queue_size=1)
        rospy.Subscriber('/map_debug/perception_view', String, self.on_perception_view, queue_size=1)
        rospy.Subscriber('/leo/avoidance_status', String, self.on_ftg_status, queue_size=1)
        for suffix, kind, callback in (
                ('tracking_state', String, self.on_tracking_state),
                ('local_path', NavPath, self.on_local_path),
                ('obstacles', String, self.on_obstacles),
                ('lane_geometry', String, self.on_lane_geometry)):
            rospy.Subscriber('/leo/waypoint_debug/'+suffix, kind, callback,
                             callback_args='leo', queue_size=1)
        self.update_perception_subscriptions('raceline')
        rospy.Timer(rospy.Duration(0.1), self.refresh)
        self.log_project_counts()

    def log_project_counts(self):
        counts = ', '.join('S{} R{}={}점'.format(sector['number'], lane, len(points))
                           for sector in self.project['sectors']
                           for lane, points in sector['racelines'].items())
        rospy.loginfo('Map debug saved JSON: %s | %s', self.project_file, counts)

    def reload_project_if_changed(self):
        now = time.monotonic()
        if now < self.next_project_check:
            return
        self.next_project_check = now + 1.0
        try:
            stamp = self.project_file.stat().st_mtime_ns
            if stamp == self.project_stamp:
                return
            project = load_project(self.project_file)
            route, sectors = primary_route(project)
            xy = np.asarray([(point['x'], point['y']) for point in route])
            width = raceline_layout_settings(project, self.raceline_config)
            corridor = LaneCorridor(project_lane_geometry(project, width))
        except (OSError, ValueError, KeyError, TypeError) as error:
            rospy.logwarn_throttle(5, 'Could not reload saved map project: %s', error)
            return
        self.project, self.route, self.sectors, self.xy = project, route, sectors, xy
        self.project_stamp = stamp
        self.saved_lane_corridor = corridor
        if self.live_lane_geometry is None:
            self.lane_corridor = corridor
        self.publish_paths()
        self.log_project_counts()

    def publish_static(self):
        origin_x, origin_y, raster = self.aerial.raster(self.reference)
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = rospy.Time.now()
        grid.info.resolution = self.aerial.resolution
        grid.info.width = raster.shape[1]
        grid.info.height = raster.shape[0]
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = raster.ravel().astype('int8').tolist()
        self.map_pub.publish(grid)
        photo = marker('color_photo', 0, Marker.MESH_RESOURCE)
        photo.mesh_resource = self.aerial.color_mesh_resource(self.mesh_directory.name)
        photo.mesh_use_embedded_materials = True
        photo.scale.x = photo.scale.y = photo.scale.z = 1.0
        photo.color.r = photo.color.g = photo.color.b = photo.color.a = 1.0
        self.photo_pub.publish(photo)

        self.publish_paths()

    def publish_paths(self):

        markers = []
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.append(clear)
        for sector in self.project['sectors']:
            for lane_number in range(1, sector['lane_count'] + 1):
                points = sector['racelines'][str(lane_number)]
                if not points:
                    continue
                line = marker('sector_paths', sector['number'] * 10 + lane_number,
                              Marker.LINE_STRIP)
                line.scale.x = 2.0
                line.color.r, line.color.g, line.color.b, line.color.a = (
                    (0.0, 0.95, 1.0, 0.9) if lane_number == 1 else
                    (0.1, 0.55, 1.0, 0.75) if lane_number == 2 else
                    (0.1, 1.0, 0.4, 0.75))
                line.points = [Point(p['x'], p['y'], 0.5 + lane_number * 0.05) for p in points]
                if len(points) >= 2:
                    markers.append(line)
                dots = marker('sector_points', sector['number'] * 10 + lane_number,
                              Marker.POINTS)
                dots.scale.x = dots.scale.y = 0.9
                dots.color = line.color
                dots.points = [Point(p['x'], p['y'], 0.8 + lane_number * 0.05) for p in points]
                markers.append(dots)
        for index, name in enumerate(CHECKPOINT_ORDER):
            x, y = self.checkpoints[name]['position'][:2]
            box = marker('checkpoint_boxes', index, Marker.CUBE)
            box.pose.position.x, box.pose.position.y, box.pose.position.z = x, y, 2.5
            box.scale.x = 26.0 if name == 'start' else 17.0
            box.scale.y, box.scale.z = 17.0, 1.0
            box.color.r, box.color.g, box.color.b, box.color.a = 0.0, 0.55, 0.02, 0.45
            markers.append(box)
            label = marker('checkpoint_numbers', index, Marker.TEXT_VIEW_FACING)
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, 3.7
            label.scale.z = 11.0
            label.text = name.upper()
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 0.78
            markers.append(label)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def on_pose(self, message):
        p, q = message.pose.position, message.pose.orientation
        age = rospy.Time.now().to_sec()-message.header.stamp.to_sec()
        if not -.1 <= age <= .5 or message.header.frame_id != 'map' or not all(math.isfinite(v) for v in
                (p.x, p.y, q.x, q.y, q.z, q.w)):
            return
        norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if not 0.5 <= norm <= 1.5:
            return
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.pose = (p.x, p.y, yaw)
        self.pose_time = time.monotonic()

    def on_valid(self, message):
        self.valid = message.data

    def on_speed(self, message):
        if math.isfinite(message.data):
            self.speed = message.data
            self.speed_time = time.monotonic()

    def on_tracking_state(self, message, source='raceline'):
        try:
            state = json.loads(message.data)
            if (type(state['valid']) is not bool or
                    type(state['raceline']) is not int or state['raceline'] not in (1, 2, 3) or
                    (state['sector'] is not None and
                     (type(state['sector']) is not int or not 1 <= state['sector'] <= 15))):
                return
            if 'stamp' in state and (not math.isfinite(float(state['stamp'])) or
                    not -.1 <= rospy.Time.now().to_sec()-state['stamp'] <= .5):
                return
        except (ValueError, KeyError, TypeError):
            return
        snapshot = (state, time.monotonic())
        self.source_snapshots[source]['tracking'] = snapshot
        if source == 'raceline':
            self.tracking = snapshot

    def current_tracking(self, now, require_valid=True):
        tracking = (self.tracking if self.active_source == 'raceline' else
                    self.source_snapshots.get(self.active_source, {}).get('tracking'))
        if (tracking is None or now - tracking[1] >= .5
                or require_valid and not tracking[0]['valid']
                or 'stamp' in tracking[0] and
                not -.1 <= rospy.Time.now().to_sec()-tracking[0]['stamp'] <= .5):
            return None
        return tracking[0]

    def on_obstacles(self, message, source='raceline'):
        try:
            data = json.loads(message.data)
            stamp = float(data['stamp'])
            age = rospy.Time.now().to_sec()-stamp
            if data['frame_id'] != 'map' or not math.isfinite(stamp) or not -.1 <= age <= .5:
                return
            obstacles, identifiers = [], set()
            for item in data['obstacles']:
                identifier = item['id']
                values = [float(item[key]) for key in ('x', 'y', 'radius', 'vx', 'vy')]
                kind = item['kind']
                if (type(identifier) is not int or not 0 <= identifier < 2**31 or
                        identifier in identifiers or not all(math.isfinite(v) for v in values) or
                        not .1 <= values[2] <= (100. if source in ('route', 'ftg')
                                               else 15. if source == 'raceline' else 6.)
                        or kind not in ('dynamic', 'static', 'unknown')):
                    return
                # Detector rectangle; older streams carry only the circle.
                box = [float(item.get(key, 0.)) for key in ('length', 'width', 'yaw')]
                if not all(math.isfinite(v) for v in box) or min(box[:2]) < 0.:
                    return
                if box[0] <= 0. or box[1] <= 0.:
                    box = [2.*values[2], 2.*values[2], 0.]
                identifiers.add(identifier)
                lane = (int(item.get('raceline', 0)),
                        bool(item.get('lane_stable', False)),
                        bool(item.get('lane_boundary_crossing', True)))
                # Optional diagnostic bounds already use the cloud's map Z.
                bounds = None
                try:
                    lower, upper = float(item['z_min']), float(item['z_max'])
                    if math.isfinite(lower) and math.isfinite(upper) and lower <= upper:
                        bounds = lower, upper
                except (KeyError, ValueError, TypeError, OverflowError):
                    pass
                obstacles.append((identifier, *values, kind, *box, *lane, bounds))
        except (ValueError, KeyError, TypeError, OverflowError):
            return
        corridor = (self.lane_corridor if source == 'raceline' else
                    self.source_snapshots[source].get('corridor', self.saved_lane_corridor))
        if corridor is None:
            return
        if source not in ('route', 'ftg'):
            mask = corridor.contains([(o[1], o[2]) for o in obstacles])
            obstacles = [o for o, inside in zip(obstacles, mask) if inside]
        snapshot = (obstacles, time.monotonic(), stamp)
        if hasattr(self, 'source_snapshots'):
            self.source_snapshots[source]['obstacles'] = snapshot
        if source == 'raceline':
            self.obstacle_snapshot = snapshot

    def on_lane_geometry(self, message, source='raceline'):
        try:
            geometry = json.loads(message.data)
            corridor = LaneCorridor(geometry)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            rospy.logwarn_throttle(5., 'Invalid debug lane geometry: %s', error)
            return
        self.source_snapshots[source]['corridor'] = corridor
        self.source_snapshots[source]['obstacles'] = None
        if source == 'raceline':
            self.live_lane_geometry = geometry
            self.lane_corridor = corridor
            self.obstacle_snapshot = None
        # Live geometry is only for matching the detector's obstacle filter.
        # Visible map paths always come from DEBUG_PROJECT_FILE.

    def on_config(self, message, owner):
        try:
            data = json.loads(message.data)
            if not isinstance(data['values'], dict) or not data['node'].startswith('/'):
                return
            for name, value in data['values'].items():
                if name in SETTINGS:
                    validated_value(name, value)
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        with self.config_lock:
            self.config_snapshots[owner] = (data, time.monotonic())

    def on_parameter_edit(self, message):
        try:
            data = json.loads(message.data)
            name = data['name']
            value = validated_value(name, data['value'])
            owner = SETTINGS[name][0]
            with self.config_lock:
                snapshot = self.config_snapshots.get(owner)
            node = snapshot[0]['node'] if snapshot else '/' + owner
            rospy.set_param(node + '/' + name, value)
            with self.config_lock:
                self.pending_parameters[name] = (value, time.monotonic())
                self.parameter_error = ''
        except (ValueError, KeyError, TypeError, OSError, rospy.ROSException) as error:
            with self.config_lock:
                self.parameter_error = str(error)

    def parameter_status(self, now):
        rows = []
        with self.config_lock:
            for name, spec in SETTINGS.items():
                owner, default, _, unit, description = spec
                snapshot = self.config_snapshots.get(owner)
                fresh = snapshot is not None and now-snapshot[1] < 1.
                value = snapshot[0]['values'].get(name, default) if snapshot else default
                pending = self.pending_parameters.get(name)
                if pending and fresh and snapshot[1] >= pending[1] and value == pending[0]:
                    del self.pending_parameters[name]
                    pending = None
                error = snapshot[0].get('errors', {}).get(name, '') if fresh else ''
                rows.append(dict(name=name, value=value, unit=unit, description=description,
                                 applied=fresh, editable=True,
                                 pending_value=pending[0] if pending else None, error=error))
            error = self.parameter_error
            width_applied = all(
                owner in self.config_snapshots and now-self.config_snapshots[owner][1] < 1.
                and self.config_snapshots[owner][0]['values'].get('raceline_width')
                == self.saved_lane_corridor.width
                for owner in ('lidar_obstacles', 'gps_waypoint_follower'))
        rows.append(dict(name='raceline_width', value=self.saved_lane_corridor.width, unit='m',
                         description='튜너 저장값 자동 연동 · 튜너에서 변경 후 저장',
                         applied=width_applied, editable=False, pending_value=None, error=''))
        return rows, error

    def on_horizon(self, message):
        try:
            data = json.loads(message.data)
            points = np.asarray(data['points'], dtype=float).reshape((-1, 2))
            if (data['frame_id'] != 'map' or not np.isfinite(points).all()
                    or not math.isfinite(float(data['stamp']))
                    or not math.isfinite(float(data['horizon_m'])) or data['horizon_m'] <= 0.
                    or not math.isfinite(float(data['half_width_m'])) or data['half_width_m'] <= 0.):
                return
        except (ValueError, KeyError, TypeError, OverflowError):
            return
        self.horizon_snapshot = (data, points, time.monotonic())

    def publish_horizon(self, now):
        snapshot = self.horizon_snapshot
        if snapshot is None or now-snapshot[2] > .5:
            return
        data, points, _ = snapshot
        age = rospy.Time.now().to_sec()-data['stamp']
        if not -.1 <= age <= .5 or len(points) < 2:
            return
        directions = np.diff(points, axis=0)
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths < 1e-9):
            return
        normals = np.column_stack((-directions[:, 1], directions[:, 0])) / lengths[:, None]
        normals = np.vstack((normals, normals[-1])) * data['half_width_m']
        left, right = points+normals, points-normals
        area = marker('obstacle_horizon_area', 0, Marker.TRIANGLE_LIST)
        area.scale.x = area.scale.y = area.scale.z = 1.
        area.color.r, area.color.g, area.color.b, area.color.a = .95, .35, 1., .25
        for i in range(len(points)-1):
            area.points.extend(Point(float(p[0]), float(p[1]), 1.1) for p in
                               (left[i], right[i], left[i+1], right[i], right[i+1], left[i+1]))
        border = marker('obstacle_horizon_border', 0, Marker.LINE_STRIP)
        border.scale.x = .25
        border.color.r, border.color.g, border.color.b, border.color.a = .95, .35, 1., 1.
        outline = np.vstack((left, right[::-1], left[:1]))
        border.points = [Point(float(p[0]), float(p[1]), 1.2) for p in outline]
        label = marker('obstacle_horizon_label', 0, Marker.TEXT_VIEW_FACING)
        label.pose.position.x, label.pose.position.y = points[-1]
        label.pose.position.z = 3.
        label.scale.z = 2.
        label.color = border.color
        actual = float(lengths.sum())
        label.text = 'obstacle_horizon {:.1f} m'.format(data['horizon_m'])
        if actual < data['horizon_m']-.1:
            label.text += '\nSaved path {:.1f} m'.format(actual)
        for item in (area, border, label):
            item.lifetime = rospy.Duration(.5)
        self.horizon_pub.publish(MarkerArray(markers=[area, border, label]))

    def publish_obstacles(self, now):
        source = self.perception_source() if hasattr(self, 'active_source') else 'raceline'
        snapshot = (self.obstacle_snapshot if source == 'raceline' else
                    self.source_snapshots.get(source, {}).get('obstacles'))
        visible = (snapshot is not None and now-snapshot[1] <= .5 and
                   -.1 <= rospy.Time.now().to_sec()-snapshot[2] <= .5)
        obstacles = snapshot[0] if visible else []
        corridor = (self.lane_corridor if source == 'raceline' else
                    self.source_snapshots.get(source, {}).get('corridor', self.saved_lane_corridor))
        if corridor is None:
            obstacles = []
        elif obstacles and source not in ('route', 'ftg'):
            mask = corridor.contains([(o[1], o[2]) for o in obstacles])
            obstacles = [o for o, inside in zip(obstacles, mask) if inside]
        identifiers = {o[0] for o in obstacles}
        markers = []
        vertical_bounds = {}
        scan_points = self.obstacle_scan_points(source, snapshot[2], now) if obstacles else None
        namespaces = ('obstacle_boxes', 'obstacle_edges', 'obstacle_labels')
        for identifier in self.obstacle_marker_ids-identifiers:
            for namespace in namespaces:
                deleted = marker(namespace, identifier, Marker.CUBE)
                deleted.action = Marker.DELETE
                markers.append(deleted)
        for identifier, x, y, radius, vx, vy, kind, length, width, yaw, lane, stable, crossing, bounds in obstacles:
            if bounds is None and scan_points is not None:
                bounds = obstacle_z_bounds(scan_points, x, y, length, width, yaw)
            vertical_bounds[identifier] = bounds
            rgb = ((1., .15, .1) if kind == 'dynamic' else
                   (.1, .65, 1.) if kind == 'static' else (1., .85, .1))
            # The detector's lane-aligned rectangle; height is for visibility.
            height, base = 1.5, 3.
            box = marker('obstacle_boxes', identifier, Marker.CUBE)
            box.pose.position.x, box.pose.position.y = x, y
            box.pose.position.z = base+height*.5
            box.pose.orientation.z, box.pose.orientation.w = math.sin(yaw*.5), math.cos(yaw*.5)
            box.scale.x, box.scale.y = length, width
            box.scale.z = height
            box.color.r, box.color.g, box.color.b, box.color.a = (*rgb, .35)
            box.lifetime = rospy.Duration(.5)
            markers.append(box)
            edges = marker('obstacle_edges', identifier, Marker.LINE_LIST)
            edges.scale.x = .18
            edges.color.r, edges.color.g, edges.color.b, edges.color.a = (*rgb, 1.)
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [Point(x+c*sx*length*.5-s*sy*width*.5, y+s*sx*length*.5+c*sy*width*.5, z)
                       for z in (base, base+height)
                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            pairs = [(i, (i+1) % 4) for i in range(4)]
            pairs += [(a+4, b+4) for a, b in pairs]+[(i, i+4) for i in range(4)]
            edges.points = [corners[i] for pair in pairs for i in pair]
            edges.lifetime = rospy.Duration(.5)
            markers.append(edges)
            label = marker('obstacle_labels', identifier, Marker.TEXT_VIEW_FACING)
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, base+height+1.
            label.scale.z = 2.
            label.color.r, label.color.g, label.color.b, label.color.a = (*rgb, 1.)
            name = {'dynamic': 'DYNAMIC', 'static': 'STATIC', 'unknown': 'UNKNOWN'}[kind]
            label.text = '{} #{}'.format(name, identifier)
            if lane in (1, 2, 3):
                label.text += '\nR{}{}'.format(lane, '' if stable else ' ?')
                if crossing:
                    label.text += ' EDGE'
            if kind == 'dynamic':
                label.text += '\n{:.1f} km/h'.format(math.hypot(vx, vy)*3.6)
            label.lifetime = rospy.Duration(.5)
            markers.append(label)
        if markers:
            self.obstacle_pub.publish(MarkerArray(markers=markers))
            # Preserve the aerial overlay. The 3D view uses only observed Z
            # from this source and scan; no measured height means a footprint.
            lidar_markers = copy.deepcopy(markers)
            for original, item in zip(markers, lidar_markers):
                if item.action == Marker.DELETE:
                    continue
                bounds = vertical_bounds[item.id]
                lower, upper = bounds if bounds is not None else (self.ground_z+.03,)*2
                if item.type == Marker.LINE_LIST:
                    item.pose.position.z = 0.
                    item.scale.x = .06
                    # Each corner occurs three times in the line list. Assign
                    # fresh points: shifting aliased corners in place sinks
                    # them three times while the cube moves only once.
                    item.points = [Point(p.x, p.y, lower+(p.z-3.)/1.5*(upper-lower))
                                   for p in original.points]
                elif item.type == Marker.TEXT_VIEW_FACING:
                    item.pose.position.z = upper+.3
                    item.scale.z = .65
                elif item.type == Marker.CUBE:
                    item.pose.position.z = (lower+upper)*.5
                    item.scale.z = upper-lower
                    item.color.a = .18
                    if bounds is None or upper <= lower:
                        item.action = Marker.DELETE
            self.lidar_obstacle_pub.publish(MarkerArray(markers=lidar_markers))
        self.obstacle_marker_ids = identifiers

    def obstacle_scan_points(self, source, stamp, now):
        """Match height to the obstacle scan, never a different source/scan."""
        snapshots = getattr(self, 'source_snapshots', {}).get(source, {})
        for cloud, received, points in reversed(snapshots.get('used_history', [])):
            if now-received <= .5 and abs(cloud.header.stamp.to_sec()-stamp) < 1e-6:
                return points
        return None

    def on_local_path(self, message, source='raceline'):
        age = rospy.Time.now().to_sec()-message.header.stamp.to_sec()
        if (message.header.frame_id != 'map' or not -.1 <= age <= .5
                or any(p.header.frame_id not in ('', 'map') or
                       not all(math.isfinite(v) for v in
                               (p.pose.position.x, p.pose.position.y, p.pose.position.z))
                       for p in message.poses)):
            return
        snapshot = (copy.deepcopy(message), time.monotonic())
        self.source_snapshots[source]['path'] = snapshot
        if source == 'raceline':
            self.local_path_snapshot = snapshot

    def publish_local_path(self, now):
        snapshot = (self.local_path_snapshot if self.active_source == 'raceline' else
                    self.source_snapshots.get(self.active_source, {}).get('path'))
        fresh = (snapshot is not None and now-snapshot[1] <= .5 and
                 self.current_tracking(now) is not None and
                 -.1 <= rospy.Time.now().to_sec()-snapshot[0].header.stamp.to_sec() <= .5)
        if fresh:
            path = copy.deepcopy(snapshot[0])
            # Same actual XY path in both views. The aerial map's saved paths
            # sit above the road, so raise only its display copy above them.
            for pose in path.poses:
                pose.pose.position.z = self.ground_z+.07
            self.local_path_pub.publish(path)
            map_path = copy.deepcopy(path)
            for pose in map_path.poses:
                pose.pose.position.z = 1.4
            self.map_local_path_pub.publish(map_path)
            self.publish_path_markers(path, map_path)
            self.local_path_visible = bool(path.poses)
        elif self.local_path_visible:
            path = NavPath(header=Header(stamp=rospy.Time.now(), frame_id='map'))
            self.local_path_pub.publish(path)
            self.map_local_path_pub.publish(path)
            self.publish_path_markers(path, path)
            self.local_path_visible = False

    def publish_path_markers(self, lidar_path, map_path):
        # Explicit marker deletion avoids asking RViz's billboard Path
        # display to allocate a zero-element Ogre index buffer on stops.
        for path, publisher, width in ((lidar_path, self.local_path_marker_pub, .15),
                                       (map_path, self.map_path_marker_pub, .6)):
            line = marker('actual_local_path', 0, Marker.LINE_STRIP)
            line.header = path.header
            if len(path.poses) < 2:
                line.action = Marker.DELETE
            else:
                line.scale.x = width
                line.color.r, line.color.g, line.color.b, line.color.a = 1., .78, 0., 1.
                line.points = [copy.deepcopy(p.pose.position) for p in path.poses]
                line.lifetime = rospy.Duration(.5)
            publisher.publish(line)

    def publish_lidar_scene(self, pose):
        if pose is None:
            if self.lidar_scene_visible:
                clear = marker('lidar_scene', 0, Marker.CUBE)
                clear.action = Marker.DELETEALL
                self.lidar_scene_pub.publish(MarkerArray(markers=[clear]))
                self.lidar_scene_visible = False
            return
        x, y, yaw = pose
        self.tf_broadcaster.sendTransform((x, y, self.ground_z),
            (0., 0., math.sin(yaw*.5), math.cos(yaw*.5)), rospy.Time.now(),
            'map_debug_vehicle', 'map')
        corridor = (self.lane_corridor if self.active_source == 'raceline' else
                    self.source_snapshots.get(self.active_source, {}).get('corridor', self.saved_lane_corridor))
        edges, centers = nearby_lane_edges(corridor, pose)
        markers = []
        for namespace, points, rgb, width in (
                ('lane_boundaries', edges, (1., .85, .15), .07),
                ('lane_centers', centers, (.2, .65, 1.), .045)):
            line = marker(namespace, 0, Marker.LINE_LIST)
            line.scale.x = width
            line.color.r, line.color.g, line.color.b, line.color.a = (*rgb, .85)
            line.points = [Point(float(p[0]), float(p[1]), self.ground_z+.03) for p in points]
            markers.append(line)
        geometry = self.vehicle_geometry
        corners = vehicle_box_corners(pose, geometry['front'], geometry['rear'],
            geometry['half_width'], self.ground_z, self.vehicle_height or 0.)
        box = marker('vehicle_dimensions', 0, Marker.LINE_LIST)
        box.scale.x = .09
        box.color.r, box.color.g, box.color.b, box.color.a = 1., .75, .1, 1.
        box.points = [Point(*corners[i]) for pair in BOX_EDGE_PAIRS for i in pair]
        markers.append(box)
        nose = marker('vehicle_forward', 0, Marker.ARROW)
        nose.scale.x, nose.scale.y, nose.scale.z = .15, .35, .35
        nose.color = box.color
        nose.points = [Point(x, y, self.ground_z+.15),
                       Point(x+2.*math.cos(yaw), y+2.*math.sin(yaw), self.ground_z+.15)]
        markers.append(nose)
        tracking = self.current_tracking(time.monotonic()) or {}
        target = tracking.get('pursuit_target')
        if isinstance(target, dict) and all(type(target.get(k)) in (int, float)
                and math.isfinite(target[k]) for k in ('x', 'y')):
            sphere = marker('pursuit_target', 0, Marker.SPHERE)
            sphere.pose.position.x, sphere.pose.position.y = target['x'], target['y']
            sphere.pose.position.z = .4
            sphere.scale.x = sphere.scale.y = sphere.scale.z = .6
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1., 1., 0., 1.
            markers.append(sphere)
        for item in markers:
            item.lifetime = rospy.Duration(.5)
        self.lidar_scene_pub.publish(MarkerArray(markers=markers))
        self.lidar_scene_visible = True

    def on_ground_points(self, message):
        if message.header.frame_id != 'map':
            return
        # Map background is at z=-0.1. Project only this 2D display above it;
        # original physical heights remain on /waypoint_debug/ground_points.
        points = [(x, y, .15) for x, y, _ in point_cloud2.read_points(
            message, field_names=('x', 'y', 'z'), skip_nans=True)]
        self.ground_display_pub.publish(point_cloud2.create_cloud_xyz32(
            Header(stamp=message.header.stamp, frame_id='map'), points))

    def on_ground_status(self, message, source='raceline'):
        try:
            data = json.loads(message.data)
            if (data.get('frame_id') != 'map' or not math.isfinite(float(data['stamp'])) or
                    (data.get('mode') not in ('ground_filtered', 'ground_unconfirmed', 'imu_missing_or_stale',
                        'waiting_cloud', 'waiting_lane_geometry', 'wrong_cloud_frame', 'cloud_time_mismatch', 'waiting_pose',
                        'local_filtered', 'route_filtered', 'ftg_filtered')
                     and data.get('debug_valid') is not False)):
                return
            for key in ('pitch_deg', 'roll_deg', 'imu_skew_s', 'residual_p95_m'):
                if data.get(key) is not None and not math.isfinite(float(data[key])):
                    return
        except (ValueError, TypeError, KeyError, AttributeError):
            return
        with self.config_lock:
            snapshot = (data, time.monotonic())
            self.source_snapshots[source]['ground'] = snapshot
            if source == 'raceline':
                self.ground_snapshot = snapshot
        if source in ('route', 'ftg'):
            self.on_obstacles(String(data=json.dumps(dict(
                stamp=data['stamp'], frame_id='map', obstacles=data.get('obstacles', [])))), source)

    def ground_status(self, now):
        with self.config_lock:
            source = self.perception_source()
            snapshot = (self.ground_snapshot if source == 'raceline' else
                        self.source_snapshots.get(source, {}).get('ground'))
        if snapshot is None or now-snapshot[1] > .5:
            return None
        age = rospy.Time.now().to_sec()-snapshot[0]['stamp']
        return snapshot[0] if -.1 <= age <= .5 else None

    def on_guard_status(self, message):
        try:
            data = json.loads(message.data)
            if not isinstance(data.get('state'), str):
                return
        except (ValueError, TypeError, AttributeError):
            return
        self.guard_snapshot = (data, time.monotonic())

    def on_ftg_status(self, message):
        try:
            data = json.loads(message.data)
            if type(data.get('valid')) is not bool:
                return
        except (ValueError, TypeError, AttributeError):
            return
        self.source_snapshots['ftg']['decision'] = (data, time.monotonic())

    def on_perception_view(self, message):
        if message.data in ('auto', 'collision'):
            self.perception_view = message.data

    def select_source(self, now):
        if self.guard_snapshot is None:
            return 'raceline'
        if now-self.guard_snapshot[1] > .5:
            return None
        if self.guard_snapshot[0]['state'] == 'hold':
            return None
        return 'raceline' if self.guard_snapshot[0]['state'] == 'passthrough' else 'leo'

    def perception_source(self):
        if self.active_source != 'leo':
            return self.active_source
        tracking = self.current_tracking(time.monotonic(), require_valid=False) or {}
        if (self.perception_view == 'collision' and
                tracking.get('planner_mode') == 'route_and_local_avoidance'):
            return 'leo'
        if tracking.get('planner_mode') == 'external_blackout_target':
            return 'ftg'
        if tracking.get('local_path_source') == 'route_avoidance':
            return 'route'
        return 'leo'

    def on_debug_cloud(self, message, source_kind):
        if message.header.frame_id != 'map':
            return
        source, kind = source_kind
        received = time.monotonic()
        self.source_snapshots[source][kind] = (message, received)
        if kind == 'used':
            if (message.point_step == 12 and message.row_step == 12*message.width
                    and [(f.name, f.offset, f.datatype, f.count) for f in message.fields]
                    == [(name, i*4, 7, 1) for i, name in enumerate(('x', 'y', 'z'))]):
                dtype = '>f4' if message.is_bigendian else '<f4'
                points = np.frombuffer(message.data, dtype=dtype).reshape(-1, 3)
            else:
                points = np.asarray(list(point_cloud2.read_points(
                    message, field_names=('x', 'y', 'z'), skip_nans=True))).reshape(-1, 3)
            # Cloud and obstacle callbacks can arrive in either order. Keep a
            # short bounded history to pair the original scan at render time.
            history = self.source_snapshots[source].get('used_history', [])
            self.source_snapshots[source]['used_history'] = (
                history+[(message, received, points)])[-4:]

    def update_perception_subscriptions(self, source):
        if source == self.perception_subscription_source:
            return
        for subscriber in self.perception_subscribers:
            subscriber.unregister()
        self.perception_subscribers = []
        self.perception_subscription_source = source
        if source is None:
            return
        prefix, status_name = {
            'raceline': ('/waypoint_debug/', 'ground_status'),
            'leo': ('/leo/waypoint_debug/', 'ground_status'),
            'route': ('/leo/route_debug/', 'status'),
            'ftg': ('/leo/ftg_debug/', 'status')}[source]
        # Only the displayed pipeline needs to build diagnostic clouds. This
        # keeps inactive perception workers out of the rendering workload.
        for kind, suffix in (('used', 'obstacle_points'), ('discarded', 'discarded_points')):
            self.perception_subscribers.append(rospy.Subscriber(prefix+suffix, PointCloud2,
                self.on_debug_cloud, callback_args=(source, kind), queue_size=1, buff_size=2**24))
        self.perception_subscribers.append(rospy.Subscriber(prefix+status_name, String,
            self.on_ground_status, callback_args=source, queue_size=1))

    def publish_debug_clouds(self, now):
        source = self.perception_source()
        for kind, publisher in (('used', self.used_points_pub), ('discarded', self.discarded_points_pub)):
            snapshot = self.source_snapshots.get(source, {}).get(kind)
            if (snapshot is not None and now-snapshot[1] <= .5 and
                    -.1 <= rospy.Time.now().to_sec()-snapshot[0].header.stamp.to_sec() <= .5):
                publisher.publish(snapshot[0])
            else:
                publisher.publish(point_cloud2.create_cloud_xyz32(
                    Header(stamp=rospy.Time.now(), frame_id='map'), []))

    def refresh(self, _event):
        self.reload_project_if_changed()
        now = time.monotonic()
        self.active_source = self.select_source(now)
        self.update_perception_subscriptions(self.perception_source())
        self.publish_debug_clouds(now)
        self.publish_obstacles(now)
        self.publish_local_path(now)
        parameters, parameter_error = self.parameter_status(now)
        tracking = self.current_tracking(now)
        recent_state = tracking
        decision = dict(self.current_tracking(now, require_valid=False) or {})
        ftg = self.source_snapshots['ftg'].get('decision')
        if (decision.get('planner_mode') == 'external_blackout_target' and
                ftg is not None and now-ftg[1] <= .5):
            decision['ftg'] = ftg[0]
            decision['planner_reason'] = ftg[0].get('reason', '')
            decision['planner_state'] = ftg[0].get('avoidance_phase', '')
        raceline = None
        available = (self.valid and self.pose is not None and
                     now - self.pose_time < 1.0)
        pose = self.pose
        if tracking is not None and tracking.get('frame_id') == 'map':
            controller_pose = tuple(tracking.get(key) for key in ('x', 'y', 'yaw'))
            if all(type(v) in (int, float) and math.isfinite(v) for v in controller_pose):
                pose = controller_pose
                available = True
        speed = (self.speed if self.speed_time is not None and
                 now - self.speed_time < 1.0 else None)
        if available:
            x, y, yaw = pose
            index = int(np.argmin(np.sum((self.xy - (x, y)) ** 2, axis=1)))
            sector = self.sectors[index]
            if tracking is not None:
                sector = tracking['sector'] or sector
                raceline = tracking['raceline']
            arrow = marker('vehicle', 0, Marker.ARROW)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 2.0, 4.0, 3.0
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.75, 0.0, 1.0
            arrow.points = [Point(x, y, 3.0),
                            Point(x + 14*math.cos(yaw), y + 14*math.sin(yaw), 3.0)]
            label = marker('vehicle_heading', 0, Marker.TEXT_VIEW_FACING)
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, 7.0
            label.scale.z = 5.0
            label.text = '{:.1f}°'.format(math.degrees(yaw))
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            self.vehicle_pub.publish(MarkerArray(markers=[arrow, label]))
            self.last_vehicle_visible = True
        else:
            x = y = yaw = sector = None
            if self.last_vehicle_visible:
                deleted = marker('vehicle', 0, Marker.ARROW)
                deleted.action = Marker.DELETEALL
                self.vehicle_pub.publish(MarkerArray(markers=[deleted]))
                self.last_vehicle_visible = False
        self.publish_lidar_scene(pose if available else None)
        self.status_pub.publish(String(data=json.dumps({
            'available': available, 'speed_kmh': speed, 'x': x, 'y': y,
            'heading_deg': math.degrees(yaw) if yaw is not None else None,
            'sector': sector, 'raceline': raceline,
            'driving_state': recent_state.get('driving_state', 'RACELINE') if recent_state else None,
            'obstacle_kind': recent_state.get('obstacle_kind', '') if recent_state else '',
            'obstacle_distance_m': recent_state.get('obstacle_distance_m') if recent_state else None,
            'local_lane': recent_state.get('local_lane') if recent_state else None,
            'preferred_raceline': recent_state.get('preferred_raceline') if recent_state else None,
            'current_raceline': recent_state.get('current_raceline') if recent_state else None,
            'preferred_raceline_reason': recent_state.get('preferred_raceline_reason', '') if recent_state else '',
            'waypoint_index': recent_state.get('waypoint_index') if recent_state else None,
            'project_file': str(self.project_file),
            'raceline_width': self.saved_lane_corridor.width,
            'avoidance_parameters': parameters, 'parameter_error': parameter_error,
            'ground_filter': self.ground_status(now),
            'decision': decision,
            'controller_source': self.active_source,
            'perception_source': self.perception_source(),
            'guard': self.guard_snapshot[0] if self.guard_snapshot and now-self.guard_snapshot[1] <= .5 else {},
            'vehicle_dimensions': dict(self.vehicle_geometry, height=self.vehicle_height),
            'lane_change_reason': recent_state.get('lane_change_reason', '') if recent_state else ''})))


if __name__ == '__main__':
    rospy.init_node('map_debug')
    MapDebug()
    rospy.spin()
