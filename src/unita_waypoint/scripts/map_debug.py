#!/usr/bin/env python3
"""K-City map, saved route, vehicle state and detected obstacle boxes."""

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(rospkg.RosPack().get_path('unita_waypoint'))
DEBUG_PROJECT_FILE = Path('/home/unita/unita_ws/src/unita_waypoint/config/map_tunner_project.json')
sys.path.insert(0, str(ROOT / 'scripts'))
from map_tunner_core import (CHECKPOINT_ORDER, load_checkpoints, load_csv,
                             load_project, primary_route, raceline_layout_settings)
from map_tunner_map import AerialMap
from lane_corridor import LaneCorridor, project_lane_geometry


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
        width = raceline_layout_settings(self.project, self.raceline_config, self.raceline_width_override)
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
        self.photo_pub = rospy.Publisher('/map_debug/photo', Marker, queue_size=1, latch=True)
        self.map_pub = rospy.Publisher('/map_debug/map', OccupancyGrid, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher('/map_debug/markers', MarkerArray, queue_size=1, latch=True)
        self.vehicle_pub = rospy.Publisher('/map_debug/vehicle', MarkerArray, queue_size=1)
        self.obstacle_pub = rospy.Publisher('/map_debug/obstacles', MarkerArray, queue_size=1)
        self.status_pub = rospy.Publisher('/map_debug/status', String, queue_size=1, latch=True)
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
            width = raceline_layout_settings(project, self.raceline_config, self.raceline_width_override)
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
        # The same saved Sector 11 endpoint drives the follower's finish rule.
        endpoint = self.project['sectors'][10]['racelines']['1'][-1]
        finish = marker('cp11_finish_line', 0, Marker.LINE_LIST)
        finish.scale.x = .6
        finish.color.r = finish.color.g = finish.color.b = finish.color.a = 1.
        finish.points = [Point(endpoint['x']-20., endpoint['y'], 1.2),
                         Point(endpoint['x']+20., endpoint['y'], 1.2)]
        markers.append(finish)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def on_pose(self, message):
        p, q = message.pose.position, message.pose.orientation
        if message.header.frame_id != 'map' or not all(math.isfinite(v) for v in
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

    def on_tracking_state(self, message):
        try:
            state = json.loads(message.data)
            if (type(state['valid']) is not bool or
                    type(state['raceline']) is not int or state['raceline'] not in (1, 2, 3) or
                    (state['sector'] is not None and
                     (type(state['sector']) is not int or not 1 <= state['sector'] <= 15))):
                return
        except (ValueError, KeyError, TypeError):
            return
        self.tracking = (state, time.monotonic())

    def current_tracking(self, now):
        tracking = self.tracking
        if tracking is None or now - tracking[1] >= 1.0 or not tracking[0]['valid']:
            return None
        return tracking[0]

    def on_obstacles(self, message):
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
                        not .1 <= values[2] <= 6. or kind not in ('dynamic', 'static', 'unknown')):
                    return
                identifiers.add(identifier)
                obstacles.append((identifier, *values, kind))
        except (ValueError, KeyError, TypeError, OverflowError):
            return
        corridor = self.lane_corridor
        if corridor is None:
            return
        mask = corridor.contains([(o[1], o[2]) for o in obstacles])
        obstacles = [o for o, inside in zip(obstacles, mask) if inside]
        self.obstacle_snapshot = (obstacles, time.monotonic(), stamp)

    def on_lane_geometry(self, message):
        try:
            geometry = json.loads(message.data)
            corridor = LaneCorridor(geometry)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            rospy.logwarn_throttle(5., 'Invalid debug lane geometry: %s', error)
            return
        self.live_lane_geometry = geometry
        self.lane_corridor = corridor
        self.obstacle_snapshot = None
        # Live geometry is only for matching the detector's obstacle filter.
        # Visible map paths always come from DEBUG_PROJECT_FILE.

    def publish_obstacles(self, now):
        snapshot = self.obstacle_snapshot
        visible = (snapshot is not None and now-snapshot[1] <= .5 and
                   -.1 <= rospy.Time.now().to_sec()-snapshot[2] <= .5)
        obstacles = snapshot[0] if visible else []
        corridor = self.lane_corridor
        if corridor is None:
            obstacles = []
        elif obstacles:
            mask = corridor.contains([(o[1], o[2]) for o in obstacles])
            obstacles = [o for o, inside in zip(obstacles, mask) if inside]
        identifiers = {o[0] for o in obstacles}
        markers = []
        namespaces = ('obstacle_boxes', 'obstacle_edges', 'obstacle_labels')
        for identifier in self.obstacle_marker_ids-identifiers:
            for namespace in namespaces:
                deleted = marker(namespace, identifier, Marker.CUBE)
                deleted.action = Marker.DELETE
                markers.append(deleted)
        for identifier, x, y, radius, vx, vy, kind in obstacles:
            rgb = ((1., .15, .1) if kind == 'dynamic' else
                   (.1, .65, 1.) if kind == 'static' else (1., .85, .1))
            # The tracker reports a 2D collision radius, not an object mesh.
            # Enclose that footprint in a box; height is for map visibility.
            width, height, base = 2.*radius, 1.5, 3.
            box = marker('obstacle_boxes', identifier, Marker.CUBE)
            box.pose.position.x, box.pose.position.y = x, y
            box.pose.position.z = base+height*.5
            box.scale.x = box.scale.y = width
            box.scale.z = height
            box.color.r, box.color.g, box.color.b, box.color.a = (*rgb, .35)
            box.lifetime = rospy.Duration(.5)
            markers.append(box)
            edges = marker('obstacle_edges', identifier, Marker.LINE_LIST)
            edges.scale.x = .18
            edges.color.r, edges.color.g, edges.color.b, edges.color.a = (*rgb, 1.)
            corners = [Point(x+sx*radius, y+sy*radius, z)
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
            if kind == 'dynamic':
                label.text += '\n{:.1f} km/h'.format(math.hypot(vx, vy)*3.6)
            label.lifetime = rospy.Duration(.5)
            markers.append(label)
        if markers:
            self.obstacle_pub.publish(MarkerArray(markers=markers))
        self.obstacle_marker_ids = identifiers

    def refresh(self, _event):
        self.reload_project_if_changed()
        now = time.monotonic()
        self.publish_obstacles(now)
        tracking = self.current_tracking(now)
        recent_state = tracking
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
            'finish_line_y': recent_state.get('finish_line_y') if recent_state else None,
            'finish_line_rear_y': recent_state.get('finish_line_rear_y') if recent_state else None,
            'finish_line_passed': recent_state.get('finish_line_passed') if recent_state else None,
            'lane_change_reason': recent_state.get('lane_change_reason', '') if recent_state else ''})))


if __name__ == '__main__':
    rospy.init_node('map_debug')
    MapDebug()
    rospy.spin()
