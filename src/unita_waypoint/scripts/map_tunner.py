#!/usr/bin/env python3
"""Edit independent sector racelines over a georeferenced K-City aerial in RViz."""

import json
import math
import sys
import tempfile
import threading
from pathlib import Path

import rospkg
import rospy
from geometry_msgs.msg import Point, Point32, PointStamped, PolygonStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    InteractiveMarkerFeedback, Marker, MarkerArray)

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from map_tunner_core import (ProjectEditor, create_project, insert_index,
                             load_checkpoints, load_csv, load_project,
                             nearest_point, save_project)
from map_tunner_map import AerialMap


COLORS = {1: (1.0, 0.16, 0.12), 2: (0.05, 0.65, 1.0),
          3: (0.15, 1.0, 0.35)}


def color(marker, rgb, alpha=1.0):
    marker.color.r, marker.color.g, marker.color.b = rgb
    marker.color.a = alpha


def marker_base(namespace, marker_id, marker_type, scale, position=None):
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.header.stamp = rospy.Time.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x, marker.scale.y, marker.scale.z = scale
    if position is not None:
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
    return marker


def text_marker(namespace, marker_id, label, x, y, z, size, rgb, alpha=1.0):
    marker = marker_base(namespace, marker_id, Marker.TEXT_VIEW_FACING,
                         (0.0, 0.0, size), (x, y, z))
    marker.text = label
    color(marker, rgb, alpha)
    return marker


class MapTunner:
    def __init__(self):
        root = Path(rospkg.RosPack().get_path('unita_waypoint'))
        config = root / 'config'
        self.waypoint_file = Path(rospy.get_param('~waypoint_file', str(config / 'waypoints.csv'))).expanduser()
        self.checkpoint_file = Path(rospy.get_param('~checkpoint_file', str(config / 'map_tunner_checkpoints.yaml'))).expanduser()
        self.project_file = Path(rospy.get_param('~project_file', str(config / 'map_tunner_project.json'))).expanduser()
        self.calibration_file = Path(rospy.get_param('~calibration_file', str(config / 'map_tunner_calibration.yaml'))).expanduser()
        self.checkpoints = load_checkpoints(self.checkpoint_file)
        self.reference = load_csv(self.waypoint_file)
        if self.project_file.exists():
            project = load_project(self.project_file)
        else:
            project = create_project(self.waypoint_file, self.checkpoint_file,
                                     rospy.get_param('~default_speed_kmh', 20))
            save_project(self.project_file, project)
        self.editor = ProjectEditor(project)
        self.aerial = AerialMap(rospy.get_param('~image_file', str(config / 'kcity_map_sim.png')),
                                self.calibration_file, self.checkpoints,
                                rospy.get_param('~map_resolution', 0.5))
        self.calibration_mtime_ns = self.calibration_file.stat().st_mtime_ns
        self.photo_mesh_directory = tempfile.TemporaryDirectory(prefix='map_tunner_photo_')
        self.lock = threading.RLock()
        self.active_sector = 1
        self.active_lane = 1
        self.mode = 'select'
        self.selected_id = None
        self.range_start_id = None
        self.range_end_id = None
        self.pending_calibration = None
        self.interpolation_start = None
        self.resize_base = None

        self.map_pub = rospy.Publisher('/map_tunner/map', OccupancyGrid,
                                       queue_size=1, latch=True)
        self.photo_pub = rospy.Publisher('/map_tunner/photo', Marker,
                                         queue_size=1, latch=True)
        self.image_corners_pub = rospy.Publisher('/map_tunner/image_corners',
                                                 PolygonStamped, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher('/map_tunner/markers', MarkerArray,
                                          queue_size=1, latch=True)
        self.status_pub = rospy.Publisher('/map_tunner/status', String,
                                          queue_size=1, latch=True)
        self.state_pub = rospy.Publisher('/map_tunner/state', String,
                                         queue_size=1, latch=True)
        self.server = InteractiveMarkerServer('map_tunner/interactive')
        self.cp_menu = MenuHandler()
        self.cp_actions = {}
        for label, action in [
            ('Select outgoing sector', 'select_sector'),
            ('Add lane', 'add_lane'), ('Remove last lane', 'remove_lane'),
            ('Raceline 1', 'lane_1'), ('Raceline 2', 'lane_2'),
            ('Raceline 3', 'lane_3'), ('Add point mode', 'mode_add'),
            ('Select point mode', 'mode_select'),
            ('Calibrate image here: click aerial checkpoint', 'calibrate'),
            ('Undo', 'undo'), ('Save project', 'save')]:
            self.cp_actions[self.cp_menu.insert(label, callback=self.on_feedback)] = action
        self.point_menu = MenuHandler()
        self.point_actions = {}
        for label, action in [('+1 km/h', 'speed_up'), ('-1 km/h', 'speed_down'),
                              ('Delete point', 'delete_point'), ('Undo', 'undo'),
                              ('Calibrate image to this waypoint: click road', 'calibrate_point'),
                              ('Save project', 'save')]:
            self.point_actions[self.point_menu.insert(label, callback=self.on_feedback)] = action
        rospy.Subscriber('/clicked_point', PointStamped, self.on_click, queue_size=10)
        rospy.Subscriber('/map_tunner/smooth_point', PointStamped,
                         self.on_smooth_click, queue_size=10)
        rospy.Subscriber('/map_tunner/command', String, self.on_command, queue_size=10)
        self.publish_map()
        self.publish_color_photo()
        self.publish_image_corners()
        self.refresh()
        rospy.Timer(rospy.Duration(1.0), self.reload_calibration_if_changed)
        rospy.loginfo('Map tuner ready. Select a checkpoint or use the RViz controls.')

    def reload_calibration_if_changed(self, _event):
        with self.lock:
            try:
                mtime_ns = self.calibration_file.stat().st_mtime_ns
                if mtime_ns == self.calibration_mtime_ns:
                    return
                aerial = AerialMap(self.aerial.image_path, self.calibration_file,
                                   self.checkpoints, self.aerial.resolution)
                self.aerial = aerial
                self.calibration_mtime_ns = mtime_ns
                self.publish_map()
                self.publish_color_photo()
                self.publish_image_corners()
                self.refresh('사진 배치를 다시 읽었습니다.')
            except (OSError, ValueError) as error:
                rospy.logwarn_throttle(5, 'Could not reload photo calibration: %s', error)

    def status(self, message=None):
        sector = self.editor.sector(self.active_sector)
        state = ('Sector {} ({} -> {}) | {} lane(s) | raceline {} | {} mode | {}'
                 .format(self.active_sector, sector['from'], sector['to'],
                         sector['lane_count'], self.active_lane, self.mode,
                         'UNSAVED' if self.editor.dirty else 'saved'))
        if self.pending_calibration:
            state += ' | ALIGN {}: click aerial with Publish Point'.format(
                self.pending_calibration[1])
        if message:
            state += ' | ' + message
            rospy.loginfo('%s', state)
        self.status_pub.publish(String(data=state))
        selected = self.selected_point()
        range_points = self.speed_range_points()
        self.state_pub.publish(String(data=json.dumps({
            'sector': self.active_sector,
            'from': sector['from'],
            'to': sector['to'],
            'lane_count': sector['lane_count'],
            'lane': self.active_lane,
            'mode': self.mode,
            'point_id': selected['id'] if selected else None,
            'speed_kmh': selected['speed_kmh'] if selected else None,
            'range_start_id': self.range_start_id,
            'range_end_id': self.range_end_id,
            'range_count': len(range_points),
            'dirty': self.editor.dirty,
            'calibration': self.pending_calibration is not None,
            'interpolation_start': self.interpolation_start is not None,
            'message': message or '',
        })))

    def publish_map(self):
        origin_x, origin_y, data = self.aerial.raster(self.reference)
        message = OccupancyGrid()
        message.header.frame_id = 'map'
        message.header.stamp = rospy.Time.now()
        message.info.resolution = self.aerial.resolution
        message.info.width = data.shape[1]
        message.info.height = data.shape[0]
        message.info.origin.position.x = origin_x
        message.info.origin.position.y = origin_y
        message.info.origin.orientation.w = 1.0
        message.data = data.ravel().astype('int8').tolist()
        self.map_pub.publish(message)

    def publish_color_photo(self):
        marker = marker_base('color_photo', 0, Marker.MESH_RESOURCE,
                             (1.0, 1.0, 1.0))
        marker.mesh_resource = self.aerial.color_mesh_resource(self.photo_mesh_directory.name)
        marker.mesh_use_embedded_materials = True
        color(marker, (1.0, 1.0, 1.0))
        self.photo_pub.publish(marker)

    def publish_image_corners(self):
        message = PolygonStamped()
        message.header.frame_id = 'map'
        message.header.stamp = rospy.Time.now()
        message.polygon.points = [Point32(float(x), float(y), 0.0)
                                  for x, y in self.aerial.image_corners_world()]
        self.image_corners_pub.publish(message)

    def selected_point(self):
        if self.selected_id is None:
            return None
        return next((point for point in self.editor.lane(self.active_sector, self.active_lane)
                     if point['id'] == self.selected_id), None)

    def speed_range_points(self):
        if self.range_start_id is None or self.range_end_id is None:
            return []
        points = self.editor.lane(self.active_sector, self.active_lane)
        indices = {point['id']: index for index, point in enumerate(points)}
        if self.range_start_id not in indices or self.range_end_id not in indices:
            return []
        first, last = sorted((indices[self.range_start_id], indices[self.range_end_id]))
        return points[first:last + 1]

    def clear_speed_range(self):
        self.range_start_id = None
        self.range_end_id = None

    def refresh(self, message=None):
        with self.lock:
            if self.active_lane > self.editor.sector(self.active_sector)['lane_count']:
                self.active_lane = self.editor.sector(self.active_sector)['lane_count']
                self.selected_id = None
            if self.selected_point() is None:
                self.selected_id = None
            ids = {point['id'] for point in self.editor.lane(self.active_sector,
                                                              self.active_lane)}
            if self.range_start_id not in ids:
                self.clear_speed_range()
            elif self.range_end_id not in ids:
                self.range_end_id = None
            self.publish_markers()
            self.publish_interactive()
            self.status(message)

    def publish_markers(self):
        markers = []
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.append(delete)
        for sector in self.editor.project['sectors']:
            number = sector['number']
            for lane_number in range(1, sector['lane_count'] + 1):
                points = sector['racelines'][str(lane_number)]
                if len(points) >= 2:
                    marker = marker_base('sector_paths', number * 10 + lane_number,
                                         Marker.LINE_STRIP, (3.2 if number == self.active_sector else 1.8,
                                                             0.0, 0.0))
                    inactive = ((0.0, 0.95, 1.0) if lane_number == 1
                                else COLORS[lane_number])
                    color(marker, COLORS[lane_number] if number == self.active_sector
                          else inactive, 1.0 if number == self.active_sector else 0.9)
                    marker.points = [Point(point['x'], point['y'], 0.5 + lane_number * 0.05)
                                     for point in points]
                    markers.append(marker)
                if points and number == self.active_sector:
                    dots = marker_base('active_points', lane_number, Marker.POINTS,
                                        (0.9, 0.9, 0.0))
                    color(dots, COLORS[lane_number], 0.85)
                    stride = max(1, len(points) // 400)
                    dots.points = [Point(point['x'], point['y'], 0.8)
                                   for point in points[::stride]]
                    markers.append(dots)
        for index, name in enumerate(['start'] + [str(i) for i in range(1, 15)]):
            x, y = self.checkpoints[name]['position'][:2]
            width = 26.0 if name == 'start' else 17.0
            body = marker_base('checkpoint_boxes', index, Marker.CUBE,
                               (width, 17.0, 1.0), (x, y, 2.5))
            color(body, (0.12, 0.95, 0.16) if index == self.active_sector - 1
                  else (0.0, 0.55, 0.02), 0.45)
            markers.append(body)
            markers.append(text_marker('checkpoint_numbers', index,
                                       name.upper(), x, y, 3.7, 11.0,
                                       (1.0, 1.0, 1.0), 0.78))
        range_points = self.speed_range_points()
        if range_points:
            line = marker_base('speed_range', 0, Marker.LINE_STRIP, (4.0, 0.0, 0.0))
            line.points = [Point(point['x'], point['y'], 1.25) for point in range_points]
            color(line, (1.0, 0.85, 0.05), 0.9)
            markers.append(line)
        current_points = self.editor.lane(self.active_sector, self.active_lane)
        for marker_id, point_id, label, rgb in (
                (0, self.range_start_id, '속도 시작', (1.0, 0.55, 0.05)),
                (1, self.range_end_id, '속도 끝', (0.95, 0.2, 0.8))):
            point = next((item for item in current_points if item['id'] == point_id), None)
            if point is None:
                continue
            ball = marker_base('speed_range_ends', marker_id, Marker.SPHERE,
                               (4.0, 4.0, 2.0), (point['x'], point['y'], 2.0))
            color(ball, rgb, 0.9)
            markers.append(ball)
            markers.append(text_marker('speed_range_labels', marker_id, label,
                                       point['x'], point['y'], 4.5, 4.0, rgb))
        if self.interpolation_start:
            x, y = self.interpolation_start
            start = marker_base('interpolation_start', 0, Marker.SPHERE,
                                (5.0, 5.0, 1.0), (x, y, 2.5))
            color(start, (1.0, 0.95, 0.1))
            markers.append(start)
            markers.append(text_marker('interpolation_start_label', 0,
                                       'START → 끝점 좌클릭', x, y, 5.0, 6.0,
                                       (1.0, 1.0, 0.1)))
        selected = self.selected_point()
        if selected:
            markers.append(text_marker('selected_point', 1,
                                       'S{} R{}  point {}  {} km/h'.format(
                                           self.active_sector, self.active_lane,
                                           selected['id'], selected['speed_kmh']),
                                       selected['x'], selected['y'], 4.0, 3.5, (1.0, 1.0, 0.0)))
        self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_interactive(self):
        self.server.clear()
        if self.mode == 'resize':
            corners = self.aerial.image_corners_world()
            locations = {'tl': corners[0], 'tr': corners[1],
                         'br': corners[2], 'bl': corners[3],
                         'top': (corners[0] + corners[1]) / 2,
                         'right': (corners[1] + corners[2]) / 2,
                         'bottom': (corners[2] + corners[3]) / 2,
                         'left': (corners[3] + corners[0]) / 2}
            for name, (x, y) in locations.items():
                marker = InteractiveMarker()
                marker.header.frame_id = 'map'
                marker.name = 'resize/' + name
                marker.description = ('비율 유지' if len(name) == 2 else '한 방향 크기 변경')
                marker.scale = 12.0
                marker.pose.position.x = float(x)
                marker.pose.position.y = float(y)
                marker.pose.position.z = 2.0
                marker.pose.orientation.w = 1.0
                control = InteractiveMarkerControl()
                control.name = 'drag'
                control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
                control.always_visible = True
                control.orientation.w = math.sqrt(0.5)
                control.orientation.y = math.sqrt(0.5)
                body = marker_base('resize_handles', 0, Marker.CUBE, (12.0, 12.0, 1.0))
                color(body, (1.0, 0.78, 0.0) if len(name) == 2
                      else (0.1, 0.6, 1.0))
                control.markers.append(body)
                marker.controls.append(control)
                self.server.insert(marker, self.on_feedback)
        selected = self.selected_point()
        if selected:
            marker = InteractiveMarker()
            marker.header.frame_id = 'map'
            marker.name = 'selected_point'
            marker.description = '좌클릭 드래그: 위치 이동 · 우클릭: 속도/삭제'
            marker.scale = 8.0
            marker.pose.position.x = selected['x']
            marker.pose.position.y = selected['y']
            marker.pose.position.z = 1.5
            marker.pose.orientation.w = 1.0
            control = InteractiveMarkerControl()
            control.name = 'move_xy'
            control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            control.always_visible = True
            control.orientation.w = math.sqrt(0.5)
            control.orientation.y = math.sqrt(0.5)
            body = marker_base('selected', 0, Marker.SPHERE, (6.0, 6.0, 2.5))
            color(body, (1.0, 1.0, 0.0))
            control.markers.append(body)
            marker.controls.append(control)
            self.server.insert(marker, self.on_feedback)
            self.point_menu.apply(self.server, marker.name)
        self.server.applyChanges()

    def on_click(self, message):
        with self.lock:
            if message.header.frame_id != 'map':
                rospy.logwarn('Publish Point fixed frame must be map (got %s)',
                              message.header.frame_id)
                return
            x, y = message.point.x, message.point.y
            if not math.isfinite(x) or not math.isfinite(y):
                return
            if self.pending_calibration:
                calibration = self.pending_calibration
                try:
                    if calibration[0] == 'checkpoint':
                        name = calibration[1]
                        pixel = self.aerial.update_checkpoint_anchor(name, (x, y))
                    else:
                        name, world_xy = calibration[1:]
                        pixel = self.aerial.update_waypoint_anchor(name, world_xy, (x, y))
                except ValueError as error:
                    self.refresh(str(error))
                    return
                self.pending_calibration = None
                self.aerial.save_calibration()
                self.calibration_mtime_ns = self.calibration_file.stat().st_mtime_ns
                self.publish_map()
                self.publish_color_photo()
                self.refresh('Image anchor {} -> ({:.1f}, {:.1f})'.format(name, *pixel))
                return
            if self.mode == 'interpolate':
                if self.editor.lane(self.active_sector, self.active_lane):
                    self.refresh('선 보간은 빈 Raceline에서만 가능합니다.')
                    return
                if self.interpolation_start is None:
                    self.interpolation_start = (x, y)
                    self.refresh('시작점 지정 완료. 끝점을 좌클릭하세요.')
                    return
                try:
                    count = self.editor.fill_interpolated_lane(
                        self.active_sector, self.active_lane,
                        self.interpolation_start, (x, y))
                except ValueError as error:
                    self.refresh(str(error))
                    return
                self.interpolation_start = None
                self.selected_id = None
                self.refresh('선 보간 완료: 시작/끝 포함 {}점, 사이에 {}점 생성'.format(
                    count, count - 2))
                return
            if self.mode == 'resize':
                return
            if self.mode == 'smooth':
                self.smooth_at(x, y)
                return
            points = self.editor.lane(self.active_sector, self.active_lane)
            if self.mode == 'speed_range':
                index, distance = nearest_point(points, x, y)
                if index is None or distance > 8.0:
                    self.refresh('8 m 이내에 경로점이 없습니다. 확대해서 다시 선택하세요.')
                    return
                point_id = points[index]['id']
                if self.range_start_id is None or self.range_end_id is not None:
                    self.range_start_id = point_id
                    self.range_end_id = None
                    self.refresh('속도 시작점 #{} 선택. 끝점을 좌클릭하세요.'.format(point_id))
                else:
                    self.range_end_id = point_id
                    self.refresh('속도 구간 #{} → #{} ({}점) 선택. 속도를 입력하세요.'.format(
                        self.range_start_id, point_id, len(self.speed_range_points())))
                return
            if self.mode == 'select':
                index, distance = nearest_point(points, x, y)
                if index is None or distance > 8.0:
                    self.refresh('8 m 이내에 점이 없습니다. 확대하거나 점 추가 모드를 사용하세요.')
                    return
                self.selected_id = points[index]['id']
                self.refresh('점 #{} 선택 ({:.2f} m 거리)'.format(self.selected_id, distance))
            else:
                index = insert_index(points, x, y)
                reference_index = min(range(len(self.reference)),
                                      key=lambda i: (self.reference[i][0] - x) ** 2
                                                  + (self.reference[i][1] - y) ** 2)
                z = self.reference[reference_index][2]
                point = self.editor.add_point(self.active_sector, self.active_lane,
                                              x, y, z, 20, index)
                self.selected_id = point['id']
                self.refresh('점 #{} 추가 ({}번째)'.format(point['id'], index + 1))

    def on_smooth_click(self, message):
        with self.lock:
            if self.mode != 'smooth' or message.header.frame_id != 'map':
                return
            self.smooth_at(message.point.x, message.point.y)

    def smooth_at(self, x, y):
        points = self.editor.lane(self.active_sector, self.active_lane)
        index, distance = nearest_point(points, x, y)
        if index is None or distance > 8.0:
            self.refresh('8 m 이내에 경로점이 없습니다.')
            return
        self.selected_id = points[index]['id']
        try:
            changed = self.editor.smooth_neighborhood(
                self.active_sector, self.active_lane, self.selected_id)
        except ValueError as error:
            self.refresh(str(error))
            return
        self.refresh('점 #{} 주변 앞뒤 15개로 {}개 점 스무딩'.format(
            self.selected_id, changed))

    def on_feedback(self, feedback):
        with self.lock:
            if feedback.marker_name.startswith('resize/'):
                name = feedback.marker_name.split('/', 1)[1]
                if feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
                    self.resize_base = self.aerial.image_corners_world().copy()
                elif feedback.event_type in (InteractiveMarkerFeedback.POSE_UPDATE,
                                              InteractiveMarkerFeedback.MOUSE_UP):
                    try:
                        self.aerial.resize_handle(
                            name, (feedback.pose.position.x, feedback.pose.position.y),
                            self.resize_base)
                    except ValueError as error:
                        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                            self.resize_base = None
                            self.refresh(str(error))
                        return
                    self.publish_image_corners()
                    if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                        self.resize_base = None
                        self.aerial.save_calibration()
                        self.calibration_mtime_ns = self.calibration_file.stat().st_mtime_ns
                        self.publish_map()
                        self.publish_color_photo()
                        self.refresh('사진 크기와 위치를 저장했습니다.')
                return
            if feedback.marker_name == 'selected_point':
                if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                    selected = self.selected_point()
                    if selected and math.isfinite(feedback.pose.position.x) and math.isfinite(feedback.pose.position.y) and math.hypot(
                            feedback.pose.position.x - selected['x'],
                            feedback.pose.position.y - selected['y']) > 1e-4:
                        self.editor.change_point(self.active_sector, self.active_lane,
                                                 selected['id'],
                                                 x=feedback.pose.position.x,
                                                 y=feedback.pose.position.y)
                        self.refresh('점 #{} 이동'.format(selected['id']))
                    return
                if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
                    action = self.point_actions.get(feedback.menu_entry_id)
                    if action:
                        self.perform(action)
                return
            if feedback.marker_name.startswith('checkpoint/'):
                name = feedback.marker_name.split('/', 1)[1]
                self.active_sector = 1 if name == 'start' else int(name) + 1
                if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
                    action = self.cp_actions.get(feedback.menu_entry_id)
                    if action == 'calibrate':
                        self.pending_calibration = ('checkpoint', name)
                        self.refresh('Click the aerial position of checkpoint ' + name)
                    elif action:
                        self.perform(action)
                elif feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
                    self.perform('select_sector')
                return
    def on_command(self, message):
        """Commands from the fixed RViz panel; all edits still pass through perform()."""
        command = message.data.strip()
        with self.lock:
            try:
                if command.startswith('sector:'):
                    sector = int(command.split(':', 1)[1])
                    self.editor.sector(sector)
                    self.active_sector = sector
                    self.selected_id = None
                    self.clear_speed_range()
                    self.interpolation_start = None
                    self.refresh('S{} 구간 선택'.format(sector))
                elif command.startswith('lane:'):
                    lane = int(command.split(':', 1)[1])
                    self.perform('lane_{}'.format(lane))
                elif command.startswith('speed:'):
                    value = command.split(':', 1)[1]
                    if not value.isdecimal() or not 0 <= int(value) <= 200:
                        raise ValueError('속도는 0~200 km/h 정수로 입력하세요.')
                    selected = self.selected_point()
                    if selected is None:
                        raise ValueError('먼저 점을 선택하세요.')
                    self.editor.change_point(self.active_sector, self.active_lane,
                                             selected['id'], speed_kmh=int(value))
                    self.refresh('점 #{} 속도 {} km/h'.format(selected['id'], value))
                elif command.startswith('speed_range:'):
                    value = command.split(':', 1)[1]
                    if not value.isdecimal() or not 0 <= int(value) <= 200:
                        raise ValueError('속도는 0~200 km/h 정수로 입력하세요.')
                    if self.range_start_id is None or self.range_end_id is None:
                        raise ValueError('시작점과 끝점을 차례로 선택하세요.')
                    count = self.editor.set_speed_range(
                        self.active_sector, self.active_lane,
                        self.range_start_id, self.range_end_id, int(value))
                    self.refresh('현재 Raceline의 {}개 점에 {} km/h 적용'.format(count, value))
                else:
                    self.perform(command)
            except (ValueError, TypeError) as error:
                self.refresh(str(error))

    def perform(self, action):
        with self.lock:
            try:
                if action == 'select_sector':
                    self.selected_id = None
                    self.clear_speed_range()
                    self.interpolation_start = None
                elif action in ('prev_sector', 'next_sector'):
                    step = -1 if action == 'prev_sector' else 1
                    self.active_sector = (self.active_sector - 1 + step) % 15 + 1
                    self.selected_id = None
                    self.clear_speed_range()
                    self.interpolation_start = None
                elif action == 'add_lane':
                    self.active_lane = self.editor.add_lane(self.active_sector)
                    self.selected_id = None
                    self.clear_speed_range()
                    self.mode = 'add'
                    self.interpolation_start = None
                elif action == 'remove_lane':
                    removed = self.editor.remove_lane(self.active_sector)
                    self.clear_speed_range()
                    if self.active_lane == removed:
                        self.active_lane -= 1
                        self.selected_id = None
                    self.interpolation_start = None
                elif action.startswith('lane_'):
                    lane = int(action.rsplit('_', 1)[1])
                    self.editor.lane(self.active_sector, lane)
                    self.active_lane = lane
                    self.selected_id = None
                    self.clear_speed_range()
                    self.interpolation_start = None
                elif action in ('mode_add', 'mode_select', 'mode_interpolate', 'mode_smooth'):
                    self.mode = action.split('_', 1)[1]
                    if action == 'mode_select':
                        self.selected_id = None
                    self.interpolation_start = None
                elif action == 'mode_speed_range':
                    self.mode = 'speed_range'
                    self.selected_id = None
                    self.clear_speed_range()
                    self.interpolation_start = None
                elif action == 'mode_resize':
                    self.mode = 'resize'
                    self.interpolation_start = None
                elif action in ('speed_up', 'speed_down'):
                    selected = self.selected_point()
                    if not selected:
                        raise ValueError('select a point first')
                    change = 1 if action == 'speed_up' else -1
                    self.editor.change_point(self.active_sector, self.active_lane,
                                             selected['id'],
                                             speed_kmh=selected['speed_kmh'] + change)
                elif action == 'delete_point':
                    if self.selected_id is None:
                        raise ValueError('select a point first')
                    self.editor.remove_point(self.active_sector, self.active_lane,
                                             self.selected_id)
                    self.selected_id = None
                elif action == 'undo':
                    if not self.editor.undo():
                        raise ValueError('nothing to undo')
                elif action == 'calibrate_cp':
                    name = self.editor.sector(self.active_sector)['from']
                    self.pending_calibration = ('checkpoint', name)
                elif action == 'calibrate_point':
                    selected = self.selected_point()
                    if not selected:
                        raise ValueError('select a point first')
                    self.pending_calibration = ('point', selected['id'],
                                                (selected['x'], selected['y']))
                elif action == 'save':
                    save_project(self.project_file, self.editor.project)
                    self.editor.dirty = False
                else:
                    raise ValueError('unknown action ' + action)
            except ValueError as error:
                rospy.logwarn('%s', error)
                self.refresh(str(error))
                return
            labels = {
                'select_sector': '구간 선택',
                'add_lane': '빈 차선 추가',
                'remove_lane': '마지막 차선 삭제',
                'mode_add': '점 추가 모드: 지도에서 좌클릭하세요.',
                'mode_select': '점 선택 모드: 점을 좌클릭하면 노란 점을 바로 드래그할 수 있습니다.',
                'mode_speed_range': '속도 구간: 시작점과 끝점을 차례로 좌클릭하세요.',
                'mode_interpolate': '선 보간 모드: 빈 Raceline에 시작점과 끝점을 좌클릭하세요.',
                'mode_smooth': '스무딩 모드: 점을 좌클릭하면 주변 앞뒤 15개를 스무딩합니다.',
                'mode_resize': '사진 손잡이를 드래그하세요. 모서리는 비율 유지, 변은 한 방향만 변경합니다.',
                'speed_up': '속도 +1 km/h',
                'speed_down': '속도 -1 km/h',
                'delete_point': '점 삭제',
                'undo': '마지막 편집 실행 취소',
                'calibrate_cp': '사진 속 체크포인트를 Publish Point로 클릭하세요.',
                'calibrate_point': '사진 속 경로점 위치를 Publish Point로 클릭하세요.',
                'save': 'JSON 저장 완료',
            }
            self.refresh(labels.get(action, action.replace('_', ' ')))


if __name__ == '__main__':
    rospy.init_node('map_tunner')
    try:
        MapTunner()
    except Exception as error:
        rospy.logfatal('Map tuner startup failed: %s', error)
        raise
    rospy.spin()
