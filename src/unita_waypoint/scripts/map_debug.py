#!/usr/bin/env python3
"""Read-only K-City map, saved route, checkpoints, and live vehicle state."""

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
sys.path.insert(0, str(ROOT / 'scripts'))
from map_tunner_core import CHECKPOINT_ORDER, load_checkpoints, load_csv, load_project, primary_route
from map_tunner_map import AerialMap


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
        project_path = rospy.get_param('~project_file', str(config / 'map_tunner_project.json'))
        checkpoint_path = rospy.get_param('~checkpoint_file', str(config / 'map_tunner_checkpoints.yaml'))
        reference_path = rospy.get_param('~waypoint_file', str(config / 'waypoints.csv'))
        image_path = rospy.get_param('~image_file', str(config / 'kcity_map_sim.png'))
        calibration_path = rospy.get_param('~calibration_file', str(config / 'map_tunner_calibration.yaml'))
        self.project = load_project(project_path)
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
        self.valid = False
        self.last_vehicle_visible = False
        self.photo_pub = rospy.Publisher('/map_debug/photo', Marker, queue_size=1, latch=True)
        self.map_pub = rospy.Publisher('/map_debug/map', OccupancyGrid, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher('/map_debug/markers', MarkerArray, queue_size=1, latch=True)
        self.vehicle_pub = rospy.Publisher('/map_debug/vehicle', MarkerArray, queue_size=1)
        self.status_pub = rospy.Publisher('/map_debug/status', String, queue_size=1, latch=True)
        self.publish_static()
        rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'),
                         PoseStamped, self.on_pose, queue_size=1)
        rospy.Subscriber(rospy.get_param('~valid_topic', '/localization/valid'),
                         Bool, self.on_valid, queue_size=1)
        rospy.Subscriber(rospy.get_param('~speed_topic', '/competition/ego_speed_kmh'),
                         Float64, self.on_speed, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.refresh)
        rospy.loginfo('Map debug ready: %d saved Raceline 1 points', len(self.route))

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

        markers = []
        for sector in self.project['sectors']:
            for lane_number in range(1, sector['lane_count'] + 1):
                points = sector['racelines'][str(lane_number)]
                if len(points) < 2:
                    continue
                line = marker('sector_paths', sector['number'] * 10 + lane_number,
                              Marker.LINE_STRIP)
                line.scale.x = 2.0
                line.color.r, line.color.g, line.color.b, line.color.a = (
                    (0.0, 0.95, 1.0, 0.9) if lane_number == 1 else
                    (0.1, 0.55, 1.0, 0.75) if lane_number == 2 else
                    (0.1, 1.0, 0.4, 0.75))
                line.points = [Point(p['x'], p['y'], 0.55) for p in points]
                markers.append(line)
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

    def refresh(self, _event):
        now = time.monotonic()
        available = (self.valid and self.pose is not None and
                     now - self.pose_time < 1.0)
        speed = (self.speed if self.speed_time is not None and
                 now - self.speed_time < 1.0 else None)
        if available:
            x, y, yaw = self.pose
            index = int(np.argmin(np.sum((self.xy - (x, y)) ** 2, axis=1)))
            sector = self.sectors[index]
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
            'sector': sector})))


if __name__ == '__main__':
    rospy.init_node('map_debug')
    MapDebug()
    rospy.spin()
