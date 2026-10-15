#!/usr/bin/env python3
"""Show only LiDAR returns FTG treats as obstacle hits, grouped into boxes.

Also draws the GPS-route avoidance plan (/leo/route_avoidance): the local
path, the global route it leaves, the obstacles it judged and its state.
"""
import json
import math
import time

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from leo_avoidance.ftg_local import Config, FollowTheGap
from leo_avoidance.ftg_visualization import (cluster_boxes,
                                             line_candidate_points,
                                             relative_lidar_xy)
from leo_avoidance.route_avoidance import RouteFrame
from leo_guard.route_file import load_route_points

ROUTE_STATE_COLORS = {
    'PASSING': (.2, 1., .2),
    'WAIT_PASS': (1., .85, 0.),
    'WAIT_SOLID_LINE': (1., .15, .15),
}


class FTGVisualizationNode:
    def __init__(self):
        self.planner = FollowTheGap(Config(**{
            key: rospy.get_param('~' + key, default)
            for key, default in vars(Config()).items()}))
        self.period_s = 1. / float(rospy.get_param('~visualization_hz', 5.))
        self.target_distance_m = float(rospy.get_param(
            '~target_distance_from_lidar_m', 4.0))
        if not np.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError('visualization_hz must be positive')
        self.last_update = 0.
        self.points_pub = rospy.Publisher('/leo/ftg_obstacle_points', PointCloud2,
                                          queue_size=1)
        self.rear_points_pub = rospy.Publisher(
            '/leo/ftg_rear_obstacle_points', PointCloud2, queue_size=1)
        self.boxes_pub = rospy.Publisher('/leo/ftg_obstacle_boxes', MarkerArray,
                                         queue_size=1)
        self.pose_pub = rospy.Publisher('/leo/ftg_pose_diagnostics', MarkerArray,
                                        queue_size=1)
        self.planning_pub = rospy.Publisher('/leo/ftg_planning', MarkerArray,
                                            queue_size=1)
        self.last_planning_update = 0.
        self.jump_event = None
        rospy.Subscriber('/lidar3D', PointCloud2, self.on_cloud, queue_size=1,
                         buff_size=2 ** 24)
        rospy.Subscriber('/leo/driving_status', String, self.on_driving_status,
                         queue_size=1)
        rospy.Subscriber('/leo/avoidance_status', String,
                         self.on_avoidance_status, queue_size=1)
        # GPS-route avoidance plan, drawn in the same LiDAR view.
        waypoint_file = rospy.get_param('~waypoint_file', '')
        self.route = RouteFrame(load_route_points(waypoint_file)) if waypoint_file else None
        self.route_view_m = (-10., 60.)
        self.route_pose = None
        self.last_route_update = 0.
        self.route_pub = rospy.Publisher('/leo/route_avoidance_markers', MarkerArray,
                                         queue_size=1)
        rospy.Subscriber(rospy.get_param('~pose_topic', '/localization/pose'),
                         PoseStamped, self.on_route_pose, queue_size=1)
        rospy.Subscriber('/leo/route_avoidance', String, self.on_route_avoidance,
                         queue_size=1)

    def on_route_pose(self, message):
        p, q = message.pose.position, message.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if all(math.isfinite(v) for v in (p.x, p.y, yaw)):
            self.route_pose = (p.x, p.y, yaw)

    def route_xy(self, xy):
        """Map point -> current LiDAR view, or None outside the drawn window."""
        local = relative_lidar_xy(xy, self.route_pose,
                                  self.planner.config.lidar_to_rear_x_m)
        if local is None or not self.route_view_m[0] <= local[0] <= self.route_view_m[1]:
            return None
        return local

    def route_line(self, namespace, xy_points, color, width, z):
        line = self.pose_marker('lidar', namespace, 0, Marker.LINE_STRIP, (0., 0.), color)
        line.pose.position.z = 0.
        line.points = [Point(x=x, y=y, z=z) for x, y in
                       filter(None, (self.route_xy(p) for p in xy_points))]
        line.scale.x = width
        line.lifetime = rospy.Duration(1.)
        return line if len(line.points) >= 2 else None

    def on_route_avoidance(self, message):
        now = time.monotonic()
        if now - self.last_route_update < .1 or self.route_pose is None:
            return
        self.last_route_update = now
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        state = status.get('state')
        color = ROUTE_STATE_COLORS.get(state, (.25, .85, 1.))
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'lidar'
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        if self.route is not None:
            ego_s, _, _ = self.route.project(self.route_pose[0], self.route_pose[1],
                                             0, len(self.route.lengths))
            ahead = [self.route.pose(s)[0] for s in
                     np.arange(ego_s + self.route_view_m[0], ego_s + self.route_view_m[1], 1.)]
            route = self.route_line('route_global_path', ahead, (.6, .6, .6), .06, .02)
            if route is not None:
                route.color.a = .6
                markers.markers.append(route)
        path = self.route_line('route_local_path', status.get('path') or [], color, .18, .08)
        if path is not None:
            markers.markers.append(path)
        for index, obstacle in enumerate(status.get('obstacles') or []):
            if self.route is None:
                break
            xy = self.route_xy(self.route.pose(float(obstacle['s']), float(obstacle['d']))[0])
            if xy is None:
                continue
            moving = bool(obstacle.get('moving'))
            radius = max(.2, float(obstacle.get('r', .2)))
            disc = self.pose_marker('lidar', 'route_obstacles', index, Marker.CYLINDER, xy,
                                    (1., .55, 0.) if moving else (1., .1, .1))
            disc.scale.x = disc.scale.y = 2. * radius
            disc.scale.z = .3
            disc.color.a = .6
            disc.lifetime = rospy.Duration(1.)
            markers.markers.append(disc)
            label = self.pose_marker(
                'lidar', 'route_obstacle_labels', index, Marker.TEXT_VIEW_FACING,
                (xy[0], xy[1] + radius + .4), (1., 1., 1.),
                '#{} {} d={:+.2f}m'.format(obstacle.get('id'),
                                            'moving' if moving else 'static',
                                            float(obstacle['d'])))
            label.lifetime = rospy.Duration(1.)
            markers.markers.append(label)
        text = self.pose_marker(
            'lidar', 'route_avoidance_status', 0, Marker.TEXT_VIEW_FACING,
            (6., 3.), color,
            'route: {} | {} | cap {:.0f} km/h | offset {:+.2f} m'.format(
                state, status.get('reason'), float(status.get('speed_cap_kmh') or 0.),
                float(status.get('offset_m') or 0.)))
        text.lifetime = rospy.Duration(1.)
        markers.markers.append(text)
        self.route_pub.publish(markers)

    def on_avoidance_status(self, message):
        now = time.monotonic()
        if now - self.last_planning_update < .1:
            return
        self.last_planning_update = now
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'lidar'
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        reach_m = self.target_distance_m
        for index, angle in enumerate(status.get('candidate_bearings_rad') or []):
            if not isinstance(angle, (float, int)) or not np.isfinite(angle):
                continue
            candidate = Marker()
            candidate.header.frame_id = 'lidar'
            candidate.header.stamp = rospy.Time.now()
            candidate.ns = 'ftg_candidate_directions'
            candidate.id = index
            candidate.type = Marker.LINE_STRIP
            candidate.action = Marker.ADD
            candidate.pose.orientation.w = 1.
            candidate.points = [Point(x=0., y=0., z=.15),
                                Point(x=reach_m*np.cos(angle),
                                      y=reach_m*np.sin(angle), z=.15)]
            candidate.scale.x = .045
            candidate.color.r, candidate.color.g, candidate.color.b = .25, .7, 1.
            candidate.color.a = .8
            candidate.lifetime = rospy.Duration(.3)
            markers.markers.append(candidate)
        selected = status.get('target_bearing_rad')
        valid = (status.get('valid') and isinstance(selected, (float, int))
                 and np.isfinite(selected)
                 and float(status.get('target_speed_mps') or 0.) > 0.)
        if valid:
            path = Marker()
            path.header.frame_id = 'lidar'
            path.header.stamp = rospy.Time.now()
            path.ns = 'ftg_selected_direction'
            path.id = 0
            path.type = Marker.ARROW
            path.action = Marker.ADD
            path.pose.orientation.w = 1.
            path.points = [Point(x=0., y=0., z=.25),
                           Point(x=reach_m*np.cos(selected),
                                 y=reach_m*np.sin(selected), z=.25)]
            path.scale.x, path.scale.y, path.scale.z = .12, .25, .3
            path.color.g = path.color.a = 1.
            path.lifetime = rospy.Duration(.3)
            markers.markers.append(path)
        label = Marker()
        label.header.frame_id = 'lidar'
        label.header.stamp = rospy.Time.now()
        label.ns = 'ftg_planning_status'
        label.id = 0
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = 2.
        label.pose.position.z = 1.5
        label.pose.orientation.w = 1.
        label.scale.z = .4
        label.color.r = 0. if valid else 1.
        label.color.g = 1. if valid else .2
        label.color.a = 1.
        label.text = ('FTG direction: {:.1f} deg | {} candidates'.format(
            np.degrees(selected), len(status.get('candidate_bearings_rad') or []))
            if valid else 'FTG unavailable: {}'.format(status.get('reason')))
        label.lifetime = rospy.Duration(.3)
        markers.markers.append(label)
        self.planning_pub.publish(markers)

    @staticmethod
    def pose_marker(frame, namespace, marker_id, marker_type, xy, color,
                    text=None):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y = xy
        marker.pose.position.z = .5 if marker_type == Marker.TEXT_VIEW_FACING else .1
        marker.pose.orientation.w = 1.
        marker.scale.x = marker.scale.y = .6
        marker.scale.z = .38 if marker_type == Marker.TEXT_VIEW_FACING else .6
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.
        marker.lifetime = rospy.Duration(.35)
        if text is not None:
            marker.text = text
        return marker

    def on_driving_status(self, message):
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        output = status.get('driving_pose') or status.get('previous_driving_pose')
        if output is None:
            return
        frame = 'lidar'
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        locations = (
            ('LiDAR estimate', status.get('lidar_pose_before_select'), (0., .85, 1.)),
            ('MORAI actual', status.get('ego_pose'), (1., .25, .85)),
            ('Driving output' if status.get('driving_pose') is not None
             else 'Last driving output', output, (.25, 1., .25)))
        for index, (name, pose, color) in enumerate(locations):
            xy = relative_lidar_xy(pose, output, self.planner.config.lidar_to_rear_x_m)
            if xy is None:
                continue
            markers.markers.append(self.pose_marker(
                frame, 'pose_sources', index, Marker.SPHERE, xy, color))
            markers.markers.append(self.pose_marker(
                frame, 'pose_labels', index, Marker.TEXT_VIEW_FACING,
                (xy[0], xy[1] + .8), color,
                '{} ({:.1f}, {:.1f})'.format(name, pose[0], pose[1])))
        lidar = status.get('lidar_pose_before_select')
        ego = status.get('ego_pose')
        lidar_xy = relative_lidar_xy(lidar, output, self.planner.config.lidar_to_rear_x_m)
        ego_xy = relative_lidar_xy(ego, output, self.planner.config.lidar_to_rear_x_m)
        if lidar_xy is not None and ego_xy is not None:
            separation = float(np.hypot(float(lidar[0])-float(ego[0]),
                                        float(lidar[1])-float(ego[1])))
            line = self.pose_marker(frame, 'estimate_error', 0, Marker.LINE_STRIP,
                                    (0., 0.), (1., .7, 0.))
            line.points = [Point(x=x, y=y, z=.1) for x, y in (lidar_xy, ego_xy)]
            line.scale.x = .08
            markers.markers.append(line)
            markers.markers.append(self.pose_marker(
                frame, 'estimate_error_label', 0, Marker.TEXT_VIEW_FACING,
                ((lidar_xy[0]+ego_xy[0])/2., (lidar_xy[1]+ego_xy[1])/2.),
                (1., .7, 0.), 'LiDAR-MORAI {:.1f} m'.format(separation)))
        jump_m = status.get('driving_pose_jump_m')
        previous = status.get('previous_driving_pose')
        if (isinstance(jump_m, (float, int)) and np.isfinite(jump_m)
                and jump_m > 10. and previous is not None):
            self.jump_event = (time.monotonic(), previous, output,
                               float(jump_m), status.get('mode'))
        if self.jump_event is not None and time.monotonic()-self.jump_event[0] < 20.:
            _, old, new, distance, jump_mode = self.jump_event
            old_xy = relative_lidar_xy(old, output, self.planner.config.lidar_to_rear_x_m)
            new_xy = relative_lidar_xy(new, output, self.planner.config.lidar_to_rear_x_m)
            if old_xy is not None and new_xy is not None:
                line = self.pose_marker(frame, 'pose_jump', 0, Marker.LINE_STRIP,
                                        (0., 0.), (1., 0., 0.))
                line.points = [Point(x=x, y=y, z=.2) for x, y in (old_xy, new_xy)]
                line.scale.x = .15
                markers.markers.append(line)
                markers.markers.append(self.pose_marker(
                    frame, 'pose_jump_label', 0, Marker.TEXT_VIEW_FACING,
                    new_xy, (1., 0., 0.),
                    'POSE JUMP {:.1f} m | {} | ({:.1f},{:.1f}) -> ({:.1f},{:.1f})'.format(
                        distance, jump_mode, old[0], old[1], new[0], new[1])))
        markers.markers.append(self.pose_marker(
            frame, 'driving_mode', 0, Marker.TEXT_VIEW_FACING,
            (-self.planner.config.lidar_to_rear_x_m, -1.5), (1., 1., 1.),
            'mode: {} | {}'.format(status.get('mode'), status.get('reason'))))
        self.pose_pub.publish(markers)

    def on_cloud(self, message):
        now = time.monotonic()
        if now - self.last_update < self.period_s:
            return
        self.last_update = now
        if message.header.frame_id != 'lidar':
            rospy.logwarn_throttle(5., 'FTG visualization expects lidar frame')
            return
        cloud = np.asarray(list(point_cloud2.read_points(
            message, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=float)
        cloud = cloud.reshape((-1, 3))
        # Share the ground/height filtering pass across both displays.
        filtered = self.planner._in_collision_height(
            self.planner._remove_ground(cloud[np.isfinite(cloud).all(axis=1)]))
        all_candidates = self.planner._in_avoidance_sector(filtered)
        points, wall_fits = self.planner._track_obstacles(all_candidates)
        rear_points = self.planner._rear_obstacle_points(filtered)
        self.points_pub.publish(point_cloud2.create_cloud_xyz32(
            message.header, points.astype(np.float32).tolist()))
        self.rear_points_pub.publish(point_cloud2.create_cloud_xyz32(
            message.header, rear_points.astype(np.float32).tolist()))

        markers = MarkerArray()
        clear = Marker()
        clear.header = message.header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        bubble_m = (self.planner.config.vehicle_half_width_m
                    + self.planner.config.side_margin_m)
        for index, (minimum, maximum, count, distance, center, size, yaw, group) in enumerate(
                cluster_boxes(points, oriented=True)):
            front = maximum[0] > 0 and minimum[1] <= bubble_m and maximum[1] >= -bubble_m
            line_points = line_candidate_points(group, size, yaw)
            box = Marker()
            box.header = message.header
            box.ns = ('ftg_linear_candidates' if line_points is not None
                      else 'ftg_candidate_boxes')
            box.id = index
            box.type = Marker.LINE_STRIP if line_points is not None else Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x, box.pose.position.y, box.pose.position.z = center
            box.pose.orientation.z = np.sin(yaw / 2.)
            box.pose.orientation.w = np.cos(yaw / 2.)
            if line_points is not None:
                box.pose.position.x = box.pose.position.y = box.pose.position.z = 0.
                box.pose.orientation.z = 0.
                box.pose.orientation.w = 1.
                box.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                              for p in line_points]
                box.scale.x = .12
            else:
                box.scale.x, box.scale.y, box.scale.z = size
            box.color.r = 1.
            box.color.g = .15 if front else .7
            box.color.b = .15 if front else 0.
            box.color.a = 1. if line_points is not None else .32
            box.lifetime = rospy.Duration(.5)
            markers.markers.append(box)

            label = Marker()
            label.header = message.header
            label.ns = 'ftg_candidate_labels'
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = center[:2]
            label.pose.position.z = maximum[2] + .35
            label.pose.orientation.w = 1.
            label.scale.z = .35
            label.color.r = label.color.g = label.color.b = label.color.a = 1.
            label.text = '{}{} pts / {:.1f} m'.format(
                'linear candidate: ' if line_points is not None else '',
                count, distance)
            label.lifetime = rospy.Duration(.5)
            markers.markers.append(label)
        for index, fit in enumerate(wall_fits):
            if fit is None:
                continue
            _, _, slope, offset, curvature, first_x, last_x = fit
            edge = Marker()
            edge.header = message.header
            edge.ns = 'track_boundaries'
            edge.id = index
            edge.type = Marker.LINE_STRIP
            edge.action = Marker.ADD
            edge.points = [Point(x=float(x),
                                 y=float(curvature*x*x+slope*x+offset), z=0.)
                           for x in np.linspace(first_x, last_x, 20)]
            edge.scale.x = .12
            edge.color.g = edge.color.a = 1.
            edge.lifetime = rospy.Duration(.5)
            markers.markers.append(edge)
        wall = wall_fits[0][:2] if wall_fits[0] is not None else None
        if wall is not None:
            clearance, heading = wall
            wall_y = -(clearance + self.planner.config.vehicle_half_width_m)
            ruler = Marker()
            ruler.header = message.header
            ruler.ns = 'right_wall_distance'
            ruler.id = 0
            ruler.type = Marker.LINE_STRIP
            ruler.action = Marker.ADD
            ruler.points = [Point(x=0., y=0., z=0.),
                            Point(x=0., y=wall_y, z=0.)]
            ruler.scale.x = .08
            ruler.color.g = ruler.color.a = 1.
            ruler.lifetime = rospy.Duration(.5)
            markers.markers.append(ruler)

            wall_label = Marker()
            wall_label.header = message.header
            wall_label.ns = 'right_wall_distance_label'
            wall_label.id = 0
            wall_label.type = Marker.TEXT_VIEW_FACING
            wall_label.action = Marker.ADD
            wall_label.pose.position.y = wall_y / 2.
            wall_label.pose.position.z = .4
            wall_label.pose.orientation.w = 1.
            wall_label.scale.z = .4
            wall_label.color.g = wall_label.color.a = 1.
            wall_label.text = 'right wall: {:.2f} m clearance / {:.1f} deg'.format(
                clearance, np.degrees(heading))
            wall_label.lifetime = rospy.Duration(.5)
            markers.markers.append(wall_label)
        self.boxes_pub.publish(markers)


if __name__ == '__main__':
    rospy.init_node('leo_ftg_visualization')
    FTGVisualizationNode()
    rospy.spin()
