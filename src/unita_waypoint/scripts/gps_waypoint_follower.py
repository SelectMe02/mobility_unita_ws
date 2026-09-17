#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re

import rospy

from pyproj import Proj
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

from morai_msgs.msg import GPSMessage, CtrlCmd


class GPSWaypointFollower:
    def __init__(self):
        rospy.init_node("gps_waypoint_follower")

        # ============================================================
        # ROS parameters
        # ============================================================

        self.waypoint_file = rospy.get_param(
            "~waypoint_file",
            ""
        )

        # latlon : latitude, longitude
        # local_xy : MORAI map local x, y
        self.waypoint_format = rospy.get_param(
            "~waypoint_format",
            "latlon"
        )

        self.gps_topic = rospy.get_param(
            "~gps_topic",
            "/gps"
        )

        self.imu_topic = rospy.get_param(
            "~imu_topic",
            "/imu"
        )

        self.ctrl_topic = rospy.get_param(
            "~ctrl_topic",
            "/ctrl_cmd"
        )

        # Korea / MORAI example
        self.utm_zone = rospy.get_param(
            "~utm_zone",
            52
        )

        # ------------------------------------------------------------
        # Vehicle / Pure Pursuit parameters
        # ------------------------------------------------------------

        self.wheelbase = rospy.get_param(
            "~wheelbase",
            2.7
        )

        self.lookahead_distance = rospy.get_param(
            "~lookahead_distance",
            5.0
        )

        self.target_speed_kmh = rospy.get_param(
            "~target_speed_kmh",
            20.0
        )

        self.max_steering_rad = rospy.get_param(
            "~max_steering_rad",
            0.5
        )

        # 차량이 반대로 조향하면 -1.0으로 변경
        self.steering_sign = rospy.get_param(
            "~steering_sign",
            1.0
        )

        self.goal_tolerance = rospy.get_param(
            "~goal_tolerance",
            2.0
        )
        self.loop_path = rospy.get_param("~loop_path", True)

        self.control_rate = rospy.get_param(
            "~control_rate",
            20.0
        )
        self.position_reset_distance = float(rospy.get_param(
            "~position_reset_distance", 10.0))
        if (not math.isfinite(self.position_reset_distance)
                or self.position_reset_distance <= 0):
            raise ValueError("position_reset_distance must be positive and finite")

        # ============================================================
        # State
        # ============================================================

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.east_offset = None
        self.north_offset = None

        self.gps_ready = False
        self.imu_ready = False
        self.path_ready = False

        self.current_waypoint_index = 0
        self.first_nearest_search = True
        self.last_tracking_position = None

        # raw waypoint:
        # latlon -> [(lat, lon), ...]
        # local_xy -> [(x, y), ...]
        self.raw_waypoints = []

        # controller always uses local xy
        self.waypoints = []

        # ============================================================
        # UTM projection
        # ============================================================

        self.proj_utm = Proj(
            proj="utm",
            zone=self.utm_zone,
            ellps="WGS84",
            preserve_units=False
        )

        # ============================================================
        # ROS publisher / subscriber
        # ============================================================

        self.ctrl_pub = rospy.Publisher(
            self.ctrl_topic,
            CtrlCmd,
            queue_size=1
        )

        # RViz debug
        self.path_pub = rospy.Publisher(
            "/waypoint_path",
            Path,
            queue_size=1,
            latch=True
        )

        rospy.Subscriber(
            self.gps_topic,
            GPSMessage,
            self.gps_callback,
            queue_size=1
        )

        rospy.Subscriber(
            self.imu_topic,
            Imu,
            self.imu_callback,
            queue_size=1
        )

        # ============================================================
        # Load waypoint file
        # ============================================================

        self.load_waypoints()

        # local_xy는 GPS offset 필요 없음
        if self.waypoint_format == "local_xy":
            self.waypoints = list(self.raw_waypoints)
            self.path_ready = True
            self.publish_path()

        rospy.loginfo("======================================")
        rospy.loginfo(" GPS Waypoint Follower started")
        rospy.loginfo(" GPS topic      : %s", self.gps_topic)
        rospy.loginfo(" IMU topic      : %s", self.imu_topic)
        rospy.loginfo(" CTRL topic     : %s", self.ctrl_topic)
        rospy.loginfo(" waypoint file  : %s", self.waypoint_file)
        rospy.loginfo(" waypoint type  : %s", self.waypoint_format)
        rospy.loginfo(" target speed   : %.1f km/h",
                      self.target_speed_kmh)
        rospy.loginfo(" lookahead      : %.2f m",
                      self.lookahead_distance)
        rospy.loginfo(" loop path      : %s", self.loop_path)
        rospy.loginfo("======================================")

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate),
            self.control_loop
        )

    # ================================================================
    # waypoint loader
    # ================================================================

    def load_waypoints(self):
        if not self.waypoint_file:
            rospy.logfatal("waypoint_file is empty")
            rospy.signal_shutdown("No waypoint file")
            return

        path = os.path.expanduser(self.waypoint_file)

        if not os.path.isfile(path):
            rospy.logfatal(
                "Waypoint file not found: %s",
                path
            )
            rospy.signal_shutdown("Waypoint file not found")
            return

        points = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                # comma / space / tab 모두 지원
                tokens = re.split(r"[,\s]+", line)

                if len(tokens) < 2:
                    continue

                try:
                    a = float(tokens[0])
                    b = float(tokens[1])
                except ValueError:
                    # latitude,longitude 같은 header 무시
                    continue

                points.append((a, b))

        if len(points) < 2:
            rospy.logfatal(
                "Not enough waypoint points: %d",
                len(points)
            )
            rospy.signal_shutdown(
                "Need at least two waypoints"
            )
            return

        self.raw_waypoints = points

        rospy.loginfo(
            "Loaded %d waypoints",
            len(self.raw_waypoints)
        )

    # ================================================================
    # GPS
    # ================================================================

    def gps_callback(self, msg):
        try:
            utm_x, utm_y = self.proj_utm(
                msg.longitude,
                msg.latitude
            )
        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                "UTM conversion failed: %s" % str(e)
            )
            return

        self.east_offset = msg.eastOffset
        self.north_offset = msg.northOffset

        # MORAI local map coordinate
        self.current_x = utm_x - self.east_offset
        self.current_y = utm_y - self.north_offset

        self.gps_ready = True

        # GPS waypoint는 offset을 알아야 local xy로 변환 가능
        if (
            self.waypoint_format == "latlon"
            and not self.path_ready
        ):
            self.convert_waypoints_to_local()

    # ================================================================
    # IMU
    # ================================================================

    def imu_callback(self, msg):
        quaternion = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

        norm = math.sqrt(
            quaternion[0] ** 2
            + quaternion[1] ** 2
            + quaternion[2] ** 2
            + quaternion[3] ** 2
        )

        if norm < 1e-6:
            return

        _, _, yaw = euler_from_quaternion(
            quaternion
        )

        self.current_yaw = yaw
        self.imu_ready = True

    # ================================================================
    # Lat/Lon waypoint -> MORAI local xy
    # ================================================================

    def convert_waypoints_to_local(self):
        if (
            self.east_offset is None
            or self.north_offset is None
        ):
            return

        local_points = []

        for lat, lon in self.raw_waypoints:
            utm_x, utm_y = self.proj_utm(
                lon,
                lat
            )

            x = utm_x - self.east_offset
            y = utm_y - self.north_offset

            local_points.append((x, y))

        self.waypoints = local_points
        self.path_ready = True

        rospy.loginfo(
            "Converted %d GPS waypoints to MORAI local XY",
            len(self.waypoints)
        )

        self.publish_path()

    # ================================================================
    # RViz Path visualization
    # ================================================================

    def publish_path(self):
        if not self.waypoints:
            return

        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        for x, y in self.waypoints:
            pose = PoseStamped()

            pose.header = msg.header

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            pose.pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.path_pub.publish(msg)

    # ================================================================
    # nearest waypoint
    # ================================================================

    def tracking_waypoint_count(self):
        count = len(self.waypoints)
        # A closing copy of the start point is useful for Path display,
        # but must not compete with the start index during loop tracking.
        if self.loop_path and count > 1:
            first, last = self.waypoints[0], self.waypoints[-1]
            if math.hypot(first[0] - last[0], first[1] - last[1]) <= 1e-6:
                count -= 1
        return count

    def find_nearest_waypoint(self):
        if not self.waypoints:
            return 0

        count = self.tracking_waypoint_count()
        # 첫 실행에서는 전체 waypoint 탐색
        if self.first_nearest_search:
            indices = range(count)
            self.first_nearest_search = False

        elif self.loop_path:
            # Search across the end/start boundary in path order.
            start = self.current_waypoint_index - 10
            indices = ((start + offset) % count
                       for offset in range(min(count, 310)))
        else:
            # 이후에는 현재 인덱스 주변만 탐색
            start = max(
                self.current_waypoint_index - 10,
                0
            )

            end = min(
                self.current_waypoint_index + 300,
                count
            )
            indices = range(start, end)

        min_distance = float("inf")
        min_index = self.current_waypoint_index

        for i in indices:
            wx, wy = self.waypoints[i]

            distance = math.hypot(
                wx - self.current_x,
                wy - self.current_y
            )

            if distance < min_distance:
                min_distance = distance
                min_index = i

        return min_index

    # ================================================================
    # look-ahead waypoint
    # ================================================================

    def find_lookahead_point(self):
        start = self.current_waypoint_index
        count = self.tracking_waypoint_count()
        if self.loop_path:
            indices = ((start + offset) % count for offset in range(count))
        else:
            indices = range(start, count)

        for i in indices:
            wx, wy = self.waypoints[i]

            dx = wx - self.current_x
            dy = wy - self.current_y

            # global -> vehicle coordinate
            local_x = (
                math.cos(self.current_yaw) * dx
                + math.sin(self.current_yaw) * dy
            )

            local_y = (
                -math.sin(self.current_yaw) * dx
                + math.cos(self.current_yaw) * dy
            )

            # 뒤쪽 waypoint 무시
            if local_x <= 0.0:
                continue

            distance = math.hypot(
                local_x,
                local_y
            )

            if distance >= self.lookahead_distance:
                return (
                    i,
                    local_x,
                    local_y,
                    distance
                )

        return None

    # ================================================================
    # Pure Pursuit
    # ================================================================

    def calculate_steering(
        self,
        local_x,
        local_y,
        lookahead
    ):
        alpha = math.atan2(
            local_y,
            local_x
        )

        steering = math.atan2(
            2.0
            * self.wheelbase
            * math.sin(alpha),
            lookahead
        )

        steering *= self.steering_sign

        steering = max(
            -self.max_steering_rad,
            min(
                self.max_steering_rad,
                steering
            )
        )

        return steering

    # ================================================================
    # publish control
    # ================================================================

    def publish_control(
        self,
        steering,
        velocity
    ):
        cmd = CtrlCmd()

        # Velocity control
        cmd.longlCmdType = 2

        cmd.accel = 0.0
        cmd.brake = 0.0

        cmd.steering = steering

        # km/h
        cmd.velocity = velocity

        cmd.acceleration = 0.0

        self.ctrl_pub.publish(cmd)

    # ================================================================
    # main control loop
    # ================================================================

    def control_loop(self, event):
        if not self.gps_ready:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for GPS..."
            )
            return

        if not self.imu_ready:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for IMU..."
            )
            return

        if not self.path_ready:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for waypoint path..."
            )
            return

        # ------------------------------------------------------------
        # nearest waypoint
        # ------------------------------------------------------------

        position = (self.current_x, self.current_y)
        if (self.last_tracking_position is not None
                and math.hypot(position[0] - self.last_tracking_position[0],
                               position[1] - self.last_tracking_position[1])
                > self.position_reset_distance):
            # SIM initialization/teleport invalidates the local index window.
            self.current_waypoint_index = 0
            self.first_nearest_search = True
            rospy.loginfo("Position reset detected; searching the full waypoint path")
        self.last_tracking_position = position

        self.current_waypoint_index = \
            self.find_nearest_waypoint()

        # ------------------------------------------------------------
        # check final goal
        # ------------------------------------------------------------

        goal_x, goal_y = self.waypoints[-1]

        goal_distance = math.hypot(
            goal_x - self.current_x,
            goal_y - self.current_y
        )

        if (
            not self.loop_path
            and self.current_waypoint_index
            >= len(self.waypoints) - 3
            and goal_distance
            <= self.goal_tolerance
        ):
            self.publish_control(
                0.0,
                0.0
            )

            rospy.loginfo_throttle(
                1.0,
                "GOAL REACHED | distance = %.2f m"
                % goal_distance
            )

            return

        # ------------------------------------------------------------
        # look-ahead point
        # ------------------------------------------------------------

        target = self.find_lookahead_point()

        if target is None:
            self.publish_control(
                0.0,
                0.0
            )

            rospy.logwarn_throttle(
                1.0,
                "No forward waypoint found"
            )

            return

        target_idx, local_x, local_y, ld = target

        # ------------------------------------------------------------
        # Pure Pursuit steering
        # ------------------------------------------------------------

        steering = self.calculate_steering(
            local_x,
            local_y,
            ld
        )

        # ------------------------------------------------------------
        # MORAI control
        # ------------------------------------------------------------

        self.publish_control(
            steering,
            self.target_speed_kmh
        )

        rospy.loginfo_throttle(
            0.5,
            (
                "WP %d/%d | target=%d | "
                "pos=(%.2f, %.2f) | "
                "yaw=%.1f deg | "
                "steer=%.3f rad | "
                "speed=%.1f km/h"
            )
            % (
                self.current_waypoint_index,
                len(self.waypoints) - 1,
                target_idx,
                self.current_x,
                self.current_y,
                math.degrees(
                    self.current_yaw
                ),
                steering,
                self.target_speed_kmh
            )
        )


if __name__ == "__main__":
    try:
        follower = GPSWaypointFollower()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
