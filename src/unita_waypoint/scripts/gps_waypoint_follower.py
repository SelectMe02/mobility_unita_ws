#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re
import time

import rospy

from pyproj import Proj
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, String
from tf.transformations import euler_from_quaternion

from morai_msgs.msg import GPSMessage, CtrlCmd


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class PIDController:
    """PID with bounded integral/output for MORAI's normalized pedals."""

    def __init__(self, kp, ki, kd, integral_limit=20.0, output_limit=1.0):
        values = (kp, ki, kd, integral_limit, output_limit)
        if (not all(math.isfinite(value) for value in values)
                or min(kp, ki, kd) < 0.0
                or integral_limit <= 0.0 or output_limit <= 0.0):
            raise ValueError("invalid PID gains or limits")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.previous_error = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, target, current, dt):
        if not all(math.isfinite(value) for value in (target, current, dt)):
            raise ValueError("PID input must be finite")
        dt = clamp(dt, 1e-3, 0.2)
        error = target - current
        derivative = (0.0 if self.previous_error is None
                      else (error - self.previous_error) / dt)
        candidate = clamp(self.integral + error * dt,
                          -self.integral_limit, self.integral_limit)
        unclamped = self.kp * error + self.ki * candidate + self.kd * derivative
        output = clamp(unclamped, -self.output_limit, self.output_limit)
        # Anti-windup: integrate while unsaturated or while unwinding saturation.
        if (output == unclamped
                or (output > 0.0 and error < 0.0)
                or (output < 0.0 and error > 0.0)):
            self.integral = candidate
        self.previous_error = error
        return output


def three_point_curvature(first, middle, last):
    """Absolute curvature (1/m) of the circumcircle through three points."""
    ab = math.hypot(first[0] - middle[0], first[1] - middle[1])
    bc = math.hypot(middle[0] - last[0], middle[1] - last[1])
    ca = math.hypot(last[0] - first[0], last[1] - first[1])
    if min(ab, bc, ca) <= 1e-6:
        return 0.0
    twice_area = abs((middle[0] - first[0]) * (last[1] - first[1])
                     - (middle[1] - first[1]) * (last[0] - first[0]))
    return 2.0 * twice_area / (ab * bc * ca)


def build_curvature_speed_profile(points, speed_limits_kmh, loop_path,
                                  curvature_window_m, lateral_accel_limit,
                                  min_curve_speed_kmh, max_accel, max_decel):
    """Plan speed from path curvature, then add acceleration/braking envelopes."""
    count = len(points)
    if count < 3 or len(speed_limits_kmh) != count:
        raise ValueError("speed profile requires matching path and limit arrays")
    segment_lengths = [math.hypot(points[i + 1][0] - points[i][0],
                                  points[i + 1][1] - points[i][1])
                       for i in range(count - 1)]
    if loop_path:
        segment_lengths.append(math.hypot(points[0][0] - points[-1][0],
                                          points[0][1] - points[-1][1]))
    positive = [length for length in segment_lengths if length > 1e-6]
    if not positive:
        raise ValueError("waypoint path has no length")
    mean_spacing = sum(positive) / len(positive)
    span = max(1, int(round(curvature_window_m / mean_spacing)))
    span = min(span, max(1, (count - 1) // 2))

    curvatures = []
    raw_speed_mps = []
    for index in range(count):
        if loop_path:
            previous = points[(index - span) % count]
            following = points[(index + span) % count]
        else:
            previous = points[max(0, index - span)]
            following = points[min(count - 1, index + span)]
        curvature = three_point_curvature(previous, points[index], following)
        curvatures.append(curvature)
        curve_speed_kmh = math.sqrt(
            lateral_accel_limit / max(curvature, 1e-6)) * 3.6
        planned_kmh = max(min_curve_speed_kmh,
                          min(speed_limits_kmh[index], curve_speed_kmh))
        raw_speed_mps.append(planned_kmh / 3.6)

    speeds = list(raw_speed_mps)
    # Multiple circular passes make the start/end boundary obey the same
    # acceleration and braking constraints as the rest of a loop.
    passes = 3 if loop_path else 1
    for _ in range(passes):
        for index in range(count):
            if index == 0 and not loop_path:
                continue
            previous = (index - 1) % count
            distance = segment_lengths[previous]
            speeds[index] = min(
                speeds[index],
                math.sqrt(max(0.0, speeds[previous] ** 2
                                   + 2.0 * max_accel * distance)))
        for index in range(count - 1, -1, -1):
            if index == count - 1 and not loop_path:
                continue
            following = (index + 1) % count
            distance = segment_lengths[index]
            speeds[index] = min(
                speeds[index],
                math.sqrt(max(0.0, speeds[following] ** 2
                                   + 2.0 * max_decel * distance)))
    return [speed * 3.6 for speed in speeds], curvatures


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

        self.enable_velocity_planning = bool(rospy.get_param(
            "~enable_velocity_planning", False))
        self.longitudinal_control_mode = rospy.get_param(
            "~longitudinal_control_mode", "velocity")
        if self.longitudinal_control_mode not in ("velocity", "pid"):
            raise ValueError("longitudinal_control_mode must be velocity or pid")
        self.speed_topic = rospy.get_param(
            "~speed_topic", "/competition/ego_speed_kmh")
        self.speed_feedback_timeout = float(rospy.get_param(
            "~speed_feedback_timeout", 0.5))
        self.min_lookahead_distance = float(rospy.get_param(
            "~min_lookahead_distance", self.lookahead_distance))
        self.max_lookahead_distance = float(rospy.get_param(
            "~max_lookahead_distance", self.lookahead_distance))
        self.lookahead_speed_gain = float(rospy.get_param(
            "~lookahead_speed_gain", 0.0))

        self.normal_max_speed_kmh = float(rospy.get_param(
            "~normal_max_speed_kmh", self.target_speed_kmh))
        self.high_speed_max_speed_kmh = float(rospy.get_param(
            "~high_speed_max_speed_kmh", self.normal_max_speed_kmh))
        self.high_speed_start_checkpoint = rospy.get_param(
            "~high_speed_start_checkpoint", 10)
        self.high_speed_end_checkpoint = rospy.get_param(
            "~high_speed_end_checkpoint", 13)
        self.checkpoint_match_tolerance = float(rospy.get_param(
            "~checkpoint_match_tolerance", 1.0))
        self.checkpoints = rospy.get_param("~checkpoints", [])
        self.curvature_window_m = float(rospy.get_param(
            "~curvature_window_m", 25.0))
        self.lateral_accel_limit = float(rospy.get_param(
            "~lateral_accel_limit", 2.94))
        self.min_curve_speed_kmh = float(rospy.get_param(
            "~min_curve_speed_kmh", 25.0))
        self.max_profile_accel = float(rospy.get_param(
            "~max_profile_accel_mps2", 2.0))
        self.max_profile_decel = float(rospy.get_param(
            "~max_profile_decel_mps2", 3.5))

        positive_parameters = (
            self.speed_feedback_timeout, self.min_lookahead_distance,
            self.max_lookahead_distance, self.normal_max_speed_kmh,
            self.high_speed_max_speed_kmh, self.checkpoint_match_tolerance,
            self.curvature_window_m, self.lateral_accel_limit,
            self.min_curve_speed_kmh, self.max_profile_accel,
            self.max_profile_decel)
        if (not all(math.isfinite(value) and value > 0.0
                    for value in positive_parameters)
                or not math.isfinite(self.lookahead_speed_gain)
                or self.lookahead_speed_gain < 0.0
                or self.min_lookahead_distance > self.max_lookahead_distance
                or self.normal_max_speed_kmh > self.high_speed_max_speed_kmh
                or self.min_curve_speed_kmh > self.normal_max_speed_kmh):
            raise ValueError("invalid velocity planning/look-ahead parameters")

        self.speed_pid = PIDController(
            float(rospy.get_param("~pid_kp", 0.08)),
            float(rospy.get_param("~pid_ki", 0.002)),
            float(rospy.get_param("~pid_kd", 0.01)),
            float(rospy.get_param("~pid_integral_limit", 20.0)))

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
        self.current_speed_kmh = None
        self.last_speed_received = None
        self.last_pid_update = None
        self.speed_profile = []
        self.path_curvatures = []
        self.checkpoint_indices = {}

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
        self.target_speed_pub = rospy.Publisher(
            "/waypoint_debug/target_speed_kmh", Float64, queue_size=1)
        self.current_speed_pub = rospy.Publisher(
            "/waypoint_debug/current_speed_kmh", Float64, queue_size=1)
        self.curvature_pub = rospy.Publisher(
            "/waypoint_debug/path_curvature", Float64, queue_size=1)
        self.speed_zone_pub = rospy.Publisher(
            "/waypoint_debug/speed_zone", String, queue_size=1)

        rospy.Subscriber(
            self.gps_topic,
            GPSMessage,
            self.gps_callback,
            queue_size=1
        )

        if self.longitudinal_control_mode == "pid":
            rospy.Subscriber(
                self.speed_topic,
                Float64,
                self.speed_callback,
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
            self.prepare_velocity_profile()
            self.publish_path()

        rospy.loginfo("======================================")
        rospy.loginfo(" GPS Waypoint Follower started")
        rospy.loginfo(" GPS topic      : %s", self.gps_topic)
        rospy.loginfo(" IMU topic      : %s", self.imu_topic)
        rospy.loginfo(" CTRL topic     : %s", self.ctrl_topic)
        rospy.loginfo(" waypoint file  : %s", self.waypoint_file)
        rospy.loginfo(" waypoint type  : %s", self.waypoint_format)
        if self.enable_velocity_planning:
            rospy.loginfo(" speed limits   : normal %.1f / CP %s-%s %.1f km/h",
                          self.normal_max_speed_kmh,
                          self.high_speed_start_checkpoint,
                          self.high_speed_end_checkpoint,
                          self.high_speed_max_speed_kmh)
        else:
            rospy.loginfo(" target speed   : %.1f km/h", self.target_speed_kmh)
        rospy.loginfo(" longitudinal   : %s", self.longitudinal_control_mode)
        rospy.loginfo(" lookahead      : %.2f..%.2f m (gain %.2f)",
                      self.min_lookahead_distance,
                      self.max_lookahead_distance,
                      self.lookahead_speed_gain)
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

    def speed_callback(self, msg):
        if math.isfinite(msg.data) and msg.data >= 0.0:
            self.current_speed_kmh = msg.data
            self.last_speed_received = time.monotonic()

    def speed_feedback_is_fresh(self):
        return (self.current_speed_kmh is not None
                and self.last_speed_received is not None
                and 0.0 <= time.monotonic() - self.last_speed_received
                <= self.speed_feedback_timeout)

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
        self.prepare_velocity_profile()

        rospy.loginfo(
            "Converted %d GPS waypoints to MORAI local XY",
            len(self.waypoints)
        )

        self.publish_path()

    # ================================================================
    # checkpoint zones / curvature velocity planning
    # ================================================================

    def prepare_velocity_profile(self):
        if not self.enable_velocity_planning:
            self.speed_profile = []
            self.path_curvatures = []
            return

        count = self.tracking_waypoint_count()
        tracking_points = self.waypoints[:count]
        if not self.checkpoints:
            raise ValueError("velocity planning requires checkpoint configuration")

        self.checkpoint_indices = {}
        for checkpoint in self.checkpoints:
            try:
                identifier = str(checkpoint["id"])
                x, y = float(checkpoint["x"]), float(checkpoint["y"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("invalid checkpoint entry: {!r}".format(checkpoint))
            index, _ = min(
                enumerate(tracking_points),
                key=lambda item: math.hypot(item[1][0] - x, item[1][1] - y))
            distance = math.hypot(tracking_points[index][0] - x,
                                  tracking_points[index][1] - y)
            if distance > self.checkpoint_match_tolerance:
                raise ValueError(
                    "checkpoint {} is {:.2f} m from waypoint path".format(
                        identifier, distance))
            self.checkpoint_indices[identifier] = index

        start_id = str(self.high_speed_start_checkpoint)
        end_id = str(self.high_speed_end_checkpoint)
        if start_id not in self.checkpoint_indices or end_id not in self.checkpoint_indices:
            raise ValueError("high-speed checkpoint IDs are missing")
        start_index = self.checkpoint_indices[start_id]
        end_index = self.checkpoint_indices[end_id]

        speed_limits = [self.normal_max_speed_kmh] * count
        if start_index <= end_index:
            high_speed_indices = range(start_index, end_index + 1)
        elif self.loop_path:
            high_speed_indices = list(range(start_index, count)) + list(
                range(0, end_index + 1))
        else:
            raise ValueError("high-speed zone runs backward on a non-loop path")
        for index in high_speed_indices:
            speed_limits[index] = self.high_speed_max_speed_kmh

        self.speed_profile, self.path_curvatures = build_curvature_speed_profile(
            tracking_points,
            speed_limits,
            self.loop_path,
            self.curvature_window_m,
            self.lateral_accel_limit,
            self.min_curve_speed_kmh,
            self.max_profile_accel,
            self.max_profile_decel)
        rospy.loginfo(
            "Velocity profile ready: CP %s idx=%d -> CP %s idx=%d, "
            "planned %.1f..%.1f km/h",
            start_id, start_index, end_id, end_index,
            min(self.speed_profile), max(self.speed_profile))

    def target_speed_for_index(self, index):
        if self.enable_velocity_planning and self.speed_profile:
            return self.speed_profile[index % len(self.speed_profile)]
        return self.target_speed_kmh

    def speed_zone_for_index(self, index):
        if not self.enable_velocity_planning or not self.checkpoint_indices:
            return "constant"
        start = self.checkpoint_indices[str(self.high_speed_start_checkpoint)]
        end = self.checkpoint_indices[str(self.high_speed_end_checkpoint)]
        inside = (start <= index <= end if start <= end
                  else index >= start or index <= end)
        return ("high_speed_cp{}_{}".format(
            self.high_speed_start_checkpoint,
            self.high_speed_end_checkpoint) if inside else "normal")

    def update_dynamic_lookahead(self):
        if self.lookahead_speed_gain <= 0.0:
            return
        speed_kmh = (self.current_speed_kmh
                     if self.current_speed_kmh is not None else 0.0)
        self.lookahead_distance = clamp(
            self.lookahead_speed_gain * speed_kmh / 3.6,
            self.min_lookahead_distance,
            self.max_lookahead_distance)

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
        cmd.steering = steering
        cmd.acceleration = 0.0

        if self.longitudinal_control_mode == "pid":
            now = time.monotonic()
            dt = (1.0 / self.control_rate if self.last_pid_update is None
                  else now - self.last_pid_update)
            self.last_pid_update = now
            output = self.speed_pid.update(
                velocity, self.current_speed_kmh, dt)
            cmd.longlCmdType = 1
            cmd.velocity = 0.0
            cmd.accel = max(0.0, output)
            cmd.brake = max(0.0, -output)
        else:
            cmd.longlCmdType = 2
            cmd.accel = 0.0
            cmd.brake = 0.0
            cmd.velocity = velocity

        self.ctrl_pub.publish(cmd)

    def publish_full_stop(self):
        cmd = CtrlCmd()
        cmd.longlCmdType = 1
        cmd.accel = 0.0
        cmd.brake = 1.0
        cmd.steering = 0.0
        cmd.velocity = 0.0
        cmd.acceleration = 0.0
        self.ctrl_pub.publish(cmd)
        self.speed_pid.reset()
        self.last_pid_update = None

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

        if (self.longitudinal_control_mode == "pid"
                and not self.speed_feedback_is_fresh()):
            self.publish_full_stop()
            rospy.logwarn_throttle(
                1.0,
                "Waiting for fresh vehicle speed on %s" % self.speed_topic)
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

        self.update_dynamic_lookahead()

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

        target_speed_kmh = self.target_speed_for_index(
            self.current_waypoint_index)
        self.publish_control(steering, target_speed_kmh)

        self.target_speed_pub.publish(Float64(data=target_speed_kmh))
        if self.current_speed_kmh is not None:
            self.current_speed_pub.publish(Float64(data=self.current_speed_kmh))
        if self.path_curvatures:
            self.curvature_pub.publish(Float64(
                data=self.path_curvatures[self.current_waypoint_index]))
        self.speed_zone_pub.publish(String(
            data=self.speed_zone_for_index(self.current_waypoint_index)))

        rospy.loginfo_throttle(
            0.5,
            (
                "WP %d/%d | target=%d | "
                "pos=(%.2f, %.2f) | "
                "yaw=%.1f deg | "
                "steer=%.3f rad | "
                "target/current=%.1f/%.1f km/h | Ld=%.1f m | zone=%s"
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
                target_speed_kmh,
                self.current_speed_kmh if self.current_speed_kmh is not None else -1.0,
                self.lookahead_distance,
                self.speed_zone_for_index(self.current_waypoint_index)
            )
        )


if __name__ == "__main__":
    try:
        follower = GPSWaypointFollower()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
