import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "waypoint_follower", PACKAGE / "scripts/gps_waypoint_follower.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LoopTrackingTests(unittest.TestCase):
    def follower(self, points, loop=True):
        obj = module.GPSWaypointFollower.__new__(module.GPSWaypointFollower)
        obj.waypoints = points
        obj.loop_path = loop
        obj.path_ready = obj.gps_ready = obj.imu_ready = True
        obj.current_waypoint_index = 0
        obj.first_nearest_search = True
        obj.last_tracking_position = None
        obj.position_reset_distance = 10.0
        obj.current_x, obj.current_y = points[0]
        obj.current_yaw = 0.0
        obj.goal_tolerance = 2.0
        obj.lookahead_distance = 5.0
        obj.wheelbase = 2.7
        obj.steering_sign = 1.0
        obj.max_steering_rad = 0.5
        obj.target_speed_kmh = 20.0
        obj.enable_velocity_planning = False
        obj.longitudinal_control_mode = "velocity"
        obj.lookahead_speed_gain = 0.0
        obj.current_speed_kmh = None
        obj.speed_profile = []
        obj.path_curvatures = []
        obj.checkpoint_indices = {}
        obj.target_speed_pub = MagicMock()
        obj.current_speed_pub = MagicMock()
        obj.curvature_pub = MagicMock()
        obj.speed_zone_pub = MagicMock()
        obj.commands = []
        obj.publish_control = lambda steering, velocity: obj.commands.append(
            (steering, velocity))
        return obj

    def run_loop(self, obj):
        with patch.object(module.rospy, "loginfo"), \
                patch.object(module.rospy, "loginfo_throttle"), \
                patch.object(module.rospy, "logwarn_throttle"):
            obj.control_loop(None)

    def actual_points(self):
        with (PACKAGE / "config/waypoints.csv").open() as stream:
            return [tuple(map(float, line.split()[:2])) for line in stream
                    if line.strip()]

    def set_pose(self, obj, index):
        count = obj.tracking_waypoint_count()
        obj.current_x, obj.current_y = obj.waypoints[index]
        # Duplicate samples have no heading. Use the next distinct sample,
        # as a real IMU keeps measuring the vehicle's direction there.
        for offset in range(1, count):
            next_x, next_y = obj.waypoints[(index + offset) % count]
            if math.hypot(next_x - obj.current_x, next_y - obj.current_y) > 1e-6:
                obj.current_yaw = math.atan2(next_y - obj.current_y,
                                            next_x - obj.current_x)
                break

    def test_overlapping_start_end_drives_at_start(self):
        obj = self.follower(self.actual_points())
        self.assertEqual(obj.waypoints[0], obj.waypoints[-1])
        self.set_pose(obj, 0)
        self.run_loop(obj)
        self.assertEqual(obj.current_waypoint_index, 0)
        self.assertEqual(obj.commands[-1][1], 20.0)

    def test_lookahead_wraps_before_end_without_stopping(self):
        obj = self.follower(self.actual_points())
        count = obj.tracking_waypoint_count()
        obj.current_waypoint_index = count - 1
        obj.first_nearest_search = False
        self.set_pose(obj, count - 1)
        target = obj.find_lookahead_point()
        self.assertIsNotNone(target)
        self.assertLess(target[0], 30)
        self.assertGreater(target[1], 0.0)
        self.run_loop(obj)
        self.assertEqual(obj.commands[-1][1], 20.0)

    def test_nearest_window_wraps_to_start(self):
        obj = self.follower(self.actual_points())
        obj.current_waypoint_index = obj.tracking_waypoint_count() - 2
        obj.first_nearest_search = False
        self.set_pose(obj, 2)
        self.run_loop(obj)
        self.assertEqual(obj.current_waypoint_index, 2)
        self.assertEqual(obj.commands[-1][1], 20.0)

    def test_two_complete_laps_keep_driving(self):
        obj = self.follower(self.actual_points())
        count = obj.tracking_waypoint_count()
        for lap in range(2):
            for index in range(count):
                self.set_pose(obj, index)
                self.run_loop(obj)
                steering, velocity = obj.commands[-1]
                self.assertEqual(velocity, 20.0, (lap, index))
                self.assertTrue(math.isfinite(steering), (lap, index))
                self.assertLessEqual(abs(steering), obj.max_steering_rad)
            self.set_pose(obj, 0)
            self.run_loop(obj)
            self.assertEqual(obj.current_waypoint_index, 0)
            self.assertEqual(obj.commands[-1][1], 20.0)

    def test_teleport_reacquires_start_after_lap(self):
        obj = self.follower(self.actual_points())
        obj.current_waypoint_index = 2000
        obj.first_nearest_search = False
        obj.last_tracking_position = obj.waypoints[2000]
        self.set_pose(obj, 0)
        self.run_loop(obj)
        self.assertEqual(obj.current_waypoint_index, 0)
        self.assertEqual(obj.commands[-1][1], 20.0)

    def test_open_path_still_stops_when_loop_is_disabled(self):
        obj = self.follower([(float(i), 0.0) for i in range(20)], loop=False)
        obj.current_waypoint_index = 19
        obj.first_nearest_search = False
        obj.current_x = 19.0
        self.run_loop(obj)
        self.assertEqual(obj.commands[-1], (0.0, 0.0))

    def test_no_forward_target_still_stops(self):
        obj = self.follower([(0.0, 0.0), (-10.0, 0.0), (0.0, 0.0)])
        self.run_loop(obj)
        self.assertEqual(obj.commands[-1], (0.0, 0.0))

    def test_pid_accelerates_brakes_and_clamps_pedals(self):
        pid = module.PIDController(0.08, 0.002, 0.01)
        self.assertEqual(pid.update(100.0, 0.0, 0.05), 1.0)
        pid.reset()
        self.assertEqual(pid.update(0.0, 100.0, 0.05), -1.0)
        pid.reset()
        output = pid.update(50.0, 48.0, 0.05)
        self.assertGreater(output, 0.0)
        self.assertLess(output, 1.0)

    def test_pid_mode_publishes_normalized_throttle_control(self):
        obj = module.GPSWaypointFollower.__new__(module.GPSWaypointFollower)
        obj.longitudinal_control_mode = "pid"
        obj.control_rate = 20.0
        obj.last_pid_update = None
        obj.current_speed_kmh = 40.0
        obj.speed_pid = module.PIDController(0.08, 0.002, 0.01)
        obj.ctrl_pub = MagicMock()
        obj.publish_control(0.12, 60.0)
        command = obj.ctrl_pub.publish.call_args.args[0]
        self.assertEqual(command.longlCmdType, 1)
        self.assertGreater(command.accel, 0.0)
        self.assertEqual(command.brake, 0.0)
        self.assertEqual(command.velocity, 0.0)
        self.assertAlmostEqual(command.steering, 0.12)

    def test_speed_proportional_lookahead_is_bounded(self):
        obj = module.GPSWaypointFollower.__new__(module.GPSWaypointFollower)
        obj.lookahead_speed_gain = 0.78
        obj.min_lookahead_distance = 5.0
        obj.max_lookahead_distance = 30.0
        obj.current_speed_kmh = 100.0
        obj.update_dynamic_lookahead()
        self.assertAlmostEqual(obj.lookahead_distance, 21.6666667, places=5)
        obj.current_speed_kmh = 0.0
        obj.update_dynamic_lookahead()
        self.assertEqual(obj.lookahead_distance, 5.0)

    def test_checkpoint_coordinates_match_the_recorded_path(self):
        points = self.actual_points()[:-1]
        with (PACKAGE / "config/competition_speed_profile.yaml").open() as stream:
            config = yaml.safe_load(stream)
        matched = {}
        for checkpoint in config["checkpoints"]:
            index, point = min(
                enumerate(points),
                key=lambda item: math.hypot(
                    item[1][0] - checkpoint["x"],
                    item[1][1] - checkpoint["y"]))
            distance = math.hypot(point[0] - checkpoint["x"],
                                  point[1] - checkpoint["y"])
            self.assertLess(distance, 1e-6, checkpoint["id"])
            matched[str(checkpoint["id"])] = index
        self.assertEqual(matched["10"], 2275)
        self.assertEqual(matched["13"], 3529)

    def test_curvature_profile_obeys_60_and_100_kmh_zones(self):
        points = self.actual_points()[:-1]
        limits = [60.0] * len(points)
        for index in range(2275, 3529 + 1):
            limits[index] = 100.0
        speeds, curvatures = module.build_curvature_speed_profile(
            points, limits, True, 25.0, 2.94, 25.0, 2.0, 3.5)
        self.assertEqual(len(speeds), len(points))
        self.assertEqual(len(curvatures), len(points))
        self.assertLessEqual(max(speeds[:2275]), 60.0 + 1e-9)
        self.assertLessEqual(max(speeds[3530:]), 60.0 + 1e-9)
        self.assertGreater(max(speeds[2275:3530]), 99.0)
        self.assertLess(speeds[3231], 80.0)  # bend around checkpoint 12
        self.assertLess(speeds[1297], 45.0)  # tight bend around checkpoint 7


if __name__ == "__main__":
    unittest.main()
