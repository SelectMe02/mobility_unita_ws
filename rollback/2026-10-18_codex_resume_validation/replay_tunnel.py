"""Offline scan comparison, never publishes ROS commands.

GPS return + integrated recorded speed estimate route progress backwards.
This checks component behavior on recorded scans, not a closed-loop drive.
Ego odometry was not recorded, so temporal crossing tracking is not replayed.
Run from the workspace after sourcing ROS and devel/setup.bash.
"""
import bisect
from dataclasses import fields, replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import rosbag
import rospy
from sensor_msgs import point_cloud2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/leo_avoidance/src'))
from leo_avoidance.ftg_local import Config, FollowTheGap
from leo_guard.route_file import anchor_on_route, load_route_points, load_route_speeds
from leo_guard.traffic_local import RouteProgress


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


old = module('ftg_before_resume', ROOT / 'rollback/2026-10-18_codex_resume_before' /
             'src/leo_avoidance/src/leo_avoidance/ftg_local.py')
node = module('exit_guidance_replay', ROOT / 'src/leo_avoidance/scripts/driving_state_node_local.py')
cfg_data = yaml.safe_load((ROOT / 'src/leo_avoidance/config/avoidance_local.yaml').read_text())
cfg = Config(**{f.name: cfg_data[f.name] for f in fields(Config) if f.name in cfg_data})
old_cfg = old.Config(**{f.name: getattr(cfg, f.name) for f in fields(old.Config)})
odom_cfg = yaml.safe_load((ROOT / 'src/leo_avoidance/config/odometry_local.yaml').read_text())
route_file = ROOT / 'src/unita_waypoint/config/map_tunner_project.json'
route = RouteProgress(load_route_points(str(route_file)), max_error=7., backward_window=30)
speeds = load_route_speeds(str(route_file))
state = node.DrivingStateNode.__new__(node.DrivingStateNode)
state.route = route
state.route_lookahead_m = cfg_data['route_lookahead_m']
state.exit_rejoin_start_m = odom_cfg['exit_rejoin_start_m']
state.exit_rejoin_full_m = odom_cfg['exit_rejoin_full_m']
exit_index, _ = anchor_on_route(route.points, odom_cfg['exit_anchor'])
tangent = route.delta[exit_index] / route.lengths[exit_index]
exit_xy = np.array([odom_cfg['exit_anchor'][axis] for axis in ('x', 'y')])
state.exit_route_s = route.s[exit_index] + np.clip(
    (exit_xy - route.points[exit_index]) @ tangent, 0., route.lengths[exit_index])
bag_path = ROOT / 'src/unita_waypoint/log/run_20261018_205532/sensors_2026-10-18-20-55-32_0.bag'
speed_samples, headings, poses = [], [], []
return_time = None
denied = False
with rosbag.Bag(str(bag_path)) as bag:
    for topic, message, stamp in bag.read_messages(topics=[
            '/gps', '/competition/ego_speed_kmh', '/imu', '/localization/pose']):
        t = stamp.to_sec()
        if topic == '/gps':
            valid = abs(message.latitude) > 1e-9 and abs(message.longitude) > 1e-9
            if denied and valid and return_time is None:
                return_time = t
            denied = denied or not valid
        elif topic == '/competition/ego_speed_kmh':
            speed_samples.append((t, message.data / 3.6))
        elif topic == '/imu':
            headings.append((message.header.stamp.to_sec(), node.DrivingStateNode.yaw(message.orientation)))
        else:
            p, q = message.pose.position, message.pose.orientation
            poses.append((t, (p.x, p.y, node.DrivingStateNode.yaw(q))))
assert return_time is not None
return_pose = next(p for t, p in poses if t >= return_time)
return_s, return_error = route.locate(*return_pose)
speed_times, speed_values = np.array(speed_samples).T
heading_times, heading_values = np.array(headings).T
travel = np.r_[0., np.cumsum(np.diff(speed_times) * (speed_values[:-1] + speed_values[1:]) * .5)]
return_travel = np.interp(return_time, speed_times, travel)
now = [0.]
before = old.FollowTheGap(old_cfg, clock=lambda: now[0])
unchanged = FollowTheGap(cfg, clock=lambda: now[0])
after = FollowTheGap(cfg, clock=lambda: now[0])
rows, unchanged_differences = [], 0
with rosbag.Bag(str(bag_path)) as bag:
    for _, message, stamp in bag.read_messages(
            topics=['/lidar3D'], start_time=rospy.Time.from_sec(return_time - 19.),
            end_time=rospy.Time.from_sec(return_time)):
        t = message.header.stamp.to_sec()
        now[0] = t
        speed = float(np.interp(t, speed_times, speed_values))
        heading = float(heading_values[min(bisect.bisect_left(heading_times, t), len(heading_values)-1)])
        state.route_progress_m = return_s - (return_travel - np.interp(t, speed_times, travel))
        index = min(len(speeds)-1, max(0, int(np.searchsorted(route.s, state.route_progress_m)-1)))
        for planner in (before, unchanged, after):
            planner.config = replace(planner.config, max_speed_mps=float(speeds[index])/3.6)
        points = np.asarray(list(point_cloud2.read_points(
            message, field_names=('x', 'y', 'z'), skip_nans=True)), float).reshape(-1, 3)
        original = before.plan(points, speed, 0., wall_follow=True, stamp=t)
        same = unchanged.plan(points, speed, 0., wall_follow=True, stamp=t)
        unchanged_differences += vars(original) != vars(same)
        error, weight, remaining = state.exit_rejoin_guidance(heading)
        revised = after.plan(points, speed, 0., wall_follow=True, stamp=t,
                             wall_follow_weight=weight, route_rejoin_remaining_m=remaining,
                             exit_heading_error_rad=error)
        rows.append(dict(seconds_before_gps=round(t-return_time, 3),
                         remaining_m=remaining, heading_error_rad=error,
                         phase_before=original.avoidance_phase, phase_after=revised.avoidance_phase,
                         bearing_before=original.target_bearing_rad, bearing_after=revised.target_bearing_rad,
                         speed_before=original.speed_mps, speed_after=revised.speed_mps,
                         obstacle_points_before=original.obstacle_points,
                         obstacle_points_after=revised.obstacle_points))
exit_rows = [r for r in rows if r['remaining_m'] is not None]
report = dict(scans=len(rows), unchanged_mode_differences=unchanged_differences,
              exit_scans=len(exit_rows), gps_return_projection_error_m=return_error,
              passing_before=sum(r['phase_before'] == 'PASSING' for r in exit_rows),
              passing_after=sum(r['phase_after'] == 'PASSING' for r in exit_rows),
              changed_obstacle_counts=sum(r['obstacle_points_before'] != r['obstacle_points_after'] for r in rows),
              last_exit_scan=exit_rows[-1], rows=rows)
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
assert unchanged_differences == 0
assert report['changed_obstacle_counts'] == 0
assert report['passing_after'] < report['passing_before']
